"""Qulacs operator, compiled from the shared sparse Pauli representation.

This wrapper used to hold a Qiskit ``SparsePauliOp``; it now compiles the
framework-independent representation directly, which is what removed Qiskit
from this backend.

It is one observable.  Evaluating several against one circuit is
:class:`QulacsObservableBatch` below, which is what the multi-observable
gradient path uses.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import sympy as sp
from qulacs import GeneralQuantumOperator, PauliOperator  # pylint: disable=no-name-in-module
from sympy import lambdify

from ..base.observable_batch import ObservableBatch
from ..base.operator_base import QuantumOperatorBase
from ..base.operator_ir import PauliIR
from ..parameters import Parameter, sort_parameters


def _pauli_term(label: str) -> str:
    """Render a Pauli label as the term string Qulacs parses.

    Qubit ``q`` is character ``q`` of the label, matching the Qulacs term
    string, so no reversal happens here.

    Args:
        label: A Pauli label such as ``"ZI"``.

    Returns:
        The Qulacs term string, e.g. ``"Z 0 I 1 "``.
    """
    return " ".join(f"{pauli} {qubit}" for qubit, pauli in enumerate(label)) + " "


def _as_number(coeff: Any) -> complex | float:
    """Reduce a numeric coefficient, dropping a negligible imaginary part."""
    value = np.real_if_close(complex(coeff))
    return float(value.real) if not np.iscomplexobj(value) else complex(value)


class QulacsOperator(QuantumOperatorBase):
    """An observable that compiles to Qulacs operators.

    Built like any other operator -- ``QulacsOperator(["ZI"], [1.0])`` -- or
    converted from an existing one with :meth:`from_quantum_operator`.
    Compilation is lazy, so the inherited algebra (:meth:`compose`,
    :meth:`adjoint`, :meth:`apply_layout`, ...) works without paying for it.

    Args:
        paulis: Pauli labels, qubit 0 leftmost.
        coeffs: One coefficient per label.
        num_qubits: Width, required only when no labels are given.
        _ir: Adopt this representation instead of building one.
    """

    def __init__(
        self,
        paulis: "Sequence[str] | None" = None,
        coeffs: "Sequence[Any] | None" = None,
        num_qubits: "int | None" = None,
        *,
        _ir: "PauliIR | None" = None,
    ) -> None:
        super().__init__(paulis, coeffs, num_qubits, _ir=_ir)
        self._terms: List[str] = []
        self._coeff_funcs: List[Callable] = []
        self._coeff_grad_funcs: List[List[Callable]] = []
        self._used_parameters: List[List[Parameter]] = []
        self._compiled = False

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def _build_native(self) -> Callable:
        """Compile the representation into a Qulacs operator factory."""
        return self.get_operator_func()

    def _ensure_compiled(self) -> None:
        """Compile the Pauli terms and coefficient callables, once."""
        if self._compiled:
            return
        self._terms = []
        self._coeff_funcs = []
        self._coeff_grad_funcs = []
        self._used_parameters = []
        for label, coeff in zip(self._ir.to_labels(), self._ir.coeffs):
            coeff_func, grads, used = self._build_coeff_functions(coeff)
            self._terms.append(_pauli_term(label))
            self._coeff_funcs.append(coeff_func)
            self._coeff_grad_funcs.append(grads)
            self._used_parameters.append(used)
        self._compiled = True

    @property
    def _symbol_tuple_obs(self) -> Tuple[Parameter, ...]:
        """Symbols the lambdified coefficient callables accept, in argument order."""
        return tuple(self.parameters)

    def _build_coeff_functions(self, coeff: Any) -> Tuple[Any, List[Callable], List[Parameter]]:
        """Build the value and derivative callables for one term's coefficient.

        Args:
            coeff: The coefficient, a number or a SymPy expression.

        Returns:
            Tuple of ``(coeff_func, grad_funcs, used_parameters)``.  A constant
            coefficient yields a zero derivative and no used parameters.
        """
        if isinstance(coeff, sp.Basic) and coeff.free_symbols:
            coeff_func = lambdify(self._symbol_tuple_obs, coeff)
            grad_funcs: List[Callable] = []
            used_parameters: List[Parameter] = []
            for parameter in sort_parameters(
                s for s in coeff.free_symbols if isinstance(s, Parameter)
            ):
                used_parameters.append(parameter)
                derivative = sp.diff(coeff, parameter)
                if derivative.free_symbols:
                    grad_funcs.append(lambdify(self._symbol_tuple_obs, derivative))
                else:
                    # Call-by-value so the closure keeps this term's constant.
                    value = _as_number(derivative)
                    grad_funcs.append(lambda *_args, value=value: value)
            return coeff_func, grad_funcs, used_parameters

        constant = _as_number(coeff)
        return (lambda *_args, constant=constant: constant), [lambda *_args: 0.0], []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def parameter_dimensions(self) -> Dict[str, int]:
        """Number of parameter occurrences, keyed by vector name."""
        dimensions: Dict[str, int] = {}
        for parameter in self.parameters:
            dimensions[parameter.vector_name] = dimensions.get(parameter.vector_name, 0) + 1
        return dimensions

    @property
    def parameter_names(self) -> list:
        """List of operator parameter names"""
        return list(self.parameter_dimensions)

    @property
    def free_parameters(self) -> set:
        """Return the set of free (non-bound) parameters in the operator."""
        return set(self._ir.free_parameters)

    @property
    def terms(self) -> List[str]:
        """The Qulacs term strings, one per Pauli."""
        self._ensure_compiled()
        return list(self._terms)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def build_operator(self, values: Sequence) -> GeneralQuantumOperator:
        """Build the Qulacs operator for one set of parameter values.

        Args:
            values: Values for this operator's parameters, in order.

        Returns:
            The Qulacs operator.
        """
        self._ensure_compiled()
        operator = GeneralQuantumOperator(self.num_qubits)
        for term, coeff in zip(self._terms, self._coeff_funcs):
            operator.add_operator(coeff(*values), term)
        return operator

    def gradient_terms(self, gradient_parameters: Sequence[Parameter]) -> List[PauliOperator]:
        """Return the Pauli terms whose coefficient depends on a requested parameter."""
        self._ensure_compiled()
        return [
            PauliOperator(term, 1.0)
            for index, term in enumerate(self._terms)
            if any(p in gradient_parameters for p in self._used_parameters[index])
        ]

    def coefficient_jacobian(
        self, gradient_parameters: Sequence[Parameter], values: Sequence
    ) -> np.ndarray:
        """Return d(coefficient)/d(parameter) for the terms that depend on one.

        Args:
            gradient_parameters: Parameters the gradient is taken with respect to.
            values: Values for this operator's parameters, in order.

        Returns:
            An array of shape ``(relevant terms, len(gradient_parameters))``.
        """
        self._ensure_compiled()
        positions = {p: i for i, p in enumerate(gradient_parameters)}
        relevant = [
            index
            for index in range(len(self._terms))
            if any(p in gradient_parameters for p in self._used_parameters[index])
        ]
        jacobian = np.zeros((len(relevant), len(gradient_parameters)))
        for row, index in enumerate(relevant):
            for slot, parameter in enumerate(self._used_parameters[index]):
                if parameter in gradient_parameters:
                    jacobian[row, positions[parameter]] = self._coeff_grad_funcs[index][slot](
                        *values
                    )
        return jacobian

    def _rebuild(self, ir: PauliIR) -> "QulacsOperator":
        """Wrap a new representation in this operator's type."""
        return type(self)(_ir=ir)


class QulacsObservableBatch(ObservableBatch):
    """Several Qulacs observables evaluated against one circuit."""

    def get_operator_func(self) -> Callable:
        """Return a callable building one Qulacs operator per observable.

        The callable takes the parameter values already flattened, one per
        symbol -- the convention the executor uses when evaluating an
        expectation value.
        """

        def operator_func(*args):
            return [
                operator.build_operator(slice_)
                for operator, slice_ in zip(self, self.split_arguments(list(args)))
            ]

        return operator_func

    def get_operators_for_gradient(
        self, gradient_parameters: "Parameter | List[Parameter] | None" = None
    ) -> Callable:
        """Return a callable giving the gradient-relevant Pauli terms per observable."""
        selected = _as_parameter_list(gradient_parameters)

        def operator_func(*_args):
            return [operator.gradient_terms(selected) for operator in self]

        return operator_func

    def get_gradient_outer_jacobian_operators_new(
        self, gradient_parameters: "Parameter | List[Parameter] | None" = None
    ) -> Callable:
        """Return a callable giving the coefficient Jacobian per observable.

        Qulacs differentiates only bare rotation angles, so a coefficient that
        is an expression needs this outer factor for the chain rule.
        """
        selected = _as_parameter_list(gradient_parameters)

        def outer_jacobian(*args):
            # Grouped per parameter vector here, unlike get_operator_func.
            values = _flatten(args, len(self.parameter_names))
            return [
                operator.coefficient_jacobian(selected, slice_)
                for operator, slice_ in zip(self, self.split_arguments(values))
            ]

        return outer_jacobian


def _as_parameter_list(
    gradient_parameters: "Parameter | List[Parameter] | None",
) -> List[Parameter]:
    """Normalise a gradient-parameter argument to a list."""
    if isinstance(gradient_parameters, Parameter):
        return [gradient_parameters]
    return list(gradient_parameters) if gradient_parameters is not None else []


def _flatten(args: Sequence, count: int) -> List:
    """Flatten the first ``count`` positional argument groups into one list."""
    return sum([list(args[i]) for i in range(count)], [])
