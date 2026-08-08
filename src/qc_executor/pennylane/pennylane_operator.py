"""PennyLane operator, compiled from the shared sparse Pauli representation.

This wrapper used to hold a Qiskit ``SparsePauliOp``; it now compiles the
framework-independent representation directly, which is what removed Qiskit
from this backend.  Labels need no reversal any more: qubit 0 is leftmost on
both sides.

It is one observable.  Evaluating several against one circuit is
:class:`PennyLaneObservableBatch` below, whose stacked measurement is what makes
a multi-observable gradient one differentiation rather than a loop.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

import pennylane as qml
import pennylane.numpy as pnp
import sympy as sp
from pennylane import pauli
from sympy import lambdify

from ..base.observable_batch import ObservableBatch
from ..base.operator_base import QuantumOperatorBase
from ..base.operator_ir import PauliIR
from ..parameters import Parameter
from ._sympy_interface import _get_sympy_interface


def _resolve_coefficient(coeff, symbol_tuple, printer, modules):
    """Convert one coefficient to a float or a PennyLane-compatible callable.

    Args:
        coeff: The coefficient, a number or a SymPy expression.
        symbol_tuple: Symbols the callable takes, in argument order.
        printer: SymPy printer targeting PennyLane's autograd numpy.
        modules: Module list for :func:`sympy.lambdify`.

    Returns:
        A float for a constant coefficient, else a callable of the symbols.

    Raises:
        ValueError: If a constant coefficient has an imaginary part.
    """
    if isinstance(coeff, sp.Basic) and coeff.free_symbols:
        return lambdify(symbol_tuple, coeff, modules=modules, printer=printer)
    value = complex(coeff)
    if value.imag != 0:
        raise ValueError("Imaginary part of operator coefficient is not supported")
    return float(value.real)


class PennyLaneOperator(QuantumOperatorBase):
    """An observable that compiles to a PennyLane measurement.

    Built like any other operator -- ``PennyLaneOperator(["ZI"], [1.0])`` -- or
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
        self._pennylane_words: List[Any] = []
        self._coeff_functions: List[Any] = []
        self._compiled = False

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def _build_native(self) -> Callable:
        """Compile the representation into a PennyLane observable callable."""
        return self.build_pennylane_observable()

    def _ensure_compiled(self) -> None:
        """Compile the Pauli words and coefficient callables, once."""
        if self._compiled:
            return
        printer, modules = _get_sympy_interface()
        symbol_tuple: Tuple[Parameter, ...] = tuple(self.parameters)
        self._coeff_functions = [
            _resolve_coefficient(coeff, symbol_tuple, printer, modules)
            for coeff in self._ir.coeffs
        ]
        # Qubit 0 is leftmost on both sides, so the label passes through.
        self._pennylane_words = [
            pauli.string_to_pauli_word(label) for label in self._ir.to_labels()
        ]
        self._compiled = True

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
    def pennylane_words(self) -> List[Any]:
        """The compiled PennyLane Pauli words."""
        self._ensure_compiled()
        return self._pennylane_words

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def measurement(self, values: Sequence):
        """Build the PennyLane measurement for one set of parameter values.

        Coefficients are weights, not decoration: an unweighted ``qml.sum``
        here silently returned the value for all-ones coefficients.

        Args:
            values: Values for this operator's parameters, in order.

        Returns:
            The measurement process, or ``0.0`` for an empty operator.
        """
        self._ensure_compiled()
        if len(self._pennylane_words) == 0:
            return 0.0
        coeffs = [coeff(*values) if callable(coeff) else coeff for coeff in self._coeff_functions]
        return qml.expval(qml.Hamiltonian(coeffs, self._pennylane_words))

    def build_pennylane_observable(self) -> Callable:
        """Return a callable that measures this observable inside a QNode.

        Returns:
            A callable taking one sequence of values per operator parameter.
        """

        def pennylane_observable(*args):
            """PennyLane observable that can be called with parameters"""
            values = _flatten(args, len(self.parameter_names))
            return self.measurement(values)

        return pennylane_observable

    def _rebuild(self, ir: PauliIR) -> "PennyLaneOperator":
        """Wrap a new representation in this operator's type."""
        return type(self)(_ir=ir)


class PennyLaneObservableBatch(ObservableBatch):
    """Several PennyLane observables measured in one QNode.

    Returning them stacked is what lets a multi-observable gradient be a single
    differentiation of one QNode rather than a loop over the set.
    """

    def build_pennylane_observable(self) -> Callable:
        """Return a callable measuring every observable, stacked."""

        def pennylane_observable(*args):
            values = _flatten(args, len(self.parameter_names))
            slices = self.split_arguments(values)
            return pnp.stack(
                tuple(operator.measurement(slice_) for operator, slice_ in zip(self, slices))
            )

        return pennylane_observable


def _flatten(args: Sequence, count: int) -> List:
    """Flatten the first ``count`` positional argument groups into one list."""
    return sum([list(args[i]) for i in range(count)], [])
