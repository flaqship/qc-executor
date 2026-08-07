"""Qulacs operator implementation for quantum expectation value computation.

This wrapper used to hold a Qiskit ``SparsePauliOp`` and read its labels,
coefficients and parameter expressions.  It now compiles the shared sparse Pauli
representation directly, which is what removes Qiskit from this backend.

Two consequences of owning the representation:

* Labels need no reversal.  Qiskit renders qubit 0 rightmost, so the old code
  reversed every label before building the Qulacs term string; both sides now
  index character ``i`` as qubit ``i``.
* Coefficient derivatives come from :func:`sympy.diff` rather than
  ``ParameterExpression.gradient``, so the fallback chain the Qiskit path needed
  when ``gradient`` refused an expression is gone.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple, cast

import numpy as np
import sympy as sp
from qulacs import GeneralQuantumOperator, PauliOperator  # pylint: disable=no-name-in-module
from sympy import lambdify

from ..base import QuantumOperatorBase
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


class QulacsOperator:
    """Qulacs native operator wrapper for expectation value and gradient computation.

    Args:
        operator: The operator, or a list of operators to evaluate together.

    Raises:
        ValueError: If the argument is not an operator or a list of operators.
    """

    @classmethod
    def from_quantum_operator(
        cls, operator: QuantumOperatorBase | List[QuantumOperatorBase]
    ) -> "QulacsOperator":
        """Create a Qulacs native operator from generic operator(s)."""
        return cls(operator)

    def __init__(
        self,
        operator: QuantumOperatorBase | List[QuantumOperatorBase],
    ) -> None:

        if isinstance(operator, QuantumOperatorBase):
            self.multiple_operators = False
            self._irs: List[PauliIR] = [operator.ir]
        elif isinstance(operator, list):
            if not all(isinstance(obs, QuantumOperatorBase) for obs in operator):
                raise ValueError("Unsupported operator type")
            if not operator:
                raise ValueError("Unsupported operator type")
            self.multiple_operators = True
            self._irs = [cast(QuantumOperatorBase, obs).ir for obs in operator]
        else:
            raise ValueError("Unsupported operator type")

        self._num_qubits = self._irs[0].num_qubits

        self.new_operators: List[List[str]] = []
        self.new_operators_coeff: List[List[Callable]] = []
        self.new_operators_coeff_grad: List[List[List[Callable]]] = []
        self.new_operators_used_parameters: List[List[List[Parameter]]] = []
        self._qulacs_op_parameters: Dict[str, int] = {}
        self._free_parameters: set = set()
        self._symbol_tuple_obs: Tuple[Parameter, ...] = ()
        self.build_operator_instructions(self._irs)

        self._outer_jacobi_obs_cache: dict = {}

    @property
    def num_qubits(self) -> int:
        """Number of qubits of the circuit"""
        return self._num_qubits

    @property
    def parameter_names(self) -> list:
        """List of operator parameter names"""
        return list(self._qulacs_op_parameters.keys())

    @property
    def parameter_dimensions(self) -> dict:
        """Dictionary with the dimension of each circuit parameter"""
        return self._qulacs_op_parameters

    @property
    def hash(self) -> bytes:
        """Hashable object of the operator for caching."""
        return b"".join(ir.fingerprint() for ir in self._irs)

    @property
    def free_parameters(self) -> set:
        """Return the set of free (non-bound) parameters in the operator."""
        return self._free_parameters

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
                self._free_parameters.add(parameter)
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

    def build_operator_instructions(self, operators: Sequence[PauliIR]) -> None:
        """Compile the sparse Pauli representation into Qulacs terms and callables.

        Args:
            operators: One representation per observable to evaluate.
        """
        self._qulacs_op_parameters = {}
        for ir in operators:
            for parameter in sort_parameters(ir.free_parameters):
                name = parameter.vector_name
                self._qulacs_op_parameters[name] = self._qulacs_op_parameters.get(name, 0) + 1

        # One symbol slot per parameter occurrence, in the same order the
        # dimensions above were counted, so the executor's flattened argument
        # list lines up with what the lambdified callables expect.
        self._symbol_tuple_obs = tuple(
            parameter for ir in operators for parameter in sort_parameters(ir.free_parameters)
        )

        self.new_operators = []
        self.new_operators_coeff = []
        self.new_operators_coeff_grad = []
        self.new_operators_used_parameters = []

        for ir in operators:
            terms: List[str] = []
            coeff_funcs: List[Callable] = []
            grad_funcs: List[List[Callable]] = []
            used_parameters: List[List[Parameter]] = []

            for label, coeff in zip(ir.to_labels(), ir.coeffs):
                coeff_func, term_grads, term_parameters = self._build_coeff_functions(coeff)
                terms.append(_pauli_term(label))
                coeff_funcs.append(coeff_func)
                grad_funcs.append(term_grads)
                used_parameters.append(term_parameters)

            self.new_operators.append(terms)
            self.new_operators_coeff.append(coeff_funcs)
            self.new_operators_coeff_grad.append(grad_funcs)
            self.new_operators_used_parameters.append(used_parameters)

    def get_operator_func(self) -> Callable:
        """Returns the Qulacs operator function for the operator depending on parameters."""

        def operator_func(*args):

            list_operators = []
            for i, operator in enumerate(self.new_operators):
                new_operator = GeneralQuantumOperator(self.num_qubits)
                for j, term in enumerate(operator):
                    new_operator.add_operator(self.new_operators_coeff[i][j](*args), term)
                list_operators.append(new_operator)

            return list_operators

        return operator_func

    def get_gradient_outer_jacobian_operators_new(
        self,
        gradient_parameters: "Parameter | List[Parameter] | None" = None,
    ) -> Callable:
        """Returns the outer jacobian needed for the chain rule in circuit derivatives.

        Qulacs does not support multiple parameters and parameter expressions,
        so we need to calculate a transformation which also includes the gradient of the
        parameter expression.

        Args:
            gradient_parameters (Parameter | List[Parameter] | None):
                Parameters to calculate the gradient for.
        """

        if isinstance(gradient_parameters, Parameter):
            gradient_parameters = [gradient_parameters]
        gradient_parameters = list(gradient_parameters) if gradient_parameters is not None else []
        gradient_param_dict = {p: i for i, p in enumerate(gradient_parameters)}

        def outer_jacobian(*args):

            # Collects the args values connected to the operator parameters
            op_param_list = sum([list(args[i]) for i in range(len(self.parameter_names))], [])

            outer_jacobians = []

            for iop, operator in enumerate(self.new_operators_coeff_grad):

                relevant_operations = [
                    i
                    for i in range(len(operator))
                    if any(
                        param in gradient_parameters
                        for param in self.new_operators_used_parameters[iop][i]
                    )
                ]

                outer_jacobian = np.zeros((len(relevant_operations), len(gradient_parameters)))
                for i, operation in enumerate(relevant_operations):
                    for j, param in enumerate(self.new_operators_used_parameters[iop][operation]):
                        if param in gradient_parameters:
                            outer_jacobian[i, gradient_param_dict[param]] = (
                                self.new_operators_coeff_grad[iop][operation][j](*op_param_list)
                            )
                outer_jacobians.append(outer_jacobian)
            return outer_jacobians

        return outer_jacobian

    def get_operators_for_gradient(
        self,
        gradient_parameters: "Parameter | List[Parameter] | None" = None,
    ) -> Callable:
        """Returns the Qulacs operator function for the operators depending on parameters."""

        if isinstance(gradient_parameters, Parameter):
            gradient_parameters = [gradient_parameters]
        gradient_parameters = list(gradient_parameters) if gradient_parameters is not None else []

        def operator_func(*_args):

            list_operators = []
            for iop, operator in enumerate(self.new_operators):

                relevant_operations = [
                    i
                    for i in range(len(operator))
                    if any(
                        param in gradient_parameters
                        for param in self.new_operators_used_parameters[iop][i]
                    )
                ]

                list_paulis = []
                for op in relevant_operations:
                    list_paulis.append(PauliOperator(operator[op], 1.0))
                list_operators.append(list_paulis)

            return list_operators

        return operator_func
