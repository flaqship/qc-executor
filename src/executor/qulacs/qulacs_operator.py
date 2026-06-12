"""Qulacs operator implementation for quantum expectation value computation."""

from __future__ import annotations

from typing import Any, List, cast

import numpy as np
import sympy as sp
from qiskit.circuit import ParameterExpression
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.quantum_info import SparsePauliOp
from qulacs import GeneralQuantumOperator, PauliOperator  # pylint: disable=no-name-in-module
from sympy import lambdify

from ..base import QuantumOperatorBase
from ..utils.qiskit_compat import _param_free_symbols, _param_to_sympy


class QulacsOperator:
    """Qulacs native operator wrapper for expectation value and gradient computation."""

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
            self._qiskit_operator = cast(Any, operator)._qiskit_operator
            self._num_qubits = self._qiskit_operator.num_qubits
        elif isinstance(operator, list):
            if all(isinstance(obs, QuantumOperatorBase) for obs in operator):
                self._qiskit_operator = [cast(Any, obs)._qiskit_operator for obs in operator]
            else:
                raise ValueError("Unsupported operator type")
            self._num_qubits = self._qiskit_operator[0].num_qubits
        else:
            raise ValueError("Unsupported operator type")

        self.new_operators = []
        self.new_operators_coeff = []
        self.new_operators_coeff_grad = []
        self.new_operators_used_parameters = []
        self._qulacs_op_parameters = {}
        self._free_parameters = set()
        self.build_operator_instructions(self._qiskit_operator)

        self._outer_jacobi_obs_cache = {}

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
    def hash(self) -> str:
        """Hashable object of the circuit and operator for caching"""
        return str(self._qiskit_operator)

    @property
    def free_parameters(self) -> set:
        """Return the set of free (non-bound) parameters in the operator."""
        return self._free_parameters

    def _build_coeff_functions(self, c):
        """Build coefficient function and gradient functions for a single operator term.

        Args:
            c: The coefficient, which may be a ParameterVectorElement,
               ParameterExpression, or a plain numeric value.

        Returns:
            Tuple of (coeff_func, grad_funcs, used_params).
        """
        if isinstance(c, ParameterVectorElement):
            self._free_parameters.add(c)
            return lambdify(self._symbol_tuple_obs, _param_to_sympy(c)), [lambda *arg: 1.0], [c]

        if isinstance(c, ParameterExpression):
            coeff_func = lambdify(self._symbol_tuple_obs, _param_to_sympy(c))
            grad_funcs = []
            used_params = []
            for param_element in _param_free_symbols(c):
                self._free_parameters.add(param_element)
                used_params.append(param_element)
                try:
                    param_grad = c.gradient(param_element)
                except (
                    TypeError,
                    ValueError,
                    AttributeError,
                    NotImplementedError,
                ):
                    c_sym = _param_to_sympy(c)
                    p_sym = _param_to_sympy(param_element)
                    param_grad = c_sym.diff(p_sym)
                if isinstance(param_grad, complex) and param_grad.imag == 0:
                    param_grad = param_grad.real
                if isinstance(param_grad, (float, complex)):
                    grad_funcs.append(lambda *arg, param_grad=param_grad: param_grad)
                elif isinstance(param_grad, sp.Basic):
                    grad_funcs.append(lambdify(self._symbol_tuple_obs, param_grad))
                else:
                    grad_funcs.append(
                        lambdify(self._symbol_tuple_obs, _param_to_sympy(param_grad))
                    )
            return coeff_func, grad_funcs, used_params

        return lambda *arg, c=c: c, [lambda *arg: 0.0], []

    def build_operator_instructions(self, operator: List[SparsePauliOp] | SparsePauliOp):
        """
        Function to build the instructions for the Qulacs operator from the Qiskit operator.

        This functions converts the Qiskit SparsePauli and parameter expressions to Qulacs
        compatible Pauli words and functions.

        Args:
            operator (List[SparsePauliOp] | SparsePauliOp): Qiskit operator to convert
                                                                    to Qulacs

        Returns:
            Tuple with lists of Qulacs operator parameter functions, Qulacs Pauli words,
            Qulacs operator parameters and Qulacs operator parameter dimensions
        """

        self.multiple_operators = False
        if isinstance(operator, SparsePauliOp):
            operator = [operator]
        elif isinstance(operator, list):
            self.multiple_operators = True
        else:
            raise ValueError("Unsupported operator type")

        self._symbol_tuple_obs = tuple()

        self._qulacs_op_parameters = {}

        for op in operator:
            for param in op.parameters:
                name = param.vector.name
                if name not in self._qulacs_op_parameters:
                    self._qulacs_op_parameters[name] = 1
                else:
                    self._qulacs_op_parameters[name] += 1

        self._symbol_tuple_obs = tuple(
            sum(
                [
                    [
                        _param_to_sympy(p)
                        for p in sorted(op.parameters, key=lambda param: param.index)
                    ]
                    for op in operator
                ],
                [],
            )
        )

        # new version
        self.new_operators = []
        self.new_operators_coeff = []
        self.new_operators_coeff_grad = []
        self.new_operators_used_parameters = []
        for op in operator:

            paulis = [str(p) for p in op.paulis]
            coeff = list(np.real_if_close(np.asarray(cast(Any, op.coeffs))))

            new_operator = []
            new_operators_coeff = []
            new_operators_coeff_grad = []
            new_operators_used_parameters = []
            for c, p in zip(coeff, paulis):
                string = " ".join(f"{p_} {i}" for i, p_ in enumerate(p)) + " "
                coeff_func, grad_funcs, used_params = self._build_coeff_functions(c)
                new_operator.append(string)
                new_operators_coeff.append(coeff_func)
                new_operators_coeff_grad.append(grad_funcs)
                new_operators_used_parameters.append(used_params)

            self.new_operators.append(new_operator)
            self.new_operators_coeff.append(new_operators_coeff)
            self.new_operators_coeff_grad.append(new_operators_coeff_grad)
            self.new_operators_used_parameters.append(new_operators_used_parameters)

    def get_operator_func(self):
        """Returns the Qulacs operator function for the operator depending on parameters."""

        def operator_func(*args):

            list_operators = []
            for i, operator in enumerate(self.new_operators):
                new_operator = GeneralQuantumOperator(self.num_qubits)
                for j, op in enumerate(operator):
                    new_operator.add_operator(self.new_operators_coeff[i][j](*args), op)
                list_operators.append(new_operator)

            return list_operators

        return operator_func

    def get_gradient_outer_jacobian_operators_new(
        self,
        gradient_parameters: ParameterVectorElement | List[ParameterVectorElement] | None = None,
    ):
        """Returns the outer jacobian needed for the chain rule in circuit derivatives.

        Qulacs does not support multiple parameters and parameter expressions,
        so we need to calculate a transformation which also includes the gradient of the
        parameter expression.

        Args:
            gradient_parameters (ParameterVectorElement | List[ParameterVectorElement] | None):
                Parameters to calculate the gradient for.
        """

        if isinstance(gradient_parameters, ParameterVectorElement):
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
        gradient_parameters: ParameterVectorElement | List[ParameterVectorElement] | None = None,
    ):
        """Returns the Qulacs operator function for the operators depending on parameters."""

        if isinstance(gradient_parameters, ParameterVectorElement):
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
