"""PennyLane operator conversion utilities for Qiskit-based quantum operators."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, cast

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from pennylane import pauli
from qiskit.circuit import ParameterExpression
from qiskit.quantum_info import SparsePauliOp
from sympy import lambdify
from sympy.printing.numpy import NumPyPrinter as _NumPyPrinter

from ..base import QuantumOperatorBase
from ..utils.qiskit_compat import _param_is_constant, _param_to_sympy

if TYPE_CHECKING:
    from ..quantum_operator import QuantumOperator


def _get_sympy_interface():
    """
    Returns the sympy interface that is used in the parameter conversion.

    Necessary for the correct conversion of sympy expressions in Qiskit to
    python functions in PennyLane.

    Returns:
        Tuple of sympy printer and sympy modules
    """
    # SymPy printer for pennylane numpy implementation has to be set manually,
    # otherwise math functions are used in lambdify instead of pennylane.numpy functions
    user_functions = {}
    printer = _NumPyPrinter(
        {
            "fully_qualified_modules": False,
            "inline": True,
            "allow_unknown_functions": True,
            "user_functions": user_functions,
        }
    )
    # Use Pennylane numpy for sympy lambdify
    modules = pnp

    # The functions down below can be used to switch between different gradient engines
    # as tensorflow, jax and torch. However, this is not supported and implemented yet.

    #     # SymPy printer for pennylane numpy implementation has to be set manually,
    #     # otherwise math functions are used in lambdify instead of pennylane.numpy functions
    #     from sympy.printing.tensorflow import TensorflowPrinter as Printer  # type: ignore

    #     user_functions = {}
    #     printer = Printer(
    #         {
    #             "fully_qualified_modules": False,
    #             "inline": True,
    #             "allow_unknown_functions": True,
    #             "user_functions": user_functions,
    #         }
    #     )  #
    #     modules = tf

    # elif self._gradient_engine == "jax":
    #     from sympy.printing.numpy import JaxPrinter as Printer  # type: ignore

    #     user_functions = {}
    #     printer = Printer(
    #         {
    #             "fully_qualified_modules": False,
    #             "inline": True,
    #             "allow_unknown_functions": True,
    #             "user_functions": user_functions,
    #         }
    #     )  #
    #     modules = jnp
    # elif self._gradient_engine == "torch" or self._gradient_engine == "pytorch":
    #     from sympy.printing.pycode import PythonCodePrinter as Printer  # type: ignore

    #     user_functions = {}
    #     printer = Printer(
    #         {
    #             "fully_qualified_modules": False,
    #             "inline": True,
    #             "allow_unknown_functions": True,
    #             "user_functions": user_functions,
    #         }
    #     )  #
    #     modules = torch

    # else:
    #     # tbd for jax and tensorflow
    #     printer = None
    #     modules = None

    return printer, modules


def _resolve_coefficient(coeff, symbol_tuple, printer, modules):
    """Convert a Qiskit operator coefficient to a float or PennyLane-compatible callable."""
    if isinstance(coeff, ParameterExpression):
        if not _param_is_constant(coeff):
            symbol_expr = _param_to_sympy(coeff)
            return lambdify(symbol_tuple, symbol_expr, modules=modules, printer=printer)
        coeff = complex(coeff)
    if np.imag(coeff) != 0:
        raise ValueError("Imaginary part of operator coefficient is not supported")
    return float(np.real(coeff))


class PennyLaneOperator:
    """Convert generic quantum operators to PennyLane-native operators.

    Args:
        operator (QuantumOperatorBase | list[QuantumOperatorBase]):
            Operator definition(s) to convert.
    """

    @classmethod
    def from_quantum_operator(cls, operator: QuantumOperatorBase) -> "PennyLaneOperator":
        """Create a PennyLane native operator from a generic operator."""
        return cls(operator)

    def __init__(
        self,
        operator: QuantumOperatorBase | List[QuantumOperatorBase],
    ) -> None:

        if isinstance(operator, QuantumOperatorBase):
            self._qiskit_operator = cast("QuantumOperator", operator).qiskit_operator
            self._num_qubits = self._qiskit_operator.num_qubits
        elif isinstance(operator, list):
            if all(isinstance(op, QuantumOperatorBase) for op in operator):
                self._qiskit_operator = [
                    cast("QuantumOperator", op).qiskit_operator for op in operator
                ]
            else:
                raise ValueError("Unsupported operator type")
            self._num_qubits = self._qiskit_operator[0].num_qubits
        else:
            raise ValueError("Unsupported operator type")

        self._pennylane_operator_param_functions = []
        self._pennylane_operator_parameters = []
        self._pennylane_words = []
        self._pennylane_operator_parameter_dimensions = {}

        self.build_operator_instructions(self._qiskit_operator)

    @property
    def parameter_names(self) -> list:
        """List of operator parameter names"""
        return self._pennylane_operator_parameters

    @property
    def parameter_dimensions(self) -> dict:
        """Dictionary with the dimension of each operator parameter"""
        return self._pennylane_operator_parameter_dimensions

    @property
    def hash(self) -> int:
        """Hashable object of the circuit and operator for caching"""
        return hash(str(self._qiskit_operator))

    def build_operator_instructions(self, operator: List[SparsePauliOp] | SparsePauliOp):
        """
        Function to build the instructions for the PennyLane operator from the Qiskit operator.

        This functions converts the Qiskit SparsePauli and parameter expressions to PennyLane
        compatible Pauli words and functions.

        Args:
            operator (List[SparsePauliOp] | SparsePauliOp): Qiskit operator to convert
                                                                    to PennyLane

        Returns:
            Tuple with lists of PennyLane operator parameter functions, PennyLane Pauli words,
            PennyLane operator parameters and PennyLane operator parameter dimensions
        """

        self._pennylane_operator_param_functions = []
        self._pennylane_operator_parameters = []
        self._pennylane_words = []
        self._pennylane_operator_parameter_dimensions = {}

        islist = True
        if not isinstance(operator, list):
            islist = False
            operator = [operator]

        def sort_parameters_after_index(parameter_vector):
            index_list = [p.index for p in parameter_vector]
            argsort_list = np.argsort(index_list)
            return [parameter_vector[i] for i in argsort_list]

        printer, modules = _get_sympy_interface()

        for op in operator:
            for param in op.parameters:
                if param.vector.name not in self._pennylane_operator_parameters:
                    self._pennylane_operator_parameters.append(param.vector.name)
                    self._pennylane_operator_parameter_dimensions[param.vector.name] = 1
                else:
                    self._pennylane_operator_parameter_dimensions[param.vector.name] += 1

        # Handle operator parameter expressions and convert them to compatible python functions

        symbol_tuple = tuple(
            sum(
                [
                    [_param_to_sympy(p) for p in sort_parameters_after_index(op.parameters)]
                    for op in operator
                ],
                [],
            )
        )

        self._pennylane_operator_param_functions = []
        for op in operator:
            coeffs = op.coeffs if op.coeffs is not None else []
            self._pennylane_operator_param_functions.append(
                [
                    _resolve_coefficient(coeff, symbol_tuple, printer, modules)
                    for coeff in coeffs
                ]
            )

        # Convert Pauli strings into PennyLane Pauli words
        for op in operator:
            self._pennylane_words.append(
                [pauli.string_to_pauli_word(p) for p in op.paulis.to_labels()]
            )

        if not islist:
            self._pennylane_operator_param_functions = self._pennylane_operator_param_functions[0]
            self._pennylane_words = self._pennylane_words[0]

    def build_pennylane_observable(self):
        """
        Function to build the PennyLane circuit from the Qiskit circuit and observable.

        The functions returns a callable PennyLane circuit that can be called with parameters.
        The PennyLane circuit is built from the instructions previously generated from the Qiskit
        circuit and observable.

        Returns:
            Callable PennyLane circuit
        """

        def pennylane_observable(*args):
            """PennyLane circuit that can be called with parameters"""

            # Collects the args values connected to the observable parameters
            obs_param_list = sum(
                [list(args[i]) for i in range(len(self._pennylane_operator_parameters))],
                [],
            )

            if isinstance(self._qiskit_operator, list):
                expval_list = []
                for i, obs in enumerate(self._pennylane_words):
                    if len(obs_param_list) > 0:
                        coeff_list = [
                            coeff(*obs_param_list) if callable(coeff) else coeff
                            for coeff in self._pennylane_operator_param_functions[i]
                        ]
                        expval_list.append(qml.expval(qml.Hamiltonian(coeff_list, obs)))
                    else:
                        # In case no parameters are present in the observable
                        # Calculate the expectation value of sum of the observables
                        # since this is more compatible with hardware backends
                        if len(self._pennylane_words[i]) == 0:
                            expval_list.append(0.0)
                        else:
                            expval_list.append(
                                qml.expval(qml.sum(*self._pennylane_words[i]))
                            )
                return pnp.stack(tuple(expval_list))
            if len(obs_param_list) > 0:
                coeff_list = [
                    coeff(*obs_param_list) if callable(coeff) else coeff
                    for coeff in self._pennylane_operator_param_functions
                ]
                return qml.expval(qml.Hamiltonian(coeff_list, self._pennylane_words))
            # In case no parameters are present in the observable
            # Calculate the expectation value of sum of the observables
            # since this is more compatible with hardware backends
            if len(self._pennylane_words) == 0:
                return 0.0
            return qml.expval(qml.sum(*self._pennylane_words))

        return pennylane_observable
