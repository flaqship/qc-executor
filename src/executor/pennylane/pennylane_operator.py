from __future__ import annotations

from typing import List

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import pennylane.pauli as pauli
from qiskit.circuit import ParameterExpression
from qiskit.quantum_info import SparsePauliOp
from sympy import lambdify

from ..base import QuantumOperatorBase
from ..qiskit.qiskit_operator import to_qiskit_operator
from ..utils.qiskit_compat import _param_is_constant, _param_to_sympy


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
    from sympy.printing.numpy import NumPyPrinter as Printer

    user_functions = {}
    printer = Printer(
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
            self._qiskit_operator = to_qiskit_operator(operator)
            self._num_qubits = self._qiskit_operator.num_qubits
        elif isinstance(operator, list):
            if all([isinstance(op, QuantumOperatorBase) for op in operator]):
                self._qiskit_operator = [to_qiskit_operator(op) for op in operator]
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
            pennylane_operator_param_function_ = []
            for coeff in op.coeffs:
                if isinstance(coeff, ParameterExpression):
                    if _param_is_constant(coeff):
                        coeff = complex(coeff)
                        if isinstance(coeff, (np.complex128, np.complex64, complex)):
                            if np.imag(coeff) != 0:
                                raise ValueError(
                                    "Imaginary part of operator coefficient is not supported"
                                )
                            coeff = float(np.real(coeff))
                        else:
                            coeff = float(coeff)
                    else:
                        symbol_expr = _param_to_sympy(coeff)
                        f = lambdify(symbol_tuple, symbol_expr, modules=modules, printer=printer)
                        pennylane_operator_param_function_.append(f)
                else:
                    if isinstance(coeff, np.complex128) or isinstance(coeff, np.complex64):
                        if np.imag(coeff) != 0:
                            raise ValueError(
                                "Imaginary part of operator coefficient is not supported"
                            )
                        coeff = float(np.real(coeff))
                    else:
                        coeff = float(coeff)
                    pennylane_operator_param_function_.append(coeff)
            self._pennylane_operator_param_functions.append(pennylane_operator_param_function_)

        # Convert Pauli strings into PennyLane Pauli words. Qiskit labels are
        # little-endian (rightmost char = qubit 0) whereas PennyLane's
        # ``string_to_pauli_word`` assigns left-to-right (leftmost char = wire 0),
        # so the label is reversed to keep operator wires aligned with the circuit.
        for op in operator:
            self._pennylane_words.append(
                [pauli.string_to_pauli_word(p[::-1]) for p in op.paulis.to_labels()]
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
                        coeff_list = []
                        for coeff in self._pennylane_operator_param_functions[i]:
                            if callable(coeff):
                                evaluated_param = coeff(*obs_param_list)
                                coeff_list.append(evaluated_param)
                            else:
                                coeff_list.append(coeff)
                        expval_list.append(qml.expval(qml.Hamiltonian(coeff_list, obs)))
                    else:
                        # No symbolic parameters: coefficients are plain numbers.
                        # Weight each Pauli word by its numeric coefficient (a bare
                        # sum would drop them) and measure the combined observable.
                        if len(self._pennylane_words[i]) == 0:
                            expval_list.append(0.0)
                        else:
                            expval_list.append(
                                qml.expval(
                                    sum(
                                        coeff * word
                                        for coeff, word in zip(
                                            self._pennylane_operator_param_functions[i],
                                            self._pennylane_words[i],
                                        )
                                    )
                                )
                            )
                return pnp.stack(tuple(expval_list))
            else:
                if len(obs_param_list) > 0:
                    coeff_list = []
                    for coeff in self._pennylane_operator_param_functions:
                        if callable(coeff):
                            evaluated_param = coeff(*obs_param_list)
                            coeff_list.append(evaluated_param)
                        else:
                            coeff_list.append(coeff)
                    return qml.expval(qml.Hamiltonian(coeff_list, self._pennylane_words))
                else:
                    # No symbolic parameters: coefficients are plain numbers.
                    # Weight each Pauli word by its numeric coefficient (a bare
                    # sum would drop them) and measure the combined observable.
                    if len(self._pennylane_words) == 0:
                        return 0.0
                    else:
                        return qml.expval(
                            sum(
                                coeff * word
                                for coeff, word in zip(
                                    self._pennylane_operator_param_functions,
                                    self._pennylane_words,
                                )
                            )
                        )

        return pennylane_observable
