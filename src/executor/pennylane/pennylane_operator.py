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
        observable (QuantumOperatorBase | list[QuantumOperatorBase]):
            Operator definition(s) to convert.
    """

    @classmethod
    def from_quantum_operator(cls, operator: QuantumOperatorBase) -> "PennyLaneOperator":
        """Create a PennyLane native operator from a generic operator."""
        return cls(operator)

    def __init__(
        self,
        observable: QuantumOperatorBase | List[QuantumOperatorBase],
    ) -> None:

        if isinstance(observable, QuantumOperatorBase):
            self._qiskit_operator = observable._qiskit_operator
            self._num_qubits = self._qiskit_operator.num_qubits
        elif isinstance(observable, list):
            if all([isinstance(obs, QuantumOperatorBase) for obs in observable]):
                self._qiskit_operator = [obs._qiskit_operator for obs in observable]
            else:
                raise ValueError("Unsupported observable type")
            self._num_qubits = self._qiskit_operator[0].num_qubits
        else:
            raise ValueError("Unsupported observable type")

        self._pennylane_obs_param_function = []
        self._pennylane_obs_parameters = []
        self._pennylane_words = []
        self._pennylane_obs_parameters_dimensions = {}

        self.build_observable_instructions(self._qiskit_operator)

    @property
    def parameter_names(self) -> list:
        """List of observable parameter names"""
        return self._pennylane_obs_parameters

    @property
    def parameter_dimensions(self) -> dict:
        """Dictionary with the dimension of each observable parameter"""
        return self._pennylane_obs_parameters_dimensions

    @property
    def hash(self) -> int:
        """Hashable object of the circuit and operator for caching"""
        return hash(str(self._qiskit_operator))

    def build_observable_instructions(self, observable: List[SparsePauliOp] | SparsePauliOp):
        """
        Function to build the instructions for the PennyLane observable from the Qiskit observable.

        This functions converts the Qiskit SparsePauli and parameter expressions to PennyLane
        compatible Pauli words and functions.

        Args:
            observable (List[SparsePauliOp] | SparsePauliOp): Qiskit observable to convert
                                                                    to PennyLane

        Returns:
            Tuple with lists of PennyLane observable parameter functions, PennyLane Pauli words,
            PennyLane observable parameters and PennyLane observable parameter dimensions
        """

        self._pennylane_obs_param_function = []
        self._pennylane_obs_parameters = []
        self._pennylane_words = []
        self._pennylane_obs_parameters_dimensions = {}

        islist = True
        if not isinstance(observable, list):
            islist = False
            observable = [observable]

        def sort_parameters_after_index(parameter_vector):
            index_list = [p.index for p in parameter_vector]
            argsort_list = np.argsort(index_list)
            return [parameter_vector[i] for i in argsort_list]

        printer, modules = _get_sympy_interface()

        for obs in observable:
            for param in obs.parameters:
                if param.vector.name not in self._pennylane_obs_parameters:
                    self._pennylane_obs_parameters.append(param.vector.name)
                    self._pennylane_obs_parameters_dimensions[param.vector.name] = 1
                else:
                    self._pennylane_obs_parameters_dimensions[param.vector.name] += 1

        # Handle observable parameter expressions and convert them to compatible python functions

        symbol_tuple = tuple(
            sum(
                [
                    [_param_to_sympy(p) for p in sort_parameters_after_index(obs.parameters)]
                    for obs in observable
                ],
                [],
            )
        )

        self._pennylane_obs_param_function = []
        for obs in observable:
            pennylane_obs_param_function_ = []
            for coeff in obs.coeffs:
                if isinstance(coeff, ParameterExpression):
                    if _param_is_constant(coeff):
                        coeff = complex(coeff)
                        if isinstance(coeff, (np.complex128, np.complex64, complex)):
                            if np.imag(coeff) != 0:
                                raise ValueError(
                                    "Imaginary part of observable coefficient is not supported"
                                )
                            coeff = float(np.real(coeff))
                        else:
                            coeff = float(coeff)
                    else:
                        symbol_expr = _param_to_sympy(coeff)
                        f = lambdify(symbol_tuple, symbol_expr, modules=modules, printer=printer)
                        pennylane_obs_param_function_.append(f)
                else:
                    if isinstance(coeff, np.complex128) or isinstance(coeff, np.complex64):
                        if np.imag(coeff) != 0:
                            raise ValueError(
                                "Imaginary part of observable coefficient is not supported"
                            )
                        coeff = float(np.real(coeff))
                    else:
                        coeff = float(coeff)
                    pennylane_obs_param_function_.append(coeff)
            self._pennylane_obs_param_function.append(pennylane_obs_param_function_)

        # Convert Pauli strings into PennyLane Pauli words
        for obs in observable:
            self._pennylane_words.append(
                [pauli.string_to_pauli_word(p) for p in obs.paulis.to_labels()]
            )

        if not islist:
            self._pennylane_obs_param_function = self._pennylane_obs_param_function[0]
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
                [list(args[i]) for i in range(len(self._pennylane_obs_parameters))],
                [],
            )

            if isinstance(self._qiskit_observable, list):
                expval_list = []
                for i, obs in enumerate(self._pennylane_words):
                    if len(obs_param_list) > 0:
                        coeff_list = []
                        for coeff in self._pennylane_obs_param_function[i]:
                            if callable(coeff):
                                evaluated_param = coeff(*obs_param_list)
                                coeff_list.append(evaluated_param)
                            else:
                                coeff_list.append(coeff)
                        expval_list.append(qml.expval(qml.Hamiltonian(coeff_list, obs)))
                    else:
                        # In case no parameters are present in the observable
                        # Calculate the expectation value of sum of the observables
                        # since this is more compatible with hardware backends
                        if len(self._pennylane_words[i]) == 0:
                            expval_list.append(0.0)
                        else:
                            expval_list.append(
                                qml.expval(sum([obs for obs in self._pennylane_words[i]]))
                            )
                return pnp.stack(tuple(expval_list))
            else:
                if len(obs_param_list) > 0:
                    coeff_list = []
                    for coeff in self._pennylane_obs_param_function:
                        if callable(coeff):
                            evaluated_param = coeff(*obs_param_list)
                            coeff_list.append(evaluated_param)
                        else:
                            coeff_list.append(coeff)
                    return qml.expval(qml.Hamiltonian(coeff_list, self._pennylane_words))
                else:
                    # In case no parameters are present in the observable
                    # Calculate the expectation value of sum of the observables
                    # since this is more compatible with hardware backends
                    if len(self._pennylane_words) == 0:
                        return 0.0
                    else:
                        return qml.expval(sum([obs for obs in self._pennylane_words]))

        return pennylane_observable
