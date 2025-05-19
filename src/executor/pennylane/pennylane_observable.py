import numpy as np
from typing import Union, List
from sympy import lambdify, sympify

from qiskit.circuit import ParameterExpression
from qiskit.quantum_info import SparsePauliOp

import pennylane as qml
import pennylane.numpy as pnp
import pennylane.pauli as pauli
from pennylane.operation import Observable as PennyLaneObservable

from ..base_classes import QuantumOperatorBase

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


class PennyLaneObservable:
    """
    Class for converting a Qiskit circuit to a PennyLane circuit.

    Args:
        circuit (QuantumCircuit): Qiskit circuit to convert to PennyLane
        observable (Union[None, SparsePauliOp, List[SparsePauliOp], str]): Observable to be measured
                                                                           Can be also a string like ``"probs"`` or ``"state"``
        executor (Executor): Executor object to handle the PennyLane circuit. Has to be initialized with a PennyLane device.

    Attributes:
    -----------

    Attributes:
        pennylane_circuit (qml.qnode): PennyLane circuit that can be called with parameters
        circuit_parameter_names (list): List of circuit parameter names
        observable_parameter_names (list): List of observable parameter names
        circuit_parameter_dimensions (dict): Dictionary with the dimension of each circuit parameter
        observable_parameter_dimension (dict): Dictionary with the dimension of each observable parameter
        circuit_arguments (list): List of all circuit and observable parameters names
        hash (str): Hashable object of the circuit and observable for caching

    Methods:
    --------
    """

    def __init__(
        self,
        observable: Union[QuantumOperatorBase, List[QuantumOperatorBase],],
    ) -> None:


        if isinstance(observable, QuantumOperatorBase):
            self._qiskit_observable = observable._qiskit_operator
            self._num_qubits = self._qiskit_observable.num_qubits
        elif isinstance(observable, list):
            if all([isinstance(obs, QuantumOperatorBase) for obs in observable]):
                self._qiskit_observable = [obs._qiskit_operator for obs in observable]
            else:
                raise ValueError("Unsupported observable type")
            self._num_qubits = self._qiskit_observable[0].num_qubits
        else:
            raise ValueError("Unsupported observable type")

        self._pennylane_obs_param_function = []
        self._pennylane_obs_parameters = []
        self._pennylane_words = []
        self._pennylane_obs_parameters_dimensions = {}

        self.build_observable_instructions(self._qiskit_observable)

    @property
    def parameter_names(self) -> list:
        """List of observable parameter names"""
        return self._pennylane_obs_parameters

    @property
    def observable_parameter_dimensions(self) -> dict:
        """Dictionary with the dimension of each observable parameter"""
        return self._pennylane_obs_parameters_dimensions

    @property
    def hash(self) -> str:
        """Hashable object of the circuit and observable for caching"""
        return hash(str(self._qiskit_observable))

    def build_observable_instructions(self, observable: Union[List[SparsePauliOp], SparsePauliOp]):
        """
        Function to build the instructions for the PennyLane observable from the Qiskit observable.

        This functions converts the Qiskit SparsePauli and parameter expressions to PennyLane
        compatible Pauli words and functions.

        Args:
            observable (Union[List[SparsePauliOp], SparsePauliOp]): Qiskit observable to convert
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
                    [sympify(p._symbol_expr) for p in sort_parameters_after_index(obs.parameters)]
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
                    if coeff._symbol_expr == None:
                        coeff = coeff._coeff
                        if isinstance(coeff, np.complex128) or isinstance(coeff, np.complex64):
                            if np.imag(coeff) != 0:
                                raise ValueError(
                                    "Imaginary part of observable coefficient is not supported"
                                )
                            coeff = float(np.real(coeff))
                        else:
                            coeff = float(coeff)
                    else:
                        symbol_expr = sympify(coeff._symbol_expr)
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
                [pauli.string_to_pauli_word(str(p[::-1])) for p in obs._pauli_list]
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
                [
                    list(args[i])
                    for i in range(len(self._pennylane_obs_parameters))
                ],
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
