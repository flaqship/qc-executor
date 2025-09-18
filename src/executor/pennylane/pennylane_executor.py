import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union
from collections import Counter
from itertools import product
from qiskit.circuit import ParameterExpression, Parameter, ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement


import pennylane as qml
import pennylane.numpy as pnp

from ..base import QuantumOperatorBase, QuantumCircuitBase, ExecutorBase

from .pennylane_circuit import PennyLaneCircuit
from .pennylane_observable import PennyLaneObservable

from ..utils.data_preprocessing import adjust_features, adjust_parameters, to_tuple

class PennylaneExecutor(ExecutorBase):
    """Base class for quantum circuit executors.

    Args:
        shots (int, optional): Number of shots for sampling. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        log_file (str, optional): Path to the log file. Defaults to None.
        caching (bool, optional): Whether to use caching. Defaults to None.
        cache_dir (str, optional): Directory for caching. Defaults to "cache".
    """

    def __init__(
        self,
        shots: Union[int, None] = None,
        seed: Union[int, None] = None,
        log_file: Union[str, None] = None,
        caching: Union[bool, None] = None,
        cache_dir: str = "cache",
    ):

        super().__init__(
            shots=shots, seed=seed, log_file=log_file, caching=caching, cache_dir=cache_dir
        )

        self._circuit_cache = {}
        self._operator_cache = {}

        if seed is not None:
            self._random = np.random.default_rng(seed)
        else:
            self._random = np.random.default_rng()

        self._device = qml.device("default.qubit", wires=1)

    @property
    def shots(self) -> Union[int, None]:
        """Return the number of shots."""
        return self._shots

    @shots.setter
    def shots(self, value: Union[int, None]) -> None:
        """Set the number of shots."""
        raise NotImplementedError

    @property
    def remote(self) -> bool:
        """Return True if the execution access a remote backend."""
        return False

    def _preprocess_circuits(self, circuit: QuantumCircuitBase):

        multiple_circuits = True
        circuits = circuit
        if not isinstance(circuit,List):
            circuits = [circuit]
            multiple_circuits = False

        qulacs_circuits = []

        # Check the cache for already converted circuits
        for circ in circuits:
            if circ in self._circuit_cache:
                qulacs_circuits.append(self._circuit_cache[circ])
            else:
                qulacs_circuit = PennyLaneCircuit(circ)
                self._circuit_cache[circ] = qulacs_circuit
                qulacs_circuits.append(qulacs_circuit)

        return qulacs_circuits, multiple_circuits

    def _preprocess_operators(self, operator: QuantumOperatorBase):

        # todo operator cache
        multiple_operators = True
        operators = operator
        if not isinstance(operator, List):
            operators = [operator]
            multiple_operators = False
        qulacs_observables = []

        for op in operators:
            if op in self._operator_cache:
                qulacs_observables.append(self._operator_cache[op])
            else:
                qulacs_observable = PennyLaneObservable(op)
                self._operator_cache[op] = qulacs_observable
                qulacs_observables.append(qulacs_observable)

        return qulacs_observables, multiple_operators

    def expectation_value(
        self, circuit: QuantumCircuitBase, operator: QuantumOperatorBase, **parameter_values
    ) -> float:
        """
        Calculate the expectation value of the operator with respect to the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.
            operator (QuantumOperatorBase): The quantum operator.

        Returns:
            float: The expectation value.
        """

        pennylane_circuits, multiple_circuits = self._preprocess_circuits(circuit)
        pennylane_observables, multiple_operators = self._preprocess_operators(operator)

        if circuit.num_qubits != len(self._device.wires):
            self._device = qml.device(self._device.name, wires=circuit.num_qubits)

        values = []

        for pennylane_circuit in pennylane_circuits:

            circuit_parameters = []
            multiple_circuit_parameters = []
            circuit_parameters_dimension = []

            circuit_values = []
            for param in pennylane_circuit.parameter_names:
                if param not in parameter_values:
                    raise ValueError(f"Parameter '{param}' not found in provided parameter values.")

                param_values, multiple_params = adjust_features(parameter_values[param], pennylane_circuit.parameter_dimensions[param])
                circuit_parameters.append(param_values)
                multiple_circuit_parameters.append(multiple_params)
                circuit_parameters_dimension.append(pennylane_circuit.parameter_dimensions[param])

            circuit_parameter_tuples = product(*circuit_parameters)

            for pennylane_observable in pennylane_observables:
                observable_values = []
                observable_parameters = []
                multiple_observable_parameters = []
                observable_parameters_dimension = []
                for param in pennylane_observable.parameter_names:
                    if param not in parameter_values:
                        raise ValueError(f"Parameter '{param}' not found in provided parameter values.")

                    param_values, multiple_params = adjust_features(parameter_values[param], pennylane_observable.parameter_dimensions[param])
                    observable_parameters.append(param_values)
                    multiple_observable_parameters.append(multiple_params)
                    observable_parameters_dimension.append(pennylane_observable.parameter_dimensions[param])

                observable_parameter_tuples = product(*observable_parameters)

                @qml.qnode(self._device)
                def circuit_func(*args):
                    pennylane_circuit.build_pennylane_circuit()(*args)
                    return pennylane_observable.build_pennylane_observable()(
                        *args[len(pennylane_circuit.parameter_names) :]
                    )

                for cp in circuit_parameter_tuples:
                    cp_values = []
                    for op in observable_parameter_tuples:
                        cp_values.append(circuit_func(*cp, *op))

                    observable_values.append(cp_values)
                circuit_values.append(observable_values)
            values.append(circuit_values)
        values = np.array(values)

        shape = list(values.shape)
        num_circuit_param = shape.pop(-1)
        if num_circuit_param > 1:
            raise NotImplementedError("Multiple parameters per circuit not supported yet.")
        num_obs_param = shape.pop(-1)
        if num_obs_param > 1:
            raise NotImplementedError("Multiple parameters per observable not supported yet.")
        values = values.reshape(shape)

        if not multiple_circuits:
            values = values[0]
            if not multiple_operators:
                values = values[0]
        else:
            if not multiple_operators:
                values = values.reshape(-1)

        return values

    def expectation_value_derivatives(
        self,
        circuit: QuantumCircuitBase,
        operator: QuantumOperatorBase,
        *values: Union[
            str,
            ParameterVector,
            ParameterVectorElement,
            tuple,
        ],
        **parameter_values,
    ) -> Union[np.array, dict]:
        """
        Calculate the derivatives of the expectation value with respect to the parameters

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.
            operator (QuantumOperatorBase): The quantum operator.
            values: Values for which the derivatives are calculated. Can be strings (e.g.
                "expectation_value" or the name of parameters), or
                ParameterVectors, ParameterVectorElements. Tuples are used for higher
                order derivatives.
            parameter_values: Parameters to evaluate the circuit and observable given as
                keyword arguments.

        Returns:
            Union[np.array, dict]: The derivatives of the expectation value. If a single value
                is provided, a numpy array is returned. If multiple values are provided, a
                dictionary with the values as keys and the derivatives as values is returned.
        """


        def remove_brackets(s: str) -> str:
            return re.sub(r"\[.*?\]", "", s)

        pennylane_circuits, multiple_circuits = self._preprocess_circuits(circuit)
        pennylane_observables, multiple_operators = self._preprocess_operators(operator)

        # TODO: multiple circuits and operators not implemented yet
        pennylane_circuit = pennylane_circuits[0]
        pennylane_observable = pennylane_observables[0]

        circuit_parameters = []
        multiple_circuit_parameters = []
        circuit_parameters_dimension = []

        # preprocess the parameter values
        for param in pennylane_circuit.parameter_names:
            if param not in parameter_values:
                raise ValueError(f"Parameter '{param}' not found in provided parameter values.")

            param_values, multiple_params = adjust_features(parameter_values[param], pennylane_circuit.parameter_dimensions[param])
            circuit_parameters.append(param_values[0])
            multiple_circuit_parameters.append(multiple_params)
            circuit_parameters_dimension.append(pennylane_circuit.parameter_dimensions[param])

        observable_parameters = []
        multiple_observable_parameters = []
        observable_parameters_dimension = []
        for param in pennylane_observable.parameter_names:
            if param not in parameter_values:
                raise ValueError(f"Parameter '{param}' not found in provided parameter values.")

            param_values, multiple_params = adjust_features(parameter_values[param], pennylane_observable.parameter_dimensions[param])
            observable_parameters.append(param_values[0])
            multiple_observable_parameters.append(multiple_params)
            observable_parameters_dimension.append(pennylane_observable.parameter_dimensions[param])

        result_dict = {}

        if values is None or len(values) == 0:
            values = ("expectation_value",)

        # Convert and sort the values
        values = list(values)
        indices = np.argsort([str(t) for t in values])
        values = [values[i] for i in indices]
        values = [to_tuple(v) for v in values]

        if circuit.num_qubits != len(self._device.wires):
            self._device = qml.device(self._device.name, wires=circuit.num_qubits)

        @qml.qnode(self._device)
        def circuit_func(*args):
            pennylane_circuit.build_pennylane_circuit()(*args)
            return pennylane_observable.build_pennylane_observable()(
                *args[len(pennylane_circuit.parameter_names) :]
            )

        argnum_dict = {}
        argnum=0
        for param in pennylane_circuit.parameter_names:
            argnum_dict[param] = argnum
            argnum+=1
        for param in pennylane_observable.parameter_names:
            argnum_dict[param] = argnum
            argnum+=1

        print(argnum_dict)

        # Loop over all requested derivatives
        for todo in values:
            print(todo)




    def statevector(self, circuit: QuantumCircuitBase, **parameter_values) -> np.ndarray:
        """
        Get the statevector of the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.

        Returns:
            np.ndarray: The statevector of the circuit.
        """

        def reverse_bits_array(n, num_bits):
            indices = np.arange(n)
            reversed_indices = np.zeros_like(indices)
            for _ in range(num_bits):
                # Shift left to make space for the next bit
                reversed_indices = (reversed_indices << 1) | (indices & 1)
                # Shift original indices right to move to the next bit
                indices >>= 1
            return reversed_indices

        pennylane_circuits, multiple_circuits = self._preprocess_circuits(circuit)

        state_vectors = []
        for pennylane_circuit in pennylane_circuits:

            circuit_parameters = []
            multiple_circuit_parameters = []
            circuit_parameters_dimension = []
            circuit_values = []
            for param in pennylane_circuit.parameter_names:
                if param not in parameter_values:
                    raise ValueError(f"Parameter '{param}' not found in provided parameter values.")
                param_values, multiple_params = adjust_features(parameter_values[param], pennylane_circuit.parameter_dimensions[param])
                circuit_parameters.append(param_values)
                multiple_circuit_parameters.append(multiple_params)
                circuit_parameters_dimension.append(pennylane_circuit.parameter_dimensions[param])

            circuit_parameter_tuples = product(*circuit_parameters)

            if pennylane_circuit.num_qubits != len(self._device.wires):
                self._device = qml.device(self._device.name, wires=circuit.num_qubits)

            @qml.qnode(self._device)
            def circuit_func(*args):
                pennylane_circuit.build_pennylane_circuit()(*args)
                return qml.state()

            for cp in circuit_parameter_tuples:
                state_wrong_sort = np.array(circuit_func(*cp))
                n = len(state_wrong_sort)
                num_bits = circuit.num_qubits
                indices = reverse_bits_array(n, num_bits)
                circuit_values.append(state_wrong_sort[indices])
            state_vectors.append(circuit_values)

        state_vectors = np.array(state_vectors)
        # Remove the parameter dimension list (has to be fixed for multiple parameters)
        shape = list(state_vectors.shape)
        shape.pop(1)
        state_vectors = state_vectors.reshape(shape)

        if not multiple_circuits:
            state_vectors = state_vectors[0]

        return state_vectors
