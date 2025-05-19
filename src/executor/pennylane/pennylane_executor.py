import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union
from collections import Counter

from qiskit.circuit import ParameterExpression, Parameter, ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement


import pennylane as qml
import pennylane.numpy as pnp

from ..base import QuantumOperatorBase, QuantumCircuitBase, ExecutorBase

from .pennylane_circuit import PennyLaneCircuit
from .pennylane_observable import PennyLaneObservable
class PennylaneExecutor(ExecutorBase):
    """Base class for quantum circuit executors.

    Args:
        shots (int, optional): Number of shots for sampling. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        log_file (str, optional): Path to the log file. Defaults to None.
        caching (bool, optional): Whether to use caching. Defaults to None.
        cache_dir (str, optional): Directory for caching. Defaults to "cache".
    """
    def __init__(self,
                 shots: Union[int,None] = None,
                 seed: Union[int,None] = None,
                 log_file: Union[str,None] = None,
                 caching: Union[bool, None] = None,
                 cache_dir: str ="cache"):

        super().__init__(shots=shots, seed=seed, log_file=log_file, caching=caching, cache_dir=cache_dir)

        self._circuit_cache = {}
        self._operator_cache = {}

        if seed is not None:
            self._random = np.random.default_rng(seed)
        else:
            self._random = np.random.default_rng()

        self._device =  qml.device("default.qubit", wires=1)

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

    def expectation_value(self, circuit: QuantumCircuitBase, operator: QuantumOperatorBase, **parameter_values) -> float:
        """
        Calculate the expectation value of the operator with respect to the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.
            operator (QuantumOperatorBase): The quantum operator.

        Returns:
            float: The expectation value.
        """

        if circuit in self._circuit_cache:
            pennylane_circuit = self._circuit_cache[circuit]
        else:
            pennylane_circuit = PennyLaneCircuit(circuit)
            self._circuit_cache[circuit] = pennylane_circuit

        #todo operator cache
        pennylane_observable = PennyLaneObservable(operator)

        circuit_parameters = [parameter_values[param] for param in pennylane_circuit.parameter_names]
        obs_parameters = [parameter_values[param] for param in pennylane_observable.parameter_names]


        if circuit.num_qubits != len(self._device.wires):
            self._device = qml.device(self._device.name, wires=circuit.num_qubits)

        @qml.qnode(self._device)
        def circuit_func(*args):
            pennylane_circuit.build_pennylane_circuit()(*args)
            return pennylane_observable.build_pennylane_observable()(*args[len(pennylane_circuit.parameter_names):])

        # Execute the circuit
        result = circuit_func(*circuit_parameters, *obs_parameters)
        return result

    def expectation_value_derivatives(self, circuit: QuantumCircuitBase, operator: QuantumOperatorBase, parameter,  *values: Union[
            str,
            ParameterVector,
            ParameterVectorElement,
            tuple,
        ]) -> dict:
        """
        Calculate the derivatives of the expectation value with respect to the parameters of the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.
            operator (QuantumOperatorBase): The quantum operator.
            args: Additional arguments for the derivative calculation.

        Returns:
            List[float]: The derivatives of the expectation value.
        """
        raise NotImplementedError

    def sample(self, circuit: QuantumCircuitBase, **parameter_values) -> dict:
        """
        Sample the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.

        Returns:
            dict: The samples from the circuit.
        """
        if circuit in self._circuit_cache:
            pennylane_circuit = self._circuit_cache[circuit]
        else:
            pennylane_circuit = PennyLaneCircuit(circuit)
            self._circuit_cache[circuit] = pennylane_circuit

        circuit_parameters = [parameter_values[param] for param in pennylane_circuit.parameter_names]

        #if circuit.num_qubits != len(self._device.wires):

        self._device = qml.device("default.qubit", wires=circuit.num_qubits, shots=self._shots, seed = self._random)

        @qml.qnode(self._device)
        def circuit_func(*args):
            pennylane_circuit.build_pennylane_circuit()(*args)
            return qml.sample(wires=list(range(circuit.num_qubits)))

        samples = circuit_func(*circuit_parameters)

        # Convert samples to bitstrings
        bitstrings = ["".join(str(bit) for bit in sample[::-1]) for sample in samples]

        # Count occurrences of each bitstring
        return dict(Counter(bitstrings))




    def statevector(self, circuit: QuantumCircuitBase, **parameter_values) -> np.ndarray:
        """
        Get the statevector of the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.

        Returns:
            np.ndarray: The statevector of the circuit.
        """
        if circuit in self._circuit_cache:
            pennylane_circuit = self._circuit_cache[circuit]
        else:
            pennylane_circuit = PennyLaneCircuit(circuit)
            self._circuit_cache[circuit] = pennylane_circuit

        circuit_parameters = [parameter_values[param] for param in pennylane_circuit.parameter_names]

        if circuit.num_qubits != len(self._device.wires):
            self._device = qml.device(self._device.name, wires=circuit.num_qubits)

        @qml.qnode(self._device)
        def circuit_func(*args):
            pennylane_circuit.build_pennylane_circuit()(*args)
            return qml.state()

        # Execute the circuit
        state_wrong_sort = np.array(circuit_func(*circuit_parameters))

        def reverse_bits_array(n, num_bits):
            indices = np.arange(n)
            reversed_indices = np.zeros_like(indices)
            for _ in range(num_bits):
                # Shift left to make space for the next bit
                reversed_indices = (reversed_indices << 1) | (indices & 1)
                # Shift original indices right to move to the next bit
                indices >>= 1
            return reversed_indices

        n = len(state_wrong_sort)
        num_bits = circuit.num_qubits

        indices = reverse_bits_array(n, num_bits)
        state = state_wrong_sort[indices]

        return state

