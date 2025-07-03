import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union
from collections import Counter

from qiskit.circuit import ParameterExpression, Parameter, ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement

from ..base import QuantumOperatorBase, QuantumCircuitBase, ExecutorBase

from qoqo import Circuit, operations as ops
from qoqo_quest import Backend

from .qoqo_circuit import QoqoCircuit
from .qoqo_observable import QoqoObservable


class QulacsExecutor(ExecutorBase):
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

    @property
    def shots(self) -> Union[int, None]:
        """Return the number of shots."""
        return self._shots

    @shots.setter
    def shots(self, value: Union[int, None]) -> None:
        """Set the number of shots."""
        self._shots = value

    @property
    def remote(self) -> bool:
        """Return True if the execution access a remote backend."""
        return False

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

        if circuit in self._circuit_cache:
            qoqo_circuit = self._circuit_cache[circuit]
        else:
            qoqo_circuit = QoqoCircuit(circuit)
            self._circuit_cache[circuit] = qoqo_circuit

        constant_circuit = qoqo_circuit.get_qoqo_circuit()
        qoqo_observable = QoqoObservable(operator)

        measurement, id_shift = qoqo_observable.get_qoqo_observable_measurement(
            constant_circuit, qoqo_observable, self._shots
        )
        backend = Backend(qoqo_circuit.num_qubits)

        expectations = backend.run_measurement(measurement)

        return expectations["expectation_value"] + id_shift

    def expectation_value_derivatives(
        self,
        circuit: QuantumCircuitBase,
        operator: QuantumOperatorBase,
        parameter,
        *values: Union[
            str,
            ParameterVector,
            ParameterVectorElement,
            tuple,
        ],
    ) -> dict:
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

        statevector = self.statevector(circuit, **parameter_values)

        # Get the probabilities
        probabilities = np.square(np.abs(statevector))

        # Generate bitstrings corresponding to each basis state
        num_qubits = circuit.num_qubits
        bitstrings = [format(i, f"0{num_qubits}b") for i in range(len(statevector))]

        # Sample from the distribution
        samples = self._random.choice(bitstrings, size=self._shots, p=probabilities)

        # Count occurrences
        counts = dict(Counter(samples))
        return counts

    def statevector(self, circuit: QuantumCircuitBase, **parameter_values) -> np.ndarray:
        """
        Get the statevector of the circuit.

        Args:
            circuit (QuantumCircuitBase): The quantum circuit.

        Returns:
            np.ndarray: The statevector of the circuit.
        """
        if circuit in self._circuit_cache:
            qoqo_circuit = self._circuit_cache[circuit]
        else:
            qoqo_circuit = QoqoCircuit(circuit)
            self._circuit_cache[circuit] = qoqo_circuit

        running_circuit = Circuit()
        running_circuit += ops.DefinitionComplex("state_vector", qoqo_circuit.num_qubits, True)
        running_circuit += qoqo_circuit.get_qoqo_circuit()
        running_circuit += ops.PragmaGetStateVector(qoqo_circuit.num_qubits, None)

        backend = Backend(qoqo_circuit.num_qubits)

        results = backend.run_circuit(running_circuit)

        return results[2].get("state_vector")[0]
