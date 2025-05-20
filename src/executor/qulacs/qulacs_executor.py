import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union
from collections import Counter

from qiskit.circuit import ParameterExpression, Parameter, ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement

from qulacs import QuantumState, GradCalculator, GeneralQuantumOperator

from ..base import QuantumOperatorBase, QuantumCircuitBase, ExecutorBase

from .qulacs_circuit import QulacsCircuit
from .qulacs_observable import QulacsObservable


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
        raise NotImplementedError

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
            qulacs_circuit = self._circuit_cache[circuit]
        else:
            qulacs_circuit = QulacsCircuit(circuit)
            self._circuit_cache[circuit] = qulacs_circuit

        # todo operator cache
        qulacs_observable = QulacsObservable(operator)

        obs_param_list = sum(
            [list(parameter_values[param]) for param in qulacs_observable.parameter_names], []
        )
        circ = qulacs_circuit.get_circuit_func()(
            *[parameter_values[param] for param in qulacs_circuit.parameter_names]
        )
        state = QuantumState(circuit.num_qubits)
        circ.update_quantum_state(state)
        operators = qulacs_observable.get_observable_func()(*obs_param_list)

        param_values = np.array([o.get_expectation_value(state) for o in operators])
        values = np.real_if_close(param_values)

        if not qulacs_observable.multiple_observables:
            return values[0]

        return values

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
            qulacs_circuit = self._circuit_cache[circuit]
        else:
            qulacs_circuit = QulacsCircuit(circuit)
            self._circuit_cache[circuit] = qulacs_circuit

        qulacs_circuit = qulacs_circuit.get_circuit_func()(
            *[parameter_values[param] for param in qulacs_circuit.parameter_names]
        )
        quantum_state = QuantumState(circuit.num_qubits)
        qulacs_circuit.update_quantum_state(quantum_state)

        # Get the state vector
        state_vector = quantum_state.get_vector()

        return state_vector
