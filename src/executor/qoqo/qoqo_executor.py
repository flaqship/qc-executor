from __future__ import annotations

import numpy as np
from typing import Union, List
from collections import Counter

from qiskit.circuit import ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement

from ..base import QuantumOperatorBase, QuantumCircuitBase, ExecutorBase

from qoqo import Circuit, operations as ops
from qoqo_quest import Backend

from .qoqo_circuit import QoqoCircuit
from .qoqo_operator import QoqoOperator
from qollage import draw_circuit


class QoqoExecutor(ExecutorBase):
    """Base class for quantum circuit executors.

    Args:
        shots (int, optional): Number of shots for sampling. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        log_file (str, optional): Path to the log file. Defaults to None.
        caching (bool, optional): Whether to use caching. Defaults to None.
        cache_dir (str, optional): Directory for caching. Defaults to "cache".
    """

    _native_circuit_class = QoqoCircuit
    _native_operator_class = QoqoOperator

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

    def _expectation_value(
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
            if isinstance(circuit, QoqoCircuit):
                qoqo_circuit = circuit
            else:
                qoqo_circuit = QoqoCircuit(circuit)
            self._circuit_cache[circuit] = qoqo_circuit

        qoqo_observable = QoqoOperator(operator)
        flat_circuit_parameters = {}
        for key, vals in parameter_values.items():
            is_list = isinstance(vals, list)
            if key in qoqo_circuit.parameter_names:
                if is_list:
                    for i, v in enumerate(vals):
                        flat_circuit_parameters[f"{key}[{i}]"] = v
                else:
                    flat_circuit_parameters[key] = vals
        obs_parameters = [parameter_values[key] for key in qoqo_observable.parameter_names]
        qoqo_circuit.assign_parameters(flat_circuit_parameters)
        qoqo_observable.assign_parameters(np.array(obs_parameters).flatten().tolist())

        constant_circuit = qoqo_circuit.get_qoqo_circuit()
        measurement, id_shift = qoqo_observable.get_qoqo_observable_measurement(
            constant_circuit, self._shots
        )
        backend = Backend(qoqo_circuit.num_qubits)
        expectations = backend.run_measurement(measurement)

        return expectations["expectation_value"] + id_shift

    def _expectation_value_derivatives(
        self,
        circuit: QuantumCircuitBase | List[QuantumCircuitBase],
        observable: QuantumOperatorBase | List[QuantumOperatorBase],
        *derivative,
        **parameters,
    ) -> float | np.ndarray | dict:
        """
        Calculate the derivatives of the expectation value with respect to the
        parameters of the circuit.

        Args:
            circuit (QuantumCircuitBase | List[QuantumCircuitBase]): The quantum circuit
                or a list of circuits.
            observable (QuantumOperatorBase | List[QuantumOperatorBase]): The quantum
                observable or a list of observables.
            derivative: The parameter(s) with respect to which the derivative is calculated.
            parameters: Additional values for the free parameters of the circuit(s) and
                the observable(s) given as keyword arguments.
                Both vector-style keys (e.g., ``x=[0.1, 0.2]``) and indexed keys
                (e.g., ``x[0]=0.1, x[1]=0.2``) are accepted and normalized.

        Returns:
            float | np.array | dict: The derivative of the expectation value:
                - single float/array if one derivative parameter is requested
                - dictionary mapping parameter names to gradient arrays if multiple
                  parameters are requested
        """
        raise NotImplementedError

    def _sample(self, circuit: QuantumCircuitBase, **parameter_values) -> dict:
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

    def _statevector(self, circuit: QuantumCircuitBase, **parameter_values) -> np.ndarray:
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
        running_circuit += ops.PragmaGetStateVector("state_vector", None)

        backend = Backend(qoqo_circuit.num_qubits)
        results = backend.run_circuit(running_circuit)

        return results[2].get("state_vector")[0]

    def _transpile_circuit(self, circuit: QuantumCircuitBase) -> QuantumCircuitBase:
        """Abstract implementation of circuit transpilation."""
        if isinstance(circuit, self._native_circuit_class):
            return circuit
        return self._native_circuit_class.from_quantum_circuit(circuit)

    def _transpile_operator(self, operator: QuantumOperatorBase) -> QuantumOperatorBase:
        """Abstract implementation of operator transpilation.

        Subclasses override this to convert generic QuantumOperator to
        backend-native types. For backends supporting symmetry (e.g.,
        Pauli Propagation), the symmetry_strategy parameter allows
        assigning a symmetry strategy to the operator.

        Args:
            operator (QuantumOperatorBase): The operator to transpile.

        Returns:
            QuantumOperatorBase: The transpiled operator in backend-native format.
        """
        if isinstance(operator, self._native_operator_class):
            return operator
        return self._native_operator_class.from_quantum_operator(operator)

    @classmethod
    def get_accepted_backend_types(cls) -> List[type]:
        """Return a list of backend object types accepted by this executor.

        This is used for auto-detection when a non-string backend is passed to
        :meth:`Executor.create`.  If the backend object is an instance of any
        of the returned types, this executor will be selected automatically.

        Returns:
            List[type]: List of accepted backend types
                (e.g., Qiskit ``Backend`` / ``BackendV2`` classes)
        """
        raise NotImplementedError
