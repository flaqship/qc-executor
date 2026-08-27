"""Prototype Qrisp-backed executor."""

from __future__ import annotations

from typing import Any, List

import numpy as np

from ..base import ExecutorBase, QuantumCircuitBase, QuantumOperatorBase
from ..base.parameters_base import build_binding, evaluate, flatten_indexed
from .qrisp_circuit import QrispCircuit
from .qrisp_operator import QrispOperator


class QrispExecutor(ExecutorBase):
    """Execute Executor circuits using Qrisp's local simulator."""

    _native_circuit_class = QrispCircuit
    _native_operator_class = QrispOperator

    def __init__(self, backend: Any = None, **kwargs):
        if backend not in (None, "default"):
            raise ValueError("The Qrisp prototype currently supports only its local simulator")
        super().__init__(backend=backend, **kwargs)

    @property
    def shots(self) -> int | None:
        return self._shots

    @shots.setter
    def shots(self, value: int | None) -> None:
        self._shots = value

    @property
    def remote(self) -> bool:
        return False

    @classmethod
    def get_accepted_backend_types(cls) -> List[type]:
        return []

    @classmethod
    def get_accepted_backend_aliases(cls) -> List[str]:
        return ["qrisp", "default"]

    def _transpile_circuit(self, circuit: QuantumCircuitBase) -> QrispCircuit:
        return QrispCircuit.from_quantum_circuit(circuit)

    def _transpile_operator(self, operator: QuantumOperatorBase, **options) -> QrispOperator:
        return QrispOperator.from_quantum_operator(operator, **options)

    @staticmethod
    def _statevector_for(circuit: QrispCircuit, parameters) -> np.ndarray:
        compiled = circuit.build_qrisp_circuit(parameters)
        little_endian = np.asarray(compiled.statevector_array(), dtype=complex)
        width = circuit.num_qubits
        permutation = [int(f"{index:0{width}b}"[::-1], 2) for index in range(len(little_endian))]
        return little_endian[permutation]

    @staticmethod
    def _pauli_matrix(label: str) -> np.ndarray:
        matrices = {
            "I": np.eye(2, dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.diag([1, -1]).astype(complex),
        }
        result = matrices[label[0]]
        for char in label[1:]:
            result = np.kron(result, matrices[char])
        return result

    def _expectation_one(self, circuit, observable, parameters):
        binding = build_binding(circuit.parameters + observable.parameters, parameters)
        state = self._statevector_for(circuit, parameters)
        value = 0j
        for label, coefficient in zip(observable.paulis, observable.coeffs):
            coefficient_value = (
                evaluate(coefficient, binding)
                if hasattr(coefficient, "free_symbols")
                else complex(coefficient)
            )
            value += coefficient_value * np.vdot(
                state, self._pauli_matrix(label) @ state
            )
        return float(np.real_if_close(value).real)

    def _expectation_value(self, circuit, observable, **parameters):
        circuits = circuit if isinstance(circuit, list) else [circuit]
        observables = observable if isinstance(observable, list) else [observable]
        results = [self._expectation_one(c, o, parameters) for c in circuits for o in observables]
        if not isinstance(circuit, list) and not isinstance(observable, list):
            return results[0]
        return np.asarray(results)

    def _expectation_value_derivatives(self, circuit, observable, *derivative, **parameters):
        if isinstance(circuit, list):
            raise NotImplementedError("Qrisp derivatives do not support multiple circuits")
        requested = derivative or tuple(sorted({p.vector_name for p in circuit.parameters}))
        epsilon = 1e-3
        result = {}
        flat = flatten_indexed(parameters)
        for name in requested:
            elements = [key for key in flat if key == name or key.startswith(f"{name}[")]
            gradients = []
            for key in elements:
                vector = list(parameters[name]) if isinstance(parameters.get(name), (list, tuple)) else [parameters[name]]
                index = int(key.split("[")[1][:-1]) if "[" in key else 0
                plus_vector = vector.copy()
                minus_vector = vector.copy()
                plus_vector[index] += epsilon
                minus_vector[index] -= epsilon
                plus = dict(parameters)
                minus = dict(parameters)
                plus[name] = plus_vector
                minus[name] = minus_vector
                gradients.append(
                    (self._expectation_value(circuit, observable, **plus)
                     - self._expectation_value(circuit, observable, **minus))
                    / (2 * epsilon)
                )
            result[name] = np.asarray(gradients)
        return next(iter(result.values())) if len(result) == 1 else result

    def _statevector(self, circuit, **parameters):
        if isinstance(circuit, list):
            return np.asarray([self._statevector_for(item, parameters) for item in circuit])
        return self._statevector_for(circuit, parameters)

    def _sample(self, circuit, **parameters):
        if self._shots is None:
            raise ValueError("Qrisp sampling requires shots to be configured")
        state = self._statevector_for(circuit, parameters)
        probabilities = np.abs(state) ** 2
        probabilities = probabilities / probabilities.sum()
        indices = np.random.default_rng(self._seed).choice(len(state), self._shots, p=probabilities)
        counts = {}
        for index in indices:
            bitstring = f"{index:0{circuit.num_qubits}b}"
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts