"""Qiskit circuit wrapper for parameter management and circuit caching."""

from __future__ import annotations

from collections import OrderedDict
from typing import List

from qc_executor.circuit_idendity_mixin import CircuitIdentityMixin
from qc_executor.qiskit._param_binding import build_params_dict
from qc_executor.utils.qiskit_hash_functions import _circuit_key


class QiskitCircuit(CircuitIdentityMixin):
    """
    Wrapper for Qiskit circuits used by QiskitExecutor.
    Provides parameter management and circuit caching functionality.
    """

    def __init__(self, circuit):
        """
        Initialize QiskitCircuit wrapper.

        Args:
            circuit: QuantumCircuit object (from qc_executor.quantum_circuit)
        """
        # Extract the internal qiskit circuit
        self._qiskit_circuit = getattr(circuit, "qiskit_circuit", circuit)

        self._num_qubits = self._qiskit_circuit.num_qubits

        # Group parameters by ParameterVector name
        self._parameter_dimensions = OrderedDict()
        self._free_parameters = set()

        for p in self._qiskit_circuit.parameters:
            self._free_parameters.add(p)
            name = p.vector.name
            self._parameter_dimensions[name] = self._parameter_dimensions.get(name, 0) + 1

    @classmethod
    def from_quantum_circuit(cls, circuit):
        """Create a native Qiskit circuit wrapper from a generic circuit."""
        return cls(circuit)

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the circuit."""
        return self._num_qubits

    @property
    def parameter_names(self) -> List[str]:
        """List of parameter vector names."""
        return list(self._parameter_dimensions.keys())

    @property
    def parameter_dimensions(self) -> dict:
        """Dictionary mapping parameter names to their dimensions."""
        return dict(self._parameter_dimensions)

    @property
    def qiskit_circuit(self):
        """Access to the underlying Qiskit circuit."""
        return self._qiskit_circuit

    @property
    def free_parameters(self) -> set:
        """Set of all free parameters in the circuit."""
        return self._free_parameters

    def bind_parameters(self, parameter_values: dict):
        """
        Bind parameter values to the circuit.

        Args:
            parameter_values: Dictionary mapping parameter names to values

        Returns:
            Bound Qiskit circuit
        """
        params_dict = build_params_dict(self._free_parameters, parameter_values)

        # Bind parameters to circuit
        if params_dict:
            return self._qiskit_circuit.assign_parameters(params_dict)
        return self._qiskit_circuit

    @classmethod
    def from_qiskit(cls, qiskit_circuit) -> "QiskitCircuit":
        """Create a :class:`QiskitCircuit` directly from a Qiskit ``QuantumCircuit``.

        This bypasses the normal ``__init__`` path which expects
        an executor ``QuantumCircuit`` wrapper and instead accepts
        an already-transpiled Qiskit circuit.
        """
        wrapper = object.__new__(cls)
        wrapper._qiskit_circuit = qiskit_circuit
        wrapper._num_qubits = qiskit_circuit.num_qubits

        wrapper._parameter_dimensions = OrderedDict()
        wrapper._free_parameters = set()
        for p in qiskit_circuit.parameters:
            wrapper._free_parameters.add(p)
            name = p.vector.name if hasattr(p, "vector") else p.name
            wrapper._parameter_dimensions[name] = wrapper._parameter_dimensions.get(name, 0) + 1
        return wrapper

    def copy(self):
        """Return a copy of the circuit wrapper."""
        return QiskitCircuit(self._qiskit_circuit.copy())

    def _circuit_hash_key(self):
        return (_circuit_key(self._qiskit_circuit),)

    def __str__(self):
        return str(self._qiskit_circuit)

    def __repr__(self):
        return f"QiskitCircuit({self.num_qubits} qubits, {len(self._free_parameters)} parameters)"
