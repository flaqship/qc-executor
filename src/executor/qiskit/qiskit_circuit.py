
from __future__ import annotations
from collections import OrderedDict
from typing import List

import numpy as np


class QiskitCircuit:
    """
    Wrapper for Qiskit circuits used by QiskitExecutor.
    Provides parameter management and circuit caching functionality.
    """

    def __init__(self, circuit):
        """
        Initialize QiskitCircuit wrapper.

        Args:
            circuit: QuantumCircuit object (from executor.quantum_circuit)
        """
        # Extract the internal qiskit circuit
        if hasattr(circuit, "_qiskit_circuit"):
            self._qiskit_circuit = circuit._qiskit_circuit
        else:
            self._qiskit_circuit = circuit

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
    def hash(self) -> int:
        """Hash of the circuit for caching."""
        return hash(str(self._qiskit_circuit))

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
        # Convert parameter_values dict (with vector names) to Qiskit format
        params_dict = {}

        for param_name, values in parameter_values.items():
            # Ensure values is a list
            if not isinstance(values, (list, np.ndarray)):
                values = [values]

            # Match parameters from circuit with provided values
            matching_params = [p for p in self._free_parameters if p.vector.name == param_name]

            # Sort by index to ensure correct ordering
            matching_params = sorted(matching_params, key=lambda x: x.index)

            for i, param in enumerate(matching_params):
                if i < len(values):
                    params_dict[param] = values[i]

        # Bind parameters to circuit
        if params_dict:
            return self._qiskit_circuit.assign_parameters(params_dict)
        else:
            return self._qiskit_circuit

    def copy(self):
        """Return a copy of the circuit wrapper."""
        return QiskitCircuit(self._qiskit_circuit.copy())

    def __str__(self):
        return str(self._qiskit_circuit)

    def __repr__(self):
        return f"QiskitCircuit({self.num_qubits} qubits, {len(self._free_parameters)} parameters)"
