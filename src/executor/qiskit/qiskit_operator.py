"""Qiskit operator wrapper for use with QiskitExecutor."""

from __future__ import annotations

from collections import OrderedDict
from typing import List

from executor.qiskit._param_binding import build_params_dict


class QiskitOperator:
    """
    Wrapper for Qiskit operators (SparsePauliOp) used by QiskitExecutor.
    Handles parameter management for parametrized operators.
    """

    def __init__(self, operator):
        """
        Initialize QiskitOperator wrapper.

        Args:
            operator: QuantumOperator object (from executor.quantum_operator)
        """
        # Extract the internal qiskit operator
        if hasattr(operator, "_qiskit_operator"):
            self._qiskit_operator = operator._qiskit_operator
        else:
            self._qiskit_operator = operator

        self._num_qubits = self._qiskit_operator.num_qubits

        # Group parameters by ParameterVector name
        self._parameter_dimensions = OrderedDict()
        self._free_parameters = set()

        for p in self._qiskit_operator.parameters:
            self._free_parameters.add(p)
            name = p.vector.name
            self._parameter_dimensions[name] = self._parameter_dimensions.get(name, 0) + 1

    @classmethod
    def from_quantum_operator(cls, operator):
        """Create a native Qiskit operator wrapper from a generic operator."""
        return cls(operator)

    @property
    def num_qubits(self) -> int:
        """Number of qubits the operator acts on."""
        return self._num_qubits

    @property
    def hash(self) -> int:
        """Hash of the operator for caching."""
        return hash(str(self._qiskit_operator))

    @property
    def parameter_names(self) -> List[str]:
        """List of parameter vector names."""
        return list(self._parameter_dimensions.keys())

    @property
    def parameter_dimensions(self) -> dict:
        """Dictionary mapping parameter names to their dimensions."""
        return dict(self._parameter_dimensions)

    @property
    def qiskit_operator(self):
        """Access to the underlying Qiskit SparsePauliOp."""
        return self._qiskit_operator

    @property
    def free_parameters(self) -> set:
        """Set of all free parameters in the operator."""
        return self._free_parameters

    def bind_parameters(self, parameter_values: dict):
        """
        Bind parameter values to the operator.

        Args:
            parameter_values: Dictionary mapping parameter names to values

        Returns:
            Bound Qiskit SparsePauliOp
        """
        params_dict = build_params_dict(self._free_parameters, parameter_values)

        # Bind parameters to operator
        if params_dict:
            return self._qiskit_operator.assign_parameters(params_dict)
        return self._qiskit_operator

    def copy(self):
        """Return a copy of the operator wrapper."""
        return QiskitOperator(self._qiskit_operator.copy())

    def __str__(self):
        return str(self._qiskit_operator)

    def __repr__(self):
        return f"QiskitOperator({self.num_qubits} qubits, {len(self._free_parameters)} parameters)"
