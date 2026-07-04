from __future__ import annotations

from collections import OrderedDict
from typing import List

import numpy as np
import sympy as sp
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.circuit import ParameterVector as QiskitParameterVector

from ..abstraction.abstract_parameter import Parameter, free_parameters
from ..abstraction.gates import Barrier, CliffordGate, RotationGate


def _build_qiskit_parameter_map(circuit) -> dict:
    """Map each native :class:`Parameter` to a Qiskit ParameterVector element.

    Collects all parameters used across gates, groups them by vector name, and
    builds one Qiskit ``ParameterVector`` per group sized to the highest index
    seen. Returns a dict ``{native Parameter: qiskit element}``.
    """
    max_index: dict = {}
    for gate in circuit._gates:
        if isinstance(gate, RotationGate):
            for param in free_parameters(gate.angle):
                name = param.vector_name
                max_index[name] = max(max_index.get(name, -1), param.index)

    symbol_to_qiskit: dict = {}
    for name, top in sorted(max_index.items()):
        qiskit_vec = QiskitParameterVector(name, top + 1)
        for i in range(top + 1):
            symbol_to_qiskit[Parameter(name, i)] = qiskit_vec[i]
    return symbol_to_qiskit


def _angle_to_qiskit(angle, symbol_to_qiskit: dict):
    """Convert one angle (numeric or symbolic) to a Qiskit-usable value."""
    if not isinstance(angle, sp.Basic):
        return angle  # plain float / int

    symbols = free_parameters(angle)
    if not symbols:
        return float(angle)  # constant expression

    # Replay the symbolic expression onto Qiskit parameter objects. Qiskit
    # ParameterExpression supports +, -, *, /, ** so arithmetic angles work.
    func = sp.lambdify(symbols, angle, modules="math")
    qiskit_args = [symbol_to_qiskit[s] for s in symbols]
    return func(*qiskit_args)


def _abstract_to_qiskit(circuit) -> QiskitQuantumCircuit:
    """Build a Qiskit ``QuantumCircuit`` from an ``AbstractQuantumCircuit``.

    Gate names in the abstract circuit match the Qiskit method names, so each
    typed gate maps directly onto the corresponding Qiskit instruction.
    """
    qc = QiskitQuantumCircuit(circuit.num_qubits)
    symbol_to_qiskit = _build_qiskit_parameter_map(circuit)

    for gate in circuit._gates:
        if isinstance(gate, Barrier):
            qc.barrier(list(gate.qubits))
        elif isinstance(gate, CliffordGate):
            getattr(qc, gate.name)(*gate.qubits)
        elif isinstance(gate, RotationGate):
            angle = _angle_to_qiskit(gate.angle, symbol_to_qiskit)
            getattr(qc, gate.name)(angle, *gate.qubits)
        else:  # pragma: no cover - defensive
            raise NotImplementedError(f"Unknown gate type {type(gate).__name__}")
    return qc


def to_qiskit_circuit(circuit):
    """Return the underlying Qiskit circuit for any supported circuit type.

    * executor wrappers expose ``_qiskit_circuit`` and are returned directly;
    * an ``AbstractQuantumCircuit`` exposes a typed ``_gates`` list and is
      converted here (cached on the instance, keyed by content hash so repeated
      access returns identical parameter objects);
    * anything else is assumed to already be a raw Qiskit circuit.
    """
    qiskit_circuit = getattr(circuit, "_qiskit_circuit", None)
    if qiskit_circuit is not None:
        return qiskit_circuit

    if hasattr(circuit, "_gates"):
        key = hash(circuit)
        cache = getattr(circuit, "_qiskit_conversion_cache", None)
        if cache is None or cache[0] != key:
            converted = _abstract_to_qiskit(circuit)
            try:
                circuit._qiskit_conversion_cache = (key, converted)
            except (AttributeError, TypeError):  # pragma: no cover - defensive
                pass
            return converted
        return cache[1]

    return circuit


class QiskitCircuit:
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
        # Extract / build the internal qiskit circuit (handles abstract circuits).
        self._qiskit_circuit = to_qiskit_circuit(circuit)

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

            # Match parameters from circuit with provided values.
            # Guard against standalone Parameter objects (no .vector/.index).
            def _param_name(p) -> str:
                return p.vector.name if hasattr(p, "vector") else p.name

            def _param_index(p) -> int:
                return p.index if hasattr(p, "index") else 0

            matching_params = [p for p in self._free_parameters if _param_name(p) == param_name]
            # Sort by index to ensure correct ordering
            matching_params = sorted(matching_params, key=_param_index)

            for i, param in enumerate(matching_params):
                if i < len(values):
                    params_dict[param] = values[i]

        # Bind parameters to circuit
        if params_dict:
            return self._qiskit_circuit.assign_parameters(params_dict)
        else:
            return self._qiskit_circuit

    @classmethod
    def _from_qiskit(cls, qiskit_circuit) -> "QiskitCircuit":
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

    def __str__(self):
        return str(self._qiskit_circuit)

    def __repr__(self):
        return f"QiskitCircuit({self.num_qubits} qubits, {len(self._free_parameters)} parameters)"
