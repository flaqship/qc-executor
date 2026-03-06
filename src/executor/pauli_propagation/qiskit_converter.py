"""Convert Qiskit circuits to internal gate representation."""

import hashlib
import pickle
from typing import Dict, List, Optional, Union

import numpy as np
import sympy as sp

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter, ParameterExpression
    from qiskit.circuit.library import (
        CXGate,
        CZGate,
        HGate,
        RXGate,
        RXXGate,
        RYGate,
        RYYGate,
        RZGate,
        RZZGate,
        SGate,
        SwapGate,
        TGate,
        XGate,
        YGate,
        ZGate,
    )

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from .gates import CliffordGate, Gate, LayerBarrier, PauliRotation


class CircuitConversionCache:
    """Cache for converted circuits to avoid re-conversion."""

    def __init__(self):
        """Initialize empty cache."""
        self._cache = {}

    def get_hash(self, circuit) -> str:
        """Compute hash of circuit structure (ignoring parameter values).

        Args:
            circuit: Qiskit QuantumCircuit

        Returns:
            Hash string
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit conversion")

        # Create a string representation of circuit structure
        circuit_str = f"nqubits={circuit.num_qubits}\n"
        for instruction in circuit.data:
            gate = instruction.operation
            qubits = [circuit.find_bit(q).index for q in instruction.qubits]
            circuit_str += f"{gate.name}({qubits})\n"

        # Hash the string
        return hashlib.md5(circuit_str.encode()).hexdigest()

    def get(self, circuit_hash: str) -> Optional[List[Union[Gate, LayerBarrier]]]:
        """Get cached conversion.

        Args:
            circuit_hash: Circuit hash

        Returns:
            List of gates if cached, None otherwise
        """
        return self._cache.get(circuit_hash)

    def set(self, circuit_hash: str, gates: List[Union[Gate, LayerBarrier]]) -> None:
        """Cache conversion.

        Args:
            circuit_hash: Circuit hash
            gates: List of internal gates
        """
        self._cache[circuit_hash] = gates

    def clear(self) -> None:
        """Clear cache."""
        self._cache = {}


# Global cache instance
_circuit_cache = CircuitConversionCache()


def convert_circuit(
    circuit,
    use_cache: bool = True,
) -> List[Union[Gate, LayerBarrier]]:
    """Convert Qiskit QuantumCircuit to list of internal Gates.

    Args:
        circuit: Qiskit QuantumCircuit
        use_cache: Whether to use caching

    Returns:
        List of internal Gate or LayerBarrier objects

    Raises:
        ImportError: If Qiskit is not available
        ValueError: If circuit contains unsupported gates
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for circuit conversion")

    # Check cache
    if use_cache:
        circuit_hash = _circuit_cache.get_hash(circuit)
        cached_gates = _circuit_cache.get(circuit_hash)
        if cached_gates is not None:
            return cached_gates

    nqubits = circuit.num_qubits
    gates = []

    for instruction in circuit.data:
        gate_op = instruction.operation
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]

        # Convert to internal gate
        internal_gate = _convert_single_gate(gate_op, qubits, nqubits)
        if internal_gate is not None:
            gates.append(internal_gate)

    # Cache result
    if use_cache:
        _circuit_cache.set(circuit_hash, gates)

    return gates


def _convert_single_gate(gate_op, qubits: List[int], nqubits: int) -> Optional[Gate]:
    """Convert a single Qiskit gate to internal representation.

    Args:
        gate_op: Qiskit gate operation
        qubits: List of qubit indices
        nqubits: Total number of qubits

    Returns:
        Internal Gate object or None if gate should be skipped

    Raises:
        ValueError: If gate is not supported
    """
    gate_name = gate_op.name.upper()

    # Pauli rotations (parametric gates)
    if gate_name == "RX":
        param_expr, param_value = (
            _extract_parameter(gate_op.params[0]) if gate_op.params else (None, None)
        )
        return PauliRotation(
            ["X"], qubits[0], nqubits, param_expr=param_expr, param_value=param_value
        )

    elif gate_name == "RY":
        param_expr, param_value = (
            _extract_parameter(gate_op.params[0]) if gate_op.params else (None, None)
        )
        return PauliRotation(
            ["Y"], qubits[0], nqubits, param_expr=param_expr, param_value=param_value
        )

    elif gate_name == "RZ":
        param_expr, param_value = (
            _extract_parameter(gate_op.params[0]) if gate_op.params else (None, None)
        )
        return PauliRotation(
            ["Z"], qubits[0], nqubits, param_expr=param_expr, param_value=param_value
        )

    elif gate_name == "RXX":
        param_expr, param_value = (
            _extract_parameter(gate_op.params[0]) if gate_op.params else (None, None)
        )
        return PauliRotation(
            ["X", "X"], qubits, nqubits, param_expr=param_expr, param_value=param_value
        )

    elif gate_name == "RYY":
        param_expr, param_value = (
            _extract_parameter(gate_op.params[0]) if gate_op.params else (None, None)
        )
        return PauliRotation(
            ["Y", "Y"], qubits, nqubits, param_expr=param_expr, param_value=param_value
        )

    elif gate_name == "RZZ":
        param_expr, param_value = (
            _extract_parameter(gate_op.params[0]) if gate_op.params else (None, None)
        )
        return PauliRotation(
            ["Z", "Z"], qubits, nqubits, param_expr=param_expr, param_value=param_value
        )

    # Clifford gates (non-parametric)
    elif gate_name in ["H", "S", "T", "X", "Y", "Z"]:
        return CliffordGate(gate_name, qubits[0], nqubits)

    elif gate_name in ["CX", "CNOT"]:
        return CliffordGate("CNOT", qubits, nqubits)

    elif gate_name == "CZ":
        return CliffordGate("CZ", qubits, nqubits)

    elif gate_name == "SWAP":
        return CliffordGate("SWAP", qubits, nqubits)

    # Identity gates are skipped
    elif gate_name == "ID":
        return None

    # Barriers mark layer boundaries
    elif gate_name == "BARRIER":
        return LayerBarrier()

    else:
        raise ValueError(f"Unsupported gate: {gate_name}")


def _extract_parameter_name(param) -> Optional[str]:
    """Extract parameter name from Qiskit parameter.

    Args:
        param: Qiskit parameter (could be Parameter, float, or ParameterExpression)

    Returns:
        Parameter name string or None if it's a concrete value
    """
    if not QISKIT_AVAILABLE:
        return None

    if isinstance(param, Parameter):
        return param.name
    elif isinstance(param, ParameterExpression):
        # For expressions, get the first parameter name
        params = param.parameters
        if len(params) > 0:
            return list(params)[0].name
        return None
    else:
        # Concrete value (float)
        return None


def _extract_parameter(param) -> tuple[sp.Expr | None, float | None]:
    """Extract parameter as sympy expression or concrete value from Qiskit parameter.

    Args:
        param: Qiskit parameter (could be Parameter, float, or ParameterExpression)

    Returns:
        Tuple of (sympy_expr, concrete_value)
        - For Parameter/ParameterExpression: (sympy_expr, None)
        - For float: (None, value)
    """
    if not QISKIT_AVAILABLE:
        return None, None

    from executor.utils.qiskit_compat import _param_is_constant, _param_to_float, _param_to_sympy

    if isinstance(param, (Parameter, ParameterExpression)):
        if _param_is_constant(param):
            # Constant expression, return as float
            return None, _param_to_float(param)
        else:
            # Symbolic expression, convert to sympy
            return _param_to_sympy(param), None
    else:
        # Concrete value (float)
        return None, float(param)


def bind_parameters(
    gates: List[Union[Gate, LayerBarrier]],
    parameter_values: Dict[str, float],
) -> Dict[str, float]:
    """Create a complete parameter binding dict for gates.

    Extracts concrete parameter values from gates and merges them with
    provided parameter_values.

    Args:
        gates: List of internal gates (may include LayerBarrier markers)
        parameter_values: Dict mapping parameter names to values

    Returns:
        Complete parameter dict with all required values

    Raises:
        ValueError: If required parameters are missing
        TypeError: If an unexpected object type is in the gates list
    """
    # First, expand any list values in parameter_values to indexed format
    # This converts {"x": [0.1], "p": [0.5, 0.6]} to individual indexed entries
    expanded_params = {}
    for key, value in parameter_values.items():
        if isinstance(value, (list, tuple)):
            # For each value in the list, create an indexed key
            for idx, val in enumerate(value):
                indexed_key = f"{key}[{idx}]"
                if not isinstance(val, (int, float, np.number)):
                    raise TypeError(
                        f"Parameter '{indexed_key}' has invalid type {type(val)}. "
                        f"Expected float or numeric value."
                    )
                expanded_params[indexed_key] = float(val)
        elif isinstance(value, (int, float, np.number)):
            expanded_params[key] = float(value)
        else:
            raise TypeError(
                f"Parameter '{key}' has invalid value type {type(value)}. "
                f"Expected float or list of floats."
            )

    # Start with the expanded parameter dict
    result = dict(expanded_params)

    # Collect required parameters and extract concrete values
    required_params = set()
    for gate in gates:
        # Explicitly skip LayerBarrier markers
        if isinstance(gate, LayerBarrier):
            continue

        # All other objects must be Gate instances
        if not isinstance(gate, Gate):
            raise TypeError(
                f"Unexpected object in gates list: {type(gate)!r}; "
                "expected a Gate or LayerBarrier."
            )

        if gate.is_parametric():
            if hasattr(gate, "param_expr") and gate.param_expr is not None:
                for symbol in gate.param_expr.free_symbols:
                    if symbol.name not in result:
                        required_params.add(symbol.name)
            elif gate.param_name:
                if gate.param_name not in result:
                    if hasattr(gate, "param_value") and gate.param_value is not None:
                        result[gate.param_name] = gate.param_value
                    else:
                        required_params.add(gate.param_name)
            elif hasattr(gate, "param_value") and gate.param_value is not None:
                pass

    # Check for missing parameters
    missing = required_params - set(result.keys())
    if missing:
        raise ValueError(f"Missing parameter values for: {missing}")

    return result


def clear_cache() -> None:
    """Clear the circuit conversion cache."""
    _circuit_cache.clear()
