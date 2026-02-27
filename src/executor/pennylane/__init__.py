from .pennylane_circuit import PennyLaneCircuit
from ..quantum_circuit import QuantumCircuit as _QuantumCircuit


def transpile_circuit(circuit: _QuantumCircuit) -> PennyLaneCircuit:
    """Transpile a generic QuantumCircuit to a PennyLaneCircuit.

    Args:
        circuit: The generic QuantumCircuit to transpile.

    Returns:
        The corresponding PennyLaneCircuit.
    """
    return PennyLaneCircuit(circuit)


__all__ = [
    "PennyLaneCircuit",
    "transpile_circuit",
]