from .qulacs_circuit import QulacsCircuit
from .qulacs_observable import QulacsObservable
from ..quantum_circuit import QuantumCircuit as _QuantumCircuit


def transpile_circuit(circuit: _QuantumCircuit) -> QulacsCircuit:
    """Transpile a generic QuantumCircuit to a QulacsCircuit.

    Args:
        circuit: The generic QuantumCircuit to transpile.

    Returns:
        The corresponding QulacsCircuit.
    """
    return QulacsCircuit(circuit)


__all__ = [
    "QulacsCircuit",
    "QulacsObservable",
    "transpile_circuit",
]
