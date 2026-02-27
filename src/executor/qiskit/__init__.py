from .qiskit_circuit import QiskitCircuit
from .qiskit_observable import QiskitObservable
from ..quantum_circuit import QuantumCircuit as _QuantumCircuit
from qiskit import QuantumCircuit as _QiskitQuantumCircuit


def transpile_circuit(circuit: _QuantumCircuit) -> _QiskitQuantumCircuit:
    """Transpile a generic QuantumCircuit to a Qiskit QuantumCircuit.

    Args:
        circuit: The generic QuantumCircuit to transpile.

    Returns:
        The corresponding Qiskit QuantumCircuit.
    """
    return circuit._qiskit_circuit


__all__ = [
    "QiskitCircuit",
    "QiskitObservable",
    "transpile_circuit",
]
