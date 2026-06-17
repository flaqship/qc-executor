"""Framework-independent abstraction layer for circuits, operators, parameters.

Define circuits and operators here without depending on any quantum framework,
then convert them to the backend of your choice (e.g. ``to_qiskit()``).
"""

from .abstract_parameter import Parameter, ParameterVector, free_parameters
from .abstract_quantum_circuit import AbstractQuantumCircuit
from .gates import AbstractGate, Barrier, CliffordGate, RotationGate

__all__ = [
    "Parameter",
    "ParameterVector",
    "free_parameters",
    "AbstractQuantumCircuit",
    "AbstractGate",
    "CliffordGate",
    "RotationGate",
    "Barrier",
]
