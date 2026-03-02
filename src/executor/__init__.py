"""A library for quantum machine learning following the scikit-learn standard."""

# Import factory first to ensure it's available for backend registration
from .factory import Executor

from . import base, pennylane, qiskit, qulacs
from .quantum_circuit import QuantumCircuit
from .quantum_operator import QuantumOperator
from .parameters import Parameters

__version__ = "0.1.0"

__all__ = [
    "Executor",
    "base",
    "pennylane",
    "qiskit",
    "qulacs",
    "QuantumCircuit",
    "QuantumOperator",
    "Parameters",
]
