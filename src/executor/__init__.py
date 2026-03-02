"""A library for quantum machine learning following the scikit-learn standard."""

import importlib

from . import base
from .quantum_circuit import QuantumCircuit
from .quantum_operator import QuantumOperator
from .parameters import Parameters
from ._factory import create_executor

__version__ = "0.1.0"

__all__ = [
    "base",
    "pennylane",
    "qiskit",
    "qulacs",
    "create_executor",
    "QuantumCircuit",
    "QuantumOperator",
    "Parameters",
]

_LAZY_SUBMODULES = {"pennylane", "qiskit", "qulacs"}


def __getattr__(name: str):
    """Lazily import backend submodules on first access (PEP 562)."""
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
