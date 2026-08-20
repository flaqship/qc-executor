"""A framework-independent layer for building and executing quantum circuits.

Importing this package pulls in no quantum framework.  The circuit, operator and
parameter types are built on a columnar instruction store and SymPy, and every
backend -- Qiskit included -- is an optional extra imported on first use.

``qc_executor.qiskit`` and friends resolve lazily through the module ``__getattr__``
below, so ``import qc_executor`` stays cheap and works in an environment with no
backend installed at all.  A backend whose dependency is missing resolves to
``None`` rather than raising.
"""

import importlib
import logging
from typing import Any

from . import base
from .factory import Executor
from .parameters import Parameters
from .quantum_circuit import QuantumCircuit
from .quantum_operator import QuantumOperator

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

#: Backends resolved on first attribute access.  Importing one eagerly here
#: would make its framework a hard dependency of the core package.
_OPTIONAL_BACKENDS = ("qiskit", "pennylane", "qulacs", "pauli_propagation")

__all__ = [
    "Executor",
    "base",
    "QuantumCircuit",
    "QuantumOperator",
    "Parameters",
    *_OPTIONAL_BACKENDS,
]


def __getattr__(name: str) -> Any:
    """Import an optional backend on first access.

    Args:
        name: Attribute being looked up.

    Returns:
        The backend module, or ``None`` if its dependency is not installed.

    Raises:
        AttributeError: For any name that is not an optional backend.
    """
    if name not in _OPTIONAL_BACKENDS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(f".{name}", __name__)
    except ImportError as error:
        logger.debug("%s backend not available: %s", name, error)
        module = None
    # Cache the result so the next lookup skips __getattr__ entirely.
    globals()[name] = module
    return module


def __dir__() -> list:
    """List the public names, including backends not yet imported."""
    return sorted(__all__)
