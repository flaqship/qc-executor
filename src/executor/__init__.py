"""A library for quantum machine learning following the scikit-learn standard."""

import logging

logger = logging.getLogger(__name__)

# Import factory first to ensure it's available for backend registration
from .factory import Executor

from . import base
from .quantum_circuit import QuantumCircuit
from .quantum_operator import QuantumOperator
from .parameters import Parameters

try:
    from . import qiskit
except ImportError as e:
    logger.debug(f"Qiskit backend not available: {e}")
    qiskit = None

# Lazy load optional backends
try:
    from . import pennylane
except ImportError as e:
    logger.debug(f"PennyLane backend not available: {e}")
    pennylane = None

try:
    from . import qulacs
except ImportError as e:
    logger.debug(f"Qulacs backend not available: {e}")
    qulacs = None

__version__ = "0.1.0"

__all__ = [
    "Executor",
    "base",
    "QuantumCircuit",
    "QuantumOperator",
    "Parameters",
]

# Add optional backends to __all__ if available
if qiskit is not None:
    __all__.append("qiskit")
if pennylane is not None:
    __all__.append("pennylane")
if qulacs is not None:
    __all__.append("qulacs")
