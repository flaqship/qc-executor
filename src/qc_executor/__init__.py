"""A library for quantum machine learning following the scikit-learn standard."""

import logging

from . import base
from .factory import Executor
from .parameters import Parameters, Parameter
from .quantum_circuit import QuantumCircuit
from .quantum_operator import QuantumOperator

logger = logging.getLogger(__name__)

try:
    from . import qiskit
except ImportError as e:
    logger.debug("Qiskit backend not available: %s", e)
    qiskit = None

# Lazy load optional backends
try:
    from . import pennylane
except ImportError as e:
    logger.debug("PennyLane backend not available: %s", e)
    pennylane = None

try:
    from . import qulacs
except ImportError as e:
    logger.debug("Qulacs backend not available: %s", e)
    qulacs = None

try:
    from . import pauli_propagation
except ImportError as e:
    logger.debug("Pauli propagation backend not available: %s", e)
    pauli_propagation = None

__version__ = "0.1.0"

__all__ = ["Executor", "base", "QuantumCircuit", "QuantumOperator", "Parameters", "Parameter"]

# Add optional backends to __all__ if available
if qiskit is not None:
    __all__.append("qiskit")
if pennylane is not None:
    __all__.append("pennylane")
if qulacs is not None:
    __all__.append("qulacs")
if pauli_propagation is not None:
    __all__.append("pauli_propagation")
