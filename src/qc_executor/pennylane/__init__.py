"""PennyLane backend for Executor."""

# Register PennyLaneExecutor with the factory
from qc_executor.factory import Executor

from .pennylane_circuit import PennyLaneCircuit
from .pennylane_executor import PennyLaneExecutor
from .pennylane_operator import PennyLaneOperator

Executor.register("pennylane")(PennyLaneExecutor)

__all__ = [
    "PennyLaneCircuit",
    "PennyLaneExecutor",
    "PennyLaneOperator",
]
