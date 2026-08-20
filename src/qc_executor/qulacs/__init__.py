"""Qulacs backend for Executor."""

# Register QulacsExecutor with the factory
from qc_executor.factory import Executor

from .qulacs_circuit import QulacsCircuit
from .qulacs_executor import QulacsExecutor
from .qulacs_operator import QulacsOperator

Executor.register("qulacs")(QulacsExecutor)

__all__ = [
    "QulacsCircuit",
    "QulacsExecutor",
    "QulacsOperator",
]
