"""Qrisp backend for QC Executor."""

from qc_executor.factory import Executor

from .qrisp_circuit import QrispCircuit
from .qrisp_executor import QrispExecutor
from .qrisp_operator import QrispOperator

Executor.register("qrisp")(QrispExecutor)

__all__ = ["QrispCircuit", "QrispExecutor", "QrispOperator"]