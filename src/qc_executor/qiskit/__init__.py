"""Qiskit backend for the executor framework.

Qiskit is an optional extra like every other backend, so importing this package
raises ``ImportError`` when it is absent.  ``qc_executor.qiskit`` then resolves
to ``None`` and :meth:`Executor.create` reports which extra to install.
"""

from qc_executor.factory import Executor

from .qiskit_circuit import QiskitCircuit
from .qiskit_executor import QiskitExecutor
from .qiskit_operator import QiskitOperator

Executor.register("qiskit")(QiskitExecutor)

__all__ = [
    "QiskitCircuit",
    "QiskitExecutor",
    "QiskitOperator",
]
