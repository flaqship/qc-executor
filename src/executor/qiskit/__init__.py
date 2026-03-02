from .qiskit_circuit import QiskitCircuit
from .qiskit_executor import QiskitExecutor
from .qiskit_observable import QiskitObservable

# Register QiskitExecutor with the factory
from executor.factory import Executor

@Executor.register("qiskit")
class _RegisteredQiskitExecutor(QiskitExecutor):
    """QiskitExecutor registered with the factory."""
    pass

# Replace QiskitExecutor reference to ensure decorator is applied
QiskitExecutor = _RegisteredQiskitExecutor

__all__ = [
    "QiskitCircuit",
    "QiskitExecutor",
    "QiskitObservable",
]
