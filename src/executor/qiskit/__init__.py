# Register QiskitExecutor with the factory
from executor.factory import Executor

from .qiskit_circuit import QiskitCircuit
from .qiskit_operator import QiskitOperator

try:
    from .qiskit_executor import QiskitExecutor
except ImportError as e:
    import warnings

    warnings.warn(
        f"Qiskit executor backend not available: {e}. "
        "Install with: pip install executor[qiskit-full]",
        UserWarning,
    )

    __all__ = [
        "QiskitCircuit",
        "QiskitOperator",
    ]
else:
    Executor.register("qiskit")(QiskitExecutor)

    __all__ = [
        "QiskitCircuit",
        "QiskitExecutor",
        "QiskitOperator",
    ]
