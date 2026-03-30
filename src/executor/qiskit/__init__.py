# Register QiskitExecutor with the factory
from executor.factory import Executor

from .qiskit_circuit import QiskitCircuit
from .qiskit_observable import QiskitObservable

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
        "QiskitObservable",
    ]
else:
    Executor.register("qiskit")(QiskitExecutor)

    __all__ = [
        "QiskitCircuit",
        "QiskitExecutor",
        "QiskitObservable",
    ]
