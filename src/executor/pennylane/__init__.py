"""PennyLane backend for Executor."""

try:
    # Register PennyLaneExecutor with the factory
    from executor.factory import Executor

    from .pennylane_circuit import PennyLaneCircuit
    from .pennylane_executor import PennyLaneExecutor
    from .pennylane_observable import PennyLaneObservable

    Executor.register("pennylane")(PennyLaneExecutor)

    __all__ = [
        "PennyLaneCircuit",
        "PennyLaneExecutor",
        "PennyLaneObservable",
    ]

except ImportError as e:
    import warnings

    warnings.warn(
        f"PennyLane backend not available: {e}. " "Install with: pip install executor[pennylane]",
        UserWarning,
    )
    raise
