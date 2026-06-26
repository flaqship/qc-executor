"""PennyLane backend for Executor."""

try:
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

except ImportError as e:
    import warnings

    warnings.warn(
        f"PennyLane backend not available: {e}. "
        "Install with: pip install qc-executor[pennylane]",
        UserWarning,
    )
    raise
