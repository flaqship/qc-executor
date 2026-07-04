"""Qulacs backend for Executor."""

try:
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

except ImportError as e:
    import warnings

    warnings.warn(
        f"Qulacs backend not available: {e}. " "Install with: pip install executor[qulacs]",
        UserWarning,
    )
    raise
