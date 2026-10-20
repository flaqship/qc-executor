"""qoqo backend for Executor."""

try:
    # Register QoqoExecutor with the factory
    from executor.factory import Executor

    from .qoqo_circuit import QoqoCircuit
    from .qoqo_executor import QoqoExecutor
    from .qoqo_operator import QoqoOperator

    Executor.register("qoqo")(QoqoExecutor)

    __all__ = [
        "QoqoCircuit",
        "QoqoExecutor",
        "QoqoOperator",
    ]

except ImportError as e:
    import warnings

    warnings.warn(
        f"qoqo backend not available: {e}. " "Install with: pip install executor[qoqo]",
        UserWarning,
    )
    raise
