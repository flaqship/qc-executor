"""Qulacs backend for Executor."""

try:
    from .qulacs_circuit import QulacsCircuit
    from .qulacs_executor import QulacsExecutor
    from .qulacs_observable import QulacsObservable
    
    # Register QulacsExecutor with the factory
    from executor.factory import Executor
    Executor.register("qulacs")(QulacsExecutor)
    
    __all__ = [
        "QulacsCircuit",
        "QulacsExecutor",
        "QulacsObservable",
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(
        f"Qulacs backend not available: {e}. "
        "Install with: pip install executor[qulacs]",
        ImportWarning
    )
    raise
