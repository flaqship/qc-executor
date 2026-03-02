"""Factory function for creating backend-specific executor instances with lazy imports."""

import importlib

_BACKENDS = {
    "pennylane": {
        "module": "executor.pennylane.pennylane_executor",
        "class": "PennylaneExecutor",
        "install_hint": "pip install pennylane",
    },
    "qiskit": {
        "module": "executor.qiskit.executor_qiskit",
        "class": "QiskitExecutor",
        "install_hint": "pip install qiskit",
    },
    "qulacs": {
        "module": "executor.qulacs.qulacs_executor",
        "class": "QulacsExecutor",
        "install_hint": "pip install qulacs",
    },
}


def create_executor(backend: str, **kwargs):
    """Create an executor instance for the specified backend.

    Args:
        backend: Backend identifier (case-insensitive).
            Supported values: ``"pennylane"``, ``"qiskit"``, ``"qulacs"``.
        **kwargs: Additional keyword arguments forwarded to the executor constructor.

    Returns:
        An instance of the backend-specific :class:`ExecutorBase` subclass.

    Raises:
        ValueError: If *backend* is not one of the supported backends.
        ImportError: If the third-party library required by the chosen backend
            is not installed.
    """
    key = backend.lower()
    if key not in _BACKENDS:
        supported = ", ".join(sorted(_BACKENDS))
        raise ValueError(
            f"Unknown backend '{backend}'. Supported backends are: {supported}"
        )

    info = _BACKENDS[key]
    try:
        module = importlib.import_module(info["module"])
    except ImportError as exc:
        raise ImportError(
            f"Backend '{backend}' requires an additional package. "
            f"Install it with: {info['install_hint']}"
        ) from exc

    cls = getattr(module, info["class"])
    return cls(**kwargs)
