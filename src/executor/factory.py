"""Factory class for creating executor instances with plugin support."""

from typing import Any, Callable, Type, Union
import logging

logger = logging.getLogger(__name__)


class Executor:
    """Factory class for creating executor instances based on backend name.

    This class provides a plugin-based architecture for executor backends.
    Backends can be registered using the @Executor.register() decorator or
    discovered automatically via entry points.

    Example:
        >>> executor = Executor.create("qiskit", shots=1024)
        >>> backends = Executor.available_backends()
        >>> print(backends)  # ['qiskit', 'pennylane', 'qulacs']
    """

    _registry: dict[str, Type["ExecutorBase"]] = {}
    _plugins_discovered: bool = False
    _backend_extra_map: dict[str, str] = {
        "qiskit": "qiskit-full",
        "pennylane": "pennylane",
        "qulacs": "qulacs",
    }

    def __init__(self):
        """Executor cannot be instantiated. Use Executor.create() instead."""
        raise TypeError(
            "Executor cannot be instantiated directly. "
            "Use Executor.create(backend_name, **kwargs) instead."
        )

    @classmethod
    def register(cls, name: str) -> Callable[[Type["ExecutorBase"]], Type["ExecutorBase"]]:
        """Decorator to register a backend implementation.

        Args:
            name: The name of the backend (e.g., "qiskit", "pennylane", "qulacs")

        Returns:
            Decorator function that registers the backend class

        Raises:
            TypeError: If the decorated class does not inherit from ExecutorBase

        Example:
            >>> @Executor.register("qiskit")
            ... class QiskitExecutor(ExecutorBase):
            ...     pass
        """

        def decorator(backend_class: Type["ExecutorBase"]) -> Type["ExecutorBase"]:
            # Import here to avoid circular imports
            from executor.base.executor_base import ExecutorBase

            if not issubclass(backend_class, ExecutorBase):
                raise TypeError(
                    f"{backend_class.__name__} must inherit from ExecutorBase "
                    f"to be registered as a backend"
                )

            cls._registry[name] = backend_class
            logger.debug(f"Registered backend '{name}': {backend_class.__name__}")
            return backend_class

        return decorator

    @classmethod
    def create(cls, backend: Union[str, Any], **kwargs) -> "ExecutorBase":
        """Create an executor instance for the specified backend.

        Args:
            backend: Name of the backend (e.g., "qiskit", "pennylane", "qulacs").
                May also be a Qiskit ``Backend`` / ``BackendV2`` instance, in
                which case the ``"qiskit"`` executor is used automatically and
                the object is forwarded as ``backend=<instance>``.
            **kwargs: Configuration parameters passed to the backend constructor

        Returns:
            An instance of the requested backend executor

        Raises:
            ValueError: If the backend is not found or not installed

        Example:
            >>> executor = Executor.create("qiskit", shots=1024, seed=42)
            >>> executor = Executor.create("pennylane", shots=1000)
        """
        # Accept a Backend object and route to the qiskit executor
        if not isinstance(backend, str):
            return cls.from_backend(backend, **kwargs)

        # Try to find backend in registry
        if backend not in cls._registry:
            # Try discovering plugins if not done yet
            if not cls._plugins_discovered:
                cls._discover_plugins()

        # Check if backend is available
        if backend not in cls._registry:
            available = cls.available_backends()
            available_str = ", ".join(f"'{b}'" for b in available) if available else "none"
            extra_name = cls._backend_extra_map.get(backend, backend)

            raise ValueError(
                f"Backend '{backend}' not found. "
                f"Available backends: {available_str}. "
                f"Install with: pip install executor[{extra_name}]"
            )

        # Create and return backend instance
        backend_class = cls._registry[backend]
        logger.info(f"Creating {backend_class.__name__} with config: {kwargs}")
        return backend_class(**kwargs)

    @classmethod
    def from_backend(cls, backend, **kwargs) -> "ExecutorBase":
        """Create an executor from a Qiskit ``Backend`` / ``BackendV2`` instance.

        This is a convenience entry-point for IBM Quantum hardware and
        noise-aware fake backends.  The *backend* object is forwarded to
        :class:`~executor.qiskit.qiskit_executor.QiskitExecutor`.

        Args:
            backend: A Qiskit :class:`~qiskit.providers.Backend` instance
                (e.g. obtained from ``QiskitRuntimeService``).
            **kwargs: Extra arguments forwarded to ``QiskitExecutor``
                (``shots``, ``seed``, ``execution_mode``, ``options``, …).

        Returns:
            A ``QiskitExecutor`` instance configured for the given backend.

        Raises:
            ValueError: If the ``"qiskit"`` backend is not registered.

        Example:
            >>> from qiskit_ibm_runtime import QiskitRuntimeService
            >>> service = QiskitRuntimeService()
            >>> ibm_backend = service.least_busy(operational=True, simulator=False)
            >>> executor = Executor.from_backend(ibm_backend, shots=1024)
        """
        # Ensure the qiskit backend is discovered
        if "qiskit" not in cls._registry:
            if not cls._plugins_discovered:
                cls._discover_plugins()

        if "qiskit" not in cls._registry:
            raise ValueError(
                "The 'qiskit' backend is not available. "
                "Install with: pip install executor[qiskit-full]"
            )

        backend_class = cls._registry["qiskit"]
        logger.info(
            "Creating %s from backend instance: %s (kwargs: %s)",
            backend_class.__name__,
            backend,
            kwargs,
        )
        return backend_class(backend=backend, **kwargs)

    @classmethod
    def _discover_plugins(cls) -> None:
        """Discover and load plugins via entry points.

        This method searches for entry points in the 'executor.backends' group
        and loads them. Loading a plugin module triggers its @register decorator,
        which adds the backend to the registry.
        """
        cls._plugins_discovered = True

        from importlib.metadata import entry_points

        # Get entry points for executor backends
        try:
            # Python 3.12+ API: supports group keyword argument
            eps = entry_points(group="executor.backends")
        except TypeError:
            # Python 3.10–3.11 API: entry_points() returns a Selection object
            all_eps = entry_points()
            # Selection.select() is available on these versions; fall back to
            # dict-style access only if needed for very old backports.
            select = getattr(all_eps, "select", None)
            if callable(select):
                eps = select(group="executor.backends")
            else:
                eps = all_eps.get("executor.backends", [])

        # Load each entry point
        for ep in eps:
            try:
                logger.debug(f"Loading plugin entry point: {ep.name}")
                ep.load()  # This triggers the @register decorator
            except Exception as e:
                logger.warning(f"Failed to load plugin '{ep.name}': {e}")

    @classmethod
    def available_backends(cls) -> list[str]:
        """Get a list of available (installed) backends.

        Returns:
            List of backend names that can be used with create()

        Example:
            >>> backends = Executor.available_backends()
            >>> print(backends)  # ['qiskit', 'pennylane', 'qulacs']
        """
        # Ensure plugins are discovered
        if not cls._plugins_discovered:
            cls._discover_plugins()

        return sorted(cls._registry.keys())
