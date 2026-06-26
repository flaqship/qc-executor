"""Factory class for creating executor instances with plugin support."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any, Callable, Type, cast

from .base.executor_base import ExecutorBase

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
    _backend_alias_map: dict[str, str] = {}
    _alias_registry_size: int = 0
    _plugins_discovered: bool = False
    _backend_extra_map: dict[str, str] = {
        "qiskit": "qiskit-full",
        "pennylane": "pennylane",
        "qulacs": "qulacs",
        "pauli_propagation": "pauli_propagation",
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
            if not issubclass(backend_class, ExecutorBase):
                raise TypeError(
                    f"{backend_class.__name__} must inherit from ExecutorBase "
                    f"to be registered as a backend"
                )

            cls._registry[name] = backend_class
            cls._index_backend_aliases(name, backend_class)
            cls._alias_registry_size = len(cls._registry)
            logger.debug("Registered backend '%s': %s", name, backend_class.__name__)
            return backend_class

        return decorator

    @classmethod
    def create(cls, target: str | Any, **kwargs) -> "ExecutorBase":
        """Create an executor instance for the specified backend.

        Args:
            target: Name of the backend (e.g., "qiskit", "pennylane", "qulacs").
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

        # Try discovering plugins if not done yet
        if not cls._plugins_discovered:
            cls._discover_plugins()

        # Keep alias map in sync if tests or callers manipulated _registry directly.
        cls._ensure_alias_map_consistency()

        # String path: look up in registry and create instance
        if isinstance(target, str):
            target_alias = cls._normalize_backend_alias(target)

            # Exact backend names take precedence over aliases.
            if target in cls._registry:
                backend_class = cls._registry[target]
                logger.info("Creating %s with config: %s", backend_class.__name__, kwargs)
                return backend_class(**kwargs)

            alias_backend_name = cls._backend_alias_map.get(target_alias)
            if alias_backend_name is not None:
                backend_class = cls._registry.get(alias_backend_name)
                if backend_class is None:
                    cls._rebuild_backend_alias_map()
                    alias_backend_name = cls._backend_alias_map.get(target_alias)
                    backend_class = (
                        cls._registry.get(alias_backend_name)
                        if alias_backend_name is not None
                        else None
                    )

                if backend_class is not None:
                    if "backend" in kwargs:
                        raise ValueError(
                            "Conflicting 'backend' argument: the backend was specified "
                            f"both via alias routing ('{target}') and explicitly in "
                            "**kwargs. Please specify the backend only once."
                        )
                    logger.info(
                        "Routing string target '%s' to backend '%s' via alias.",
                        target,
                        alias_backend_name,
                    )
                    return backend_class(backend=target_alias, **kwargs)

            raise cls._build_backend_not_found_error(target, target_alias)

        # Non-string path: auto-detect executor from accepted_types
        for name, executor_class in cls._registry.items():
            try:
                accepted = executor_class.get_accepted_backend_types()
            except (ImportError, NotImplementedError, AttributeError):
                logger.debug(
                    "get_accepted_backend_types() failed for '%s'; skipping.", name, exc_info=True
                )
                continue

            if any(isinstance(target, t) for t in accepted):
                logger.info(
                    "Auto-detected backend '%s' for object of type %s.",
                    name,
                    type(target).__name__,
                )
                return executor_class(backend=target, **kwargs)

        # No plugin matched — produce a helpful error message
        accepted_summary = {
            name: [t.__name__ for t in executor_class.get_accepted_backend_types()]
            for name, executor_class in cls._registry.items()
            if executor_class.get_accepted_backend_types()
        }
        raise ValueError(
            f"No registered executor accepts an object of type "
            f"'{type(target).__qualname__}'. "
            f"Pass the backend name as a string instead, or install the "
            f"matching plugin.\n"
            f"Accepted types per backend: {accepted_summary}"
        )

    @classmethod
    def _discover_plugins(cls) -> None:
        """Discover and load plugins via entry points.

        This method searches for entry points in the 'qc_executor.backends' group
        and loads them. Loading a plugin module triggers its @register decorator,
        which adds the backend to the registry.
        """
        cls._plugins_discovered = True

        # Get entry points for executor backends
        try:
            # Python 3.10+ API: supports group keyword argument
            eps = entry_points(group="qc_executor.backends")
        except TypeError:
            # Python < 3.10: entry_points() returns a dict-like mapping
            _all_eps = entry_points()
            _select = getattr(_all_eps, "select", None)
            if callable(_select):
                eps = cast(list, _select(group="qc_executor.backends"))
            else:
                eps = cast(
                    list, getattr(_all_eps, "get", lambda *a: [])("qc_executor.backends", [])
                )

        # Load each entry point
        for ep in eps:
            try:
                logger.debug("Loading plugin entry point: %s", ep.name)
                ep.load()  # This triggers the @register decorator
            except (ImportError, AttributeError) as e:
                logger.warning("Failed to load plugin '%s': %s", ep.name, e)

    @classmethod
    def _build_backend_not_found_error(cls, target: str, target_alias: str) -> ValueError:
        """Build a helpful ValueError for an unrecognised backend name."""
        available = cls.available_backends()
        available_str = ", ".join(f"'{b}'" for b in available) if available else "none"
        aliases = sorted(cls._backend_alias_map.keys())
        aliases_str = ", ".join(f"'{a}'" for a in aliases) if aliases else "none"

        resolved = cls._backend_alias_map.get(target_alias)
        if resolved in cls._backend_extra_map:
            backend_key = resolved
        elif target in cls._backend_extra_map:
            backend_key = target
        else:
            backend_key = None

        message = (
            f"Backend '{target}' not found. "
            f"Available backends: {available_str}. "
            f"Known backend aliases: {aliases_str}."
        )
        if backend_key is not None:
            extra_name = cls._backend_extra_map[backend_key]
            message += f" Install with: pip install qc-executor[{extra_name}]"
        return ValueError(message)

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

    @classmethod
    def _normalize_backend_alias(cls, alias: str) -> str:
        """Normalize a backend alias for stable lookups and duplicate checks."""
        return alias.strip().lower()

    @classmethod
    def _index_backend_aliases(
        cls, backend_name: str, backend_class: Type["ExecutorBase"]
    ) -> None:
        """Add aliases for a backend to the central alias map.

        Raises:
            ValueError: If an alias is already owned by another backend.
        """
        aliases = backend_class.get_accepted_backend_aliases()
        for alias in aliases:
            normalized_alias = cls._normalize_backend_alias(str(alias))
            if not normalized_alias:
                continue

            existing_backend = cls._backend_alias_map.get(normalized_alias)
            if existing_backend is not None and existing_backend != backend_name:
                raise ValueError(
                    f"Duplicate backend alias '{normalized_alias}' declared by "
                    f"'{backend_name}' and '{existing_backend}'."
                )

            cls._backend_alias_map[normalized_alias] = backend_name

    @classmethod
    def _rebuild_backend_alias_map(cls) -> None:
        """Rebuild the alias map from the current backend registry.

        This is a safety net for tests or integrations that modify ``_registry``
        directly instead of using :meth:`register`.
        """
        new_alias_map: dict[str, str] = {}
        for backend_name, backend_class in cls._registry.items():
            aliases = backend_class.get_accepted_backend_aliases()
            for alias in aliases:
                normalized_alias = cls._normalize_backend_alias(str(alias))
                if not normalized_alias:
                    continue

                existing_backend = new_alias_map.get(normalized_alias)
                if existing_backend is not None and existing_backend != backend_name:
                    raise ValueError(
                        f"Duplicate backend alias '{normalized_alias}' declared by "
                        f"'{backend_name}' and '{existing_backend}'."
                    )

                new_alias_map[normalized_alias] = backend_name

        cls._backend_alias_map = new_alias_map
        cls._alias_registry_size = len(cls._registry)

    @classmethod
    def _ensure_alias_map_consistency(cls) -> None:
        """Rebuild alias map when registry changes outside of ``register``."""
        if len(cls._registry) != cls._alias_registry_size:
            cls._rebuild_backend_alias_map()

    @classmethod
    def switch_backend(
        cls, executor: "ExecutorBase", backend: str | Any, **overrides
    ) -> "ExecutorBase":
        """Switch an executor to a different backend while preserving its configuration.

        Creates a new executor instance with the specified backend, copying
        the current configuration and applying any overrides.

        Args:
            executor (ExecutorBase): The existing executor whose configuration
                should be copied.
            backend: Name of the backend to switch to (e.g., ``"qiskit"``,
                ``"pennylane"``, ``"qulacs"``).  May also be a Qiskit
                ``Backend`` / ``BackendV2`` instance, in which case the
                ``"qiskit"`` executor is used automatically.
            **overrides: Configuration parameters to override (e.g., shots=2048)

        Returns:
            ExecutorBase: New executor instance with the specified backend

        Example:
            >>> executor = Executor.create("qiskit", shots=1024, seed=42)
            >>> pennylane_executor = Executor.switch_backend(executor, "pennylane")
            >>> # pennylane_executor has shots=1024, seed=42
            >>>
            >>> # Override specific parameters
            >>> qulacs_executor = Executor.switch_backend(executor, "qulacs", shots=2048)
            >>> # qulacs_executor has shots=2048, seed=42
            >>>
            >>> # Switch to a real IBM Quantum backend
            >>> from qiskit_ibm_runtime import QiskitRuntimeService
            >>> service = QiskitRuntimeService()
            >>> ibm_backend = service.least_busy(operational=True, simulator=False)
            >>> ibm_executor = Executor.switch_backend(executor, ibm_backend)
        """
        config = executor.get_config()
        config.update(overrides)
        logger.info("Switching backend from %s to %s", type(executor).__name__, backend)
        return cls.create(backend, **config)
