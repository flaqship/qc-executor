from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, List, overload

import numpy as np

from ..parameters import Parameter, Parameters
from .circuit_base import QuantumCircuitBase
from .operator_base import QuantumOperatorBase


class _BoundedCache(OrderedDict):
    """An ordered dictionary that evicts the oldest entry when a size limit is reached.

    Args:
        max_size (int, optional): Maximum number of entries. Defaults to None (unlimited).
    """

    def __init__(self, max_size: int | None = None):
        super().__init__()
        if max_size is not None and (not isinstance(max_size, int) or max_size <= 0):
            raise ValueError(f"max_size must be None or a positive integer, got {max_size!r}.")
        self.max_size = max_size

    def __setitem__(self, key, value):
        if self.max_size is not None and key not in self and len(self) >= self.max_size:
            self.popitem(last=False)  # evict oldest entry
        super().__setitem__(key, value)


class ExecutorBase(ABC):
    """Base class for quantum circuit executors.

    Args:
        shots (int | None, optional): Number of shots for sampling.
        seed (int | None, optional): Random seed for reproducibility.
        log_file (str | None, optional): Path to the log file.
        log_level (str, optional): Logging level (for example ``"DEBUG"``,
            ``"INFO"``, ``"WARNING"``, ``"ERROR"``).
        caching (bool | None, optional): Whether to cache computation results
            in memory.
        cache_dir (str, optional): Directory for caching.
        max_cache_size (int | None, optional): Maximum number of entries kept
            in each in-memory cache. ``None`` means unlimited.
    """

    _native_circuit_class = None
    _native_observable_class = None

    # ========================================================================
    # Initialization & Configuration
    # ========================================================================

    def __init__(
        self,
        shots: int | None = None,
        seed: int | None = None,
        log_file: str | None = None,
        log_level: str = "WARNING",
        caching: bool | None = None,
        cache_dir: str = "cache",
        max_cache_size: int | None = None,
    ):
        self._shots = shots
        self._seed = seed
        self._log_file = log_file
        self._caching = caching
        self._cache_dir = cache_dir
        self._max_cache_size = max_cache_size

        # Result cache – shared across all public interface methods (method name
        # is part of the key to prevent cross-method collisions).
        self._result_cache = self._make_cache() if caching else None

        # Validate and resolve log level
        _valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        log_level_upper = log_level.upper()
        if log_level_upper not in _valid_levels:
            raise ValueError(
                f"Invalid log_level '{log_level}'. "
                f"Must be one of: {', '.join(sorted(_valid_levels))}."
            )
        level = getattr(logging, log_level_upper)

        # Set up logger using a dotted hierarchy so handlers can be
        # configured at the 'executor' package level by callers.
        logger_name = f"{type(self).__module__}.{type(self).__qualname__}"
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(level)
        if log_file is not None:
            log_file_abs = os.path.abspath(log_file)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            # Avoid registering duplicate file handlers for the same path;
            # update level/formatter on an existing handler instead.
            existing_handler = next(
                (
                    h
                    for h in self._logger.handlers
                    if isinstance(h, logging.FileHandler) and h.baseFilename == log_file_abs
                ),
                None,
            )
            if existing_handler is not None:
                existing_handler.setLevel(level)
                existing_handler.setFormatter(formatter)
            else:
                handler = logging.FileHandler(log_file_abs)
                handler.setLevel(level)
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)

    def _make_cache(self) -> _BoundedCache:
        """Create a new bounded cache with the configured size limit."""
        return _BoundedCache(self._max_cache_size)

    @property
    def shots(self) -> int | None:
        """Return the number of shots."""
        return self._shots

    @shots.setter
    def shots(self, value: int | None) -> None:
        """Set the number of shots."""
        raise NotImplementedError

    @property
    def remote(self) -> bool:
        """Return True if the execution access a remote backend."""
        raise NotImplementedError

    def get_config(self) -> dict:
        """Get the current executor configuration.

        Returns:
            dict: Dictionary containing the executor configuration parameters
                (shots, seed, log_file, log_level, caching, cache_dir, max_cache_size)

        Example:
            >>> executor = Executor.create("qiskit", shots=1024, seed=42)
            >>> config = executor.get_config()
            >>> print(config)  # {'shots': 1024, 'seed': 42, ...}
        """
        return {
            "shots": self._shots,
            "seed": self._seed,
            "log_file": self._log_file,
            "log_level": logging.getLevelName(self._logger.level),
            "caching": self._caching,
            "cache_dir": self._cache_dir,
            "max_cache_size": self._max_cache_size,
        }

    def switch_backend(self, backend: str | Any, **overrides) -> "ExecutorBase":
        """Switch to a different backend while preserving configuration.

        Creates a new executor instance with the specified backend, copying
        the current configuration and applying any overrides.

        Args:
            backend: Name of the backend to switch to (e.g., ``"qiskit"``,
                ``"pennylane"``, ``"qulacs"``).  May also be a Qiskit
                ``Backend`` / ``BackendV2`` instance (e.g. obtained from
                ``QiskitRuntimeService``), in which case the ``"qiskit"``
                executor is used automatically and the object is forwarded
                as ``backend=<instance>``.
            **overrides: Configuration parameters to override (e.g., shots=2048)

        Returns:
            ExecutorBase: New executor instance with the specified backend

        Example:
            >>> executor = Executor.create("qiskit", shots=1024, seed=42)
            >>> pennylane_executor = executor.switch_backend("pennylane")
            >>> # pennylane_executor has shots=1024, seed=42
            >>>
            >>> # Override specific parameters
            >>> qulacs_executor = executor.switch_backend("qulacs", shots=2048)
            >>> # qulacs_executor has shots=2048, seed=42
            >>>
            >>> # Switch to a real IBM Quantum backend
            >>> from qiskit_ibm_runtime import QiskitRuntimeService
            >>> service = QiskitRuntimeService()
            >>> ibm_backend = service.least_busy(operational=True, simulator=False)
            >>> ibm_executor = executor.switch_backend(ibm_backend)
        """
        # Lazy import to avoid circular dependencies
        from executor.factory import Executor  # pylint: disable=cyclic-import

        # Get current config and apply overrides
        config = self.get_config()
        config.update(overrides)

        # Create new executor with specified backend
        self._logger.info("Switching backend from %s to %s", type(self).__name__, backend)
        return Executor.create(backend, **config)

    # ========================================================================
    # Internal Infrastructure
    # ========================================================================

    @staticmethod
    def _make_result_key(method_name: str, *args, **kwargs) -> tuple:
        """Build a hashable cache key for a public-interface call.

        Args:
            method_name: Name of the calling method (prevents key collisions
                between different methods).
            *args: Positional arguments passed to the method.
            ``**kwargs``: Keyword arguments passed to the method.

        Returns:
            A hashable tuple that uniquely identifies the call.
        """

        def _to_hashable(v):
            if isinstance(v, np.ndarray):
                # Include dtype, shape, and strides (memory layout) to avoid collisions
                # between arrays that share the same raw bytes but differ structurally.
                return (v.dtype.str, v.shape, v.strides, v.tobytes())
            if isinstance(v, (list, tuple)):
                return tuple(_to_hashable(i) for i in v)
            try:
                hash(v)
                return v
            except TypeError:
                # Fall back to object identity for unhashable types
                return id(v)

        return (
            (method_name,)
            + tuple(_to_hashable(a) for a in args)
            + tuple(sorted((k, _to_hashable(v)) for k, v in kwargs.items()))
        )

    @staticmethod
    def _normalize_parameter_values(**parameters) -> dict:
        """Normalize parameter values by converting indexed keys to vector keys.

        Converts indexed parameter keys like x[0], x[1] to vector form x=[...].
        This allows unifying parameter passing across backends that expect
        vector-style keys (e.g., ParameterVector names like "x", "p").

        Example:
            Input: x=[0.1, 0.2, 0.3], p=[1.0]
            Output: x=[0.1, 0.2, 0.3], p=[1.0]

            Input: x[0]=0.1, x[1]=0.2, x[2]=0.3, p=[1.0]
            Output: x=[0.1, 0.2, 0.3], p=[1.0]

        Args:
            **parameters: Parameter keyword arguments as passed to public methods.

        Returns:
            Dictionary with normalized parameters using vector keys.
        """
        import re

        normalized = {}
        indexed_params = {}  # Maps "x" -> ["x[0]", "x[1]", ...]

        for key, value in parameters.items():
            # Check for indexed key pattern: "x[i]" or "p[i]"
            match = re.match(r"^([a-zA-Z_]\w*)\[(\d+)\]$", key)
            if match:
                param_name = match.group(1)
                index = int(match.group(2))
                if param_name not in indexed_params:
                    indexed_params[param_name] = {}
                indexed_params[param_name][index] = value
            else:
                # Non-indexed key; keep as-is
                if (
                    key not in indexed_params
                ):  # Don't overwrite vector key if indexed version exists
                    normalized[key] = value

        # Convert collected indexed params to vector form
        for param_name, index_dict in indexed_params.items():
            if param_name in normalized:
                # Both indexed and vector forms provided; indexed takes precedence
                pass
            max_index = max(index_dict.keys())
            vector_form = [index_dict.get(i) for i in range(max_index + 1)]
            normalized[param_name] = vector_form

        return normalized

    # ========================================================================
    # Public API – Core Quantum Operations
    # ========================================================================

    def expectation_value(
        self,
        circuit: QuantumCircuitBase | List[QuantumCircuitBase],
        operator: QuantumOperatorBase | List[QuantumOperatorBase],
        **parameters,
    ) -> float | np.array:
        """
        Calculate the expectation value of the operator with respect to the circuit.

        Args:
            circuit (QuantumCircuitBase | List[QuantumCircuitBase]): The quantum circuit or a list of circuits.
            operator (QuantumOperatorBase | List[QuantumOperatorBase]): The quantum operator or a list of operators.
            parameters: Additional values for the free parameters of the circuit(s) and the operator(s) given as keyword arguments.
                Both vector-style keys (e.g., ``x=[0.1, 0.2]``) and indexed keys
                (e.g., ``x[0]=0.1, x[1]=0.2``) are accepted and normalized.

        Returns:
            float | np.array: The expectation value either as a single float or as an numpy array if multiple circuits/operators are provided.
        """
        self._logger.info("Computing expectation value")
        parameters = self._normalize_parameter_values(**parameters)
        if self._result_cache is not None:
            key = self._make_result_key("expectation_value", circuit, operator, **parameters)
            if key in self._result_cache:
                self._logger.debug("Result cache hit for expectation_value")
                return self._result_cache[key]
            result = self._expectation_value(circuit, operator, **parameters)
            self._result_cache[key] = result
            return result
        return self._expectation_value(circuit, operator, **parameters)

    @abstractmethod
    def _expectation_value(
        self,
        circuit: QuantumCircuitBase | List[QuantumCircuitBase],
        operator: QuantumOperatorBase | List[QuantumOperatorBase],
        **parameters,
    ) -> float | np.array:
        """Abstract implementation of expectation value computation."""
        raise NotImplementedError

    def expectation_value_derivatives(
        self,
        circuit: QuantumCircuitBase | List[QuantumCircuitBase],
        operator: QuantumOperatorBase | List[QuantumOperatorBase],
        *derivative,
        **parameters,
    ) -> float | np.array | dict:
        """
        Calculate the derivatives of the expectation value with respect to the parameters of the circuit.

        Args:
            circuit (QuantumCircuitBase | List[QuantumCircuitBase]): The quantum circuit or a list of circuits.
            operator (QuantumOperatorBase | List[QuantumOperatorBase]): The quantum operator or a list of operators.
            derivative: The parameter(s) with respect to which the derivative is calculated.
            parameters: Additional values for the free parameters of the circuit(s) and the operator(s) given as keyword arguments.
                Both vector-style keys (e.g., ``x=[0.1, 0.2]``) and indexed keys
                (e.g., ``x[0]=0.1, x[1]=0.2``) are accepted and normalized.

        Returns:
            float | np.array | dict: The derivative of the expectation value:
                - single float/array if one derivative parameter is requested
                - dictionary mapping parameter names to gradient arrays if multiple parameters are requested
        """
        self._logger.info("Computing expectation value derivatives")
        parameters = self._normalize_parameter_values(**parameters)
        if self._result_cache is not None:
            key = self._make_result_key(
                "expectation_value_derivatives", circuit, operator, derivative, **parameters
            )
            if key in self._result_cache:
                self._logger.debug("Result cache hit for expectation_value_derivatives")
                return self._result_cache[key]
            result = self._expectation_value_derivatives(
                circuit, operator, *derivative, **parameters
            )
            self._result_cache[key] = result
            return result
        return self._expectation_value_derivatives(circuit, operator, *derivative, **parameters)

    @abstractmethod
    def _expectation_value_derivatives(
        self,
        circuit: QuantumCircuitBase | List[QuantumCircuitBase],
        operator: QuantumOperatorBase | List[QuantumOperatorBase],
        *derivative,
        **parameters,
    ) -> float | np.array | dict:
        """Abstract implementation of expectation value derivatives computation."""
        raise NotImplementedError

    def sample(
        self, circuit: QuantumCircuitBase | List[QuantumCircuitBase], **parameters
    ) -> dict | List[dict]:
        """
        Computes samples of the quantumstate of the given circuit.

        Args:
            circuit (QuantumCircuitBase | List[QuantumCircuitBase]): The quantum circuit or a list of circuits.
            parameters: Additional values for the free parameters of the circuit(s) given as keyword arguments.
                Both vector-style keys (e.g., ``x=[0.1, 0.2]``) and indexed keys
                (e.g., ``x[0]=0.1, x[1]=0.2``) are accepted and normalized.

        Returns:
            dict | List[dict]: The sampled results either as a single dictionary or a list of dictionaries if multiple circuits are provided.
        """
        self._logger.info("Sampling circuit (shots=%s)", self._shots)
        parameters = self._normalize_parameter_values(**parameters)
        if self._result_cache is not None:
            # Include shots in the key so that changing shots invalidates cached samples.
            key = self._make_result_key("sample", circuit, self._shots, **parameters)
            if key in self._result_cache:
                self._logger.debug("Result cache hit for sample")
                return self._result_cache[key]
            result = self._sample(circuit, **parameters)
            self._result_cache[key] = result
            return result
        return self._sample(circuit, **parameters)

    @abstractmethod
    def _sample(
        self, circuit: QuantumCircuitBase | List[QuantumCircuitBase], **parameters
    ) -> dict | List[dict]:
        """Abstract implementation of circuit sampling."""
        raise NotImplementedError

    def statevector(
        self, circuit: QuantumCircuitBase | List[QuantumCircuitBase], **parameters
    ) -> np.ndarray:
        """
        Computes the statevector of the quantum circuit.

        Args:
            circuit (QuantumCircuitBase | List[QuantumCircuitBase]): The quantum circuit or a list of circuits.
            parameters: Additional values for the free parameters of the circuit(s) given as keyword arguments.
                Both vector-style keys (e.g., ``x=[0.1, 0.2]``) and indexed keys
                (e.g., ``x[0]=0.1, x[1]=0.2``) are accepted and normalized.

        Returns:
            np.ndarray: The statevector of the circuit(s).
        """
        self._logger.info("Computing statevector")
        parameters = self._normalize_parameter_values(**parameters)
        if self._result_cache is not None:
            key = self._make_result_key("statevector", circuit, **parameters)
            if key in self._result_cache:
                self._logger.debug("Result cache hit for statevector")
                return self._result_cache[key]
            result = self._statevector(circuit, **parameters)
            self._result_cache[key] = result
            return result
        return self._statevector(circuit, **parameters)

    @abstractmethod
    def _statevector(
        self, circuit: QuantumCircuitBase | List[QuantumCircuitBase], **parameters
    ) -> np.ndarray:
        """Abstract implementation of statevector computation."""
        raise NotImplementedError

    # ========================================================================
    # Public API – Circuit/Observable Handling
    # ========================================================================

    def transpile_circuit(
        self, circuit: QuantumCircuitBase | List[QuantumCircuitBase]
    ) -> QuantumCircuitBase | List[QuantumCircuitBase]:
        """
        Transpile the circuit for execution on this executor's backend.

        Subclasses may override :meth:`_transpile_circuit` to apply
        backend-specific optimisations (e.g. gate decomposition, qubit
        routing).  When a list of circuits is provided, each circuit is
        transpiled and cached individually.

        Args:
            circuit (QuantumCircuitBase | List[QuantumCircuitBase]): The
                quantum circuit or a list of circuits to transpile.

        Returns:
            QuantumCircuitBase | List[QuantumCircuitBase]: The transpiled
                circuit(s).
        """
        self._logger.info("Transpiling circuit")
        if isinstance(circuit, list):
            return [self._transpile_single_cached(c) for c in circuit]
        return self._transpile_single_cached(circuit)

    def _transpile_single_cached(self, circuit: QuantumCircuitBase) -> QuantumCircuitBase:
        """Transpile a single circuit, consulting the result cache if enabled."""
        if self._result_cache is not None:
            key = self._make_result_key("transpile_circuit", circuit)
            if key in self._result_cache:
                self._logger.debug("Result cache hit for transpile_circuit")
                return self._result_cache[key]
            result = self._transpile_circuit(circuit)
            self._result_cache[key] = result
            return result
        return self._transpile_circuit(circuit)

    @abstractmethod
    def _transpile_circuit(self, circuit: QuantumCircuitBase) -> QuantumCircuitBase:
        """Abstract implementation of circuit transpilation."""
        raise NotImplementedError

    @overload
    def transpile_observable(self, operator: QuantumOperatorBase) -> QuantumOperatorBase: ...

    @overload
    def transpile_observable(
        self, operator: List[QuantumOperatorBase]
    ) -> List[QuantumOperatorBase]: ...

    def transpile_observable(
        self,
        operator: QuantumOperatorBase | List[QuantumOperatorBase],
    ) -> QuantumOperatorBase | List[QuantumOperatorBase]:
        """
        Transpile the operator for execution on this executor's backend.

        Subclasses may override :meth:`_transpile_observable` to apply
        backend-specific conversions (e.g., to wrapper types).  When a list
        of operators is provided, each operator is transpiled and cached individually.

        Args:
            operator (QuantumOperatorBase | List[QuantumOperatorBase]): The
                quantum operator or a list of operators to transpile.

        Returns:
            QuantumOperatorBase | List[QuantumOperatorBase]: The transpiled
                operator(s).
        """
        self._logger.info("Transpiling operator")
        if isinstance(operator, list):
            return [self._transpile_observable_cached(op) for op in operator]
        return self._transpile_observable_cached(operator)

    def _transpile_observable_cached(self, operator: QuantumOperatorBase) -> QuantumOperatorBase:
        """Transpile a single observable, consulting the result cache if enabled."""
        if self._result_cache is not None:
            key = self._make_result_key("transpile_observable", operator)
            if key in self._result_cache:
                self._logger.debug("Result cache hit for transpile_observable")
                return self._result_cache[key]
            result = self._transpile_observable(operator)
            self._result_cache[key] = result
            return result
        return self._transpile_observable(operator)

    @abstractmethod
    def _transpile_observable(self, operator: QuantumOperatorBase) -> QuantumOperatorBase:
        """Abstract implementation of observable transpilation.

        Subclasses override this to convert generic QuantumOperator to
        backend-native types. For backends supporting symmetry (e.g.,
        Pauli Propagation), the symmetry_strategy parameter allows
        assigning a symmetry strategy to the operator.

        Args:
            operator (QuantumOperatorBase): The operator to transpile.

        Returns:
            QuantumOperatorBase: The transpiled operator in backend-native format.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_accepted_backend_types(cls) -> List[type]:
        """Return a list of backend object types accepted by this executor.

        This is used for auto-detection when a non-string backend is passed to
        :meth:`Executor.create`.  If the backend object is an instance of any
        of the returned types, this executor will be selected automatically.

        Returns:
            List[type]: List of accepted backend types (e.g., Qiskit ``Backend`` / ``BackendV2`` classes)
        """
        raise NotImplementedError

    @classmethod
    def get_accepted_backend_aliases(cls) -> List[str]:
        """Return string aliases accepted by this executor.

        This optional list is used by :meth:`Executor.create` when a string
        target is not a registered backend name. The factory resolves aliases
        to the owning plugin and forwards the original string via
        ``backend=<target>``.

        Returns:
            List[str]: String aliases accepted by the executor.
        """
        return []
