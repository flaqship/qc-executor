"""Base class for quantum circuit executors across different quantum frameworks."""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, overload

import numpy as np

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
            in each in-memory cache; ``None`` makes them unbounded.
    """

    _native_circuit_class = None
    _native_operator_class = None

    # ========================================================================
    # Initialization & Configuration
    # ========================================================================

    def __init__(
        self,
        backend: Any = None,
        shots: int | None = None,
        seed: int | None = None,
        log_file: str | None = None,
        log_level: str = "WARNING",
        caching: bool | None = None,
        cache_dir: str = "cache",
        max_cache_size: int | None = 4096,
    ):
        self._backend = backend
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

    def switch_backend(self, backend: Any, **overrides) -> "ExecutorBase":
        """Switch to a different backend while preserving configuration.

        Delegates to :meth:`Executor.switch_backend <qc_executor.factory.Executor.switch_backend>`.

        Args:
            backend: Name of the backend (e.g., ``"qiskit"``, ``"pennylane"``)
                or a backend instance for auto-detection.
            **overrides: Configuration parameters to override (e.g., shots=2048)

        Returns:
            ExecutorBase: New executor instance with the specified backend
        """
        from qc_executor.factory import (  # pylint: disable=import-outside-toplevel,cyclic-import
            Executor,
        )

        return Executor.switch_backend(self, backend, **overrides)

    # ========================================================================
    # Internal Infrastructure
    # ========================================================================

    @staticmethod
    def _structural_cache_key(obj):
        """Return a mutation-aware cache key for a circuit or operator."""
        method = getattr(obj, "structural_key", None)
        if method is not None:
            try:
                return (type(obj).__qualname__, method())
            except NotImplementedError:
                pass
        return obj

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
            if isinstance(v, (QuantumCircuitBase, QuantumOperatorBase)):
                # Key circuits/operators by their current structure so that
                # in-place mutations never hit stale cached results.
                return ExecutorBase._structural_cache_key(v)
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

            Mixing x=[...] with x[0]=... is not allowed.

        Args:
            **parameters: Parameter keyword arguments as passed to public methods.

        Returns:
            Dictionary with normalized parameters using vector keys.
        """
        normalized = {}
        indexed_params = {}  # Maps "x" -> ["x[0]", "x[1]", ...]
        vector_params = set()

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
                vector_params.add(key)
                normalized[key] = value

        conflicting_params = sorted(vector_params.intersection(indexed_params))
        if conflicting_params:
            raise ValueError(
                "Cannot mix vector and indexed parameter forms for: "
                f"{', '.join(conflicting_params)}"
            )

        # Convert collected indexed params to vector form
        for param_name, index_dict in indexed_params.items():
            max_index = max(index_dict.keys())
            vector_form = [index_dict.get(i) for i in range(max_index + 1)]
            if any(value is None for value in vector_form):
                raise ValueError(
                    f"Incomplete indexed parameters for '{param_name}': "
                    "missing indices would produce None values in the vector form."
                )
            normalized[param_name] = vector_form

        return normalized

    # ========================================================================
    # Public API – Core Quantum Operations
    # ========================================================================

    def expectation_value(
        self,
        circuit: QuantumCircuitBase | List[QuantumCircuitBase],
        observable: QuantumOperatorBase | List[QuantumOperatorBase],
        **parameters,
    ) -> float | np.ndarray:
        """
        Calculate the expectation value of the observable with respect to the circuit.

        Args:
            circuit (QuantumCircuitBase | List[QuantumCircuitBase]): The quantum circuit
                or a list of circuits.
            observable (QuantumOperatorBase | List[QuantumOperatorBase]): The quantum
                observable or a list of observables.
            parameters: Additional values for the free parameters of the circuit(s) and
                the observable(s) given as keyword arguments.
                Both vector-style keys (e.g., ``x=[0.1, 0.2]``) and indexed keys
                (e.g., ``x[0]=0.1, x[1]=0.2``) are accepted and normalized.

        Returns:
            float | np.array: The expectation value either as a single float or as a
                numpy array if multiple circuits/observables are provided.
        """
        self._logger.info("Computing expectation value")
        parameters = self._normalize_parameter_values(**parameters)
        if self._result_cache is not None:
            key = self._make_result_key("expectation_value", circuit, observable, **parameters)
            if key in self._result_cache:
                self._logger.debug("Result cache hit for expectation_value")
                return self._result_cache[key]
            result = self._expectation_value(circuit, observable, **parameters)
            self._result_cache[key] = result
            return result
        return self._expectation_value(circuit, observable, **parameters)

    @abstractmethod
    def _expectation_value(
        self,
        circuit: QuantumCircuitBase | List[QuantumCircuitBase],
        observable: QuantumOperatorBase | List[QuantumOperatorBase],
        **parameters,
    ) -> float | np.ndarray:
        """Abstract implementation of expectation value computation."""
        raise NotImplementedError

    def expectation_value_derivatives(
        self,
        circuit: QuantumCircuitBase | List[QuantumCircuitBase],
        observable: QuantumOperatorBase | List[QuantumOperatorBase],
        *derivative,
        **parameters,
    ) -> float | np.ndarray | dict:
        """
        Calculate the derivatives of the expectation value with respect to the
        parameters of the circuit.

        The return format and the ordering below are part of the public API;
        downstream consumers rely on both. The full gradient of a circuit is
        obtained by supplying all of its parameter-vector names (see
        :attr:`QuantumCircuitBase.parameter_vector_names`) as ``derivative``
        arguments.

        Args:
            circuit (QuantumCircuitBase | List[QuantumCircuitBase]): The quantum circuit
                or a list of circuits.
            observable (QuantumOperatorBase | List[QuantumOperatorBase]): The quantum
                observable or a list of observables. Lists of circuits and
                observables are evaluated combinatorially: every circuit is
                paired with every observable.
            derivative: The parameter(s) with respect to which the derivative is calculated.
            parameters: Additional values for the free parameters of the circuit(s) and
                the observable(s) given as keyword arguments.
                Both vector-style keys (e.g., ``x=[0.1, 0.2]``) and indexed keys
                (e.g., ``x[0]=0.1, x[1]=0.2``) are accepted and normalized.

        Returns:
            float | np.array | dict: The derivative of the expectation value:
                - single float/array if one derivative parameter is requested
                - dictionary mapping parameter names to gradient arrays if multiple
                  parameters are requested

            A list input adds a leading axis per list: ``(n_circuits, ...)``
            for a circuit list, ``(n_observables, ...)`` for an observable
            list, and ``(n_circuits, n_observables, ...)`` for both.

            Entries within each parameter vector are ordered by numeric
            element index, i.e. ``theta[2]`` precedes ``theta[10]``.
        """
        self._logger.info("Computing expectation value derivatives")
        parameters = self._normalize_parameter_values(**parameters)

        key = None
        if self._result_cache is not None:
            key = self._make_result_key(
                "expectation_value_derivatives", circuit, observable, derivative, **parameters
            )
            if key in self._result_cache:
                self._logger.debug("Result cache hit for expectation_value_derivatives")
                return self._result_cache[key]

        # Plugins only implement single-pair derivatives; list inputs are
        # expanded combinatorially and the per-pair results are stacked with
        # one leading axis per list input.
        # TODO: Every circuit/observable pair is evaluated separately. Allow
        # plugins to override this loop with native batching (e.g. qiskit
        # Estimator PUBs, pennylane's qml.execute) where supported.
        multiple_circuits = isinstance(circuit, (list, tuple))
        multiple_observables = isinstance(observable, (list, tuple))
        if not multiple_circuits and not multiple_observables:
            result = self._expectation_value_derivatives(
                circuit, observable, *derivative, **parameters
            )
        else:
            circuits = list(circuit) if multiple_circuits else [circuit]
            observables = list(observable) if multiple_observables else [observable]
            pair_results = [
                [
                    self._expectation_value_derivatives(c, o, *derivative, **parameters)
                    for o in observables
                ]
                for c in circuits
            ]

            def collapse(nested) -> np.ndarray:
                values = np.asarray(nested)
                if not multiple_circuits:
                    return values[0]
                if not multiple_observables:
                    return values[:, 0]
                return values

            if isinstance(pair_results[0][0], dict):
                # Multiple derivative parameters: stack the per-pair dicts per name.
                result = {
                    name: collapse([[pair[name] for pair in row] for row in pair_results])
                    for name in pair_results[0][0]
                }
            else:
                result = collapse(pair_results)

        if self._result_cache is not None:
            self._result_cache[key] = result
        return result

    @abstractmethod
    def _expectation_value_derivatives(
        self,
        circuit: QuantumCircuitBase | List[QuantumCircuitBase],
        observable: QuantumOperatorBase | List[QuantumOperatorBase],
        *derivative,
        **parameters,
    ) -> float | np.ndarray | dict:
        """Abstract implementation of expectation value derivatives computation."""
        raise NotImplementedError

    def sample(
        self, circuit: QuantumCircuitBase | List[QuantumCircuitBase], **parameters
    ) -> dict | List[dict]:
        """
        Computes samples of the quantumstate of the given circuit.

        Bitstring keys use the big-endian convention ``q[0]q[1]...q[n-1]``
        (qubit 0 is the leftmost character).

        Args:
            circuit (QuantumCircuitBase | List[QuantumCircuitBase]): The quantum circuit
                or a list of circuits.
            parameters: Additional values for the free parameters of the circuit(s)
                given as keyword arguments.
                Both vector-style keys (e.g., ``x=[0.1, 0.2]``) and indexed keys
                (e.g., ``x[0]=0.1, x[1]=0.2``) are accepted and normalized.

        Returns:
            dict | List[dict]: The sampled results either as a single dictionary or a
                list of dictionaries if multiple circuits are provided.
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
        """Abstract implementation of circuit sampling.

        Returned bitstring keys must use the big-endian convention
        ``q[0]q[1]...q[n-1]`` (qubit 0 is the leftmost character).
        """
        raise NotImplementedError

    def statevector(
        self, circuit: QuantumCircuitBase | List[QuantumCircuitBase], **parameters
    ) -> np.ndarray:
        """
        Computes the statevector of the quantum circuit.

        Amplitudes use the big-endian basis ordering: index ``i`` encodes
        qubit 0 as the most significant bit.

        Args:
            circuit (QuantumCircuitBase | List[QuantumCircuitBase]): The quantum circuit
                or a list of circuits.
            parameters: Additional values for the free parameters of the circuit(s)
                given as keyword arguments.
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
        """Abstract implementation of statevector computation.

        The returned amplitudes must use the big-endian basis ordering
        (qubit 0 is the most significant bit of the index).
        """
        raise NotImplementedError

    def probabilities(
        self, circuit: QuantumCircuitBase, *, cutoff: float = 0.0, **parameters
    ) -> Dict[int, float]:
        """Compute the measurement probabilities of a circuit.

        With ``shots=None`` the probabilities are exact (derived from the
        statevector); otherwise they are estimated from sampled counts.

        Args:
            circuit (QuantumCircuitBase): A single quantum circuit.
            cutoff (float, optional): Only probabilities strictly greater
                than this threshold are returned. The default of ``0.0``
                omits exact zeros only, matching what the backends report
                natively; pass a larger value to keep the result sparse for
                many qubits.
            parameters: Values for the free parameters of the circuit given as
                keyword arguments.

        Returns:
            Dict[int, float]: Mapping of basis-state index (big-endian
            ordering, qubit 0 = most significant bit) to probability.

        Raises:
            ValueError: If a list of circuits or multiple parameter sets are
                passed.
        """
        if isinstance(circuit, list):
            raise ValueError("probabilities supports a single circuit only")
        if self._shots is None:
            state_vector = np.asarray(self.statevector(circuit, **parameters))
            probability_values = np.abs(state_vector) ** 2
            return {
                index: float(value)
                for index, value in enumerate(probability_values)
                if value > cutoff
            }
        counts = self.sample(circuit, **parameters)
        # unwrap the single-set case since some backends return one counts dict per parameter set.
        if isinstance(counts, list):
            if len(counts) != 1:
                raise ValueError("probabilities supports a single parameter set only")
            counts = counts[0]
        total = sum(counts.values())
        # Prune with the same threshold as the exact branch, so that switching
        # `shots` on or off does not change which entries are reported.
        probabilities = ((int(str(bits), 2), count / total) for bits, count in counts.items())
        return {index: value for index, value in probabilities if value > cutoff}

    # ========================================================================
    # Public API – Circuit/Operator Handling
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
    def transpile_operator(self, operator: QuantumOperatorBase) -> QuantumOperatorBase: ...

    @overload
    def transpile_operator(
        self, operator: List[QuantumOperatorBase]
    ) -> List[QuantumOperatorBase]: ...

    def transpile_operator(
        self,
        operator: QuantumOperatorBase | List[QuantumOperatorBase],
    ) -> QuantumOperatorBase | List[QuantumOperatorBase]:
        """
        Transpile the operator for execution on this executor's backend.

        Subclasses may override :meth:`_transpile_operator` to apply
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
            return [self._transpile_operator_cached(operator) for operator in operator]
        return self._transpile_operator_cached(operator)

    def _transpile_operator_cached(self, operator: QuantumOperatorBase) -> QuantumOperatorBase:
        """Transpile a single observable, consulting the result cache if enabled."""
        if self._result_cache is not None:
            key = self._make_result_key("transpile_operator", operator)
            if key in self._result_cache:
                self._logger.debug("Result cache hit for transpile_operator")
                return self._result_cache[key]
            result = self._transpile_operator(operator)
            self._result_cache[key] = result
            return result
        return self._transpile_operator(operator)

    @abstractmethod
    def _transpile_operator(self, operator: QuantumOperatorBase) -> QuantumOperatorBase:
        """Abstract implementation of operator transpilation.

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
            List[type]: List of accepted backend types
                (e.g., Qiskit ``Backend`` / ``BackendV2`` classes)
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
