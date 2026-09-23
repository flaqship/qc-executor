"""Base class for quantum circuit executors across different quantum frameworks."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, overload

import numpy as np

from .circuit_base import QuantumCircuitBase
from .gate_set import OpCode
from .operator_base import QuantumOperatorBase
from .parameters_base import normalize_values

#: Non-unitary opcodes that make a statevector ill defined.  Both collapse the
#: state randomly, so the "statevector" would be one sample from a mixture.
_STOCHASTIC_OPCODES = {OpCode.MEASURE: "mid-circuit measurement", OpCode.RESET: "reset"}


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

    #: The circuit type this backend executes natively, if any.
    _native_circuit_class: type | None = None
    #: The operator type this backend executes natively, if any.
    _native_operator_class: type | None = None

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
        # Conversion cache – generic circuits converted to the native type,
        # keyed by content. Always on: converting (and compiling) the same
        # circuit again for every call would dominate iterative workloads such
        # as VQE, where one circuit is evaluated thousands of times.
        self._conversion_cache = self._make_cache()

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

            Mixing x=[...] with x[0]=... is not allowed.

        Args:
            **parameters: Parameter keyword arguments as passed to public methods.

        Returns:
            Dictionary with normalized parameters using vector keys.
        """
        return normalize_values(**parameters)

    # ========================================================================
    # Input coercion
    # ========================================================================

    def _to_native_circuit(self, circuit):
        """Convert a single circuit to this backend's native type.

        The result is cached by circuit content and handed to every later
        call with an equal circuit, so it must be a pure function of its input
        and the backend must not mutate the returned object afterwards.

        This is a *type* conversion only.  Backends whose
        :meth:`transpile_circuit` also targets a specific device — for example
        mapping onto a machine's qubit count — override this so that the device
        step stays out of the coercion path, where it would desynchronise the
        circuit from its observable.

        Args:
            circuit: A single circuit.

        Returns:
            The circuit in native form.
        """
        return self.transpile_circuit(circuit)

    def _coerce_circuit(self, circuit):
        """Return ``circuit`` in this backend's native form.

        Circuits already native to this backend pass through untouched;
        anything else is converted.  This is what lets the public methods accept
        a generic circuit or a native one interchangeably.

        Args:
            circuit: A single circuit or a list of them.

        Returns:
            The circuit(s) in native form.
        """
        if isinstance(circuit, list):
            return [self._coerce_circuit(item) for item in circuit]
        # pylint: disable=isinstance-second-argument-not-valid-type
        native_class = self._native_circuit_class
        if native_class is not None and isinstance(circuit, native_class):
            return circuit
        ir = getattr(circuit, "ir", None)
        if ir is None:
            return self._to_native_circuit(circuit)
        # The fingerprint is cached per IR revision, so a mutated circuit gets
        # a new key and an unchanged one is looked up at the cost of a hash.
        key = (type(circuit).__qualname__, ir.fingerprint())
        native = self._conversion_cache.get(key)
        if native is None:
            native = self._to_native_circuit(circuit)
            self._conversion_cache[key] = native
        return native

    def _coerce_operator(self, operator, **options):
        """Return ``operator`` in this backend's native form.

        Args:
            operator: A single operator or a list of them.
            ``**options``: Backend-specific transpilation options.

        Returns:
            The operator(s) in native form.
        """
        if isinstance(operator, list):
            return [self._coerce_operator(item, **options) for item in operator]
        # pylint: disable=isinstance-second-argument-not-valid-type
        native_class = self._native_operator_class
        if native_class is not None and isinstance(operator, native_class) and not options:
            return operator
        return self.transpile_operator(operator, **options)

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
        # Coerce before keying so a generic input and its native equivalent
        # share one cache entry.
        circuit = self._coerce_circuit(circuit)
        observable = self._coerce_operator(observable)
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
                paired with every observable. Observable lists are handled
                natively by each backend; a circuit list is expanded here.
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
        circuit = self._coerce_circuit(circuit)
        observable = self._coerce_operator(observable)

        key = None
        if self._result_cache is not None:
            key = self._make_result_key(
                "expectation_value_derivatives", circuit, observable, derivative, **parameters
            )
            if key in self._result_cache:
                self._logger.debug("Result cache hit for expectation_value_derivatives")
                return self._result_cache[key]

        # Every backend differentiates a list of observables natively (in one
        # pass where the framework allows it), but only one circuit at a time:
        # each circuit needs its own state.  A circuit list is therefore
        # expanded here and the per-circuit results are stacked with a
        # leading axis, so the list convention holds on every backend.
        if isinstance(circuit, (list, tuple)):
            per_circuit = [
                self._expectation_value_derivatives(one, observable, *derivative, **parameters)
                for one in circuit
            ]
            if isinstance(per_circuit[0], dict):
                # Several derivative names: stack the per-circuit dicts per name.
                result = {
                    name: np.asarray([entry[name] for entry in per_circuit])
                    for name in per_circuit[0]
                }
            else:
                result = np.asarray(per_circuit)
        else:
            result = self._expectation_value_derivatives(
                circuit, observable, *derivative, **parameters
            )

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
        circuit = self._coerce_circuit(circuit)
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
        self._reject_stochastic_circuit(circuit)
        parameters = self._normalize_parameter_values(**parameters)
        circuit = self._coerce_circuit(circuit)
        if self._result_cache is not None:
            key = self._make_result_key("statevector", circuit, **parameters)
            if key in self._result_cache:
                self._logger.debug("Result cache hit for statevector")
                return self._result_cache[key]
            result = self._statevector(circuit, **parameters)
            self._result_cache[key] = result
            return result
        return self._statevector(circuit, **parameters)

    @staticmethod
    def _reject_stochastic_circuit(
        circuit: QuantumCircuitBase | List[QuantumCircuitBase],
    ) -> None:
        """Refuse to return a statevector for a circuit that collapses its state.

        A mid-circuit measurement or a reset turns the final state into one
        sample from a mixture, so simulators that accept such circuits return a
        different vector on each call.  Reporting that as *the* statevector is
        wrong, and caching it makes the wrongness stick, so this is an error
        rather than a warning.

        Args:
            circuit: The circuit, or circuits, about to be simulated.

        Raises:
            NotImplementedError: If any circuit measures or resets a qubit.
        """
        for one in circuit if isinstance(circuit, list) else [circuit]:
            ir = getattr(one, "ir", None)
            if ir is None:
                continue
            for opcode, _, _ in ir.iter_ops():
                described = _STOCHASTIC_OPCODES.get(OpCode(opcode))
                if described is not None:
                    raise NotImplementedError(
                        f"statevector is not defined for a circuit containing a "
                        f"{described}: the state collapses randomly, so the result "
                        f"would be one sample from a mixture rather than a state. "
                        f"Use sample() or expectation_value() instead."
                    )

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
    ) -> Dict[int, float] | List[Dict[int, float]]:
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
                keyword arguments. Passing more than one parameter set (the
                same batching convention as :meth:`expectation_value`) is
                supported on backends whose :meth:`statevector`/:meth:`sample`
                support it.

        Returns:
            Dict[int, float] | List[Dict[int, float]]: Mapping of basis-state
            index (big-endian ordering, qubit 0 = most significant bit) to
            probability - one mapping for a single parameter set, or a list
            of mappings, one per set, for a batch of parameter sets.

        Raises:
            ValueError: If a list of circuits is passed.
        """
        if isinstance(circuit, list):
            raise ValueError("probabilities supports a single circuit only")
        if self._shots is None:
            state_vector = np.asarray(self.statevector(circuit, **parameters))
            if state_vector.ndim > 1:
                return [self._probabilities_from_statevector(sv, cutoff) for sv in state_vector]
            return self._probabilities_from_statevector(state_vector, cutoff)
        counts = self.sample(circuit, **parameters)
        # Some backends return one counts dict per parameter set even for a
        # single set (a length-1 list); unwrap that case so a single
        # parameter set always yields a single dict here too, regardless of
        # backend. More than one entry is a genuine batch.
        if isinstance(counts, list):
            if len(counts) == 1:
                counts = counts[0]
            else:
                return [self._probabilities_from_counts(c, cutoff) for c in counts]
        return self._probabilities_from_counts(counts, cutoff)

    @staticmethod
    def _probabilities_from_statevector(
        state_vector: np.ndarray, cutoff: float
    ) -> Dict[int, float]:
        """Convert one statevector to a probability mapping, pruned at ``cutoff``."""
        probability_values = np.abs(state_vector) ** 2
        return {
            index: float(value) for index, value in enumerate(probability_values) if value > cutoff
        }

    @staticmethod
    def _probabilities_from_counts(counts: Dict[str, int], cutoff: float) -> Dict[int, float]:
        """Convert one bitstring-count mapping to a probability mapping, pruned at ``cutoff``.

        Prunes with the same threshold as :meth:`_probabilities_from_statevector`,
        so that switching ``shots`` on or off does not change which entries
        are reported.
        """
        total = sum(counts.values())
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
    def transpile_operator(
        self, operator: QuantumOperatorBase, **options: Any
    ) -> QuantumOperatorBase: ...

    @overload
    def transpile_operator(
        self, operator: List[QuantumOperatorBase], **options: Any
    ) -> List[QuantumOperatorBase]: ...

    def transpile_operator(
        self,
        operator: QuantumOperatorBase | List[QuantumOperatorBase],
        **options: Any,
    ) -> QuantumOperatorBase | List[QuantumOperatorBase]:
        """
        Transpile the operator for execution on this executor's backend.

        Subclasses may override :meth:`_transpile_operator` to apply
        backend-specific conversions (e.g., to wrapper types).  When a list
        of operators is provided, each operator is transpiled and cached individually.

        Args:
            operator (QuantumOperatorBase | List[QuantumOperatorBase]): The
                quantum operator or a list of operators to transpile.
            ``**options``: Backend-specific transpilation options, forwarded to
                :meth:`_transpile_operator`.  The Pauli-propagation backend uses
                this for ``symmetry_strategy``.

        Returns:
            QuantumOperatorBase | List[QuantumOperatorBase]: The transpiled
                operator(s).
        """
        self._logger.info("Transpiling operator")
        if isinstance(operator, list):
            return [self._transpile_operator_cached(item, **options) for item in operator]
        return self._transpile_operator_cached(operator, **options)

    def _transpile_operator_cached(
        self, operator: QuantumOperatorBase, **options: Any
    ) -> QuantumOperatorBase:
        """Transpile a single observable, consulting the result cache if enabled."""
        if self._result_cache is not None:
            key = self._make_result_key("transpile_operator", operator, **options)
            if key in self._result_cache:
                self._logger.debug("Result cache hit for transpile_operator")
                return self._result_cache[key]
            result = self._transpile_operator(operator, **options)
            self._result_cache[key] = result
            return result
        return self._transpile_operator(operator, **options)

    @abstractmethod
    def _transpile_operator(
        self, operator: QuantumOperatorBase, **options: Any
    ) -> QuantumOperatorBase:
        """Abstract implementation of operator transpilation.

        Subclasses override this to convert a generic operator into their
        backend-native type.

        Args:
            operator (QuantumOperatorBase): The operator to transpile.
            ``**options``: Backend-specific options.  A backend that accepts
                none should still declare ``**options`` and ignore them.

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
