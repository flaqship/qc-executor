import logging
import numpy as np
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Union

from qiskit.circuit import ParameterExpression, Parameter, ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement

from .circuit_base import QuantumCircuitBase
from .operator_base import QuantumOperatorBase

from ..parameters import Parameter, Parameters


class _BoundedCache(OrderedDict):
    """An ordered dictionary that evicts the oldest entry when a size limit is reached.

    Args:
        max_size (int, optional): Maximum number of entries. Defaults to None (unlimited).
    """

    def __init__(self, max_size: Union[int, None] = None):
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key, value):
        if self.max_size is not None and key not in self and len(self) >= self.max_size:
            self.popitem(last=False)  # evict oldest entry
        super().__setitem__(key, value)


class ExecutorBase(ABC):
    """Base class for quantum circuit executors.

    Args:
        shots (int, optional): Number of shots for sampling. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        log_file (str, optional): Path to the log file. Defaults to None.
        log_level (str, optional): Logging level. One of ``"DEBUG"``, ``"INFO"``,
            ``"WARNING"``, ``"ERROR"``. Defaults to ``"WARNING"``.
        caching (bool, optional): Whether to cache computation results. When ``True``,
            the results of :meth:`expectation_value`,
            :meth:`expectation_value_derivatives`, :meth:`sample`,
            :meth:`statevector`, and :meth:`transpile_circuit` are stored in an
            in-memory cache and returned directly on repeated calls with the same
            arguments. Defaults to None (no caching).
        cache_dir (str, optional): Directory for caching. Defaults to "cache".
        max_cache_size (int, optional): Maximum number of entries kept in each
            in-memory cache. ``None`` means unlimited. Defaults to None.
    """

    def __init__(
        self,
        shots: Union[int, None] = None,
        seed: Union[int, None] = None,
        log_file: Union[str, None] = None,
        log_level: str = "WARNING",
        caching: Union[bool, None] = None,
        cache_dir: str = "cache",
        max_cache_size: Union[int, None] = None,
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
            # Avoid registering duplicate file handlers for the same path
            existing_paths = {
                h.baseFilename
                for h in self._logger.handlers
                if isinstance(h, logging.FileHandler)
            }
            if log_file not in existing_paths:
                handler = logging.FileHandler(log_file)
                handler.setLevel(level)
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)

    def _make_cache(self) -> _BoundedCache:
        """Create a new bounded cache with the configured size limit."""
        return _BoundedCache(self._max_cache_size)

    @staticmethod
    def _make_result_key(method_name: str, *args, **kwargs) -> tuple:
        """Build a hashable cache key for a public-interface call.

        Args:
            method_name: Name of the calling method (prevents key collisions
                between different methods).
            *args: Positional arguments passed to the method.
            **kwargs: Keyword arguments passed to the method.

        Returns:
            A hashable tuple that uniquely identifies the call.
        """

        def _to_hashable(v):
            if isinstance(v, np.ndarray):
                return v.tobytes()
            elif isinstance(v, (list, tuple)):
                return tuple(_to_hashable(i) for i in v)
            else:
                try:
                    hash(v)
                    return v
                except TypeError:
                    # Fall back to object identity for unhashable types
                    return id(v)

        return (method_name,) + tuple(_to_hashable(a) for a in args) + tuple(
            sorted((k, _to_hashable(v)) for k, v in kwargs.items())
        )

    @property
    def shots(self) -> Union[int, None]:
        """Return the number of shots."""
        return self._shots

    @shots.setter
    def shots(self, value: Union[int, None]) -> None:
        """Set the number of shots."""
        raise NotImplementedError

    @property
    def remote(self) -> bool:
        """Return True if the execution access a remote backend."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public interface – logging and result caching are centralized here
    # via the Template Method pattern.  Subclasses implement the
    # _-prefixed counterparts.
    # ------------------------------------------------------------------

    def expectation_value(
        self,
        circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]],
        operator: Union[QuantumOperatorBase, List[QuantumOperatorBase]],
        **parameters,
    ) -> Union[float, np.array]:
        """
        Calculate the expectation value of the operator with respect to the circuit.

        Args:
            circuit (Union[QuantumCircuitBase, List[QuantumCircuitBase]]): The quantum circuit or a list of circuits.
            operator (Union[QuantumOperatorBase, List[QuantumOperatorBase]]): The quantum operator or a list of operators.
            parameters: Additional values for the free parameters of the circuit(s) and the operator(s) given as keyword arguments.

        Returns:
            Union[float,np.array]: The expectation value either as a single float or as an numpy array if multiple circuits/operators are provided.
        """
        self._logger.info("Computing expectation value")
        if self._result_cache is not None:
            key = self._make_result_key("expectation_value", circuit, operator, **parameters)
            if key in self._result_cache:
                self._logger.debug("Result cache hit for expectation_value")
                return self._result_cache[key]
            result = self._expectation_value(circuit, operator, **parameters)
            self._result_cache[key] = result
            return result
        return self._expectation_value(circuit, operator, **parameters)

    def expectation_value_derivatives(
        self,
        circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]],
        operator: Union[QuantumOperatorBase, List[QuantumOperatorBase]],
        *derivative,
        **parameters,
    ) -> Union[float, np.array]:
        """
        Calculate the derivatives of the expectation value with respect to the parameters of the circuit.

        Args:
            circuit (Union[QuantumCircuitBase, List[QuantumCircuitBase]]): The quantum circuit or a list of circuits.
            operator (Union[QuantumOperatorBase, List[QuantumOperatorBase]]): The quantum operator or a list of operators.
            derivative: The parameter(s) with respect to which the derivative is calculated.
            parameters: Additional values for the free parameters of the circuit(s) and the operator(s) given as keyword arguments.

        Returns:
            Union[float, np.array]: The derivative of the expectation value either as a single float or as an numpy array.
        """
        self._logger.info("Computing expectation value derivatives")
        if self._result_cache is not None:
            key = self._make_result_key(
                "expectation_value_derivatives", circuit, operator, derivative, **parameters
            )
            if key in self._result_cache:
                self._logger.debug("Result cache hit for expectation_value_derivatives")
                return self._result_cache[key]
            result = self._expectation_value_derivatives(circuit, operator, *derivative, **parameters)
            self._result_cache[key] = result
            return result
        return self._expectation_value_derivatives(circuit, operator, *derivative, **parameters)

    # @abstractmethod
    # def _convert(self, circuit: QuantumCircuitBase, observables: List[QuantumOperatorBase]):
    #    pass

    def sample(
        self, circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]], **parameters
    ) -> Union[dict, List[dict]]:
        """
        Computes samples of the quantumstate of the given circuit.

        Args:
            circuit (Union[QuantumCircuitBase, List[QuantumCircuitBase]]): The quantum circuit or a list of circuits.
            parameters: Additional values for the free parameters of the circuit(s) given as keyword arguments.

        Returns:
            Union[dict, List[dict]]: The sampled results either as a single dictionary or a list of dictionaries if multiple circuits are provided.
        """
        self._logger.info("Sampling circuit (shots=%s)", self._shots)
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

    def statevector(
        self, circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]], **parameters
    ) -> np.ndarray:
        """
        Computes the statevector of the quantum circuit.

        Args:
            circuit (Union[QuantumCircuitBase, List[QuantumCircuitBase]]): The quantum circuit or a list of circuits.
            parameters: Additional values for the free parameters of the circuit(s) given as keyword arguments.

        Returns:
            np.ndarray: The statevector of the circuit(s).
        """
        self._logger.info("Computing statevector")
        if self._result_cache is not None:
            key = self._make_result_key("statevector", circuit, **parameters)
            if key in self._result_cache:
                self._logger.debug("Result cache hit for statevector")
                return self._result_cache[key]
            result = self._statevector(circuit, **parameters)
            self._result_cache[key] = result
            return result
        return self._statevector(circuit, **parameters)

    def transpile_circuit(
        self, circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]]
    ) -> Union[QuantumCircuitBase, List[QuantumCircuitBase]]:
        """
        Transpile the circuit for execution on this executor's backend.

        Subclasses may override :meth:`_transpile_circuit` to apply
        backend-specific optimisations (e.g. gate decomposition, qubit
        routing).  The default implementation returns the circuit unchanged.

        Args:
            circuit (Union[QuantumCircuitBase, List[QuantumCircuitBase]]): The
                quantum circuit or a list of circuits to transpile.

        Returns:
            Union[QuantumCircuitBase, List[QuantumCircuitBase]]: The transpiled
                circuit(s).
        """
        self._logger.info("Transpiling circuit")
        if self._result_cache is not None:
            key = self._make_result_key("transpile_circuit", circuit)
            if key in self._result_cache:
                self._logger.debug("Result cache hit for transpile_circuit")
                return self._result_cache[key]
            result = self._transpile_circuit(circuit)
            self._result_cache[key] = result
            return result
        return self._transpile_circuit(circuit)

    # ------------------------------------------------------------------
    # Abstract implementation hooks – subclasses override these.
    # ------------------------------------------------------------------

    @abstractmethod
    def _expectation_value(
        self,
        circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]],
        operator: Union[QuantumOperatorBase, List[QuantumOperatorBase]],
        **parameters,
    ) -> Union[float, np.array]:
        raise NotImplementedError

    @abstractmethod
    def _expectation_value_derivatives(
        self,
        circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]],
        operator: Union[QuantumOperatorBase, List[QuantumOperatorBase]],
        *derivative,
        **parameters,
    ) -> Union[float, np.array]:
        raise NotImplementedError

    @abstractmethod
    def _sample(
        self, circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]], **parameters
    ) -> Union[dict, List[dict]]:
        raise NotImplementedError

    @abstractmethod
    def _statevector(
        self, circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]], **parameters
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _transpile_circuit(
        self, circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]]
    ) -> Union[QuantumCircuitBase, List[QuantumCircuitBase]]:
        raise NotImplementedError
