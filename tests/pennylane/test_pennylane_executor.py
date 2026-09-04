"""
Test suite for PennyLane executor.

This module tests the PennyLaneExecutor class which executes quantum circuits
using PennyLane backend, including:
- Expectation value computation
- Sampling
- Statevector computation
- Derivative computation
- Caching
- Error handling
"""

import logging
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pennylane as qml
import pytest

from qc_executor import Executor, QuantumCircuit, QuantumOperator
from qc_executor.base.circuit_base import QuantumCircuitBase
from qc_executor.base.executor_base import ExecutorBase
from qc_executor.base.operator_base import QuantumOperatorBase
from qc_executor.parameters import Parameters
from qc_executor.pennylane.pennylane_circuit import PennyLaneCircuit
from qc_executor.pennylane.pennylane_executor import PennyLaneExecutor


def _build_circuit(num_qubits, operations):
    """
    Helper function to build a quantum circuit from a list of operations.

    Args:
        num_qubits (int): Number of qubits in the circuit
        operations (list): List of tuples (gate_name, gate_args)
                          Example: [("h", [0]), ("cx", [0, 1])]

    Returns:
        QuantumCircuit: The constructed quantum circuit
    """
    qc = QuantumCircuit(num_qubits)
    for gate_name, gate_args in operations:
        getattr(qc, gate_name)(*gate_args)
    return qc


class TestPennylaneExecutorInitialization:
    """Test suite for PennyLane executor initialization."""

    def test_get_accepted_backend_aliases(self):
        aliases = PennyLaneExecutor.get_accepted_backend_aliases()
        assert "default.qubit" in aliases
        assert isinstance(aliases, list)

    def test_initialization_default(self):
        """Test executor initialization with default parameters."""
        executor = PennyLaneExecutor()
        assert executor is not None
        assert executor.shots is None

    def test_initialization_with_shots(self):
        """Test executor initialization with shots parameter."""
        executor = PennyLaneExecutor(shots=1000)
        assert executor.shots == 1000

    def test_initialization_with_seed(self):
        """Test executor initialization with seed parameter."""
        executor = PennyLaneExecutor(seed=42)
        assert executor is not None
        assert executor.shots is None

    def test_initialization_with_all_params(self):
        """Test executor initialization with all parameters."""
        executor = PennyLaneExecutor(shots=500, seed=123, log_file="test.log")
        assert executor.shots == 500


class TestPennylaneExpectationValue:
    """Test suite for PennyLane executor expectation values."""

    def test_expectation_value_bell_state_z_basis(self):
        """Test expectation value of Bell state with Z observables."""
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        operator = QuantumOperator(["ZI", "IZ"], [1.0, 1.0])

        executor = PennyLaneExecutor()
        result = executor.expectation_value(qc, operator)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 0.0, atol=1e-5)

    def test_expectation_value_bell_state_zz(self):
        """Test expectation value of Bell state with ZZ observable."""
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        operator = QuantumOperator(["ZZ"], [1.0])

        executor = PennyLaneExecutor()
        result = executor.expectation_value(qc, operator)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_hadamard_x_basis(self):
        """Test expectation value of Hadamard state with X observable."""
        qc = _build_circuit(1, [("h", [0])])
        operator = QuantumOperator(["X"], [1.0])

        executor = PennyLaneExecutor()
        result = executor.expectation_value(qc, operator)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_y_basis(self):
        """Test expectation value in Y basis (H followed by S gate)."""
        qc = _build_circuit(1, [("h", [0]), ("s", [0])])
        operator = QuantumOperator(["Y"], [1.0])

        executor = PennyLaneExecutor()
        result = executor.expectation_value(qc, operator)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_with_circuit_parameter(self):
        """Test expectation value with parametric circuit (RX gate)."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        operator = QuantumOperator(["Z"], [1.0])

        executor = PennyLaneExecutor()
        result = executor.expectation_value(qc, operator, x=[np.pi])

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, -1.0, atol=1e-5)

    def test_expectation_value_with_multiple_circuit_parameters(self):
        """Test expectation value with multiple circuit parameters."""
        x = Parameters("x", 2)
        qc = _build_circuit(2, [("rx", [0, x[0]]), ("ry", [1, x[1]])])
        operator = QuantumOperator(["ZZ"], [1.0])

        executor = PennyLaneExecutor()
        result = executor.expectation_value(qc, operator, x=[0.0, 0.0])

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_with_observable_parameters(self):
        """Test expectation value with parametric observable."""
        p_obs = Parameters("p_obs", 2)
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        operator = QuantumOperator(["ZI", "IZ"], [p_obs[0], p_obs[1]])

        executor = PennyLaneExecutor()
        result = executor.expectation_value(qc, operator, p_obs=[0.5, 0.5])

        assert isinstance(result, (float, np.ndarray))

    def test_expectation_value_with_circuit_and_observable_parameters(self):
        """Test expectation value with both circuit and observable parameters."""
        x = Parameters("x", 1)
        p_obs = Parameters("p_obs", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        operator = QuantumOperator(["Z"], [p_obs[0]])

        executor = PennyLaneExecutor()
        result = executor.expectation_value(qc, operator, x=[0.0], p_obs=[1.0])

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_three_qubit_chain(self):
        """Test expectation value with three-qubit GHZ-type state."""
        qc = _build_circuit(3, [("h", [0]), ("cx", [0, 1]), ("cx", [1, 2])])
        operator = QuantumOperator(["ZZZ"], [1.0])

        executor = PennyLaneExecutor()
        result = executor.expectation_value(qc, operator)

        assert isinstance(result, (float, np.ndarray))


class TestPennylaneSampling:
    """Test suite for PennyLane executor sampling."""

    def test_sample_bell_state(self):
        """Test sampling from Bell state (should get 00 and 11)."""
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])

        executor = PennyLaneExecutor(shots=1000, seed=42)
        result = executor.sample(qc)

        assert isinstance(result, list)
        assert len(result) == 1
        samples = result[0]
        assert isinstance(samples, dict)
        # Bell state should have 00 or 11 outcomes
        assert any(bit in samples for bit in ["00", "11"])

    def test_sample_x_gate(self):
        """Test sampling after X gate (should get all 1s)."""
        qc = _build_circuit(2, [("x", [0]), ("x", [1])])

        executor = PennyLaneExecutor(shots=100, seed=42)
        result = executor.sample(qc)

        samples = result[0]
        assert "11" in samples
        assert samples["11"] == 100

    def test_sample_with_parameter(self):
        """Test sampling with parametric circuit."""
        x = Parameters("x", 1)
        qc = _build_circuit(2, [("rx", [0, x[0]])])

        executor = PennyLaneExecutor(shots=1000, seed=42)
        result = executor.sample(qc, x=[np.pi])

        samples = result[0]
        assert isinstance(samples, dict)
        # After RX(pi), qubit 0 should be flipped; the public bitstring
        # convention puts qubit 0 leftmost.
        assert "10" in samples
        assert samples["10"] >= 900  # Should have high count

    def test_sample_hadamard(self):
        """Test sampling from Hadamard state."""
        # Use 2 qubits to avoid scalar sample issues
        qc = _build_circuit(2, [("h", [0])])

        executor = PennyLaneExecutor(shots=1000, seed=42)
        result = executor.sample(qc)

        samples = result[0]
        assert isinstance(samples, dict)
        assert len(samples) > 0
        # Total counts should equal shots
        total_counts = sum(samples.values())
        assert total_counts == 1000


class TestPennylaneStatevector:
    """Test suite for PennyLane executor statevector."""

    def test_statevector_empty_circuit(self):
        """Test statevector of empty circuit (should be |00...0>)."""
        qc = _build_circuit(2, [])

        executor = PennyLaneExecutor()
        statevector = executor.statevector(qc)

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 4
        # Should be |00> state: [1, 0, 0, 0]
        assert np.isclose(abs(statevector[0]), 1.0, atol=1e-5)
        assert np.allclose(abs(statevector[1:]), 0.0, atol=1e-5)

    def test_statevector_x_gate(self):
        """Test statevector after X gate (should be |1>)."""
        qc = _build_circuit(1, [("x", [0])])

        executor = PennyLaneExecutor()
        statevector = executor.statevector(qc)

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 2
        # Should be |1> state: [0, 1]
        assert np.isclose(abs(statevector[1]), 1.0, atol=1e-5)

    def test_statevector_hadamard(self):
        """Test statevector of Hadamard state (should be equal superposition)."""
        qc = _build_circuit(1, [("h", [0])])

        executor = PennyLaneExecutor()
        statevector = executor.statevector(qc)

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 2
        # Should be (|0> + |1>)/sqrt(2)
        assert np.isclose(abs(statevector[0]), 1 / np.sqrt(2), atol=1e-5)
        assert np.isclose(abs(statevector[1]), 1 / np.sqrt(2), atol=1e-5)

    def test_statevector_bell_state(self):
        """Test statevector of Bell state."""
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])

        executor = PennyLaneExecutor()
        statevector = executor.statevector(qc)

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 4
        # Bell state: (|00> + |11>)/sqrt(2)
        assert np.isclose(abs(statevector[0]), 1 / np.sqrt(2), atol=1e-5)
        assert np.isclose(abs(statevector[3]), 1 / np.sqrt(2), atol=1e-5)

    def test_statevector_with_parameter(self):
        """Test statevector with parametric circuit."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])

        executor = PennyLaneExecutor()
        statevector = executor.statevector(qc, x=[np.pi / 2])

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 2
        # Statevector should be normalized
        assert np.isclose(np.sum(np.abs(statevector) ** 2), 1.0, atol=1e-5)

    def test_statevector_with_multiple_parameters(self):
        """Test statevector with multiple parameters."""
        x = Parameters("x", 2)
        qc = _build_circuit(2, [("rx", [0, x[0]]), ("ry", [1, x[1]])])

        executor = PennyLaneExecutor()
        statevector = executor.statevector(qc, x=[0.5, 0.3])

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 4
        # Statevector should be normalized
        assert np.isclose(np.sum(np.abs(statevector) ** 2), 1.0, atol=1e-5)


class TestPennylaneDerivatives:
    """Test suite for PennyLane executor derivatives."""

    def test_expectation_value_derivatives_single_parameter(self):
        """Test derivative with respect to a single parameter."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        operator = QuantumOperator(["Z"], [1.0])

        executor = PennyLaneExecutor()
        result = executor.expectation_value_derivatives(qc, operator, "x", x=[0.0])

        assert isinstance(result, (float, np.ndarray))

    def test_expectation_value_derivatives_indexed_parameter(self):
        """Test derivative with respect to indexed parameter (e.g., x[0])."""
        x = Parameters("x", 2)
        qc = _build_circuit(2, [("rx", [0, x[0]]), ("ry", [1, x[1]])])
        operator = QuantumOperator(["ZI"], [1.0])

        executor = PennyLaneExecutor()
        result = executor.expectation_value_derivatives(qc, operator, "x[0]", x=[0.0, 0.0])

        assert isinstance(result, (float, np.ndarray))

    def test_expectation_value_derivatives_multiple_values(self):
        """Test requesting multiple derivatives (expectation value and parameter)."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        operator = QuantumOperator(["Z"], [1.0])

        executor = PennyLaneExecutor()
        result = executor.expectation_value_derivatives(
            qc, operator, "expectation_value", "x", x=[0.0]
        )

        assert isinstance(result, dict)
        assert "expectation_value" in result or "x" in result

    def test_expectation_value_derivatives_known_value(self):
        """Test derivative computation with known analytical result."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [1.0])

        executor = PennyLaneExecutor()
        derivative = executor.expectation_value_derivatives(qc, operator, "x", x=[0.0])

        assert isinstance(derivative, (float, np.ndarray))
        # Derivative should be close to 0 at x=0
        assert np.isclose(derivative, 0.0, atol=1e-5)


class TestPennylaneErrorHandling:
    """Test suite for PennyLane executor error handling."""

    def test_missing_parameter_error_in_expectation_value_circuit(self):
        """Test that missing parameter raises ValueError in expectation_value."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        operator = QuantumOperator(["Z"], [1.0])

        executor = PennyLaneExecutor()

        with pytest.raises(ValueError, match="Parameter 'x' not found"):
            executor.expectation_value(qc, operator)  # Missing x parameter

    def test_missing_parameter_error_in_expectation_value_observable(self):
        """Test that missing observable parameter raises ValueError."""
        x = Parameters("x", 1)
        y = Parameters("y", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        operator = QuantumOperator(["Z"], [y[0]])

        executor = PennyLaneExecutor()

        with pytest.raises(ValueError, match="Parameter 'y' not found"):
            executor.expectation_value(qc, operator, x=[0.5])  # Missing y parameter

    def test_missing_parameter_error_in_sample(self):
        """Test that missing parameter raises ValueError in sample."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])

        executor = PennyLaneExecutor(shots=1000)

        with pytest.raises(ValueError, match="Parameter 'x' not found"):
            executor.sample(qc)  # Missing x parameter

    def test_missing_parameter_error_in_statevector(self):
        """Test that missing parameter raises ValueError in statevector."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])

        executor = PennyLaneExecutor()

        with pytest.raises(ValueError, match="Parameter 'x' not found"):
            executor.statevector(qc)  # Missing x parameter

    def test_missing_parameter_error_in_derivatives_circuit(self):
        """Test that missing parameter raises ValueError in expectation_value_derivatives."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        operator = QuantumOperator(["Z"], [1.0])

        executor = PennyLaneExecutor()

        with pytest.raises(ValueError, match="Parameter 'x' not found"):
            executor.expectation_value_derivatives(qc, operator, "x")  # Missing x parameter

    def test_missing_parameter_error_in_derivatives_observable(self):
        """Test that missing parameter raises ValueError in expectation_value_derivatives."""
        x = Parameters("x", 1)
        y = Parameters("y", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        operator = QuantumOperator(["Z"], [y[0]])

        executor = PennyLaneExecutor()

        with pytest.raises(ValueError, match="Parameter 'y' not found"):
            executor.expectation_value_derivatives(qc, operator, x=[0.5])  # Missing y parameter

    def test_derivatives_list_inputs_are_expanded_by_the_base(self):
        """List inputs are expanded combinatorially before reaching the plugin."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        operator = QuantumOperator(["Z"], [1.0])

        executor = PennyLaneExecutor()
        single = np.asarray(
            executor.expectation_value_derivatives(qc, operator, "x", x=[0.1]), dtype=float
        )
        per_circuit = np.asarray(
            executor.expectation_value_derivatives([qc, qc], operator, "x", x=[0.1]), dtype=float
        )
        per_observable = np.asarray(
            executor.expectation_value_derivatives(qc, [operator, operator], "x", x=[0.1]),
            dtype=float,
        )

        assert per_circuit.shape[0] == 2
        assert per_observable.shape[0] == 2
        np.testing.assert_allclose(per_circuit[0], single)
        np.testing.assert_allclose(per_observable[1], single)

    def test_device_kwargs_raises(self):
        with pytest.raises(TypeError, match="'device' is not a supported argument"):
            PennyLaneExecutor(device="default.qubit")


class TestPennylaneCaching:
    """Test suite for PennyLane executor caching."""

    def test_circuit_caching(self):
        """Test that circuits are cached under their structural key."""
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        executor = PennyLaneExecutor()

        # First call should add to cache
        executor._preprocess_circuits(qc)
        assert ExecutorBase._structural_cache_key(qc) in executor._circuit_cache

        # Second call should use cache
        cached_circuits, _ = executor._preprocess_circuits(qc)
        assert len(cached_circuits) == 1
        assert (
            cached_circuits[0] is executor._circuit_cache[ExecutorBase._structural_cache_key(qc)]
        )

        # A structurally identical fresh object hits the same entry
        qc_clone = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        cached_clone, _ = executor._preprocess_circuits(qc_clone)
        assert cached_clone[0] is cached_circuits[0]
        assert len(executor._circuit_cache) == 1

        # An in-place mutation produces a fresh conversion
        qc.x(0)
        mutated, _ = executor._preprocess_circuits(qc)
        assert mutated[0] is not cached_circuits[0]
        assert len(executor._circuit_cache) == 2

    def test_observable_caching(self):
        """Test that operators are cached under their structural key."""
        operator = QuantumOperator(["ZI"], [1.0])
        executor = PennyLaneExecutor()

        # First call should add to cache
        executor._preprocess_operators(operator)
        assert ExecutorBase._structural_cache_key(operator) in executor._operator_cache

        # Second call should use cache
        cached_operators, _ = executor._preprocess_operators(operator)
        assert len(cached_operators) == 1
        assert (
            cached_operators[0]
            is executor._operator_cache[ExecutorBase._structural_cache_key(operator)]
        )

        # A structurally identical fresh object hits the same entry
        operator_clone = QuantumOperator(["ZI"], [1.0])
        cached_clone, _ = executor._preprocess_operators(operator_clone)
        assert cached_clone[0] is cached_operators[0]
        assert len(executor._operator_cache) == 1


class TestPennylaneProperties:
    """Test suite for PennyLane executor properties."""

    def test_shots_property_getter(self):
        """Test that shots property returns correct value."""
        executor = PennyLaneExecutor(shots=500)
        assert executor.shots == 500

    def test_shots_property_setter_updates_shots(self):
        """Test that shots setter actually changes the reported shot count."""
        executor = PennyLaneExecutor()
        assert executor.shots is None

        executor.shots = 1000
        assert executor.shots == 1000

        executor.shots = None
        assert executor.shots is None

    def test_remote_property(self):
        """Test that remote property returns False."""
        executor = PennyLaneExecutor()
        assert executor.remote is False


class TestPennylaneLogging:
    """Test suite for PennyLane executor logging."""

    def _close_file_handlers(self, executor):
        """Helper to close and remove file handlers from an executor's logger."""
        for handler in executor._logger.handlers[:]:
            handler.close()
            executor._logger.removeHandler(handler)

    def test_logging_default_level(self):
        """Test that default logging level is WARNING."""
        executor = PennyLaneExecutor()
        assert executor._logger.level == logging.WARNING

    def test_logging_info_level(self):
        """Test that INFO logging level is set correctly."""
        executor = PennyLaneExecutor(log_level="INFO")
        assert executor._logger.level == logging.INFO

    def test_logging_debug_level(self):
        """Test that DEBUG logging level is set correctly."""
        executor = PennyLaneExecutor(log_level="DEBUG")
        assert executor._logger.level == logging.DEBUG

    def test_logging_error_level(self):
        """Test that ERROR logging level is set correctly."""
        executor = PennyLaneExecutor(log_level="ERROR")
        assert executor._logger.level == logging.ERROR

    def test_logging_invalid_level_raises(self):
        """Test that an invalid log_level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log_level"):
            PennyLaneExecutor(log_level="VERBOSE")

    def test_logging_to_file(self, tmp_path):
        """Test that log messages are written to the specified log file."""

        log_file = str(tmp_path / "executor.log")
        executor = PennyLaneExecutor(log_level="INFO", log_file=log_file)
        executor._logger.info("test log message")

        with open(log_file, encoding="utf-8") as f:
            content = f.read()
        assert "test log message" in content

        self._close_file_handlers(executor)

    def test_logging_no_duplicate_handlers(self, tmp_path):
        """Two executors sharing a log file must not add duplicate handlers."""
        log_file = str(tmp_path / "executor.log")
        executor1 = PennyLaneExecutor(log_level="INFO", log_file=log_file)
        handler_count_before = len(executor1._logger.handlers)

        executor2 = PennyLaneExecutor(log_level="INFO", log_file=log_file)
        assert len(executor2._logger.handlers) == handler_count_before

        self._close_file_handlers(executor1)


class TestPennylaneCacheSizeRestriction:
    """Test suite for PennyLane executor cache size restrictions."""

    def test_cache_size_restriction_circuits(self):
        """Test that circuit cache respects max_cache_size with FIFO eviction."""
        executor = PennyLaneExecutor(max_cache_size=2)

        qc1 = _build_circuit(1, [])
        qc2 = _build_circuit(2, [])
        qc3 = _build_circuit(3, [])

        executor._preprocess_circuits(qc1)
        executor._preprocess_circuits(qc2)
        assert len(executor._circuit_cache) == 2

        # Adding a third circuit should evict the oldest (qc1)
        executor._preprocess_circuits(qc3)
        assert len(executor._circuit_cache) == 2
        assert ExecutorBase._structural_cache_key(qc1) not in executor._circuit_cache
        assert ExecutorBase._structural_cache_key(qc2) in executor._circuit_cache
        assert ExecutorBase._structural_cache_key(qc3) in executor._circuit_cache
        # qc2 was inserted before qc3, so it should be first in the ordered dict
        assert list(executor._circuit_cache.keys()) == [
            ExecutorBase._structural_cache_key(qc2),
            ExecutorBase._structural_cache_key(qc3),
        ]

    def test_cache_size_restriction_observables(self):
        """Test that operator cache respects max_cache_size with FIFO eviction."""
        executor = PennyLaneExecutor(max_cache_size=2)

        op1 = QuantumOperator(["Z"], [1.0])
        op2 = QuantumOperator(["X"], [1.0])
        op3 = QuantumOperator(["Y"], [1.0])

        executor._preprocess_operators(op1)
        executor._preprocess_operators(op2)
        assert len(executor._operator_cache) == 2

        # Adding a third operator should evict the oldest (op1)
        executor._preprocess_operators(op3)
        assert len(executor._operator_cache) == 2
        assert ExecutorBase._structural_cache_key(op1) not in executor._operator_cache
        assert ExecutorBase._structural_cache_key(op2) in executor._operator_cache
        assert ExecutorBase._structural_cache_key(op3) in executor._operator_cache
        assert list(executor._operator_cache.keys()) == [
            ExecutorBase._structural_cache_key(op2),
            ExecutorBase._structural_cache_key(op3),
        ]

    def test_default_cache_size_is_bounded(self):
        """Test that caches use the default bound when max_cache_size is not specified."""
        executor = PennyLaneExecutor()
        assert executor._max_cache_size == 4096
        assert executor._circuit_cache.max_size == 4096
        assert executor._operator_cache.max_size == 4096


class TestPennylaneResultCaching:
    """Test suite for PennyLane executor result caching."""

    def test_result_cache_disabled_by_default(self):
        """Test that result cache is None when caching is not enabled."""
        executor = PennyLaneExecutor()
        assert executor._result_cache is None

    def test_result_cache_disabled_when_caching_false(self):
        """Test that result cache is None when caching=False."""
        executor = PennyLaneExecutor(caching=False)
        assert executor._result_cache is None

    def test_result_cache_enabled_when_caching_true(self):
        """Test that result cache is created when caching=True."""
        executor = PennyLaneExecutor(caching=True)
        assert executor._result_cache is not None

    def test_result_cache_respects_max_cache_size(self):
        """Test that result cache respects max_cache_size."""
        executor = PennyLaneExecutor(caching=True, max_cache_size=5)
        assert executor._result_cache.max_size == 5

    def test_expectation_value_result_caching(self):
        """Test that repeated expectation_value calls use the result cache."""
        qc = _build_circuit(1, [("h", [0])])
        operator = QuantumOperator(["Z"], [1.0])

        executor = PennyLaneExecutor(caching=True)
        result1 = executor.expectation_value(qc, operator)

        # Cache should contain one entry
        assert len(executor._result_cache) == 1

        # Second call with same args must not add a new entry
        result2 = executor.expectation_value(qc, operator)
        assert len(executor._result_cache) == 1

        # Both calls must return the same value
        assert np.isclose(result1, result2, atol=1e-10)

    def test_statevector_result_caching(self):
        """Test that repeated statevector calls use the result cache."""
        qc = _build_circuit(1, [("h", [0])])

        executor = PennyLaneExecutor(caching=True)
        sv1 = executor.statevector(qc)
        assert len(executor._result_cache) == 1

        sv2 = executor.statevector(qc)
        assert len(executor._result_cache) == 1

        np.testing.assert_array_equal(sv1, sv2)

    def test_different_args_produce_distinct_cache_entries(self):
        """Test that calls with different arguments create separate cache entries."""
        qc1 = _build_circuit(1, [("h", [0])])
        qc2 = _build_circuit(1, [("x", [0])])
        operator = QuantumOperator(["Z"], [1.0])

        executor = PennyLaneExecutor(caching=True)
        executor.expectation_value(qc1, operator)
        executor.expectation_value(qc2, operator)

        # Two distinct circuits → two distinct cache entries
        assert len(executor._result_cache) == 2

    def test_transpile_circuit_returns_pennylane_circuit(self):
        """Test that transpile_circuit converts a QuantumCircuit to a PennyLaneCircuit."""
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        executor = PennyLaneExecutor()
        result = executor.transpile_circuit(qc)
        assert isinstance(result, PennyLaneCircuit)

    def test_transpile_circuit_result_caching(self):
        """Test that repeated transpile_circuit calls use the result cache."""
        qc = _build_circuit(1, [("h", [0])])
        executor = PennyLaneExecutor(caching=True)

        result1 = executor.transpile_circuit(qc)
        assert len(executor._result_cache) == 1

        result2 = executor.transpile_circuit(qc)
        assert len(executor._result_cache) == 1
        assert result1 is result2

    def test_transpile_circuit_no_cache_when_caching_disabled(self):
        """Test that transpile_circuit doesn't cache when caching is disabled."""
        qc = _build_circuit(1, [("h", [0])])
        executor = PennyLaneExecutor()  # caching=None by default
        result = executor.transpile_circuit(qc)
        assert isinstance(result, PennyLaneCircuit)
        assert executor._result_cache is None


class TestPennylaneDeviceConfiguration:
    """Test suite for PennyLane executor device configuration."""

    def test_default_device_name(self):
        """Test that the default device is 'default.qubit'."""
        executor = PennyLaneExecutor()
        assert executor.device_name == "default.qubit"
        assert executor._device.name == "default.qubit"

    def test_custom_device_name(self):
        """Test that a custom device is stored and used."""
        executor = PennyLaneExecutor(backend="default.mixed")
        assert executor.device_name == "default.mixed"
        assert executor._device.name == "default.mixed"

    def test_device_name_property_readonly(self):
        """Test that device_name is a read-only property."""
        executor = PennyLaneExecutor()
        with pytest.raises(AttributeError):
            executor.device_name = "lightning.qubit"

    def test_expectation_value_with_default_mixed_device(self):
        """Test expectation value computation with default.mixed device."""
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        operator = QuantumOperator(["ZZ"], [1.0])

        executor_default = PennyLaneExecutor()
        executor_mixed = PennyLaneExecutor(backend="default.mixed")

        result_default = executor_default.expectation_value(qc, operator)
        result_mixed = executor_mixed.expectation_value(qc, operator)

        assert np.isclose(result_default, result_mixed, atol=1e-5)

    def test_statevector_with_default_mixed_device(self):
        """Test that statevector can be computed with default.mixed device.

        Note: default.mixed returns a density matrix, so we only check
        that the computation succeeds and returns a valid result.
        """
        qc = _build_circuit(1, [("h", [0])])

        executor_mixed = PennyLaneExecutor(backend="default.mixed")
        sv_mixed = executor_mixed.statevector(qc)

        assert sv_mixed is not None
        assert sv_mixed.size > 0

    def test_sample_with_default_mixed_device(self):
        """Test sampling with default.mixed device."""
        qc = _build_circuit(2, [("x", [0]), ("x", [1])])

        executor = PennyLaneExecutor(shots=100, seed=42, backend="default.mixed")
        result = executor.sample(qc)

        samples = result[0]
        assert "11" in samples
        assert samples["11"] == 100

    def test_derivatives_with_default_mixed_device(self):
        """Test that expectation values can be computed with default.mixed device."""
        qc = _build_circuit(1, [("h", [0])])
        operator = QuantumOperator(["X"], [1.0])

        executor = PennyLaneExecutor(backend="default.mixed")
        result = executor.expectation_value(qc, operator)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_with_default_mixed_nontrivial_circuit(self):
        """Test expectation value with default.mixed on a non-trivial circuit."""
        qc = _build_circuit(1, [("x", [0])])
        operator = QuantumOperator(["Z"], [1.0])

        executor = PennyLaneExecutor(backend="default.mixed")
        result = executor.expectation_value(qc, operator)

        assert np.isclose(result, -1.0, atol=1e-5)

    def test_factory_with_device_name(self):
        """Test creating executor via factory with device_name."""
        executor = Executor.create("pennylane", backend="default.mixed")
        assert executor.device_name == "default.mixed"


class TestDeviceInit:
    """Tests for string-vs-instance device initialisation and config/shots handling."""

    # -- String device init -------------------------------------------------

    def test_init_string_device_default(self):
        """String device with defaults stores the correct internal state."""
        executor = PennyLaneExecutor("default.qubit")
        assert executor._custom_device is False
        assert executor.device_name == "default.qubit"
        assert not executor._device_args
        assert not executor._device_kwargs

    def test_init_string_device_with_kwargs(self):
        """Extra **kwargs are stored and forwarded to qml.device()."""
        # "wires" is accepted by every PennyLane version; device-specific
        # kwargs come and go across releases and would make this version-fragile.
        executor = PennyLaneExecutor("default.qubit", wires=2)
        assert executor._custom_device is False
        assert executor._device_kwargs == {"wires": 2}

    # -- Device instance init -----------------------------------------------

    def test_init_device_instance(self):
        """Passing a Device instance stores it directly."""
        dev = qml.device("default.qubit", wires=2)
        executor = PennyLaneExecutor(dev)
        assert executor._custom_device is True
        assert executor._device is dev
        assert executor.device_name == dev.name

    def test_init_device_instance_rejects_shots_or_seed(self):
        """Device-instance path rejects executor-level shots/seed overrides."""
        dev = qml.device("default.qubit", wires=2)
        with pytest.raises(ValueError, match="shots' and 'seed'"):
            PennyLaneExecutor(dev, shots=50)
        with pytest.raises(ValueError, match="shots' and 'seed'"):
            PennyLaneExecutor(dev, seed=42)

    def test_init_device_instance_rejects_extra_kwargs(self):
        """Passing extra **kwargs together with a Device instance is an error."""
        dev = qml.device("default.qubit", wires=2)
        with pytest.raises(TypeError, match="Extra positional or keyword arguments"):
            PennyLaneExecutor(dev, custom_decomps={})

    def test_init_rejects_wires_positional_and_keyword(self):
        """Passing wires both positionally and as keyword is rejected clearly."""
        with pytest.raises(ValueError, match="provided both positionally"):
            PennyLaneExecutor("default.qubit", 2, wires=2)

    # -- Config / shots conflict --------------------------------------------

    def test_shots_config_conflict_warning_dict(self):
        """A dict config with 'shots' triggers a UserWarning and overrides shots."""
        mock_dev = MagicMock(spec=qml.devices.Device)
        with patch.object(PennyLaneExecutor, "_create_device", return_value=mock_dev):
            with pytest.warns(UserWarning, match="overridden"):
                executor = PennyLaneExecutor("default.qubit", shots=100, config={"shots": 500})
        assert executor.shots == 500

    def test_shots_config_conflict_warning_object(self):
        """An object config with a .shots attribute triggers a UserWarning."""

        class _FakeConfig:
            shots = 200

        mock_dev = MagicMock(spec=qml.devices.Device)
        with patch.object(PennyLaneExecutor, "_create_device", return_value=mock_dev):
            with pytest.warns(UserWarning, match="overridden"):
                executor = PennyLaneExecutor("default.qubit", shots=100, config=_FakeConfig())
        assert executor.shots == 200

    def test_no_warning_when_shots_none(self):
        """No warning when shots is None, even if config has shots."""
        mock_dev = MagicMock(spec=qml.devices.Device)
        with patch.object(PennyLaneExecutor, "_create_device", return_value=mock_dev):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                executor = PennyLaneExecutor("default.qubit", config={"shots": 500})
        # shots stays None since it was not explicitly set
        assert executor.shots is None

    def test_no_warning_when_config_has_no_shots(self):
        """No warning when config is present but contains no shots."""
        mock_dev = MagicMock(spec=qml.devices.Device)
        with patch.object(PennyLaneExecutor, "_create_device", return_value=mock_dev):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                executor = PennyLaneExecutor("default.qubit", shots=100, config={"backend": "aer"})
        assert executor.shots == 100

    # -- Device recreation behaviour ----------------------------------------

    def test_custom_device_not_recreated(self):
        """A custom Device instance is never replaced when the qubit count changes."""
        dev = qml.device("default.qubit", wires=4)
        executor = PennyLaneExecutor(dev)

        qc1 = _build_circuit(1, [("h", [0])])
        op1 = QuantumOperator(["Z"], [1.0])
        executor.expectation_value(qc1, op1)
        assert executor._device is dev

        qc2 = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        op2 = QuantumOperator(["ZZ"], [1.0])
        executor.expectation_value(qc2, op2)
        assert executor._device is dev  # still the same object

    def test_string_device_not_recreated_on_wire_change(self):
        """A string-based device is never recreated on qubit-count changes."""
        executor = PennyLaneExecutor("default.mixed")

        qc1 = _build_circuit(1, [("h", [0])])
        op1 = QuantumOperator(["Z"], [1.0])
        executor.expectation_value(qc1, op1)
        dev_after_1q = executor._device

        qc2 = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        op2 = QuantumOperator(["ZZ"], [1.0])
        executor.expectation_value(qc2, op2)
        assert executor._device is dev_after_1q
        assert executor.device_name == "default.mixed"

    def test_custom_device_with_too_few_wires_raises_clear_error(self):
        """A custom device must have enough wires for the executed circuit."""
        dev = qml.device("default.qubit", wires=1)
        executor = PennyLaneExecutor(dev)
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        operator = QuantumOperator(["ZZ"], [1.0])

        with pytest.raises(ValueError, match="has only 1 wires"):
            executor.expectation_value(qc, operator)

    # -- get_accepted_backend_types ------------------------------------------

    def test_get_accepted_backend_types_returns_list(self):
        """get_accepted_backend_types returns a list of types."""
        accepted = PennyLaneExecutor.get_accepted_backend_types()
        assert isinstance(accepted, list)
        assert len(accepted) > 0
        assert all(isinstance(t, type) for t in accepted)

    def test_get_accepted_backend_types_contains_device(self):
        """The PennyLane Device base class must be in accepted types."""
        accepted = PennyLaneExecutor.get_accepted_backend_types()
        assert qml.devices.Device in accepted

    def test_get_accepted_backend_types_matches_device_instance(self):
        """A concrete PennyLane device must match one accepted type."""
        dev = qml.device("default.qubit", wires=1)
        accepted = PennyLaneExecutor.get_accepted_backend_types()
        assert any(isinstance(dev, t) for t in accepted)

    def test_get_accepted_backend_types_rejects_non_device(self):
        """A plain string or unrelated object must not match any accepted type."""
        accepted = PennyLaneExecutor.get_accepted_backend_types()
        assert not any(isinstance("default.qubit", t) for t in accepted)
        assert not any(isinstance(42, t) for t in accepted)


class TestPennylaneExecutorHelpers:
    """Test suite for helper methods in PennyLaneExecutor."""

    def test_preprocess_operators_native_operator_is_passed_through(self):
        """Native PennyLane operators pass through _preprocess_operators unchanged."""
        executor = PennyLaneExecutor()

        native_op = MagicMock(spec=executor._native_operator_class)
        native_op.__class__ = executor._native_operator_class
        result = executor._preprocess_operators(native_op)
        assert result == ([native_op], False)

    def test_transpile_circuit_native_is_returned_directly(self):
        """Native circuit bypasses conversion."""
        executor = PennyLaneExecutor()

        native_circuit = MagicMock(spec=executor._native_circuit_class)
        native_circuit.__class__ = executor._native_circuit_class

        with patch.object(executor._native_circuit_class, "from_quantum_circuit") as mock_from:
            result = executor._transpile_circuit(native_circuit)

        assert result is native_circuit
        mock_from.assert_not_called()

    def test_transpile_circuit_foreign_is_converted(self):
        """Non-native circuit is converted via from_quantum_circuit."""
        executor = PennyLaneExecutor()

        foreign_circuit = MagicMock(spec=QuantumCircuitBase)
        converted = MagicMock(spec=executor._native_circuit_class)

        with patch.object(
            executor._native_circuit_class, "from_quantum_circuit", return_value=converted
        ) as mock_from:
            result = executor._transpile_circuit(foreign_circuit)

        mock_from.assert_called_once_with(foreign_circuit)
        assert result is converted

    def test_transpile_operator_native_is_returned_directly(self):
        """Native operator bypasses conversion."""
        executor = PennyLaneExecutor()

        native_op = MagicMock(spec=executor._native_operator_class)
        native_op.__class__ = executor._native_operator_class

        with patch.object(executor._native_operator_class, "from_quantum_operator") as mock_from:
            result = executor._transpile_operator(native_op)

        assert result is native_op
        mock_from.assert_not_called()

    def test_transpile_operator_foreign_is_converted(self):
        """Non-native operator is converted via from_quantum_operator."""
        executor = PennyLaneExecutor()

        foreign_op = MagicMock(spec=QuantumOperatorBase)
        converted = MagicMock(spec=executor._native_operator_class)

        with patch.object(
            executor._native_operator_class, "from_quantum_operator", return_value=converted
        ) as mock_from:
            result = executor._transpile_operator(foreign_op)

        mock_from.assert_called_once_with(foreign_op)
        assert result is converted
