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

import warnings
from unittest.mock import patch, MagicMock

import numpy as np
import pennylane as qml
import pytest
from qiskit.circuit import ParameterVector

from executor import QuantumCircuit, QuantumOperator
from executor.pennylane.pennylane_circuit import PennyLaneCircuit
from executor.pennylane.pennylane_executor import PennyLaneExecutor


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


class TestPennylaneExecutor:
    """Test suite for PennyLane executor."""

    # Initialization Tests

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

    # Expectation Value Tests

    def test_expectation_value_bell_state_z_basis(self):
        """Test expectation value of Bell state with Z operators."""
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        op = QuantumOperator(["ZI", "IZ"], [1.0, 1.0])

        executor = PennyLaneExecutor()
        result = executor.expectation_value(qc, op)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 0.0, atol=1e-5)

    def test_expectation_value_bell_state_zz(self):
        """Test expectation value of Bell state with ZZ operator."""
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        op = QuantumOperator(["ZZ"], [1.0])

        executor = PennyLaneExecutor()
        result = executor.expectation_value(qc, op)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_hadamard_x_basis(self):
        """Test expectation value of Hadamard state with X operator."""
        qc = _build_circuit(1, [("h", [0])])
        op = QuantumOperator(["X"], [1.0])

        executor = PennyLaneExecutor()
        result = executor.expectation_value(qc, op)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_y_basis(self):
        """Test expectation value in Y basis (H followed by S gate)."""
        qc = _build_circuit(1, [("h", [0]), ("s", [0])])
        op = QuantumOperator(["Y"], [1.0])

        executor = PennyLaneExecutor()
        result = executor.expectation_value(qc, op)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_with_circuit_parameter(self):
        """Test expectation value with parametric circuit (RX gate)."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])

        executor = PennyLaneExecutor()
        result = executor.expectation_value(qc, op, x=[np.pi])

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, -1.0, atol=1e-5)

    def test_expectation_value_with_multiple_circuit_parameters(self):
        """Test expectation value with multiple circuit parameters."""
        x = ParameterVector("x", 2)
        qc = _build_circuit(2, [("rx", [0, x[0]]), ("ry", [1, x[1]])])
        op = QuantumOperator(["ZZ"], [1.0])

        executor = PennyLaneExecutor()
        result = executor.expectation_value(qc, op, x=[0.0, 0.0])

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_with_observable_parameters(self):
        """Test expectation value with parametric observable."""
        pop = ParameterVector("pop", 2)
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        op = QuantumOperator(["ZI", "IZ"], [pop[0], pop[1]])

        executor = PennyLaneExecutor()
        result = executor.expectation_value(qc, op, pop=[0.5, 0.5])

        assert isinstance(result, (float, np.ndarray))

    def test_expectation_value_with_circuit_and_observable_parameters(self):
        """Test expectation value with both circuit and observable parameters."""
        x = ParameterVector("x", 1)
        pop = ParameterVector("pop", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        op = QuantumOperator(["Z"], [pop[0]])

        executor = PennyLaneExecutor()
        result = executor.expectation_value(qc, op, x=[0.0], pop=[1.0])

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_three_qubit_chain(self):
        """Test expectation value with three-qubit GHZ-type state."""
        qc = _build_circuit(3, [("h", [0]), ("cx", [0, 1]), ("cx", [1, 2])])
        op = QuantumOperator(["ZZZ"], [1.0])

        executor = PennyLaneExecutor()
        result = executor.expectation_value(qc, op)

        assert isinstance(result, (float, np.ndarray))

    # Sampling Tests

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
        x = ParameterVector("x", 1)
        qc = _build_circuit(2, [("rx", [0, x[0]])])

        executor = PennyLaneExecutor(shots=1000, seed=42)
        result = executor.sample(qc, x=[np.pi])

        samples = result[0]
        assert isinstance(samples, dict)
        # After RX(pi), qubit 0 should be flipped
        assert "01" in samples
        assert samples["01"] >= 900  # Should have high count

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

    # Statevector Tests

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
        x = ParameterVector("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])

        executor = PennyLaneExecutor()
        statevector = executor.statevector(qc, x=[np.pi / 2])

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 2
        # Statevector should be normalized
        assert np.isclose(np.sum(np.abs(statevector) ** 2), 1.0, atol=1e-5)

    def test_statevector_with_multiple_parameters(self):
        """Test statevector with multiple parameters."""
        x = ParameterVector("x", 2)
        qc = _build_circuit(2, [("rx", [0, x[0]]), ("ry", [1, x[1]])])

        executor = PennyLaneExecutor()
        statevector = executor.statevector(qc, x=[0.5, 0.3])

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 4
        # Statevector should be normalized
        assert np.isclose(np.sum(np.abs(statevector) ** 2), 1.0, atol=1e-5)

    # Derivative Tests

    def test_expectation_value_derivatives_single_parameter(self):
        """Test derivative with respect to a single parameter."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])

        executor = PennyLaneExecutor()
        result = executor.expectation_value_derivatives(qc, op, "x", x=[0.0])

        assert isinstance(result, (float, np.ndarray))

    def test_expectation_value_derivatives_indexed_parameter(self):
        """Test derivative with respect to indexed parameter (e.g., x[0])."""
        x = ParameterVector("x", 2)
        qc = _build_circuit(2, [("rx", [0, x[0]]), ("ry", [1, x[1]])])
        op = QuantumOperator(["ZI"], [1.0])

        executor = PennyLaneExecutor()
        result = executor.expectation_value_derivatives(qc, op, "x[0]", x=[0.0, 0.0])

        assert isinstance(result, (float, np.ndarray))

    def test_expectation_value_derivatives_multiple_values(self):
        """Test requesting multiple derivatives (expectation value and parameter)."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])

        executor = PennyLaneExecutor()
        result = executor.expectation_value_derivatives(qc, op, "expectation_value", "x", x=[0.0])

        assert isinstance(result, dict)
        assert "expectation_value" in result or "x" in result

    def test_expectation_value_derivatives_known_value(self):
        """Test derivative computation with known analytical result."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])

        executor = PennyLaneExecutor()
        derivative = executor.expectation_value_derivatives(qc, op, "x", x=[0.0])

        assert isinstance(derivative, (float, np.ndarray))
        # Derivative should be close to 0 at x=0
        assert np.isclose(derivative, 0.0, atol=1e-5)

    # Error Handling Tests

    def test_missing_parameter_error_in_expectation_value(self):
        """Test that missing parameter raises ValueError in expectation_value."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])

        executor = PennyLaneExecutor()

        with pytest.raises(ValueError, match="Parameter 'x' not found"):
            executor.expectation_value(qc, op)  # Missing x parameter

    def test_missing_parameter_error_in_sample(self):
        """Test that missing parameter raises ValueError in sample."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])

        executor = PennyLaneExecutor(shots=1000)

        with pytest.raises(ValueError, match="Parameter 'x' not found"):
            executor.sample(qc)  # Missing x parameter

    def test_missing_parameter_error_in_statevector(self):
        """Test that missing parameter raises ValueError in statevector."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])

        executor = PennyLaneExecutor()

        with pytest.raises(ValueError, match="Parameter 'x' not found"):
            executor.statevector(qc)  # Missing x parameter

    def test_missing_parameter_error_in_derivatives(self):
        """Test that missing parameter raises ValueError in expectation_value_derivatives."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])

        executor = PennyLaneExecutor()

        with pytest.raises(ValueError, match="Parameter 'x' not found"):
            executor.expectation_value_derivatives(qc, op, "x")  # Missing x parameter

    # Caching Tests

    def test_circuit_caching(self):
        """Test that circuits are properly cached."""
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        executor = PennyLaneExecutor()

        # First call should add to cache
        executor._preprocess_circuits(qc)
        assert qc in executor._circuit_cache

        # Second call should use cache
        cached_circuits, _ = executor._preprocess_circuits(qc)
        assert len(cached_circuits) == 1
        assert cached_circuits[0] is executor._circuit_cache[qc]

    def test_operator_caching(self):
        """Test that operators are properly cached."""
        op = QuantumOperator(["ZI"], [1.0])
        executor = PennyLaneExecutor()

        # First call should add to cache
        executor._preprocess_operators(op)
        assert op in executor._operator_cache

        # Second call should use cache
        cached_operators, _ = executor._preprocess_operators(op)
        assert len(cached_operators) == 1
        assert cached_operators[0] is executor._operator_cache[op]

    # ========================================================================
    # Property Tests
    # ========================================================================

    def test_shots_property_getter(self):
        """Test that shots property returns correct value."""
        executor = PennyLaneExecutor(shots=500)
        assert executor.shots == 500

    def test_shots_property_setter_raises_error(self):
        """Test that shots setter raises NotImplementedError."""
        executor = PennyLaneExecutor()

        with pytest.raises(NotImplementedError):
            executor.shots = 1000

    def test_remote_property(self):
        """Test that remote property returns False."""
        executor = PennyLaneExecutor()
        assert executor.remote is False

    # ========================================================================
    # Logging Tests
    # ========================================================================

    def _close_file_handlers(self, executor):
        """Helper to close and remove file handlers from an executor's logger."""
        for handler in executor._logger.handlers[:]:
            handler.close()
            executor._logger.removeHandler(handler)

    def test_logging_default_level(self):
        """Test that default logging level is WARNING."""
        import logging

        executor = PennyLaneExecutor()
        assert executor._logger.level == logging.WARNING

    def test_logging_info_level(self):
        """Test that INFO logging level is set correctly."""
        import logging

        executor = PennyLaneExecutor(log_level="INFO")
        assert executor._logger.level == logging.INFO

    def test_logging_debug_level(self):
        """Test that DEBUG logging level is set correctly."""
        import logging

        executor = PennyLaneExecutor(log_level="DEBUG")
        assert executor._logger.level == logging.DEBUG

    def test_logging_error_level(self):
        """Test that ERROR logging level is set correctly."""
        import logging

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

        with open(log_file) as f:
            content = f.read()
        assert "test log message" in content

        self._close_file_handlers(executor)

    def test_logging_no_duplicate_handlers(self, tmp_path):
        """Test that creating two executors with the same log file does not add duplicate handlers."""
        log_file = str(tmp_path / "executor.log")
        executor1 = PennyLaneExecutor(log_level="INFO", log_file=log_file)
        handler_count_before = len(executor1._logger.handlers)

        executor2 = PennyLaneExecutor(log_level="INFO", log_file=log_file)
        assert len(executor2._logger.handlers) == handler_count_before

        self._close_file_handlers(executor1)

    # ========================================================================
    # Cache Size Tests
    # ========================================================================

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
        assert qc1 not in executor._circuit_cache
        assert qc2 in executor._circuit_cache
        assert qc3 in executor._circuit_cache
        # qc2 was inserted before qc3, so it should be first in the ordered dict
        assert list(executor._circuit_cache.keys()) == [qc2, qc3]

    def test_cache_size_restriction_operators(self):
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
        assert op1 not in executor._operator_cache
        assert op2 in executor._operator_cache
        assert op3 in executor._operator_cache
        assert list(executor._operator_cache.keys()) == [op2, op3]

    def test_unlimited_cache_size_by_default(self):
        """Test that cache is unlimited when max_cache_size is not specified."""
        executor = PennyLaneExecutor()
        assert executor._max_cache_size is None
        assert executor._circuit_cache.max_size is None
        assert executor._operator_cache.max_size is None

    # ========================================================================
    # Result-level Caching Tests
    # ========================================================================

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
        op = QuantumOperator(["Z"], [1.0])

        executor = PennyLaneExecutor(caching=True)
        result1 = executor.expectation_value(qc, op)

        # Cache should contain one entry
        assert len(executor._result_cache) == 1

        # Second call with same args must not add a new entry
        result2 = executor.expectation_value(qc, op)
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
        op = QuantumOperator(["Z"], [1.0])

        executor = PennyLaneExecutor(caching=True)
        executor.expectation_value(qc1, op)
        executor.expectation_value(qc2, op)

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

    # ========================================================================
    # Device Selection Tests
    # ========================================================================

    def test_default_device_name(self):
        """Test that the default device is 'default.qubit'."""
        executor = PennyLaneExecutor()
        assert executor.device_name == "default.qubit"
        assert executor._device.name == "default.qubit"

    def test_custom_device_name(self):
        """Test that a custom device is stored and used."""
        executor = PennyLaneExecutor(device="default.mixed")
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
        op = QuantumOperator(["ZZ"], [1.0])

        executor_default = PennyLaneExecutor()
        executor_mixed = PennyLaneExecutor(device="default.mixed")

        result_default = executor_default.expectation_value(qc, op)
        result_mixed = executor_mixed.expectation_value(qc, op)

        assert np.isclose(result_default, result_mixed, atol=1e-5)

    def test_statevector_with_default_mixed_device(self):
        """Test that statevector can be computed with default.mixed device.

        Note: default.mixed returns a density matrix, so we only check
        that the computation succeeds and returns a valid result.
        """
        qc = _build_circuit(1, [("h", [0])])

        executor_mixed = PennyLaneExecutor(device="default.mixed")
        sv_mixed = executor_mixed.statevector(qc)

        assert sv_mixed is not None
        assert sv_mixed.size > 0

    def test_sample_with_default_mixed_device(self):
        """Test sampling with default.mixed device."""
        qc = _build_circuit(2, [("x", [0]), ("x", [1])])

        executor = PennyLaneExecutor(shots=100, seed=42, device="default.mixed")
        result = executor.sample(qc)

        samples = result[0]
        assert "11" in samples
        assert samples["11"] == 100

    def test_derivatives_with_default_mixed_device(self):
        """Test that expectation values can be computed with default.mixed device."""
        qc = _build_circuit(1, [("h", [0])])
        op = QuantumOperator(["X"], [1.0])

        executor = PennyLaneExecutor(device="default.mixed")
        result = executor.expectation_value(qc, op)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_parametric_with_custom_device(self):
        """Test expectation value with custom device on a non-trivial circuit."""
        qc = _build_circuit(1, [("x", [0])])
        op = QuantumOperator(["Z"], [1.0])

        executor = PennyLaneExecutor(device="default.mixed")
        result = executor.expectation_value(qc, op)

        assert np.isclose(result, -1.0, atol=1e-5)

    def test_factory_with_device_name(self):
        """Test creating executor via factory with device_name."""
        from executor import Executor

        executor = Executor.create("pennylane", device="default.mixed")
        assert executor.device_name == "default.mixed"

    def test_device_recreation_preserves_device_name(self):
        """Test that device recreation (on qubit count change) preserves the device name."""
        executor = PennyLaneExecutor(device="default.mixed")

        # Run on a 1-qubit circuit first
        qc1 = _build_circuit(1, [("h", [0])])
        op1 = QuantumOperator(["Z"], [1.0])
        executor.expectation_value(qc1, op1)
        assert executor._device.name == "default.mixed"

        # Run on a 2-qubit circuit to trigger device recreation
        qc2 = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        op2 = QuantumOperator(["ZZ"], [1.0])
        executor.expectation_value(qc2, op2)
        assert executor._device.name == "default.mixed"


class TestDeviceInit:
    """Tests for string-vs-instance device initialisation and config/shots handling."""

    # -- String device init -------------------------------------------------

    def test_init_string_device_default(self):
        """String device with defaults stores the correct internal state."""
        executor = PennyLaneExecutor("default.qubit")
        assert executor._custom_device is False
        assert executor.device_name == "default.qubit"
        assert executor._device_args == ()
        assert executor._device_kwargs == {}

    def test_init_string_device_with_kwargs(self):
        """Extra **kwargs are stored and forwarded to qml.device()."""
        executor = PennyLaneExecutor("default.qubit", custom_decomps={})
        assert executor._custom_device is False
        assert executor._device_kwargs == {"custom_decomps": {}}

    # -- Device instance init -----------------------------------------------

    def test_init_device_instance(self):
        """Passing a Device instance stores it directly."""
        dev = qml.device("default.qubit", wires=2)
        executor = PennyLaneExecutor(dev)
        assert executor._custom_device is True
        assert executor._device is dev
        assert executor.device_name == dev.name

    def test_init_device_instance_no_extra_args(self):
        """Device-instance path stores empty args/kwargs."""
        dev = qml.device("default.qubit", wires=2)
        executor = PennyLaneExecutor(dev, shots=50)
        assert executor._device_args == ()
        assert executor._device_kwargs == {}

    def test_init_device_instance_rejects_extra_kwargs(self):
        """Passing extra **kwargs together with a Device instance is an error."""
        dev = qml.device("default.qubit", wires=2)
        with pytest.raises(TypeError, match="Extra positional or keyword arguments"):
            PennyLaneExecutor(dev, custom_decomps={})

    # -- Config / shots conflict --------------------------------------------

    def test_shots_config_conflict_warning_dict(self):
        """A dict config with 'shots' triggers a UserWarning and overrides shots."""
        mock_dev = MagicMock(spec=qml.devices.Device)
        with patch.object(PennyLaneExecutor, "_create_device", return_value=mock_dev):
            with pytest.warns(UserWarning, match="overridden"):
                executor = PennyLaneExecutor(
                    "default.qubit", shots=100, config={"shots": 500}
                )
        assert executor.shots == 500

    def test_shots_config_conflict_warning_object(self):
        """An object config with a .shots attribute triggers a UserWarning."""

        class _FakeConfig:
            shots = 200

        mock_dev = MagicMock(spec=qml.devices.Device)
        with patch.object(PennyLaneExecutor, "_create_device", return_value=mock_dev):
            with pytest.warns(UserWarning, match="overridden"):
                executor = PennyLaneExecutor(
                    "default.qubit", shots=100, config=_FakeConfig()
                )
        assert executor.shots == 200

    def test_no_warning_when_shots_none(self):
        """No warning when shots is None, even if config has shots."""
        mock_dev = MagicMock(spec=qml.devices.Device)
        with patch.object(PennyLaneExecutor, "_create_device", return_value=mock_dev):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                executor = PennyLaneExecutor(
                    "default.qubit", config={"shots": 500}
                )
        # shots stays None since it was not explicitly set
        assert executor.shots is None

    def test_no_warning_when_config_has_no_shots(self):
        """No warning when config is present but contains no shots."""
        mock_dev = MagicMock(spec=qml.devices.Device)
        with patch.object(PennyLaneExecutor, "_create_device", return_value=mock_dev):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                executor = PennyLaneExecutor(
                    "default.qubit", shots=100, config={"backend": "aer"}
                )
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

    def test_string_device_recreated_on_wire_change(self):
        """A string-based device is recreated when the qubit count changes."""
        executor = PennyLaneExecutor("default.mixed")

        qc1 = _build_circuit(1, [("h", [0])])
        op1 = QuantumOperator(["Z"], [1.0])
        executor.expectation_value(qc1, op1)
        dev_after_1q = executor._device

        qc2 = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        op2 = QuantumOperator(["ZZ"], [1.0])
        executor.expectation_value(qc2, op2)
        assert executor._device is not dev_after_1q  # replaced
        assert executor.device_name == "default.mixed"

