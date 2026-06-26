import logging

import numpy as np
import pytest

from qc_executor import QuantumCircuit, QuantumOperator
from qc_executor.parameters import Parameters
from qc_executor.qulacs import QulacsCircuit, QulacsExecutor, QulacsOperator


def _build_circuit(num_qubits, operations):
    qc = QuantumCircuit(num_qubits)
    for gate_name, gate_args in operations:
        getattr(qc, gate_name)(*gate_args)
    return qc


class TestQulacsExecutorMetadata:
    def test_get_accepted_backend_types(self):
        assert QulacsExecutor.get_accepted_backend_types() == []

    def test_get_accepted_backend_aliases(self):
        assert QulacsExecutor.get_accepted_backend_aliases() == []

    def test_shots_property_and_remote(self):
        """Test basic public properties of the executor."""
        executor = QulacsExecutor(shots=128)
        assert executor.shots == 128
        assert executor.remote is False

    def test_shots_setter_raises(self):
        """Test that setting shots via property is not implemented."""
        executor = QulacsExecutor()
        with pytest.raises(NotImplementedError):
            executor.shots = 64


class TestQulacsExecutorLoggingAndCache:

    def test_logging_default_level(self):
        """Test that default logging level is WARNING."""
        executor = QulacsExecutor()
        assert executor._logger.level == logging.WARNING

    def test_logging_info_level(self):
        """Test that INFO logging level is set correctly."""
        executor = QulacsExecutor(log_level="INFO")
        assert executor._logger.level == logging.INFO

    def test_logging_debug_level(self):
        """Test that DEBUG logging level is set correctly."""
        executor = QulacsExecutor(log_level="DEBUG")
        assert executor._logger.level == logging.DEBUG

    def test_logging_invalid_level_raises(self):
        """Test that an invalid log_level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log_level"):
            QulacsExecutor(log_level="TRACE")

    def test_logging_to_file(self, tmp_path):
        """Test that log messages are written to the specified log file."""
        log_file = str(tmp_path / "qulacs_executor.log")
        executor = QulacsExecutor(log_level="INFO", log_file=log_file)
        executor._logger.info("qulacs test log message")

        with open(log_file) as f:
            content = f.read()
        assert "qulacs test log message" in content

        for handler in executor._logger.handlers[:]:
            handler.close()
            executor._logger.removeHandler(handler)

    def test_cache_size_restriction_circuits(self):
        """Test that circuit cache respects max_cache_size."""
        executor = QulacsExecutor(max_cache_size=1)
        assert executor._circuit_cache.max_size == 1

    def test_cache_size_restriction_observables(self):
        """Test that operator cache respects max_cache_size."""
        executor = QulacsExecutor(max_cache_size=1)
        assert executor._operator_cache.max_size == 1

    def test_unlimited_cache_size_by_default(self):
        """Test that caches are unlimited when max_cache_size is not specified."""
        executor = QulacsExecutor()
        assert executor._max_cache_size is None
        assert executor._circuit_cache.max_size is None
        assert executor._operator_cache.max_size is None


class TestQulacsExecutorPreprocessingAndTranspile:
    def test_preprocess_circuits_cache_miss_and_hit(self):
        """Test circuit preprocessing cache miss on first call and hit on second."""
        executor = QulacsExecutor()
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])

        converted_first, multiple_first = executor._preprocess_circuits(qc)
        converted_second, multiple_second = executor._preprocess_circuits(qc)

        assert multiple_first is False
        assert multiple_second is False
        assert isinstance(converted_first[0], QulacsCircuit)
        assert converted_first[0] is converted_second[0]

    def test_preprocess_operators_cache_miss_hit_and_native_passthrough(self):
        """Test operator preprocessing for cache behavior and native passthrough."""
        executor = QulacsExecutor()
        op = QuantumOperator(["Z"], [1.0])

        converted_first, multiple_first = executor._preprocess_operators(op)
        converted_second, multiple_second = executor._preprocess_operators(op)
        native_op = converted_first[0]
        converted_native, multiple_native = executor._preprocess_operators(native_op)

        assert multiple_first is False
        assert multiple_second is False
        assert multiple_native is False
        assert isinstance(converted_first[0], QulacsOperator)
        assert converted_first[0] is converted_second[0]
        assert converted_native[0] is native_op

    def test_transpile_circuit_native_and_generic(self):
        """Test circuit transpilation for both native and generic inputs."""
        executor = QulacsExecutor()
        generic = _build_circuit(1, [("h", [0])])
        native = QulacsCircuit(generic)

        transpiled_generic = executor.transpile_circuit(generic)
        transpiled_native = executor.transpile_circuit(native)

        assert isinstance(transpiled_generic, QulacsCircuit)
        assert transpiled_native is native

    def test_transpile_operator_native_and_generic(self):
        """Test operator transpilation for both native and generic inputs."""
        executor = QulacsExecutor()
        generic = QuantumOperator(["Z"], [1.0])
        native = QulacsOperator(generic)

        transpiled_generic = executor._transpile_operator(generic)
        transpiled_native = executor._transpile_operator(native)

        assert isinstance(transpiled_generic, QulacsOperator)
        assert transpiled_native is native


class TestQulacsExecutorExpectationAndStatevector:
    def test_expectation_value_missing_circuit_parameter_raises(self):
        """Test missing circuit parameter error handling in expectation value."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])

        executor = QulacsExecutor()
        with pytest.raises(ValueError, match="Parameter 'x' not found"):
            executor.expectation_value(qc, op)

    def test_expectation_value_missing_observable_parameter_raises(self):
        """Test missing observable parameter error handling in expectation value."""
        p = Parameters("p", 1)
        qc = _build_circuit(1, [])
        op = QuantumOperator(["Z"], [p[0]])

        executor = QulacsExecutor()
        with pytest.raises(ValueError, match="Parameter 'p' not found"):
            executor.expectation_value(qc, op)

    def test_expectation_value_multiple_circuits_single_observable_shape(self):
        """Test shape handling for multiple circuits with one observable."""
        qc1 = _build_circuit(1, [])
        qc2 = _build_circuit(1, [("x", [0])])
        op = QuantumOperator(["Z"], [1.0])

        executor = QulacsExecutor()
        values = executor.expectation_value([qc1, qc2], op)

        assert isinstance(values, np.ndarray)
        assert values.shape == (2,)
        assert np.isclose(values[0], 1.0, atol=1e-6)
        assert np.isclose(values[1], -1.0, atol=1e-6)

    def test_statevector_missing_parameter_raises(self):
        """Test missing parameter error handling in statevector."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])

        executor = QulacsExecutor()
        with pytest.raises(ValueError, match="Parameter 'x' not found"):
            executor.statevector(qc)


class TestQulacsExecutorDerivatives:
    def test_derivatives_default_expectation_value(self):
        """Test default derivative request branch (expectation value)."""
        qc = _build_circuit(1, [])
        op = QuantumOperator(["Z"], [1.0])

        executor = QulacsExecutor()
        value = executor.expectation_value_derivatives(qc, op)

        assert isinstance(value, (float, np.ndarray))

    def test_derivatives_higher_order_tuple_raises(self):
        """Test that higher-order derivative tuples are rejected."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])

        executor = QulacsExecutor()
        with pytest.raises(ValueError, match="Higher order derivatives"):
            executor.expectation_value_derivatives(qc, op, ("x", "x"), x=[0.1])

    def test_derivatives_unknown_parameter_type_in_circuit_branch_raises(self):
        """Test unknown derivative type handling for circuit parameter loop."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])

        executor = QulacsExecutor()
        with pytest.raises(ValueError, match="Unknown parameter type"):
            executor.expectation_value_derivatives(qc, op, 123, x=[0.1])

    def test_derivatives_unknown_parameter_type_in_observable_branch_raises(self):
        """Test unknown derivative type handling for observable parameter loop."""
        p = Parameters("p", 1)
        qc = _build_circuit(1, [])
        op = QuantumOperator(["Z"], [p[0]])

        executor = QulacsExecutor()
        with pytest.raises(ValueError, match="Unknown parameter type"):
            executor.expectation_value_derivatives(qc, op, 123, p=[0.1])

    def test_derivatives_unknown_derivative_name_raises(self):
        """Test unknown derivative string handling."""
        qc = _build_circuit(1, [])
        op = QuantumOperator(["Z"], [1.0])

        executor = QulacsExecutor()
        with pytest.raises(ValueError, match="Unknown derivative"):
            executor.expectation_value_derivatives(qc, op, "not_a_derivative")

    def test_derivatives_missing_circuit_parameter_raises(self):
        """Test missing circuit parameter error in derivatives."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])

        executor = QulacsExecutor()
        with pytest.raises(ValueError, match="Parameter 'x' not found"):
            executor.expectation_value_derivatives(qc, op, "x")

    def test_derivatives_missing_observable_parameter_raises(self):
        """Test missing observable parameter error in derivatives."""
        p = Parameters("p", 1)
        qc = _build_circuit(1, [])
        op = QuantumOperator(["Z"], [p[0]])

        executor = QulacsExecutor()
        with pytest.raises(ValueError, match="Parameter 'p' not found"):
            executor.expectation_value_derivatives(qc, op, "p")

    def test_derivatives_parameter_name_and_indexed_name(self):
        """Test circuit derivative resolution for vector and indexed names."""
        x = Parameters("x", 1)
        p = Parameters("p", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        op = QuantumOperator(["Z"], [p[0]])

        executor = QulacsExecutor()
        grad_by_vector = executor.expectation_value_derivatives(qc, op, "x", x=[0.0], p=[1.0])
        grad_by_index = executor.expectation_value_derivatives(qc, op, "x[0]", x=[0.0], p=[1.0])

        assert isinstance(grad_by_vector, np.ndarray)
        assert np.issubdtype(grad_by_vector.dtype, np.number)

        assert isinstance(grad_by_index, np.ndarray)
        assert np.issubdtype(grad_by_index.dtype, np.number)

    def test_derivatives_observable_parameter_name_and_indexed_name(self):
        """Test observable derivative resolution for vector and indexed names."""
        p = Parameters("p", 1)
        qc = _build_circuit(1, [])
        op = QuantumOperator(["Z"], [p[0]])

        executor = QulacsExecutor()
        grad_by_vector = executor.expectation_value_derivatives(qc, op, "p", p=[1.0])
        grad_by_index = executor.expectation_value_derivatives(qc, op, "p[0]", p=[1.0])

        assert isinstance(grad_by_vector, np.ndarray)
        assert np.issubdtype(grad_by_vector.dtype, np.number)

        assert isinstance(grad_by_index, np.ndarray)
        assert np.issubdtype(grad_by_index.dtype, np.number)

    def test_derivatives_circuit_and_observable_parameter_mix_raises(self):
        """Test that mixed circuit+observable first-order request is rejected."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        op = QuantumOperator(["Z"], [x[0]])

        executor = QulacsExecutor()
        with pytest.raises(ValueError, match="Higher order derivatives"):
            executor.expectation_value_derivatives(qc, op, "x", x=[0.2])

    def test_derivatives_multiple_outputs_with_empty_key(self):
        """Test dictionary output mapping for empty derivative key and named key."""
        x = Parameters("x", 1)
        p = Parameters("p", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        op = QuantumOperator(["Z"], [p[0]])

        executor = QulacsExecutor()
        result = executor.expectation_value_derivatives(qc, op, "", "x", x=[0.0], p=[1.0])

        assert isinstance(result, dict)
        assert "expectation_value" in result
        assert "x" in result

    def test_derivatives_circuit_gradient_multiple_operators_shape(self):
        """Test circuit gradient return path for multiple operators."""
        x = Parameters("x", 1)
        p = Parameters("p", 2)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        op = QuantumOperator(["Z", "X"], [p[0], p[1]])

        executor = QulacsExecutor()
        result = executor.expectation_value_derivatives(qc, op, "x", x=[0.3], p=[1.0, 1.0])

        assert isinstance(result, np.ndarray)
        assert result.ndim >= 1

    def test_derivatives_observable_gradient_multiple_operators_shape(self):
        """Test observable gradient return path for multiple operators."""
        p = Parameters("p", 2)
        qc = _build_circuit(1, [])
        op = QuantumOperator(["Z", "X"], [p[0], p[1]])

        executor = QulacsExecutor()
        result = executor.expectation_value_derivatives(qc, op, "p", p=[0.2, 0.1])

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 2
