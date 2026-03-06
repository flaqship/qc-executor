"""Tests for PauliPropagationExecutor (strict native API)."""

import numpy as np
import pytest

from executor.parameters import Parameters
from executor.pauli_propagation import (
    PauliPropagationCircuit,
    PauliPropagationExecutor,
    PauliPropagationObservable,
)


class TestPauliPropagationExecutor:
    def test_init(self):
        executor = PauliPropagationExecutor(
            shots=1000,
            seed=42,
            truncate_threshold=1e-10,
            max_weight=5,
        )
        assert executor.shots == 1000
        assert executor.remote is False
        assert executor.truncate_threshold == 1e-10
        assert executor.max_weight == 5

    def test_expectation_value_identity_circuit(self):
        executor = PauliPropagationExecutor()
        circuit = PauliPropagationCircuit(2)
        operator = PauliPropagationObservable(["ZZ"], [1.0])

        result = executor.expectation_value(circuit, operator)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_expectation_value_hadamard_x(self):
        executor = PauliPropagationExecutor()
        circuit = PauliPropagationCircuit(1)
        circuit.h(0)
        operator = PauliPropagationObservable(["X"], [1.0])

        result = executor.expectation_value(circuit, operator)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_expectation_value_rotation_gate(self):
        executor = PauliPropagationExecutor()
        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, np.pi / 2)
        operator = PauliPropagationObservable(["Z"], [1.0])

        result = executor.expectation_value(circuit, operator)
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_expectation_value_cnot_bell_state(self):
        executor = PauliPropagationExecutor()
        circuit = PauliPropagationCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        operator = PauliPropagationObservable(["ZZ"], [1.0])

        result = executor.expectation_value(circuit, operator)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_expectation_value_parametric_circuit(self):
        executor = PauliPropagationExecutor()
        theta = Parameters("theta", 1)

        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, theta[0])
        operator = PauliPropagationObservable(["Z"], [1.0])

        result = executor.expectation_value(circuit, operator, **{"theta[0]": 0.0})
        assert np.isclose(result, 1.0, atol=1e-10)

        result = executor.expectation_value(circuit, operator, **{"theta[0]": np.pi})
        assert np.isclose(result, -1.0, atol=1e-10)

    def test_truncation_statistics(self):
        executor = PauliPropagationExecutor(truncate_threshold=0.1)
        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, 0.1)
        operator = PauliPropagationObservable(["Z"], [1.0])

        executor.expectation_value(circuit, operator)
        stats = executor.get_truncation_stats()
        assert stats is not None
        assert stats.coeff_norm_total > 0

    def test_batch_execution_multiple_circuits(self):
        executor = PauliPropagationExecutor()

        circuit1 = PauliPropagationCircuit(1)
        circuit2 = PauliPropagationCircuit(1)
        circuit2.x(0)

        operator = PauliPropagationObservable(["Z"], [1.0])

        results = executor.expectation_value([circuit1, circuit2], operator)

        assert len(results) == 2
        assert np.isclose(results[0], 1.0, atol=1e-10)
        assert np.isclose(results[1], -1.0, atol=1e-10)


class TestBatchExpectationValue:
    def test_multi_operator_matches_single_calls(self):
        executor = PauliPropagationExecutor()

        circuit = PauliPropagationCircuit(1)
        circuit.h(0)

        op_x = PauliPropagationObservable(["X"], [1.0])
        op_z = PauliPropagationObservable(["Z"], [1.0])

        batch = executor.expectation_value(circuit, [op_x, op_z])
        single_x = executor.expectation_value(circuit, op_x)
        single_z = executor.expectation_value(circuit, op_z)

        assert np.isclose(batch[0], single_x, atol=1e-10)
        assert np.isclose(batch[1], single_z, atol=1e-10)

    def test_multi_circuit_multi_operator_ordering(self):
        executor = PauliPropagationExecutor()

        c0 = PauliPropagationCircuit(1)
        c1 = PauliPropagationCircuit(1)
        c1.x(0)

        op_z = PauliPropagationObservable(["Z"], [1.0])
        op_x = PauliPropagationObservable(["X"], [1.0])

        results = executor.expectation_value([c0, c1], [op_z, op_x])

        expected = np.array([1.0, 0.0, -1.0, 0.0])
        assert np.allclose(results, expected, atol=1e-10)

    def test_single_operator_still_returns_float(self):
        executor = PauliPropagationExecutor()
        circuit = PauliPropagationCircuit(1)
        operator = PauliPropagationObservable(["Z"], [1.0])

        result = executor.expectation_value(circuit, operator)
        assert isinstance(result, float)


class TestStrictInputs:
    def test_rejects_non_native_circuit(self):
        executor = PauliPropagationExecutor()
        operator = PauliPropagationObservable(["Z"], [1.0])

        with pytest.raises(TypeError, match="PauliPropagationCircuit"):
            executor.expectation_value("not a circuit", operator)

    def test_rejects_non_native_operator(self):
        executor = PauliPropagationExecutor()
        circuit = PauliPropagationCircuit(1)

        with pytest.raises(TypeError, match="PauliPropagationObservable"):
            executor.expectation_value(circuit, "not an operator")


class TestParameterNormalization:
    def test_expectation_value_with_list_parameters(self):
        """Test that parameters can be passed as lists like x=[0.1], p=[0.3]."""
        executor = PauliPropagationExecutor()
        theta = Parameters("theta", 1)

        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, theta[0])
        operator = PauliPropagationObservable(["Z"], [1.0])

        # Test list format: theta=[0.0]
        result = executor.expectation_value(circuit, operator, theta=[0.0])
        assert np.isclose(result, 1.0, atol=1e-10)

        # Test list format: theta=[pi]
        result = executor.expectation_value(circuit, operator, theta=[np.pi])
        assert np.isclose(result, -1.0, atol=1e-10)

    def test_expectation_value_with_multiple_list_parameters(self):
        """Test multiple parameters in list format."""
        executor = PauliPropagationExecutor()
        # Use simple Parameters that match what bind_parameters expects
        theta = Parameters("theta", 2)

        circuit = PauliPropagationCircuit(2)
        circuit.rx(0, theta[0])
        circuit.ry(1, theta[1])

        operator = PauliPropagationObservable(["ZI", "IZ"], [1.0, 1.0])

        # Test with list parameters in correct format
        result = executor.expectation_value(
            circuit, operator, theta=[0.0, 0.0]
        )
        assert np.isfinite(result)

    def test_expectation_value_with_indexed_parameters(self):
        """Test that indexed format still works (backward compatibility)."""
        executor = PauliPropagationExecutor()
        theta = Parameters("theta", 1)

        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, theta[0])
        operator = PauliPropagationObservable(["Z"], [1.0])

        # Test indexed format: {"theta[0]": 0.0}
        result = executor.expectation_value(circuit, operator, **{"theta[0]": 0.0})
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_expectation_value_derivatives_with_list_parameters(self):
        """Test derivatives with list parameters."""
        executor = PauliPropagationExecutor()
        theta = Parameters("theta", 1)

        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, theta[0])
        operator = PauliPropagationObservable(["X"], [1.0])

        # Test with list parameters
        result = executor.expectation_value_derivatives(
            circuit, operator, "theta", theta=[0.0]
        )
        assert isinstance(result, (float, np.ndarray))

    def test_statevector_with_list_parameters(self):
        """Test statevector with list parameters."""
        executor = PauliPropagationExecutor()
        theta = Parameters("theta", 1)

        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, theta[0])

        # Test with list parameters
        result = executor.statevector(circuit, theta=[0.0])
        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_sample_with_list_parameters(self):
        """Test sampling with list parameters."""
        executor = PauliPropagationExecutor(shots=100, seed=42)
        theta = Parameters("theta", 1)

        circuit = PauliPropagationCircuit(1)
        circuit.h(0)
        circuit.rx(0, theta[0])

        # Test with list parameters
        result = executor.sample(circuit, theta=[0.0])
        assert isinstance(result, dict)
        assert sum(result.values()) == 100  # Total shots
