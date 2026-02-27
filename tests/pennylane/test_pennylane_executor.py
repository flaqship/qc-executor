"""
Test suite for PennyLane executor.

This module tests the PennylaneExecutor class which executes quantum circuits
using PennyLane backend, including:
- Expectation value computation
- Sampling
- Statevector computation
- Derivative computation
- Caching
- Error handling
"""

import numpy as np
import pytest
from qiskit.circuit import ParameterVector

from executor import QuantumCircuit, QuantumOperator
from executor.pennylane.pennylane_executor import PennylaneExecutor


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
        executor = PennylaneExecutor()
        assert executor is not None
        assert executor.shots is None

    def test_initialization_with_shots(self):
        """Test executor initialization with shots parameter."""
        executor = PennylaneExecutor(shots=1000)
        assert executor.shots == 1000

    def test_initialization_with_seed(self):
        """Test executor initialization with seed parameter."""
        executor = PennylaneExecutor(seed=42)
        assert executor is not None
        assert executor.shots is None

    def test_initialization_with_all_params(self):
        """Test executor initialization with all parameters."""
        executor = PennylaneExecutor(shots=500, seed=123, log_file="test.log")
        assert executor.shots == 500

    # Expectation Value Tests

    def test_expectation_value_bell_state_z_basis(self):
        """Test expectation value of Bell state with Z operators."""
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        op = QuantumOperator(["ZI", "IZ"], [1.0, 1.0])

        executor = PennylaneExecutor()
        result = executor.expectation_value(qc, op)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 0.0, atol=1e-5)

    def test_expectation_value_bell_state_zz(self):
        """Test expectation value of Bell state with ZZ operator."""
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        op = QuantumOperator(["ZZ"], [1.0])

        executor = PennylaneExecutor()
        result = executor.expectation_value(qc, op)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_hadamard_x_basis(self):
        """Test expectation value of Hadamard state with X operator."""
        qc = _build_circuit(1, [("h", [0])])
        op = QuantumOperator(["X"], [1.0])

        executor = PennylaneExecutor()
        result = executor.expectation_value(qc, op)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_y_basis(self):
        """Test expectation value in Y basis (H followed by S gate)."""
        qc = _build_circuit(1, [("h", [0]), ("s", [0])])
        op = QuantumOperator(["Y"], [1.0])

        executor = PennylaneExecutor()
        result = executor.expectation_value(qc, op)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_with_circuit_parameter(self):
        """Test expectation value with parametric circuit (RX gate)."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])

        executor = PennylaneExecutor()
        result = executor.expectation_value(qc, op, x=[np.pi])

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, -1.0, atol=1e-5)

    def test_expectation_value_with_multiple_circuit_parameters(self):
        """Test expectation value with multiple circuit parameters."""
        x = ParameterVector("x", 2)
        qc = _build_circuit(2, [("rx", [0, x[0]]), ("ry", [1, x[1]])])
        op = QuantumOperator(["ZZ"], [1.0])

        executor = PennylaneExecutor()
        result = executor.expectation_value(qc, op, x=[0.0, 0.0])

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_with_observable_parameters(self):
        """Test expectation value with parametric observable."""
        pop = ParameterVector("pop", 2)
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        op = QuantumOperator(["ZI", "IZ"], [pop[0], pop[1]])

        executor = PennylaneExecutor()
        result = executor.expectation_value(qc, op, pop=[0.5, 0.5])

        assert isinstance(result, (float, np.ndarray))

    def test_expectation_value_with_circuit_and_observable_parameters(self):
        """Test expectation value with both circuit and observable parameters."""
        x = ParameterVector("x", 1)
        pop = ParameterVector("pop", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        op = QuantumOperator(["Z"], [pop[0]])

        executor = PennylaneExecutor()
        result = executor.expectation_value(qc, op, x=[0.0], pop=[1.0])

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_three_qubit_chain(self):
        """Test expectation value with three-qubit GHZ-type state."""
        qc = _build_circuit(3, [("h", [0]), ("cx", [0, 1]), ("cx", [1, 2])])
        op = QuantumOperator(["ZZZ"], [1.0])

        executor = PennylaneExecutor()
        result = executor.expectation_value(qc, op)

        assert isinstance(result, (float, np.ndarray))

    # Sampling Tests

    def test_sample_bell_state(self):
        """Test sampling from Bell state (should get 00 and 11)."""
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])

        executor = PennylaneExecutor(shots=1000, seed=42)
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

        executor = PennylaneExecutor(shots=100, seed=42)
        result = executor.sample(qc)

        samples = result[0]
        assert "11" in samples
        assert samples["11"] == 100

    def test_sample_with_parameter(self):
        """Test sampling with parametric circuit."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(2, [("rx", [0, x[0]])])

        executor = PennylaneExecutor(shots=1000, seed=42)
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

        executor = PennylaneExecutor(shots=1000, seed=42)
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

        executor = PennylaneExecutor()
        statevector = executor.statevector(qc)

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 4
        # Should be |00> state: [1, 0, 0, 0]
        assert np.isclose(abs(statevector[0]), 1.0, atol=1e-5)
        assert np.allclose(abs(statevector[1:]), 0.0, atol=1e-5)

    def test_statevector_x_gate(self):
        """Test statevector after X gate (should be |1>)."""
        qc = _build_circuit(1, [("x", [0])])

        executor = PennylaneExecutor()
        statevector = executor.statevector(qc)

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 2
        # Should be |1> state: [0, 1]
        assert np.isclose(abs(statevector[1]), 1.0, atol=1e-5)

    def test_statevector_hadamard(self):
        """Test statevector of Hadamard state (should be equal superposition)."""
        qc = _build_circuit(1, [("h", [0])])

        executor = PennylaneExecutor()
        statevector = executor.statevector(qc)

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 2
        # Should be (|0> + |1>)/sqrt(2)
        assert np.isclose(abs(statevector[0]), 1 / np.sqrt(2), atol=1e-5)
        assert np.isclose(abs(statevector[1]), 1 / np.sqrt(2), atol=1e-5)

    def test_statevector_bell_state(self):
        """Test statevector of Bell state."""
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])

        executor = PennylaneExecutor()
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

        executor = PennylaneExecutor()
        statevector = executor.statevector(qc, x=[np.pi / 2])

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 2
        # Statevector should be normalized
        assert np.isclose(np.sum(np.abs(statevector) ** 2), 1.0, atol=1e-5)

    def test_statevector_with_multiple_parameters(self):
        """Test statevector with multiple parameters."""
        x = ParameterVector("x", 2)
        qc = _build_circuit(2, [("rx", [0, x[0]]), ("ry", [1, x[1]])])

        executor = PennylaneExecutor()
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

        executor = PennylaneExecutor()
        result = executor.expectation_value_derivatives(qc, op, "x", x=[0.0])

        assert isinstance(result, (float, np.ndarray))

    def test_expectation_value_derivatives_indexed_parameter(self):
        """Test derivative with respect to indexed parameter (e.g., x[0])."""
        x = ParameterVector("x", 2)
        qc = _build_circuit(2, [("rx", [0, x[0]]), ("ry", [1, x[1]])])
        op = QuantumOperator(["ZI"], [1.0])

        executor = PennylaneExecutor()
        result = executor.expectation_value_derivatives(qc, op, "x[0]", x=[0.0, 0.0])

        assert isinstance(result, (float, np.ndarray))

    def test_expectation_value_derivatives_multiple_values(self):
        """Test requesting multiple derivatives (expectation value and parameter)."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])

        executor = PennylaneExecutor()
        result = executor.expectation_value_derivatives(qc, op, "expectation_value", "x", x=[0.0])

        assert isinstance(result, dict)
        assert "expectation_value" in result or "x" in result

    def test_expectation_value_derivatives_known_value(self):
        """Test derivative computation with known analytical result."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])

        executor = PennylaneExecutor()
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

        executor = PennylaneExecutor()

        with pytest.raises(ValueError, match="Parameter 'x' not found"):
            executor.expectation_value(qc, op)  # Missing x parameter

    def test_missing_parameter_error_in_sample(self):
        """Test that missing parameter raises ValueError in sample."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])

        executor = PennylaneExecutor(shots=1000)

        with pytest.raises(ValueError, match="Parameter 'x' not found"):
            executor.sample(qc)  # Missing x parameter

    def test_missing_parameter_error_in_statevector(self):
        """Test that missing parameter raises ValueError in statevector."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])

        executor = PennylaneExecutor()

        with pytest.raises(ValueError, match="Parameter 'x' not found"):
            executor.statevector(qc)  # Missing x parameter

    def test_missing_parameter_error_in_derivatives(self):
        """Test that missing parameter raises ValueError in expectation_value_derivatives."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])

        executor = PennylaneExecutor()

        with pytest.raises(ValueError, match="Parameter 'x' not found"):
            executor.expectation_value_derivatives(qc, op, "x")  # Missing x parameter

    # Caching Tests

    def test_circuit_caching(self):
        """Test that circuits are properly cached."""
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        executor = PennylaneExecutor()

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
        executor = PennylaneExecutor()

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
        executor = PennylaneExecutor(shots=500)
        assert executor.shots == 500

    def test_shots_property_setter_raises_error(self):
        """Test that shots setter raises NotImplementedError."""
        executor = PennylaneExecutor()

        with pytest.raises(NotImplementedError):
            executor.shots = 1000

    def test_remote_property(self):
        """Test that remote property returns False."""
        executor = PennylaneExecutor()
        assert executor.remote is False
