"""Tests for Pauli Propagation Executor."""

import numpy as np
import pytest

from executor.pauli_propagation.executor import PauliPropagationExecutor
from executor.pauli_propagation.operator_converter import (
    convert_operator, pauli_sum_to_sparse_pauli_op)
from executor.pauli_propagation.pauli_types import PauliSum
from executor.pauli_propagation.qiskit_converter import clear_cache

# Try to import qiskit for validation tests
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


@pytest.fixture(autouse=True)
def clear_circuit_cache():
    """Clear circuit conversion cache before each test."""
    clear_cache()
    yield
    clear_cache()


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
class TestOperatorConverter:
    """Test operator conversion between Qiskit and PauliSum."""

    def test_convert_string_operator(self):
        """Convert string Pauli operator."""
        psum = convert_operator("XYZ", nqubits=3)
        assert psum.nqubits == 3
        assert len(psum) == 1
        assert np.isclose(psum.get_coeff("XYZ"), 1.0)

    def test_convert_sparse_pauli_op_single_term(self):
        """Convert SparsePauliOp with single term."""
        # Qiskit uses opposite qubit ordering
        sparse_op = SparsePauliOp("ZYX", coeffs=[2.0])  # Qiskit: rightmost = qubit 0
        psum = convert_operator(sparse_op)

        # Our convention: leftmost = qubit 0, so this becomes XYZ
        assert psum.nqubits == 3
        assert len(psum) == 1
        assert np.isclose(psum.get_coeff("XYZ"), 2.0)

    def test_convert_sparse_pauli_op_multiple_terms(self):
        """Convert SparsePauliOp with multiple terms."""
        sparse_op = SparsePauliOp(["II", "ZZ", "XX"], coeffs=[1.0, 2.0, 3.0])
        psum = convert_operator(sparse_op)

        assert psum.nqubits == 2
        assert len(psum) == 3
        assert np.isclose(psum.get_coeff("II"), 1.0)
        assert np.isclose(psum.get_coeff("ZZ"), 2.0)
        assert np.isclose(psum.get_coeff("XX"), 3.0)

    def test_convert_sparse_pauli_op_complex_coeffs(self):
        """Convert SparsePauliOp with complex coefficients."""
        sparse_op = SparsePauliOp("XY", coeffs=[1.0 + 2.0j])
        psum = convert_operator(sparse_op)

        assert np.isclose(psum.get_coeff("YX"), 1.0 + 2.0j)  # Reversed

    def test_pauli_sum_to_sparse_pauli_op(self):
        """Convert PauliSum back to SparsePauliOp."""
        psum = PauliSum(2)
        psum.add_term("XY", 1.5)
        psum.add_term("ZZ", 2.5)

        sparse_op = pauli_sum_to_sparse_pauli_op(psum)

        assert sparse_op.num_qubits == 2
        # Should have 2 terms
        assert len(sparse_op.paulis) == 2

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion: SparsePauliOp → PauliSum → SparsePauliOp."""
        original = SparsePauliOp(["II", "XX", "ZZ"], coeffs=[1.0, 2.0, 3.0])

        # Convert to PauliSum and back
        psum = convert_operator(original)
        reconstructed = pauli_sum_to_sparse_pauli_op(psum)

        # Check equivalence (may have different term ordering)
        assert reconstructed.num_qubits == original.num_qubits
        assert len(reconstructed.paulis) == len(original.paulis)


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
class TestPauliPropagationExecutor:
    """Test PauliPropagationExecutor."""

    def test_init(self):
        """Test executor initialization."""
        executor = PauliPropagationExecutor(
            shots=1000, seed=42, truncate_threshold=1e-10, max_weight=5
        )
        assert executor.shots == 1000
        assert executor.remote == False
        assert executor.truncate_threshold == 1e-10
        assert executor.max_weight == 5

    def test_expectation_value_identity_circuit(self):
        """Identity circuit should not change observable."""
        executor = PauliPropagationExecutor()

        circuit = QuantumCircuit(2)
        # No gates - identity

        operator = SparsePauliOp("ZZ", coeffs=[1.0])

        result = executor.expectation_value(circuit, operator)

        # ⟨00|ZZ|00⟩ = 1
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_expectation_value_hadamard_x(self):
        """H gate transforms Z to X."""
        executor = PauliPropagationExecutor()

        circuit = QuantumCircuit(1)
        circuit.h(0)

        # Measure X (in propagated frame, this is Z before H)
        operator = SparsePauliOp("X", coeffs=[1.0])

        result = executor.expectation_value(circuit, operator)

        # |+⟩ = H|0⟩, ⟨+|X|+⟩ = 1
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_expectation_value_rotation_gate(self):
        """Test rotation gate RX(π/2)."""
        executor = PauliPropagationExecutor()

        circuit = QuantumCircuit(1)
        circuit.rx(np.pi / 2, 0)

        operator = SparsePauliOp("Z", coeffs=[1.0])

        result = executor.expectation_value(circuit, operator)

        # After RX(π/2), Z expectation should be 0
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_expectation_value_cnot_bell_state(self):
        """Bell state preparation and ZZ measurement."""
        executor = PauliPropagationExecutor()

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        operator = SparsePauliOp("ZZ", coeffs=[1.0])

        result = executor.expectation_value(circuit, operator)

        # Bell state: ⟨Φ+|ZZ|Φ+⟩ = 1
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_expectation_value_with_string_operator(self):
        """Test with string operator."""
        executor = PauliPropagationExecutor()

        circuit = QuantumCircuit(2)
        operator = "ZZ"

        result = executor.expectation_value(circuit, operator)

        # ⟨00|ZZ|00⟩ = 1
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_expectation_value_parametric_circuit(self):
        """Test parametric circuit with parameter binding."""
        executor = PauliPropagationExecutor()

        from qiskit.circuit import Parameter

        theta = Parameter("theta")

        circuit = QuantumCircuit(1)
        circuit.rx(theta, 0)

        operator = SparsePauliOp("Z", coeffs=[1.0])

        # RX(0) should leave Z unchanged
        result = executor.expectation_value(circuit, operator, theta=0.0)
        assert np.isclose(result, 1.0, atol=1e-10)

        # RX(π) should flip Z to -Z
        result = executor.expectation_value(circuit, operator, theta=np.pi)
        assert np.isclose(result, -1.0, atol=1e-10)

    def test_truncation_statistics(self):
        """Test that truncation stats are tracked."""
        executor = PauliPropagationExecutor(truncate_threshold=0.1)

        circuit = QuantumCircuit(1)
        circuit.rx(0.1, 0)  # Small rotation creates small terms

        operator = SparsePauliOp("Z", coeffs=[1.0])

        executor.expectation_value(circuit, operator)

        # Should have truncation stats
        stats = executor.get_truncation_stats()
        assert stats is not None
        assert stats.coeff_norm_total > 0

    def test_batch_execution_multiple_circuits(self):
        """Test batch execution with multiple circuits."""
        executor = PauliPropagationExecutor()

        circuit1 = QuantumCircuit(1)
        circuit2 = QuantumCircuit(1)
        circuit2.x(0)

        operator = SparsePauliOp("Z", coeffs=[1.0])

        results = executor.expectation_value([circuit1, circuit2], operator)

        assert len(results) == 2
        assert np.isclose(results[0], 1.0, atol=1e-10)  # ⟨0|Z|0⟩ = 1
        assert np.isclose(results[1], -1.0, atol=1e-10)  # ⟨1|Z|1⟩ = -1


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
class TestQiskitValidation:
    """Validate executor results against Qiskit statevector simulation."""

    def test_validate_single_qubit_gates(self):
        """Validate single-qubit gate results against Qiskit."""
        executor = PauliPropagationExecutor()

        test_cases = [
            ("h", {"gate": "h", "qubit": 0}, "X", 1.0),
            ("h", {"gate": "h", "qubit": 0}, "Z", 0.0),
            ("x", {"gate": "x", "qubit": 0}, "Z", -1.0),
            ("s", {"gate": "s", "qubit": 0}, "X", 0.0),
        ]

        for name, gate_spec, obs_str, _ in test_cases:
            circuit = QuantumCircuit(1)
            getattr(circuit, gate_spec["gate"])(gate_spec["qubit"])

            operator = SparsePauliOp(obs_str, coeffs=[1.0])

            # Our result
            our_result = executor.expectation_value(circuit, operator)

            # Qiskit result
            sv = Statevector.from_label("0")
            sv = sv.evolve(circuit)
            qiskit_result = sv.expectation_value(operator).real

            assert np.isclose(
                our_result, qiskit_result, atol=1e-10
            ), f"Mismatch for {name} gate with {obs_str} observable"

    def test_validate_two_qubit_gates(self):
        """Validate two-qubit gate results."""
        executor = PauliPropagationExecutor()

        # CNOT + ZZ observable
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        operator = SparsePauliOp("ZZ", coeffs=[1.0])

        our_result = executor.expectation_value(circuit, operator)

        sv = Statevector.from_label("00")
        sv = sv.evolve(circuit)
        qiskit_result = sv.expectation_value(operator).real

        assert np.isclose(our_result, qiskit_result, atol=1e-10)

    def test_validate_rotation_gates(self):
        """Validate parametric rotation gates."""
        executor = PauliPropagationExecutor()

        from qiskit.circuit import Parameter

        theta = Parameter("theta")

        circuit = QuantumCircuit(1)
        circuit.rx(theta, 0)

        operator = SparsePauliOp("Y", coeffs=[1.0])

        # Test multiple parameter values
        for angle in [0.0, np.pi / 4, np.pi / 2, np.pi]:
            our_result = executor.expectation_value(circuit, operator, theta=angle)

            # Qiskit validation
            bound_circuit = circuit.assign_parameters({theta: angle})
            sv = Statevector.from_label("0")
            sv = sv.evolve(bound_circuit)
            qiskit_result = sv.expectation_value(operator).real

            assert np.isclose(
                our_result, qiskit_result, atol=1e-10
            ), f"Mismatch for RX({angle}) with Y observable"

    def test_validate_complex_circuit(self):
        """Validate more complex circuit."""
        executor = PauliPropagationExecutor()

        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.h(1)
        circuit.cx(0, 2)
        circuit.cx(1, 2)
        circuit.rz(0.5, 2)

        operator = SparsePauliOp("ZZI", coeffs=[1.0])

        our_result = executor.expectation_value(circuit, operator)

        sv = Statevector.from_label("000")
        sv = sv.evolve(circuit)
        qiskit_result = sv.expectation_value(operator).real

        assert np.isclose(our_result, qiskit_result, atol=1e-10)

    def test_validate_with_truncation(self):
        """Verify truncation doesn't significantly affect result."""
        circuit = QuantumCircuit(2)
        circuit.rx(0.1, 0)
        circuit.ry(0.15, 1)
        circuit.cx(0, 1)

        operator = SparsePauliOp("ZZ", coeffs=[1.0])

        # Without truncation
        executor_exact = PauliPropagationExecutor()
        exact_result = executor_exact.expectation_value(circuit, operator)

        # With aggressive truncation
        executor_trunc = PauliPropagationExecutor(truncate_threshold=1e-6)
        trunc_result = executor_trunc.expectation_value(circuit, operator)

        # Qiskit reference
        sv = Statevector.from_label("00")
        sv = sv.evolve(circuit)
        qiskit_result = sv.expectation_value(operator).real

        # All should be close
        assert np.isclose(exact_result, qiskit_result, atol=1e-10)
        assert np.isclose(trunc_result, qiskit_result, atol=1e-5)  # Slightly looser tolerance

        # Check that truncation stats show error is small
        stats = executor_trunc.get_truncation_stats()
        assert stats.relative_error_bound < 0.01  # Less than 1% error


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
class TestDerivatives:
    """Test expectation_value_derivatives() method."""

    def test_rx_derivative_single_qubit(self):
        """Test derivative for single-qubit RX rotation."""
        from qiskit.circuit import Parameter

        executor = PauliPropagationExecutor()
        theta = Parameter("theta")

        circuit = QuantumCircuit(1)
        circuit.rx(theta, 0)
        operator = SparsePauliOp("Z")

        # Compute derivative at θ = π/4
        theta_val = np.pi / 4
        grad = executor.expectation_value_derivatives(circuit, operator, "theta", theta=theta_val)

        # Finite difference approximation
        eps = 1e-5
        f_plus = executor.expectation_value(circuit, operator, theta=theta_val + eps)
        f_minus = executor.expectation_value(circuit, operator, theta=theta_val - eps)
        fd_grad = (f_plus - f_minus) / (2 * eps)

        assert np.isclose(grad, fd_grad, atol=1e-6)

    def test_ry_derivative(self):
        """Test derivative for RY rotation."""
        from qiskit.circuit import Parameter

        executor = PauliPropagationExecutor()
        theta = Parameter("theta")

        circuit = QuantumCircuit(1)
        circuit.ry(theta, 0)
        operator = SparsePauliOp("X")

        theta_val = np.pi / 3
        grad = executor.expectation_value_derivatives(circuit, operator, "theta", theta=theta_val)

        # Finite difference
        eps = 1e-5
        f_plus = executor.expectation_value(circuit, operator, theta=theta_val + eps)
        f_minus = executor.expectation_value(circuit, operator, theta=theta_val - eps)
        fd_grad = (f_plus - f_minus) / (2 * eps)

        assert np.isclose(grad, fd_grad, atol=1e-6)

    def test_rz_derivative(self):
        """Test derivative for RZ rotation."""
        from qiskit.circuit import Parameter

        executor = PauliPropagationExecutor()
        theta = Parameter("theta")

        circuit = QuantumCircuit(1)
        circuit.h(0)  # Put in superposition
        circuit.rz(theta, 0)
        operator = SparsePauliOp("X")

        theta_val = np.pi / 6
        grad = executor.expectation_value_derivatives(circuit, operator, "theta", theta=theta_val)

        # Finite difference
        eps = 1e-5
        f_plus = executor.expectation_value(circuit, operator, theta=theta_val + eps)
        f_minus = executor.expectation_value(circuit, operator, theta=theta_val - eps)
        fd_grad = (f_plus - f_minus) / (2 * eps)

        assert np.isclose(grad, fd_grad, atol=1e-6)

    def test_multiple_parameters(self):
        """Test derivative with multiple parameters."""
        from qiskit.circuit import Parameter

        executor = PauliPropagationExecutor()
        theta1 = Parameter("theta1")
        theta2 = Parameter("theta2")

        circuit = QuantumCircuit(2)
        circuit.rx(theta1, 0)
        circuit.ry(theta2, 1)
        operator = SparsePauliOp("ZZ")

        theta1_val = 0.5
        theta2_val = 0.7

        # Test individual derivatives
        grad1 = executor.expectation_value_derivatives(
            circuit, operator, "theta1", theta1=theta1_val, theta2=theta2_val
        )
        grad2 = executor.expectation_value_derivatives(
            circuit, operator, "theta2", theta1=theta1_val, theta2=theta2_val
        )

        # Finite differences
        eps = 1e-5
        f_plus1 = executor.expectation_value(
            circuit, operator, theta1=theta1_val + eps, theta2=theta2_val
        )
        f_minus1 = executor.expectation_value(
            circuit, operator, theta1=theta1_val - eps, theta2=theta2_val
        )
        fd_grad1 = (f_plus1 - f_minus1) / (2 * eps)

        f_plus2 = executor.expectation_value(
            circuit, operator, theta1=theta1_val, theta2=theta2_val + eps
        )
        f_minus2 = executor.expectation_value(
            circuit, operator, theta1=theta1_val, theta2=theta2_val - eps
        )
        fd_grad2 = (f_plus2 - f_minus2) / (2 * eps)

        assert np.isclose(grad1, fd_grad1, atol=1e-6)
        assert np.isclose(grad2, fd_grad2, atol=1e-6)

    def test_derivative_list(self):
        """Test computing multiple derivatives at once."""
        from qiskit.circuit import Parameter

        executor = PauliPropagationExecutor()
        theta1 = Parameter("theta1")
        theta2 = Parameter("theta2")

        circuit = QuantumCircuit(2)
        circuit.rx(theta1, 0)
        circuit.ry(theta2, 1)
        operator = SparsePauliOp("ZZ")

        theta1_val = 0.5
        theta2_val = 0.7

        # Compute both derivatives
        grads = executor.expectation_value_derivatives(
            circuit, operator, ["theta1", "theta2"], theta1=theta1_val, theta2=theta2_val
        )

        assert len(grads) == 2
        assert isinstance(grads, np.ndarray)

    def test_batch_circuits(self):
        """Test derivatives with batch circuits."""
        from qiskit.circuit import Parameter

        executor = PauliPropagationExecutor()
        theta = Parameter("theta")

        circuit1 = QuantumCircuit(1)
        circuit1.rx(theta, 0)

        circuit2 = QuantumCircuit(1)
        circuit2.ry(theta, 0)

        operator = SparsePauliOp("Z")

        theta_val = 0.3
        grads = executor.expectation_value_derivatives(
            [circuit1, circuit2], operator, "theta", theta=theta_val
        )

        assert len(grads) == 2


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
class TestStatevector:
    """Test statevector() method."""

    def test_identity_circuit(self):
        """Test statevector for identity circuit (no gates)."""
        executor = PauliPropagationExecutor()
        circuit = QuantumCircuit(2)

        sv = executor.statevector(circuit)

        # Should be |00⟩ = [1, 0, 0, 0]
        expected = np.array([1, 0, 0, 0], dtype=complex)
        assert np.allclose(sv, expected, atol=1e-10)

    def test_hadamard_single_qubit(self):
        """Test statevector for Hadamard gate."""
        executor = PauliPropagationExecutor()
        circuit = QuantumCircuit(1)
        circuit.h(0)

        sv = executor.statevector(circuit)

        # Should be |+⟩ = [1/√2, 1/√2]
        expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
        assert np.allclose(sv, expected, atol=1e-10)

    def test_x_gate(self):
        """Test statevector for X gate."""
        executor = PauliPropagationExecutor()
        circuit = QuantumCircuit(1)
        circuit.x(0)

        sv = executor.statevector(circuit)

        # Should be |1⟩ = [0, 1]
        expected = np.array([0, 1], dtype=complex)
        assert np.allclose(sv, expected, atol=1e-10)

    def test_bell_state(self):
        """Test statevector for Bell state preparation."""
        executor = PauliPropagationExecutor()
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        sv = executor.statevector(circuit)

        # Should be (|00⟩ + |11⟩)/√2 = [1/√2, 0, 0, 1/√2]
        expected = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=complex)
        assert np.allclose(sv, expected, atol=1e-10)

    def test_normalization(self):
        """Test that statevector is properly normalized."""
        executor = PauliPropagationExecutor()
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.ry(0.5, 1)
        circuit.cx(0, 1)

        sv = executor.statevector(circuit)

        # Check normalization
        norm = np.sum(np.abs(sv) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)

    def test_parametric_circuit(self):
        """Test statevector with parametric circuit."""
        from qiskit.circuit import Parameter

        executor = PauliPropagationExecutor()
        theta = Parameter("theta")

        circuit = QuantumCircuit(1)
        circuit.ry(theta, 0)

        sv = executor.statevector(circuit, theta=np.pi / 2)

        # RY(π/2)|0⟩ = [cos(π/4), sin(π/4)] = [1/√2, 1/√2]
        expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
        assert np.allclose(sv, expected, atol=1e-10)

    def test_compare_with_qiskit(self):
        """Test statevector matches Qiskit."""
        executor = PauliPropagationExecutor()
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.rz(0.5, 2)
        circuit.cx(1, 2)

        our_sv = executor.statevector(circuit)

        # Qiskit reference
        qiskit_sv = Statevector.from_label("000")
        qiskit_sv = qiskit_sv.evolve(circuit)

        assert np.allclose(our_sv, qiskit_sv.data, atol=1e-10)


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
class TestSampling:
    """Test sample() method."""

    def test_identity_circuit(self):
        """Test sampling from identity circuit."""
        executor = PauliPropagationExecutor(shots=1000, seed=42)
        circuit = QuantumCircuit(2)

        counts = executor.sample(circuit)

        # Should only get '00'
        assert "00" in counts
        assert counts["00"] == 1000
        assert len(counts) == 1

    def test_x_gate_deterministic(self):
        """Test sampling with X gate (deterministic outcome)."""
        executor = PauliPropagationExecutor(shots=500, seed=42)
        circuit = QuantumCircuit(1)
        circuit.x(0)

        counts = executor.sample(circuit)

        # Should only get '1'
        assert "1" in counts
        assert counts["1"] == 500
        assert len(counts) == 1

    def test_hadamard_distribution(self):
        """Test sampling from Hadamard superposition."""
        executor = PauliPropagationExecutor(shots=10000, seed=42)
        circuit = QuantumCircuit(1)
        circuit.h(0)

        counts = executor.sample(circuit)

        # Should get roughly 50-50 distribution
        assert "0" in counts
        assert "1" in counts
        assert 4000 < counts["0"] < 6000  # Allow statistical variation
        assert 4000 < counts["1"] < 6000

    def test_bell_state_sampling(self):
        """Test sampling from Bell state."""
        executor = PauliPropagationExecutor(shots=10000, seed=42)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        counts = executor.sample(circuit)

        # Should get roughly 50% '00' and 50% '11'
        assert "00" in counts
        assert "11" in counts
        assert counts.get("00", 0) > 4000
        assert counts.get("11", 0) > 4000
        # Should get very few (ideally zero) of '01' and '10'
        assert counts.get("01", 0) + counts.get("10", 0) < 500

    def test_seed_reproducibility(self):
        """Test that same seed gives same samples."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        executor1 = PauliPropagationExecutor(shots=1000, seed=123)
        counts1 = executor1.sample(circuit)

        executor2 = PauliPropagationExecutor(shots=1000, seed=123)
        counts2 = executor2.sample(circuit)

        # Should get identical counts
        assert counts1 == counts2

    def test_shot_count(self):
        """Test that shot count is respected."""
        executor = PauliPropagationExecutor(shots=500, seed=42)
        circuit = QuantumCircuit(1)
        circuit.h(0)

        counts = executor.sample(circuit)

        # Total counts should equal shots
        total = sum(counts.values())
        assert total == 500

    def test_batch_sampling(self):
        """Test sampling with batch circuits."""
        executor = PauliPropagationExecutor(shots=1000, seed=42)

        circuit1 = QuantumCircuit(1)
        circuit1.x(0)

        circuit2 = QuantumCircuit(1)
        # Identity

        counts_list = executor.sample([circuit1, circuit2])

        assert len(counts_list) == 2
        assert counts_list[0]["1"] == 1000  # X gate
        assert counts_list[1]["0"] == 1000  # Identity

    def test_parametric_sampling(self):
        """Test sampling with parametric circuit."""
        from qiskit.circuit import Parameter

        executor = PauliPropagationExecutor(shots=10000, seed=42)
        theta = Parameter("theta")

        circuit = QuantumCircuit(1)
        circuit.ry(theta, 0)

        # θ=π/2 should give equal superposition
        counts = executor.sample(circuit, theta=np.pi / 2)

        assert 4000 < counts.get("0", 0) < 6000
        assert 4000 < counts.get("1", 0) < 6000


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
class TestBatchExpectationValue:
    """Test batch expectation_value() using batch_propagate under the hood."""

    def test_multi_operator_matches_single_calls(self):
        """expectation_value with operator list must match individual calls."""
        executor = PauliPropagationExecutor()

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        ops = [
            SparsePauliOp("ZZ"),
            SparsePauliOp("XX"),
            SparsePauliOp("IZ"),
        ]

        batch_result = executor.expectation_value(circuit, ops)

        for i, op in enumerate(ops):
            single = executor.expectation_value(circuit, op)
            assert np.isclose(
                batch_result[i], single
            ), f"Mismatch at operator {i}: batch={batch_result[i]}, single={single}"

    def test_multi_circuit_multi_operator_ordering(self):
        """Result ordering for multi-circuit × multi-operator must be circuit-major."""
        executor = PauliPropagationExecutor()

        circ1 = QuantumCircuit(1)
        circ1.x(0)

        circ2 = QuantumCircuit(1)
        # identity

        ops = [SparsePauliOp("Z"), SparsePauliOp("X")]

        results = executor.expectation_value([circ1, circ2], ops)

        # Expected: [<Z>_circ1, <X>_circ1, <Z>_circ2, <X>_circ2]
        assert results.shape == (4,)
        assert np.isclose(results[0], -1.0)  # <Z> after X gate = -1
        assert np.isclose(results[1], 0.0)  # <X> after X gate = 0
        assert np.isclose(results[2], 1.0)  # <Z> on |0> = 1
        assert np.isclose(results[3], 0.0)  # <X> on |0> = 0

    def test_single_operator_still_returns_float(self):
        """Single operator input must still return a scalar float."""
        executor = PauliPropagationExecutor()
        circuit = QuantumCircuit(1)
        result = executor.expectation_value(circuit, SparsePauliOp("Z"))
        assert isinstance(result, float)
        assert np.isclose(result, 1.0)

    def test_multi_operator_parametric_circuit(self):
        """Batch evaluation works correctly with parametric circuits."""
        from qiskit.circuit import Parameter

        executor = PauliPropagationExecutor()
        theta = Parameter("theta")

        circuit = QuantumCircuit(1)
        circuit.ry(theta, 0)

        ops = [SparsePauliOp("Z"), SparsePauliOp("X")]

        # At theta=pi/2: RY(pi/2)|0> = |+x> (up to global phase)
        # <Z> = 0, <X> = 1; here we only check batch vs single-call consistency
        results = executor.expectation_value(circuit, ops, theta=np.pi / 2)

        single_z = executor.expectation_value(circuit, ops[0], theta=np.pi / 2)
        single_x = executor.expectation_value(circuit, ops[1], theta=np.pi / 2)

        assert np.isclose(results[0], single_z)
        assert np.isclose(results[1], single_x)
