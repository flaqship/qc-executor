"""Integration tests for PauliPropagation within the executor library.

Tests that PauliPropagationExecutor works as a proper executor:
- Inherits from ExecutorBase
- Accepts the library's QuantumCircuit/QuantumOperator wrapper types
- Produces results consistent with other executors (QiskitExecutor)
- Importable from expected paths
"""

import pytest
import numpy as np

# Test import paths
try:
    from executor.pauli_propagation import PauliPropagationExecutor
    from executor.pauli_propagation import PauliSum, PauliString
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False

try:
    from executor import QuantumCircuit, QuantumOperator, Parameters
    from executor.base import ExecutorBase
    EXECUTOR_AVAILABLE = True
except ImportError:
    EXECUTOR_AVAILABLE = False

try:
    from executor.qiskit.qiskit_executor import QiskitExecutor
    QISKIT_EXECUTOR_AVAILABLE = True
except ImportError:
    QISKIT_EXECUTOR_AVAILABLE = False

try:
    from qiskit.circuit import Parameter as QiskitParameter, ParameterVector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


# ─── Import path tests ───────────────────────────────────────────────────────

class TestImportPaths:
    """Test that all expected import paths work."""

    def test_import_executor_class(self):
        from executor.pauli_propagation import PauliPropagationExecutor
        assert PauliPropagationExecutor is not None

    def test_import_pauli_types(self):
        from executor.pauli_propagation import PauliSum, PauliString
        assert PauliSum is not None
        assert PauliString is not None

    def test_import_via_top_level(self):
        from executor import pauli_propagation
        assert hasattr(pauli_propagation, 'PauliPropagationExecutor')

    def test_import_truncation(self):
        from executor.pauli_propagation import TruncationStats
        assert TruncationStats is not None


# ─── Inheritance & interface tests ────────────────────────────────────────────

@pytest.mark.skipif(not EXECUTOR_AVAILABLE or not IMPORT_AVAILABLE,
                    reason="executor package not installed")
class TestInheritance:
    """Test that PauliPropagationExecutor is a proper ExecutorBase subclass."""

    def test_isinstance_executor_base(self):
        executor = PauliPropagationExecutor()
        assert isinstance(executor, ExecutorBase)

    def test_remote_property(self):
        executor = PauliPropagationExecutor()
        assert executor.remote is False

    def test_shots_property(self):
        executor = PauliPropagationExecutor(shots=500)
        assert executor.shots == 500

    def test_default_shots_none(self):
        executor = PauliPropagationExecutor()
        assert executor.shots is None

    def test_truncation_params(self):
        executor = PauliPropagationExecutor(
            truncate_threshold=1e-8, max_weight=3
        )
        assert executor.truncate_threshold == 1e-8
        assert executor.max_weight == 3


# ─── Wrapper type tests (core integration) ───────────────────────────────────

@pytest.mark.skipif(not EXECUTOR_AVAILABLE or not IMPORT_AVAILABLE or not QISKIT_AVAILABLE,
                    reason="Required packages not installed")
class TestWrapperTypes:
    """Test executor with the library's QuantumCircuit/QuantumOperator wrappers."""

    def test_expectation_value_wrapper_types(self):
        """expectation_value with library QuantumCircuit + QuantumOperator."""
        executor = PauliPropagationExecutor()

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        op = QuantumOperator(["ZZ"], [1.0])

        result = executor.expectation_value(qc, op)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_expectation_value_identity(self):
        """Identity circuit, Z observable."""
        executor = PauliPropagationExecutor()

        qc = QuantumCircuit(1)
        op = QuantumOperator(["Z"], [1.0])

        result = executor.expectation_value(qc, op)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_expectation_value_x_gate(self):
        """X gate flips Z expectation."""
        executor = PauliPropagationExecutor()

        qc = QuantumCircuit(1)
        qc.x(0)
        op = QuantumOperator(["Z"], [1.0])

        result = executor.expectation_value(qc, op)
        assert np.isclose(result, -1.0, atol=1e-10)

    def test_expectation_value_parametric(self):
        """Parametric circuit with wrapper types."""
        executor = PauliPropagationExecutor()

        p = Parameters('theta', 1)
        qc = QuantumCircuit(1)
        qc.rx(0, p[0])
        op = QuantumOperator(["Z"], [1.0])

        result = executor.expectation_value(qc, op, **{'theta[0]': 0.0})
        assert np.isclose(result, 1.0, atol=1e-10)

        result = executor.expectation_value(qc, op, **{'theta[0]': np.pi})
        assert np.isclose(result, -1.0, atol=1e-10)

    def test_expectation_value_multi_term_operator(self):
        """Multi-term operator with wrapper types."""
        executor = PauliPropagationExecutor()

        qc = QuantumCircuit(2)
        op = QuantumOperator(["ZI", "IZ"], [0.5, 0.5])

        # <00| (0.5*ZI + 0.5*IZ) |00> = 0.5 + 0.5 = 1.0
        result = executor.expectation_value(qc, op)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_batch_circuits_wrapper(self):
        """Batch execution with wrapper type circuits."""
        executor = PauliPropagationExecutor()

        qc1 = QuantumCircuit(1)
        qc2 = QuantumCircuit(1)
        qc2.x(0)

        op = QuantumOperator(["Z"], [1.0])

        results = executor.expectation_value([qc1, qc2], op)
        assert len(results) == 2
        assert np.isclose(results[0], 1.0, atol=1e-10)
        assert np.isclose(results[1], -1.0, atol=1e-10)

    def test_batch_operators_wrapper(self):
        """Batch execution with wrapper type operators."""
        executor = PauliPropagationExecutor()

        qc = QuantumCircuit(1)
        qc.h(0)

        op_x = QuantumOperator(["X"], [1.0])
        op_z = QuantumOperator(["Z"], [1.0])

        results = executor.expectation_value(qc, [op_x, op_z])
        assert len(results) == 2
        assert np.isclose(results[0], 1.0, atol=1e-10)   # <+|X|+> = 1
        assert np.isclose(results[1], 0.0, atol=1e-10)    # <+|Z|+> = 0

    def test_sample_wrapper(self):
        """sample() with library QuantumCircuit."""
        executor = PauliPropagationExecutor(shots=1000, seed=42)

        qc = QuantumCircuit(1)
        qc.x(0)

        counts = executor.sample(qc)
        assert '1' in counts
        assert counts['1'] == 1000

    def test_statevector_wrapper(self):
        """statevector() with library QuantumCircuit."""
        executor = PauliPropagationExecutor()

        qc = QuantumCircuit(1)
        qc.h(0)

        sv = executor.statevector(qc)
        expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
        assert np.allclose(sv, expected, atol=1e-10)

    def test_statevector_parametric_wrapper(self):
        """statevector() with parametric wrapper circuit."""
        executor = PauliPropagationExecutor()

        p = Parameters('theta', 1)
        qc = QuantumCircuit(1)
        qc.ry(0, p[0])

        sv = executor.statevector(qc, **{'theta[0]': np.pi / 2})
        expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
        assert np.allclose(sv, expected, atol=1e-10)


# ─── Derivative tests with wrapper types ──────────────────────────────────────

@pytest.mark.skipif(not EXECUTOR_AVAILABLE or not IMPORT_AVAILABLE or not QISKIT_AVAILABLE,
                    reason="Required packages not installed")
class TestDerivativesWrapper:
    """Test expectation_value_derivatives with wrapper types and Parameter objects."""

    def test_derivative_with_string_param(self):
        """Derivative using string parameter name."""
        executor = PauliPropagationExecutor()

        p = Parameters('theta', 1)
        qc = QuantumCircuit(1)
        qc.rx(0, p[0])
        op = QuantumOperator(["Z"], [1.0])

        theta_val = np.pi / 4
        grad = executor.expectation_value_derivatives(
            qc, op, 'theta[0]', **{'theta[0]': theta_val}
        )

        # Finite difference check
        eps = 1e-5
        f_plus = executor.expectation_value(qc, op, **{'theta[0]': theta_val + eps})
        f_minus = executor.expectation_value(qc, op, **{'theta[0]': theta_val - eps})
        fd_grad = (f_plus - f_minus) / (2 * eps)

        assert np.isclose(grad, fd_grad, atol=1e-6)

    def test_derivative_with_parameter_object(self):
        """Derivative using Parameter object (converted to name internally)."""
        executor = PauliPropagationExecutor()

        p = Parameters('theta', 1)
        qc = QuantumCircuit(1)
        qc.rx(0, p[0])
        op = QuantumOperator(["Z"], [1.0])

        theta_val = np.pi / 3
        # Pass the Parameter object itself — its .name is 'theta[0]'
        grad = executor.expectation_value_derivatives(
            qc, op, p[0], **{'theta[0]': theta_val}
        )

        eps = 1e-5
        f_plus = executor.expectation_value(qc, op, **{'theta[0]': theta_val + eps})
        f_minus = executor.expectation_value(qc, op, **{'theta[0]': theta_val - eps})
        fd_grad = (f_plus - f_minus) / (2 * eps)

        assert np.isclose(grad, fd_grad, atol=1e-6)

    def test_derivative_multiple_params(self):
        """Multiple derivatives at once with wrapper types."""
        executor = PauliPropagationExecutor()

        p = Parameters('p', 2)
        qc = QuantumCircuit(2)
        qc.rx(0, p[0])
        qc.ry(1, p[1])
        op = QuantumOperator(["ZZ"], [1.0])

        grads = executor.expectation_value_derivatives(
            qc, op, ['p[0]', 'p[1]'], **{'p[0]': 0.5, 'p[1]': 0.7}
        )

        assert len(grads) == 2
        assert isinstance(grads, np.ndarray)


# ─── Cross-executor validation ────────────────────────────────────────────────

@pytest.mark.skipif(not EXECUTOR_AVAILABLE or not IMPORT_AVAILABLE
                    or not QISKIT_EXECUTOR_AVAILABLE or not QISKIT_AVAILABLE,
                    reason="Required packages not installed")
class TestCrossExecutorValidation:
    """Compare PauliPropagationExecutor results against QiskitExecutor."""

    def test_expectation_value_matches_qiskit_executor(self):
        """Expectation values should match between executors."""
        pp_exec = PauliPropagationExecutor()
        qk_exec = QiskitExecutor(seed=0)

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(1, 0.5)

        op = QuantumOperator(["ZZ"], [1.0])

        pp_result = pp_exec.expectation_value(qc, op)
        qk_result = qk_exec.expectation_value(qc, op)

        assert np.isclose(pp_result, qk_result, atol=1e-8)

    def test_expectation_value_multi_operator_matches(self):
        """Multi-term operator results should match."""
        pp_exec = PauliPropagationExecutor()
        qk_exec = QiskitExecutor(seed=0)

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)

        op = QuantumOperator(["ZI", "IZ", "XX"], [0.3, 0.5, 0.2])

        pp_result = pp_exec.expectation_value(qc, op)
        qk_result = qk_exec.expectation_value(qc, op)

        assert np.isclose(pp_result, qk_result, atol=1e-8)

    def test_parametric_circuit_matches(self):
        """Parametric circuit results should match."""
        pp_exec = PauliPropagationExecutor()
        qk_exec = QiskitExecutor(seed=0)

        p = Parameters('theta', 2)
        qc = QuantumCircuit(2)
        qc.rx(0, p[0])
        qc.ry(1, p[1])
        qc.cx(0, 1)

        op = QuantumOperator(["ZZ"], [1.0])

        pp_result = pp_exec.expectation_value(qc, op, theta=[0.3, 0.7])
        qk_result = qk_exec.expectation_value(qc, op, theta=[0.3, 0.7])

        assert np.isclose(pp_result, qk_result, atol=1e-8)

    def test_statevector_matches_qiskit_executor(self):
        """Statevectors should match between executors."""
        pp_exec = PauliPropagationExecutor()
        qk_exec = QiskitExecutor(seed=0)

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        pp_sv = pp_exec.statevector(qc)
        qk_sv = qk_exec.statevector(qc)

        assert np.allclose(pp_sv, qk_sv, atol=1e-10)

    def test_complex_circuit_matches(self):
        """More complex circuit results should match."""
        pp_exec = PauliPropagationExecutor()
        qk_exec = QiskitExecutor(seed=0)

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        qc.rz(0, 0.5)
        qc.ry(1, 0.3)

        op = QuantumOperator(["ZZI", "IZZ"], [1.0, 0.5])

        pp_result = pp_exec.expectation_value(qc, op)
        qk_result = qk_exec.expectation_value(qc, op)

        assert np.isclose(pp_result, qk_result, atol=1e-8)
