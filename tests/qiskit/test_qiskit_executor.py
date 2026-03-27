import logging

import numpy as np
import pytest
from qiskit.circuit import Parameter, ParameterVector
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

from executor import QuantumCircuit
from executor.qiskit.qiskit_circuit import QiskitCircuit
from executor.qiskit.qiskit_executor import QiskitExecutor
from executor.quantum_operator import QuantumOperator


def _build_circuit(num_qubits, operations):
    qc = QuantumCircuit(num_qubits)
    for gate_name, gate_args in operations:
        getattr(qc, gate_name)(*gate_args)
    return qc


class TestQiskitExecutor:

    def test_init_default(self):
        """Test executor initialization with default parameters."""
        executor = QiskitExecutor()
        assert isinstance(executor, QiskitExecutor)
        assert executor.shots is None
        assert executor.remote is False

    def test_init_with_shots(self):
        """Test executor initialization with shots parameter."""
        executor = QiskitExecutor(shots=1024)
        assert executor.shots == 1024

    def test_init_with_seed(self):
        """Test executor initialization with seed parameter."""
        executor = QiskitExecutor(seed=42)
        assert isinstance(executor, QiskitExecutor)
        assert executor.shots is None

    def test_init_with_all_params(self):
        """Test executor initialization with all parameters."""
        executor = QiskitExecutor(
            shots=2048,
            log_file="test.log",
            caching=True,
            cache_dir="/tmp/cache",
        )
        assert executor.shots == 2048

    def test_init_with_injected_estimator_autocreates_sampler(self):
        estimator = StatevectorEstimator()
        executor = QiskitExecutor(backend=estimator, shots=1024)

        assert executor._estimator is estimator
        assert isinstance(executor._sampler, StatevectorSampler)

    def test_init_with_injected_sampler_autocreates_estimator(self):
        sampler = StatevectorSampler()
        executor = QiskitExecutor(backend=sampler, shots=1024)

        assert executor._sampler is sampler
        assert isinstance(executor._estimator, StatevectorEstimator)

    def test_init_with_primitive_and_options_raises(self):
        with pytest.raises(ValueError, match="cannot be combined with injected"):
            QiskitExecutor(backend=StatevectorEstimator(), options={"resilience_level": 1})

    # ========================================================================
    # Logging Tests
    # ========================================================================

    def test_logging_default_level(self):
        """Test that default logging level is WARNING."""
        executor = QiskitExecutor()
        assert executor._logger.level == logging.WARNING

    def test_logging_info_level(self):
        """Test that INFO logging level is set correctly."""
        executor = QiskitExecutor(log_level="INFO")
        assert executor._logger.level == logging.INFO

    def test_logging_debug_level(self):
        """Test that DEBUG logging level is set correctly."""
        executor = QiskitExecutor(log_level="DEBUG")
        assert executor._logger.level == logging.DEBUG

    def test_logging_invalid_level_raises(self):
        """Test that an invalid log_level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log_level"):
            QiskitExecutor(log_level="VERBOSE")

    def test_logging_to_file(self, tmp_path):
        """Test that log messages are written to the specified log file."""
        log_file = str(tmp_path / "qiskit_executor.log")
        executor = QiskitExecutor(log_level="INFO", log_file=log_file)
        executor._logger.info("qiskit test log message")

        with open(log_file) as f:
            content = f.read()
        assert "qiskit test log message" in content

        for handler in executor._logger.handlers[:]:
            handler.close()
            executor._logger.removeHandler(handler)

    def test_logging_centralized_expectation_value(self, tmp_path):
        """Test that expectation_value emits an INFO log via the base class."""
        log_file = str(tmp_path / "qiskit_ev.log")
        executor = QiskitExecutor(log_level="INFO", log_file=log_file)

        qc = _build_circuit(1, [("h", [0])])
        op = QuantumOperator(["Z"], [1.0])
        executor.expectation_value(qc, op)

        with open(log_file) as f:
            content = f.read()
        assert "Computing expectation value" in content

        for handler in executor._logger.handlers[:]:
            handler.close()
            executor._logger.removeHandler(handler)

    def test_logging_centralized_statevector(self, tmp_path):
        """Test that statevector emits an INFO log via the base class."""
        log_file = str(tmp_path / "qiskit_sv.log")
        executor = QiskitExecutor(log_level="INFO", log_file=log_file)

        qc = _build_circuit(1, [])
        executor.statevector(qc)

        with open(log_file) as f:
            content = f.read()
        assert "Computing statevector" in content

        for handler in executor._logger.handlers[:]:
            handler.close()
            executor._logger.removeHandler(handler)

    # ========================================================================
    # Cache Size Tests
    # ========================================================================

    def test_max_cache_size_accepted(self):
        """Test that max_cache_size parameter is accepted."""
        executor = QiskitExecutor(max_cache_size=64)
        assert executor._max_cache_size == 64

    def test_unlimited_cache_size_by_default(self):
        """Test that cache is unlimited when max_cache_size is not specified."""
        executor = QiskitExecutor()
        assert executor._max_cache_size is None

    def test_expectation_value_bell_state_z_basis(self):
        """Test expectation value of Bell state with Z operators."""
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        op = QuantumOperator(["ZI", "IZ"], [1.0, 1.0])

        executor = QiskitExecutor()
        result = executor.expectation_value(qc, op)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 0.0, atol=1e-5)

    def test_expectation_value_bell_state_zz(self):
        """Test expectation value of Bell state with ZZ operator."""
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        op = QuantumOperator(["ZZ"], [1.0])

        executor = QiskitExecutor()
        result = executor.expectation_value(qc, op)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_hadamard_x_basis(self):
        """Test expectation value of Hadamard state with X operator."""
        qc = _build_circuit(1, [("h", [0])])
        op = QuantumOperator(["X"], [1.0])

        executor = QiskitExecutor()
        result = executor.expectation_value(qc, op)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_y_basis(self):
        """Test expectation value in Y basis (H followed by S gate)."""
        qc = _build_circuit(1, [("h", [0]), ("s", [0])])
        op = QuantumOperator(["Y"], [1.0])

        executor = QiskitExecutor()
        result = executor.expectation_value(qc, op)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_with_circuit_parameter(self):
        """Test expectation value with parametric circuit (RX gate)."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])

        executor = QiskitExecutor()
        result = executor.expectation_value(qc, op, x=[np.pi])

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, -1.0, atol=1e-5)

    def test_expectation_value_with_multiple_circuit_parameters(self):
        """Test expectation value with multiple circuit parameters."""
        x = ParameterVector("x", 2)
        qc = _build_circuit(2, [("rx", [0, x[0]]), ("ry", [1, x[1]])])
        op = QuantumOperator(["ZZ"], [1.0])

        executor = QiskitExecutor()
        result = executor.expectation_value(qc, op, x=[0.0, 0.0])

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_with_observable_parameters(self):
        """Test expectation value with parametric observable."""
        pop = ParameterVector("pop", 2)
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        op = QuantumOperator(["ZI", "IZ"], [pop[0], pop[1]])

        executor = QiskitExecutor()
        result = executor.expectation_value(qc, op, pop=[0.5, 0.5])

        assert isinstance(result, (float, np.ndarray))

    def test_expectation_value_with_circuit_and_observable_parameters(self):
        """Test expectation value with both circuit and observable parameters."""
        x = ParameterVector("x", 1)
        pop = ParameterVector("pop", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        op = QuantumOperator(["Z"], [pop[0]])

        executor = QiskitExecutor()
        result = executor.expectation_value(qc, op, x=[0.0], pop=[1.0])

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_three_qubit_chain(self):
        """Test expectation value with three-qubit GHZ-type state."""
        qc = _build_circuit(3, [("h", [0]), ("cx", [0, 1]), ("cx", [1, 2])])
        op = QuantumOperator(["ZZZ"], [1.0])

        executor = QiskitExecutor()
        result = executor.expectation_value(qc, op)

        assert isinstance(result, (float, np.ndarray))

    def test_sample_bell_state(self):
        """Test sampling from Bell state (should get 00 and 11)."""
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])

        executor = QiskitExecutor(shots=1000, seed=42)
        result = executor.sample(qc)

        assert isinstance(result, dict)
        # Bell state should have 00 or 11 outcomes
        assert any(bit in result for bit in ["00", "11"])

    def test_sample_x_gate(self):
        """Test sampling after X gate (should get all 1s)."""
        qc = _build_circuit(2, [("x", [0]), ("x", [1])])

        executor = QiskitExecutor(shots=100, seed=42)
        result = executor.sample(qc)

        assert isinstance(result, dict)
        assert "11" in result
        assert result["11"] == 100

    def test_sample_with_parameter(self):
        """Test sampling with parametric circuit."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(2, [("rx", [0, x[0]])])

        executor = QiskitExecutor(shots=1000, seed=42)
        result = executor.sample(qc, x=[np.pi])

        assert isinstance(result, dict)
        # After RX(pi), qubit 0 should be flipped
        assert "10" in result
        assert result["10"] >= 900  # Should have high count

    def test_sample_hadamard(self):
        """Test sampling from Hadamard state."""
        # Use 2 qubits to avoid scalar sample issues
        qc = _build_circuit(2, [("h", [0])])

        executor = QiskitExecutor(shots=1000, seed=42)
        result = executor.sample(qc)

        assert isinstance(result, dict)
        assert len(result) > 0
        # Total counts should equal shots
        total_counts = sum(result.values())
        assert total_counts == 1000

    def test_statevector_empty_circuit(self):
        """Test statevector of empty circuit (should be |00...0>)."""
        qc = _build_circuit(2, [])

        executor = QiskitExecutor()
        statevector = executor.statevector(qc)

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 4
        # Should be |00> state: [1, 0, 0, 0]
        assert np.isclose(abs(statevector[0]), 1.0, atol=1e-5)
        assert np.allclose(abs(statevector[1:]), 0.0, atol=1e-5)

    def test_statevector_x_gate(self):
        """Test statevector after X gate (should be |1>)."""
        qc = _build_circuit(1, [("x", [0])])

        executor = QiskitExecutor()
        statevector = executor.statevector(qc)

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 2
        # Should be |1> state: [0, 1]
        assert np.isclose(abs(statevector[1]), 1.0, atol=1e-5)

    def test_statevector_hadamard(self):
        """Test statevector of Hadamard state (should be equal superposition)."""
        qc = _build_circuit(1, [("h", [0])])

        executor = QiskitExecutor()
        statevector = executor.statevector(qc)

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 2
        # Should be (|0> + |1>)/sqrt(2)
        assert np.isclose(abs(statevector[0]), 1 / np.sqrt(2), atol=1e-5)
        assert np.isclose(abs(statevector[1]), 1 / np.sqrt(2), atol=1e-5)

    def test_statevector_bell_state(self):
        """Test statevector of Bell state."""
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])

        executor = QiskitExecutor()
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

        executor = QiskitExecutor()
        statevector = executor.statevector(qc, x=[np.pi / 2])

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 2
        # Statevector should be normalized
        assert np.isclose(np.sum(np.abs(statevector) ** 2), 1.0, atol=1e-5)

    def test_statevector_with_multiple_parameters(self):
        """Test statevector with multiple parameters."""
        x = ParameterVector("x", 2)
        qc = _build_circuit(2, [("rx", [0, x[0]]), ("ry", [1, x[1]])])

        executor = QiskitExecutor()
        statevector = executor.statevector(qc, x=[0.5, 0.3])

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 4
        # Statevector should be normalized
        assert np.isclose(np.sum(np.abs(statevector) ** 2), 1.0, atol=1e-5)

    def test_expectation_value_derivatives_single_parameter(self):
        """Test derivative with respect to a single parameter."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])

        executor = QiskitExecutor()
        result = executor.expectation_value_derivatives(qc, op, "x", x=[0.0])

        assert isinstance(result, (float, np.ndarray))

    def test_expectation_value_derivatives_indexed_parameter(self):
        """Test derivative with respect to indexed parameter (e.g., x[0])."""
        x = ParameterVector("x", 2)
        qc = _build_circuit(2, [("rx", [0, x[0]]), ("ry", [1, x[1]])])
        op = QuantumOperator(["ZI"], [1.0])

        executor = QiskitExecutor()
        result = executor.expectation_value_derivatives(qc, op, "x[0]", x=[0.0, 0.0])

        assert isinstance(result, (float, np.ndarray))

    def test_expectation_value_derivatives_multiple_values(self):
        """Test requesting multiple derivatives (expectation value and parameter)."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])

        executor = QiskitExecutor()
        result = executor.expectation_value_derivatives(qc, op, "expectation_value", "x", x=[0.0])

        assert isinstance(result, dict)
        assert "expectation_value" in result or "x" in result

    def test_expectation_value_derivatives_known_value(self):
        """Test derivative computation with known analytical result."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])

        executor = QiskitExecutor()
        derivative = executor.expectation_value_derivatives(qc, op, "x", x=[0.0])

        assert isinstance(derivative, (float, np.ndarray))
        # Derivative should be close to 0 at x=0
        assert np.isclose(derivative, 0.0, atol=1e-5)

    def test_linear_combination_operator(self):
        executor = QiskitExecutor()

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        operator = QuantumOperator(["ZZ", "XX"], [0.5, 0.5], 2)

        result = executor.expectation_value(QiskitCircuit(circuit), operator)

        assert np.allclose(result, 1.0)


# =============================================================================
# Parameter vs ParameterVector compatibility tests
#
# These tests verify that _build_param_dict, _prepare_parameter_dicts and the
# derivative machinery handle both standalone ``Parameter`` objects (no .vector /
# .index attributes) and ``ParameterVector`` elements (with .vector.name and
# .index) correctly.
# =============================================================================


class TestStandaloneParameterSupport:
    """Tests for circuits built with standalone Parameter (not ParameterVector)."""

    def test_expectation_value_standalone_parameter(self):
        """expectation_value works with a standalone Parameter (no .vector)."""
        theta = Parameter("theta")
        qc = _build_circuit(1, [("ry", [0, theta])])
        op = QuantumOperator(["Z"], [1.0])

        executor = QiskitExecutor()
        # RY(0)|0> = |0>, <Z> = 1
        result = executor.expectation_value(qc, op, theta=0.0)

        assert isinstance(result, (float, np.floating, np.ndarray))
        assert np.isclose(float(np.real(result)), 1.0, atol=1e-5)

    def test_expectation_value_standalone_parameter_pi(self):
        """RY(pi)|0> = |1>, <Z> = -1 with a standalone Parameter."""
        theta = Parameter("theta")
        qc = _build_circuit(1, [("ry", [0, theta])])
        op = QuantumOperator(["Z"], [1.0])

        executor = QiskitExecutor()
        result = executor.expectation_value(qc, op, theta=np.pi)

        assert np.isclose(float(np.real(result)), -1.0, atol=1e-5)

    def test_statevector_standalone_parameter(self):
        """statevector works with a standalone Parameter."""
        theta = Parameter("theta")
        qc = _build_circuit(1, [("rx", [0, theta])])

        executor = QiskitExecutor()
        sv = executor.statevector(qc, theta=0.0)

        assert isinstance(sv, np.ndarray)
        assert np.isclose(np.sum(np.abs(sv) ** 2), 1.0, atol=1e-5)
        # RX(0)|0> = |0>
        assert np.isclose(abs(sv[0]), 1.0, atol=1e-5)

    def test_sample_standalone_parameter(self):
        """sample works with a standalone Parameter."""
        theta = Parameter("theta")
        qc = _build_circuit(1, [("rx", [0, theta])])

        executor = QiskitExecutor(shots=500, seed=0)
        result = executor.sample(qc, theta=np.pi)

        assert isinstance(result, dict)
        # RX(pi)|0> = |1> → all counts in "1"
        assert "1" in result
        assert result["1"] >= 450

    def test_multiple_standalone_parameters(self):
        """Multiple standalone Parameters can be passed as separate kwargs."""
        alpha = Parameter("alpha")
        beta = Parameter("beta")
        qc = _build_circuit(2, [("ry", [0, alpha]), ("ry", [1, beta])])
        op = QuantumOperator(["ZZ"], [1.0])

        executor = QiskitExecutor()
        # RY(0)⊗RY(0)|00> = |00>, <ZZ> = 1
        result = executor.expectation_value(qc, op, alpha=0.0, beta=0.0)

        assert np.isclose(float(np.real(result)), 1.0, atol=1e-5)


class TestParameterVectorSupport:
    """Tests confirming ParameterVector behaviour is unchanged."""

    def test_expectation_value_parameter_vector(self):
        """expectation_value works with a ParameterVector (baseline check)."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])

        executor = QiskitExecutor()
        result = executor.expectation_value(qc, op, x=[0.0])

        assert np.isclose(result, 1.0, atol=1e-5)

    def test_statevector_parameter_vector(self):
        """statevector works with a ParameterVector."""
        x = ParameterVector("x", 2)
        qc = _build_circuit(2, [("rx", [0, x[0]]), ("ry", [1, x[1]])])

        executor = QiskitExecutor()
        sv = executor.statevector(qc, x=[0.0, 0.0])

        assert np.isclose(np.sum(np.abs(sv) ** 2), 1.0, atol=1e-5)

    def test_sample_parameter_vector(self):
        """sample works with a ParameterVector."""
        x = ParameterVector("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])

        executor = QiskitExecutor(shots=500, seed=0)
        result = executor.sample(qc, x=[np.pi])

        assert isinstance(result, dict)
        assert "1" in result
        assert result["1"] >= 450

    def test_parameter_vector_index_binding(self):
        """ParameterVector elements are bound by index correctly."""
        x = ParameterVector("x", 2)
        qc = _build_circuit(2, [("rx", [0, x[0]]), ("ry", [1, x[1]])])
        op = QuantumOperator(["IZ"], [1.0])

        executor = QiskitExecutor()
        # x[0]=pi → qubit 0 flipped → <IZ> = -1; x[1]=0 → qubit 1 unchanged
        result = executor.expectation_value(qc, op, x=[np.pi, 0.0])

        assert np.isclose(result, -1.0, atol=1e-5)


class TestMixedParameterEdgeCases:
    """Edge-case tests mixing or comparing both parameter styles."""

    def test_standalone_and_parametervector_same_result(self):
        """Standalone Parameter and ParameterVector[0] yield the same expectation value."""
        angle = np.pi / 4

        executor = QiskitExecutor()

        # Circuit A: standalone Parameter
        theta = Parameter("theta")
        qc_a = _build_circuit(1, [("ry", [0, theta])])
        op_a = QuantumOperator(["Z"], [1.0])
        result_a = executor.expectation_value(qc_a, op_a, theta=angle)

        # Circuit B: ParameterVector
        x = ParameterVector("x", 1)
        qc_b = _build_circuit(1, [("ry", [0, x[0]])])
        op_b = QuantumOperator(["Z"], [1.0])
        result_b = executor.expectation_value(qc_b, op_b, x=[angle])

        assert np.isclose(float(np.real(result_a)), float(np.real(result_b)), atol=1e-5)

    def test_scalar_value_for_standalone_parameter(self):
        """A scalar (non-list) value can be passed for a standalone Parameter."""
        theta = Parameter("theta")
        qc = _build_circuit(1, [("ry", [0, theta])])
        op = QuantumOperator(["Z"], [1.0])

        executor = QiskitExecutor()
        result = executor.expectation_value(qc, op, theta=0.0)

        assert np.isclose(float(np.real(result)), 1.0, atol=1e-5)

    def test_list_value_for_standalone_parameter(self):
        """A single-element list is also accepted for a standalone Parameter."""
        theta = Parameter("theta")
        qc = _build_circuit(1, [("ry", [0, theta])])
        op = QuantumOperator(["Z"], [1.0])

        executor = QiskitExecutor()
        result = executor.expectation_value(qc, op, theta=[0.0])

        assert np.isclose(float(np.real(result)), 1.0, atol=1e-5)
