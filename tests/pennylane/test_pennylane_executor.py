import numpy as np
import pytest
from qiskit.circuit import ParameterVector

from executor import QuantumCircuit, QuantumOperator
from executor.pennylane.pennylane_executor import PennylaneExecutor


class TestPennylaneExecutor:
    """Test suite for PennyLane executor."""

    def test_executor_initialization(self):
        """Test basic executor initialization."""
        executor = PennylaneExecutor()
        assert executor is not None
        assert executor.shots is None

    def test_executor_initialization_with_shots(self):
        """Test executor initialization with shots."""
        executor = PennylaneExecutor(shots=1000)
        assert executor.shots == 1000

    def test_executor_initialization_with_seed(self):
        """Test executor initialization with seed."""
        executor = PennylaneExecutor(seed=42)
        assert executor is not None

    def test_expectation_value_simple_circuit(self):
        """Test expectation value calculation with simple circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        op = QuantumOperator(["ZI", "IZ"], [1.0, 1.0])

        executor = PennylaneExecutor()
        result = executor.expectation_value(qc, op)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 0.0, atol=1e-5)

    def test_expectation_value_with_parameters(self):
        """Test expectation value with parametrized circuit."""
        x = ParameterVector('x', 1)
        qc = QuantumCircuit(1)
        qc.rx(0, x[0])

        op = QuantumOperator(["Z"], [1.0])

        executor = PennylaneExecutor()
        result = executor.expectation_value(qc, op, x=[np.pi])

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, -1.0, atol=1e-5)

    def test_expectation_value_with_multiple_parameters(self):
        """Test expectation value with multiple parameters."""
        x = ParameterVector('x', 2)
        qc = QuantumCircuit(2)
        qc.rx(0, x[0])
        qc.ry(1, x[1])

        op = QuantumOperator(["ZZ"], [1.0])

        executor = PennylaneExecutor()
        result = executor.expectation_value(qc, op, x=[0.0, 0.0])

        assert isinstance(result, (float, np.ndarray))
        # With x=[0,0], both qubits are in |0>, so <ZZ> = 1
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_with_parametrized_observable(self):
        """Test expectation value with parametrized observable."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        pop = ParameterVector('pop', 2)
        op = QuantumOperator(["ZI", "IZ"], [pop[0], pop[1]])

        executor = PennylaneExecutor()
        result = executor.expectation_value(qc, op, pop=[0.5, 0.5])

        assert isinstance(result, (float, np.ndarray))

    def test_expectation_value_with_circuit_and_observable_parameters(self):
        """Test expectation value with both circuit and observable parameters."""
        x = ParameterVector('x', 1)
        qc = QuantumCircuit(1)
        qc.rx(0, x[0])

        pop = ParameterVector('pop', 1)
        op = QuantumOperator(["Z"], [pop[0]])

        executor = PennylaneExecutor()
        result = executor.expectation_value(qc, op, x=[0.0], pop=[1.0])

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_missing_parameter_error(self):
        """Test that missing parameter raises error."""
        x = ParameterVector('x', 1)
        qc = QuantumCircuit(1)
        qc.rx(0, x[0])

        op = QuantumOperator(["Z"], [1.0])

        executor = PennylaneExecutor()
        with pytest.raises(ValueError, match="Parameter 'x' not found"):
            executor.expectation_value(qc, op)

    def test_expectation_value_bell_state(self):
        """Test expectation value for Bell state."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        op = QuantumOperator(["ZZ"], [1.0])

        executor = PennylaneExecutor()
        result = executor.expectation_value(qc, op)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_x_basis(self):
        """Test expectation value in X basis."""
        qc = QuantumCircuit(1)
        qc.h(0)

        op = QuantumOperator(["X"], [1.0])

        executor = PennylaneExecutor()
        result = executor.expectation_value(qc, op)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_expectation_value_y_basis(self):
        """Test expectation value in Y basis."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.s(0)

        op = QuantumOperator(["Y"], [1.0])

        executor = PennylaneExecutor()
        result = executor.expectation_value(qc, op)

        assert isinstance(result, (float, np.ndarray))
        assert np.isclose(result, 1.0, atol=1e-5)

    def test_sample_simple_circuit(self):
        """Test sampling from simple circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        executor = PennylaneExecutor(shots=1000, seed=42)
        result = executor.sample(qc)

        # Result is a list with one dict when no parameters
        assert isinstance(result, list)
        assert len(result) == 1
        samples = result[0]
        assert isinstance(samples, dict)
        assert len(samples) > 0
        assert '00' in samples or '11' in samples

    def test_sample_with_parameters(self):
        """Test sampling with parametrized circuit."""
        x = ParameterVector('x', 1)
        qc = QuantumCircuit(2)
        qc.rx(0, x[0])

        executor = PennylaneExecutor(shots=1000, seed=42)
        result = executor.sample(qc, x=[np.pi])

        assert isinstance(result, list)
        assert len(result) == 1
        samples = result[0]
        assert isinstance(samples, dict)
        # After RX(pi) on qubit 0, state should be |10> which is '01' in little-endian
        assert '01' in samples
        # Most samples should be in this state
        assert samples.get('01', 0) > 900

    def test_sample_missing_parameter_error(self):
        """Test that missing parameter raises error in sampling."""
        x = ParameterVector('x', 1)
        qc = QuantumCircuit(1)
        qc.rx(0, x[0])

        executor = PennylaneExecutor(shots=1000)
        with pytest.raises(ValueError, match="Parameter 'x' not found"):
            executor.sample(qc)

    def test_sample_deterministic_circuit(self):
        """Test sampling from deterministic circuit."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(1)

        executor = PennylaneExecutor(shots=100, seed=42)
        result = executor.sample(qc)

        assert isinstance(result, list)
        assert len(result) == 1
        samples = result[0]
        assert isinstance(samples, dict)
        assert '11' in samples
        assert samples['11'] == 100

    def test_statevector_simple_circuit(self):
        """Test statevector calculation with simple circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        executor = PennylaneExecutor()
        statevector = executor.statevector(qc)

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 4
        # Bell state: (|00> + |11>)/sqrt(2)
        assert np.isclose(abs(statevector[0]), 1 / np.sqrt(2), atol=1e-5)
        assert np.isclose(abs(statevector[3]), 1 / np.sqrt(2), atol=1e-5)

    def test_statevector_with_parameters(self):
        """Test statevector with parametrized circuit."""
        x = ParameterVector('x', 1)
        qc = QuantumCircuit(1)
        qc.rx(0, x[0])

        executor = PennylaneExecutor()
        statevector = executor.statevector(qc, x=[np.pi / 2])

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 2

    def test_statevector_missing_parameter_error(self):
        """Test that missing parameter raises error in statevector."""
        x = ParameterVector('x', 1)
        qc = QuantumCircuit(1)
        qc.rx(0, x[0])

        executor = PennylaneExecutor()
        with pytest.raises(ValueError, match="Parameter 'x' not found"):
            executor.statevector(qc)

    def test_statevector_ground_state(self):
        """Test statevector for ground state."""
        qc = QuantumCircuit(2)

        executor = PennylaneExecutor()
        statevector = executor.statevector(qc)

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 4
        assert np.isclose(abs(statevector[0]), 1.0, atol=1e-5)
        assert np.isclose(abs(statevector[1]), 0.0, atol=1e-5)
        assert np.isclose(abs(statevector[2]), 0.0, atol=1e-5)
        assert np.isclose(abs(statevector[3]), 0.0, atol=1e-5)

    def test_statevector_excited_state(self):
        """Test statevector for excited state."""
        qc = QuantumCircuit(1)
        qc.x(0)

        executor = PennylaneExecutor()
        statevector = executor.statevector(qc)

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 2
        assert np.isclose(abs(statevector[0]), 0.0, atol=1e-5)
        assert np.isclose(abs(statevector[1]), 1.0, atol=1e-5)

    def test_statevector_superposition(self):
        """Test statevector for superposition state."""
        qc = QuantumCircuit(1)
        qc.h(0)

        executor = PennylaneExecutor()
        statevector = executor.statevector(qc)

        assert isinstance(statevector, np.ndarray)
        assert len(statevector) == 2
        assert np.isclose(abs(statevector[0]), 1 / np.sqrt(2), atol=1e-5)
        assert np.isclose(abs(statevector[1]), 1 / np.sqrt(2), atol=1e-5)

    def test_statevector_normalization(self):
        """Test that statevector is normalized."""
        x = ParameterVector('x', 2)
        qc = QuantumCircuit(2)
        qc.rx(0, x[0])
        qc.ry(1, x[1])

        executor = PennylaneExecutor()
        statevector = executor.statevector(qc, x=[0.5, 0.3])

        assert isinstance(statevector, np.ndarray)
        norm = np.sum(np.abs(statevector) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_expectation_value_derivatives_simple(self):
        """Test expectation value derivatives calculation."""
        x = ParameterVector('x', 1)
        qc = QuantumCircuit(1)
        qc.rx(0, x[0])

        op = QuantumOperator(["Z"], [1.0])

        executor = PennylaneExecutor()
        result = executor.expectation_value_derivatives(qc, op, 'x', x=[0.0])

        assert isinstance(result, (float, np.ndarray))

    def test_expectation_value_derivatives_with_param_vector(self):
        """Test derivatives with ParameterVector."""
        x = ParameterVector('x', 2)
        qc = QuantumCircuit(2)
        qc.rx(0, x[0])
        qc.ry(1, x[1])

        op = QuantumOperator(["ZI"], [1.0])

        executor = PennylaneExecutor()
        # Use string with index instead of ParameterVectorElement
        result = executor.expectation_value_derivatives(qc, op, 'x[0]', x=[0.0, 0.0])

        assert isinstance(result, (float, np.ndarray))

    def test_expectation_value_derivatives_multiple_values(self):
        """Test derivatives with multiple values."""
        x = ParameterVector('x', 1)
        qc = QuantumCircuit(1)
        qc.rx(0, x[0])

        op = QuantumOperator(["Z"], [1.0])

        executor = PennylaneExecutor()
        result = executor.expectation_value_derivatives(
            qc, op, 'expectation_value', 'x', x=[0.0]
        )

        assert isinstance(result, dict)
        assert 'expectation_value' in result or 'x' in result

    def test_expectation_value_derivatives_missing_parameter_error(self):
        """Test that missing parameter raises error in derivatives."""
        x = ParameterVector('x', 1)
        qc = QuantumCircuit(1)
        qc.rx(0, x[0])

        op = QuantumOperator(["Z"], [1.0])

        executor = PennylaneExecutor()
        with pytest.raises(ValueError, match="Parameter 'x' not found"):
            executor.expectation_value_derivatives(qc, op, 'x')

    def test_remote_property(self):
        """Test remote property."""
        executor = PennylaneExecutor()
        assert executor.remote == False

    def test_circuit_caching(self):
        """Test that circuits are cached."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        executor = PennylaneExecutor()

        # First call should cache
        executor._preprocess_circuits(qc)
        assert qc in executor._circuit_cache

        # Second call should use cache
        cached_circuits, _ = executor._preprocess_circuits(qc)
        assert len(cached_circuits) == 1

    def test_operator_caching(self):
        """Test that operators are cached."""
        op = QuantumOperator(["ZI"], [1.0])

        executor = PennylaneExecutor()

        # First call should cache
        executor._preprocess_operators(op)
        assert op in executor._operator_cache

        # Second call should use cache
        cached_ops, _ = executor._preprocess_operators(op)
        assert len(cached_ops) == 1

    def test_expectation_value_with_complex_circuit(self):
        """Test expectation value with more complex circuit."""
        x = ParameterVector('x', 2)
        p = ParameterVector('p', 2)
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cry(0, 1, p[0] * x[0])
        qc.crx(1, 0, p[1] * x[0])

        pop = ParameterVector('pop', 2)
        op = QuantumOperator(["ZI", "IZ"], [pop[0], pop[1]])

        executor = PennylaneExecutor()
        result = executor.expectation_value(
            qc, op, x=[0.1], p=[0.3, 0.5], pop=[0.5, 0.6]
        )

        assert isinstance(result, (float, np.ndarray))

    def test_expectation_value_three_qubit_circuit(self):
        """Test expectation value with three-qubit circuit."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)

        op = QuantumOperator(["ZZZ"], [1.0])

        executor = PennylaneExecutor()
        result = executor.expectation_value(qc, op)

        assert isinstance(result, (float, np.ndarray))
