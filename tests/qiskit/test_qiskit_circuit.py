import numpy as np
import pytest
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.circuit import ParameterVector

from executor import QuantumCircuit
from executor.qiskit import QiskitCircuit, transpile_circuit


def _make_parametrized_circuit(length=3, vector_name="vec"):
    """Create a real Qiskit circuit that uses a ParameterVector of given length."""
    vec = ParameterVector(vector_name, length)
    qc = QuantumCircuit(1)
    for i in range(length):
        qc.ry(0, vec[i])
    return qc


class TestQiskitCircuit:
    def test_empty_circuit(self):
        """Test conversion of an empty circuit."""
        qc = QuantumCircuit(2)
        qiskit_circ = QiskitCircuit(qc)

        assert qiskit_circ.num_qubits == 2
        assert len(qiskit_circ.parameter_names) == 0
        assert isinstance(qiskit_circ.hash, int)

    def test_single_hadamard_gate(self):
        """Test circuit with a single Hadamard gate."""
        qc = QuantumCircuit(1)
        qc.h(0)

        qiskit_circ = QiskitCircuit(qc)
        assert qiskit_circ.num_qubits == 1
        assert len(qiskit_circ.parameter_names) == 0

    def test_bell_state_circuit(self):
        """Test Bell state preparation: H(0), CNOT(0,1)."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        qiskit_circ = QiskitCircuit(qc)
        assert qiskit_circ.num_qubits == 2
        assert len(qiskit_circ.parameter_names) == 0

    @pytest.mark.parametrize("gate_name", ["x", "y", "z", "s", "t"])
    def test_single_qubit_gates(self, gate_name):
        """Test individual single-qubit Pauli and phase gates."""
        qc = QuantumCircuit(1)
        getattr(qc, gate_name)(0)

        qiskit_circ = QiskitCircuit(qc)
        assert qiskit_circ.num_qubits == 1
        assert len(qiskit_circ.parameter_names) == 0

    @pytest.mark.parametrize(
        "gate_name,num_qubits",
        [
            ("cx", 2),  # CNOT
            ("cy", 2),  # Controlled-Y
            ("cz", 2),  # Controlled-Z
            ("swap", 2),  # SWAP
        ],
    )
    def test_two_qubit_gates(self, gate_name, num_qubits):
        """Test two-qubit gates."""
        qc = QuantumCircuit(num_qubits)
        getattr(qc, gate_name)(0, 1)

        qiskit_circ = QiskitCircuit(qc)
        assert qiskit_circ.num_qubits == num_qubits
        assert len(qiskit_circ.parameter_names) == 0

    def test_multi_qubit_chain(self):
        """Test a chain of CNOT gates across multiple qubits."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)

        qiskit_circ = QiskitCircuit(qc)
        assert qiskit_circ.num_qubits == 4

    @pytest.mark.parametrize(
        "theta,gate_name",
        [
            (np.pi / 4, "rx"),
            (np.pi / 2, "ry"),
            (np.pi, "rz"),
            (0.5, "rx"),
            (2 * np.pi, "ry"),
        ],
    )
    def test_single_qubit_rotation_gates_with_float(self, theta, gate_name):
        """Test single-qubit rotation gates with float angle."""
        qc = QuantumCircuit(1)
        getattr(qc, gate_name)(0, theta)

        qiskit_circ = QiskitCircuit(qc)
        assert qiskit_circ.num_qubits == 1
        # Float parameters are not tracked as parameters
        assert len(qiskit_circ.parameter_names) == 0

    @pytest.mark.parametrize(
        "theta,gate_name",
        [
            (np.pi / 4, "crx"),
            (np.pi / 2, "cry"),
            (np.pi, "crz"),
        ],
    )
    def test_controlled_rotation_gates_with_float(self, theta, gate_name):
        """Test controlled rotation gates with float angle."""
        qc = QuantumCircuit(2)
        getattr(qc, gate_name)(0, 1, theta)

        qiskit_circ = QiskitCircuit(qc)
        assert qiskit_circ.num_qubits == 2
        assert len(qiskit_circ.parameter_names) == 0

    @pytest.mark.parametrize("theta", [np.pi / 4, np.pi / 2, np.pi, 0.25])
    def test_phase_gate_with_float(self, theta):
        """Test phase shift gate with float angle."""
        qc = QuantumCircuit(1)
        qc.p(0, theta)

        qiskit_circ = QiskitCircuit(qc)
        assert qiskit_circ.num_qubits == 1
        assert len(qiskit_circ.parameter_names) == 0

    @pytest.mark.parametrize("theta", [np.pi / 4, np.pi / 2])
    def test_controlled_phase_gate_with_float(self, theta):
        """Test controlled phase shift gate with float angle."""
        qc = QuantumCircuit(2)
        qc.cp(0, 1, theta)

        qiskit_circ = QiskitCircuit(qc)
        assert qiskit_circ.num_qubits == 2
        assert len(qiskit_circ.parameter_names) == 0

    def test_single_parameter(self):
        """Test circuit with a single parameter."""
        x = ParameterVector("x", 1)
        qc = QuantumCircuit(1)
        qc.rx(0, x[0])

        qiskit_circ = QiskitCircuit(qc)
        assert "x" in qiskit_circ.parameter_names
        assert qiskit_circ.parameter_dimensions["x"] == 1

    def test_multiple_parameters_same_vector(self):
        """Test circuit with multiple parameters from the same vector."""
        x = ParameterVector("x", 3)
        qc = QuantumCircuit(3)
        qc.rx(0, x[0])
        qc.ry(1, x[1])
        qc.rz(2, x[2])

        qiskit_circ = QiskitCircuit(qc)
        assert "x" in qiskit_circ.parameter_names
        assert qiskit_circ.parameter_dimensions["x"] == 3

    def test_multiple_parameter_vectors(self):
        """Test circuit with multiple different parameter vectors."""
        x = ParameterVector("x", 2)
        y = ParameterVector("y", 1)
        qc = QuantumCircuit(3)
        qc.rx(0, x[0])
        qc.ry(1, x[1])
        qc.rz(2, y[0])

        qiskit_circ = QiskitCircuit(qc)
        assert "x" in qiskit_circ.parameter_names
        assert "y" in qiskit_circ.parameter_names
        assert qiskit_circ.parameter_dimensions["x"] == 2
        assert qiskit_circ.parameter_dimensions["y"] == 1

    def test_parametric_two_qubit_gates(self):
        """Test two-qubit gates with parameters."""
        theta = ParameterVector("theta", 3)
        qc = QuantumCircuit(2)
        qc.crx(0, 1, theta[0])
        qc.cry(0, 1, theta[1])
        qc.crz(0, 1, theta[2])

        qiskit_circ = QiskitCircuit(qc)
        assert "theta" in qiskit_circ.parameter_names
        assert qiskit_circ.parameter_dimensions["theta"] == 3

    def test_parameter_arithmetic_multiplication(self):
        """Test parameter expression with multiplication: 2 * x[0]."""
        x = ParameterVector("x", 1)
        qc = QuantumCircuit(1)
        qc.rx(0, 2 * x[0])

        qiskit_circ = QiskitCircuit(qc)
        assert "x" in qiskit_circ.parameter_names

    def test_parameter_arithmetic_addition(self):
        """Test parameter expression with addition: x[0] + 0.5."""
        x = ParameterVector("x", 1)
        qc = QuantumCircuit(1)
        qc.rx(0, x[0] + 0.5)

        qiskit_circ = QiskitCircuit(qc)
        assert "x" in qiskit_circ.parameter_names

    def test_parameter_arithmetic_subtraction(self):
        """Test parameter expression with subtraction: x[0] - 0.2."""
        x = ParameterVector("x", 1)
        qc = QuantumCircuit(1)
        qc.rx(0, x[0] - 0.2)

        qiskit_circ = QiskitCircuit(qc)
        assert "x" in qiskit_circ.parameter_names

    def test_parameter_arithmetic_division(self):
        """Test parameter expression with division: x[0] / 2."""
        x = ParameterVector("x", 1)
        qc = QuantumCircuit(1)
        qc.rx(0, x[0] / 2)

        qiskit_circ = QiskitCircuit(qc)
        assert "x" in qiskit_circ.parameter_names

    def test_parameter_multiplication_between_vectors(self):
        """Test parameter expression multiplying two different vectors: x[0] * y[0]."""
        x = ParameterVector("x", 1)
        y = ParameterVector("y", 1)
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.crx(0, 1, x[0] * y[0])

        qiskit_circ = QiskitCircuit(qc)
        assert "x" in qiskit_circ.parameter_names
        assert "y" in qiskit_circ.parameter_names

    def test_complex_parameter_expression(self):
        """Test complex parameter expression: 2 * x[0] - 1."""
        x = ParameterVector("x", 1)
        qc = QuantumCircuit(1)
        qc.rx(0, 2 * x[0] - 1)

        qiskit_circ = QiskitCircuit(qc)
        assert "x" in qiskit_circ.parameter_names

    def test_num_qubits_property(self):
        """Test that num_qubits property returns correct value."""
        qc = QuantumCircuit(5)
        qiskit_circ = QiskitCircuit(qc)
        assert qiskit_circ.num_qubits == 5

    def test_parameter_names_property_empty(self):
        """Test parameter_names property for non-parametric circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qiskit_circ = QiskitCircuit(qc)

        assert isinstance(qiskit_circ.parameter_names, list)
        assert len(qiskit_circ.parameter_names) == 0

    def test_parameter_names_property_with_params(self):
        """Test parameter_names property for parametric circuit."""
        x = ParameterVector("x", 2)
        y = ParameterVector("y", 1)
        qc = QuantumCircuit(2)
        qc.rx(0, x[0])
        qc.ry(1, x[1])
        qc.rz(0, y[0])
        qiskit_circ = QiskitCircuit(qc)

        assert "x" in qiskit_circ.parameter_names
        assert "y" in qiskit_circ.parameter_names

    def test_parameter_dimensions_property(self):
        """Test parameter_dimensions property."""
        x = ParameterVector("x", 3)
        qc = QuantumCircuit(3)
        qc.rx(0, x[0])
        qc.rx(1, x[1])
        qc.rx(2, x[2])
        qiskit_circ = QiskitCircuit(qc)

        assert isinstance(qiskit_circ.parameter_dimensions, dict)
        assert qiskit_circ.parameter_dimensions["x"] == 3

    def test_hash_property(self):
        """Test that hash property returns a valid integer."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qiskit_circ = QiskitCircuit(qc)

        hash_value = qiskit_circ.hash
        assert isinstance(hash_value, int)

    def test_hash_consistency(self):
        """Test that hash remains consistent for the same circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qiskit_circ = QiskitCircuit(qc)

        hash1 = qiskit_circ.hash
        hash2 = qiskit_circ.hash
        assert hash1 == hash2

    @pytest.mark.parametrize(
        "vec_name, vec_len, param_values, expected_unbound_count",
        [
            ("v", 3, {"v": 0.5}, 2),  # scalar binds only first
            ("w", 3, {"w": [0.1, 0.2, 0.3]}, 0),  # list binds all
            ("arr", 4, {"arr": np.array([9.0, 8.0])}, 2),  # numpy array partial bind
            ("more", 3, {"more": [1, 2, 3, 4, 5]}, 0),  # extra values ignored
        ],
    )
    def test_bind_parameters_various_cases(
        self, vec_name, vec_len, param_values, expected_unbound_count
    ):
        """Test bind_parameters with various parameter value formats."""
        qc = _make_parametrized_circuit(vec_len, vec_name)
        qiskit_circ = QiskitCircuit(qc)

        bound = qiskit_circ.bind_parameters(param_values)
        assert len(bound.parameters) == expected_unbound_count

    @pytest.mark.parametrize(
        "param_values, expect_identity",
        [
            ({"does_not_exist": [1, 2]}, True),  # no matching vector -> original circuit
            ({}, True),  # empty dict -> original circuit
            ({"a": [0.1], "nope": [7]}, False),
        ],
    )
    def test_bind_parameters_identity_and_mixed_cases(self, param_values, expect_identity):
        """Test bind_parameters for identity and mixed binding cases."""
        qc = _make_parametrized_circuit(2, "a")
        qiskit_circ = QiskitCircuit(qc)

        original_parameters = set(qc.parameters)
        original_str = str(qc)

        ret = qiskit_circ.bind_parameters(param_values)

        if expect_identity:
            # No parameters should be bound
            assert set(ret.parameters) == original_parameters
            # Circuit structure must be identical
            assert str(ret) == original_str
        else:
            # at least one parameter was bound
            assert len(ret.parameters) < len(original_parameters)


class TestTranspileCircuitQiskit:
    def test_returns_qiskit_quantum_circuit(self):
        """Test that transpile_circuit returns a Qiskit QuantumCircuit."""
        qc = QuantumCircuit(2)
        result = transpile_circuit(qc)
        assert isinstance(result, QiskitQuantumCircuit)

    def test_empty_circuit(self):
        """Test transpile_circuit with an empty circuit."""
        qc = QuantumCircuit(2)
        result = transpile_circuit(qc)
        assert result.num_qubits == 2

    def test_single_gate_circuit(self):
        """Test transpile_circuit with a single Hadamard gate."""
        qc = QuantumCircuit(1)
        qc.h(0)
        result = transpile_circuit(qc)
        assert result.num_qubits == 1

    def test_bell_state_circuit(self):
        """Test transpile_circuit preserves Bell state circuit structure."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = transpile_circuit(qc)
        assert result.num_qubits == 2

    def test_parametrized_circuit_preserves_parameters(self):
        """Test that transpile_circuit preserves circuit parameters."""
        x = ParameterVector("x", 2)
        qc = QuantumCircuit(2)
        qc.rx(0, x[0])
        qc.ry(1, x[1])
        result = transpile_circuit(qc)
        assert len(result.parameters) == 2
