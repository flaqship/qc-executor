import numpy as np
from qiskit.circuit import ParameterVector

from executor import QuantumCircuit
from executor.pennylane.pennylane_circuit import PennyLaneCircuit


class TestPennyLaneCircuit:
    """Test suite for PennyLane circuit conversion."""

    def test_simple_circuit_initialization(self):
        """Test basic circuit initialization."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        plc = PennyLaneCircuit(qc)
        assert plc.num_qubits == 2
        assert plc is not None

    def test_circuit_with_parameters(self):
        """Test circuit with parameters."""
        x = ParameterVector('x', 2)
        qc = QuantumCircuit(2)
        qc.rx(0, x[0])
        qc.ry(1, x[1])

        plc = PennyLaneCircuit(qc)
        assert plc.num_qubits == 2
        assert 'x' in plc.parameter_names
        assert plc.parameter_dimensions['x'] == 2

    def test_circuit_with_multiple_parameter_vectors(self):
        """Test circuit with multiple parameter vectors."""
        x = ParameterVector('x', 2)
        p = ParameterVector('p', 2)

        qc = QuantumCircuit(2)
        qc.rx(0, x[0])
        qc.ry(1, p[0])
        qc.rz(0, x[1])
        qc.rz(1, p[1])

        plc = PennyLaneCircuit(qc)
        assert 'x' in plc.parameter_names
        assert 'p' in plc.parameter_names
        assert plc.parameter_dimensions['x'] == 2
        assert plc.parameter_dimensions['p'] == 2

    def test_circuit_with_parametrized_gate(self):
        """Test circuit with parametrized two-qubit gates."""
        x = ParameterVector('x', 1)
        p = ParameterVector('p', 2)

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cry(0, 1, p[0] * x[0])
        qc.crx(1, 0, p[1] * x[0])

        plc = PennyLaneCircuit(qc)
        assert 'x' in plc.parameter_names
        assert 'p' in plc.parameter_names

    def test_build_pennylane_circuit(self):
        """Test building executable PennyLane circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        plc = PennyLaneCircuit(qc)
        pennylane_circuit = plc.build_pennylane_circuit()

        assert callable(pennylane_circuit)

    def test_get_pennylane_circuit(self):
        """Test get_pennylane_circuit method."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        plc = PennyLaneCircuit(qc)
        pennylane_circuit = plc.get_pennylane_circuit()

        assert callable(pennylane_circuit)

    def test_circuit_hash_property(self):
        """Test that circuit has a hash property."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        plc = PennyLaneCircuit(qc)
        hash_value = plc.hash

        assert hash_value is not None
        assert isinstance(hash_value, int)

    def test_circuit_with_single_qubit_gates(self):
        """Test circuit with various single-qubit gates."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.x(0)
        qc.y(0)
        qc.z(0)
        qc.s(0)
        qc.t(0)

        plc = PennyLaneCircuit(qc)
        assert plc.num_qubits == 1

    def test_circuit_with_two_qubit_gates(self):
        """Test circuit with various two-qubit gates."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cy(0, 1)
        qc.cz(0, 1)
        qc.swap(0, 1)

        plc = PennyLaneCircuit(qc)
        assert plc.num_qubits == 2

    def test_circuit_with_rotation_gates(self):
        """Test circuit with rotation gates."""
        theta = np.pi / 4
        qc = QuantumCircuit(1)
        qc.rx(0, theta)
        qc.ry(0, theta)
        qc.rz(0, theta)

        plc = PennyLaneCircuit(qc)
        assert plc.num_qubits == 1

    def test_circuit_with_controlled_rotation_gates(self):
        """Test circuit with controlled rotation gates."""
        theta = np.pi / 4
        qc = QuantumCircuit(2)
        qc.crx(0, 1, theta)
        qc.cry(0, 1, theta)
        qc.crz(0, 1, theta)

        plc = PennyLaneCircuit(qc)
        assert plc.num_qubits == 2

    def test_circuit_with_phase_gates(self):
        """Test circuit with phase gates."""
        theta = np.pi / 4
        qc = QuantumCircuit(2)
        qc.p(0, theta)
        qc.cp(0, 1, theta)

        plc = PennyLaneCircuit(qc)
        assert plc.num_qubits == 2

    def test_circuit_with_parameter_expression(self):
        """Test circuit with parameter expressions."""
        x = ParameterVector('x', 1)
        qc = QuantumCircuit(1)
        qc.rx(0, 2 * x[0])
        qc.ry(0, x[0] + 0.5)

        plc = PennyLaneCircuit(qc)
        assert 'x' in plc.parameter_names

    def test_circuit_num_qubits_property(self):
        """Test num_qubits property."""
        for n_qubits in [1, 2, 3, 5]:
            qc = QuantumCircuit(n_qubits)
            plc = PennyLaneCircuit(qc)
            assert plc.num_qubits == n_qubits

    def test_circuit_parameter_names_property(self):
        """Test parameter_names property."""
        x = ParameterVector('x', 2)
        y = ParameterVector('y', 3)

        qc = QuantumCircuit(2)
        qc.rx(0, x[0])
        qc.ry(1, y[0])

        plc = PennyLaneCircuit(qc)
        assert 'x' in plc.parameter_names
        assert 'y' in plc.parameter_names

    def test_circuit_parameter_dimensions_property(self):
        """Test parameter_dimensions property."""
        x = ParameterVector('x', 3)

        qc = QuantumCircuit(2)
        qc.rx(0, x[0])
        qc.ry(1, x[1])
        qc.rz(0, x[2])

        plc = PennyLaneCircuit(qc)
        assert plc.parameter_dimensions['x'] == 3

    def test_circuit_call_method(self):
        """Test that circuit can be called directly."""
        x = ParameterVector('x', 1)
        qc = QuantumCircuit(1)
        qc.rx(0, x[0])

        plc = PennyLaneCircuit(qc)
        plc._pennylane_circuit = plc.build_pennylane_circuit()

        # Call should work (though may not produce output without device)
        assert callable(plc)


    def test_empty_circuit(self):
        """Test empty circuit."""
        qc = QuantumCircuit(2)

        plc = PennyLaneCircuit(qc)
        assert plc.num_qubits == 2
        assert len(plc.parameter_names) == 0

    def test_circuit_with_no_parameters(self):
        """Test circuit without parameters."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.x(1)

        plc = PennyLaneCircuit(qc)
        assert len(plc.parameter_names) == 0
        assert len(plc.parameter_dimensions) == 0
