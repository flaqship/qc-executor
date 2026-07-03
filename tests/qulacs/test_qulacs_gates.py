"""Tests for Qulacs gate functions."""

from unittest.mock import Mock

import pytest
from qulacs import ParametricQuantumCircuit  # pylint: disable=no-name-in-module
from qulacs.gate import (  # pylint: disable=no-name-in-module
    CZ,
    RX,
    RY,
    RZ,
    SWAP,
    TOFFOLI,
    U1,
    H,
    Identity,
    S,
    Sdag,
    T,
    Tdag,
    X,
    Y,
    Z,
)

from qc_executor.qulacs.qulacs_gates import (
    qiskit_qulacs_gate_dict,
    qiskit_qulacs_param_gate_dict,
    qulacs_gate_cnot,
    qulacs_gate_cz,
    qulacs_gate_h,
    qulacs_gate_i,
    qulacs_gate_rx,
    qulacs_gate_ry,
    qulacs_gate_rz,
    qulacs_gate_s,
    qulacs_gate_sdg,
    qulacs_gate_swap,
    qulacs_gate_t,
    qulacs_gate_tdg,
    qulacs_gate_toffoli,
    qulacs_gate_u1,
    qulacs_gate_x,
    qulacs_gate_y,
    qulacs_gate_z,
    qulacs_param_gate_rx,
    qulacs_param_gate_ry,
    qulacs_param_gate_rz,
)


@pytest.fixture
def mock_circuit():
    """Fixture providing a fresh mock ParametricQuantumCircuit for each test."""
    circuit = Mock(spec=ParametricQuantumCircuit)
    yield circuit
    # Reset after test
    circuit.reset_mock()


class TestSingleQubitGates:
    """Test single-qubit gate functions (I, H, X, Y, Z, S, T)."""

    def test_qulacs_gate_i_adds_identity_gate(self, mock_circuit):
        """Test that I gate adds Identity gate to circuit."""
        qubit = 0

        qulacs_gate_i(mock_circuit, qubit)

        mock_circuit.add_gate.assert_called_once()
        call_args = mock_circuit.add_gate.call_args[0][0]
        assert isinstance(call_args, type(Identity(0)))

    def test_qulacs_gate_h_adds_hadamard_gate(self, mock_circuit):
        """Test that H gate adds Hadamard gate to circuit."""
        qubit = 0

        qulacs_gate_h(mock_circuit, qubit)

        mock_circuit.add_gate.assert_called_once()
        call_args = mock_circuit.add_gate.call_args[0][0]
        assert isinstance(call_args, type(H(0)))

    def test_qulacs_gate_x_adds_pauli_x_gate(self, mock_circuit):
        """Test that X gate adds Pauli-X gate to circuit."""
        qubit = 2

        qulacs_gate_x(mock_circuit, qubit)

        mock_circuit.add_gate.assert_called_once()
        call_args = mock_circuit.add_gate.call_args[0][0]
        assert isinstance(call_args, type(X(0)))

    def test_qulacs_gate_y_adds_pauli_y_gate(self, mock_circuit):
        """Test that Y gate adds Pauli-Y gate to circuit."""
        qubit = 1

        qulacs_gate_y(mock_circuit, qubit)

        mock_circuit.add_gate.assert_called_once()
        call_args = mock_circuit.add_gate.call_args[0][0]
        assert isinstance(call_args, type(Y(0)))

    def test_qulacs_gate_z_adds_pauli_z_gate(self, mock_circuit):
        """Test that Z gate adds Pauli-Z gate to circuit."""
        qubit = 3

        qulacs_gate_z(mock_circuit, qubit)

        mock_circuit.add_gate.assert_called_once()
        call_args = mock_circuit.add_gate.call_args[0][0]
        assert isinstance(call_args, type(Z(0)))

    def test_qulacs_gate_s_adds_s_gate(self, mock_circuit):
        """Test that S gate adds S gate to circuit."""
        qubit = 0

        qulacs_gate_s(mock_circuit, qubit)

        mock_circuit.add_gate.assert_called_once()
        call_args = mock_circuit.add_gate.call_args[0][0]
        assert isinstance(call_args, type(S(0)))

    def test_qulacs_gate_sdg_adds_s_dagger_gate(self, mock_circuit):
        """Test that Sdg gate adds S-dagger gate to circuit."""
        qubit = 1

        qulacs_gate_sdg(mock_circuit, qubit)

        mock_circuit.add_gate.assert_called_once()
        call_args = mock_circuit.add_gate.call_args[0][0]
        assert isinstance(call_args, type(Sdag(0)))

    def test_qulacs_gate_t_adds_t_gate(self, mock_circuit):
        """Test that T gate adds T gate to circuit."""
        qubit = 0

        qulacs_gate_t(mock_circuit, qubit)

        mock_circuit.add_gate.assert_called_once()
        call_args = mock_circuit.add_gate.call_args[0][0]
        assert isinstance(call_args, type(T(0)))

    def test_qulacs_gate_tdg_adds_t_dagger_gate(self, mock_circuit):
        """Test that Tdg gate adds T-dagger gate to circuit."""
        qubit = 2

        qulacs_gate_tdg(mock_circuit, qubit)

        mock_circuit.add_gate.assert_called_once()
        call_args = mock_circuit.add_gate.call_args[0][0]
        assert isinstance(call_args, type(Tdag(0)))

    def test_single_qubit_gates_with_different_qubits(self, mock_circuit):
        """Test single-qubit gates work with different qubit indices."""
        for qubit_idx in [0, 1, 5, 10]:
            mock_circuit.reset_mock()
            qulacs_gate_h(mock_circuit, qubit_idx)
            assert mock_circuit.add_gate.called


class TestMultiQubitGates:
    """Test multi-qubit gate functions (SWAP, CNOT, CZ)."""

    def test_qulacs_gate_swap_adds_swap_gate(self, mock_circuit):
        """Test that SWAP gate adds SWAP gate to circuit."""
        qubit1, qubit2 = 0, 1

        qulacs_gate_swap(mock_circuit, qubit1, qubit2)

        mock_circuit.add_gate.assert_called_once()
        call_args = mock_circuit.add_gate.call_args[0][0]
        assert isinstance(call_args, type(SWAP(0, 1)))

    def test_qulacs_gate_cnot_adds_cnot_via_add_cnot_gate(self, mock_circuit):
        """Test that CNOT gate uses add_CNOT_gate method."""
        control, target = 0, 1

        qulacs_gate_cnot(mock_circuit, control, target)

        mock_circuit.add_CNOT_gate.assert_called_once_with(control, target)

    def test_qulacs_gate_cz_adds_cz_gate(self, mock_circuit):
        """Test that CZ gate adds CZ gate to circuit."""
        control, target = 1, 2

        qulacs_gate_cz(mock_circuit, control, target)

        mock_circuit.add_gate.assert_called_once()
        call_args = mock_circuit.add_gate.call_args[0][0]
        assert isinstance(call_args, type(CZ(0, 1)))

    def test_qulacs_gate_toffoli_adds_toffoli_gate(self, mock_circuit):
        """Test that Toffoli gate adds a TOFFOLI gate to circuit."""
        control1, control2, target = 0, 1, 2

        qulacs_gate_toffoli(mock_circuit, control1, control2, target)

        mock_circuit.add_gate.assert_called_once()
        call_args = mock_circuit.add_gate.call_args[0][0]
        assert isinstance(call_args, type(TOFFOLI(0, 1, 2)))

    def test_toffoli_with_different_qubit_triples(self, mock_circuit):
        """Test Toffoli gate with various control/target triples."""
        triples = [(0, 1, 2), (2, 1, 0), (0, 3, 5), (4, 2, 7)]
        for c1, c2, tgt in triples:
            mock_circuit.reset_mock()
            qulacs_gate_toffoli(mock_circuit, c1, c2, tgt)
            assert mock_circuit.add_gate.called

    def test_swap_with_different_qubit_pairs(self, mock_circuit):
        """Test SWAP gate with various qubit pairs."""
        pairs = [(0, 1), (1, 2), (0, 5), (3, 7)]
        for q1, q2 in pairs:
            mock_circuit.reset_mock()
            qulacs_gate_swap(mock_circuit, q1, q2)
            assert mock_circuit.add_gate.called

    def test_cnot_with_different_control_target_pairs(self, mock_circuit):
        """Test CNOT gate with various control-target pairs."""
        pairs = [(0, 1), (1, 0), (2, 5), (0, 3)]
        for ctrl, tgt in pairs:
            mock_circuit.reset_mock()
            qulacs_gate_cnot(mock_circuit, ctrl, tgt)
            mock_circuit.add_CNOT_gate.assert_called_with(ctrl, tgt)


class TestRotationGates:
    """Test rotation gate functions (RX, RY, RZ with constant angles)."""

    def test_qulacs_gate_rx_adds_rx_gate_with_angle(self, mock_circuit):
        """Test that RX gate adds RX gate with correct angle."""
        angle = 1.5
        qubit = 0

        qulacs_gate_rx(mock_circuit, angle, qubit)

        mock_circuit.add_gate.assert_called_once()
        call_args = mock_circuit.add_gate.call_args[0][0]
        assert isinstance(call_args, type(RX(0, 0.0)))

    def test_qulacs_gate_ry_adds_ry_gate_with_angle(self, mock_circuit):
        """Test that RY gate adds RY gate with correct angle."""
        angle = 0.5
        qubit = 1

        qulacs_gate_ry(mock_circuit, angle, qubit)

        mock_circuit.add_gate.assert_called_once()
        call_args = mock_circuit.add_gate.call_args[0][0]
        assert isinstance(call_args, type(RY(0, 0.0)))

    def test_qulacs_gate_rz_adds_rz_gate_with_angle(self, mock_circuit):
        """Test that RZ gate adds RZ gate with correct angle."""
        angle = 2.0
        qubit = 2

        qulacs_gate_rz(mock_circuit, angle, qubit)

        mock_circuit.add_gate.assert_called_once()
        call_args = mock_circuit.add_gate.call_args[0][0]
        assert isinstance(call_args, type(RZ(0, 0.0)))

    def test_rotation_gates_with_various_angles(self, mock_circuit):
        """Test rotation gates with different angle values."""
        angles = [0.0, 0.5, 1.57, 3.14, 6.28]
        for angle in angles:
            mock_circuit.reset_mock()
            qulacs_gate_rx(mock_circuit, angle, 0)
            assert mock_circuit.add_gate.called

    def test_rotation_gates_with_various_qubits(self, mock_circuit):
        """Test rotation gates work with different qubit indices."""
        for qubit in [0, 1, 3, 5, 10]:
            mock_circuit.reset_mock()
            qulacs_gate_ry(mock_circuit, 1.0, qubit)
            assert mock_circuit.add_gate.called


class TestParametricRotationGates:
    """Test parameterized rotation gate functions (parametric RX, RY, RZ)."""

    def test_qulacs_param_gate_rx_uses_add_parametric_rx_gate(self, mock_circuit):
        """Test that parametric RX gate uses add_parametric_RX_gate method."""
        angle = 1.5
        qubit = 0

        qulacs_param_gate_rx(mock_circuit, angle, qubit)

        mock_circuit.add_parametric_RX_gate.assert_called_once_with(qubit, angle)

    def test_qulacs_param_gate_ry_uses_add_parametric_ry_gate(self, mock_circuit):
        """Test that parametric RY gate uses add_parametric_RY_gate method."""
        angle = 0.5
        qubit = 1

        qulacs_param_gate_ry(mock_circuit, angle, qubit)

        mock_circuit.add_parametric_RY_gate.assert_called_once_with(qubit, angle)

    def test_qulacs_param_gate_rz_uses_add_parametric_rz_gate(self, mock_circuit):
        """Test that parametric RZ gate uses add_parametric_RZ_gate method."""
        angle = 2.0
        qubit = 2

        qulacs_param_gate_rz(mock_circuit, angle, qubit)

        mock_circuit.add_parametric_RZ_gate.assert_called_once_with(qubit, angle)

    def test_parametric_gates_preserve_angle_values(self, mock_circuit):
        """Test that parametric gates pass correct angle values."""
        angles = [0.1, 0.5, 1.57, 3.14, 6.28]
        for angle in angles:
            mock_circuit.reset_mock()
            qulacs_param_gate_rx(mock_circuit, angle, 0)
            mock_circuit.add_parametric_RX_gate.assert_called_with(0, angle)

    def test_parametric_gates_with_various_qubits(self, mock_circuit):
        """Test parametric gates work with different qubit indices."""
        for qubit in [0, 1, 3, 5, 10]:
            mock_circuit.reset_mock()
            qulacs_param_gate_ry(mock_circuit, 1.0, qubit)
            mock_circuit.add_parametric_RY_gate.assert_called_with(qubit, 1.0)

    def test_parametric_gates_with_negative_angles(self, mock_circuit):
        """Test parametric gates work with negative angle values."""
        angles = [-0.5, -1.57, -3.14]
        for angle in angles:
            mock_circuit.reset_mock()
            qulacs_param_gate_rz(mock_circuit, angle, 0)
            mock_circuit.add_parametric_RZ_gate.assert_called_with(0, angle)


class TestSpecialGates:
    """Test special gate functions (U1, combined gate dictionary tests)."""

    def test_qulacs_gate_u1_adds_u1_gate(self, mock_circuit):
        """Test that U1 gate adds U1 gate to circuit."""
        angle = 0.75
        qubit = 0

        qulacs_gate_u1(mock_circuit, angle, qubit)

        mock_circuit.add_gate.assert_called_once()
        call_args = mock_circuit.add_gate.call_args[0][0]
        assert isinstance(call_args, type(U1(0, 0.0)))

    def test_u1_gate_with_various_angles(self, mock_circuit):
        """Test U1 gate with different angle values."""
        angles = [0.0, 0.5, 1.57, 3.14, 6.28]
        for angle in angles:
            mock_circuit.reset_mock()
            qulacs_gate_u1(mock_circuit, angle, 0)
            assert mock_circuit.add_gate.called

    def test_u1_gate_with_various_qubits(self, mock_circuit):
        """Test U1 gate works with different qubit indices."""
        for qubit in [0, 1, 3, 5]:
            mock_circuit.reset_mock()
            qulacs_gate_u1(mock_circuit, 1.0, qubit)
            assert mock_circuit.add_gate.called


class TestGateDictionaries:
    """Test gate dictionary structure and completeness."""

    def test_qiskit_qulacs_gate_dict_contains_standard_gates(self):
        """Test that gate dictionary contains standard single and multi-qubit gates."""
        expected_gates = [
            "id",
            "h",
            "x",
            "y",
            "z",
            "s",
            "t",
            "swap",
            "cx",
            "cz",
            "ccx",
            "sdg",
            "tdg",
        ]
        for gate_name in expected_gates:
            assert gate_name in qiskit_qulacs_gate_dict

    def test_qiskit_qulacs_gate_dict_contains_rotation_gates(self):
        """Test that gate dictionary contains rotation gates."""
        expected_rotation_gates = ["rx", "ry", "rz"]
        for gate_name in expected_rotation_gates:
            assert gate_name in qiskit_qulacs_gate_dict

    def test_qiskit_qulacs_param_gate_dict_contains_parametric_gates(self):
        """Test that parametric gate dictionary contains parametric rotation gates."""
        expected_param_gates = ["rx", "ry", "rz"]
        for gate_name in expected_param_gates:
            assert gate_name in qiskit_qulacs_param_gate_dict

    def test_gate_dict_functions_are_callable(self):
        """Test that all functions in gate dictionary are callable."""
        for gate_name, gate_func in qiskit_qulacs_gate_dict.items():
            assert callable(gate_func), f"Gate {gate_name} is not callable"

    def test_param_gate_dict_functions_are_callable(self):
        """Test that all functions in parametric gate dictionary are callable."""
        for gate_name, gate_func in qiskit_qulacs_param_gate_dict.items():
            assert callable(gate_func), f"Parametric gate {gate_name} is not callable"

    def test_gate_dict_values_match_function_names(self):
        """Test that gate dictionary values are the expected gate functions."""
        assert qiskit_qulacs_gate_dict["id"] == qulacs_gate_i
        assert qiskit_qulacs_gate_dict["h"] == qulacs_gate_h
        assert qiskit_qulacs_gate_dict["x"] == qulacs_gate_x
        assert qiskit_qulacs_gate_dict["y"] == qulacs_gate_y
        assert qiskit_qulacs_gate_dict["z"] == qulacs_gate_z
        assert qiskit_qulacs_gate_dict["s"] == qulacs_gate_s
        assert qiskit_qulacs_gate_dict["t"] == qulacs_gate_t
        assert qiskit_qulacs_gate_dict["swap"] == qulacs_gate_swap
        assert qiskit_qulacs_gate_dict["cx"] == qulacs_gate_cnot
        assert qiskit_qulacs_gate_dict["cz"] == qulacs_gate_cz
        assert qiskit_qulacs_gate_dict["ccx"] == qulacs_gate_toffoli
        assert qiskit_qulacs_gate_dict["sdg"] == qulacs_gate_sdg
        assert qiskit_qulacs_gate_dict["tdg"] == qulacs_gate_tdg
        assert qiskit_qulacs_gate_dict["rx"] == qulacs_gate_rx
        assert qiskit_qulacs_gate_dict["ry"] == qulacs_gate_ry
        assert qiskit_qulacs_gate_dict["rz"] == qulacs_gate_rz

    def test_param_gate_dict_values_match_function_names(self):
        """Test that parametric gate dictionary values are the expected functions."""
        assert qiskit_qulacs_param_gate_dict["rx"] == qulacs_param_gate_rx
        assert qiskit_qulacs_param_gate_dict["ry"] == qulacs_param_gate_ry
        assert qiskit_qulacs_param_gate_dict["rz"] == qulacs_param_gate_rz


class TestGateEdgeCases:
    """Test edge cases and boundary conditions for gate functions."""

    def test_gate_with_zero_angle(self, mock_circuit):
        """Test rotation gates with zero angle."""
        qulacs_gate_rx(mock_circuit, 0.0, 0)
        mock_circuit.add_gate.assert_called_once()

    def test_gate_with_negative_angle(self, mock_circuit):
        """Test rotation gates with negative angles."""
        qulacs_param_gate_ry(mock_circuit, -1.5, 0)
        mock_circuit.add_parametric_RY_gate.assert_called_once_with(0, -1.5)

    def test_gate_with_large_angle(self, mock_circuit):
        """Test rotation gates with angles greater than 2π."""
        qulacs_param_gate_rz(mock_circuit, 10.0, 0)
        mock_circuit.add_parametric_RZ_gate.assert_called_once_with(0, 10.0)

    def test_gate_with_high_qubit_index(self, mock_circuit):
        """Test gates work with high qubit indices."""
        qulacs_gate_h(mock_circuit, 99)
        mock_circuit.add_gate.assert_called_once()

    def test_multi_qubit_gate_with_different_qubits_edge_case(self, mock_circuit):
        """Test multi-qubit gates with adjacent qubit indices."""
        qulacs_gate_cz(mock_circuit, 0, 1)
        mock_circuit.add_gate.assert_called_once()

    def test_gate_function_returns_none(self, mock_circuit):
        """Test that gate functions return None."""
        result = qulacs_gate_h(mock_circuit, 0)
        assert result is None

    def test_parametric_gate_function_returns_none(self, mock_circuit):
        """Test that parametric gate functions return None."""
        result = qulacs_param_gate_rx(mock_circuit, 1.0, 0)
        assert result is None
