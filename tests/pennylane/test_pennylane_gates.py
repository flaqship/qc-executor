import pennylane as qml
import pytest

from executor.pennylane.pennylane_gates import (
    RXX, RYY, RZZ, RZX, reset, tdg, sdg, cs, csx,
    qiskit_pennylane_gate_dict, pennylane_target
)


class TestPennyLaneGates:
    """Test suite for PennyLane gate conversions."""

    def test_rxx_gate(self):
        """Test the RXX gate implementation."""
        theta = 0.5
        wires = [0, 1]
        gate = RXX(theta, wires)
        assert gate is not None
        # RXX should be equivalent to PauliRot with 'XX'
        expected = qml.PauliRot(theta, "XX", wires=wires)
        assert type(gate).__name__ == type(expected).__name__

    def test_ryy_gate(self):
        """Test the RYY gate implementation."""
        theta = 0.5
        wires = [0, 1]
        gate = RYY(theta, wires)
        assert gate is not None
        # RYY should be equivalent to PauliRot with 'YY'
        expected = qml.PauliRot(theta, "YY", wires=wires)
        assert type(gate).__name__ == type(expected).__name__

    def test_rzz_gate(self):
        """Test the RZZ gate implementation."""
        theta = 0.5
        wires = [0, 1]
        gate = RZZ(theta, wires)
        assert gate is not None
        # RZZ should be equivalent to PauliRot with 'ZZ'
        expected = qml.PauliRot(theta, "ZZ", wires=wires)
        assert type(gate).__name__ == type(expected).__name__

    def test_rzx_gate(self):
        """Test the RZX gate implementation."""
        theta = 0.5
        wires = [0, 1]
        gate = RZX(theta, wires)
        assert gate is not None
        # RZX should be equivalent to PauliRot with 'ZX'
        expected = qml.PauliRot(theta, "ZX", wires=wires)
        assert type(gate).__name__ == type(expected).__name__

    def test_reset_gate(self):
        """Test the reset gate implementation."""
        wires = 0
        gate = reset(wires)
        assert gate is not None
        # Reset should use measure with reset=True
        expected = qml.measure(wires=wires, reset=True)
        assert type(gate).__name__ == type(expected).__name__

    def test_tdg_gate(self):
        """Test the T-dagger gate implementation."""
        wires = 0
        gate = tdg(wires)
        assert gate is not None
        # Tdg should be the adjoint of T
        assert "Adjoint" in str(type(gate))

    def test_sdg_gate(self):
        """Test the S-dagger gate implementation."""
        wires = 0
        gate = sdg(wires)
        assert gate is not None
        # Sdg should be the adjoint of S
        assert "Adjoint" in str(type(gate))

    def test_cs_gate(self):
        """Test the CS gate implementation."""
        wires = [0, 1]
        gate = cs(wires)
        assert gate is not None
        # CS should be controlled S gate
        assert "Controlled" in str(type(gate))

    def test_cs_gate_wrong_number_of_wires(self):
        """Test that CS gate raises error with wrong number of wires."""
        with pytest.raises(ValueError, match="CS gate requires two wires"):
            cs([0])

    def test_csx_gate(self):
        """Test the CSX gate implementation."""
        wires = [0, 1]
        gate = csx(wires)
        assert gate is not None
        # CSX should be controlled SX gate
        assert "Controlled" in str(type(gate))

    def test_csx_gate_wrong_number_of_wires(self):
        """Test that CSX gate raises error with wrong number of wires."""
        with pytest.raises(ValueError, match="CSX gate requires two wires"):
            csx([0])

    def test_gate_dict_contains_standard_gates(self):
        """Test that gate dictionary contains all standard gates."""
        expected_gates = [
            "id", "h", "x", "y", "z", "s", "t", "sx",
            "rx", "ry", "rz", "p", "cx", "cy", "cz",
            "cp", "crx", "cry", "crz", "swap", "ccx"
        ]
        for gate in expected_gates:
            assert gate in qiskit_pennylane_gate_dict, f"Gate {gate} not found in dictionary"

    def test_gate_dict_contains_custom_gates(self):
        """Test that gate dictionary contains custom gates."""
        custom_gates = ["rxx", "ryy", "rzz", "rzx", "tdg", "sdg", "cs", "csx", "reset"]
        for gate in custom_gates:
            assert gate in qiskit_pennylane_gate_dict, f"Custom gate {gate} not found"

    def test_gate_dict_values_are_callable(self):
        """Test that gate dictionary values are callable."""
        for gate_name, gate_func in qiskit_pennylane_gate_dict.items():
            assert callable(gate_func), f"Gate {gate_name} is not callable"

    def test_pennylane_target_is_valid(self):
        """Test that pennylane_target is a valid Qiskit Target."""
        assert pennylane_target is not None
        assert hasattr(pennylane_target, 'operations')
        # Check that target contains expected gates
        assert len(pennylane_target.operations) > 0

    def test_pennylane_target_contains_gates(self):
        """Test that pennylane_target contains gates from the dictionary."""
        operation_names = [op.name for op in pennylane_target.operations]
        for gate_name in qiskit_pennylane_gate_dict.keys():
            if gate_name == "measure":  # measure might not be in operations
                continue
            # Target should contain the gate operations
            assert gate_name in operation_names or \
                   gate_name == "barrier" or gate_name == "reset", \
                   f"Gate {gate_name} not in target operations"
