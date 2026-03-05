"""
Test suite for PennyLane gate conversions.

This module tests the gate conversion functions that map Qiskit gates to
PennyLane equivalents, including:
- Custom two-qubit rotation gates (RXX, RYY, RZZ, RZX)
- Adjoint gates (Tdg, Sdg)
- Controlled gates (CS, CSX)
- Reset gate
- Gate dictionary completeness
- PennyLane target validation
"""

import numpy as np
import pennylane as qml
import pytest

from executor.pennylane.pennylane_gates import (
    RXX,
    RYY,
    RZZ,
    RZX,
    reset,
    tdg,
    sdg,
    cs,
    csx,
    qiskit_pennylane_gate_dict,
    pennylane_target,
)


class TestPennyLaneGates:
    """Test suite for PennyLane gate conversions."""

    # Two-Qubit Rotation Gates

    def test_rxx_gate(self):
        """Test RXX gate (two-qubit XX rotation)."""
        theta = 0.5
        wires = [0, 1]
        gate = RXX(theta, wires)

        assert gate is not None
        expected = qml.PauliRot(theta, "XX", wires=wires)
        assert type(gate).__name__ == type(expected).__name__

    def test_ryy_gate(self):
        """Test RYY gate (two-qubit YY rotation)."""
        theta = 0.7
        wires = [0, 1]
        gate = RYY(theta, wires)

        assert gate is not None
        expected = qml.PauliRot(theta, "YY", wires=wires)
        assert type(gate).__name__ == type(expected).__name__

    def test_rzz_gate(self):
        """Test RZZ gate (two-qubit ZZ rotation)."""
        theta = np.pi / 4
        wires = [0, 1]
        gate = RZZ(theta, wires)

        assert gate is not None
        expected = qml.PauliRot(theta, "ZZ", wires=wires)
        assert type(gate).__name__ == type(expected).__name__

    def test_rzx_gate(self):
        """Test RZX gate (two-qubit ZX rotation)."""
        theta = np.pi / 2
        wires = [1, 2]
        gate = RZX(theta, wires)

        assert gate is not None
        expected = qml.PauliRot(theta, "ZX", wires=wires)
        assert type(gate).__name__ == type(expected).__name__

    @pytest.mark.parametrize("theta", [0.0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi])
    def test_rxx_gate_various_angles(self, theta):
        """Test RXX gate with various rotation angles."""
        wires = [0, 1]
        gate = RXX(theta, wires)
        assert gate is not None

    # Adjoint (Dagger) Gates

    def test_tdg_gate(self):
        """Test T-dagger gate (adjoint of T gate)."""
        wires = 0
        gate = tdg(wires)

        assert gate is not None
        assert "Adjoint" in str(type(gate))

    def test_sdg_gate(self):
        """Test S-dagger gate (adjoint of S gate)."""
        wires = 0
        gate = sdg(wires)

        assert gate is not None
        assert "Adjoint" in str(type(gate))

    def test_tdg_gate_different_wire(self):
        """Test T-dagger gate on different wire."""
        wires = 2
        gate = tdg(wires)
        assert gate is not None

    # Controlled Gates

    def test_cs_gate(self):
        """Test CS gate (controlled-S)."""
        wires = [0, 1]
        gate = cs(wires)

        assert gate is not None
        assert "Controlled" in str(type(gate))

    def test_csx_gate(self):
        """Test CSX gate (controlled-SX)."""
        wires = [0, 1]
        gate = csx(wires)

        assert gate is not None
        assert "Controlled" in str(type(gate))

    def test_cs_gate_reversed_wires(self):
        """Test CS gate with reversed control/target."""
        wires = [1, 0]
        gate = cs(wires)
        assert gate is not None

    def test_cs_gate_wrong_wire_count_raises_error(self):
        """Test that CS gate raises ValueError for wrong number of wires."""
        with pytest.raises(ValueError, match="CS gate requires two wires"):
            cs([0])

    def test_csx_gate_wrong_wire_count_raises_error(self):
        """Test that CSX gate raises ValueError for wrong number of wires."""
        with pytest.raises(ValueError, match="CSX gate requires two wires"):
            csx([0])

    def test_cs_gate_too_many_wires_raises_error(self):
        """Test that CS gate raises ValueError for too many wires."""
        with pytest.raises(ValueError, match="CS gate requires two wires"):
            cs([0, 1, 2])

    # Reset Gate

    def test_reset_gate(self):
        """Test reset gate (measurement with reset)."""
        wires = 0
        gate = reset(wires)

        assert gate is not None
        expected = qml.measure(wires=wires, reset=True)
        assert type(gate).__name__ == type(expected).__name__

    def test_reset_gate_different_wire(self):
        """Test reset gate on different wire."""
        wires = 3
        gate = reset(wires)
        assert gate is not None

    # Gate Dictionary Tests

    def test_gate_dictionary_exists(self):
        """Test that gate dictionary exists and is not empty."""
        assert qiskit_pennylane_gate_dict is not None
        assert len(qiskit_pennylane_gate_dict) > 0

    @pytest.mark.parametrize(
        "gate_name",
        [
            "id",
            "h",
            "x",
            "y",
            "z",
            "s",
            "t",
            "sx",  # Single-qubit gates
            "rx",
            "ry",
            "rz",
            "p",  # Single-qubit parametric gates
            "cx",
            "cy",
            "cz",
            "swap",
            "ccx",  # Multi-qubit gates
            "cp",
            "crx",
            "cry",
            "crz",  # Multi-qubit parametric gates
        ],
    )
    def test_standard_gates_in_dictionary(self, gate_name):
        """Test that standard gates are in the dictionary and callable."""
        assert gate_name in qiskit_pennylane_gate_dict
        gate_func = qiskit_pennylane_gate_dict[gate_name]
        assert callable(gate_func)

    @pytest.mark.parametrize(
        "gate_name",
        [
            "rxx",
            "ryy",
            "rzz",
            "rzx",  # Custom two-qubit rotations
            "tdg",
            "sdg",  # Adjoint gates
            "cs",
            "csx",  # Controlled gates
            "reset",  # Reset gate
        ],
    )
    def test_custom_gates_in_dictionary(self, gate_name):
        """Test that custom gates are in the dictionary and callable."""
        assert gate_name in qiskit_pennylane_gate_dict
        gate_func = qiskit_pennylane_gate_dict[gate_name]
        assert callable(gate_func)

    def test_gate_dictionary_has_measure(self):
        """Test that measure gate is in the dictionary."""
        assert "measure" in qiskit_pennylane_gate_dict
        assert qiskit_pennylane_gate_dict["measure"] == qml.measure

    # PennyLane Target Tests

    def test_pennylane_target_exists(self):
        """Test that PennyLane target object exists."""
        assert pennylane_target is not None

    def test_pennylane_target_has_operations(self):
        """Test that PennyLane target has operations attribute."""
        assert hasattr(pennylane_target, "operations")
        assert len(pennylane_target.operations) > 0

    def test_pennylane_target_contains_gates(self):
        """Test that PennyLane target contains expected gate operations."""
        operation_names = [op.name for op in pennylane_target.operations]

        # Check a sample of important gates
        important_gates = ["h", "x", "cx", "rx", "ry", "rz"]
        for gate in important_gates:
            assert gate in operation_names, f"Gate '{gate}' not in target operations"

    def test_pennylane_target_gate_count(self):
        """Test that PennyLane target has reasonable number of gates."""
        # Should have at least 20 gates
        assert len(pennylane_target.operations) >= 20
