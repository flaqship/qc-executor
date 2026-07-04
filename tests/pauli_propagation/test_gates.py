"""Tests for gates module."""

import numpy as np
import pytest

from qc_executor.pauli_propagation.utils.gates import CliffordGate, PauliRotation
from qc_executor.pauli_propagation.utils.pauli_algebra import string_to_term


class TestPauliRotation:
    """Test Pauli rotation gates."""

    def test_init_single_qubit(self):
        """Test RX, RY, RZ initialization."""
        rx = PauliRotation(["X"], 0, nqubits=2, param_name="theta")
        assert rx.symbols == ["X"]
        assert rx.qubits == [0]
        assert rx.nqubits == 2
        assert rx.param_name == "theta"

    def test_init_two_qubit(self):
        """Test RXX, RZZ initialization."""
        rxx = PauliRotation(["X", "X"], [0, 1], nqubits=3)
        assert rxx.symbols == ["X", "X"]
        assert rxx.qubits == [0, 1]
        assert rxx.nqubits == 3

    def test_init_mismatch_symbols_qubits(self):
        """Test error on mismatched symbols and qubits."""
        with pytest.raises(ValueError, match="must match"):
            PauliRotation(["X", "Y"], [0], nqubits=2)

    def test_is_parametric(self):
        """Pauli rotations are parametric."""
        rx = PauliRotation(["X"], 0, nqubits=1)
        assert rx.is_parametric()

    def test_commutes_with_same_pauli(self):
        """RX commutes with X."""
        rx = PauliRotation(["X"], 0, nqubits=1)
        x_term = string_to_term("X", 1)
        assert rx.commutes_with(x_term)

    def test_anticommutes_with_different_pauli(self):
        """RX anticommutes with Z."""
        rx = PauliRotation(["X"], 0, nqubits=1)
        z_term = string_to_term("Z", 1)
        assert not rx.commutes_with(z_term)

    def test_commutes_with_identity(self):
        """All rotations commute with identity."""
        rz = PauliRotation(["Z"], 1, nqubits=3)
        i_term = string_to_term("III", 3)
        assert rz.commutes_with(i_term)


class TestCliffordGate:
    """Test Clifford gates."""

    def test_init_single_qubit(self):
        """Test H, S gate initialization."""
        h = CliffordGate("H", 0, nqubits=2)
        assert h.gate_type == "H"
        assert h.qubits == [0]
        assert h.nqubits == 2

    def test_init_two_qubit(self):
        """Test CNOT initialization."""
        cnot = CliffordGate("CNOT", [0, 1], nqubits=3)
        assert cnot.gate_type == "CNOT"
        assert cnot.qubits == [0, 1]

    def test_invalid_gate_type(self):
        """Test error on unknown gate."""
        with pytest.raises(ValueError, match="Unknown"):
            CliffordGate("INVALID", 0, nqubits=2)

    def test_wrong_qubit_count_single(self):
        """Test error on wrong qubit count for single-qubit gate."""
        with pytest.raises(ValueError, match="exactly 1 qubit"):
            CliffordGate("H", [0, 1], nqubits=2)

    def test_wrong_qubit_count_two(self):
        """Test error on wrong qubit count for two-qubit gate."""
        with pytest.raises(ValueError, match="exactly 2 qubits"):
            CliffordGate("CNOT", 0, nqubits=2)

    def test_is_not_parametric(self):
        """Clifford gates are not parametric."""
        h = CliffordGate("H", 0, nqubits=1)
        assert not h.is_parametric()

    def test_hadamard_transforms_x_to_z(self):
        """H: X → Z."""
        h = CliffordGate("H", 0, nqubits=1)
        x_term = string_to_term("X", 1)
        z_term = string_to_term("Z", 1)

        new_term, phase = h.transform_pauli_term(x_term)
        assert new_term == z_term
        assert np.isclose(phase, 1.0)

    def test_hadamard_transforms_z_to_x(self):
        """H: Z → X."""
        h = CliffordGate("H", 0, nqubits=1)
        z_term = string_to_term("Z", 1)
        x_term = string_to_term("X", 1)

        new_term, phase = h.transform_pauli_term(z_term)
        assert new_term == x_term
        assert np.isclose(phase, 1.0)

    def test_hadamard_transforms_y_to_minus_y(self):
        """H: Y → -Y."""
        h = CliffordGate("H", 0, nqubits=1)
        y_term = string_to_term("Y", 1)

        new_term, phase = h.transform_pauli_term(y_term)
        assert new_term == y_term
        assert np.isclose(phase, -1.0)

    def test_s_gate_transforms_x_to_y(self):
        """S: X → Y."""
        s = CliffordGate("S", 0, nqubits=1)
        x_term = string_to_term("X", 1)
        y_term = string_to_term("Y", 1)

        new_term, phase = s.transform_pauli_term(x_term)
        assert new_term == y_term
        assert np.isclose(phase, 1.0)

    def test_s_gate_transforms_y_to_minus_x(self):
        """S: Y → -X."""
        s = CliffordGate("S", 0, nqubits=1)
        y_term = string_to_term("Y", 1)
        x_term = string_to_term("X", 1)

        new_term, phase = s.transform_pauli_term(y_term)
        assert new_term == x_term
        assert np.isclose(phase, -1.0)

    def test_swap_exchanges_paulis(self):
        """SWAP exchanges Pauli operators on two qubits."""
        swap = CliffordGate("SWAP", [0, 1], nqubits=2)
        xy_term = string_to_term("XY", 2)
        yx_term = string_to_term("YX", 2)

        new_term, phase = swap.transform_pauli_term(xy_term)
        assert new_term == yx_term
        assert np.isclose(phase, 1.0)

    def test_multiquet_clifford_preserves_identity(self):
        """Clifford gates preserve identity on untouched qubits."""
        h = CliffordGate("H", 0, nqubits=3)
        xii_term = string_to_term("XII", 3)
        zii_term = string_to_term("ZII", 3)

        new_term, phase = h.transform_pauli_term(xii_term)
        assert new_term == zii_term
