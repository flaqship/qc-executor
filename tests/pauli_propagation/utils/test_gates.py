"""Tests for gates module."""

import numpy as np
import pytest
import sympy as sp

from qc_executor.pauli_propagation.utils.gates import (
    CliffordGate,
    LayerBarrier,
    PauliRotation,
)
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

    def test_param_name_inferred_from_symbolic_expr(self):
        """Parameter name is inferred from symbolic expression."""
        theta = sp.Symbol("theta")
        rz = PauliRotation(["Z"], 0, nqubits=1, param_expr=theta)
        assert rz.param_name == "theta"

    def test_param_name_none_for_non_symbol_expr(self):
        """Non-symbol expression does not expose a parameter name."""
        expr = sp.sin(sp.Symbol("theta"))
        rz = PauliRotation(["Z"], 0, nqubits=1, param_expr=expr)
        assert rz.param_name is None

    def test_repr_single_qubit(self):
        """Single-qubit repr uses scalar qubit formatting."""
        rx = PauliRotation(["X"], 0, nqubits=1)
        assert repr(rx) == "PauliRotation(X, qubits=0)"


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

    def test_commutes_with_returns_false(self):
        """Clifford commutation check is conservatively false."""
        h = CliffordGate("H", 0, nqubits=1)
        x_term = string_to_term("X", 1)
        assert not h.commutes_with(x_term)

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
        assert np.isclose(phase, 1.0)

    def test_t_gate_transforms_x_with_complex_phase(self):
        """T: X → X with exp(-iπ/4) phase."""
        t = CliffordGate("T", 0, nqubits=1)
        x_term = string_to_term("X", 1)

        new_term, phase = t.transform_pauli_term(x_term)
        assert new_term == x_term
        assert np.isclose(phase, np.exp(-1j * np.pi / 4))

    def test_init_accepts_cx_alias(self):
        """CX alias is normalized and accepted."""
        cx = CliffordGate("cx", [0, 1], nqubits=2)
        assert cx.gate_type == "CX"
        assert cx.qubits == [0, 1]

    def test_repr_two_qubit(self):
        """Two-qubit repr uses list formatting."""
        cx = CliffordGate("CX", [0, 1], nqubits=2)
        assert repr(cx) == "CliffordGate(CX, qubits=[0, 1])"

    @pytest.mark.parametrize(
        ("input_term", "expected_term", "expected_phase"),
        [
            ("XY", "YZ", -1.0),
            ("IZ", "ZZ", 1.0),
            ("XZ", "YY", 1.0),
            ("YZ", "XY", -1.0),
            ("ZZ", "IZ", 1.0),
            ("IY", "ZY", 1.0),
            ("YY", "XZ", 1.0),
            ("ZY", "IY", 1.0),
            ("XX", "XI", 1.0),
            ("YX", "YI", 1.0),
        ],
    )
    def test_cnot_transform_branches(self, input_term, expected_term, expected_phase):
        """CNOT branch table follows implemented transform logic."""
        cnot = CliffordGate("CNOT", [0, 1], nqubits=2)
        in_term = string_to_term(input_term, 2)
        out_term = string_to_term(expected_term, 2)

        new_term, phase = cnot.transform_pauli_term(in_term)
        assert new_term == out_term
        assert np.isclose(phase, expected_phase)

    def test_cz_transforms_x_on_second_qubit(self):
        """CZ applies the symmetric X→XZ rule on second qubit."""
        cz = CliffordGate("CZ", [0, 1], nqubits=2)
        in_term = string_to_term("IX", 2)
        expected = string_to_term("ZX", 2)

        new_term, phase = cz.transform_pauli_term(in_term)
        assert new_term == expected
        assert np.isclose(phase, 1.0)

    def test_cz_transforms_x_on_first_qubit(self):
        """CZ applies the X→XZ rule on first qubit."""
        cz = CliffordGate("CZ", [0, 1], nqubits=2)
        in_term = string_to_term("XI", 2)
        expected = string_to_term("XZ", 2)

        new_term, phase = cz.transform_pauli_term(in_term)
        assert new_term == expected
        assert np.isclose(phase, 1.0)

    def test_transform_pauli_term_unknown_type_raises(self):
        """Unknown gate type in dispatcher raises error."""
        h = CliffordGate("H", 0, nqubits=1)
        h.gate_type = "UNKNOWN"

        with pytest.raises(ValueError, match="Transformation not implemented"):
            h.transform_pauli_term(string_to_term("X", 1))


class TestLayerBarrier:
    """Test layer barrier marker."""

    def test_repr(self):
        """LayerBarrier repr is stable."""
        barrier = LayerBarrier()
        assert repr(barrier) == "LayerBarrier()"
