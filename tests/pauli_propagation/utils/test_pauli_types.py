"""Tests for pauli_types module."""

import importlib
import sys
import typing

import numpy as np
import pytest

from qc_executor.pauli_propagation.utils.pauli_types import PauliString, PauliSum


class TestPauliString:
    """Test PauliString class."""

    def test_init_default(self):
        """Test default initialization (all identities)."""
        ps = PauliString(3)
        assert ps.nqubits == 3
        assert ps.to_string() == "III"
        assert ps.coeff == 1.0

    def test_init_from_string(self):
        """Test initialization from string."""
        ps = PauliString(4, "IXYZ", coeff=2.0)
        assert ps.nqubits == 4
        assert ps.to_string() == "IXYZ"
        assert ps.coeff == 2.0

    def test_init_invalid_term_type(self):
        """Test error on unsupported term type."""
        with pytest.raises(TypeError, match="term must be int, str, or None"):
            PauliString(2, ["X"])

    def test_init_from_int(self):
        """Test initialization from integer."""
        # 0b11100100 = IXYZ
        ps = PauliString(4, 0b11100100, coeff=1.5)
        assert ps.to_string() == "IXYZ"
        assert ps.coeff == 1.5

    def test_init_from_numpy_int(self):
        """Test initialization from numpy integer."""
        ps = PauliString(4, np.int64(0b11100100), coeff=1.5)
        assert ps.to_string() == "IXYZ"
        assert ps.coeff == 1.5

    def test_from_symbols(self):
        """Test creation from symbols and indices."""
        ps = PauliString.from_symbols(["X", "Z"], [0, 2], 3, coeff=2.0)
        assert ps.to_string() == "XIZ"
        assert ps.coeff == 2.0

    def test_from_symbols_empty(self):
        """Test creation with no symbols (identity)."""
        ps = PauliString.from_symbols([], [], 3)
        assert ps.to_string() == "III"

    def test_from_symbols_mismatch(self):
        """Test error on mismatched symbols and indices."""
        with pytest.raises(ValueError, match="same length"):
            PauliString.from_symbols(["X", "Z"], [0], 3)

    def test_weight(self):
        """Test weight calculation."""
        assert PauliString(3, "III").weight() == 0
        assert PauliString(3, "XII").weight() == 1
        assert PauliString(3, "XYZ").weight() == 3

    def test_multiply(self):
        """Test Pauli string multiplication."""
        ps1 = PauliString(2, "XY", coeff=2.0)
        ps2 = PauliString(2, "YX", coeff=3.0)

        result = ps1.multiply(ps2)
        assert result.to_string() == "ZZ"
        # Coefficient: 2.0 * 3.0 * (i * -i) = 6.0 * 1 = 6.0
        assert np.isclose(result.coeff, 6.0)

    def test_multiply_different_nqubits(self):
        """Test error on multiplying different qubit counts."""
        ps1 = PauliString(2, "XY")
        ps2 = PauliString(3, "XYZ")
        with pytest.raises(ValueError, match="different nqubits"):
            ps1.multiply(ps2)

    def test_commutes_with(self):
        """Test commutation checking."""
        ps1 = PauliString(2, "XY")
        ps2 = PauliString(2, "YX")
        assert ps1.commutes_with(ps2)  # 2 anticommute positions → commute

        ps1 = PauliString(1, "X")
        ps2 = PauliString(1, "Y")
        assert not ps1.commutes_with(ps2)  # Anticommute

    def test_commutes_with_different_nqubits(self):
        """Test error on commutation with different qubit counts."""
        ps1 = PauliString(2, "XY")
        ps2 = PauliString(3, "XYZ")

        with pytest.raises(ValueError, match="different nqubits"):
            ps1.commutes_with(ps2)

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        ps = PauliString(2, "XY", coeff=2.0)

        result = ps * 3.0
        assert result.to_string() == "XY"
        assert result.coeff == 6.0

        result = 3.0 * ps
        assert result.coeff == 6.0

    def test_scalar_multiplication_complex(self):
        """Test complex scalar multiplication."""
        ps = PauliString(2, "XY", coeff=1.0)
        result = ps * 1j
        assert np.isclose(result.coeff, 1j)

    def test_equality(self):
        """Test equality comparison."""
        ps1 = PauliString(2, "XY", coeff=2.0)
        ps2 = PauliString(2, "XY", coeff=2.0)
        ps3 = PauliString(2, "XY", coeff=3.0)
        ps4 = PauliString(2, "XZ", coeff=2.0)

        assert ps1 == ps2
        assert ps1 != ps3  # Different coefficient
        assert ps1 != ps4  # Different Pauli string
        assert ps1 != object()

    def test_hash(self):
        """Test hashing (based on term, not coefficient)."""
        ps1 = PauliString(2, "XY", coeff=2.0)
        ps2 = PauliString(2, "XY", coeff=3.0)
        ps3 = PauliString(2, "XZ", coeff=2.0)

        assert hash(ps1) == hash(ps2)  # Same term
        assert hash(ps1) != hash(ps3)  # Different term

    def test_repr_and_str(self):
        """Test string representations."""
        ps = PauliString(3, "XYZ", coeff=2.5)
        assert "2.5" in str(ps)
        assert "XYZ" in str(ps)
        assert "PauliString" in repr(ps)

    def test_repr_and_str_complex(self):
        """Test string representations with complex coefficient."""
        ps = PauliString(3, "XYZ", coeff=2.5j)
        assert "j" in str(ps)
        assert "j" in repr(ps)


class TestPauliSum:
    """Test PauliSum class."""

    def test_init_empty(self):
        """Test empty initialization."""
        psum = PauliSum(3)
        assert psum.nqubits == 3
        assert len(psum) == 0

    def test_add_term_from_string(self):
        """Test adding term from string."""
        psum = PauliSum(3)
        psum.add_term("XYZ", coeff=2.0)
        assert len(psum) == 1
        assert psum.get_coeff("XYZ") == 2.0

    def test_add_term_from_int(self):
        """Test adding term from integer."""
        psum = PauliSum(3)
        term_int = 0b111011  # XYZ
        psum.add_term(term_int, coeff=3.0)
        assert len(psum) == 1
        assert psum.get_coeff(term_int) == 3.0

    def test_add_term_from_numpy_int(self):
        """Test adding term from numpy integer."""
        psum = PauliSum(3)
        term_int = np.int64(0b111011)  # XYZ
        psum.add_term(term_int, coeff=3.0)
        assert len(psum) == 1
        assert psum.get_coeff(term_int) == 3.0

    def test_add_term_from_pauli_string(self):
        """Test adding PauliString."""
        psum = PauliSum(3)
        ps = PauliString(3, "XYZ", coeff=2.5)
        psum.add_term(ps)
        assert len(psum) == 1
        assert psum.get_coeff("XYZ") == 2.5

    def test_add_term_accumulates(self):
        """Test that adding same term accumulates coefficients."""
        psum = PauliSum(3)
        psum.add_term("XYZ", coeff=2.0)
        psum.add_term("XYZ", coeff=3.0)
        assert len(psum) == 1  # Still one term
        assert psum.get_coeff("XYZ") == 5.0

    def test_add_term_cancels_to_zero(self):
        """Test that equal and opposite coefficients remove the term."""
        psum = PauliSum(3)
        psum.add_term("XYZ", coeff=2.0)
        psum.add_term("XYZ", coeff=-2.0)

        assert len(psum) == 0
        assert psum.get_coeff("XYZ") == 0.0

    def test_add_term_different_nqubits(self):
        """Test error on wrong nqubits."""
        psum = PauliSum(3)
        ps = PauliString(4, "XYZI")
        with pytest.raises(ValueError, match="expected 3"):
            psum.add_term(ps)

    def test_add_term_invalid_type(self):
        """Test error on unsupported term type."""
        psum = PauliSum(3)

        with pytest.raises(TypeError, match="term must be int, str, or PauliString"):
            psum.add_term(["X"])

    def test_get_coeff_missing(self):
        """Test getting coefficient of missing term."""
        psum = PauliSum(3)
        assert psum.get_coeff("XYZ") == 0.0

    def test_set_coeff(self):
        """Test setting coefficient."""
        psum = PauliSum(3)
        psum.set_coeff("XYZ", 2.0)
        assert psum.get_coeff("XYZ") == 2.0

        psum.set_coeff("XYZ", 5.0)
        assert psum.get_coeff("XYZ") == 5.0

    def test_set_coeff_from_int(self):
        """Test setting coefficient from integer term."""
        psum = PauliSum(3)
        term_int = 0b111011  # XYZ
        psum.set_coeff(term_int, 2.0)
        assert psum.get_coeff(term_int) == 2.0

    def test_set_coeff_zero_removes(self):
        """Test that setting coefficient to zero removes term."""
        psum = PauliSum(3)
        psum.add_term("XYZ", 2.0)
        assert len(psum) == 1

        psum.set_coeff("XYZ", 1e-16)  # Essentially zero
        assert len(psum) == 0

    def test_copy(self):
        """Test deep copy."""
        psum1 = PauliSum(3)
        psum1.add_term("XYZ", 2.0)
        psum1.add_term("III", 1.0)

        psum2 = psum1.copy()
        assert len(psum2) == 2
        assert psum2.get_coeff("XYZ") == 2.0

        # Modify copy shouldn't affect original
        psum2.add_term("XYZ", 3.0)
        assert psum1.get_coeff("XYZ") == 2.0
        assert psum2.get_coeff("XYZ") == 5.0

    def test_len(self):
        """Test length."""
        psum = PauliSum(3)
        assert len(psum) == 0

        psum.add_term("XYZ", 1.0)
        assert len(psum) == 1

        psum.add_term("III", 2.0)
        assert len(psum) == 2

    def test_iter(self):
        """Test iteration."""
        psum = PauliSum(3)
        psum.add_term("XYZ", 2.0)
        psum.add_term("III", 3.0)

        terms_dict = dict(psum)
        assert len(terms_dict) == 2

    def test_add_pauli_sums(self):
        """Test addition of two PauliSums."""
        psum1 = PauliSum(3)
        psum1.add_term("XYZ", 2.0)
        psum1.add_term("III", 1.0)

        psum2 = PauliSum(3)
        psum2.add_term("XYZ", 3.0)
        psum2.add_term("ZZZ", 4.0)

        result = psum1 + psum2
        assert len(result) == 3
        assert result.get_coeff("XYZ") == 5.0  # 2 + 3
        assert result.get_coeff("III") == 1.0
        assert result.get_coeff("ZZZ") == 4.0

    def test_add_different_nqubits(self):
        """Test error on adding different nqubits."""
        psum1 = PauliSum(3)
        psum2 = PauliSum(4)
        with pytest.raises(ValueError, match="different nqubits"):
            _ = psum1 + psum2

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        psum = PauliSum(3)
        psum.add_term("XYZ", 2.0)
        psum.add_term("III", 3.0)

        result = psum * 2.0
        assert result.get_coeff("XYZ") == 4.0
        assert result.get_coeff("III") == 6.0

        result = 2.0 * psum
        assert result.get_coeff("XYZ") == 4.0

    def test_scalar_multiplication_complex(self):
        """Test complex scalar multiplication."""
        psum = PauliSum(2)
        psum.add_term("XY", 1.0)

        result = psum * 1j
        assert np.isclose(result.get_coeff("XY"), 1j)

    def test_str_empty(self):
        """Test string representation of empty sum."""
        psum = PauliSum(3)
        assert str(psum) == "0"

    def test_str_single_term(self):
        """Test string representation of single term."""
        psum = PauliSum(3)
        psum.add_term("XYZ", 2.5)
        s = str(psum)
        assert "2.5" in s or "2.500" in s
        assert "XYZ" in s

    def test_str_single_term_complex(self):
        """Test string representation of single complex term."""
        psum = PauliSum(3)
        psum.add_term("XYZ", 2.5j)
        s = str(psum)
        assert "j" in s

    def test_repr(self):
        """Test repr."""
        psum = PauliSum(3)
        psum.add_term("XYZ", 2.0)
        psum.add_term("III", 1.0)
        r = repr(psum)
        assert "PauliSum" in r
        assert "nqubits=3" in r
        assert "terms=2" in r

    def test_repr_empty(self):
        """Test repr for empty sum."""
        psum = PauliSum(3)
        assert repr(psum) == "PauliSum(nqubits=3, terms=0)"

    def test_has_active_symmetry_default(self):
        """Default symmetry is inactive."""
        psum = PauliSum(3)
        assert not psum.has_active_symmetry

    def test_has_active_symmetry_custom(self):
        """Custom symmetry marks the PauliSum as active."""

        class DummySymmetry:
            pass

        psum = PauliSum(3, symmetry=DummySymmetry())
        assert psum.has_active_symmetry


class TestPauliTypesModule:
    """Test module-level import behavior."""

    def test_type_checking_import_branch(self, monkeypatch):
        """TYPE_CHECKING branch imports SymmetryStrategy when enabled."""
        module_name = "qc_executor.pauli_propagation.utils.pauli_types"

        with monkeypatch.context() as patch_ctx:
            patch_ctx.setattr(typing, "TYPE_CHECKING", True)
            sys.modules.pop(module_name, None)
            reloaded = importlib.import_module(module_name)
            assert hasattr(reloaded, "SymmetryStrategy")

        sys.modules.pop(module_name, None)
        importlib.import_module(module_name)
