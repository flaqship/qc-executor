"""Tests for pauli_algebra module."""

import numpy as np
import pytest

from executor.pauli_propagation.utils.pauli_algebra import (
    commutes,
    contains_x_or_y,
    count_xy,
    count_weight,
    get_pauli,
    get_uint_type,
    int_to_symbol,
    pauli_multiply,
    pauli_sum_product,
    set_pauli,
    string_to_term,
    symbol_to_int,
    term_to_string,
)
from executor.pauli_propagation.utils.pauli_types import PauliSum


class TestUintType:
    """Test uint type selection."""

    def test_small_systems(self):
        assert get_uint_type(1) == np.uint8
        assert get_uint_type(4) == np.uint8

    def test_medium_systems(self):
        assert get_uint_type(5) == np.uint16
        assert get_uint_type(8) == np.uint16

    def test_large_systems(self):
        assert get_uint_type(9) == np.uint32
        assert get_uint_type(16) == np.uint32

    def test_very_large_systems(self):
        assert get_uint_type(17) == np.uint64
        assert get_uint_type(32) == np.uint64

    def test_large_qubit_fallback(self):
        assert get_uint_type(33) == int
        assert get_uint_type(100) == int


class TestSymbolConversion:
    """Test symbol to integer conversion."""

    def test_valid_symbols(self):
        assert symbol_to_int("I") == 0
        assert symbol_to_int("X") == 1
        assert symbol_to_int("Y") == 2
        assert symbol_to_int("Z") == 3

    def test_invalid_symbol(self):
        with pytest.raises(ValueError, match="Invalid Pauli symbol"):
            symbol_to_int("A")

    def test_int_to_symbol(self):
        assert int_to_symbol(0) == "I"
        assert int_to_symbol(1) == "X"
        assert int_to_symbol(2) == "Y"
        assert int_to_symbol(3) == "Z"

    def test_invalid_int(self):
        with pytest.raises(ValueError, match="Invalid Pauli integer"):
            int_to_symbol(4)


class TestBitOperations:
    """Test get_pauli and set_pauli operations."""

    def test_get_pauli_single_qubit(self):
        # Term = 0b01 = X on qubit 0
        term = 0b01
        assert get_pauli(term, 0, 1) == 1  # X

    def test_get_pauli_multiple_qubits(self):
        # Term = 0b11_10_01_00 = Z_3 Y_2 X_1 I_0
        # Qubit 0 (bits 0-1): 00 = I
        # Qubit 1 (bits 2-3): 01 = X
        # Qubit 2 (bits 4-5): 10 = Y
        # Qubit 3 (bits 6-7): 11 = Z
        term = 0b11100100
        assert get_pauli(term, 0, 4) == 0  # I
        assert get_pauli(term, 1, 4) == 1  # X
        assert get_pauli(term, 2, 4) == 2  # Y
        assert get_pauli(term, 3, 4) == 3  # Z

    def test_set_pauli_single_qubit(self):
        term = 0
        term = set_pauli(term, 0, 1, 1)  # Set X on qubit 0
        assert term == 0b01
        assert get_pauli(term, 0, 1) == 1

    def test_set_pauli_multiple_qubits(self):
        term = 0
        term = set_pauli(term, 0, 0, 4)  # I on qubit 0
        term = set_pauli(term, 1, 1, 4)  # X on qubit 1
        term = set_pauli(term, 2, 2, 4)  # Y on qubit 2
        term = set_pauli(term, 3, 3, 4)  # Z on qubit 3
        assert term == 0b11100100

    def test_set_pauli_overwrite(self):
        term = 0b11  # Z on qubit 0
        term = set_pauli(term, 0, 1, 1)  # Change to X
        assert get_pauli(term, 0, 1) == 1

    def test_invalid_qubit_index(self):
        with pytest.raises(ValueError, match="Invalid qubit index"):
            get_pauli(0, 5, 4)

        with pytest.raises(ValueError, match="Invalid qubit index"):
            set_pauli(0, -1, 1, 4)

    def test_invalid_pauli_integer(self):
        with pytest.raises(ValueError, match="Invalid Pauli integer"):
            set_pauli(0, 0, 4, 1)


class TestStringConversion:
    """Test string to term conversion."""

    def test_term_to_string(self):
        # Term = 0b11_10_01_00 = IXYZ (qubit 0 leftmost)
        term = 0b11100100
        assert term_to_string(term, 4) == "IXYZ"

    def test_string_to_term(self):
        assert string_to_term("IXYZ", 4) == 0b11100100
        assert string_to_term("IIII", 4) == 0
        assert string_to_term("XXXX", 4) == 0b01010101

    def test_roundtrip_conversion(self):
        strings = ["IXYZ", "XYZX", "IIII", "ZZZZ", "XYZI"]
        for s in strings:
            term = string_to_term(s, len(s))
            assert term_to_string(term, len(s)) == s

    def test_invalid_string_length(self):
        with pytest.raises(ValueError, match="doesn't match nqubits"):
            string_to_term("XYZ", 4)


class TestWeightCounting:
    """Test Pauli weight counting."""

    def test_identity_weight(self):
        term = string_to_term("IIII", 4)
        assert count_weight(term, 4) == 0

    def test_single_pauli_weight(self):
        term = string_to_term("IIIIX", 5)
        assert count_weight(term, 5) == 1

    def test_multiple_pauli_weight(self):
        term = string_to_term("IXYZ", 4)
        assert count_weight(term, 4) == 3

    def test_all_pauli_weight(self):
        term = string_to_term("XYZX", 4)
        assert count_weight(term, 4) == 4


class TestPauliMultiplication:
    """Test Pauli multiplication."""

    def test_identity_multiplication(self):
        # I * X = X (phase 1)
        term1 = string_to_term("I", 1)
        term2 = string_to_term("X", 1)
        result, phase = pauli_multiply(term1, term2, 1)
        assert term_to_string(result, 1) == "X"
        assert np.isclose(phase, 1.0)

    def test_square_to_identity(self):
        # X * X = I (phase 1)
        term = string_to_term("X", 1)
        result, phase = pauli_multiply(term, term, 1)
        assert term_to_string(result, 1) == "I"
        assert np.isclose(phase, 1.0)

        # Y * Y = I
        term = string_to_term("Y", 1)
        result, phase = pauli_multiply(term, term, 1)
        assert term_to_string(result, 1) == "I"
        assert np.isclose(phase, 1.0)

        # Z * Z = I
        term = string_to_term("Z", 1)
        result, phase = pauli_multiply(term, term, 1)
        assert term_to_string(result, 1) == "I"
        assert np.isclose(phase, 1.0)

    def test_cyclic_multiplication(self):
        # X * Y = iZ
        term1 = string_to_term("X", 1)
        term2 = string_to_term("Y", 1)
        result, phase = pauli_multiply(term1, term2, 1)
        assert term_to_string(result, 1) == "Z"
        assert np.isclose(phase, 1j)

        # Y * Z = iX
        term1 = string_to_term("Y", 1)
        term2 = string_to_term("Z", 1)
        result, phase = pauli_multiply(term1, term2, 1)
        assert term_to_string(result, 1) == "X"
        assert np.isclose(phase, 1j)

        # Z * X = iY
        term1 = string_to_term("Z", 1)
        term2 = string_to_term("X", 1)
        result, phase = pauli_multiply(term1, term2, 1)
        assert term_to_string(result, 1) == "Y"
        assert np.isclose(phase, 1j)

    def test_anticyclic_multiplication(self):
        # Y * X = -iZ
        term1 = string_to_term("Y", 1)
        term2 = string_to_term("X", 1)
        result, phase = pauli_multiply(term1, term2, 1)
        assert term_to_string(result, 1) == "Z"
        assert np.isclose(phase, -1j)

        # Z * Y = -iX
        term1 = string_to_term("Z", 1)
        term2 = string_to_term("Y", 1)
        result, phase = pauli_multiply(term1, term2, 1)
        assert term_to_string(result, 1) == "X"
        assert np.isclose(phase, -1j)

        # X * Z = -iY
        term1 = string_to_term("X", 1)
        term2 = string_to_term("Z", 1)
        result, phase = pauli_multiply(term1, term2, 1)
        assert term_to_string(result, 1) == "Y"
        assert np.isclose(phase, -1j)

    def test_multi_qubit_multiplication(self):
        # (X ⊗ Y) * (Y ⊗ X) = (iZ) ⊗ (-iZ) = Z ⊗ Z with phase i*(-i) = 1
        term1 = string_to_term("XY", 2)
        term2 = string_to_term("YX", 2)
        result, phase = pauli_multiply(term1, term2, 2)
        assert term_to_string(result, 2) == "ZZ"
        assert np.isclose(phase, 1.0)  # i * (-i) = -i^2 = 1


class TestCommutation:
    """Test Pauli commutation."""

    def test_self_commutation(self):
        # Any Pauli commutes with itself
        term = string_to_term("XYZ", 3)
        assert commutes(term, term, 3)

    def test_identity_commutation(self):
        # Identity commutes with everything
        identity = string_to_term("III", 3)
        term = string_to_term("XYZ", 3)
        assert commutes(identity, term, 3)
        assert commutes(term, identity, 3)

    def test_commuting_paulis(self):
        # X_0 and X_1 commute (different qubits)
        term1 = string_to_term("XI", 2)
        term2 = string_to_term("IX", 2)
        assert commutes(term1, term2, 2)

        # X_0 Z_1 and X_0 Y_1 anticommute at position 1
        term1 = string_to_term("XZ", 2)
        term2 = string_to_term("XY", 2)
        assert not commutes(term1, term2, 2)

    def test_anticommuting_paulis(self):
        # X and Y anticommute (single position)
        term1 = string_to_term("X", 1)
        term2 = string_to_term("Y", 1)
        assert not commutes(term1, term2, 1)

        # X and Z anticommute
        term1 = string_to_term("X", 1)
        term2 = string_to_term("Z", 1)
        assert not commutes(term1, term2, 1)

    def test_two_anticommute_positions_commute(self):
        # X_0 Y_1 and Y_0 X_1 anticommute at 2 positions → commute
        term1 = string_to_term("XY", 2)
        term2 = string_to_term("YX", 2)
        assert commutes(term1, term2, 2)


class TestPauliSumProduct:
    """Test PauliSum product operation."""

    def test_product_single_terms(self):
        """Test product of PauliSums with single terms."""
        psum1 = PauliSum(2)
        psum1.add_term("XY", 2.0)

        psum2 = PauliSum(2)
        psum2.add_term("YX", 3.0)

        result = pauli_sum_product(psum1, psum2)

        # XY * YX = ZZ with phase 1, coeff 2*3 = 6
        assert len(result) == 1
        assert np.isclose(result.get_coeff("ZZ"), 6.0)

    def test_product_multiple_terms(self):
        """Test product distributes over multiple terms."""
        # (X + 2Y) * (3Z)
        psum1 = PauliSum(1)
        psum1.add_term("X", 1.0)
        psum1.add_term("Y", 2.0)

        psum2 = PauliSum(1)
        psum2.add_term("Z", 3.0)

        result = pauli_sum_product(psum1, psum2)

        # X*Z = -iY, coeff = 1*3*(-i) = -3i
        # Y*Z = iX, coeff = 2*3*i = 6i
        assert len(result) == 2
        assert np.isclose(result.get_coeff("Y"), -3j)
        assert np.isclose(result.get_coeff("X"), 6j)

    def test_product_with_merging(self):
        """Test that identical resulting terms are merged."""
        # (X + Y) * (X + Y)
        psum1 = PauliSum(1)
        psum1.add_term("X", 1.0)
        psum1.add_term("Y", 1.0)

        psum2 = PauliSum(1)
        psum2.add_term("X", 1.0)
        psum2.add_term("Y", 1.0)

        result = pauli_sum_product(psum1, psum2)

        # X*X = I, Y*Y = I, X*Y = iZ, Y*X = -iZ
        # Coefficients: I appears twice (1+1=2), Z appears with (i - i = 0)
        assert len(result) == 1  # Only I remains (Z cancels)
        assert np.isclose(result.get_coeff("I"), 2.0)

    def test_product_identity(self):
        """Test product with identity."""
        psum1 = PauliSum(2)
        psum1.add_term("XY", 2.0)

        psum2 = PauliSum(2)
        psum2.add_term("II", 3.0)

        result = pauli_sum_product(psum1, psum2)

        # XY * II = XY, coeff = 2*3 = 6
        assert len(result) == 1
        assert np.isclose(result.get_coeff("XY"), 6.0)

    def test_product_different_nqubits(self):
        """Test error on different nqubits."""
        psum1 = PauliSum(2)
        psum2 = PauliSum(3)

        with pytest.raises(ValueError, match="different nqubits"):
            pauli_sum_product(psum1, psum2)

    def test_product_associativity(self):
        """Test (A*B)*C = A*(B*C) for PauliSums."""
        A = PauliSum(2)
        A.add_term("XI", 1.0)

        B = PauliSum(2)
        B.add_term("YI", 1.0)

        C = PauliSum(2)
        C.add_term("ZI", 1.0)

        # (A*B)*C
        AB = pauli_sum_product(A, B)
        ABC_left = pauli_sum_product(AB, C)

        # A*(B*C)
        BC = pauli_sum_product(B, C)
        ABC_right = pauli_sum_product(A, BC)

        # Both should give the same result
        assert len(ABC_left) == len(ABC_right)
        for term, coeff in ABC_left:
            assert np.isclose(coeff, ABC_right.get_coeff(term))


class TestPauliToMatrix:
    """Test pauli_to_matrix function."""

    def test_identity_matrix(self):
        """Test identity operator matrix."""
        from executor.pauli_propagation.utils.pauli_algebra import pauli_to_matrix, string_to_term

        term = string_to_term("I", 1)
        matrix = pauli_to_matrix(term, 1)

        expected = np.array([[1, 0], [0, 1]], dtype=complex)
        assert np.allclose(matrix, expected)

    def test_x_matrix(self):
        """Test X operator matrix."""
        from executor.pauli_propagation.utils.pauli_algebra import pauli_to_matrix, string_to_term

        term = string_to_term("X", 1)
        matrix = pauli_to_matrix(term, 1)

        expected = np.array([[0, 1], [1, 0]], dtype=complex)
        assert np.allclose(matrix, expected)

    def test_y_matrix(self):
        """Test Y operator matrix."""
        from executor.pauli_propagation.utils.pauli_algebra import pauli_to_matrix, string_to_term

        term = string_to_term("Y", 1)
        matrix = pauli_to_matrix(term, 1)

        expected = np.array([[0, -1j], [1j, 0]], dtype=complex)
        assert np.allclose(matrix, expected)

    def test_z_matrix(self):
        """Test Z operator matrix."""
        from executor.pauli_propagation.utils.pauli_algebra import pauli_to_matrix, string_to_term

        term = string_to_term("Z", 1)
        matrix = pauli_to_matrix(term, 1)

        expected = np.array([[1, 0], [0, -1]], dtype=complex)
        assert np.allclose(matrix, expected)

    def test_two_qubit_matrix(self):
        """Test two-qubit Pauli matrix (ZZ)."""
        from executor.pauli_propagation.utils.pauli_algebra import pauli_to_matrix, string_to_term

        term = string_to_term("ZZ", 2)
        matrix = pauli_to_matrix(term, 2)

        # ZZ = Z ⊗ Z
        expected = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=complex
        )
        assert np.allclose(matrix, expected)

    def test_matrix_shape(self):
        """Test that matrix has correct shape."""
        from executor.pauli_propagation.utils.pauli_algebra import pauli_to_matrix, string_to_term

        for nqubits in [1, 2, 3]:
            term = string_to_term("I" * nqubits, nqubits)
            matrix = pauli_to_matrix(term, nqubits)
            expected_dim = 2**nqubits
            assert matrix.shape == (expected_dim, expected_dim)

    def test_hermitian(self):
        """Test that Pauli matrices are Hermitian."""
        from executor.pauli_propagation.utils.pauli_algebra import pauli_to_matrix, string_to_term

        for pauli_str in ["X", "Y", "Z", "I"]:
            term = string_to_term(pauli_str, 1)
            matrix = pauli_to_matrix(term, 1)
            # Check P† = P (Hermitian)
            assert np.allclose(matrix, matrix.conj().T)


class TestXYHelpers:
    """Test X/Y detection and counting helpers."""

    def test_contains_x_or_y(self):
        assert contains_x_or_y(string_to_term("IZZI", 4), 4) is False
        assert contains_x_or_y(string_to_term("IZYI", 4), 4) is True

    def test_count_xy(self):
        assert count_xy(string_to_term("IZZI", 4), 4) == 0
        assert count_xy(string_to_term("XYZI", 4), 4) == 2
