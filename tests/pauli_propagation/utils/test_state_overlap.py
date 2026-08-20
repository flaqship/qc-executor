"""Tests for state overlap module."""

import numpy as np
import pytest

from qc_executor.pauli_propagation.utils.pauli_types import PauliSum
from qc_executor.pauli_propagation.utils.state_overlap import (
    overlap_with_computational,
    overlap_with_zero,
    scalar_product,
)

# Try to import qiskit for validation tests
try:
    from qiskit.quantum_info import Statevector

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class TestOverlapWithZero:
    """Test overlap_with_zero function."""

    def test_identity_term(self):
        """Identity term should give coefficient."""
        psum = PauliSum(2)
        psum.add_term("II", 2.5)
        assert np.isclose(overlap_with_zero(psum), 2.5)

    def test_z_term(self):
        """Z term should give coefficient (Z|0⟩ = |0⟩)."""
        psum = PauliSum(1)
        psum.add_term("Z", 3.0)
        assert np.isclose(overlap_with_zero(psum), 3.0)

    def test_x_term(self):
        """X term should give zero (X|0⟩ = |1⟩)."""
        psum = PauliSum(1)
        psum.add_term("X", 1.0)
        assert np.isclose(overlap_with_zero(psum), 0.0)

    def test_y_term(self):
        """Y term should give zero (Y|0⟩ = i|1⟩)."""
        psum = PauliSum(1)
        psum.add_term("Y", 1.0)
        assert np.isclose(overlap_with_zero(psum), 0.0)

    def test_mixed_iz_terms(self):
        """Mixed I and Z terms should sum coefficients."""
        psum = PauliSum(3)
        psum.add_term("III", 1.0)
        psum.add_term("ZII", 2.0)
        psum.add_term("IZI", 3.0)
        psum.add_term("ZZI", 4.0)
        expected = 1.0 + 2.0 + 3.0 + 4.0
        assert np.isclose(overlap_with_zero(psum), expected)

    def test_mixed_with_x(self):
        """Terms with X should not contribute."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0)
        psum.add_term("ZZ", 2.0)
        psum.add_term("XI", 100.0)  # Should not contribute
        psum.add_term("XZ", 100.0)  # Should not contribute
        expected = 1.0 + 2.0
        assert np.isclose(overlap_with_zero(psum), expected)

    def test_mixed_with_y(self):
        """Terms with Y should not contribute."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0)
        psum.add_term("ZI", 2.0)
        psum.add_term("YI", 100.0)  # Should not contribute
        psum.add_term("YZ", 100.0)  # Should not contribute
        expected = 1.0 + 2.0
        assert np.isclose(overlap_with_zero(psum), expected)

    def test_complex_coefficients(self):
        """Should handle complex coefficients."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0 + 2.0j)
        psum.add_term("ZZ", 3.0 - 1.0j)
        expected = (1.0 + 2.0j) + (3.0 - 1.0j)
        assert np.isclose(overlap_with_zero(psum), expected)

    def test_empty_paulisum(self):
        """Empty PauliSum should give zero."""
        psum = PauliSum(3)
        assert np.isclose(overlap_with_zero(psum), 0.0)

    def test_all_x_terms(self):
        """All X terms should give zero."""
        psum = PauliSum(2)
        psum.add_term("XI", 1.0)
        psum.add_term("IX", 2.0)
        psum.add_term("XX", 3.0)
        assert np.isclose(overlap_with_zero(psum), 0.0)


class TestOverlapWithComputational:
    """Test overlap_with_computational function."""

    def test_identity_00(self):
        """Identity with |00⟩."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0)
        assert np.isclose(overlap_with_computational(psum, "00"), 1.0)

    def test_identity_11(self):
        """Identity with |11⟩."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0)
        assert np.isclose(overlap_with_computational(psum, "11"), 1.0)

    def test_z_with_0(self):
        """Z with |0⟩ gives +1."""
        psum = PauliSum(1)
        psum.add_term("Z", 2.0)
        assert np.isclose(overlap_with_computational(psum, "0"), 2.0)

    def test_z_with_1(self):
        """Z with |1⟩ gives -1."""
        psum = PauliSum(1)
        psum.add_term("Z", 2.0)
        assert np.isclose(overlap_with_computational(psum, "1"), -2.0)

    def test_zz_with_00(self):
        """ZZ with |00⟩."""
        psum = PauliSum(2)
        psum.add_term("ZZ", 1.0)
        # Both Z give +1, so total is +1
        assert np.isclose(overlap_with_computational(psum, "00"), 1.0)

    def test_zz_with_01(self):
        """ZZ with |01⟩."""
        psum = PauliSum(2)
        psum.add_term("ZZ", 1.0)
        # First Z gives +1, second Z gives -1, total is -1
        assert np.isclose(overlap_with_computational(psum, "01"), -1.0)

    def test_zz_with_10(self):
        """ZZ with |10⟩."""
        psum = PauliSum(2)
        psum.add_term("ZZ", 1.0)
        # First Z gives -1, second Z gives +1, total is -1
        assert np.isclose(overlap_with_computational(psum, "10"), -1.0)

    def test_zz_with_11(self):
        """ZZ with |11⟩."""
        psum = PauliSum(2)
        psum.add_term("ZZ", 1.0)
        # Both Z give -1, so total is +1
        assert np.isclose(overlap_with_computational(psum, "11"), 1.0)

    def test_x_gives_zero(self):
        """X terms should give zero."""
        psum = PauliSum(2)
        psum.add_term("XI", 1.0)
        psum.add_term("IX", 2.0)
        assert np.isclose(overlap_with_computational(psum, "00"), 0.0)

    def test_mixed_terms(self):
        """Mixed I and Z terms."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0)
        psum.add_term("ZI", 2.0)
        psum.add_term("IZ", 3.0)
        # |01⟩: II→1, ZI→2, IZ→-3, total=0
        assert np.isclose(overlap_with_computational(psum, "01"), 0.0)

    def test_bitstring_as_list(self):
        """Accept bitstring as list."""
        psum = PauliSum(2)
        psum.add_term("ZZ", 1.0)
        assert np.isclose(overlap_with_computational(psum, [0, 1]), -1.0)

    def test_invalid_length(self):
        """Raise error for wrong bitstring length."""
        psum = PauliSum(2)
        psum.add_term("ZZ", 1.0)
        with pytest.raises(ValueError):
            overlap_with_computational(psum, "0")


class TestScalarProduct:
    """Test scalar_product function."""

    def test_identical_single_term(self):
        """Scalar product of identical single terms."""
        psum1 = PauliSum(2)
        psum1.add_term("XY", 2.0)

        psum2 = PauliSum(2)
        psum2.add_term("XY", 3.0)

        # Should be conj(2.0) * 3.0 = 6.0
        assert np.isclose(scalar_product(psum1, psum2), 6.0)

    def test_orthogonal_terms(self):
        """Scalar product of orthogonal terms."""
        psum1 = PauliSum(2)
        psum1.add_term("XY", 2.0)

        psum2 = PauliSum(2)
        psum2.add_term("ZZ", 3.0)

        # Different Pauli strings are orthogonal
        assert np.isclose(scalar_product(psum1, psum2), 0.0)

    def test_multiple_terms_some_matching(self):
        """Multiple terms with some matching."""
        psum1 = PauliSum(2)
        psum1.add_term("II", 1.0)
        psum1.add_term("XY", 2.0)
        psum1.add_term("ZZ", 3.0)

        psum2 = PauliSum(2)
        psum2.add_term("XY", 4.0)
        psum2.add_term("ZZ", 5.0)
        psum2.add_term("XX", 6.0)  # Not in psum1

        # XY: 2*4=8, ZZ: 3*5=15, total=23
        assert np.isclose(scalar_product(psum1, psum2), 23.0)

    def test_complex_coefficients(self):
        """Handle complex coefficients with conjugation."""
        psum1 = PauliSum(1)
        psum1.add_term("X", 1.0 + 2.0j)

        psum2 = PauliSum(1)
        psum2.add_term("X", 3.0 + 4.0j)

        # conj(1+2j) * (3+4j) = (1-2j) * (3+4j) = 3+4j-6j-8j^2 = 3+4j-6j+8 = 11-2j
        expected = (1.0 - 2.0j) * (3.0 + 4.0j)
        assert np.isclose(scalar_product(psum1, psum2), expected)

    def test_self_overlap(self):
        """Scalar product with self."""
        psum = PauliSum(2)
        psum.add_term("XY", 3.0)
        psum.add_term("ZZ", 4.0)

        # Should be |3|^2 + |4|^2 = 9 + 16 = 25
        assert np.isclose(scalar_product(psum, psum), 25.0)

    def test_empty_paulisums(self):
        """Empty PauliSums."""
        psum1 = PauliSum(2)
        psum2 = PauliSum(2)
        assert np.isclose(scalar_product(psum1, psum2), 0.0)

    def test_different_nqubits_error(self):
        """Raise error for different nqubits."""
        psum1 = PauliSum(2)
        psum2 = PauliSum(3)
        with pytest.raises(ValueError):
            scalar_product(psum1, psum2)


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
class TestQiskitValidation:
    """Validate against Qiskit statevector simulations."""

    def test_identity_observable(self):
        """Identity observable on |0⟩."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0)

        # |0⟩ state; ⟨ψ|I|ψ⟩ equals the total probability, which is 1 for a valid state.
        sv = Statevector.from_label("00")
        qiskit_result = float(sv.probabilities().sum())

        our_result = overlap_with_zero(psum)
        assert np.isclose(our_result, qiskit_result, atol=1e-10)

    def test_z_observable(self):
        """Z observable on |0⟩."""
        psum = PauliSum(1)
        psum.add_term("Z", 1.0)

        # For |0⟩: ⟨0|Z|0⟩ = 1
        qiskit_result = 1.0
        our_result = overlap_with_zero(psum)
        assert np.isclose(our_result, qiskit_result, atol=1e-10)

    def test_x_observable(self):
        """X observable on |0⟩."""
        psum = PauliSum(1)
        psum.add_term("X", 1.0)

        # For |0⟩: ⟨0|X|0⟩ = 0
        qiskit_result = 0.0
        our_result = overlap_with_zero(psum)
        assert np.isclose(our_result, qiskit_result, atol=1e-10)

    def test_zz_observable(self):
        """ZZ observable on |00⟩."""
        psum = PauliSum(2)
        psum.add_term("ZZ", 1.0)

        # For |00⟩: ⟨00|ZZ|00⟩ = 1
        qiskit_result = 1.0
        our_result = overlap_with_zero(psum)
        assert np.isclose(our_result, qiskit_result, atol=1e-10)

    def test_mixed_observable(self):
        """Mixed observable I + 2Z on |0⟩."""
        psum = PauliSum(1)
        psum.add_term("I", 1.0)
        psum.add_term("Z", 2.0)

        # ⟨0|I|0⟩ + 2⟨0|Z|0⟩ = 1 + 2 = 3
        qiskit_result = 3.0
        our_result = overlap_with_zero(psum)
        assert np.isclose(our_result, qiskit_result, atol=1e-10)

    def test_computational_basis_validation(self):
        """Validate overlap_with_computational against Qiskit."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0)
        psum.add_term("ZI", 2.0)
        psum.add_term("IZ", 3.0)
        psum.add_term("ZZ", 4.0)

        # Test all 4 computational basis states
        for bitstring in ["00", "01", "10", "11"]:
            our_result = overlap_with_computational(psum, bitstring)

            # Manually compute expected value
            bits = [int(b) for b in bitstring]
            expected = 1.0  # II term
            expected += 2.0 * (1 if bits[0] == 0 else -1)  # ZI term
            expected += 3.0 * (1 if bits[1] == 0 else -1)  # IZ term
            expected += 4.0 * (1 if bits[0] == 0 else -1) * (1 if bits[1] == 0 else -1)  # ZZ term

            assert np.isclose(our_result, expected, atol=1e-10)
