"""Tests for truncation module."""

import pytest
import numpy as np
from executor.pauli_propagation.pauli_types import PauliSum
from executor.pauli_propagation.truncation import (
    TruncationStats,
    truncate_by_coeff,
    truncate_by_weight,
    truncate_combined,
)


class TestTruncationStats:
    """Test TruncationStats dataclass."""

    def test_relative_error_bound(self):
        """Test relative error bound calculation."""
        stats = TruncationStats(
            terms_removed=5,
            terms_remaining=10,
            coeff_norm_removed=0.1,
            coeff_norm_total=1.0
        )
        assert np.isclose(stats.relative_error_bound, 0.1)

    def test_relative_error_bound_zero_total(self):
        """Test relative error when total is zero."""
        stats = TruncationStats(
            terms_removed=0,
            terms_remaining=0,
            coeff_norm_removed=0.0,
            coeff_norm_total=0.0
        )
        assert stats.relative_error_bound == 0.0


class TestTruncateByCoeff:
    """Test coefficient-based truncation."""

    def test_remove_small_coefficients(self):
        """Remove terms with small coefficients."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0)
        psum.add_term("ZZ", 0.5)
        psum.add_term("XX", 1e-11)  # Should be removed
        psum.add_term("YY", 1e-12)  # Should be removed

        result, stats = truncate_by_coeff(psum, min_coeff=1e-10)

        assert len(result) == 2
        assert np.isclose(result.get_coeff("II"), 1.0)
        assert np.isclose(result.get_coeff("ZZ"), 0.5)
        assert result.get_coeff("XX") == 0.0
        assert result.get_coeff("YY") == 0.0

        assert stats.terms_removed == 2
        assert stats.terms_remaining == 2

    def test_keep_large_coefficients(self):
        """Keep all terms with large coefficients."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0)
        psum.add_term("ZZ", 2.0)
        psum.add_term("XX", 3.0)

        result, stats = truncate_by_coeff(psum, min_coeff=0.5)

        assert len(result) == 3
        assert stats.terms_removed == 0
        assert stats.terms_remaining == 3

    def test_edge_case_exactly_at_threshold(self):
        """Test behavior at exact threshold."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0)
        psum.add_term("ZZ", 1e-10)  # Exactly at threshold

        result, stats = truncate_by_coeff(psum, min_coeff=1e-10)

        # Should keep term at exact threshold
        assert len(result) == 2
        assert np.isclose(result.get_coeff("ZZ"), 1e-10)

    def test_stats_coefficient_norms(self):
        """Test statistics coefficient norm calculations."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0)
        psum.add_term("ZZ", 0.5)
        psum.add_term("XX", 0.01)

        result, stats = truncate_by_coeff(psum, min_coeff=0.1)

        # Total norm: 1.0 + 0.5 + 0.01 = 1.51
        assert np.isclose(stats.coeff_norm_total, 1.51)
        # Removed norm: 0.01
        assert np.isclose(stats.coeff_norm_removed, 0.01)
        # Relative error
        assert np.isclose(stats.relative_error_bound, 0.01 / 1.51)

    def test_inplace_false(self):
        """Test non-inplace truncation (default)."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0)
        psum.add_term("XX", 1e-11)

        result, stats = truncate_by_coeff(psum, min_coeff=1e-10, inplace=False)

        # Original should be unchanged
        assert len(psum) == 2
        assert abs(psum.get_coeff("XX")) > 0

        # Result should be truncated
        assert len(result) == 1
        assert result.get_coeff("XX") == 0.0

    def test_inplace_true(self):
        """Test inplace truncation."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0)
        psum.add_term("XX", 1e-11)

        result, stats = truncate_by_coeff(psum, min_coeff=1e-10, inplace=True)

        # Result should be same object as psum
        assert result is psum
        # Should be modified
        assert len(psum) == 1
        assert psum.get_coeff("XX") == 0.0

    def test_empty_paulisum(self):
        """Test truncation of empty PauliSum."""
        psum = PauliSum(2)
        result, stats = truncate_by_coeff(psum, min_coeff=1e-10)

        assert len(result) == 0
        assert stats.terms_removed == 0
        assert stats.terms_remaining == 0
        assert stats.coeff_norm_total == 0.0

    def test_complex_coefficients(self):
        """Test with complex coefficients."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0 + 2.0j)
        psum.add_term("ZZ", 1e-11 + 1e-11j)

        result, stats = truncate_by_coeff(psum, min_coeff=1e-10)

        # |1+2j| = sqrt(5) ≈ 2.236 > 1e-10, kept
        # |1e-11 + 1e-11j| = sqrt(2)*1e-11 ≈ 1.414e-11 < 1e-10, removed
        assert len(result) == 1
        assert stats.terms_removed == 1


class TestTruncateByWeight:
    """Test weight-based truncation."""

    def test_remove_high_weight_terms(self):
        """Remove terms with high weight."""
        psum = PauliSum(3)
        psum.add_term("III", 1.0)  # weight 0
        psum.add_term("ZII", 2.0)  # weight 1
        psum.add_term("ZZI", 3.0)  # weight 2
        psum.add_term("ZZZ", 4.0)  # weight 3

        result, stats = truncate_by_weight(psum, max_weight=1)

        assert len(result) == 2
        assert np.isclose(result.get_coeff("III"), 1.0)
        assert np.isclose(result.get_coeff("ZII"), 2.0)
        assert result.get_coeff("ZZI") == 0.0
        assert result.get_coeff("ZZZ") == 0.0

        assert stats.terms_removed == 2
        assert stats.terms_remaining == 2

    def test_keep_identity(self):
        """Identity (weight 0) should always pass weight filter."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0)
        psum.add_term("XY", 2.0)

        result, stats = truncate_by_weight(psum, max_weight=0)

        assert len(result) == 1
        assert np.isclose(result.get_coeff("II"), 1.0)

    def test_weight_exactly_at_threshold(self):
        """Terms at exact weight threshold should be kept."""
        psum = PauliSum(3)
        psum.add_term("ZZI", 1.0)  # weight 2
        psum.add_term("ZZZ", 2.0)  # weight 3

        result, stats = truncate_by_weight(psum, max_weight=2)

        # Weight 2 should be kept, weight 3 removed
        assert len(result) == 1
        assert np.isclose(result.get_coeff("ZZI"), 1.0)

    def test_stats_with_weight_truncation(self):
        """Test statistics for weight truncation."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0)   # weight 0
        psum.add_term("ZI", 2.0)   # weight 1
        psum.add_term("XY", 3.0)   # weight 2

        result, stats = truncate_by_weight(psum, max_weight=1)

        # Total norm: 1 + 2 + 3 = 6
        assert np.isclose(stats.coeff_norm_total, 6.0)
        # Removed norm: 3
        assert np.isclose(stats.coeff_norm_removed, 3.0)
        # Relative error
        assert np.isclose(stats.relative_error_bound, 0.5)

    def test_inplace_weight(self):
        """Test inplace weight truncation."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0)
        psum.add_term("XY", 2.0)

        result, stats = truncate_by_weight(psum, max_weight=0, inplace=True)

        assert result is psum
        assert len(psum) == 1


class TestTruncateCombined:
    """Test combined truncation with multiple criteria."""

    def test_coefficient_and_weight(self):
        """Apply both coefficient and weight filters."""
        psum = PauliSum(3)
        psum.add_term("III", 1.0)    # weight 0, coeff 1.0
        psum.add_term("ZII", 0.5)    # weight 1, coeff 0.5
        psum.add_term("ZZI", 0.01)   # weight 2, coeff 0.01
        psum.add_term("ZZZ", 2.0)    # weight 3, coeff 2.0

        result, stats = truncate_combined(
            psum,
            min_coeff=0.1,
            max_weight=2
        )

        # Should keep: III (passes both), ZII (passes both)
        # Should remove: ZZI (fails coeff), ZZZ (fails weight)
        assert len(result) == 2
        assert np.isclose(result.get_coeff("III"), 1.0)
        assert np.isclose(result.get_coeff("ZII"), 0.5)

    def test_only_coefficient_filter(self):
        """Use only coefficient filter (max_weight=None)."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0)
        psum.add_term("XY", 0.01)

        result, stats = truncate_combined(
            psum,
            min_coeff=0.1,
            max_weight=None
        )

        assert len(result) == 1
        assert np.isclose(result.get_coeff("II"), 1.0)

    def test_only_weight_filter(self):
        """Use only weight filter (min_coeff very small)."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0)
        psum.add_term("XY", 2.0)

        result, stats = truncate_combined(
            psum,
            min_coeff=1e-15,  # Effectively no coeff filter
            max_weight=0
        )

        assert len(result) == 1
        assert np.isclose(result.get_coeff("II"), 1.0)

    def test_custom_filter(self):
        """Test custom filter function."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0)
        psum.add_term("ZZ", 2.0)
        psum.add_term("XX", 3.0)
        psum.add_term("YY", 4.0)

        # Custom filter: only keep terms with even coefficient real part
        def even_coeff_filter(term, coeff):
            return int(coeff.real) % 2 == 0

        result, stats = truncate_combined(
            psum,
            min_coeff=0.1,
            custom_filter=even_coeff_filter
        )

        # Should keep: ZZ (2.0), YY (4.0)
        assert len(result) == 2
        assert np.isclose(result.get_coeff("ZZ"), 2.0)
        assert np.isclose(result.get_coeff("YY"), 4.0)

    def test_all_filters_combined(self):
        """Test coefficient + weight + custom filter."""
        psum = PauliSum(3)
        psum.add_term("III", 1.0)   # weight 0, odd coeff
        psum.add_term("ZII", 2.0)   # weight 1, even coeff
        psum.add_term("ZZI", 4.0)   # weight 2, even coeff
        psum.add_term("ZZZ", 6.0)   # weight 3, even coeff
        psum.add_term("XII", 0.01)  # weight 1, even coeff, small

        # Only even coefficients
        def even_filter(term, coeff):
            return int(coeff.real) % 2 == 0

        result, stats = truncate_combined(
            psum,
            min_coeff=0.1,
            max_weight=2,
            custom_filter=even_filter
        )

        # Should keep: ZII (even, weight 1, coeff 2), ZZI (even, weight 2, coeff 4)
        # Should remove: III (odd), ZZZ (weight 3), XII (small coeff)
        assert len(result) == 2
        assert np.isclose(result.get_coeff("ZII"), 2.0)
        assert np.isclose(result.get_coeff("ZZI"), 4.0)

    def test_inplace_combined(self):
        """Test inplace combined truncation."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0)
        psum.add_term("XY", 0.01)

        result, stats = truncate_combined(
            psum,
            min_coeff=0.1,
            inplace=True
        )

        assert result is psum
        assert len(psum) == 1


class TestTruncationIntegration:
    """Integration tests combining truncation with propagation."""

    def test_truncation_preserves_observable_type(self):
        """Truncation should preserve PauliSum structure."""
        psum = PauliSum(3)
        psum.add_term("ZZZ", 1.0)
        psum.add_term("XXX", 0.001)

        result, stats = truncate_by_coeff(psum, min_coeff=0.01)

        assert isinstance(result, PauliSum)
        assert result.nqubits == 3

    def test_error_bound_validity(self):
        """Test that error bound is meaningful."""
        psum = PauliSum(2)
        psum.add_term("II", 1.0)
        psum.add_term("ZZ", 0.1)
        psum.add_term("XX", 0.01)

        result, stats = truncate_by_coeff(psum, min_coeff=0.05)

        # Removed coefficient: 0.01
        # Total: 1.11
        # Relative error should be small
        assert stats.relative_error_bound < 0.01
        assert stats.coeff_norm_removed < stats.coeff_norm_total
