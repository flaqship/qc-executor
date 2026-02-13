"""Truncation strategies for PauliSum.

Provides functions to reduce the number of terms in a PauliSum by removing
terms with small coefficients or high weight, enabling controlled trade-off
between accuracy and computational efficiency.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Callable
from .pauli_types import PauliSum
from .pauli_algebra import count_weight


@dataclass
class TruncationStats:
    """Statistics about a truncation operation.

    Attributes:
        terms_removed: Number of terms removed
        terms_remaining: Number of terms kept
        coeff_norm_removed: L1 norm of removed coefficients
        coeff_norm_total: L1 norm of all coefficients before truncation
    """
    terms_removed: int
    terms_remaining: int
    coeff_norm_removed: float
    coeff_norm_total: float

    @property
    def relative_error_bound(self) -> float:
        """Upper bound on relative error from truncation.

        Returns:
            Relative error bound (coeff_norm_removed / coeff_norm_total)
        """
        if self.coeff_norm_total == 0:
            return 0.0
        return self.coeff_norm_removed / self.coeff_norm_total


def truncate_by_coeff(
    psum: PauliSum,
    min_coeff: float = 1e-10,
    inplace: bool = False
) -> Tuple[PauliSum, TruncationStats]:
    """Remove terms with coefficient magnitude below threshold.

    Args:
        psum: PauliSum to truncate
        min_coeff: Minimum coefficient magnitude to keep (default: 1e-10)
        inplace: If True, modify psum in place; if False, create new PauliSum

    Returns:
        Tuple of (truncated PauliSum, truncation statistics)
    """
    # Compute total coefficient norm
    coeff_norm_total = sum(abs(coeff) for coeff in psum.terms.values())

    if inplace:
        result = psum
        # Collect terms to remove (can't modify dict during iteration)
        terms_to_remove = [
            term for term, coeff in psum.terms.items()
            if abs(coeff) < min_coeff
        ]
        coeff_norm_removed = sum(abs(psum.terms[term]) for term in terms_to_remove)

        # Remove terms
        for term in terms_to_remove:
            del result.terms[term]

        terms_removed = len(terms_to_remove)
        terms_remaining = len(result)
    else:
        result = PauliSum(psum.nqubits)
        coeff_norm_removed = 0.0
        terms_removed = 0
        terms_remaining = 0

        for term, coeff in psum:
            if abs(coeff) >= min_coeff:
                result.add_term(term, coeff)
                terms_remaining += 1
            else:
                coeff_norm_removed += abs(coeff)
                terms_removed += 1

    stats = TruncationStats(
        terms_removed=terms_removed,
        terms_remaining=terms_remaining,
        coeff_norm_removed=coeff_norm_removed,
        coeff_norm_total=coeff_norm_total
    )

    return result, stats


def truncate_by_weight(
    psum: PauliSum,
    max_weight: int,
    inplace: bool = False
) -> Tuple[PauliSum, TruncationStats]:
    """Remove terms with weight (number of non-I Paulis) above threshold.

    Args:
        psum: PauliSum to truncate
        max_weight: Maximum weight to keep
        inplace: If True, modify psum in place; if False, create new PauliSum

    Returns:
        Tuple of (truncated PauliSum, truncation statistics)
    """
    # Compute total coefficient norm
    coeff_norm_total = sum(abs(coeff) for coeff in psum.terms.values())

    if inplace:
        result = psum
        # Collect terms to remove
        terms_to_remove = [
            term for term in psum.terms.keys()
            if count_weight(term, psum.nqubits) > max_weight
        ]
        coeff_norm_removed = sum(abs(psum.terms[term]) for term in terms_to_remove)

        # Remove terms
        for term in terms_to_remove:
            del result.terms[term]

        terms_removed = len(terms_to_remove)
        terms_remaining = len(result)
    else:
        result = PauliSum(psum.nqubits)
        coeff_norm_removed = 0.0
        terms_removed = 0
        terms_remaining = 0

        for term, coeff in psum:
            if count_weight(term, psum.nqubits) <= max_weight:
                result.add_term(term, coeff)
                terms_remaining += 1
            else:
                coeff_norm_removed += abs(coeff)
                terms_removed += 1

    stats = TruncationStats(
        terms_removed=terms_removed,
        terms_remaining=terms_remaining,
        coeff_norm_removed=coeff_norm_removed,
        coeff_norm_total=coeff_norm_total
    )

    return result, stats


def truncate_combined(
    psum: PauliSum,
    min_coeff: float = 1e-10,
    max_weight: Optional[int] = None,
    custom_filter: Optional[Callable[[int, complex], bool]] = None,
    inplace: bool = False
) -> Tuple[PauliSum, TruncationStats]:
    """Apply multiple truncation criteria (AND logic).

    A term is kept if ALL of the following conditions are met:
    - Coefficient magnitude >= min_coeff
    - Weight <= max_weight (if specified)
    - custom_filter(term, coeff) returns True (if specified)

    Args:
        psum: PauliSum to truncate
        min_coeff: Minimum coefficient magnitude to keep (default: 1e-10)
        max_weight: Maximum weight to keep (None = no weight limit)
        custom_filter: Optional custom filter function(term, coeff) -> bool
        inplace: If True, modify psum in place; if False, create new PauliSum

    Returns:
        Tuple of (truncated PauliSum, truncation statistics)
    """
    # Compute total coefficient norm
    coeff_norm_total = sum(abs(coeff) for coeff in psum.terms.values())

    def should_keep(term: int, coeff: complex) -> bool:
        """Check if term should be kept."""
        # Coefficient threshold
        if abs(coeff) < min_coeff:
            return False

        # Weight threshold
        if max_weight is not None:
            if count_weight(term, psum.nqubits) > max_weight:
                return False

        # Custom filter
        if custom_filter is not None:
            if not custom_filter(term, coeff):
                return False

        return True

    if inplace:
        result = psum
        # Collect terms to remove
        terms_to_remove = [
            term for term, coeff in psum.terms.items()
            if not should_keep(term, coeff)
        ]
        coeff_norm_removed = sum(abs(psum.terms[term]) for term in terms_to_remove)

        # Remove terms
        for term in terms_to_remove:
            del result.terms[term]

        terms_removed = len(terms_to_remove)
        terms_remaining = len(result)
    else:
        result = PauliSum(psum.nqubits)
        coeff_norm_removed = 0.0
        terms_removed = 0
        terms_remaining = 0

        for term, coeff in psum:
            if should_keep(term, coeff):
                result.add_term(term, coeff)
                terms_remaining += 1
            else:
                coeff_norm_removed += abs(coeff)
                terms_removed += 1

    stats = TruncationStats(
        terms_removed=terms_removed,
        terms_remaining=terms_remaining,
        coeff_norm_removed=coeff_norm_removed,
        coeff_norm_total=coeff_norm_total
    )

    return result, stats
