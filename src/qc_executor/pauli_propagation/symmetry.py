"""Symmetry strategies for Pauli propagation.

This module defines pluggable strategies that map Pauli terms to canonical
representatives so equivalent terms can be merged during propagation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class SymmetryStrategy(ABC):
    """Abstract base class for symmetry strategies.

    Symmetry strategies define how Pauli terms are grouped into equivalence
    classes. Each strategy must implement canonical_representative() to map
    a Pauli term to its canonical form.

    Canonical Representative:
        A unique representative element from each equivalence class under the
        symmetry transformation. Terms with the same canonical representative
        are considered equivalent and can be merged (coefficients summed).

    Example:
        For qubit permutation symmetry, the strings XXY, XYX, YXX all have
        the same multiset {X, X, Y} and should map to the same canonical form.

    Implementation Notes:
        - Pauli terms are encoded as integers (2 bits per qubit)
        - Encoding: I=00, X=01, Y=10, Z=11 (little-endian)
        - canonical_representative() must be deterministic
        - canonical_representative() should be fast (called many times)
    """

    @abstractmethod
    def canonical_representative(self, term: int, nqubits: int) -> int:
        """Compute canonical representative of a Pauli term.

        Maps a Pauli term to its canonical form under this symmetry.
        All terms in the same equivalence class must map to the same canonical.

        Args:
            term: Pauli term encoded as integer (2 bits per qubit, little-endian)
            nqubits: Number of qubits

        Returns:
            Canonical representative (integer, same encoding as input)
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable name of this symmetry strategy.

        Used for debugging, logging, and statistics tracking.
        """

    def __eq__(self, other: object) -> bool:
        """Compare by class and name.

        Strategies are stateless apart from :class:`CompositeSymmetry`, whose
        name lists its members, so the name determines behaviour.
        """
        return type(self) is type(other) and self.name == other.name  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        """Hash by class and name.

        Without this, a strategy passed to ``transpile_operator`` would fall
        back to identity hashing in the executor's cache key, so two equivalent
        strategies would miss each other's cache entries.
        """
        return hash((type(self).__name__, self.name))


class NoSymmetry(SymmetryStrategy):
    """Identity symmetry strategy (no merging).

    This strategy performs no merging - each Pauli term is its own canonical
    representative. Used as the default symmetry when no specific symmetry
    is requested.

    Useful for:
        - Backward compatibility (default behavior)
        - Baseline comparisons (measuring impact of symmetry merging)
        - Debugging (disable merging without changing code structure)

    Performance:
        - O(1) per canonical call (identity function)
        - Zero computational overhead
    """

    def canonical_representative(self, term: int, nqubits: int) -> int:
        """Return term unchanged (identity function).

        Args:
            term: Pauli term encoded as integer
            nqubits: Number of qubits (unused)

        Returns:
            The input term unchanged
        """
        return term

    @property
    def name(self) -> str:
        """Return 'no_symmetry' as identifier."""
        return "no_symmetry"


def _decode_pauli_to_string(term: int, nqubits: int) -> str:
    """Convert Pauli term integer to human-readable string.

    Helper function for debugging and testing. Delegates to
    :func:`~qc_executor.pauli_propagation.utils.pauli_algebra.term_to_string` so
    that the qubit-ordering convention is defined in a single place.

    Args:
        term: Pauli term encoded as integer
        nqubits: Number of qubits

    Returns:
        Human-readable Pauli string (e.g., "ZYXI", qubit 0 is leftmost)
    """
    # Imported lazily to avoid a circular import with the utils package.
    from .utils.pauli_algebra import term_to_string  # pylint: disable=import-outside-toplevel

    return term_to_string(term, nqubits)


class PermutationSymmetry(SymmetryStrategy):
    """Qubit permutation symmetry (S_n) using sorted multiset canonical form.

    Groups Pauli terms that differ only by qubit permutations. The canonical
    representative is the term with Paulis sorted lexicographically (I < X < Y < Z).

    Mathematical Background:
        Two Pauli strings P and Q are in the same orbit under S_n (symmetric group
        of n elements) if they have the same multiset of local Pauli operators.

        Example: XXY, XYX, YXX all have multiset {X, X, Y} → same orbit

    Implementation:
        Uses O(n) bit manipulation without lookup tables. Canonical form is computed
        by counting occurrences of each Pauli type and reconstructing in sorted order.

    Algorithm Complexity:
        Time: O(n) per canonical computation (one pass to count, one to reconstruct)
        Space: O(1) (fixed-size counter array [4])

    Scales efficiently to 100+ qubits.

    Example:
        >>> sym = PermutationSymmetry()
        >>> # Encode XXY, XYX, YXX at 3 qubits
        >>> term1 = 0x05  # XXY binary: 010001
        >>> term2 = 0x09  # XYX binary: 100001
        >>> term3 = 0x06  # YXX binary: 000110
        >>> sym.canonical_representative(term1, 3)  # All map to same value
        >>> sym.canonical_representative(term2, 3)
        >>> sym.canonical_representative(term3, 3)
    """

    def canonical_representative(self, term: int, nqubits: int) -> int:
        """Compute canonical Pauli term under qubit permutations.

        The canonical form is reconstructed from the multiset of local Pauli
        symbols in sorted order ``I < X < Y < Z``.

        Uses whole-word popcounts to count Pauli types and repeated-pattern
        arithmetic to reconstruct the sorted form, so the cost is O(1) word
        operations instead of a per-qubit Python loop.

        Args:
            term: Pauli term encoded as integer (2 bits per qubit, little-endian).
            nqubits: Number of qubits.

        Returns:
            Canonical representative as integer.
        """
        # Step 1: Count occurrences of each Pauli type via the low/high bit
        # planes of the 2-bit encoding (X=01, Y=10, Z=11).
        term = int(term)
        mask = ((1 << (2 * nqubits)) - 1) // 3  # 0b0101...01, one per qubit
        low = term & mask
        high = (term >> 1) & mask

        n_z = (low & high).bit_count()
        n_x = (low & ~high & mask).bit_count()
        n_y = (high & ~low & mask).bit_count()
        n_i = nqubits - n_x - n_y - n_z

        # Step 2: Reconstruct with I's first, then X's, Y's, Z's. A run of k
        # identical Paulis p is p * 0b0101...01 (k pairs) = p * (4^k - 1) / 3.
        run_x = ((1 << (2 * n_x)) - 1) // 3
        run_y = ((1 << (2 * n_y)) - 1) // 3
        run_z = ((1 << (2 * n_z)) - 1) // 3

        canonical = run_x << (2 * n_i)
        canonical |= (2 * run_y) << (2 * (n_i + n_x))
        canonical |= (3 * run_z) << (2 * (n_i + n_x + n_y))

        return canonical

    @property
    def name(self) -> str:
        """Return 'permutation' as identifier."""
        return "permutation"


class CompositeSymmetry(SymmetryStrategy):
    """Compose multiple symmetry strategies.

    Applies multiple symmetry strategies sequentially to compute the final
    canonical representative. The result of strategy i becomes the input to
    strategy i+1.

    Use Cases:
        - Combine multiple independent symmetries (e.g., permutation + point group)
        - Layer symmetries with different granularities
        - Experimental symmetry compositions for research

    Implementation:
        Strategies are applied in the order they are provided to __init__().
        For commuting symmetries, order doesn't matter. For non-commuting
        symmetries, different orders may give different canonical forms.

    Performance:
        Time: Sum of individual strategy times
        Space: O(1) beyond storage of strategy references

    Example:
        >>> perm_sym = PermutationSymmetry()
        >>> point_group_sym = PointGroupSymmetry()  # hypothetical
        >>> composite = CompositeSymmetry(perm_sym, point_group_sym)
        >>>
        >>> # Applies permutation symmetry first, then point group
        >>> canonical = composite.canonical_representative(term, nqubits)

    Notes:
        - Empty CompositeSymmetry (no strategies) acts as NoSymmetry
        - Single strategy behaves identically to using that strategy alone
        - Ordering matters only if symmetries don't commute
    """

    def __init__(self, *strategies: SymmetryStrategy):
        """Initialize composite symmetry with multiple strategies.

        Args:
            *strategies: Variable number of SymmetryStrategy instances to compose
        """
        self.strategies = list(strategies)  # Convert tuple to list

    def canonical_representative(self, term: int, nqubits: int) -> int:
        """Apply all configured strategies sequentially.

        Args:
            term: Pauli term encoded as integer.
            nqubits: Number of qubits.

        Returns:
            Canonical representative after all strategies are applied.
        """
        canonical = term
        for strategy in self.strategies:
            canonical = strategy.canonical_representative(canonical, nqubits)
        return canonical

    @property
    def name(self) -> str:
        """Return composite name listing all strategies.

        Format: 'composite(strategy1 + strategy2 + ...)'
        """
        if not self.strategies:
            return "composite(empty)"
        names = " + ".join(s.name for s in self.strategies)
        return f"composite({names})"
