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
    :func:`~executor.pauli_propagation.utils.pauli_algebra.term_to_string` so
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

        Args:
            term: Pauli term encoded as integer (2 bits per qubit, little-endian).
            nqubits: Number of qubits.

        Returns:
            Canonical representative as integer.
        """
        # Step 1: Count occurrences of each Pauli type
        # Index: 0=I, 1=X, 2=Y, 3=Z
        counts = [0, 0, 0, 0]

        for q in range(nqubits):
            # Extract Pauli for qubit q using bit manipulation
            # Shift right to bring qubit q's bits to position 0-1, then mask
            pauli = (term >> (2 * q)) & 0x3
            counts[pauli] += 1

        # Step 2: Reconstruct canonical term with Paulis in sorted order
        # Place all I's first, then all X's, then all Y's, then all Z's
        canonical = 0
        bit_pos = 0  # Current qubit position being written

        for pauli_type in range(4):  # 0=I, 1=X, 2=Y, 3=Z (ascending order)
            for _ in range(counts[pauli_type]):
                # Place this Pauli at bit_pos using bit manipulation
                # Shift pauli_type left to position 2*bit_pos, then OR into canonical
                canonical |= pauli_type << (2 * bit_pos)
                bit_pos += 1

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
