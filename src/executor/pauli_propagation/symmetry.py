"""Symmetry strategies for Pauli propagation.

This module provides pluggable symmetry strategies that enable automatic merging
of Pauli terms under symmetry transformations. When a PauliSum is assigned a
SymmetryStrategy, the propagate() function automatically groups and merges
equivalent terms after each gate, reducing computational complexity.

Core Concepts:
    - SymmetryStrategy: Abstract base for symmetry types
    - NoSymmetry: Identity strategy (no merging)
    - PermutationSymmetry: S_n qubit permutations (canonical form = sorted symbol multiset)
    - CompositeSymmetry: Chain multiple strategies

Usage:
    from .symmetry import PermutationSymmetry, CompositeSymmetry

    # Single symmetry
    psum = PauliSum(8, symmetry=PermutationSymmetry())

    # Multiple symmetries
    psum = PauliSum(8, symmetry=CompositeSymmetry(
        PermutationSymmetry(),
        # PointGroupSymmetry(),  # Future
    ))

    # Automatic merging during propagation
    propagated = propagate(gates, psum)  # Auto-merges if symmetry active

Performance:
    - PermutationSymmetry: O(n) per canonical computation (no lookup table)
    - Scales to 100+ qubits efficiently
    - Zero memory overhead beyond dict reconstruction during merging

Design Notes:
    - Symmetry is owned by PauliSum (not passed to propagate())
    - Merging is transparent to caller (happens in propagate loops)
    - All strategies use bit manipulation for Pauli term encoding
    - Canonical form is always reorganized in sorted order (I, X, Y, Z)
"""

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
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable name of this symmetry strategy.

        Used for debugging, logging, and statistics tracking.
        """
        pass


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
