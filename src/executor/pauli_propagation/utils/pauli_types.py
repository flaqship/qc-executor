"""Core Pauli data types: PauliString and PauliSum."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from .pauli_algebra import commutes as pauli_commutes
from .pauli_algebra import (
    count_weight,
    get_uint_type,
    pauli_multiply,
    set_pauli,
    string_to_term,
    symbol_to_int,
    term_to_string,
)

if TYPE_CHECKING:
    from ..symmetry import SymmetryStrategy


class PauliString:
    """Represents a single Pauli operator on multiple qubits.

    Stores Pauli operators efficiently as bit-encoded integers:
    - 2 bits per qubit: 00=I, 01=X, 10=Y, 11=Z
    - Uses appropriate numpy uint type based on number of qubits

    Attributes:
        nqubits: Number of qubits
        term: Bit-encoded Pauli string (numpy uint)
        coeff: Complex coefficient
    """

    def __init__(self, nqubits: int, term: int | str = None, coeff: complex = 1.0):
        """Initialize a PauliString.

        Args:
            nqubits: Number of qubits
            term: Either integer (bit-encoded) or string (e.g., "IXYZ")
                  If None, initializes to all identities
            coeff: Complex coefficient (default: 1.0)
        """
        self.nqubits = nqubits
        self.dtype = get_uint_type(nqubits)

        if term is None:
            self.term = self.dtype(0)  # All identities
        elif isinstance(term, str):
            self.term = self.dtype(string_to_term(term, nqubits))
        elif isinstance(term, (int, np.integer)):
            self.term = self.dtype(term)
        else:
            raise TypeError(f"term must be int, str, or None, got {type(term)}")

        self.coeff = complex(coeff)

    @classmethod
    def from_symbols(
        cls, symbols: List[str], indices: List[int], nqubits: int, coeff: complex = 1.0
    ):
        """Create PauliString from list of Pauli symbols and qubit indices.

        Args:
            symbols: List of Pauli symbols ('I', 'X', 'Y', 'Z')
            indices: List of qubit indices where these Paulis act
            nqubits: Total number of qubits
            coeff: Complex coefficient

        Returns:
            PauliString instance

        Example:
            PauliString.from_symbols(['X', 'Z'], [0, 2], 3) creates X_0 Z_2 on 3 qubits
        """
        if len(symbols) != len(indices):
            raise ValueError("symbols and indices must have same length")

        term = 0
        for symbol, index in zip(symbols, indices):
            pauli_int = symbol_to_int(symbol)
            term = set_pauli(term, index, pauli_int, nqubits)

        return cls(nqubits, term, coeff)

    def to_string(self) -> str:
        """Convert to string representation.

        Returns:
            String like "IXYZ..." (qubit 0 is leftmost)
        """
        return term_to_string(int(self.term), self.nqubits)

    def weight(self) -> int:
        """Count number of non-identity Pauli operators.

        Returns:
            Weight (number of non-I Paulis)
        """
        return count_weight(int(self.term), self.nqubits)

    def multiply(self, other: "PauliString") -> "PauliString":
        """Multiply with another PauliString: self * other.

        Args:
            other: Another PauliString

        Returns:
            New PauliString representing the product

        Raises:
            ValueError: If nqubits don't match
        """
        if self.nqubits != other.nqubits:
            raise ValueError(
                f"Cannot multiply PauliStrings with different nqubits: {self.nqubits} vs {other.nqubits}"
            )

        result_term, phase = pauli_multiply(int(self.term), int(other.term), self.nqubits)
        result_coeff = self.coeff * other.coeff * phase

        return PauliString(self.nqubits, result_term, result_coeff)

    def commutes_with(self, other: "PauliString") -> bool:
        """Check if this PauliString commutes with another.

        Args:
            other: Another PauliString

        Returns:
            True if they commute, False if they anticommute
        """
        if self.nqubits != other.nqubits:
            raise ValueError(
                f"Cannot compare PauliStrings with different nqubits: {self.nqubits} vs {other.nqubits}"
            )

        return pauli_commutes(int(self.term), int(other.term), self.nqubits)

    def __mul__(self, scalar: complex) -> "PauliString":
        """Scalar multiplication: self * scalar.

        Args:
            scalar: Complex number

        Returns:
            New PauliString with scaled coefficient
        """
        return PauliString(self.nqubits, int(self.term), self.coeff * scalar)

    def __rmul__(self, scalar: complex) -> "PauliString":
        """Scalar multiplication: scalar * self.

        Args:
            scalar: Complex number

        Returns:
            New PauliString with scaled coefficient
        """
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        """String representation for debugging."""
        coeff_str = (
            f"{self.coeff:.3f}" if abs(self.coeff.imag) > 1e-10 else f"{self.coeff.real:.3f}"
        )
        return f"PauliString({coeff_str} * {self.to_string()})"

    def __str__(self) -> str:
        """User-friendly string representation."""
        coeff_str = (
            f"{self.coeff:.3f}" if abs(self.coeff.imag) > 1e-10 else f"{self.coeff.real:.3f}"
        )
        return f"{coeff_str} {self.to_string()}"

    def __eq__(self, other) -> bool:
        """Check equality based on term and coefficient.

        Args:
            other: Another PauliString

        Returns:
            True if same Pauli string and coefficient
        """
        if not isinstance(other, PauliString):
            return False
        return (
            self.nqubits == other.nqubits
            and int(self.term) == int(other.term)
            and np.isclose(self.coeff, other.coeff)
        )

    def __hash__(self) -> int:
        """Hash based on term (ignoring coefficient)."""
        return hash((self.nqubits, int(self.term)))


class PauliSum:
    """Represents a sum of Pauli operators: Σ coeff_i * PauliString_i.

    Uses dictionary-based storage: {term (uint) → coefficient (complex)}
    This is efficient for sparse operator representations.

    Attributes:
        nqubits: Number of qubits
        terms: Dictionary mapping bit-encoded terms to coefficients
        dtype: Numpy unsigned integer type for terms
        symmetry: Optional symmetry strategy for automatic term merging
    """

    def __init__(self, nqubits: int, symmetry: SymmetryStrategy | None = None):
        """Initialize an empty PauliSum.

        Args:
            nqubits: Number of qubits
            symmetry: Optional SymmetryStrategy instance for automatic term merging.
                     If None, defaults to NoSymmetry (no merging).

        Example:
            # No symmetry (default)
            psum = PauliSum(4)

            # With qubit permutation symmetry
            from ..symmetry import PermutationSymmetry
            psum = PauliSum(4, symmetry=PermutationSymmetry())
        """
        from ..symmetry import NoSymmetry

        self.nqubits = nqubits
        self.dtype = get_uint_type(nqubits)
        self.terms = {}  # Dict[uint term → complex coeff]
        self.symmetry = symmetry if symmetry is not None else NoSymmetry()

    def add_term(self, term: int | str | PauliString, coeff: complex = 1.0) -> None:
        """Add a Pauli term to the sum.

        If term already exists, coefficients are added.

        Args:
            term: Bit-encoded integer, string, or PauliString
            coeff: Complex coefficient (ignored if term is PauliString)
        """
        if isinstance(term, PauliString):
            if term.nqubits != self.nqubits:
                raise ValueError(f"PauliString has {term.nqubits} qubits, expected {self.nqubits}")
            term_int = int(term.term)
            coeff = term.coeff
        elif isinstance(term, str):
            term_int = string_to_term(term, self.nqubits)
        elif isinstance(term, (int, np.integer)):
            term_int = int(term)
        else:
            raise TypeError(f"term must be int, str, or PauliString, got {type(term)}")

        # Add to existing coefficient or create new entry
        if term_int in self.terms:
            self.terms[term_int] += coeff
            # Remove if coefficient becomes essentially zero
            if abs(self.terms[term_int]) < 1e-15:
                del self.terms[term_int]
        else:
            # Only add if coefficient is non-zero
            if abs(coeff) >= 1e-15:
                self.terms[term_int] = complex(coeff)

    def get_coeff(self, term: int | str) -> complex:
        """Get coefficient for a specific term.

        Args:
            term: Bit-encoded integer or string

        Returns:
            Complex coefficient (0 if term not present)
        """
        if isinstance(term, str):
            term = string_to_term(term, self.nqubits)
        return self.terms.get(int(term), 0.0)

    def set_coeff(self, term: int | str, coeff: complex) -> None:
        """Set coefficient for a specific term.

        Args:
            term: Bit-encoded integer or string
            coeff: Complex coefficient
        """
        if isinstance(term, str):
            term = string_to_term(term, self.nqubits)

        if abs(coeff) < 1e-15:  # Remove if coefficient is essentially zero
            self.terms.pop(int(term), None)
        else:
            self.terms[int(term)] = complex(coeff)

    def copy(self) -> "PauliSum":
        """Create a deep copy of this PauliSum.

        Preserves the symmetry strategy from the original PauliSum.

        Returns:
            New PauliSum with copied terms and same symmetry
        """
        new_psum = PauliSum(self.nqubits, symmetry=self.symmetry)
        new_psum.terms = self.terms.copy()
        return new_psum

    @property
    def has_active_symmetry(self) -> bool:
        """Check if this PauliSum has an active (non-trivial) symmetry.

        Returns:
            True if symmetry is not NoSymmetry, False otherwise
        """
        from ..symmetry import NoSymmetry

        return not isinstance(self.symmetry, NoSymmetry)

    def __len__(self) -> int:
        """Return number of terms in the sum."""
        return len(self.terms)

    def __iter__(self):
        """Iterate over (term, coefficient) pairs."""
        return iter(self.terms.items())

    def __add__(self, other: "PauliSum") -> "PauliSum":
        """Add two PauliSums.

        Args:
            other: Another PauliSum

        Returns:
            New PauliSum representing the sum
        """
        if self.nqubits != other.nqubits:
            raise ValueError(
                f"Cannot add PauliSums with different nqubits: {self.nqubits} vs {other.nqubits}"
            )

        result = self.copy()
        for term, coeff in other:
            result.add_term(term, coeff)
        return result

    def __mul__(self, scalar: complex) -> "PauliSum":
        """Scalar multiplication.

        Args:
            scalar: Complex number

        Returns:
            New PauliSum with all coefficients scaled
        """
        result = PauliSum(self.nqubits, symmetry=self.symmetry)
        for term, coeff in self:
            result.terms[term] = coeff * scalar
        return result

    def __rmul__(self, scalar: complex) -> "PauliSum":
        """Scalar multiplication (reverse).

        Args:
            scalar: Complex number

        Returns:
            New PauliSum with all coefficients scaled
        """
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        """String representation for debugging."""
        if len(self.terms) == 0:
            return f"PauliSum(nqubits={self.nqubits}, terms=0)"
        return f"PauliSum(nqubits={self.nqubits}, terms={len(self.terms)})"

    def __str__(self) -> str:
        """User-friendly string representation."""
        if len(self.terms) == 0:
            return "0"

        lines = []
        for term, coeff in sorted(self.terms.items()):
            coeff_str = f"{coeff:.3f}" if abs(coeff.imag) > 1e-10 else f"{coeff.real:.3f}"
            term_str = term_to_string(term, self.nqubits)
            lines.append(f"{coeff_str} {term_str}")

        return " + ".join(lines)
