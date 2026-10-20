"""Pauli algebra operations and bit manipulations.

Encoding scheme: 2 bits per qubit
- 00 = I (identity)
- 01 = X
- 10 = Y
- 11 = Z

Internal encoding stays little-endian by qubit index, i.e. qubit 0 occupies
the least-significant two bits. The public string representation uses standard
mathematical convention: qubit 0 is the leftmost character.

Symplectic decomposition (used by the whole-word fast paths):
    With the low-bit mask M = 0b0101...01 (one pair per qubit), a term t
    decomposes into per-qubit component words
        z = (t >> 1) & M   (set where the qubit carries Y or Z)
        x = (t ^ (t >> 1)) & M   (set where the qubit carries X or Y)
    so that each qubit's Pauli is X^x Z^z up to the phase convention
    P(x, z) = i^(x*z) X^x Z^z (which makes Y = iXZ). In this encoding the
    Pauli product's result term is simply term1 ^ term2; only the phase needs
    popcount bookkeeping. This turns all per-qubit loops into O(1) whole-word
    integer operations (int.bit_count is available on Python >= 3.10).
"""

# The lazy import of pauli_types in pauli_sum_multiply is intentional.
# pylint: disable=cyclic-import

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

# Pauli symbol to integer mapping
SYMBOL_TO_INT = {"I": 0, "X": 1, "Y": 2, "Z": 3}
INT_TO_SYMBOL = {0: "I", 1: "X", 2: "Y", 3: "Z"}

# Powers of i, indexed by exponent mod 4
_PHASE_TABLE = (1.0 + 0.0j, 1j, -1.0 + 0.0j, -1j)


@lru_cache(maxsize=None)
def low_mask(nqubits: int) -> int:
    """Return the mask 0b0101...01 with one low bit per qubit pair."""
    return ((1 << (2 * nqubits)) - 1) // 3


def get_uint_type(nqubits: int):
    """Get appropriate numpy unsigned integer type based on number of qubits.

    Args:
        nqubits: Number of qubits

    Returns:
        numpy dtype (uint8, uint16, uint32, or uint64)
    """
    nbits = 2 * nqubits  # 2 bits per qubit
    if nbits <= 8:
        return np.uint8
    if nbits <= 16:
        return np.uint16
    if nbits <= 32:
        return np.uint32
    if nbits <= 64:
        return np.uint64
    return int


def symbol_to_int(symbol: str) -> int:
    """Convert Pauli symbol to integer representation.

    Args:
        symbol: 'I', 'X', 'Y', or 'Z'

    Returns:
        Integer encoding (0, 1, 2, or 3)
    """
    if symbol not in SYMBOL_TO_INT:
        raise ValueError(f"Invalid Pauli symbol: {symbol}. Must be 'I', 'X', 'Y', or 'Z'.")
    return SYMBOL_TO_INT[symbol]


def int_to_symbol(value: int) -> str:
    """Convert integer representation to Pauli symbol.

    Args:
        value: Integer encoding (0, 1, 2, or 3)

    Returns:
        Pauli symbol ('I', 'X', 'Y', or 'Z')
    """
    if value not in INT_TO_SYMBOL:
        raise ValueError(f"Invalid Pauli integer: {value}. Must be 0, 1, 2, or 3.")
    return INT_TO_SYMBOL[value]


def get_pauli(term: int, qubit_index: int, nqubits: int) -> int:
    """Extract Pauli operator at specific qubit index from encoded term.

    Args:
        term: Bit-encoded Pauli string
        qubit_index: Index of qubit (0 to nqubits-1)
        nqubits: Total number of qubits

    Returns:
        Pauli integer (0=I, 1=X, 2=Y, 3=Z) at that position
    """
    if qubit_index < 0 or qubit_index >= nqubits:
        raise ValueError(f"Invalid qubit index {qubit_index} for {nqubits} qubits.")

    # Each Pauli occupies 2 bits, LSB corresponds to qubit 0
    shift = 2 * qubit_index
    return (term >> shift) & 0b11


def set_pauli(term: int, qubit_index: int, pauli_int: int, nqubits: int) -> int:
    """Set Pauli operator at specific qubit index in encoded term.

    Args:
        term: Bit-encoded Pauli string
        qubit_index: Index of qubit (0 to nqubits-1)
        pauli_int: Pauli integer (0=I, 1=X, 2=Y, 3=Z)
        nqubits: Total number of qubits

    Returns:
        Modified bit-encoded Pauli string
    """
    if qubit_index < 0 or qubit_index >= nqubits:
        raise ValueError(f"Invalid qubit index {qubit_index} for {nqubits} qubits.")
    if pauli_int < 0 or pauli_int > 3:
        raise ValueError(f"Invalid Pauli integer {pauli_int}. Must be 0-3.")

    shift = 2 * qubit_index
    # Clear the 2 bits at this position, then set new value
    mask = ~(0b11 << shift)
    term = term & mask
    term = term | (pauli_int << shift)
    return term


def term_to_string(term: int, nqubits: int) -> str:
    """Convert bit-encoded term to string representation.

    Args:
        term: Bit-encoded Pauli string
        nqubits: Number of qubits

    Returns:
        String like "IXYZ..." (qubit 0 is leftmost)
    """
    symbols = []
    for qubit_index in range(nqubits):
        pauli_int = get_pauli(term, qubit_index, nqubits)
        symbols.append(int_to_symbol(pauli_int))
    return "".join(symbols)


def string_to_term(pauli_string: str, nqubits: int) -> int:
    """Convert string representation to bit-encoded term.

    Args:
        pauli_string: String like "IXYZ..." (qubit 0 is leftmost)
        nqubits: Number of qubits

    Returns:
        Bit-encoded Pauli string
    """
    if len(pauli_string) != nqubits:
        raise ValueError(f"String length {len(pauli_string)} doesn't match nqubits {nqubits}.")

    term = 0
    for string_index, symbol in enumerate(pauli_string):
        pauli_int = symbol_to_int(symbol)
        qubit_index = string_index
        term = set_pauli(term, qubit_index, pauli_int, nqubits)
    return term


def count_weight(term: int, nqubits: int) -> int:
    """Count number of non-identity Pauli operators.

    Args:
        term: Bit-encoded Pauli string
        nqubits: Number of qubits

    Returns:
        Weight (number of non-I Paulis)
    """
    term = int(term)  # ensure Python int semantics (bit_count, no overflow)
    return ((term | (term >> 1)) & low_mask(nqubits)).bit_count()


def pauli_multiply(term1: int, term2: int, nqubits: int) -> Tuple[int, complex]:
    """Multiply two Pauli strings: P1 * P2 = phase * P3.

    Uses multiplication table:
    I*I=I, I*X=X, I*Y=Y, I*Z=Z (identity)
    X*X=I, Y*Y=I, Z*Z=I (squares to identity)
    X*Y=iZ, Y*Z=iX, Z*X=iY (cyclic with i)
    Y*X=-iZ, Z*Y=-iX, X*Z=-iY (anti-cyclic with -i)

    Args:
        term1: First Pauli string
        term2: Second Pauli string
        nqubits: Number of qubits

    Returns:
        Tuple of (result_term, phase) where phase is complex (±1, ±i)
    """
    # In this encoding the result term is simply the XOR of both words.
    # The phase follows from P(x, z) = i^(x*z) X^x Z^z per qubit:
    # P(a) * P(b) = i^(x1*z1 + x2*z2 + 2*z1*x2 - x3*z3) P(a+b), summed over
    # qubits via popcounts (see module docstring).
    term1 = int(term1)  # ensure Python int semantics (bit_count, no overflow)
    term2 = int(term2)
    mask = low_mask(nqubits)
    z1 = (term1 >> 1) & mask
    x1 = (term1 ^ (term1 >> 1)) & mask
    z2 = (term2 >> 1) & mask
    x2 = (term2 ^ (term2 >> 1)) & mask

    k = (
        (x1 & z1).bit_count()
        + (x2 & z2).bit_count()
        + 2 * (z1 & x2).bit_count()
        - ((x1 ^ x2) & (z1 ^ z2)).bit_count()
    ) & 3

    return term1 ^ term2, _PHASE_TABLE[k]


def _multiply_single_pauli(p1: int, p2: int) -> Tuple[int, complex]:
    """Multiply two single-qubit Paulis.

    Multiplication table (indices: I=0, X=1, Y=2, Z=3):
         I   X   Y   Z
    I    I   X   Y   Z
    X    X   I  iZ -iY
    Y    Y -iZ   I  iX
    Z    Z  iY -iX   I

    Args:
        p1: First Pauli (0-3)
        p2: Second Pauli (0-3)

    Returns:
        Tuple of (result Pauli, phase)
    """
    # Multiplication table: result[p1][p2] = (result_pauli, phase)
    table = [
        # I  X           Y           Z
        [(0, 1), (1, 1), (2, 1), (3, 1)],  # I * {I,X,Y,Z}
        [(1, 1), (0, 1), (3, 1j), (2, -1j)],  # X * {I,X,Y,Z}
        [(2, 1), (3, -1j), (0, 1), (1, 1j)],  # Y * {I,X,Y,Z}
        [(3, 1), (2, 1j), (1, -1j), (0, 1)],  # Z * {I,X,Y,Z}
    ]

    return table[p1][p2]


def commutes(term1: int, term2: int, nqubits: int) -> bool:
    """Check if two Pauli strings commute.

    Two Pauli strings commute if they have an even number of positions
    where both are non-identity and different.

    Args:
        term1: First Pauli string
        term2: Second Pauli string
        nqubits: Number of qubits

    Returns:
        True if operators commute, False if they anticommute
    """
    # Symplectic form: parity of popcount(x1&z2 ^ z1&x2) over the qubit-
    # component words (commute iff even number of anticommuting positions).
    term1 = int(term1)  # ensure Python int semantics (bit_count, no overflow)
    term2 = int(term2)
    mask = low_mask(nqubits)
    z1 = (term1 >> 1) & mask
    x1 = (term1 ^ (term1 >> 1)) & mask
    z2 = (term2 >> 1) & mask
    x2 = (term2 ^ (term2 >> 1)) & mask

    return ((x1 & z2) ^ (z1 & x2)).bit_count() & 1 == 0


def pauli_sum_product(psum1, psum2):
    """Compute product of two PauliSums: psum1 * psum2.

    Distributes the product over all term pairs:
    (Σ a_i P_i) * (Σ b_j Q_j) = Σ_{i,j} a_i b_j (P_i * Q_j)

    Args:
        psum1: First PauliSum
        psum2: Second PauliSum

    Returns:
        New PauliSum representing the product

    Note:
        Import PauliSum inside function to avoid circular imports
    """
    from .pauli_types import PauliSum  # pylint: disable=import-outside-toplevel

    if psum1.nqubits != psum2.nqubits:
        raise ValueError(
            f"Cannot multiply PauliSums with different nqubits: {psum1.nqubits} vs {psum2.nqubits}"
        )

    result = PauliSum(psum1.nqubits)

    # Multiply every term in psum1 with every term in psum2
    for term1, coeff1 in psum1:
        for term2, coeff2 in psum2:
            # Multiply the Pauli strings
            result_term, phase = pauli_multiply(term1, term2, psum1.nqubits)
            result_coeff = coeff1 * coeff2 * phase

            # Add to result (will merge if term already exists)
            result.add_term(result_term, result_coeff)

    return result


def contains_x_or_y(term: int, nqubits: int) -> bool:
    """Check if Pauli string contains any X or Y operators.

    This is useful for expectation value computations, as X and Y observables
    give zero on computational basis states.

    Args:
        term: Bit-encoded Pauli string
        nqubits: Number of qubits

    Returns:
        True if term contains at least one X or Y operator, False otherwise
    """
    term = int(term)  # ensure Python int semantics (bit_count, no overflow)
    return ((term ^ (term >> 1)) & low_mask(nqubits)) != 0


def count_xy(term: int, nqubits: int) -> int:
    """Count number of X and Y Pauli operators in a Pauli string.

    Args:
        term: Bit-encoded Pauli string
        nqubits: Number of qubits

    Returns:
        Count of X and Y operators (excludes I and Z)
    """
    term = int(term)  # ensure Python int semantics (bit_count, no overflow)
    return ((term ^ (term >> 1)) & low_mask(nqubits)).bit_count()


def pauli_to_matrix(term: int, nqubits: int) -> np.ndarray:
    """Convert Pauli string to its 2^n × 2^n matrix representation.

    Uses Kronecker product to build the full matrix:
    P = P₀ ⊗ P₁ ⊗ ... ⊗ Pₙ₋₁

    where each Pᵢ is a single-qubit Pauli matrix.

    Args:
        term: Bit-encoded Pauli string
        nqubits: Number of qubits

    Returns:
        Complex numpy array of shape (2^n, 2^n) representing the Pauli operator
    """
    # Single-qubit Pauli matrices, indexed by encoding: I=0, X=1, Y=2, Z=3
    pauli_matrices = [
        np.array([[1, 0], [0, 1]], dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]

    # Start with scalar 1
    result = np.array([[1.0]], dtype=complex)

    # Build Kronecker product
    for i in range(nqubits):
        pauli_int = get_pauli(term, i, nqubits)
        matrix = pauli_matrices[pauli_int]
        result = np.kron(result, matrix)

    return result
