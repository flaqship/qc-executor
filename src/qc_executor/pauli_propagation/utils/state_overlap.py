"""State overlap and expectation value computations.

Computes expectation values for Pauli observable sums with respect to
computational-basis states.
"""

from __future__ import annotations

from typing import List

from .pauli_algebra import low_mask
from .pauli_types import PauliSum


def overlap_with_zero(psum: PauliSum) -> complex:
    """Compute the expectation value on the all-zeros basis state.

    Only Pauli strings containing only I and Z operators contribute to this overlap.
    X and Y operators give zero because:
    - X and Y terms contribute zero on computational basis states.
    - I and Z terms contribute with sign +1 for the zero state.

    Args:
        psum: PauliSum to compute overlap with

    Returns:
        Complex expectation value for the zero state.
    """
    result = 0.0 + 0.0j
    mask = low_mask(psum.nqubits)

    for term, coeff in psum.terms.items():
        # Only terms without X or Y contribute (inlined contains_x_or_y)
        if (term ^ (term >> 1)) & mask == 0:
            result += coeff

    return result


def overlap_with_computational(psum: PauliSum, bitstring: str | List[int]) -> complex:
    """Compute the expectation value on a computational basis state.

    For a computational basis state represented by a binary string:
    - Terms with X or Y give zero (they flip bits)
    - Terms with only I and Z contribute, with sign determined by Z operators

    For each qubit i:
    - If b[i] = 0: Z_i contributes +1
    - If b[i] = 1: Z_i contributes -1

    Args:
        psum: PauliSum to compute overlap with
        bitstring: Binary string or list of 0s and 1s (e.g., "0101" or [0,1,0,1])

    Returns:
        Complex expectation value for the given basis state.
    """
    # Convert bitstring to list of integers
    if isinstance(bitstring, str):
        bits = [int(b) for b in bitstring]
    else:
        bits = list(bitstring)

    if len(bits) != psum.nqubits:
        raise ValueError(f"Bitstring length {len(bits)} doesn't match nqubits {psum.nqubits}")

    # One low-mask-aligned bit per qubit whose classical bit is 1: the sign
    # of a surviving (I/Z-only) term is the parity of Z positions hitting
    # 1-bits, i.e. popcount of the Z-component word ANDed with this mask.
    ones_mask = 0
    for i, bit in enumerate(bits):
        if bit == 1:
            ones_mask |= 1 << (2 * i)

    mask = low_mask(psum.nqubits)
    result = 0.0 + 0.0j

    for term, coeff in psum.terms.items():
        # Skip terms with X or Y (they don't contribute)
        if (term ^ (term >> 1)) & mask:
            continue

        # Sign from Z operators on qubits whose bit is 1
        if ((term >> 1) & ones_mask).bit_count() & 1:
            result -= coeff
        else:
            result += coeff

    return result


def scalar_product(psum1: PauliSum, psum2: PauliSum) -> complex:
    """Compute scalar product Tr[psum1† * psum2] / 2^n.

    For Pauli operators, this simplifies to summing the product of coefficients
    for matching Pauli strings, since different Pauli strings are orthogonal:
    Tr[P_i† * P_j] / 2^n = δ_{ij}

    Args:
        psum1: First PauliSum
        psum2: Second PauliSum

    Returns:
        Complex scalar product

    Raises:
        ValueError: If PauliSums have different number of qubits
    """
    if psum1.nqubits != psum2.nqubits:
        raise ValueError(
            f"Cannot compute scalar product of PauliSums with different nqubits: "
            f"{psum1.nqubits} vs {psum2.nqubits}"
        )

    result = 0.0 + 0.0j

    # Only matching terms contribute (orthogonality of Pauli basis)
    for term, coeff1 in psum1:
        coeff2 = psum2.get_coeff(term)
        if coeff2 != 0:
            # For Pauli operators, P† = P, and Tr[P*P] = 2^n
            # So Tr[P*P] / 2^n = 1
            result += coeff1.conjugate() * coeff2

    return result
