"""State overlap and expectation value computations.

Computes expectation values ⟨ψ|O|ψ⟩ for Pauli operator sums O
with respect to various quantum states |ψ⟩.
"""

from typing import List, Union

from .pauli_algebra import contains_x_or_y, get_pauli
from .pauli_types import PauliSum


def overlap_with_zero(psum: PauliSum) -> complex:
    """Compute ⟨0|PauliSum|0⟩ where |0⟩ is the all-zeros computational basis state.

    Only Pauli strings containing only I and Z operators contribute to this overlap.
    X and Y operators give zero because:
    - ⟨0|X|0⟩ = 0 (X flips |0⟩ to |1⟩)
    - ⟨0|Y|0⟩ = 0 (Y maps |0⟩ to i|1⟩)
    - ⟨0|I|0⟩ = 1
    - ⟨0|Z|0⟩ = 1 (Z|0⟩ = |0⟩)

    Args:
        psum: PauliSum to compute overlap with

    Returns:
        Complex expectation value ⟨0|PauliSum|0⟩
    """
    result = 0.0 + 0.0j

    for term, coeff in psum:
        # Only terms without X or Y contribute
        if not contains_x_or_y(term, psum.nqubits):
            result += coeff

    return result


def overlap_with_computational(psum: PauliSum, bitstring: Union[str, List[int]]) -> complex:
    """Compute ⟨b|PauliSum|b⟩ for computational basis state |b⟩.

    For a computational basis state |b⟩ where b is a binary string:
    - Terms with X or Y give zero (they flip bits)
    - Terms with only I and Z contribute, with sign determined by Z operators

    For each qubit i:
    - If b[i] = 0: Z_i contributes +1
    - If b[i] = 1: Z_i contributes -1

    Args:
        psum: PauliSum to compute overlap with
        bitstring: Binary string or list of 0s and 1s (e.g., "0101" or [0,1,0,1])

    Returns:
        Complex expectation value ⟨b|PauliSum|b⟩
    """
    # Convert bitstring to list of integers
    if isinstance(bitstring, str):
        bits = [int(b) for b in bitstring]
    else:
        bits = list(bitstring)

    if len(bits) != psum.nqubits:
        raise ValueError(f"Bitstring length {len(bits)} doesn't match nqubits {psum.nqubits}")

    result = 0.0 + 0.0j

    for term, coeff in psum:
        # Skip terms with X or Y (they don't contribute)
        if contains_x_or_y(term, psum.nqubits):
            continue

        # Compute sign from Z operators
        sign = 1
        for i in range(psum.nqubits):
            pauli = get_pauli(term, i, psum.nqubits)
            if pauli == 3:  # Z operator
                if bits[i] == 1:
                    sign *= -1  # Z|1⟩ = -|1⟩

        result += sign * coeff

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
