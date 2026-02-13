"""Convert Qiskit operators to PauliSum.

Handles conversion from various Qiskit operator types to our internal PauliSum representation.
"""

import numpy as np
from typing import Union
from .pauli_types import PauliSum

# Qiskit imports with optional availability check
try:
    from qiskit.quantum_info import SparsePauliOp, Pauli
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def convert_operator(operator, nqubits: Union[int, None] = None) -> PauliSum:
    """Convert Qiskit operator to PauliSum.

    Supports:
    - SparsePauliOp: Sparse Pauli operator representation
    - Pauli: Single Pauli operator
    - str: Pauli string (e.g., "IXYZ")

    Args:
        operator: Qiskit operator or Pauli string
        nqubits: Number of qubits (required for string input, optional otherwise)

    Returns:
        PauliSum representation of the operator

    Raises:
        ValueError: If operator type is not supported or qiskit not available
    """
    if not QISKIT_AVAILABLE and not isinstance(operator, str):
        raise ValueError("Qiskit not available. Can only convert string operators.")

    # Handle string input
    if isinstance(operator, str):
        if nqubits is None:
            nqubits = len(operator)
        psum = PauliSum(nqubits)
        psum.add_term(operator, 1.0)
        return psum

    # Handle Qiskit types
    if QISKIT_AVAILABLE:
        if isinstance(operator, SparsePauliOp):
            return _convert_sparse_pauli_op(operator)
        elif isinstance(operator, Pauli):
            return _convert_pauli(operator)

    raise ValueError(f"Unsupported operator type: {type(operator)}")


def _convert_sparse_pauli_op(sparse_op: 'SparsePauliOp') -> PauliSum:
    """Convert Qiskit SparsePauliOp to PauliSum.

    SparsePauliOp stores Pauli operators as:
    - paulis: PauliList of Pauli strings
    - coeffs: Array of complex coefficients

    Args:
        sparse_op: Qiskit SparsePauliOp

    Returns:
        Equivalent PauliSum
    """
    nqubits = sparse_op.num_qubits
    psum = PauliSum(nqubits)

    # Iterate through Pauli terms and coefficients
    for pauli, coeff in zip(sparse_op.paulis, sparse_op.coeffs):
        # Convert Pauli to string (Qiskit uses reverse qubit ordering)
        pauli_str = pauli.to_label()

        # Qiskit labels are in reverse order (rightmost = qubit 0)
        # Our convention: leftmost = qubit 0
        # So we need to reverse the string
        pauli_str_reversed = pauli_str[::-1]

        psum.add_term(pauli_str_reversed, complex(coeff))

    return psum


def _convert_pauli(pauli: 'Pauli') -> PauliSum:
    """Convert single Qiskit Pauli to PauliSum.

    Args:
        pauli: Qiskit Pauli operator

    Returns:
        PauliSum with single term
    """
    nqubits = pauli.num_qubits
    psum = PauliSum(nqubits)

    # Convert to string and reverse (Qiskit uses opposite qubit ordering)
    pauli_str = pauli.to_label()
    pauli_str_reversed = pauli_str[::-1]

    psum.add_term(pauli_str_reversed, 1.0)
    return psum


def pauli_sum_to_sparse_pauli_op(psum: PauliSum) -> 'SparsePauliOp':
    """Convert PauliSum back to Qiskit SparsePauliOp.

    Useful for validation and testing.

    Args:
        psum: PauliSum to convert

    Returns:
        Qiskit SparsePauliOp

    Raises:
        ValueError: If Qiskit not available
    """
    if not QISKIT_AVAILABLE:
        raise ValueError("Qiskit not available")

    from .pauli_algebra import term_to_string

    pauli_strings = []
    coeffs = []

    for term, coeff in psum:
        # Convert term to string (our convention: leftmost = qubit 0)
        pauli_str = term_to_string(term, psum.nqubits)

        # Reverse for Qiskit (rightmost = qubit 0)
        pauli_str_reversed = pauli_str[::-1]

        pauli_strings.append(pauli_str_reversed)
        coeffs.append(coeff)

    if len(pauli_strings) == 0:
        # Empty operator - return zero operator
        return SparsePauliOp('I' * psum.nqubits, coeffs=[0.0])

    return SparsePauliOp(pauli_strings, coeffs=coeffs)
