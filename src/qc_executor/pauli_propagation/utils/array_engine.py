"""Vectorized Pauli propagation engine for systems of up to 32 qubits.

Terms of a PauliSum are held in parallel numpy arrays (``uint64`` bit-encoded
terms, ``complex128`` coefficients) instead of a Python dict, and every gate
application, truncation and merge is a whole-array numpy operation. With
2 bits per qubit, terms of up to 32 qubits fit exactly in a uint64.

This module contains only pure array functions; the propagation control flow
(layer loop, engine dispatch, parameter resolution) stays in propagation.py.
Semantics mirror the dict-based path: term merging replicates
PauliSum.add_term's 1e-15 pruning, and gate transforms share the exact same
lookup tables / generator components as the dict path.

Symmetry note: this engine has no symmetry-merging step. Observables with an
active (non-trivial) symmetry strategy are routed to the dict-based path by
propagation._propagate_layers, where per-layer merging is applied. Adding a
vectorized merge (e.g. for PermutationSymmetry via popcount type-counting)
would allow lifting that restriction.
"""

from __future__ import annotations

import math

import numpy as np

from .gates import CliffordGate, Gate, PauliRotation
from .pauli_algebra import low_mask
from .pauli_types import PauliSum

# Merge/prune tolerance replicating PauliSum.add_term semantics
_TOL = 1e-15

# uint64 constants (kept as np.uint64 to avoid numpy 1.x type-promotion
# surprises when mixed with Python ints)
_ONE = np.uint64(1)
_TWO = np.uint64(2)
_THREE = np.uint64(3)

# SWAR popcount constants
_M1 = np.uint64(0x5555555555555555)
_M2 = np.uint64(0x3333333333333333)
_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_H01 = np.uint64(0x0101010101010101)
_S56 = np.uint64(56)

# Powers of i shifted by one (i^(k+1)); see propagation._I_PHASE
_I_PHASE_ARR = np.array([1j, -1.0 + 0.0j, -1j, 1.0 + 0.0j], dtype=np.complex128)

_HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")


def popcount_swar(values: np.ndarray) -> np.ndarray:
    """Per-element popcount of a uint64 array via SWAR bit tricks.

    Fallback for numpy < 2.0 (which lacks np.bitwise_count). The uint64
    multiplication deliberately wraps around; the top byte after the final
    shift holds the bit count.

    Args:
        values: uint64 array

    Returns:
        uint64 array of per-element set-bit counts
    """
    v = values.copy()
    v -= (v >> _ONE) & _M1
    v = (v & _M2) + ((v >> _TWO) & _M2)
    v = (v + (v >> np.uint64(4))) & _M4
    return (v * _H01) >> _S56


if _HAS_BITWISE_COUNT:

    def popcount_u64(values: np.ndarray) -> np.ndarray:
        """Per-element popcount of a uint64 array (numpy >= 2.0 fast path)."""
        return np.bitwise_count(values).astype(np.uint64)

else:
    popcount_u64 = popcount_swar


def psum_to_arrays(psum: PauliSum):
    """Convert a PauliSum's term dict to parallel (terms, coeffs) arrays.

    Args:
        psum: PauliSum with nqubits <= 32

    Returns:
        Tuple of (uint64 terms array, complex128 coeffs array)
    """
    count = len(psum.terms)
    terms = np.fromiter(psum.terms.keys(), dtype=np.uint64, count=count)
    coeffs = np.fromiter(psum.terms.values(), dtype=np.complex128, count=count)
    return terms, coeffs


def arrays_to_psum(terms: np.ndarray, coeffs: np.ndarray, nqubits: int) -> PauliSum:
    """Convert (terms, coeffs) arrays back to a PauliSum.

    Keys become Python ints and values Python complex, matching the dict
    path's PauliSum.terms content. The result carries the default NoSymmetry,
    matching what the dict-based per-gate helpers return.

    Args:
        terms: uint64 terms array
        coeffs: complex128 coeffs array
        nqubits: Number of qubits

    Returns:
        PauliSum with the array contents as its term dict
    """
    result = PauliSum(nqubits)
    result.terms = dict(zip(terms.tolist(), coeffs.tolist()))
    return result


def merge_and_prune(terms: np.ndarray, coeffs: np.ndarray):
    """Merge duplicate terms and drop coefficients below 1e-15.

    Mirrors the accumulate-and-prune semantics of PauliSum.add_term (up to
    float summation order, which np.unique sorts by term value).

    Args:
        terms: uint64 terms array (may contain duplicates)
        coeffs: complex128 coeffs array

    Returns:
        Tuple of merged, pruned (terms, coeffs) arrays
    """
    unique_terms, inverse = np.unique(terms, return_inverse=True)
    if len(unique_terms) != len(terms):
        # complex weights are unsupported by bincount: accumulate parts
        real = np.bincount(inverse, weights=coeffs.real, minlength=len(unique_terms))
        imag = np.bincount(inverse, weights=coeffs.imag, minlength=len(unique_terms))
        terms = unique_terms
        coeffs = real + 1j * imag

    keep = np.abs(coeffs) >= _TOL
    if not keep.all():
        terms = terms[keep]
        coeffs = coeffs[keep]
    return terms, coeffs


def apply_rotation(terms: np.ndarray, coeffs: np.ndarray, gate: PauliRotation, theta: float):
    """Vectorized Pauli-rotation propagation: cos(θ)Q + i sin(θ)PQ.

    Same math as propagation._propagate_pauli_rotation, over whole arrays.

    Args:
        terms: uint64 terms array
        coeffs: complex128 coeffs array
        gate: PauliRotation with precomputed generator components
        theta: Rotation angle in radians

    Returns:
        Tuple of transformed (terms, coeffs) arrays
    """
    mask = np.uint64(low_mask(gate.nqubits))
    gen = np.uint64(gate.generator_term)
    gen_x = np.uint64(gate.generator_x)
    gen_z = np.uint64(gate.generator_z)
    gen_k = np.uint64(gate.generator_phase_count)

    z = (terms >> _ONE) & mask
    x = (terms ^ (terms >> _ONE)) & mask

    anti = (popcount_u64((gen_x & z) ^ (gen_z & x)) & _ONE).astype(bool)
    if not anti.any():
        # Everything commutes: the sum is unchanged
        return terms, coeffs

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # Pass-through rows: commuting coefficients unchanged, anticommuting
    # ones scaled by cos(θ)
    pass_coeffs = coeffs.copy()
    pass_coeffs[anti] *= cos_t

    # sin(θ) rows: P*Q term is generator XOR term; phase exponent of the
    # product (plus the folded factor i) indexes _I_PHASE_ARR. uint64
    # wraparound in the subtraction is harmless because 4 divides 2^64.
    x_anti = x[anti]
    z_anti = z[anti]
    k = (
        gen_k
        + popcount_u64(x_anti & z_anti)
        + _TWO * popcount_u64(gen_z & x_anti)
        - popcount_u64((gen_x ^ x_anti) & (gen_z ^ z_anti))
    ) & _THREE
    sin_terms = terms[anti] ^ gen
    sin_coeffs = coeffs[anti] * (_I_PHASE_ARR[k] * sin_t)

    return merge_and_prune(
        np.concatenate((terms, sin_terms)),
        np.concatenate((pass_coeffs, sin_coeffs)),
    )


def apply_clifford(terms: np.ndarray, coeffs: np.ndarray, gate: CliffordGate):
    """Vectorized Clifford propagation via the gate's lookup table.

    Args:
        terms: uint64 terms array
        coeffs: complex128 coeffs array
        gate: CliffordGate (shares its table with the dict path)

    Returns:
        Tuple of transformed (terms, coeffs) arrays
    """
    clear_mask, shifts, bits_arr, phase_arr = gate.transform_table_numpy()

    if len(shifts) == 1:
        idx = (terms >> np.uint64(shifts[0])) & _THREE
    else:
        idx = ((terms >> np.uint64(shifts[0])) & _THREE) | (
            ((terms >> np.uint64(shifts[1])) & _THREE) << _TWO
        )

    new_terms = (terms & clear_mask) | bits_arr[idx]
    new_coeffs = coeffs * phase_arr[idx]

    # Merging is still required: phases can cancel and the (pre-existing,
    # preserved) CZ/CNOT quirks make some transforms non-bijective.
    return merge_and_prune(new_terms, new_coeffs)


def apply_gate(terms: np.ndarray, coeffs: np.ndarray, gate: Gate, param_value: float | None):
    """Apply a single gate to the array representation (Heisenberg picture).

    Mirrors propagation.propagate_single_gate.

    Args:
        terms: uint64 terms array
        coeffs: complex128 coeffs array
        gate: Gate to apply
        param_value: Angle for parametric gates

    Returns:
        Tuple of transformed (terms, coeffs) arrays
    """
    if isinstance(gate, PauliRotation):
        if param_value is None:
            raise ValueError("Pauli rotation requires parameter value (angle)")
        return apply_rotation(terms, coeffs, gate, param_value)
    if isinstance(gate, CliffordGate):
        return apply_clifford(terms, coeffs, gate)
    raise TypeError(f"Unknown gate type: {type(gate)}")


def truncate_arrays(
    terms: np.ndarray,
    coeffs: np.ndarray,
    nqubits: int,
    min_coeff: float,
    max_weight: int | None,
):
    """Vectorized truncation by coefficient magnitude and Pauli weight.

    Same removal criteria as truncation.truncate_inplace_no_stats.

    Args:
        terms: uint64 terms array
        coeffs: complex128 coeffs array
        nqubits: Number of qubits
        min_coeff: Minimum coefficient magnitude to keep
        max_weight: Maximum weight to keep (None = no weight limit)

    Returns:
        Tuple of truncated (terms, coeffs) arrays
    """
    keep = np.abs(coeffs) >= min_coeff
    if max_weight is not None:
        mask = np.uint64(low_mask(nqubits))
        weights = popcount_u64((terms | (terms >> _ONE)) & mask)
        keep &= weights <= max_weight
    if keep.all():
        return terms, coeffs
    return terms[keep], coeffs[keep]


def overlap_zero_arrays(terms: np.ndarray, coeffs: np.ndarray, nqubits: int) -> complex:
    """Vectorized expectation value on the all-zeros basis state.

    Args:
        terms: uint64 terms array
        coeffs: complex128 coeffs array
        nqubits: Number of qubits

    Returns:
        Complex expectation value (sum of coefficients of I/Z-only terms)
    """
    mask = np.uint64(low_mask(nqubits))
    iz_only = ((terms ^ (terms >> _ONE)) & mask) == 0
    return complex(coeffs[iz_only].sum())
