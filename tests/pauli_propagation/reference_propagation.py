"""Independent reference implementation of Pauli propagation semantics.

This module duplicates the propagation logic with deliberately simple
per-qubit loops and hand-derived transformation rules so that parity tests
can cross-validate the (optimized, table/vector-based) production code
against a structurally different implementation.

History: it started as a bug-for-bug behavioral snapshot guarding the
performance work; after the correctness fixes (CNOT/CZ/S Heisenberg phases,
per-layer symmetry merging) it now reflects the corrected semantics, with
rules derived independently from the Clifford generator maps.

Do NOT optimize this module; simplicity is its purpose.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from qc_executor.pauli_propagation.utils.gates import (
    CliffordGate,
    LayerBarrier,
    PauliRotation,
)
from qc_executor.pauli_propagation.utils.pauli_types import PauliSum

# ---------------------------------------------------------------------------
# Frozen primitives (original per-qubit loops)
# ---------------------------------------------------------------------------


def ref_get_pauli(term: int, qubit_index: int) -> int:
    return (term >> (2 * qubit_index)) & 0b11


def ref_set_pauli(term: int, qubit_index: int, pauli_int: int) -> int:
    shift = 2 * qubit_index
    term = term & ~(0b11 << shift)
    return term | (pauli_int << shift)


def ref_count_weight(term: int, nqubits: int) -> int:
    weight = 0
    for i in range(nqubits):
        if ref_get_pauli(term, i) != 0:
            weight += 1
    return weight


_MULT_TABLE = [
    [(0, 1), (1, 1), (2, 1), (3, 1)],
    [(1, 1), (0, 1), (3, 1j), (2, -1j)],
    [(2, 1), (3, -1j), (0, 1), (1, 1j)],
    [(3, 1), (2, 1j), (1, -1j), (0, 1)],
]


def ref_pauli_multiply(term1: int, term2: int, nqubits: int) -> Tuple[int, complex]:
    result_term = 0
    phase = 1.0 + 0.0j
    for i in range(nqubits):
        p1 = ref_get_pauli(term1, i)
        p2 = ref_get_pauli(term2, i)
        p_result, p_phase = _MULT_TABLE[p1][p2]
        result_term = ref_set_pauli(result_term, i, p_result)
        phase *= p_phase
    return result_term, phase


def ref_commutes(term1: int, term2: int, nqubits: int) -> bool:
    anticommute_count = 0
    for i in range(nqubits):
        p1 = ref_get_pauli(term1, i)
        p2 = ref_get_pauli(term2, i)
        if p1 != 0 and p2 != 0 and p1 != p2:
            anticommute_count += 1
    return anticommute_count % 2 == 0


def ref_contains_x_or_y(term: int, nqubits: int) -> bool:
    for i in range(nqubits):
        if ref_get_pauli(term, i) in (1, 2):
            return True
    return False


def ref_count_xy(term: int, nqubits: int) -> int:
    count = 0
    for i in range(nqubits):
        if ref_get_pauli(term, i) in (1, 2):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Frozen Clifford transformation rules (original, quirks included)
# ---------------------------------------------------------------------------

# Heisenberg rules U† P U. S: S†XS = -Y, S†YS = X. The T entry mirrors the
# legacy approximate CliffordGate("T") behavior (converters now emit RZ(π/4)
# rotations for T instead, so this entry is unused in built circuits).
_SINGLE_QUBIT_RULES = {
    "H": {0: (0, 1.0), 1: (3, 1.0), 2: (2, -1.0), 3: (1, 1.0)},
    "S": {0: (0, 1.0), 1: (2, -1.0), 2: (1, 1.0), 3: (3, 1.0)},
    "T": {
        0: (0, 1.0),
        1: (1, np.exp(-1j * np.pi / 4)),
        2: (2, np.exp(-1j * np.pi / 4)),
        3: (3, 1.0),
    },
    "X": {0: (0, 1.0), 1: (1, 1.0), 2: (2, -1.0), 3: (3, -1.0)},
    "Y": {0: (0, 1.0), 1: (1, -1.0), 2: (2, 1.0), 3: (3, -1.0)},
    "Z": {0: (0, 1.0), 1: (1, -1.0), 2: (2, -1.0), 3: (3, 1.0)},
}


def _ref_transform_single_qubit(gate: CliffordGate, pauli_term: int):
    qubit = gate.qubits[0]
    pauli_int = ref_get_pauli(pauli_term, qubit)
    new_pauli_int, phase = _SINGLE_QUBIT_RULES[gate.gate_type][pauli_int]
    new_term = ref_set_pauli(pauli_term, qubit, new_pauli_int)
    return new_term, complex(phase)


# Conjugation images φ(P) = U† P U of single-qubit Paulis on a two-qubit
# gate's (a, b) pair, encoded as local 2-qubit terms (bits 0-1 = qubit a,
# bits 2-3 = qubit b). Derived by hand from the generator maps
# (CNOT: X_c → X_c X_t, Z_c → Z_c, X_t → X_t, Z_t → Z_c Z_t;
#  CZ: X_i → X_i Z_j, Z_i → Z_i; Y = iXZ composes with phase +1).
# The full transform of P_a ⊗ P_b follows compositionally:
# φ(P_a ⊗ P_b) = φ(P_a ⊗ I) · φ(I ⊗ P_b), multiplied with the frozen
# Pauli multiplication table — an independent derivation from the explicit
# 16-entry tables used in production.
_CNOT_IMAGES_A = {0: 0b0000, 1: 0b0101, 2: 0b0110, 3: 0b0011}
_CNOT_IMAGES_B = {0: 0b0000, 1: 0b0100, 2: 0b1011, 3: 0b1111}
_CZ_IMAGES_A = {0: 0b0000, 1: 0b1101, 2: 0b1110, 3: 0b0011}
_CZ_IMAGES_B = {0: 0b0000, 1: 0b0111, 2: 0b1011, 3: 0b1100}


def _ref_conjugate_two_qubit(gate: CliffordGate, images_a, images_b, pauli_term: int):
    qubit_a, qubit_b = gate.qubits[0], gate.qubits[1]
    p_a = ref_get_pauli(pauli_term, qubit_a)
    p_b = ref_get_pauli(pauli_term, qubit_b)

    local_product, phase = ref_pauli_multiply(images_a[p_a], images_b[p_b], 2)

    new_term = ref_set_pauli(pauli_term, qubit_a, local_product & 3)
    new_term = ref_set_pauli(new_term, qubit_b, (local_product >> 2) & 3)
    return new_term, complex(phase)


def _ref_transform_cnot(gate: CliffordGate, pauli_term: int):
    return _ref_conjugate_two_qubit(gate, _CNOT_IMAGES_A, _CNOT_IMAGES_B, pauli_term)


def _ref_transform_cz(gate: CliffordGate, pauli_term: int):
    return _ref_conjugate_two_qubit(gate, _CZ_IMAGES_A, _CZ_IMAGES_B, pauli_term)


def _ref_transform_swap(gate: CliffordGate, pauli_term: int):
    q0, q1 = gate.qubits
    p0 = ref_get_pauli(pauli_term, q0)
    p1 = ref_get_pauli(pauli_term, q1)
    new_term = ref_set_pauli(pauli_term, q0, p1)
    new_term = ref_set_pauli(new_term, q1, p0)
    return new_term, 1.0


def ref_transform_pauli_term(gate: CliffordGate, pauli_term: int):
    if gate.gate_type in ["H", "S", "T", "X", "Y", "Z"]:
        return _ref_transform_single_qubit(gate, pauli_term)
    if gate.gate_type in ["CNOT", "CX"]:
        return _ref_transform_cnot(gate, pauli_term)
    if gate.gate_type == "CZ":
        return _ref_transform_cz(gate, pauli_term)
    if gate.gate_type == "SWAP":
        return _ref_transform_swap(gate, pauli_term)
    raise ValueError(f"Transformation not implemented for {gate.gate_type}")


# ---------------------------------------------------------------------------
# Frozen PauliSum term accumulation (add_term semantics)
# ---------------------------------------------------------------------------


def _ref_add_term(terms: Dict[int, complex], term: int, coeff: complex) -> None:
    term = int(term)
    if term in terms:
        terms[term] += coeff
        if abs(terms[term]) < 1e-15:
            del terms[term]
    else:
        if abs(coeff) >= 1e-15:
            terms[term] = complex(coeff)


# ---------------------------------------------------------------------------
# Frozen propagation
# ---------------------------------------------------------------------------


def _ref_propagate_rotation(gate: PauliRotation, psum: PauliSum, theta: float) -> PauliSum:
    result = PauliSum(psum.nqubits)
    for term, coeff in psum.terms.items():
        if ref_commutes(gate.generator_term, term, psum.nqubits):
            _ref_add_term(result.terms, term, coeff)
        else:
            cos_coeff = coeff * np.cos(theta)
            _ref_add_term(result.terms, term, cos_coeff)
            pq_term, pq_phase = ref_pauli_multiply(gate.generator_term, term, psum.nqubits)
            sin_coeff = coeff * np.sin(theta) * 1j * pq_phase
            _ref_add_term(result.terms, pq_term, sin_coeff)
    return result


def _ref_propagate_clifford(gate: CliffordGate, psum: PauliSum) -> PauliSum:
    result = PauliSum(psum.nqubits)
    for term, coeff in psum.terms.items():
        new_term, phase = ref_transform_pauli_term(gate, term)
        _ref_add_term(result.terms, new_term, coeff * phase)
    return result


def _ref_split_layers(gates: List) -> List[List]:
    layers: List[List] = []
    current: List = []
    for gate in gates:
        if isinstance(gate, LayerBarrier):
            if current:
                layers.append(current)
                current = []
        else:
            current.append(gate)
    if current:
        layers.append(current)
    if not layers:
        return []
    if len(layers) == 1 and len(layers[0]) == len(gates):
        return [[gate] for gate in layers[0]]
    return layers


def _ref_truncate_inplace(psum: PauliSum, min_coeff: float, max_weight: int | None) -> None:
    to_remove = []
    for term, coeff in psum.terms.items():
        if abs(coeff) < min_coeff:
            to_remove.append(term)
        elif max_weight is not None and ref_count_weight(term, psum.nqubits) > max_weight:
            to_remove.append(term)
    for term in to_remove:
        del psum.terms[term]


def _ref_merge_by_symmetry(psum: PauliSum, strategy) -> None:
    merged: Dict[int, complex] = {}
    for term, coeff in psum.terms.items():
        canonical = strategy.canonical_representative(term, psum.nqubits)
        if canonical in merged:
            merged[canonical] += coeff
        else:
            merged[canonical] = coeff
    psum.terms = merged


def ref_propagate(
    gates: List,
    observable: PauliSum,
    parameters: Dict[str, float] | None = None,
    max_weight: int | None = None,
    truncate_threshold: float | None = None,
) -> PauliSum:
    """Reference copy of propagate() semantics.

    - Symmetry merging runs on the input observable and again after each
      layer (before truncation), using the observable's strategy.
    - Without barriers, truncation runs after every single gate.
    - Parameter resolution here supports only concrete ``param_value`` and
      simple ``param_name`` lookups (test circuits use concrete angles).
    """
    if parameters is None:
        parameters = {}

    do_truncate = max_weight is not None or truncate_threshold is not None
    result = observable.copy()

    has_active_symmetry = result.has_active_symmetry
    strategy = result.symmetry

    # Initial symmetry merge (preprocessing, as in production)
    if has_active_symmetry:
        _ref_merge_by_symmetry(result, strategy)

    for layer in reversed(_ref_split_layers(gates)):
        for gate in reversed(layer):
            if isinstance(gate, PauliRotation):
                if gate.param_value is not None:
                    theta = gate.param_value
                elif gate.param_name and gate.param_name in parameters:
                    theta = parameters[gate.param_name]
                else:
                    raise ValueError("Reference propagate requires concrete angles")
                result = _ref_propagate_rotation(gate, result, theta)
            elif isinstance(gate, CliffordGate):
                result = _ref_propagate_clifford(gate, result)
            else:
                raise TypeError(f"Unknown gate type: {type(gate)}")

        # Per-layer merge before truncation (matches production order)
        if has_active_symmetry:
            _ref_merge_by_symmetry(result, strategy)

        if do_truncate:
            _ref_truncate_inplace(
                result,
                min_coeff=truncate_threshold if truncate_threshold else 1e-15,
                max_weight=max_weight,
            )

    return result


def ref_overlap_with_zero(psum: PauliSum) -> complex:
    result = 0.0 + 0.0j
    for term, coeff in psum.terms.items():
        if not ref_contains_x_or_y(term, psum.nqubits):
            result += coeff
    return result
