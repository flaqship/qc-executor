"""Frozen reference implementation of Pauli propagation semantics.

This module is a verbatim behavioral snapshot of the propagation logic as of
the start of the performance-optimization work. It intentionally duplicates
the original per-qubit-loop primitives and gate transformation rules
(including their known quirks) so that parity tests can verify that
performance changes to the production code do not alter results.

Do NOT "fix" or optimize this module; it exists to pin behavior.
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

_SINGLE_QUBIT_RULES = {
    "H": {0: (0, 1.0), 1: (3, 1.0), 2: (2, -1.0), 3: (1, 1.0)},
    "S": {0: (0, 1.0), 1: (2, 1.0), 2: (1, -1.0), 3: (3, 1.0)},
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


def _ref_transform_cnot(gate: CliffordGate, pauli_term: int):
    control = gate.qubits[0]
    target = gate.qubits[1]

    p_c = ref_get_pauli(pauli_term, control)
    p_t = ref_get_pauli(pauli_term, target)

    new_term = pauli_term
    phase = 1.0

    if p_c == 1:
        p_t_new = p_t ^ 1
        new_term = ref_set_pauli(new_term, target, p_t_new)
        if p_t == 2:
            phase *= (-1) ** ((p_c == 2 or p_t == 2) and p_c * p_t != 0)

    if p_t == 3:
        if p_c == 0:
            new_term = ref_set_pauli(new_term, control, 3)
        elif p_c == 1:
            new_term = ref_set_pauli(new_term, control, 2)
            new_term = ref_set_pauli(new_term, target, 2)
        elif p_c == 2:
            new_term = ref_set_pauli(new_term, control, 1)
            new_term = ref_set_pauli(new_term, target, 2)
            phase = -1.0
        elif p_c == 3:
            new_term = ref_set_pauli(new_term, control, 0)
    elif p_t == 2:
        if p_c == 0:
            new_term = ref_set_pauli(new_term, control, 3)
        elif p_c == 1:
            new_term = ref_set_pauli(new_term, control, 2)
            new_term = ref_set_pauli(new_term, target, 3)
        elif p_c == 2:
            new_term = ref_set_pauli(new_term, control, 1)
            new_term = ref_set_pauli(new_term, target, 3)
        elif p_c == 3:
            new_term = ref_set_pauli(new_term, control, 0)
    elif p_t == 1:
        if p_c == 1:
            new_term = ref_set_pauli(new_term, target, 0)
        elif p_c == 2:
            new_term = ref_set_pauli(new_term, target, 0)

    return new_term, complex(phase)


def _ref_transform_cz(gate: CliffordGate, pauli_term: int):
    q0, q1 = gate.qubits[0], gate.qubits[1]
    p0 = ref_get_pauli(pauli_term, q0)
    p1 = ref_get_pauli(pauli_term, q1)

    new_term = pauli_term
    phase = 1.0

    if p0 == 1:
        p1_new = p1 ^ 3 if p1 != 3 else 0
        new_term = ref_set_pauli(new_term, q1, p1_new if p1 == 0 else (p1 ^ 3) % 4)

    if p1 == 1:
        p0_new = p0 ^ 3 if p0 != 3 else 0
        new_term = ref_set_pauli(new_term, q0, p0_new if p0 == 0 else (p0 ^ 3) % 4)

    return new_term, complex(phase)


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


def ref_propagate(
    gates: List,
    observable: PauliSum,
    parameters: Dict[str, float] | None = None,
    max_weight: int | None = None,
    truncate_threshold: float | None = None,
) -> PauliSum:
    """Frozen copy of propagate() semantics.

    Notes on pinned quirks:
    - Symmetry merging effectively runs only once, on the input observable:
      the per-gate propagation helpers return fresh PauliSums carrying the
      default NoSymmetry, so the per-layer merge is a no-op afterwards.
    - Without barriers, truncation runs after every single gate.
    - Parameter resolution here supports only concrete ``param_value`` and
      simple ``param_name`` lookups (test circuits use concrete angles).
    """
    if parameters is None:
        parameters = {}

    do_truncate = max_weight is not None or truncate_threshold is not None
    result = observable.copy()

    # Initial symmetry merge (the only one that ever fires; see docstring).
    if result.has_active_symmetry:
        merged: Dict[int, complex] = {}
        for term, coeff in result.terms.items():
            canonical = result.symmetry.canonical_representative(term, result.nqubits)
            if canonical in merged:
                merged[canonical] += coeff
            else:
                merged[canonical] = coeff
        result.terms = merged

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
