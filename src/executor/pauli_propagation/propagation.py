"""Core Pauli propagation algorithm.

Propagates observables through quantum circuits in the Heisenberg picture.
"""

from typing import Dict, List, Optional

import numpy as np

from .gates import CliffordGate, Gate, PauliRotation
from .pauli_algebra import pauli_sum_product
from .pauli_types import PauliSum
from .truncation import truncate_combined


class PropagationCache:
    """Cache for intermediate results during propagation.

    Stores main PauliSum and auxiliary PauliSum to minimize allocations.
    """

    def __init__(self, nqubits: int):
        """Initialize propagation cache.

        Args:
            nqubits: Number of qubits
        """
        self.nqubits = nqubits
        self.mainsum = PauliSum(nqubits)
        self.auxsum = PauliSum(nqubits)

    def clear(self):
        """Clear both sums."""
        self.mainsum = PauliSum(self.nqubits)
        self.auxsum = PauliSum(self.nqubits)


def _apply_symmetry_merging(psum: PauliSum) -> None:
    """Merge Pauli terms by canonical symmetry (in-place).

    Groups terms that are equivalent under PauliSum's symmetry strategy
    and combines their coefficients.

    Algorithm:
        1. For each term in psum.terms:
            a. Compute canonical representative via symmetry strategy
            b. Accumulate coefficient under canonical term
        2. Replace psum.terms with merged dictionary

    Time: O(T * n) where T = number of terms, n = number of qubits
    Space: O(T) for intermediate merged_terms dict

    Example:
        If psum has terms XXY, XYX, YXX with PermutationSymmetry:
            - All three map to canonical XXY
            - Coefficients are summed under canonical term
            - Result has only one term with combined coefficient

    Args:
        psum: PauliSum to merge (modified in-place)
    """
    if not psum.has_active_symmetry:
        return  # No merging needed

    merged_terms = {}
    for term, coeff in psum.terms.items():
        # Compute canonical representative using symmetry strategy
        canonical = psum.symmetry.canonical_representative(term, psum.nqubits)

        # Accumulate coefficient under canonical term
        if canonical in merged_terms:
            merged_terms[canonical] += coeff
        else:
            merged_terms[canonical] = coeff

    # Replace terms dict with merged version
    psum.terms = merged_terms


def propagate_single_gate(
    gate: Gate,
    psum: PauliSum,
    param_value: Optional[float] = None,
) -> PauliSum:
    """Propagate PauliSum through a single gate (Heisenberg picture).

    For parametric gates: exp(-i θ/2 P) Q exp(i θ/2 P)
    For Clifford gates: U† Q U

    Args:
        gate: Gate to apply
        psum: Input PauliSum observable
        param_value: Parameter value for parametric gates (angle in radians)

    Returns:
        Transformed PauliSum
    """
    if isinstance(gate, PauliRotation):
        return _propagate_pauli_rotation(gate, psum, param_value)
    elif isinstance(gate, CliffordGate):
        return _propagate_clifford(gate, psum)
    else:
        raise TypeError(f"Unknown gate type: {type(gate)}")


def _propagate_pauli_rotation(
    gate: PauliRotation,
    psum: PauliSum,
    theta: Optional[float],
) -> PauliSum:
    """Propagate through Pauli rotation: exp(-i θ/2 P) Q exp(i θ/2 P).

    Uses the formula:
    - If [P, Q] = 0: result = Q (commuting)
    - If {P, Q} = 0: result = cos(θ) Q + sin(θ) (PQ - QP)/(2i)
                            = cos(θ) Q + sin(θ) [P, Q]/(2i)
                            = cos(θ) Q - i sin(θ) PQ  (for anticommuting P, Q)

    For anticommuting Paulis: PQ = iR for some Pauli R, so:
    result = cos(θ) Q + sin(θ) R

    Args:
        gate: Pauli rotation gate
        psum: Input PauliSum
        theta: Rotation angle in radians

    Returns:
        Transformed PauliSum
    """
    if theta is None:
        raise ValueError("Pauli rotation requires parameter value (angle)")

    result = PauliSum(psum.nqubits)

    for term, coeff in psum:
        if gate.commutes_with(term):
            # Commuting terms pass through unchanged
            result.add_term(term, coeff)
        else:
            # Anticommuting terms: split into cos and sin components
            # cos(θ) * Q term
            cos_coeff = coeff * np.cos(theta)
            result.add_term(term, cos_coeff)

            # sin(θ) * R term from rotation formula
            # Compute P * Q
            from .pauli_algebra import pauli_multiply

            pq_term, pq_phase = pauli_multiply(gate.generator_term, term, psum.nqubits)
            # The full coefficient is: coeff * sin(θ) * i * pq_phase
            # For exp(iθP/2) Q exp(-iθP/2) = cos(θ)Q + i*sin(θ)PQ
            sin_coeff = coeff * np.sin(theta) * (1j) * pq_phase
            result.add_term(pq_term, sin_coeff)

    return result


def _propagate_clifford(gate: CliffordGate, psum: PauliSum) -> PauliSum:
    """Propagate through Clifford gate: U† Q U.

    Clifford gates map Paulis to Paulis (up to phase).

    Args:
        gate: Clifford gate
        psum: Input PauliSum

    Returns:
        Transformed PauliSum
    """
    result = PauliSum(psum.nqubits)

    for term, coeff in psum:
        # Transform the Pauli term
        new_term, phase = gate.transform_pauli_term(term)

        # Add with combined coefficient
        new_coeff = coeff * phase
        result.add_term(new_term, new_coeff)

    return result


def _resolve_param_value(
    gate: Gate,
    parameters: Dict[str, float],
) -> Optional[float]:
    """Resolve the parameter value for a parametric gate.

    Args:
        gate: Parametric gate whose parameter value should be resolved
        parameters: Dict mapping parameter names to values

    Returns:
        Resolved parameter value, or None if not found
    """
    if gate.param_name and gate.param_name in parameters:
        return parameters[gate.param_name]
    if hasattr(gate, "param_value") and gate.param_value is not None:
        return gate.param_value
    return parameters.get("theta", None)


def propagate(
    gates: List[Gate],
    observable: PauliSum,
    parameters: Optional[Dict[str, float]] = None,
    max_weight: Optional[int] = None,
    truncate_threshold: Optional[float] = None,
) -> PauliSum:
    """Propagate observable through circuit (Heisenberg picture).

    In Heisenberg picture, we evolve the observable backward through the circuit:
    O → U_n† ... U_2† U_1† O U_1 U_2 ... U_n

    Since we store gates in forward order, we apply them in reverse.

    Symmetry merging (if enabled):
        If observable has active symmetry, terms are merged:
        1. Initially (before gate loop) - reduces input term count
        2. After each gate - prevents term explosion during propagation

        This grouping of equivalent Pauli strings significantly reduces
        computational cost for equivariant circuits and large molecules.

    Args:
        gates: List of gates (in circuit order)
        observable: Initial observable (PauliSum)
        parameters: Dict mapping parameter names to values
        max_weight: Maximum Pauli weight for truncation (None = no limit)
        truncate_threshold: Coefficient threshold for truncation (None = no truncation)

    Returns:
        Evolved observable
    """
    if parameters is None:
        parameters = {}

    do_truncate = max_weight is not None or truncate_threshold is not None
    result = observable.copy()

    # Initial symmetry merging (preprocessing step)
    # Reduces input terms before propagation starts
    _apply_symmetry_merging(result)

    # Apply gates in reverse order (Heisenberg picture)
    for gate in reversed(gates):
        if gate.is_parametric():
            param_value = _resolve_param_value(gate, parameters)
            result = propagate_single_gate(gate, result, param_value)
        else:
            result = propagate_single_gate(gate, result)

        # Apply symmetry merging after each gate (inline merging)
        # Groups equivalent terms before they explode
        _apply_symmetry_merging(result)

        # Truncate after each gate to prevent term explosion
        # Note: symmetry merging happens BEFORE truncation
        # This allows truncation to work on already-reduced term set
        if do_truncate:
            result, _ = truncate_combined(
                result,
                min_coeff=truncate_threshold if truncate_threshold else 1e-15,
                max_weight=max_weight,
                inplace=True,
            )

    return result


def batch_propagate(
    gates: List[Gate],
    observables: List[PauliSum],
    parameters: Optional[Dict[str, float]] = None,
    max_weight: Optional[int] = None,
    truncate_threshold: Optional[float] = None,
) -> List[PauliSum]:
    """Propagate multiple observables through a circuit in a single gate-loop pass.

    Instead of calling propagate() N times (one per observable), this function
    iterates once over the reversed gate list and applies each gate to all
    observables simultaneously. This yields an ~N× speedup for N observables
    sharing the same circuit and parameters.

    Truncation is applied independently to each PauliSum after every gate,
    preserving the same approximation behaviour as individual propagate() calls.

    Args:
        gates: List of gates (in circuit order)
        observables: List of initial observables (PauliSum), one per operator
        parameters: Dict mapping parameter names to values
        max_weight: Maximum Pauli weight for truncation (None = no limit)
        truncate_threshold: Coefficient threshold for truncation (None = no truncation)

    Returns:
        List of evolved PauliSums, one per input observable, in the same order
    """
    if not observables:
        return []

    if parameters is None:
        parameters = {}

    do_truncate = max_weight is not None or truncate_threshold is not None

    # Copy all observables so inputs are not mutated
    results = [obs.copy() for obs in observables]

    # Single pass over gates (in reverse for Heisenberg picture)
    for gate in reversed(gates):
        # Resolve parameter value once per gate (shared across all observables)
        if gate.is_parametric():
            param_value = _resolve_param_value(gate, parameters)
            results = [propagate_single_gate(gate, r, param_value) for r in results]
        else:
            results = [propagate_single_gate(gate, r) for r in results]

        # Truncate each PauliSum independently after every gate
        if do_truncate:
            results = [
                truncate_combined(
                    r,
                    min_coeff=truncate_threshold if truncate_threshold is not None else 1e-15,
                    max_weight=max_weight,
                    inplace=True,
                )[0]
                for r in results
            ]

    return results
