"""Core Pauli propagation algorithm.

Propagates observables through quantum circuits in the Heisenberg picture.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .gates import CliffordGate, Gate, LayerBarrier, PauliRotation
from .pauli_algebra import pauli_multiply
from .pauli_types import PauliSum
from .truncation import truncate_combined


class PropagationCache:  # pylint: disable=too-few-public-methods
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
    param_value: float | None = None,
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
    if isinstance(gate, CliffordGate):
        return _propagate_clifford(gate, psum)
    raise TypeError(f"Unknown gate type: {type(gate)}")


def _propagate_pauli_rotation(
    gate: PauliRotation,
    psum: PauliSum,
    theta: float | None,
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
) -> float | None:
    """Resolve the parameter value for a parametric gate.

    Handles both symbolic expressions (param_expr with sympy symbols)
    and simple parameter names (param_name for backward compatibility).

    Args:
        gate: Parametric gate whose parameter value should be resolved
        parameters: Dict mapping parameter names or symbol names to values

    Returns:
        Resolved parameter value, or None if not found
    """
    # Try to resolve symbolic expression first
    if hasattr(gate, "param_expr") and gate.param_expr is not None:
        expr = gate.param_expr

        # Check if all free symbols are in parameters
        subs_dict = {}
        for symbol in expr.free_symbols:
            symbol_name = symbol.name
            if symbol_name in parameters:
                subs_dict[symbol] = parameters[symbol_name]

        # If we have substitutions, try to evaluate
        if subs_dict:
            try:
                result = expr.subs(subs_dict)
                if result.is_number:
                    return float(result)
            except (TypeError, ValueError):
                pass

    # Fallback to param_name for backward compatibility
    if gate.param_name and gate.param_name in parameters:
        return parameters[gate.param_name]

    # Fallback to concrete param_value
    if hasattr(gate, "param_value") and gate.param_value is not None:
        return gate.param_value

    # Try generic "theta" parameter
    return parameters.get("theta", None)


def _split_gates_by_barriers(gates: List) -> List[List[Gate]]:
    """Split gates into layers/groups separated by LayerBarrier markers.

    LayerBarrier objects in the gate list mark the end of a layer. This function
    groups gates into layers by splitting at each barrier.

    Layers are defined as:
    - All gates before the first barrier (if any)
    - All gates between consecutive barriers
    - All gates after the last barrier (if any)

    If the gate list contains no barriers, each gate forms its own layer.
    This ensures backward compatibility with circuits without explicit layer
    structure: gates are merged/truncated after each gate (per-gate granularity).

    Args:
        gates: List of Gate objects and LayerBarrier markers (from convert_circuit)

    Returns:
        List of layers, where each layer is a List[Gate] (barriers removed)

    Example:
        >>> gates = [RX(0), RY(0), LayerBarrier(), CX(0,1), LayerBarrier(), RZ(0)]
        >>> _split_gates_by_barriers(gates)
        [[RX(0), RY(0)], [CX(0,1)], [RZ(0)]]

        >>> gates = [RX(0), RY(0), CX(0,1), RZ(0)]  # No barriers
        >>> _split_gates_by_barriers(gates)
        [[RX(0)], [RY(0)], [CX(0,1)], [RZ(0)]]  # Each gate is own layer
    """
    layers = []
    current_layer = []

    for gate in gates:
        if isinstance(gate, LayerBarrier):
            # End of current layer
            if current_layer:
                layers.append(current_layer)
                current_layer = []
        else:
            # Accumulate gate in current layer
            current_layer.append(gate)

    # Add final layer if non-empty
    if current_layer:
        layers.append(current_layer)

    # If no barriers were found, split into per-gate layers for backward compatibility
    if not layers:
        # Empty gate list
        return []

    # Check if all layers are single-gate (no barriers were found)
    if len(layers) == 1 and len(layers[0]) == len(gates):
        # No barriers in original gate list - split into per-gate layers
        return [[gate] for gate in layers[0]]

    return layers


def propagate(
    gates: List[Gate],
    observable: PauliSum,
    parameters: Dict[str, float] | None = None,
    max_weight: int | None = None,
    truncate_threshold: float | None = None,
) -> PauliSum:
    """Propagate observable through circuit (Heisenberg picture).

    In Heisenberg picture, we evolve the observable backward through the circuit:
    O → U_n† ... U_2† U_1† O U_1 U_2 ... U_n

    Since we store gates in forward order, we apply them in reverse.

    Symmetry merging (if enabled):
        Layer-based merging strategy:
        1. Initially: Merge input observable (if has_active_symmetry)
        2. Per-layer: After propagating all gates in a layer, merge and truncate

        Layers are defined by LayerBarrier markers in the circuit:
        - If circuit has no barriers: each gate forms its own layer (per-gate merging)
        - If circuit has barriers: gates between barriers form a layer (per-layer merging)

        This strategy reduces term explosion while respecting the circuit's layer
        structure. For equivariant circuits, per-layer merging preserves equivariance
        (applying and merging within a layer maintains invariance).

    Args:
        gates: List of gates and LayerBarrier markers (from convert_circuit)
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
    # Reduces input terms before propagation starts (if observable has active symmetry)
    _apply_symmetry_merging(result)

    # Split gates into layers by barriers
    # If no barriers: each gate forms own layer (backward compatible per-gate granularity)
    # If barriers exist: gates between barriers form a layer (per-layer granularity)
    layers = _split_gates_by_barriers(gates)

    # Apply gates in reverse order (Heisenberg picture)
    # Process layers in reverse, and gates within each layer in reverse
    for layer in reversed(layers):
        # Propagate each gate in the layer (in reverse order)
        for gate in reversed(layer):
            if gate.is_parametric():
                param_value = _resolve_param_value(gate, parameters)
                result = propagate_single_gate(gate, result, param_value)
            else:
                result = propagate_single_gate(gate, result)

        # After each layer: apply symmetry merging
        # Groups equivalent terms that may have been created within the layer
        _apply_symmetry_merging(result)

        # Truncate after each layer to prevent term explosion
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
    parameters: Dict[str, float] | None = None,
    max_weight: int | None = None,
    truncate_threshold: float | None = None,
) -> List[PauliSum]:
    """Propagate multiple observables through a circuit in a single layer-loop pass.

    Instead of calling propagate() N times (one per observable), this function
    iterates once over the layers and applies all gates in each layer to all
    observables simultaneously. This yields an ~N× speedup for N observables
    sharing the same circuit and parameters.

    Symmetry merging and truncation are applied independently to each PauliSum
    after every layer, preserving the same approximation behaviour as
    individual propagate() calls.

    Layer-based merging strategy (same as propagate()):
        1. Initially: Merge all observables (if has_active_symmetry)
        2. Per-layer: After propagating a layer, merge and truncate each observable

        Layers are defined by LayerBarrier markers:
        - No barriers: each gate forms own layer (per-gate granularity)
        - With barriers: gates between barriers form a layer (per-layer granularity)

    Args:
        gates: List of gates and LayerBarrier markers (from convert_circuit)
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

    # Initial symmetry merging: reduce input observables before propagation
    # (only if observables have active symmetry)
    for r in results:
        _apply_symmetry_merging(r)

    # Split gates into layers by barriers
    # If no barriers: each gate forms own layer (per-gate granularity)
    # If barriers exist: gates between barriers form a layer (per-layer granularity)
    layers = _split_gates_by_barriers(gates)

    # Apply gates in reverse order (Heisenberg picture)
    # Process layers in reverse, and gates within each layer in reverse
    for layer in reversed(layers):
        # Propagate each gate in the layer to all observables (in reverse order)
        for gate in reversed(layer):
            # Resolve parameter value once per gate (shared across all observables)
            if gate.is_parametric():
                param_value = _resolve_param_value(gate, parameters)
                results = [propagate_single_gate(gate, r, param_value) for r in results]
            else:
                results = [propagate_single_gate(gate, r) for r in results]

        # After each layer: apply symmetry merging to all observables
        # Groups equivalent terms that may have been created within the layer
        for r in results:
            _apply_symmetry_merging(r)

        # Truncate each PauliSum independently after every layer
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
