"""Core Pauli propagation algorithm.

Propagates observables through quantum circuits in the Heisenberg picture.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List

import sympy as sp

from . import array_engine
from .gates import CliffordGate, Gate, LayerBarrier, PauliRotation
from .pauli_algebra import low_mask
from .pauli_types import PauliSum
from .truncation import truncate_inplace_no_stats

# Powers of i shifted by one (i^(k+1)), indexed by the Pauli-product phase
# exponent k: folds the extra factor i from the rotation formula
# cos(θ)Q + i sin(θ)PQ into a single table lookup.
_I_PHASE = (1j, -1.0 + 0.0j, -1j, 1.0 + 0.0j)

# Adaptive engine dispatch: propagation starts on the dict-based engine and
# switches to the numpy array engine once the term count makes vectorization
# pay off — but only when every term fits in a uint64 (2 bits per qubit).
# USE_ARRAY_ENGINE is a module-level escape hatch for tests and debugging.
USE_ARRAY_ENGINE = True
_ARRAY_ENGINE_MAX_QUBITS = 32
_ARRAY_ENGINE_MIN_TERMS = 128  # empirical dict/array crossover


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

    # Hot loop: everything gate-constant is hoisted, the commute check and
    # Pauli product are inlined whole-word bit operations (see the
    # pauli_algebra module docstring), and terms are merged directly in a
    # plain dict replicating PauliSum.add_term semantics (accumulate, drop
    # magnitudes below 1e-15).
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    mask = low_mask(psum.nqubits)
    gen = gate.generator_term
    gen_x = gate.generator_x
    gen_z = gate.generator_z
    gen_k = gate.generator_phase_count

    new_terms: dict = {}

    for term, coeff in psum.terms.items():
        z = (term >> 1) & mask
        x = (term ^ (term >> 1)) & mask

        if ((gen_x & z) ^ (gen_z & x)).bit_count() & 1 == 0:
            # Commuting terms pass through unchanged
            existing = new_terms.get(term)
            if existing is None:
                if abs(coeff) >= 1e-15:
                    new_terms[term] = complex(coeff)
            else:
                existing += coeff
                if abs(existing) < 1e-15:
                    del new_terms[term]
                else:
                    new_terms[term] = existing
        else:
            # Anticommuting terms split as cos(θ)Q + i sin(θ)PQ.
            # cos(θ) * Q term
            cos_coeff = coeff * cos_t
            existing = new_terms.get(term)
            if existing is None:
                if abs(cos_coeff) >= 1e-15:
                    new_terms[term] = complex(cos_coeff)
            else:
                existing += cos_coeff
                if abs(existing) < 1e-15:
                    del new_terms[term]
                else:
                    new_terms[term] = existing

            # sin(θ) * R term: P*Q is gen XOR term; the phase exponent of
            # the product plus the extra i is looked up in _I_PHASE.
            pq_term = gen ^ term
            k = (
                gen_k
                + (x & z).bit_count()
                + 2 * (gen_z & x).bit_count()
                - ((gen_x ^ x) & (gen_z ^ z)).bit_count()
            ) & 3
            sin_coeff = coeff * sin_t * _I_PHASE[k]
            existing = new_terms.get(pq_term)
            if existing is None:
                if abs(sin_coeff) >= 1e-15:
                    new_terms[pq_term] = complex(sin_coeff)
            else:
                existing += sin_coeff
                if abs(existing) < 1e-15:
                    del new_terms[pq_term]
                else:
                    new_terms[pq_term] = existing

    result = PauliSum(psum.nqubits)
    result.terms = new_terms
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
    # Hot loop: table-driven term transformation (built lazily on the gate)
    # with direct dict merging replicating PauliSum.add_term semantics.
    transform = gate.transform_pauli_term
    new_terms: dict = {}

    for term, coeff in psum.terms.items():
        new_term, phase = transform(term)
        new_coeff = coeff * phase

        existing = new_terms.get(new_term)
        if existing is None:
            if abs(new_coeff) >= 1e-15:
                new_terms[new_term] = complex(new_coeff)
        else:
            existing += new_coeff
            if abs(existing) < 1e-15:
                del new_terms[new_term]
            else:
                new_terms[new_term] = existing

    result = PauliSum(psum.nqubits)
    result.terms = new_terms
    return result


@lru_cache(maxsize=1024)
def _compile_param_expr(expr: sp.Expr):
    """Compile a sympy expression to a fast numeric callable.

    Returns (symbol_names, callable) with symbols sorted by name. Cached at
    module level: sympy expressions hash and compare structurally, so the
    cache is shared across gates and calls. Kept off gate instances because
    lambdified functions are not picklable (required for process-based
    parallel execution).
    """
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    names = tuple(s.name for s in symbols)
    func = sp.lambdify(symbols, expr, modules="math")
    return names, func


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

        # Fast path: evaluate a pre-compiled (cached) form of the expression
        # when all its symbols have values. Mirrors the subs() result below.
        try:
            names, func = _compile_param_expr(expr)
        except Exception:  # pylint: disable=broad-except
            names, func = None, None
        if names and func is not None and all(name in parameters for name in names):
            try:
                return float(func(*(parameters[name] for name in names)))
            except Exception:  # pylint: disable=broad-except
                pass  # fall through to the subs()-based path

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

    min_coeff = truncate_threshold if truncate_threshold else 1e-15
    return _propagate_layers(layers, result, parameters, do_truncate, min_coeff, max_weight)


def _propagate_layers(
    layers: List[List[Gate]],
    result: PauliSum,
    parameters: Dict[str, float],
    do_truncate: bool,
    min_coeff: float,
    max_weight: int | None,
) -> PauliSum:
    """Run the reversed layer loop on an already-merged observable copy.

    Shared core of propagate() and batch_propagate(). Starts on the
    dict-based engine and switches once to the vectorized array engine when
    the term count reaches _ARRAY_ENGINE_MIN_TERMS (dicts are faster for
    small sums, arrays for large ones) and terms fit in uint64.

    Note on symmetry: the input must already be merged by the caller. The
    per-layer merge below only applies to the dict engine and is currently a
    no-op after the first gate (the per-gate helpers return PauliSums with
    default NoSymmetry — pinned, pre-existing behavior), which is why the
    array-engine branch does not need a merging step.

    Args:
        layers: Gate layers from _split_gates_by_barriers
        result: Observable copy to evolve (consumed; may be returned)
        parameters: Dict mapping parameter names to values
        do_truncate: Whether to truncate after each layer
        min_coeff: Coefficient threshold used when truncating
        max_weight: Maximum Pauli weight used when truncating

    Returns:
        Evolved PauliSum
    """
    nqubits = result.nqubits
    use_arrays = USE_ARRAY_ENGINE and nqubits <= _ARRAY_ENGINE_MAX_QUBITS
    arrays = None  # (terms, coeffs) numpy representation once switched

    # Apply gates in reverse order (Heisenberg picture)
    # Process layers in reverse, and gates within each layer in reverse
    for layer in reversed(layers):
        # Propagate each gate in the layer (in reverse order)
        for gate in reversed(layer):
            if gate.is_parametric():
                param_value = _resolve_param_value(gate, parameters)
            else:
                param_value = None

            if arrays is None and use_arrays and len(result.terms) >= _ARRAY_ENGINE_MIN_TERMS:
                arrays = array_engine.psum_to_arrays(result)

            if arrays is None:
                result = propagate_single_gate(gate, result, param_value)
            else:
                arrays = array_engine.apply_gate(arrays[0], arrays[1], gate, param_value)

        if arrays is None:
            # After each layer: apply symmetry merging
            # Groups equivalent terms that may have been created within the layer
            _apply_symmetry_merging(result)

            # Truncate after each layer to prevent term explosion
            # Note: symmetry merging happens BEFORE truncation
            # This allows truncation to work on already-reduced term set
            # (stats-free fast path; stats were previously discarded here)
            if do_truncate:
                truncate_inplace_no_stats(result, min_coeff=min_coeff, max_weight=max_weight)
        elif do_truncate:
            arrays = array_engine.truncate_arrays(
                arrays[0], arrays[1], nqubits, min_coeff, max_weight
            )

    if arrays is not None:
        result = array_engine.arrays_to_psum(arrays[0], arrays[1], nqubits)
    return result


def batch_propagate(
    gates: List[Gate],
    observables: List[PauliSum],
    parameters: Dict[str, float] | None = None,
    max_weight: int | None = None,
    truncate_threshold: float | None = None,
) -> List[PauliSum]:
    """Propagate multiple observables through the same circuit.

    Amortizes the shared per-circuit work (layer splitting, parameter
    handling — symbolic parameter expressions are compiled once and cached)
    across observables, and evolves one observable at a time so only a single
    intermediate term store is alive at once (lower peak memory than evolving
    all observables in lock-step).

    Each observable is propagated with exactly the same semantics as an
    individual propagate() call: initial symmetry merge, reversed layer loop,
    per-layer truncation, adaptive dict/array engine.

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
    min_coeff = truncate_threshold if truncate_threshold is not None else 1e-15

    # Split gates into layers by barriers
    # If no barriers: each gate forms own layer (per-gate granularity)
    # If barriers exist: gates between barriers form a layer (per-layer granularity)
    layers = _split_gates_by_barriers(gates)

    results = []
    for observable in observables:
        # Copy so inputs are not mutated; initial symmetry merge as in propagate()
        result = observable.copy()
        _apply_symmetry_merging(result)
        results.append(
            _propagate_layers(layers, result, parameters, do_truncate, min_coeff, max_weight)
        )

    return results
