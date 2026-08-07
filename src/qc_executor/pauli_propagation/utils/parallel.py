"""Process-parallel execution helpers for the Pauli propagation executor.

The worker functions live at module level with picklable arguments so they
work with Windows' spawn start method (no lambdas, no closures, no state
bound to executor instances).
"""

from __future__ import annotations

from typing import Dict, Tuple

from .parameter_binding import bind_parameters
from .propagation import propagate
from .state_overlap import overlap_with_zero
from .truncation import TruncationStats, truncate_combined


def expectation_task(
    circuit,
    observable,
    normalized_params: Dict[str, float],
    truncate_threshold: float | None,
    max_weight: int | None,
    symmetry_strategy,
) -> Tuple[complex, TruncationStats | None]:
    """Compute one expectation value: propagate observable, overlap with |0>.

    Free function equivalent of the executor's single-pair computation so it
    can run in worker processes; the executor's serial path delegates here
    too, keeping one shared code path.

    Args:
        circuit: PauliPropagationCircuit
        observable: PauliPropagationOperator
        normalized_params: Parameters in indexed format (e.g. "theta[0]")
        truncate_threshold: Coefficient threshold (None = no truncation)
        max_weight: Maximum Pauli weight (None = no weight limit)
        symmetry_strategy: Executor-level fallback symmetry strategy

    Returns:
        Tuple of (complex expectation value, final TruncationStats or None)
    """
    gates = circuit.gates

    # Bind parameters if needed (bind_parameters expects gates list, not circuit)
    bound_params = bind_parameters(gates, normalized_params)

    # Assign parameters to observable if it has parametric coefficients
    effective_observable = observable
    if observable.is_parametrized:
        effective_observable = observable.assign_parameters(normalized_params)

    # pauli_sum returns a copy, so mutating its symmetry is side-effect free
    propagated_observable = effective_observable.pauli_sum

    # Use observable-level symmetry when explicitly configured.
    # Fall back to executor-level symmetry otherwise.
    if not propagated_observable.has_active_symmetry:
        propagated_observable.symmetry = symmetry_strategy

    # Propagate observable through circuit (Heisenberg picture)
    propagated = propagate(
        gates,
        propagated_observable,
        bound_params,
        max_weight=max_weight,
        truncate_threshold=truncate_threshold,
    )

    # Final truncation pass (cheap cleanup; keeps the stats)
    stats = None
    if truncate_threshold is not None or max_weight is not None:
        propagated, stats = truncate_combined(
            propagated,
            min_coeff=truncate_threshold if truncate_threshold else 1e-15,
            max_weight=max_weight,
            inplace=True,
        )

    return overlap_with_zero(propagated), stats


def expectation_task_star(args) -> Tuple[complex, TruncationStats | None]:
    """Tuple-argument adapter for expectation_task (for executor.map)."""
    return expectation_task(*args)
