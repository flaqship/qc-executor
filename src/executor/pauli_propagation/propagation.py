"""Core Pauli propagation algorithm.

Propagates observables through quantum circuits in the Heisenberg picture.
"""

import numpy as np
from typing import List, Dict, Optional
from .gates import Gate, PauliRotation, CliffordGate
from .pauli_types import PauliSum
from .pauli_algebra import pauli_sum_product


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


def propagate(
    gates: List[Gate],
    observable: PauliSum,
    parameters: Optional[Dict[str, float]] = None,
) -> PauliSum:
    """Propagate observable through circuit (Heisenberg picture).

    In Heisenberg picture, we evolve the observable backward through the circuit:
    O → U_n† ... U_2† U_1† O U_1 U_2 ... U_n

    Since we store gates in forward order, we apply them in reverse.

    Args:
        gates: List of gates (in circuit order)
        observable: Initial observable (PauliSum)
        parameters: Dict mapping parameter names to values

    Returns:
        Evolved observable
    """
    if parameters is None:
        parameters = {}

    result = observable.copy()

    # Apply gates in reverse order (Heisenberg picture)
    for gate in reversed(gates):
        if gate.is_parametric():
            # Get parameter value
            param_value = None
            if gate.param_name and gate.param_name in parameters:
                param_value = parameters[gate.param_name]
            elif hasattr(gate, 'param_value') and gate.param_value is not None:
                # Use concrete value stored in gate
                param_value = gate.param_value
            else:
                # Try to infer parameter from gate type
                # This is a fallback; ideally param_name should be set
                param_value = parameters.get('theta', None)

            result = propagate_single_gate(gate, result, param_value)
        else:
            result = propagate_single_gate(gate, result)

    return result
