"""Pauli Propagation Package for Quantum Computing.

A Python implementation of Pauli propagation for efficient quantum circuit simulation
in the Heisenberg picture. Based on PauliPropagation.jl.
"""

from executor.factory import Executor

from .executor import PauliPropagationExecutor
from .pauli_propagation_circuit import PauliPropagationCircuit
from .pauli_types import PauliString, PauliSum
from .state_overlap import overlap_with_computational, overlap_with_zero, scalar_product
from .symmetry import (
    CompositeSymmetry,
    NoSymmetry,
    PermutationSymmetry,
    SymmetryStrategy,
)
from .truncation import (
    TruncationStats,
    truncate_by_coeff,
    truncate_by_weight,
    truncate_combined,
)

Executor.register("pauli_propagation")(PauliPropagationExecutor)

__version__ = "0.1.0"
__all__ = [
    "PauliString",
    "PauliSum",
    "PauliPropagationExecutor",
    "PauliPropagationCircuit",
    "overlap_with_zero",
    "overlap_with_computational",
    "scalar_product",
    "truncate_by_coeff",
    "truncate_by_weight",
    "truncate_combined",
    "TruncationStats",
    "SymmetryStrategy",
    "NoSymmetry",
    "PermutationSymmetry",
    "CompositeSymmetry",
]
