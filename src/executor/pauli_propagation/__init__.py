"""Pauli Propagation Package for Quantum Computing.

A Python implementation of Pauli propagation for efficient quantum circuit simulation
in the Heisenberg picture. Based on PauliPropagation.jl.
"""

from .pauli_types import PauliString, PauliSum
from .executor import PauliPropagationExecutor
from .state_overlap import overlap_with_zero, overlap_with_computational, scalar_product
from .truncation import truncate_by_coeff, truncate_by_weight, truncate_combined, TruncationStats

__version__ = "0.1.0"
__all__ = [
    "PauliString",
    "PauliSum",
    "PauliPropagationExecutor",
    "overlap_with_zero",
    "overlap_with_computational",
    "scalar_product",
    "truncate_by_coeff",
    "truncate_by_weight",
    "truncate_combined",
    "TruncationStats",
]
