"""Pauli Propagation Package for Quantum Computing.

A Python implementation of Pauli propagation for efficient quantum circuit simulation
in the Heisenberg picture. Based on PauliPropagation.jl.
"""

from qc_executor.factory import Executor

from .pauli_propagation_circuit import PauliPropagationCircuit
from .pauli_propagation_executor import PauliPropagationExecutor
from .pauli_propagation_operator import PauliPropagationOperator
from .symmetry import (
    CompositeSymmetry,
    NoSymmetry,
    PermutationSymmetry,
    SymmetryStrategy,
)

Executor.register("pauli_propagation")(PauliPropagationExecutor)

__version__ = "0.1.0"
__all__ = [
    "PauliPropagationExecutor",
    "PauliPropagationCircuit",
    "PauliPropagationOperator",
    "SymmetryStrategy",
    "NoSymmetry",
    "PermutationSymmetry",
    "CompositeSymmetry",
]
