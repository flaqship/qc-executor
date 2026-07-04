"""Typed gate instructions for the framework-independent circuit.

These dataclasses are the building blocks stored in an
:class:`~executor.abstraction.abstract_quantum_circuit.AbstractQuantumCircuit`.
Each backend converts this typed gate list into its native representation.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Union

import sympy as sp

#: An angle is either numeric or a symbolic parameter expression.
Angle = Union[float, int, sp.Basic]


class AbstractGate(ABC):
    """Base class for all gate instructions stored in the circuit."""


@dataclass
class CliffordGate(AbstractGate):
    """Non-parametric gate (H, S, Sdg, T, Tdg, X, Y, Z, CX, CY, CZ, ECR, SWAP).

    Args:
        name: Lowercase gate name matching the Qiskit method (e.g. ``"cx"``).
        qubits: Qubit indices the gate acts on.
    """

    name: str
    qubits: tuple


@dataclass
class RotationGate(AbstractGate):
    """Single-angle gate (RX, RY, RZ, P, CP, CRX, CRY, CRZ, RXX, RYY, RZZ, RZX).

    Args:
        name: Lowercase gate name matching the Qiskit method (e.g. ``"rx"``).
        qubits: Qubit indices the gate acts on.
        angle: Rotation angle – numeric or symbolic parameter expression.
    """

    name: str
    qubits: tuple
    angle: Angle


@dataclass
class Barrier(AbstractGate):
    """Barrier marker (not a physical gate).

    Args:
        qubits: Qubit indices the barrier spans.
    """

    qubits: tuple