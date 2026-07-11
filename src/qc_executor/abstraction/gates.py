"""Typed gate instructions for the framework-independent circuit.

These dataclasses are the building blocks stored in an
:class:`~executor.abstraction.abstract_quantum_circuit.AbstractQuantumCircuit`.
Each backend converts this typed gate list into its native representation.

:data:`SUPPORTED_GATES` is the single source of truth for the gate vocabulary.
A gate name listed there must satisfy all three of:

* it is a method name on Qiskit's ``QuantumCircuit`` (the Qiskit backend
  resolves gates via ``getattr``);
* it is a key in ``pennylane.pennylane_gates.qiskit_pennylane_gate_dict`` (the
  PennyLane backend resolves gates through that dict);
* it is invertible, i.e. listed in ``_SELF_INVERSE`` or ``_CLIFFORD_INVERSE_MAP``
  in ``abstract_quantum_circuit`` (rotations invert by negating the angle).

``tests/abstraction/test_gate_backend_parity.py`` enforces all three, so adding a
gate here without wiring up a backend fails the suite instead of blowing up at
circuit-build time.
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
    """Non-parametric gate, e.g. ``h``, ``cx``, ``ccx``.

    See :data:`SUPPORTED_GATES` for the full vocabulary.

    Args:
        name: Lowercase gate name matching the Qiskit method (e.g. ``"cx"``).
        qubits: Qubit indices the gate acts on.
    """

    name: str
    qubits: tuple


@dataclass
class RotationGate(AbstractGate):
    """Single-angle gate, e.g. ``rx``, ``cp``, ``rzz``.

    See :data:`SUPPORTED_GATES` for the full vocabulary.

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


@dataclass(frozen=True)
class GateSpec:
    """Shape of a supported gate.

    Args:
        num_qubits: How many qubits the gate acts on.
        num_angles: How many rotation angles it takes (0 for a
            :class:`CliffordGate`, 1 for a :class:`RotationGate`).
    """

    num_qubits: int
    num_angles: int


#: The gate vocabulary of the abstraction – see the module docstring for the
#: invariants every entry must satisfy.
SUPPORTED_GATES: dict[str, GateSpec] = {
    # one qubit, no angle
    "id": GateSpec(1, 0),
    "h": GateSpec(1, 0),
    "s": GateSpec(1, 0),
    "sdg": GateSpec(1, 0),
    "sx": GateSpec(1, 0),
    "sxdg": GateSpec(1, 0),
    "t": GateSpec(1, 0),
    "tdg": GateSpec(1, 0),
    "x": GateSpec(1, 0),
    "y": GateSpec(1, 0),
    "z": GateSpec(1, 0),
    # one qubit, one angle
    "rx": GateSpec(1, 1),
    "ry": GateSpec(1, 1),
    "rz": GateSpec(1, 1),
    "p": GateSpec(1, 1),
    # two qubits, no angle
    "cx": GateSpec(2, 0),
    "cy": GateSpec(2, 0),
    "cz": GateSpec(2, 0),
    "ch": GateSpec(2, 0),
    "cs": GateSpec(2, 0),
    "csdg": GateSpec(2, 0),
    "ecr": GateSpec(2, 0),
    "swap": GateSpec(2, 0),
    # two qubits, one angle
    "cp": GateSpec(2, 1),
    "crx": GateSpec(2, 1),
    "cry": GateSpec(2, 1),
    "crz": GateSpec(2, 1),
    "rxx": GateSpec(2, 1),
    "ryy": GateSpec(2, 1),
    "rzz": GateSpec(2, 1),
    "rzx": GateSpec(2, 1),
    # three qubits, no angle
    "ccx": GateSpec(3, 0),
    "cswap": GateSpec(3, 0),
}


def validate_gate(gate: AbstractGate) -> None:
    """Check a gate against :data:`SUPPORTED_GATES`.

    Guards the one property no backend checks on its own: that the gate name is
    part of the shared vocabulary. The Qiskit backend would otherwise accept any
    name Qiskit happens to implement and the PennyLane backend would only fail
    much later, while building its circuit.

    Args:
        gate: The gate instruction to validate. Barriers are accepted as-is.

    Raises:
        ValueError: If the name is unknown, or the qubit or angle count does not
            match the gate's spec.
    """
    if isinstance(gate, Barrier):
        return

    spec = SUPPORTED_GATES.get(gate.name)
    if spec is None:
        raise ValueError(
            f"Unknown gate '{gate.name}'. Supported gates: "
            f"{', '.join(sorted(SUPPORTED_GATES))}."
        )

    if len(gate.qubits) != spec.num_qubits:
        raise ValueError(
            f"Gate '{gate.name}' acts on {spec.num_qubits} qubit(s), "
            f"got {len(gate.qubits)}: {list(gate.qubits)}."
        )

    if len(set(gate.qubits)) != len(gate.qubits):
        raise ValueError(
            f"Gate '{gate.name}' was given a repeated qubit: {list(gate.qubits)}."
        )

    num_angles = 1 if isinstance(gate, RotationGate) else 0
    if num_angles != spec.num_angles:
        expected = "a RotationGate" if spec.num_angles else "a CliffordGate"
        raise ValueError(
            f"Gate '{gate.name}' takes {spec.num_angles} angle(s) and must be "
            f"built as {expected}, got {type(gate).__name__}."
        )
