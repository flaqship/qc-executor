"""Framework-independent quantum circuit backed by a typed gate list.

Gates are stored as typed objects (:class:`CliffordGate`, :class:`RotationGate`,
:class:`Barrier`) rather than inside a Qiskit circuit. Angles may be plain
floats or symbolic :class:`~executor.abstraction.abstract_parameter.Parameter`
expressions (SymPy). Each backend builds its native circuit directly from this
gate list (see e.g. the PennyLane backend's ``PennyLaneCircuit``).
"""

from __future__ import annotations

from abc import ABC
from collections import Counter
from dataclasses import dataclass
from typing import List, Union

import sympy as sp

from ..base import QuantumCircuitBase
from .abstract_parameter import Parameter, free_parameters

#: An angle is either numeric or a symbolic parameter expression.
Angle = Union[float, int, sp.Basic]


# ---------------------------------------------------------------------------
# Typed gate instructions
# ---------------------------------------------------------------------------

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


# Inversion tables used by :meth:`AbstractQuantumCircuit.invert`.
_SELF_INVERSE = {"h", "x", "y", "z", "cx", "cy", "cz", "ecr", "swap"}
_CLIFFORD_INVERSE_MAP = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}


# ---------------------------------------------------------------------------
# Circuit
# ---------------------------------------------------------------------------

class AbstractQuantumCircuit(QuantumCircuitBase):
    """Quantum circuit defined independently of any quantum framework.

    Build a circuit with the usual gate methods; each backend converts this
    typed gate list into its native representation (see e.g. the PennyLane
    backend's ``PennyLaneCircuit``).

    Args:
        num_qubits: Number of qubits in the circuit.
    """

    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self._gates: List[AbstractGate] = []

    def _add(self, gate: AbstractGate) -> None:
        """Append a gate to the circuit."""
        self._gates.append(gate)

    @classmethod
    def from_quantum_circuit(cls, circuit: QuantumCircuitBase) -> "AbstractQuantumCircuit":
        """Return ``circuit`` unchanged if abstract, else copy its gate list."""
        if isinstance(circuit, cls):
            return circuit.copy()
        raise TypeError(
            f"Cannot build an AbstractQuantumCircuit from {type(circuit).__name__}."
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def parameters(self) -> List[Parameter]:
        """Return all native parameters used in the circuit, sorted."""
        seen: dict = {}
        for gate in self._gates:
            if isinstance(gate, RotationGate):
                for param in free_parameters(gate.angle):
                    seen[param] = None
        return sorted(seen, key=lambda p: (p.vector_name, p.index))

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    @property
    def is_parameterized(self) -> bool:
        return len(self.parameters) > 0

    def draw(self) -> str:
        """Return a simple text listing of the gate sequence."""
        lines = []
        for gate in self._gates:
            if isinstance(gate, Barrier):
                lines.append(f"{'BARRIER':<8} {list(gate.qubits)}")
            elif isinstance(gate, CliffordGate):
                lines.append(f"{gate.name.upper():<8} {list(gate.qubits)}")
            elif isinstance(gate, RotationGate):
                lines.append(f"{gate.name.upper():<8} {list(gate.qubits)}  angle={gate.angle}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Gate methods – single qubit, no angle
    # ------------------------------------------------------------------

    def _add_clifford_single(self, name: str, qubits: int | List[int]) -> None:
        targets = qubits if isinstance(qubits, (list, tuple)) else [qubits]
        for q in targets:
            self._add(CliffordGate(name, (q,)))

    def h(self, qubits: int | List[int]):    self._add_clifford_single("h", qubits)
    def s(self, qubits: int | List[int]):    self._add_clifford_single("s", qubits)
    def sdag(self, qubits: int | List[int]): self._add_clifford_single("sdg", qubits)
    def t(self, qubits: int | List[int]):    self._add_clifford_single("t", qubits)
    def tdag(self, qubits: int | List[int]): self._add_clifford_single("tdg", qubits)
    def x(self, qubits: int | List[int]):    self._add_clifford_single("x", qubits)
    def y(self, qubits: int | List[int]):    self._add_clifford_single("y", qubits)
    def z(self, qubits: int | List[int]):    self._add_clifford_single("z", qubits)

    # ------------------------------------------------------------------
    # Gate methods – single qubit, with angle
    # ------------------------------------------------------------------

    def _add_rotation_single(self, name: str, qubits: int | List[int], angle: Angle) -> None:
        targets = qubits if isinstance(qubits, (list, tuple)) else [qubits]
        for q in targets:
            self._add(RotationGate(name, (q,), angle))

    def rx(self, qubits: int | List[int], angle: Angle): self._add_rotation_single("rx", qubits, angle)
    def ry(self, qubits: int | List[int], angle: Angle): self._add_rotation_single("ry", qubits, angle)
    def rz(self, qubits: int | List[int], angle: Angle): self._add_rotation_single("rz", qubits, angle)
    def p(self, qubits: int | List[int], angle: Angle):  self._add_rotation_single("p", qubits, angle)

    # ------------------------------------------------------------------
    # Gate methods – two qubits, no angle
    # ------------------------------------------------------------------

    def cx(self, control_qubit: int, target_qubit: int):
        self._add(CliffordGate("cx", (control_qubit, target_qubit)))

    def cy(self, control_qubit: int, target_qubit: int):
        self._add(CliffordGate("cy", (control_qubit, target_qubit)))

    def cz(self, control_qubit: int, target_qubit: int):
        self._add(CliffordGate("cz", (control_qubit, target_qubit)))

    def ecr(self, control_qubit: int, target_qubit: int):
        self._add(CliffordGate("ecr", (control_qubit, target_qubit)))

    def swap(self, qubit1: int, qubit2: int):
        self._add(CliffordGate("swap", (qubit1, qubit2)))

    def cnot(self, control_qubit: int, target_qubit: int):
        self.cx(control_qubit, target_qubit)

    # ------------------------------------------------------------------
    # Gate methods – two qubits, with angle
    # ------------------------------------------------------------------

    def cp(self, control_qubit: int, target_qubit: int, angle: Angle):
        self._add(RotationGate("cp", (control_qubit, target_qubit), angle))

    def crx(self, control_qubit: int, target_qubit: int, angle: Angle):
        self._add(RotationGate("crx", (control_qubit, target_qubit), angle))

    def cry(self, control_qubit: int, target_qubit: int, angle: Angle):
        self._add(RotationGate("cry", (control_qubit, target_qubit), angle))

    def crz(self, control_qubit: int, target_qubit: int, angle: Angle):
        self._add(RotationGate("crz", (control_qubit, target_qubit), angle))

    def rxx(self, control_qubit: int, target_qubit: int, angle: Angle):
        self._add(RotationGate("rxx", (control_qubit, target_qubit), angle))

    def ryy(self, control_qubit: int, target_qubit: int, angle: Angle):
        self._add(RotationGate("ryy", (control_qubit, target_qubit), angle))

    def rzz(self, control_qubit: int, target_qubit: int, angle: Angle):
        self._add(RotationGate("rzz", (control_qubit, target_qubit), angle))

    def rzx(self, control_qubit: int, target_qubit: int, angle: Angle):
        self._add(RotationGate("rzx", (control_qubit, target_qubit), angle))

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def barrier(self, qubits: int | List[int]):
        targets = (qubits,) if isinstance(qubits, int) else tuple(qubits)
        self._add(Barrier(targets))

    def measure(self):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Circuit operations
    # ------------------------------------------------------------------

    def assign_parameters(self, parameters: dict):
        """Substitute numeric values for symbolic parameters in every angle."""
        for gate in self._gates:
            if isinstance(gate, RotationGate) and isinstance(gate.angle, sp.Basic):
                result = gate.angle.subs(parameters)
                gate.angle = float(result) if result.is_number else result

    def compose(self, qc: QuantumCircuitBase, qubits: List[int]) -> "AbstractQuantumCircuit":
        """Append another abstract circuit's gates, remapping qubit indices."""
        if not isinstance(qc, AbstractQuantumCircuit):
            raise ValueError("compose() expects an AbstractQuantumCircuit.")
        for gate in qc._gates:
            remapped = tuple(qubits[q] for q in gate.qubits)
            if isinstance(gate, CliffordGate):
                self._add(CliffordGate(gate.name, remapped))
            elif isinstance(gate, RotationGate):
                self._add(RotationGate(gate.name, remapped, gate.angle))
            elif isinstance(gate, Barrier):
                self._add(Barrier(remapped))
        return self

    def invert(self) -> "AbstractQuantumCircuit":
        """Return the inverse circuit (reversed order, each gate inverted)."""
        inverted = type(self)(self._num_qubits)
        for gate in reversed(self._gates):
            if isinstance(gate, Barrier):
                inverted._gates.append(Barrier(gate.qubits))
            elif isinstance(gate, CliffordGate):
                if gate.name in _SELF_INVERSE:
                    inverted._gates.append(CliffordGate(gate.name, gate.qubits))
                elif gate.name in _CLIFFORD_INVERSE_MAP:
                    inverted._gates.append(
                        CliffordGate(_CLIFFORD_INVERSE_MAP[gate.name], gate.qubits)
                    )
                else:
                    raise NotImplementedError(f"Cannot invert gate '{gate.name}'")
            elif isinstance(gate, RotationGate):
                inverted._gates.append(RotationGate(gate.name, gate.qubits, -gate.angle))
        return inverted

    def copy(self) -> "AbstractQuantumCircuit":
        """Return a copy of the circuit with an independent gate list."""
        clone = type(self)(self._num_qubits)
        for gate in self._gates:
            if isinstance(gate, CliffordGate):
                clone._gates.append(CliffordGate(gate.name, gate.qubits))
            elif isinstance(gate, RotationGate):
                clone._gates.append(RotationGate(gate.name, gate.qubits, gate.angle))
            elif isinstance(gate, Barrier):
                clone._gates.append(Barrier(gate.qubits))
        return clone

    def circuit_metrics(self) -> dict:
        """Return a count of gates by name."""
        names = [
            "barrier" if isinstance(g, Barrier) else g.name
            for g in self._gates
        ]
        return dict(Counter(names))

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __hash__(self):
        gate_tuple = tuple(
            (type(g).__name__, g.qubits,
             str(g.angle) if isinstance(g, RotationGate) else None)
            for g in self._gates
        )
        return hash((self._num_qubits, gate_tuple))

    def __str__(self):
        return self.draw()

    def __repr__(self):
        return f"AbstractQuantumCircuit(num_qubits={self._num_qubits}, gates={len(self._gates)})"
