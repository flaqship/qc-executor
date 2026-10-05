"""The gate vocabulary of the framework-independent circuit IR.

Every instruction in a :class:`~qc_executor.base.circuit_ir.CircuitIR` is one
:class:`OpCode` plus its qubits, parameters and (rarely) classical bits.  A
single :data:`GATE_DEFS` table describes each opcode's arity and algebraic
properties, so backends and the decomposition pass can be driven by table
lookup instead of long ``if`` chains over gate names.

Opcodes are grouped into numeric bands with room to grow, so a new gate can be
added without renumbering — the IR stores opcodes as raw integers and a shifted
value would silently reinterpret existing data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict

__all__ = ["OpCode", "GateDef", "GATE_DEFS", "OPCODE_BY_NAME", "VARIABLE_QUBITS", "gate_def"]

#: ``GateDef.num_qubits`` value for instructions spanning an arbitrary width.
VARIABLE_QUBITS = -1


class OpCode(IntEnum):
    """Instruction kinds storable in the circuit IR.

    Values are explicit and banded by category.  They are persisted inside the
    IR's packed arrays, so existing values must never be renumbered.
    """

    # -- Structural and non-unitary (0-15) --
    BARRIER = 0
    MEASURE = 1
    RESET = 2
    # Reserved for nested classical control flow; not emitted in this version.
    BLOCK_BEGIN = 4
    BLOCK_END = 5

    # -- Single qubit, no parameters (16-31) --
    I = 16
    H = 17
    X = 18
    Y = 19
    Z = 20
    S = 21
    SDG = 22
    T = 23
    TDG = 24
    SX = 25
    SXDG = 26

    # -- Single qubit, parameterised (32-47) --
    RX = 32
    RY = 33
    RZ = 34
    P = 35
    U = 36

    # -- Two qubit, no parameters (48-63) --
    CX = 48
    CY = 49
    CZ = 50
    CH = 51
    ECR = 52
    SWAP = 53
    ISWAP = 54
    CS = 55
    CSX = 56

    # -- Two qubit, parameterised (64-79) --
    CP = 64
    CRX = 65
    CRY = 66
    CRZ = 67
    RXX = 68
    RYY = 69
    RZZ = 70
    RZX = 71

    # -- Three qubit (80-95) --
    CCX = 80
    CSWAP = 81


@dataclass(frozen=True, slots=True)
class GateDef:
    """Static description of one :class:`OpCode`.

    Args:
        name: Lowercase canonical name, also used for round-tripping via
            :data:`OPCODE_BY_NAME`.
        num_qubits: Qubits the instruction acts on, or :data:`VARIABLE_QUBITS`.
        num_params: Number of angle parameters.
        num_clbits: Number of classical bits written.
        is_clifford: Whether the gate is in the Clifford group.  Parameterised
            rotations are not, since that depends on the angle.
        self_inverse: Whether applying the gate twice is the identity.
        inverse: The opcode implementing the adjoint, when a distinct opcode
            exists.  ``None`` means the inverse is either the gate itself
            (``self_inverse``) or obtained by negating the angle.
        pauli_rotation: For rotation gates, the Pauli generator per qubit,
            aligned with the instruction's qubit order.  ``rz`` is ``("Z",)``
            and ``rzx`` is ``("Z", "X")``.
    """

    name: str
    num_qubits: int
    num_params: int = 0
    num_clbits: int = 0
    is_clifford: bool = False
    self_inverse: bool = False
    inverse: "OpCode | None" = None
    pauli_rotation: "tuple[str, ...] | None" = None

    @property
    def is_parameterized(self) -> bool:
        """Whether the gate carries at least one angle."""
        return self.num_params > 0

    @property
    def has_variable_width(self) -> bool:
        """Whether the gate spans an arbitrary number of qubits."""
        return self.num_qubits == VARIABLE_QUBITS


#: The single source of truth for opcode arity and algebraic properties.
GATE_DEFS: Dict[OpCode, GateDef] = {
    # Structural and non-unitary.
    OpCode.BARRIER: GateDef("barrier", VARIABLE_QUBITS),
    OpCode.MEASURE: GateDef("measure", 1, num_clbits=1),
    OpCode.RESET: GateDef("reset", 1),
    OpCode.BLOCK_BEGIN: GateDef("block_begin", VARIABLE_QUBITS),
    OpCode.BLOCK_END: GateDef("block_end", VARIABLE_QUBITS),
    # Single qubit, no parameters.
    OpCode.I: GateDef("id", 1, is_clifford=True, self_inverse=True),
    OpCode.H: GateDef("h", 1, is_clifford=True, self_inverse=True),
    OpCode.X: GateDef("x", 1, is_clifford=True, self_inverse=True),
    OpCode.Y: GateDef("y", 1, is_clifford=True, self_inverse=True),
    OpCode.Z: GateDef("z", 1, is_clifford=True, self_inverse=True),
    OpCode.S: GateDef("s", 1, is_clifford=True, inverse=OpCode.SDG),
    OpCode.SDG: GateDef("sdg", 1, is_clifford=True, inverse=OpCode.S),
    OpCode.T: GateDef("t", 1, inverse=OpCode.TDG),
    OpCode.TDG: GateDef("tdg", 1, inverse=OpCode.T),
    OpCode.SX: GateDef("sx", 1, is_clifford=True, inverse=OpCode.SXDG),
    OpCode.SXDG: GateDef("sxdg", 1, is_clifford=True, inverse=OpCode.SX),
    # Single qubit, parameterised.
    OpCode.RX: GateDef("rx", 1, num_params=1, pauli_rotation=("X",)),
    OpCode.RY: GateDef("ry", 1, num_params=1, pauli_rotation=("Y",)),
    OpCode.RZ: GateDef("rz", 1, num_params=1, pauli_rotation=("Z",)),
    OpCode.P: GateDef("p", 1, num_params=1),
    OpCode.U: GateDef("u", 1, num_params=3),
    # Two qubit, no parameters.
    OpCode.CX: GateDef("cx", 2, is_clifford=True, self_inverse=True),
    OpCode.CY: GateDef("cy", 2, is_clifford=True, self_inverse=True),
    OpCode.CZ: GateDef("cz", 2, is_clifford=True, self_inverse=True),
    OpCode.CH: GateDef("ch", 2, self_inverse=True),
    OpCode.ECR: GateDef("ecr", 2, is_clifford=True, self_inverse=True),
    OpCode.SWAP: GateDef("swap", 2, is_clifford=True, self_inverse=True),
    OpCode.ISWAP: GateDef("iswap", 2, is_clifford=True),
    OpCode.CS: GateDef("cs", 2),
    OpCode.CSX: GateDef("csx", 2),
    # Two qubit, parameterised.
    OpCode.CP: GateDef("cp", 2, num_params=1),
    OpCode.CRX: GateDef("crx", 2, num_params=1),
    OpCode.CRY: GateDef("cry", 2, num_params=1),
    OpCode.CRZ: GateDef("crz", 2, num_params=1),
    OpCode.RXX: GateDef("rxx", 2, num_params=1, pauli_rotation=("X", "X")),
    OpCode.RYY: GateDef("ryy", 2, num_params=1, pauli_rotation=("Y", "Y")),
    OpCode.RZZ: GateDef("rzz", 2, num_params=1, pauli_rotation=("Z", "Z")),
    OpCode.RZX: GateDef("rzx", 2, num_params=1, pauli_rotation=("Z", "X")),
    # Three qubit.
    OpCode.CCX: GateDef("ccx", 3, self_inverse=True),
    OpCode.CSWAP: GateDef("cswap", 3, self_inverse=True),
}

#: Reverse lookup from canonical gate name to opcode.
OPCODE_BY_NAME: Dict[str, OpCode] = {
    definition.name: opcode for opcode, definition in GATE_DEFS.items()
}


def gate_def(opcode: OpCode) -> GateDef:
    """Return the definition of ``opcode``.

    Args:
        opcode: The opcode to describe.

    Returns:
        The matching :class:`GateDef`.

    Raises:
        KeyError: If the opcode has no table entry.
    """
    return GATE_DEFS[opcode]
