"""Lowering of circuit instructions into a backend's supported gate set.

Backends declare what they can execute via
:meth:`~qc_executor.base.circuit_base.QuantumCircuitBase.supported_opcodes`, and
this pass rewrites anything else into those gates.  It replaces the
``qiskit.transpile(basis_gates=...)`` call that every non-Qiskit backend used to
rely on, which is what allows Qiskit to become an optional dependency.

The rewriter runs to a fixed point, so a rule may emit gates that themselves
need lowering.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, FrozenSet, List

from .circuit_ir import CircuitIR, Instruction
from .gate_set import GATE_DEFS, OpCode

__all__ = ["UnsupportedGateError", "DECOMPOSITIONS", "decompose_ir"]


class UnsupportedGateError(NotImplementedError):
    """Raised when an instruction cannot be expressed in the target gate set."""


Rule = Callable[[Instruction], List[Instruction]]


def _op(opcode: OpCode, *qubits: int, params: tuple = ()) -> Instruction:
    """Build one replacement instruction."""
    return Instruction(opcode, tuple(qubits), params)


def _cy(i: Instruction) -> List[Instruction]:
    control, target = i.qubits
    return [
        _op(OpCode.SDG, target),
        _op(OpCode.CX, control, target),
        _op(OpCode.S, target),
    ]


def _cz(i: Instruction) -> List[Instruction]:
    control, target = i.qubits
    return [_op(OpCode.H, target), _op(OpCode.CX, control, target), _op(OpCode.H, target)]


def _ch(i: Instruction) -> List[Instruction]:
    control, target = i.qubits
    return [
        _op(OpCode.RY, target, params=(math.pi / 4,)),
        _op(OpCode.CX, control, target),
        _op(OpCode.RY, target, params=(-math.pi / 4,)),
    ]


def _swap(i: Instruction) -> List[Instruction]:
    a, b = i.qubits
    return [_op(OpCode.CX, a, b), _op(OpCode.CX, b, a), _op(OpCode.CX, a, b)]


def _iswap(i: Instruction) -> List[Instruction]:
    a, b = i.qubits
    return [
        _op(OpCode.S, a),
        _op(OpCode.S, b),
        _op(OpCode.H, a),
        _op(OpCode.CX, a, b),
        _op(OpCode.CX, b, a),
        _op(OpCode.H, b),
    ]


def _cp(i: Instruction) -> List[Instruction]:
    control, target = i.qubits
    (angle,) = i.params
    return [
        _op(OpCode.P, control, params=(angle / 2,)),
        _op(OpCode.CX, control, target),
        _op(OpCode.P, target, params=(-angle / 2,)),
        _op(OpCode.CX, control, target),
        _op(OpCode.P, target, params=(angle / 2,)),
    ]


def _cs(i: Instruction) -> List[Instruction]:
    return _cp(Instruction(OpCode.CP, i.qubits, (math.pi / 2,)))


def _csx(i: Instruction) -> List[Instruction]:
    control, target = i.qubits
    return [
        _op(OpCode.H, target),
        _op(OpCode.CP, control, target, params=(math.pi / 2,)),
        _op(OpCode.H, target),
    ]


def _crz(i: Instruction) -> List[Instruction]:
    control, target = i.qubits
    (angle,) = i.params
    return [
        _op(OpCode.RZ, target, params=(angle / 2,)),
        _op(OpCode.CX, control, target),
        _op(OpCode.RZ, target, params=(-angle / 2,)),
        _op(OpCode.CX, control, target),
    ]


def _crx(i: Instruction) -> List[Instruction]:
    """CRX conjugated into the Z basis; the emitted CRZ is lowered in turn."""
    control, target = i.qubits
    return [
        _op(OpCode.H, target),
        _op(OpCode.CRZ, control, target, params=i.params),
        _op(OpCode.H, target),
    ]


def _cry(i: Instruction) -> List[Instruction]:
    control, target = i.qubits
    (angle,) = i.params
    return [
        _op(OpCode.RY, target, params=(angle / 2,)),
        _op(OpCode.CX, control, target),
        _op(OpCode.RY, target, params=(-angle / 2,)),
        _op(OpCode.CX, control, target),
    ]


def _rzz(i: Instruction) -> List[Instruction]:
    a, b = i.qubits
    return [_op(OpCode.CX, a, b), _op(OpCode.RZ, b, params=i.params), _op(OpCode.CX, a, b)]


def _rxx(i: Instruction) -> List[Instruction]:
    a, b = i.qubits
    return [
        _op(OpCode.H, a),
        _op(OpCode.H, b),
        _op(OpCode.RZZ, a, b, params=i.params),
        _op(OpCode.H, a),
        _op(OpCode.H, b),
    ]


def _ryy(i: Instruction) -> List[Instruction]:
    a, b = i.qubits
    return [
        _op(OpCode.SDG, a),
        _op(OpCode.H, a),
        _op(OpCode.SDG, b),
        _op(OpCode.H, b),
        _op(OpCode.RZZ, a, b, params=i.params),
        _op(OpCode.H, a),
        _op(OpCode.S, a),
        _op(OpCode.H, b),
        _op(OpCode.S, b),
    ]


def _rzx(i: Instruction) -> List[Instruction]:
    a, b = i.qubits
    return [_op(OpCode.H, b), _op(OpCode.RZZ, a, b, params=i.params), _op(OpCode.H, b)]


def _ecr(i: Instruction) -> List[Instruction]:
    a, b = i.qubits
    return [
        _op(OpCode.RZX, a, b, params=(math.pi / 4,)),
        _op(OpCode.X, a),
        _op(OpCode.RZX, a, b, params=(-math.pi / 4,)),
    ]


def _ccx(i: Instruction) -> List[Instruction]:
    first, second, target = i.qubits
    return [
        _op(OpCode.H, target),
        _op(OpCode.CX, second, target),
        _op(OpCode.TDG, target),
        _op(OpCode.CX, first, target),
        _op(OpCode.T, target),
        _op(OpCode.CX, second, target),
        _op(OpCode.TDG, target),
        _op(OpCode.CX, first, target),
        _op(OpCode.T, second),
        _op(OpCode.T, target),
        _op(OpCode.H, target),
        _op(OpCode.CX, first, second),
        _op(OpCode.T, first),
        _op(OpCode.TDG, second),
        _op(OpCode.CX, first, second),
    ]


def _cswap(i: Instruction) -> List[Instruction]:
    control, a, b = i.qubits
    return [
        _op(OpCode.CX, b, a),
        _op(OpCode.CCX, control, a, b),
        _op(OpCode.CX, b, a),
    ]


def _u(i: Instruction) -> List[Instruction]:
    (qubit,) = i.qubits
    theta, phi, lam = i.params
    return [
        _op(OpCode.RZ, qubit, params=(lam,)),
        _op(OpCode.RY, qubit, params=(theta,)),
        _op(OpCode.RZ, qubit, params=(phi,)),
    ]


def _fixed_rotation(opcode: OpCode, angle: float) -> Rule:
    """Build a rule replacing a gate with one fixed single-qubit rotation."""
    return lambda i: [_op(opcode, i.qubits[0], params=(angle,))]


#: Rewrite rules keyed by the opcode they eliminate.
#:
#: The rewriter runs to a fixed point, so a rule may emit gates that themselves
#: need lowering -- CRX emits CRZ, and RXX emits RZZ.  Everything bottoms out in
#: ``{RX, RY, RZ, P, CX}``, and no rule expands a gate in terms of something
#: that expands back into it, so the fixed point is always reached.
#:
#: Global phase is dropped throughout, matching the existing behaviour of the
#: simulator backends.
DECOMPOSITIONS: Dict[OpCode, Rule] = {
    # Structural.
    OpCode.I: lambda i: [],
    # Single qubit into rotations.
    OpCode.H: lambda i: [
        _op(OpCode.RY, i.qubits[0], params=(math.pi / 2,)),
        _op(OpCode.X, i.qubits[0]),
    ],
    OpCode.X: _fixed_rotation(OpCode.RX, math.pi),
    OpCode.Y: _fixed_rotation(OpCode.RY, math.pi),
    OpCode.Z: _fixed_rotation(OpCode.RZ, math.pi),
    OpCode.S: _fixed_rotation(OpCode.P, math.pi / 2),
    OpCode.SDG: _fixed_rotation(OpCode.P, -math.pi / 2),
    OpCode.T: _fixed_rotation(OpCode.P, math.pi / 4),
    OpCode.TDG: _fixed_rotation(OpCode.P, -math.pi / 4),
    OpCode.SX: _fixed_rotation(OpCode.RX, math.pi / 2),
    OpCode.SXDG: _fixed_rotation(OpCode.RX, -math.pi / 2),
    OpCode.P: _fixed_rotation(OpCode.RZ, 0.0),  # replaced below; needs the angle
    OpCode.U: _u,
    # Two qubit.
    OpCode.CY: _cy,
    OpCode.CZ: _cz,
    OpCode.CH: _ch,
    OpCode.SWAP: _swap,
    OpCode.ISWAP: _iswap,
    OpCode.CS: _cs,
    OpCode.CSX: _csx,
    OpCode.CP: _cp,
    OpCode.CRX: _crx,
    OpCode.CRY: _cry,
    OpCode.CRZ: _crz,
    OpCode.RXX: _rxx,
    OpCode.RYY: _ryy,
    OpCode.RZZ: _rzz,
    OpCode.RZX: _rzx,
    OpCode.ECR: _ecr,
    # Three qubit.
    OpCode.CCX: _ccx,
    OpCode.CSWAP: _cswap,
}

# P carries its own angle, so it cannot use the fixed-rotation helper.
DECOMPOSITIONS[OpCode.P] = lambda i: [_op(OpCode.RZ, i.qubits[0], params=i.params)]

#: Instructions that every backend must tolerate structurally.
_ALWAYS_ALLOWED: FrozenSet[OpCode] = frozenset({OpCode.BARRIER})


def decompose_ir(ir: CircuitIR, supported: FrozenSet[OpCode], *, max_passes: int = 8) -> CircuitIR:
    """Rewrite ``ir`` so it uses only ``supported`` opcodes.

    Args:
        ir: The circuit to lower.
        supported: Opcodes the target backend can execute.
        max_passes: Safety bound on rewrite rounds, guarding against a rule
            cycle.

    Returns:
        ``ir`` itself when nothing needs lowering, otherwise a new circuit.

    Raises:
        UnsupportedGateError: If an instruction is unsupported and no rule
            applies, or if lowering fails to converge.
    """
    allowed = frozenset(supported) | _ALWAYS_ALLOWED
    if all(OpCode(opcode) in allowed for opcode, _, _ in ir.iter_ops()):
        return ir

    current = ir
    for _ in range(max_passes):
        rewritten = CircuitIR(current.num_qubits, current.num_clbits)
        changed = False

        for instruction in current:
            if instruction.opcode in allowed:
                _append(rewritten, instruction)
                continue

            rule = DECOMPOSITIONS.get(instruction.opcode)
            if rule is None:
                raise UnsupportedGateError(
                    f"'{GATE_DEFS[instruction.opcode].name}' is not supported by this "
                    "backend and no decomposition rule is registered for it"
                )
            for replacement in rule(instruction):
                _append(rewritten, replacement)
            changed = True

        current = rewritten
        if not changed:
            return current

    raise UnsupportedGateError(
        f"gate lowering did not converge within {max_passes} passes; "
        "a decomposition rule is likely cyclic"
    )


def _append(target: CircuitIR, instruction: Instruction) -> None:
    """Copy one instruction into ``target``, preserving clbits and condition."""
    target.append(
        instruction.opcode,
        instruction.qubits,
        instruction.params,
        instruction.clbits,
        instruction.condition,
    )
