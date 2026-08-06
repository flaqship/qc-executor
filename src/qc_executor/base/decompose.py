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

from typing import Callable, Dict, FrozenSet, List

from .circuit_ir import CircuitIR, Instruction
from .gate_set import GATE_DEFS, OpCode

__all__ = ["UnsupportedGateError", "DECOMPOSITIONS", "decompose_ir"]


class UnsupportedGateError(NotImplementedError):
    """Raised when an instruction cannot be expressed in the target gate set."""


#: Rewrite rules keyed by the opcode they eliminate.  Populated by the gate
#: lowering work package; an empty table means only exact matches are accepted.
Rule = Callable[[Instruction], List[Instruction]]
DECOMPOSITIONS: Dict[OpCode, Rule] = {}

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
