"""Unitary verification of the gate-lowering rule table.

Each rule is checked against a reference unitary built from first principles —
Pauli matrices and ``exp(-i t P / 2)`` — rather than from any quantum framework,
so the lowering pass that replaces ``qiskit.transpile`` is not validated against
the thing it replaces.

Lowering drops global phase, matching the simulator backends, so comparisons are
up to a phase.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from qc_executor.base.circuit_ir import CircuitIR, Instruction
from qc_executor.base.decompose import DECOMPOSITIONS, decompose_ir
from qc_executor.base.gate_set import GATE_DEFS, OpCode
from qc_executor.parameters import Parameters

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.diag([1, -1]).astype(complex)

#: Angles used when a gate takes parameters; deliberately not special values.
_ANGLES = [0.6137, 0.31, 0.87]

#: The basis every rule must eventually reduce to.
PRIMITIVES = frozenset({OpCode.RX, OpCode.RY, OpCode.RZ, OpCode.CX})


def _rot(pauli: np.ndarray, angle: float) -> np.ndarray:
    """Return ``exp(-i * angle / 2 * pauli)``."""
    return math.cos(angle / 2) * _I - 1j * math.sin(angle / 2) * pauli


def _controlled(target: np.ndarray) -> np.ndarray:
    """Return a two-qubit controlled gate, qubit 0 controlling."""
    result = np.eye(4, dtype=complex)
    result[2:, 2:] = target
    return result


_SINGLE = {
    OpCode.I: lambda p: _I,
    OpCode.H: lambda p: np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2),
    OpCode.X: lambda p: _X,
    OpCode.Y: lambda p: _Y,
    OpCode.Z: lambda p: _Z,
    OpCode.S: lambda p: np.diag([1, 1j]).astype(complex),
    OpCode.SDG: lambda p: np.diag([1, -1j]).astype(complex),
    OpCode.T: lambda p: np.diag([1, np.exp(1j * math.pi / 4)]),
    OpCode.TDG: lambda p: np.diag([1, np.exp(-1j * math.pi / 4)]),
    OpCode.SX: lambda p: np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2,
    OpCode.SXDG: lambda p: np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]) / 2,
    OpCode.RX: lambda p: _rot(_X, p[0]),
    OpCode.RY: lambda p: _rot(_Y, p[0]),
    OpCode.RZ: lambda p: _rot(_Z, p[0]),
    OpCode.P: lambda p: np.diag([1, np.exp(1j * p[0])]),
    OpCode.U: lambda p: (
        np.diag([1, np.exp(1j * p[1])]) @ _rot(_Y, p[0]) @ np.diag([1, np.exp(1j * p[2])])
    ),
}


def _two_qubit(opcode: OpCode, params) -> np.ndarray:
    """Return the two-qubit reference unitary for ``opcode``."""
    if opcode is OpCode.CX:
        return _controlled(_X)
    if opcode is OpCode.CY:
        return _controlled(_Y)
    if opcode is OpCode.CZ:
        return _controlled(_Z)
    if opcode is OpCode.CH:
        return _controlled(_SINGLE[OpCode.H](()))
    if opcode is OpCode.CS:
        return _controlled(_SINGLE[OpCode.S](()))
    if opcode is OpCode.CSX:
        return _controlled(_SINGLE[OpCode.SX](()))
    if opcode is OpCode.CP:
        return _controlled(np.diag([1, np.exp(1j * params[0])]))
    if opcode is OpCode.CRX:
        return _controlled(_rot(_X, params[0]))
    if opcode is OpCode.CRY:
        return _controlled(_rot(_Y, params[0]))
    if opcode is OpCode.CRZ:
        return _controlled(_rot(_Z, params[0]))
    if opcode is OpCode.SWAP:
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    if opcode is OpCode.ISWAP:
        return np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex)
    if opcode is OpCode.ECR:
        # ECR = (IX - XY)/sqrt(2) in Qiskit's label convention, where the
        # leftmost letter is the highest qubit.  Here qubit 0 is the leading
        # tensor factor, so the letters swap places.
        return (np.kron(_X, _I) - np.kron(_Y, _X)) / math.sqrt(2)
    paulis = {"X": _X, "Y": _Y, "Z": _Z}
    first, second = GATE_DEFS[opcode].pauli_rotation
    generator = np.kron(paulis[first], paulis[second])
    return math.cos(params[0] / 2) * np.eye(4) - 1j * math.sin(params[0] / 2) * generator


def _embed(matrix: np.ndarray, qubits, num_qubits: int) -> np.ndarray:
    """Place a gate matrix into the full space, qubit 0 as the leading factor."""
    width = len(qubits)
    full = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for row in range(2**num_qubits):
        row_bits = [(row >> (num_qubits - 1 - q)) & 1 for q in range(num_qubits)]
        sub_row = sum(row_bits[q] << (width - 1 - k) for k, q in enumerate(qubits))
        for sub_col in range(2**width):
            amplitude = matrix[sub_row, sub_col]
            if amplitude == 0:
                continue
            col_bits = list(row_bits)
            for k, q in enumerate(qubits):
                col_bits[q] = (sub_col >> (width - 1 - k)) & 1
            col = sum(bit << (num_qubits - 1 - q) for q, bit in enumerate(col_bits))
            full[row, col] += amplitude
    return full


def unitary(instruction: Instruction, num_qubits: int) -> np.ndarray:
    """Build the full-space unitary of one instruction."""
    opcode = instruction.opcode
    if opcode in _SINGLE:
        local = _SINGLE[opcode](instruction.params)
    elif opcode is OpCode.CCX:
        local = np.eye(8, dtype=complex)
        local[[6, 7]] = local[[7, 6]]
    elif opcode is OpCode.CSWAP:
        local = np.eye(8, dtype=complex)
        local[[5, 6]] = local[[6, 5]]
    else:
        local = _two_qubit(opcode, instruction.params)
    return _embed(local, instruction.qubits, num_qubits)


def product(instructions, num_qubits: int) -> np.ndarray:
    """Compose a sequence of instructions into one unitary."""
    return functools.reduce(
        lambda acc, i: unitary(i, num_qubits) @ acc,
        instructions,
        np.eye(2**num_qubits, dtype=complex),
    )


def equal_up_to_phase(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare unitaries ignoring the global phase that lowering discards."""
    index = np.unravel_index(np.argmax(np.abs(left)), left.shape)
    if abs(right[index]) < 1e-9:
        return False
    return np.allclose(left * (right[index] / left[index]), right, atol=1e-8)


def _source(opcode: OpCode) -> Instruction:
    """Build a representative instruction for ``opcode``."""
    definition = GATE_DEFS[opcode]
    return Instruction(
        opcode,
        tuple(range(definition.num_qubits)),
        tuple(_ANGLES[: definition.num_params]),
    )


LOWERABLE = sorted(DECOMPOSITIONS, key=lambda op: GATE_DEFS[op].name)
NON_TRIVIAL = [op for op in LOWERABLE if op is not OpCode.I]


class TestReferenceMatchesTheGateTable:
    def test_pauli_rotation_generators_are_consistent(self):
        """The reference builds rotations from GATE_DEFS, so pin that link."""
        for opcode in (OpCode.RXX, OpCode.RYY, OpCode.RZZ, OpCode.RZX):
            assert GATE_DEFS[opcode].pauli_rotation is not None


class TestRulesPreserveTheUnitary:
    @pytest.mark.parametrize("opcode", LOWERABLE, ids=lambda op: GATE_DEFS[op].name)
    def test_each_rule_matches_its_gate(self, opcode):
        source = _source(opcode)
        num_qubits = GATE_DEFS[opcode].num_qubits

        replacement = product(DECOMPOSITIONS[opcode](source), num_qubits)

        assert equal_up_to_phase(replacement, unitary(source, num_qubits))

    def test_every_gate_has_a_rule_or_is_a_primitive(self):
        structural = {
            OpCode.BARRIER,
            OpCode.MEASURE,
            OpCode.RESET,
            OpCode.BLOCK_BEGIN,
            OpCode.BLOCK_END,
        }

        missing = set(GATE_DEFS) - set(DECOMPOSITIONS) - PRIMITIVES - structural

        assert not missing, f"no lowering rule for {sorted(GATE_DEFS[o].name for o in missing)}"


class TestLoweringToPrimitives:
    @pytest.mark.parametrize("opcode", NON_TRIVIAL, ids=lambda op: GATE_DEFS[op].name)
    def test_lowering_reaches_the_primitive_basis(self, opcode):
        source = _source(opcode)
        ir = CircuitIR(GATE_DEFS[opcode].num_qubits)
        ir.append(opcode, source.qubits, source.params)

        lowered = decompose_ir(ir, PRIMITIVES)

        assert {OpCode(op) for op, _, _ in lowered.iter_ops()} <= PRIMITIVES

    @pytest.mark.parametrize("opcode", NON_TRIVIAL, ids=lambda op: GATE_DEFS[op].name)
    def test_lowering_preserves_the_unitary(self, opcode):
        source = _source(opcode)
        num_qubits = GATE_DEFS[opcode].num_qubits
        ir = CircuitIR(num_qubits)
        ir.append(opcode, source.qubits, source.params)

        lowered = decompose_ir(ir, PRIMITIVES)

        assert equal_up_to_phase(product(list(lowered), num_qubits), product(list(ir), num_qubits))

    def test_a_whole_circuit_survives_lowering(self):
        ir = CircuitIR(3)
        ir.append(OpCode.H, (0,))
        ir.append(OpCode.CCX, (0, 1, 2))
        ir.append(OpCode.RYY, (0, 1), (0.4,))
        ir.append(OpCode.CSWAP, (0, 1, 2))
        ir.append(OpCode.ECR, (1, 2))

        lowered = decompose_ir(ir, PRIMITIVES)

        assert equal_up_to_phase(product(list(lowered), 3), product(list(ir), 3))

    def test_identity_gates_are_dropped(self):
        ir = CircuitIR(1)
        ir.append(OpCode.I, (0,))

        assert len(decompose_ir(ir, PRIMITIVES)) == 0

    def test_symbolic_angles_survive_lowering(self):
        x = Parameters("x", 1)
        ir = CircuitIR(2)
        ir.append(OpCode.RZZ, (0, 1), (2 * x[0],))

        lowered = decompose_ir(ir, PRIMITIVES)

        assert lowered.free_parameters == frozenset({x[0]})

    def test_a_clifford_plus_rotation_basis_also_works(self):
        """A realistic backend basis, not just the minimal one."""
        basis = frozenset(
            {OpCode.H, OpCode.S, OpCode.SDG, OpCode.CX, OpCode.RX, OpCode.RY, OpCode.RZ}
        )
        ir = CircuitIR(2)
        ir.append(OpCode.ISWAP, (0, 1))
        ir.append(OpCode.RYY, (0, 1), (0.7,))

        lowered = decompose_ir(ir, basis)

        assert {OpCode(op) for op, _, _ in lowered.iter_ops()} <= basis
        assert equal_up_to_phase(product(list(lowered), 2), product(list(ir), 2))
