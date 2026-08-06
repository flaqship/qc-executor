"""Tests for lowering circuits into a backend's supported gate set."""

from __future__ import annotations

import pytest

from qc_executor.base.circuit_ir import CircuitIR, Condition, Instruction
from qc_executor.base.decompose import DECOMPOSITIONS, UnsupportedGateError, decompose_ir
from qc_executor.base.gate_set import OpCode

FULL_SET = frozenset(OpCode)


class TestPassThrough:
    def test_supported_circuit_is_returned_unchanged(self):
        ir = CircuitIR(2)
        ir.append(OpCode.H, (0,))
        ir.append(OpCode.CX, (0, 1))

        assert decompose_ir(ir, FULL_SET) is ir

    def test_barriers_are_always_allowed(self):
        ir = CircuitIR(2)
        ir.append(OpCode.BARRIER, (0, 1))

        assert decompose_ir(ir, frozenset({OpCode.H})) is ir

    def test_empty_circuit(self):
        ir = CircuitIR(1)

        assert decompose_ir(ir, frozenset()) is ir


class TestUnsupported:
    def test_missing_rule_reports_the_gate(self):
        ir = CircuitIR(2)
        ir.append(OpCode.ECR, (0, 1))

        with pytest.raises(UnsupportedGateError, match="'ecr' is not supported"):
            decompose_ir(ir, frozenset({OpCode.CX}))

    def test_error_is_a_not_implemented_error(self):
        """Subclassing NotImplementedError keeps these bodies out of coverage."""
        assert issubclass(UnsupportedGateError, NotImplementedError)


class TestRewriting:
    def test_a_registered_rule_replaces_the_instruction(self, monkeypatch):
        def cy_rule(instruction: Instruction):
            control, target = instruction.qubits
            return [
                Instruction(OpCode.SDG, (target,)),
                Instruction(OpCode.CX, (control, target)),
                Instruction(OpCode.S, (target,)),
            ]

        monkeypatch.setitem(DECOMPOSITIONS, OpCode.CY, cy_rule)

        ir = CircuitIR(2)
        ir.append(OpCode.CY, (0, 1))

        lowered = decompose_ir(ir, frozenset({OpCode.SDG, OpCode.CX, OpCode.S}))

        assert [i.name for i in lowered] == ["sdg", "cx", "s"]

    def test_rules_are_applied_until_a_fixed_point(self, monkeypatch):
        """A rule may emit gates that themselves need lowering."""
        monkeypatch.setitem(
            DECOMPOSITIONS, OpCode.CY, lambda i: [Instruction(OpCode.ECR, i.qubits)]
        )
        monkeypatch.setitem(
            DECOMPOSITIONS, OpCode.ECR, lambda i: [Instruction(OpCode.CX, i.qubits)]
        )

        ir = CircuitIR(2)
        ir.append(OpCode.CY, (0, 1))

        lowered = decompose_ir(ir, frozenset({OpCode.CX}))

        assert [i.name for i in lowered] == ["cx"]

    def test_supported_instructions_are_copied_verbatim(self, monkeypatch):
        monkeypatch.setitem(
            DECOMPOSITIONS, OpCode.CY, lambda i: [Instruction(OpCode.CX, i.qubits)]
        )

        ir = CircuitIR(2, num_clbits=1)
        ir.append(OpCode.MEASURE, (0,), (), (0,))
        ir.append(OpCode.X, (1,), condition=Condition((0,), 1))
        ir.append(OpCode.CY, (0, 1))

        lowered = decompose_ir(ir, frozenset({OpCode.MEASURE, OpCode.X, OpCode.CX}))

        assert [i.name for i in lowered] == ["measure", "x", "cx"]
        assert lowered[0].clbits == (0,)
        assert lowered[1].condition == Condition((0,), 1)

    def test_parameters_survive_lowering(self, monkeypatch):
        monkeypatch.setitem(
            DECOMPOSITIONS,
            OpCode.CRZ,
            lambda i: [Instruction(OpCode.RZ, (i.qubits[1],), i.params)],
        )

        ir = CircuitIR(2)
        ir.append(OpCode.CRZ, (0, 1), (0.75,))

        lowered = decompose_ir(ir, frozenset({OpCode.RZ}))

        assert lowered[0].params == (0.75,)

    def test_a_cyclic_rule_is_caught(self, monkeypatch):
        monkeypatch.setitem(
            DECOMPOSITIONS, OpCode.CY, lambda i: [Instruction(OpCode.ECR, i.qubits)]
        )
        monkeypatch.setitem(
            DECOMPOSITIONS, OpCode.ECR, lambda i: [Instruction(OpCode.CY, i.qubits)]
        )

        ir = CircuitIR(2)
        ir.append(OpCode.CY, (0, 1))

        with pytest.raises(UnsupportedGateError, match="did not converge"):
            decompose_ir(ir, frozenset({OpCode.CX}), max_passes=3)
