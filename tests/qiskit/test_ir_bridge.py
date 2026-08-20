"""Tests for compiling the framework-independent IR into a Qiskit circuit."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.quantum_info import Operator

from qc_executor import QuantumCircuit
from qc_executor.base.circuit_ir import CircuitIR, Condition
from qc_executor.base.gate_set import GATE_DEFS, OpCode
from qc_executor.parameters import Parameters
from qc_executor.qiskit._ir_bridge import SUPPORTED_OPCODES, ir_to_qiskit

#: Opcodes deliberately outside the emitter table.
_STRUCTURAL = {OpCode.BLOCK_BEGIN, OpCode.BLOCK_END}


class TestCoverageOfTheGateSet:
    def test_every_gate_except_reserved_ones_can_be_emitted(self):
        missing = set(GATE_DEFS) - SUPPORTED_OPCODES - _STRUCTURAL
        assert not missing, f"no Qiskit emitter for: {sorted(g.name for g in missing)}"


class TestGateEmission:
    @pytest.mark.parametrize(
        "opcode",
        [op for op in GATE_DEFS if op in SUPPORTED_OPCODES and op not in {OpCode.MEASURE}],
        ids=lambda op: GATE_DEFS[op].name,
    )
    def test_each_gate_emits_under_its_qiskit_name(self, opcode):
        definition = GATE_DEFS[opcode]
        num_qubits = 3 if definition.has_variable_width else definition.num_qubits
        ir = CircuitIR(max(num_qubits, 1))
        ir.append(
            opcode, tuple(range(num_qubits)), tuple(0.3 for _ in range(definition.num_params))
        )

        native = ir_to_qiskit(ir)

        assert [i.operation.name for i in native.data] == [definition.name]

    def test_gate_semantics_match_qiskit(self):
        """A representative circuit must produce the same unitary as Qiskit's own."""
        ir = CircuitIR(2)
        ir.append(OpCode.H, (0,))
        ir.append(OpCode.CX, (0, 1))
        ir.append(OpCode.RZ, (1,), (0.7,))

        reference = QiskitQuantumCircuit(2)
        reference.h(0)
        reference.cx(0, 1)
        reference.rz(0.7, 1)

        assert np.allclose(Operator(ir_to_qiskit(ir)).data, Operator(reference).data)

    def test_unknown_opcode_is_reported(self):
        ir = CircuitIR(1)
        ir.append(OpCode.BLOCK_BEGIN, ())

        with pytest.raises(NotImplementedError, match="no Qiskit equivalent"):
            ir_to_qiskit(ir)


class TestRegisters:
    def test_no_classical_register_when_unused(self):
        ir = CircuitIR(2)
        ir.append(OpCode.H, (0,))

        assert ir_to_qiskit(ir).num_clbits == 0

    def test_classical_register_is_created_when_needed(self):
        ir = CircuitIR(1, num_clbits=2)
        ir.append(OpCode.MEASURE, (0,), (), (1,))

        native = ir_to_qiskit(ir)

        assert native.num_clbits == 2
        assert native.data[0].operation.name == "measure"


class TestConditions:
    def test_single_bit_condition_becomes_an_if_block(self):
        ir = CircuitIR(2, num_clbits=1)
        ir.append(OpCode.MEASURE, (0,), (), (0,))
        ir.append(OpCode.X, (1,), condition=Condition((0,), 1))

        native = ir_to_qiskit(ir)

        assert [i.operation.name for i in native.data] == ["measure", "if_else"]

    def test_multi_bit_condition_becomes_an_if_block(self):
        ir = CircuitIR(1, num_clbits=2)
        ir.append(OpCode.X, (0,), condition=Condition((0, 1), 3))

        native = ir_to_qiskit(ir)

        assert [i.operation.name for i in native.data] == ["if_else"]


class TestParameters:
    def test_symbolic_angles_become_parameter_expressions(self):
        x = Parameters("x", 2)
        ir = CircuitIR(1)
        ir.append(OpCode.RX, (0,), (2 * x[0] + x[1],))

        native = ir_to_qiskit(ir)

        assert sorted(p.name for p in native.parameters) == ["x[0]", "x[1]"]

    def test_parameter_identity_is_shared_across_compilations(self):
        """Qiskit compares parameters by UUID; binding depends on this holding."""
        x = Parameters("x", 1)
        circuit = QuantumCircuit(1)
        circuit.rx(0, x[0])

        first = set(ir_to_qiskit(circuit.ir).parameters)
        second = set(ir_to_qiskit(circuit.ir).parameters)

        assert first == second

    def test_bound_circuit_evaluates(self):
        x = Parameters("x", 1)
        ir = CircuitIR(1)
        ir.append(OpCode.RX, (0,), (2 * x[0],))

        native = ir_to_qiskit(ir)
        bound = native.assign_parameters({p: 0.25 for p in native.parameters})

        assert np.allclose(Operator(bound).data, Operator(_rx_reference(0.5)).data)


def _rx_reference(angle: float):
    """Build a one-qubit Qiskit circuit with a single RX rotation."""
    circuit = QiskitQuantumCircuit(1)
    circuit.rx(angle, 0)
    return circuit
