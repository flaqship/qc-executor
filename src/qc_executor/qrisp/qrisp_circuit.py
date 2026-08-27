"""Qrisp circuit wrapper compiled from Executor's framework-independent IR."""

from __future__ import annotations

from typing import Any, Mapping

import qrisp

from ..base.circuit_base import QuantumCircuitBase
from ..base.gate_set import GATE_DEFS, OpCode
from ..base.parameters_base import build_binding, evaluate


_SUPPORTED = frozenset(
    {
        OpCode.I,
        OpCode.H,
        OpCode.X,
        OpCode.Y,
        OpCode.Z,
        OpCode.S,
        OpCode.T,
        OpCode.RX,
        OpCode.RY,
        OpCode.RZ,
        OpCode.CX,
        OpCode.CZ,
        OpCode.SWAP,
        OpCode.BARRIER,
    }
)


class QrispCircuit(QuantumCircuitBase):
    """A circuit compiled into a Qrisp ``QuantumSession``."""

    @classmethod
    def supported_opcodes(cls) -> frozenset:
        return _SUPPORTED

    def _build_native(self):
        return self.build_qrisp_circuit()

    @staticmethod
    def _emit(session, instruction, parameters):
        if instruction.opcode is OpCode.BARRIER:
            return
        if instruction.condition is not None:
            raise NotImplementedError("Qrisp prototype does not yet support classical conditions")
        if instruction.opcode in (OpCode.MEASURE, OpCode.RESET):
            raise NotImplementedError("Qrisp prototype does not yet support dynamic circuits")

        name = GATE_DEFS[instruction.opcode].name
        gate = getattr(qrisp, name, None)
        if gate is None:
            raise NotImplementedError(f"Qrisp has no gate function for '{name}'")
        angles = tuple(
            value if parameters is None else evaluate(value, parameters)
            for value in instruction.params
        )
        gate(*angles, *[session.qubits[index] for index in instruction.qubits])

    def build_qrisp_circuit(self, values: Mapping[str, Any] | None = None):
        """Build and compile a Qrisp circuit, optionally binding parameters."""
        source = self._lowered_ir()
        parameters = None if values is None and source.free_parameters else build_binding(
            source.free_parameters, values or {}
        )
        session = qrisp.QuantumSession()
        session.request_qubits(self.num_qubits)
        for instruction in source:
            self._emit(session, instruction, parameters)
        return session.compile()

    @property
    def qrisp_circuit(self):
        """Return the compiled circuit for a numeric circuit."""
        return self.native

    @classmethod
    def from_quantum_circuit(cls, circuit: QuantumCircuitBase) -> "QrispCircuit":
        if isinstance(circuit, cls):
            return circuit
        return cls(circuit.num_qubits, circuit.num_clbits, _ir=circuit.ir.copy())