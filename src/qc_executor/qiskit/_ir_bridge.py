"""Compilation of the framework-independent circuit IR into a Qiskit circuit.

This is the Qiskit plugin's half of the translation layer: the core package
knows nothing about Qiskit, and everything needed to turn a
:class:`~qc_executor.base.circuit_ir.CircuitIR` into a ``qiskit.QuantumCircuit``
lives here.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.circuit import ClassicalRegister, QuantumRegister

from ..base.circuit_ir import CircuitIR, Condition, Instruction
from ..base.gate_set import OpCode
from ._sympy_bridge import to_qiskit_expr

__all__ = ["ir_to_qiskit", "SUPPORTED_OPCODES"]


def _gate(method: str) -> Callable[[Any, Instruction, tuple], None]:
    """Build an emitter forwarding angles then qubits to a Qiskit method."""

    def emit(circuit: Any, instruction: Instruction, params: tuple) -> None:
        getattr(circuit, method)(*params, *instruction.qubits)

    return emit


#: Emitters keyed by opcode.  A table keeps this a single dispatch rather than
#: one branch per gate, which also keeps the branch coverage flat.
_EMITTERS: Dict[OpCode, Callable[[Any, Instruction, tuple], None]] = {
    OpCode.I: _gate("id"),
    OpCode.H: _gate("h"),
    OpCode.X: _gate("x"),
    OpCode.Y: _gate("y"),
    OpCode.Z: _gate("z"),
    OpCode.S: _gate("s"),
    OpCode.SDG: _gate("sdg"),
    OpCode.T: _gate("t"),
    OpCode.TDG: _gate("tdg"),
    OpCode.SX: _gate("sx"),
    OpCode.SXDG: _gate("sxdg"),
    OpCode.RX: _gate("rx"),
    OpCode.RY: _gate("ry"),
    OpCode.RZ: _gate("rz"),
    OpCode.P: _gate("p"),
    OpCode.U: _gate("u"),
    OpCode.CX: _gate("cx"),
    OpCode.CY: _gate("cy"),
    OpCode.CZ: _gate("cz"),
    OpCode.CH: _gate("ch"),
    OpCode.ECR: _gate("ecr"),
    OpCode.SWAP: _gate("swap"),
    OpCode.ISWAP: _gate("iswap"),
    OpCode.CS: _gate("cs"),
    OpCode.CSX: _gate("csx"),
    OpCode.CP: _gate("cp"),
    OpCode.CRX: _gate("crx"),
    OpCode.CRY: _gate("cry"),
    OpCode.CRZ: _gate("crz"),
    OpCode.RXX: _gate("rxx"),
    OpCode.RYY: _gate("ryy"),
    OpCode.RZZ: _gate("rzz"),
    OpCode.RZX: _gate("rzx"),
    OpCode.CCX: _gate("ccx"),
    OpCode.CSWAP: _gate("cswap"),
    OpCode.RESET: _gate("reset"),
}

#: Opcodes the Qiskit backend can emit directly.  Structural instructions are
#: handled separately by :func:`ir_to_qiskit`.
SUPPORTED_OPCODES = frozenset(_EMITTERS) | {OpCode.BARRIER, OpCode.MEASURE}


def ir_to_qiskit(ir: CircuitIR) -> QiskitQuantumCircuit:
    """Compile a circuit IR into a Qiskit circuit.

    Symbolic angles are translated to ``ParameterExpression`` through the shared
    factory, so a parameter keeps one identity across every circuit and operator
    in the process — which Qiskit's binding and differentiation both require.

    Args:
        ir: The circuit to compile.

    Returns:
        The equivalent ``qiskit.QuantumCircuit``.

    Raises:
        NotImplementedError: If the IR contains an opcode Qiskit cannot express.
    """
    registers: list = [QuantumRegister(ir.num_qubits, "q")]
    if ir.num_clbits:
        registers.append(ClassicalRegister(ir.num_clbits, "c"))
    circuit = QiskitQuantumCircuit(*registers)

    for instruction in ir:
        # The shared factory is deliberate: compiling the same IR twice, or a
        # circuit and an observable separately, must yield the *same* Qiskit
        # parameter objects, because Qiskit compares them by UUID rather than
        # by name and binding would otherwise fail.
        params = tuple(to_qiskit_expr(p) for p in instruction.params)
        if instruction.condition is None:
            _emit(circuit, instruction, params)
            continue
        # Qiskit 2.x removed c_if, so conditions become if_test blocks.
        target = _condition_target(circuit, instruction.condition)
        with circuit.if_test((target, instruction.condition.value)):
            _emit(circuit, instruction, params)

    return circuit


def _condition_target(circuit: QiskitQuantumCircuit, condition: Condition) -> Any:
    """Resolve a classical condition to something ``if_test`` accepts.

    Qiskit tests either a single classical bit or a whole register; it has no
    form for an arbitrary subset of bits.

    Args:
        circuit: The circuit being built.
        condition: The condition to resolve.

    Returns:
        A ``Clbit`` or a ``ClassicalRegister``.

    Raises:
        NotImplementedError: If the condition spans an arbitrary subset of bits.
    """
    if len(condition.clbits) == 1:
        return circuit.clbits[condition.clbits[0]]
    if tuple(condition.clbits) == tuple(range(circuit.num_clbits)):
        return circuit.cregs[0]
    raise NotImplementedError(
        "Qiskit can only condition on a single classical bit or on a whole "
        f"register, not on the subset {list(condition.clbits)}"
    )


def _emit(circuit: QiskitQuantumCircuit, instruction: Instruction, params: tuple) -> None:
    """Emit one instruction onto a Qiskit circuit."""
    if instruction.opcode is OpCode.BARRIER:
        circuit.barrier(*instruction.qubits)
        return
    if instruction.opcode is OpCode.MEASURE:
        circuit.measure(instruction.qubits[0], instruction.clbits[0])
        return

    emitter = _EMITTERS.get(instruction.opcode)
    if emitter is None:
        raise NotImplementedError(
            f"'{instruction.name}' has no Qiskit equivalent in this translation layer"
        )
    emitter(circuit, instruction, params)
