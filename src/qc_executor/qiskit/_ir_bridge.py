"""Translation between the framework-independent circuit IR and Qiskit.

This is the Qiskit plugin's half of the translation layer: the core package
knows nothing about Qiskit, and everything needed to turn a
:class:`~qc_executor.base.circuit_ir.CircuitIR` into a ``qiskit.QuantumCircuit``
and back lives here.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.circuit import ClassicalRegister, QuantumRegister
from qiskit.quantum_info import PauliList, SparsePauliOp

from ..base.circuit_ir import CircuitIR, Condition, Instruction
from ..base.decompose import UnsupportedGateError
from ..base.gate_set import OPCODE_BY_NAME, OpCode
from ..base.operator_ir import PauliIR
from ._sympy_bridge import from_qiskit_expr, to_qiskit_expr

__all__ = [
    "ir_to_qiskit",
    "qiskit_to_ir",
    "pauli_ir_to_sparse_pauli_op",
    "sparse_pauli_op_to_pauli_ir",
    "SUPPORTED_OPCODES",
]


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


#: How often :func:`qiskit_to_ir` lets Qiskit unroll composite instructions
#: before giving up.  Library circuits flatten in two or three rounds.
_MAX_DECOMPOSE_ROUNDS = 10


def qiskit_to_ir(circuit: QiskitQuantumCircuit) -> CircuitIR:
    """Import a Qiskit circuit into the framework-independent circuit IR.

    The inverse of :func:`ir_to_qiskit`.  Composite instructions -- the blocks a
    ``qiskit.circuit.library`` circuit is built from -- are unrolled with Qiskit's
    own ``decompose`` until every instruction has an :class:`OpCode`.  Angles go
    through :func:`from_qiskit_expr`, so ``ParameterVector`` elements arrive as
    :class:`~qc_executor.parameters.Parameter` symbols of the same name.

    The global phase is dropped, as in the IR's own decomposition pass; it does
    not affect expectation values, probabilities or samples.

    Unlike :meth:`QiskitCircuit.from_qiskit`, which only carries a circuit for
    the Qiskit backend, the result is a real instruction store that runs on
    every backend.  That is also why an ISA-transpiled circuit is refused: its
    layout would be silently lost.

    Args:
        circuit: The Qiskit circuit to import.

    Returns:
        The equivalent :class:`~qc_executor.base.circuit_ir.CircuitIR`.

    Raises:
        UnsupportedGateError: If the circuit carries a transpile layout, or an
            instruction (control flow, ``delay``, an opaque gate) has no
            :class:`OpCode` and cannot be decomposed into one.
    """
    if getattr(circuit, "layout", None) is not None:
        raise UnsupportedGateError(
            "a transpiled circuit carries a layout the IR cannot express; "
            "use QiskitCircuit.from_qiskit to run it on the Qiskit backend"
        )
    circuit = _flatten(circuit)

    ir = CircuitIR(circuit.num_qubits, circuit.num_clbits)
    for item in circuit.data:
        operation = item.operation
        qubits = [circuit.find_bit(bit).index for bit in item.qubits]
        clbits = [circuit.find_bit(bit).index for bit in item.clbits]
        opcode = OPCODE_BY_NAME[operation.name]
        if opcode is OpCode.BARRIER:
            ir.append(opcode, qubits)
            continue
        params = [from_qiskit_expr(param) for param in operation.params]
        ir.append(opcode, qubits, params, clbits)
    return ir


def _flatten(circuit: QiskitQuantumCircuit) -> QiskitQuantumCircuit:
    """Unroll composite instructions until every name has an :class:`OpCode`.

    Raises:
        UnsupportedGateError: If an instruction survives every round.
    """
    for _ in range(_MAX_DECOMPOSE_ROUNDS):
        unknown = _unknown_names(circuit)
        if not unknown:
            return circuit
        unrolled = circuit.decompose(sorted(unknown))
        if unrolled == circuit:
            # Qiskit has no definition for what is left, so another round won't help.
            break
        circuit = unrolled
    unknown = _unknown_names(circuit)
    if unknown:
        raise UnsupportedGateError(
            f"cannot import {sorted(unknown)} into the circuit IR: no opcode and "
            "no decomposition into supported gates"
        )
    return circuit


def _unknown_names(circuit: QiskitQuantumCircuit) -> frozenset:
    """Names of the instructions in ``circuit`` that have no :class:`OpCode`."""
    return frozenset(
        item.operation.name for item in circuit.data if item.operation.name not in OPCODE_BY_NAME
    )


def pauli_ir_to_sparse_pauli_op(ir: PauliIR) -> SparsePauliOp:
    """Translate a sparse Pauli operator into Qiskit's ``SparsePauliOp``.

    The symplectic columns map across **without** reversal: both sides index the
    ``z``/``x`` arrays by qubit number.  Only the rendered label differs, because
    this project writes qubit 0 leftmost and Qiskit writes it rightmost, so
    ``["ZI"]`` here becomes ``["IZ"]`` there and both mean Z on qubit 0.

    Args:
        ir: The operator to translate.

    Returns:
        The equivalent ``SparsePauliOp``.
    """
    pauli_list = PauliList.from_symplectic(ir.z, ir.x)
    coeffs = [to_qiskit_expr(coeff) for coeff in ir.coeffs]
    return SparsePauliOp(pauli_list, np.array(coeffs, dtype=object if ir.symbolic else complex))


def sparse_pauli_op_to_pauli_ir(operator: SparsePauliOp) -> PauliIR:
    """Translate a Qiskit ``SparsePauliOp`` into the shared representation.

    Args:
        operator: The Qiskit operator to translate.

    Returns:
        The equivalent :class:`~qc_executor.base.operator_ir.PauliIR`.
    """
    paulis = operator.paulis
    coeffs = [from_qiskit_expr(coeff) for coeff in operator.coeffs]
    # Build the coefficient column (splitting out symbolic entries) via a
    # throwaway all-identity operator, then attach the real Pauli data.
    numeric = PauliIR.from_labels(
        ["I" * operator.num_qubits] * len(coeffs), coeffs, operator.num_qubits
    )
    return PauliIR(
        operator.num_qubits,
        np.asarray(paulis.z),
        np.asarray(paulis.x),
        numeric.coeffs_array,
        numeric.symbolic,
    )


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
