"""Pauli propagation native circuit datatype."""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Sequence, cast

import sympy as sp

from qc_executor.base.circuit_base import QuantumCircuitBase
from qc_executor.base.circuit_ir import CircuitIR, Instruction
from qc_executor.base.gate_set import GATE_DEFS, OpCode
from qc_executor.parameters import Parameter

from .utils.gates import CliffordGate, Gate, LayerBarrier, PauliRotation

#: Opcodes the propagation engine executes directly.  Everything else is lowered
#: into these by the shared decomposition pass, which is how this backend gained
#: support for gates it used to reject outright (crx, cry, crz, rzx, ecr, ccx,
#: cswap, iswap, u, ...).
_SUPPORTED = frozenset(
    {
        OpCode.H,
        OpCode.S,
        OpCode.X,
        OpCode.Y,
        OpCode.Z,
        OpCode.CX,
        OpCode.CZ,
        OpCode.SWAP,
        OpCode.RX,
        OpCode.RY,
        OpCode.RZ,
        OpCode.RXX,
        OpCode.RYY,
        OpCode.RZZ,
        OpCode.BARRIER,
    }
)

#: Opcode -> the engine's Clifford gate name.
_CLIFFORD_NAMES: Dict[OpCode, str] = {
    OpCode.H: "H",
    OpCode.S: "S",
    OpCode.X: "X",
    OpCode.Y: "Y",
    OpCode.Z: "Z",
    OpCode.CX: "CNOT",
    OpCode.CZ: "CZ",
    OpCode.SWAP: "SWAP",
}


def _qubit_arg(qubits: Sequence[int]) -> int | List[int]:
    """Return the single qubit index or the full list, matching gate constructors."""
    return list(qubits) if len(qubits) > 1 else qubits[0]


def _clone_gate(gate: Gate | LayerBarrier, num_qubits: int) -> Gate | LayerBarrier | None:
    """Create a fresh copy of a gate instruction (None for unknown gate types)."""
    if isinstance(gate, LayerBarrier):
        return LayerBarrier()
    if isinstance(gate, PauliRotation):
        return PauliRotation(
            list(gate.symbols),
            _qubit_arg(gate.qubits),
            num_qubits,
            param_expr=gate.param_expr,
            param_value=gate.param_value,
        )
    if isinstance(gate, CliffordGate):
        return CliffordGate(gate.gate_type, _qubit_arg(gate.qubits), num_qubits)
    return None


def _split_angle(angle: Any) -> tuple[sp.Expr | None, float | None]:
    """Split an IR angle into the engine's symbolic / concrete pair.

    Args:
        angle: A number or a SymPy expression.

    Returns:
        ``(expression, None)`` for a symbolic angle, ``(None, value)`` otherwise.
    """
    if isinstance(angle, sp.Basic) and angle.free_symbols:
        return angle, None
    return None, float(angle)


class PauliPropagationCircuit(QuantumCircuitBase):
    """Backend-native circuit representation for Pauli propagation.

    Instructions are recorded in the shared circuit IR and compiled into the
    engine's flat gate list on demand.  Building the gate list from the IR is
    what removed this backend's dependency on Qiskit for circuit conversion.

    Args:
        num_qubits: Number of qubits in the circuit.
        num_clbits: Number of classical bits (unused; measurement is not
            representable in the Heisenberg picture).
        gates: Adopt this gate list directly instead of compiling one.  Used by
            :meth:`replace_gate` and by callers holding engine gates already.
        _ir: Adopt this instruction store instead of starting empty.
    """

    def __init__(
        self,
        num_qubits: int = 0,
        num_clbits: int = 0,
        *,
        gates: Sequence[Gate | LayerBarrier] | None = None,
        _ir: CircuitIR | None = None,
    ):
        if gates is not None and _ir is None and num_qubits == 0:
            raise ValueError("num_qubits is required when constructing from a gate list")
        super().__init__(num_qubits, num_clbits, _ir=_ir)
        self._gates_override: List[Gate | LayerBarrier] | None = (
            list(gates) if gates is not None else None
        )

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------

    @classmethod
    def supported_opcodes(cls) -> FrozenSet[OpCode]:
        """Return the opcodes the propagation engine executes directly."""
        return _SUPPORTED

    def _build_native(self) -> List[Gate | LayerBarrier]:
        """Compile the instruction store into the engine's gate list.

        The store is lowered first: the shared builder accepts the whole gate
        set, so a circuit built directly through it may hold gates the engine
        cannot execute (``sdag``, ``cy``, ``crz``, ...).
        """
        if self._gates_override is not None:
            return self._gates_override
        return [self._to_gate(instruction) for instruction in self._lowered_ir()]

    def _to_gate(self, instruction: Instruction) -> Gate | LayerBarrier:
        """Translate one instruction into an engine gate.

        Raises:
            NotImplementedError: For classically conditioned instructions, which
                the Heisenberg picture cannot express.
        """
        opcode = instruction.opcode
        if instruction.condition is not None:
            # Without this the gate would be applied unconditionally, which is a
            # wrong answer rather than a missing feature.
            raise NotImplementedError(
                "Classically conditioned gates are not supported by the "
                "Pauli-propagation backend: it propagates operators in the "
                "Heisenberg picture, where no measurement outcome exists to "
                f"condition '{instruction.name}' on."
            )
        if opcode is OpCode.BARRIER:
            return LayerBarrier()

        clifford = _CLIFFORD_NAMES.get(opcode)
        if clifford is not None:
            return CliffordGate(clifford, _qubit_arg(instruction.qubits), self.num_qubits)

        generators = GATE_DEFS[opcode].pauli_rotation
        if generators is None:
            raise NotImplementedError(f"'{instruction.name}' has no Pauli-propagation equivalent")
        expression, value = _split_angle(instruction.params[0])
        return PauliRotation(
            list(generators),
            _qubit_arg(instruction.qubits),
            self.num_qubits,
            param_expr=expression,
            param_value=value,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def gates(self) -> List[Gate | LayerBarrier]:
        """Return a shallow copy of the engine gate instructions."""
        return list(self.native)

    @property
    def parameter_symbols(self) -> Dict[str, Parameter]:
        """Return the parameters keyed by name, e.g. ``{"theta[0]": theta[0]}``."""
        if self._gates_override is not None:
            found: Dict[str, Parameter] = {}
            for gate in self._gates_override:
                if isinstance(gate, PauliRotation) and gate.param_expr is not None:
                    for symbol in gate.param_expr.free_symbols:
                        found.setdefault(symbol.name, symbol)
            return found
        return {parameter.name: parameter for parameter in self.parameters}

    @property
    def parameter_names(self) -> List[str]:
        """Return the parameter names used by the circuit."""
        return list(self.parameter_symbols)

    @property
    def parameters(self) -> List[Parameter]:
        """Return the free parameters, sorted by ``(vector_name, index)``."""
        if self._gates_override is not None:
            from qc_executor.parameters import (  # pylint: disable=import-outside-toplevel
                sort_parameters,
            )

            return sort_parameters(self.parameter_symbols.values())
        return super().parameters

    def draw(self) -> str:
        """Return a plain-text listing of the engine gate sequence."""
        return "\n".join(str(gate) for gate in self.native)

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    def replace_gate(self, index: int, gate: Gate | LayerBarrier) -> "PauliPropagationCircuit":
        """Return a copy with the gate at ``index`` replaced.

        Used by the parameter-shift rule, which needs one rotation's angle
        displaced while everything else stays put.  The result carries an
        explicit gate list, because a shifted gate need not correspond to any
        instruction in the original store.

        Args:
            index: Position of the gate to replace.
            gate: The replacement.

        Returns:
            A new circuit; this one is untouched.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        gates = self.gates
        if not 0 <= index < len(gates):
            raise IndexError(f"gate index {index} out of range for {len(gates)} gate(s)")
        gates[index] = gate
        return type(self)(self.num_qubits, gates=gates)

    def assign_parameters(self, parameters: Dict[Any, float]) -> "PauliPropagationCircuit":
        """Return a copy with values substituted for parameters.

        Only the adopted-gate-list case needs this override; a circuit backed
        by the instruction store binds through the base implementation.

        Args:
            parameters: Values keyed by :class:`~qc_executor.parameters.Parameter`
                or by name.

        Returns:
            A new bound circuit; this one is untouched.
        """
        if self._gates_override is None:
            return cast("PauliPropagationCircuit", super().assign_parameters(parameters))

        binding = {
            (key if isinstance(key, Parameter) else Parameter(str(key))): value
            for key, value in parameters.items()
        }
        bound: List[Gate | LayerBarrier] = []
        for gate in self._gates_override:
            if isinstance(gate, PauliRotation) and gate.param_expr is not None:
                result = gate.param_expr.xreplace(dict(binding))
                expression, value = _split_angle(result)
                bound.append(
                    PauliRotation(
                        list(gate.symbols),
                        _qubit_arg(gate.qubits),
                        self.num_qubits,
                        param_expr=expression,
                        param_value=value,
                    )
                )
            else:
                cloned = _clone_gate(gate, self.num_qubits)
                if cloned is not None:
                    bound.append(cloned)
        return type(self)(self.num_qubits, gates=bound)

    def copy(self) -> "PauliPropagationCircuit":
        """Return an independent copy of this circuit."""
        if self._gates_override is not None:
            return type(self)(
                self.num_qubits,
                gates=[
                    cloned
                    for cloned in (
                        _clone_gate(gate, self.num_qubits) for gate in self._gates_override
                    )
                    if cloned is not None
                ],
            )
        return type(self)(self.num_qubits, self.num_clbits, _ir=self._ir.copy())

    def measure(self, qubits=None, clbits=None):
        """Measurement is not representable in the Heisenberg picture."""
        raise NotImplementedError("Measurement is not represented in PauliPropagationCircuit.")

    def from_qasm(self, qasm: str) -> None:
        """QASM import is not supported for this datatype."""
        raise NotImplementedError("QASM import is intentionally not supported for this datatype.")

    def to_qasm(self) -> str:
        """QASM export is not supported for this datatype."""
        raise NotImplementedError("QASM export is intentionally not supported for this datatype.")

    def circuit_metrics(self) -> dict:
        """Return gate counts and depth for the compiled gate list."""
        gates = self.native
        gate_count = sum(1 for gate in gates if not isinstance(gate, LayerBarrier))
        depth = gate_count
        return {
            "num_qubits": self.num_qubits,
            "num_gates": gate_count,
            "depth": depth,
            "num_parameters": self.num_parameters,
        }

    @property
    def num_parameters(self) -> int:
        """Number of free parameters in the circuit."""
        if self._gates_override is not None:
            return len(self.parameter_symbols)
        return super().num_parameters

    @property
    def is_parameterized(self) -> bool:
        """Whether the circuit has any free parameter."""
        return self.num_parameters > 0

    def __hash__(self) -> int:
        if self._gates_override is None:
            return super().__hash__()
        return hash(
            (
                self.num_qubits,
                tuple(
                    (
                        type(gate).__name__,
                        getattr(gate, "gate_type", None),
                        tuple(getattr(gate, "symbols", ()) or ()),
                        tuple(getattr(gate, "qubits", ()) or ()),
                        str(getattr(gate, "param_expr", None)),
                        getattr(gate, "param_value", None),
                    )
                    for gate in self._gates_override
                ),
            )
        )

    def __str__(self) -> str:
        return self.draw()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(num_qubits={self.num_qubits}, "
            f"num_gates={len(self.native)})"
        )
