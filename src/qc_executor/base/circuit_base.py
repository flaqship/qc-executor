"""The quantum circuit interface shared by the generic and native circuits.

Unlike a conventional abstract base, this class is *concrete*: it owns the
circuit IR and implements the whole builder API on top of it.  A backend
subclass supplies only two things — which opcodes it can execute, and how to
compile the IR into its native representation:

.. code-block:: python

    class MyCircuit(QuantumCircuitBase):
        @classmethod
        def supported_opcodes(cls):
            return frozenset({OpCode.H, OpCode.CX, OpCode.RZ})

        def _build_native(self):
            return my_framework.build(self.ir)

Everything else — gate methods, parameter handling, composition, inversion,
hashing, and the Pauli-evolution helpers — is inherited and therefore identical
across the generic circuit and every backend's native circuit.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, FrozenSet, Iterator, List, Mapping, Sequence

import numpy as np
import sympy as sp

from ..parameters import Parameter, sort_parameters
from .circuit_ir import CircuitIR, Condition, Instruction
from .gate_set import GATE_DEFS, OpCode

__all__ = ["QuantumCircuitBase", "ConditionScope"]


class ConditionScope:
    """Context manager applying a classical condition to appended gates.

    Returned by :meth:`QuantumCircuitBase.if_`; not constructed directly.

    Args:
        circuit: The circuit whose appends should be gated.
        condition: The condition to apply while the scope is active.
    """

    __slots__ = ("_circuit", "_condition", "_previous")

    def __init__(self, circuit: "QuantumCircuitBase", condition: Condition):
        self._circuit = circuit
        self._condition = condition
        self._previous: "Condition | None" = None

    def __enter__(self) -> "ConditionScope":
        self._previous = self._circuit._pending_condition
        if self._previous is not None:
            raise RuntimeError("Nested classical conditions are not supported")
        self._circuit._pending_condition = self._condition
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._circuit._pending_condition = self._previous


class QuantumCircuitBase(ABC):
    """A quantum circuit, independent of any quantum framework.

    Args:
        num_qubits: Number of qubits in the circuit.
        num_clbits: Number of classical bits for mid-circuit measurement.
        _ir: Adopt this instruction store instead of starting empty.  Used by
            conversion helpers; not part of the public construction API.
    """

    def __init__(self, num_qubits: int, num_clbits: int = 0, *, _ir: "CircuitIR | None" = None):
        self._ir = _ir if _ir is not None else CircuitIR(num_qubits, num_clbits)
        self._pending_condition: "Condition | None" = None
        self._native_cache: Any = None
        self._native_revision: int = -1

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------

    @classmethod
    def supported_opcodes(cls) -> FrozenSet[OpCode]:
        """Return the opcodes this circuit type can represent natively.

        The default accepts the whole gate set.  Backends narrow this so that
        :meth:`from_quantum_circuit` lowers anything else into supported gates.
        """
        return frozenset(GATE_DEFS)

    def _build_native(self) -> Any:
        """Compile the IR into this backend's native circuit representation.

        Returns:
            The native object.  The generic circuit has none.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no native representation; "
            "override _build_native() in a backend subclass"
        )

    @property
    def native(self) -> Any:
        """The compiled native circuit, built on first use and cached.

        The cache is keyed on the IR revision, so mutating the circuit
        transparently invalidates it.
        """
        if self._native_revision != self._ir.revision:
            self._native_cache = self._build_native()
            self._native_revision = self._ir.revision
        return self._native_cache

    @classmethod
    def from_quantum_circuit(cls, circuit: "QuantumCircuitBase") -> "QuantumCircuitBase":
        """Convert any circuit into this circuit type.

        Instructions outside :meth:`supported_opcodes` are lowered first, so a
        backend only ever sees gates it declared support for.

        Args:
            circuit: The circuit to convert.

        Returns:
            ``circuit`` unchanged if it is already of this type, else a new
            instance sharing its content.
        """
        if isinstance(circuit, cls):
            return circuit
        # Imported here because the decomposition pass builds on this class.
        from .decompose import decompose_ir  # pylint: disable=import-outside-toplevel

        return cls(
            circuit.num_qubits,
            circuit.num_clbits,
            _ir=decompose_ir(circuit.ir, cls.supported_opcodes()),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ir(self) -> CircuitIR:
        """The underlying instruction store."""
        return self._ir

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the circuit."""
        return self._ir.num_qubits

    @property
    def num_clbits(self) -> int:
        """Number of classical bits in the circuit."""
        return self._ir.num_clbits

    @property
    def parameters(self) -> List[Parameter]:
        """The free parameters, sorted by ``(vector_name, index)``."""
        return sort_parameters(self._ir.free_parameters)

    @property
    def num_parameters(self) -> int:
        """Number of free parameters in the circuit."""
        return len(self._ir.free_parameters)

    @property
    def is_parameterized(self) -> bool:
        """Whether the circuit has any free parameter."""
        return bool(self._ir.free_parameters)

    # ------------------------------------------------------------------
    # Appending
    # ------------------------------------------------------------------

    def _append(self, opcode: OpCode, qubits: Sequence[int], params: Sequence[Any] = ()) -> None:
        """Append one instruction, honouring any active condition scope."""
        self._ir.append(opcode, qubits, params, condition=self._pending_condition)

    def _broadcast(self, opcode: OpCode, qubits: "int | Sequence[int]") -> None:
        """Append a single-qubit gate to one qubit or to each of several."""
        if isinstance(qubits, (int, np.integer)):
            self._append(opcode, (int(qubits),))
        else:
            for qubit in qubits:
                self._append(opcode, (int(qubit),))

    def _broadcast_param(
        self, opcode: OpCode, qubits: "int | Sequence[int]", *params: Any
    ) -> None:
        """Append a parameterised single-qubit gate to one or several qubits."""
        if isinstance(qubits, (int, np.integer)):
            self._append(opcode, (int(qubits),), params)
        else:
            for qubit in qubits:
                self._append(opcode, (int(qubit),), params)

    # -- single qubit, no angle --

    def i(self, qubits: "int | Sequence[int]") -> None:
        """Add identity gates."""
        self._broadcast(OpCode.I, qubits)

    def h(self, qubits: "int | Sequence[int]") -> None:
        """Add Hadamard gates."""
        self._broadcast(OpCode.H, qubits)

    def x(self, qubits: "int | Sequence[int]") -> None:
        """Add Pauli-X gates."""
        self._broadcast(OpCode.X, qubits)

    def y(self, qubits: "int | Sequence[int]") -> None:
        """Add Pauli-Y gates."""
        self._broadcast(OpCode.Y, qubits)

    def z(self, qubits: "int | Sequence[int]") -> None:
        """Add Pauli-Z gates."""
        self._broadcast(OpCode.Z, qubits)

    def s(self, qubits: "int | Sequence[int]") -> None:
        """Add S gates."""
        self._broadcast(OpCode.S, qubits)

    def sdag(self, qubits: "int | Sequence[int]") -> None:
        """Add S-dagger gates."""
        self._broadcast(OpCode.SDG, qubits)

    def t(self, qubits: "int | Sequence[int]") -> None:
        """Add T gates."""
        self._broadcast(OpCode.T, qubits)

    def tdag(self, qubits: "int | Sequence[int]") -> None:
        """Add T-dagger gates."""
        self._broadcast(OpCode.TDG, qubits)

    def sx(self, qubits: "int | Sequence[int]") -> None:
        """Add square-root-of-X gates."""
        self._broadcast(OpCode.SX, qubits)

    def sxdag(self, qubits: "int | Sequence[int]") -> None:
        """Add inverse square-root-of-X gates."""
        self._broadcast(OpCode.SXDG, qubits)

    # -- single qubit, with angle --

    def rx(self, qubits: "int | Sequence[int]", angle: Any) -> None:
        """Add RX rotations."""
        self._broadcast_param(OpCode.RX, qubits, angle)

    def ry(self, qubits: "int | Sequence[int]", angle: Any) -> None:
        """Add RY rotations."""
        self._broadcast_param(OpCode.RY, qubits, angle)

    def rz(self, qubits: "int | Sequence[int]", angle: Any) -> None:
        """Add RZ rotations."""
        self._broadcast_param(OpCode.RZ, qubits, angle)

    def p(self, qubits: "int | Sequence[int]", angle: Any) -> None:
        """Add phase gates."""
        self._broadcast_param(OpCode.P, qubits, angle)

    def u(self, qubits: "int | Sequence[int]", theta: Any, phi: Any, lam: Any) -> None:
        """Add general single-qubit U gates."""
        self._broadcast_param(OpCode.U, qubits, theta, phi, lam)

    # -- two qubit, no angle --

    def cx(self, control_qubit: int, target_qubit: int) -> None:
        """Add a CNOT gate."""
        self._append(OpCode.CX, (control_qubit, target_qubit))

    def cnot(self, control_qubit: int, target_qubit: int) -> None:
        """Add a CNOT gate (alias of :meth:`cx`)."""
        self.cx(control_qubit, target_qubit)

    def cy(self, control_qubit: int, target_qubit: int) -> None:
        """Add a controlled-Y gate."""
        self._append(OpCode.CY, (control_qubit, target_qubit))

    def cz(self, control_qubit: int, target_qubit: int) -> None:
        """Add a controlled-Z gate."""
        self._append(OpCode.CZ, (control_qubit, target_qubit))

    def ch(self, control_qubit: int, target_qubit: int) -> None:
        """Add a controlled-Hadamard gate."""
        self._append(OpCode.CH, (control_qubit, target_qubit))

    def cs(self, control_qubit: int, target_qubit: int) -> None:
        """Add a controlled-S gate."""
        self._append(OpCode.CS, (control_qubit, target_qubit))

    def csx(self, control_qubit: int, target_qubit: int) -> None:
        """Add a controlled-SX gate."""
        self._append(OpCode.CSX, (control_qubit, target_qubit))

    def ecr(self, control_qubit: int, target_qubit: int) -> None:
        """Add an echoed cross-resonance gate."""
        self._append(OpCode.ECR, (control_qubit, target_qubit))

    def swap(self, qubit1: int, qubit2: int) -> None:
        """Add a SWAP gate."""
        self._append(OpCode.SWAP, (qubit1, qubit2))

    def iswap(self, qubit1: int, qubit2: int) -> None:
        """Add an iSWAP gate."""
        self._append(OpCode.ISWAP, (qubit1, qubit2))

    # -- two qubit, with angle --

    def cp(self, control_qubit: int, target_qubit: int, angle: Any) -> None:
        """Add a controlled-phase gate."""
        self._append(OpCode.CP, (control_qubit, target_qubit), (angle,))

    def crx(self, control_qubit: int, target_qubit: int, angle: Any) -> None:
        """Add a controlled-RX gate."""
        self._append(OpCode.CRX, (control_qubit, target_qubit), (angle,))

    def cry(self, control_qubit: int, target_qubit: int, angle: Any) -> None:
        """Add a controlled-RY gate."""
        self._append(OpCode.CRY, (control_qubit, target_qubit), (angle,))

    def crz(self, control_qubit: int, target_qubit: int, angle: Any) -> None:
        """Add a controlled-RZ gate."""
        self._append(OpCode.CRZ, (control_qubit, target_qubit), (angle,))

    def rxx(self, qubit1: int, qubit2: int, angle: Any) -> None:
        """Add an XX rotation."""
        self._append(OpCode.RXX, (qubit1, qubit2), (angle,))

    def ryy(self, qubit1: int, qubit2: int, angle: Any) -> None:
        """Add a YY rotation."""
        self._append(OpCode.RYY, (qubit1, qubit2), (angle,))

    def rzz(self, qubit1: int, qubit2: int, angle: Any) -> None:
        """Add a ZZ rotation."""
        self._append(OpCode.RZZ, (qubit1, qubit2), (angle,))

    def rzx(self, qubit1: int, qubit2: int, angle: Any) -> None:
        """Add a ZX rotation, with Z on ``qubit1`` and X on ``qubit2``."""
        self._append(OpCode.RZX, (qubit1, qubit2), (angle,))

    # -- three qubit --

    def ccx(self, control_qubit1: int, control_qubit2: int, target_qubit: int) -> None:
        """Add a Toffoli gate."""
        self._append(OpCode.CCX, (control_qubit1, control_qubit2, target_qubit))

    def toffoli(self, control_qubit1: int, control_qubit2: int, target_qubit: int) -> None:
        """Add a Toffoli gate (alias of :meth:`ccx`)."""
        self.ccx(control_qubit1, control_qubit2, target_qubit)

    def cswap(self, control_qubit: int, qubit1: int, qubit2: int) -> None:
        """Add a Fredkin (controlled-SWAP) gate."""
        self._append(OpCode.CSWAP, (control_qubit, qubit1, qubit2))

    # ------------------------------------------------------------------
    # Structural and non-unitary
    # ------------------------------------------------------------------

    def barrier(self, qubits: "int | Sequence[int] | None" = None) -> None:
        """Add a barrier across the given qubits, or all of them by default."""
        if qubits is None:
            targets = tuple(range(self.num_qubits))
        elif isinstance(qubits, (int, np.integer)):
            targets = (int(qubits),)
        else:
            targets = tuple(int(q) for q in qubits)
        self._append(OpCode.BARRIER, targets)

    def measure(
        self,
        qubits: "int | Sequence[int] | None" = None,
        clbits: "int | Sequence[int] | None" = None,
    ) -> None:
        """Measure qubits into classical bits.

        Classical bits are allocated automatically when omitted, growing the
        classical register as needed.

        Args:
            qubits: Qubits to measure.  Defaults to every qubit.
            clbits: Destination bits.  Defaults to freshly allocated ones.

        Raises:
            ValueError: If the qubit and classical bit counts differ.
        """
        targets = self._normalize_targets(qubits)
        if clbits is None:
            start = self._ir.num_clbits
            self._ir.ensure_clbits(start + len(targets))
            destinations = tuple(range(start, start + len(targets)))
        elif isinstance(clbits, (int, np.integer)):
            destinations = (int(clbits),)
        else:
            destinations = tuple(int(c) for c in clbits)

        if len(destinations) != len(targets):
            raise ValueError(
                f"measure needs one classical bit per qubit, got {len(targets)} "
                f"qubit(s) and {len(destinations)} classical bit(s)"
            )
        for qubit, clbit in zip(targets, destinations):
            self._ir.append(
                OpCode.MEASURE, (qubit,), (), (clbit,), condition=self._pending_condition
            )

    def measure_all(self) -> None:
        """Measure every qubit into its own classical bit."""
        self.measure(range(self.num_qubits))

    def reset(self, qubits: "int | Sequence[int] | None" = None) -> None:
        """Reset qubits to the zero state.

        Args:
            qubits: Qubits to reset.  Defaults to every qubit.
        """
        for qubit in self._normalize_targets(qubits):
            self._append(OpCode.RESET, (qubit,))

    def if_(self, clbits: "int | Sequence[int]", value: int) -> ConditionScope:
        """Gate the instructions in the ``with`` body on a classical value.

        .. code-block:: python

            circuit.measure(0, 0)
            with circuit.if_(0, 1):
                circuit.x(1)

        Args:
            clbits: Classical bits to test, least significant first.
            value: The value those bits must equal.

        Returns:
            A context manager applying the condition to appended instructions.
        """
        if isinstance(clbits, (int, np.integer)):
            bits = (int(clbits),)
        else:
            bits = tuple(int(c) for c in clbits)
        for clbit in bits:
            if not 0 <= clbit < self.num_clbits:
                raise ValueError(
                    f"classical bit index {clbit} is out of range for "
                    f"{self.num_clbits} classical bit(s)"
                )
        return ConditionScope(self, Condition(bits, int(value)))

    def _normalize_targets(self, qubits: "int | Sequence[int] | None") -> tuple:
        """Resolve a qubit argument into a tuple of indices."""
        if qubits is None:
            return tuple(range(self.num_qubits))
        if isinstance(qubits, (int, np.integer)):
            return (int(qubits),)
        return tuple(int(q) for q in qubits)

    # ------------------------------------------------------------------
    # Backend-independent algorithms
    # ------------------------------------------------------------------

    def pauli_string(self, pauli_string: str) -> None:
        """Apply a Pauli string to the circuit.

        Args:
            pauli_string: Pauli string to apply, one character per qubit.

        Raises:
            ValueError: If the string length does not match the qubit count.
        """
        if len(pauli_string) != self.num_qubits:
            raise ValueError("Pauli string length does not match number of qubits")

        for i, pauli in enumerate(pauli_string[::-1]):
            if pauli == "X":
                self.x(i)
            elif pauli == "Y":
                self.y(i)
            elif pauli == "Z":
                self.z(i)
            elif pauli == "I":
                pass  # Identity gate (I) can be skipped as it does nothing

    def _apply_basis_change(
        self, paulis: List[str], qubits: List[int], working_qubits: List[int]
    ) -> None:
        """Apply basis change for non-trivial Paulis."""
        for p, q in zip(paulis, qubits):
            if p == "X":
                self.h(working_qubits[q])
            elif p == "Y":
                self.sdag(working_qubits[q])
                self.h(working_qubits[q])
            elif p != "Z":
                raise ValueError(f"Unknown Pauli operator: {p}")

    def _undo_basis_change(
        self, paulis: List[str], qubits: List[int], working_qubits: List[int]
    ) -> None:
        """Undo basis change for non-trivial Paulis."""
        for p, q in zip(paulis, qubits):
            if p == "X":
                self.h(working_qubits[q])
            elif p == "Y":
                self.h(working_qubits[q])
                self.s(working_qubits[q])

    def _apply_cnot_ladder(self, qubits: List[int], working_qubits: List[int]) -> None:
        """Apply the forward CNOT ladder for Pauli evolution."""
        if not qubits:
            return
        control = qubits[0]
        for target in qubits[1:]:
            self.cx(working_qubits[control], working_qubits[target])
            control = target

    def _undo_cnot_ladder(self, qubits: List[int], working_qubits: List[int]) -> None:
        """Undo the CNOT ladder after the phase rotation."""
        if not qubits:
            return
        control = qubits[-1]
        for target in reversed(qubits[:-1]):
            self.cx(working_qubits[target], working_qubits[control])
            control = target

    @staticmethod
    def _evolution_angle(coeff: Any, parameter: Any, scale: float = 2.0) -> Any:
        """Combine an operator coefficient and evolution parameter into an angle.

        Symbolic coefficients stay symbolic; numeric ones are checked for a
        vanishing imaginary part and reduced to a float.

        Args:
            coeff: The operator coefficient.
            parameter: The evolution parameter, itself possibly symbolic.
            scale: Factor applied to the product.

        Returns:
            The rotation angle, numeric or symbolic.

        Raises:
            ValueError: If a numeric coefficient has a non-zero imaginary part.
        """
        if isinstance(coeff, sp.Basic) and coeff.free_symbols:
            return scale * coeff * parameter
        value = np.real_if_close(complex(coeff))
        if np.iscomplexobj(value):
            raise ValueError("Complex coefficients are not supported")
        return scale * float(value.real) * parameter

    def pauli_evolution(
        self,
        operator: Any,
        parameter: Any,
        working_qubits: "List[int] | None" = None,
    ) -> None:
        """Apply the Pauli evolution ``exp(itP)`` for a single Pauli term.

        Args:
            operator: A quantum operator holding exactly one Pauli string.
            parameter: The evolution parameter, numeric or symbolic.
            working_qubits: Physical qubits to use, defaulting to ``0..n-1``.

        Raises:
            ValueError: If the operator holds more than one Pauli string.
        """
        pauli_str = operator.paulis[0]
        coeffs = operator.coeffs
        if len(coeffs) != 1:
            raise ValueError("Only operators with single Pauli strings are supported")

        angle = self._evolution_angle(coeffs[0], parameter)

        qubits = [i for i, p in enumerate(pauli_str[::-1]) if p != "I"][::-1]
        paulis = [p for p in pauli_str if p != "I"]

        if working_qubits is None:
            working_qubits = list(range(len(pauli_str)))

        self._apply_basis_change(paulis, qubits, working_qubits)

        if qubits:
            self._apply_cnot_ladder(qubits, working_qubits)
            self.rz(working_qubits[qubits[-1]], angle)
            self._undo_cnot_ladder(qubits, working_qubits)

        self._undo_basis_change(paulis, qubits, working_qubits)

    def controlled_pauli_evolution(
        self,
        operator: Any,
        parameter: Any,
        control_qubit: int,
        working_qubits: "List[int] | None" = None,
    ) -> None:
        """Apply a controlled Pauli evolution ``exp(itP)``.

        Args:
            operator: A quantum operator holding exactly one Pauli string.
            parameter: The evolution parameter, numeric or symbolic.
            control_qubit: The control qubit.
            working_qubits: Physical qubits to use for the Pauli support.

        Raises:
            ValueError: If the operator holds more than one Pauli string.
        """
        pauli_str = operator.paulis[0]
        coeffs = operator.coeffs
        if len(coeffs) != 1:
            raise ValueError("Only operators with single Pauli strings are supported")

        qubits = [i for i, p in enumerate(pauli_str[::-1]) if p != "I"][::-1]
        paulis = [p for p in pauli_str if p != "I"]

        if not paulis:
            self.rz(control_qubit, self._evolution_angle(coeffs[0], parameter, scale=-1.0))
            return

        angle = self._evolution_angle(coeffs[0], parameter)

        if working_qubits is None:
            working_qubits = list(range(len(pauli_str) + 1))
            working_qubits.remove(control_qubit)

        self._apply_basis_change(paulis, qubits, working_qubits)

        if qubits:
            self._apply_cnot_ladder(qubits, working_qubits)
            self.crz(control_qubit, working_qubits[qubits[-1]], angle)
            self._undo_cnot_ladder(qubits, working_qubits)

        self._undo_basis_change(paulis, qubits, working_qubits)

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    def compose(
        self,
        qc: "QuantumCircuitBase",
        qubits: "Sequence[int] | None" = None,
        clbits: "Sequence[int] | None" = None,
    ) -> "QuantumCircuitBase":
        """Append another circuit's instructions onto this one, in place.

        Args:
            qc: The circuit to append.
            qubits: Where ``qc``'s qubits land, defaulting to the identity.
            clbits: Where ``qc``'s classical bits land.

        Returns:
            This circuit, to allow chaining.

        Raises:
            TypeError: If ``qc`` is not a quantum circuit.
        """
        if not isinstance(qc, QuantumCircuitBase):
            raise TypeError(f"can only compose with a quantum circuit, got {type(qc).__name__}")
        self._ir.extend(qc.ir, qubits, clbits)
        return self

    def assign_parameters(self, parameters: Mapping[Any, float]) -> "QuantumCircuitBase":
        """Substitute values for parameters, in place.

        Args:
            parameters: Values keyed by :class:`~qc_executor.parameters.Parameter`
                or by parameter name.

        Returns:
            This circuit, to allow chaining.
        """
        binding = {
            (key if isinstance(key, Parameter) else Parameter(str(key))): value
            for key, value in parameters.items()
        }
        self._ir = self._ir.substitute(binding)
        return self

    def invert(self) -> "QuantumCircuitBase":
        """Return the adjoint of this circuit."""
        return type(self)(self.num_qubits, self.num_clbits, _ir=self._ir.inverse())

    def copy(self) -> "QuantumCircuitBase":
        """Return an independent copy of this circuit."""
        return type(self)(self.num_qubits, self.num_clbits, _ir=self._ir.copy())

    def circuit_metrics(self) -> Dict[str, int]:
        """Return how often each gate name appears in the circuit."""
        return self._ir.count_ops()

    def draw(self) -> str:
        """Return a plain-text listing of the instruction sequence."""
        lines = []
        for instruction in self._ir:
            parts = [f"{instruction.name.upper():<8} {list(instruction.qubits)}"]
            if instruction.params:
                parts.append(f"  params={list(instruction.params)}")
            if instruction.clbits:
                parts.append(f"  clbits={list(instruction.clbits)}")
            if instruction.condition is not None:
                condition = instruction.condition
                parts.append(f"  if {list(condition.clbits)} == {condition.value}")
            lines.append("".join(parts))
        return "\n".join(lines)

    def fingerprint(self) -> bytes:
        """Return a stable digest of the circuit's content."""
        return self._ir.fingerprint()

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._ir)

    def __getitem__(self, index: int) -> Instruction:
        return self._ir[index]

    def __iter__(self) -> Iterator[Instruction]:
        return iter(self._ir)

    def __hash__(self) -> int:
        return hash(self._ir.fingerprint())

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, QuantumCircuitBase) and self._ir == other.ir

    def __str__(self) -> str:
        return self.draw()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(num_qubits={self.num_qubits}, "
            f"num_clbits={self.num_clbits}, instructions={len(self._ir)})"
        )
