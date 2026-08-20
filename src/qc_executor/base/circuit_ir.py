"""The columnar instruction store behind every quantum circuit.

Circuits are held as parallel packed arrays rather than one Python object per
gate.  A gate costs roughly 20-30 bytes here against ~130 bytes for a numeric
Qiskit circuit and ~650 bytes for a symbolically parameterised one, which is
what makes million-gate circuits practical.

Layout::

    _opcodes    array('H')                    2 B per instruction
    _qubits     array('I') + _qubit_off       4 B per qubit + 4 B per instruction
    _params     array('d') + _param_off       8 B per angle + 4 B per instruction
    _symbolic   dict[slot, sympy.Expr]        sparse; NaN in _params marks a slot
    _clbits     dict[index, tuple[int, ...]]  sparse; empty for gate-only circuits
    _conditions dict[index, Condition]        sparse

Numeric angles live in the packed ``array('d')`` so backends can read them as a
single NumPy buffer; only symbolic angles pay for a Python object, and they are
flagged by a NaN sentinel in the numeric column.
"""

from __future__ import annotations

import hashlib
import math
from array import array
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import sympy as sp

from ..parameters import Parameter, canonicalize
from .gate_set import GATE_DEFS, VARIABLE_QUBITS, GateDef, OpCode

__all__ = ["Condition", "Instruction", "CircuitIR"]

#: Marks a parameter slot whose real value lives in the symbolic overlay.
_SYMBOLIC = float("nan")


def _as_float(value: Any, gate_name: str) -> float:
    """Coerce a numeric angle, rejecting foreign symbolic types clearly.

    Args:
        value: The angle to coerce.
        gate_name: Gate name, for the error message.

    Returns:
        The angle as a float.

    Raises:
        TypeError: If the value is neither numeric nor a SymPy expression.
            A framework's own parameter type lands here, which is deliberate:
            the IR is framework independent and only speaks SymPy.
    """
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"angle for gate '{gate_name}' must be a number or a SymPy expression built "
            f"from qc_executor Parameters, got {type(value).__name__}: {value!r}"
        ) from exc


@dataclass(frozen=True, slots=True)
class Condition:
    """An equality test on classical bits gating a single instruction.

    Args:
        clbits: The classical bits read, least significant first.
        value: The integer the bits must equal for the instruction to apply.
    """

    clbits: Tuple[int, ...]
    value: int


@dataclass(frozen=True, slots=True)
class Instruction:
    """A single instruction, materialised on demand from the packed arrays.

    Args:
        opcode: What the instruction does.
        qubits: Qubits acted on, in the order the gate defines.
        params: Angles, each a ``float`` or a SymPy expression.
        clbits: Classical bits written, for measurements.
        condition: Classical condition gating this instruction, if any.
    """

    opcode: OpCode
    qubits: Tuple[int, ...]
    params: Tuple[Any, ...] = ()
    clbits: Tuple[int, ...] = ()
    condition: "Condition | None" = None

    @property
    def definition(self) -> GateDef:
        """The static description of this instruction's opcode."""
        return GATE_DEFS[self.opcode]

    @property
    def name(self) -> str:
        """The canonical lowercase gate name."""
        return GATE_DEFS[self.opcode].name

    @property
    def is_parameterized(self) -> bool:
        """Whether any angle is symbolic rather than a plain number."""
        return any(isinstance(p, sp.Basic) for p in self.params)


class CircuitIR:
    """A packed, mutable sequence of quantum instructions.

    Args:
        num_qubits: Number of qubits the circuit spans.
        num_clbits: Number of classical bits available for measurement.
    """

    __slots__ = (
        "_num_qubits",
        "_num_clbits",
        "_opcodes",
        "_qubits",
        "_qubit_off",
        "_params",
        "_param_off",
        "_symbolic",
        "_clbits",
        "_conditions",
        "_revision",
        "_cache",
    )

    def __init__(self, num_qubits: int, num_clbits: int = 0):
        if num_qubits < 0:
            raise ValueError(f"num_qubits must be non-negative, got {num_qubits}")
        if num_clbits < 0:
            raise ValueError(f"num_clbits must be non-negative, got {num_clbits}")
        self._num_qubits = int(num_qubits)
        self._num_clbits = int(num_clbits)
        self._opcodes = array("H")
        self._qubits = array("I")
        self._qubit_off = array("I", [0])
        self._params = array("d")
        self._param_off = array("I", [0])
        self._symbolic: Dict[int, sp.Expr] = {}
        self._clbits: Dict[int, Tuple[int, ...]] = {}
        self._conditions: Dict[int, Condition] = {}
        self._revision = 0
        self._cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Size and identity
    # ------------------------------------------------------------------

    @property
    def num_qubits(self) -> int:
        """Number of qubits the circuit spans."""
        return self._num_qubits

    @property
    def num_clbits(self) -> int:
        """Number of classical bits available."""
        return self._num_clbits

    @property
    def revision(self) -> int:
        """Counter bumped on every mutation, for invalidating cached artifacts."""
        return self._revision

    def __len__(self) -> int:
        return len(self._opcodes)

    def _touch(self) -> None:
        """Record a mutation and drop derived caches."""
        self._revision += 1
        self._cache.clear()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def ensure_clbits(self, count: int) -> None:
        """Grow the classical register to hold at least ``count`` bits.

        Args:
            count: Required number of classical bits.
        """
        if count > self._num_clbits:
            self._num_clbits = count
            self._touch()

    def append(
        self,
        opcode: OpCode,
        qubits: Sequence[int],
        params: Sequence[Any] = (),
        clbits: Sequence[int] = (),
        condition: "Condition | None" = None,
    ) -> int:
        """Append one instruction.

        Args:
            opcode: The instruction kind.
            qubits: Qubit indices, in the order the gate defines.
            params: Angles, each a number or a SymPy expression.
            clbits: Classical bits written.
            condition: Classical condition gating this instruction.

        Returns:
            The index of the appended instruction.

        Raises:
            ValueError: If arity or indices do not match the gate definition.
        """
        definition = GATE_DEFS[opcode]
        qubit_tuple = tuple(int(q) for q in qubits)
        self._validate(definition, qubit_tuple, params, clbits)

        index = len(self._opcodes)
        self._opcodes.append(int(opcode))
        self._qubits.extend(qubit_tuple)
        self._qubit_off.append(len(self._qubits))

        for value in params:
            if isinstance(value, sp.Basic) and value.free_symbols:
                self._symbolic[len(self._params)] = canonicalize(value)
                self._params.append(_SYMBOLIC)
            else:
                self._params.append(_as_float(value, definition.name))
        self._param_off.append(len(self._params))

        if clbits:
            self._clbits[index] = tuple(int(c) for c in clbits)
        if condition is not None:
            self._conditions[index] = condition

        self._touch()
        return index

    def _validate(
        self,
        definition: GateDef,
        qubits: Tuple[int, ...],
        params: Sequence[Any],
        clbits: Sequence[int],
    ) -> None:
        """Check arity and index ranges for one instruction."""
        if definition.num_qubits != VARIABLE_QUBITS and len(qubits) != definition.num_qubits:
            raise ValueError(
                f"{definition.name} acts on {definition.num_qubits} qubit(s), "
                f"got {len(qubits)}"
            )
        if len(params) != definition.num_params:
            raise ValueError(
                f"{definition.name} takes {definition.num_params} parameter(s), "
                f"got {len(params)}"
            )
        for qubit in qubits:
            if not 0 <= qubit < self._num_qubits:
                raise ValueError(
                    f"qubit index {qubit} is out of range for a "
                    f"{self._num_qubits}-qubit circuit"
                )
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"{definition.name} received repeated qubit indices: {qubits}")
        for clbit in clbits:
            if not 0 <= clbit < self._num_clbits:
                raise ValueError(
                    f"classical bit index {clbit} is out of range for "
                    f"{self._num_clbits} classical bit(s)"
                )

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def _params_at(self, index: int) -> Tuple[Any, ...]:
        """Return the angles of instruction ``index``, symbolic slots resolved."""
        start, stop = self._param_off[index], self._param_off[index + 1]
        values: List[Any] = []
        for slot in range(start, stop):
            symbolic = self._symbolic.get(slot)
            values.append(symbolic if symbolic is not None else self._params[slot])
        return tuple(values)

    def _qubits_at(self, index: int) -> Tuple[int, ...]:
        """Return the qubits of instruction ``index``."""
        start, stop = self._qubit_off[index], self._qubit_off[index + 1]
        return tuple(self._qubits[start:stop])

    def __getitem__(self, index: int) -> Instruction:
        if not isinstance(index, int):
            raise TypeError(
                f"CircuitIR indices must be integers, not {type(index).__name__}; "
                "iterate the circuit to walk instructions"
            )
        if index < 0:
            index += len(self._opcodes)
        if not 0 <= index < len(self._opcodes):
            raise IndexError(f"instruction index {index} out of range")
        return Instruction(
            opcode=OpCode(self._opcodes[index]),
            qubits=self._qubits_at(index),
            params=self._params_at(index),
            clbits=self._clbits.get(index, ()),
            condition=self._conditions.get(index),
        )

    def __iter__(self) -> Iterator[Instruction]:
        for index in range(len(self._opcodes)):
            yield self[index]

    def iter_ops(self) -> Iterator[Tuple[int, Tuple[int, ...], Tuple[Any, ...]]]:
        """Iterate opcode, qubits and angles without building Instructions.

        Backends translating large circuits should prefer this over iterating
        the IR directly, which materialises a dataclass per gate.

        Yields:
            ``(opcode, qubits, params)`` for each instruction in order.
        """
        for index, opcode in enumerate(self._opcodes):
            yield opcode, self._qubits_at(index), self._params_at(index)

    def numeric_params(self) -> "np.ndarray | None":
        """Return all angles as one NumPy array, or ``None`` if any is symbolic.

        Returns:
            A read-only view of the packed angle column, letting backends apply
            vectorised transforms instead of per-gate Python arithmetic.
        """
        if self._symbolic:
            return None
        return np.frombuffer(self._params, dtype=np.float64)

    def count_ops(self) -> Dict[str, int]:
        """Return how often each gate name appears."""
        counts: Dict[str, int] = {}
        for opcode in self._opcodes:
            name = GATE_DEFS[OpCode(opcode)].name
            counts[name] = counts.get(name, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    @property
    def free_parameters(self) -> FrozenSet[Parameter]:
        """The parameters appearing anywhere in the circuit."""
        cached = self._cache.get("free_parameters")
        if cached is None:
            found: set = set()
            for expr in self._symbolic.values():
                found.update(s for s in expr.free_symbols if isinstance(s, Parameter))
            cached = frozenset(found)
            self._cache["free_parameters"] = cached
        return cached

    def substitute(self, binding: Mapping[Parameter, float]) -> "CircuitIR":
        """Return a copy with parameter values substituted.

        Angles that become fully numeric move into the packed column; angles
        still carrying free symbols stay in the overlay.

        Args:
            binding: Values to substitute.

        Returns:
            A new circuit; the original is untouched.
        """
        # pylint: disable=protected-access  # same-class access
        clone = self.copy()
        if not binding:
            return clone
        replacements = dict(binding)
        for slot, expr in list(clone._symbolic.items()):
            result = expr.xreplace(replacements)
            # xreplace on a bare Symbol returns the replacement object itself,
            # which is a plain Python float rather than a SymPy number.
            if isinstance(result, sp.Basic) and result.free_symbols:
                clone._symbolic[slot] = result
            else:
                del clone._symbolic[slot]
                clone._params[slot] = float(result)
        clone._touch()
        return clone

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    def copy(self) -> "CircuitIR":
        """Return an independent copy of this circuit."""
        # pylint: disable=protected-access  # same-class access
        clone = CircuitIR(self._num_qubits, self._num_clbits)
        clone._opcodes = array("H", self._opcodes)
        clone._qubits = array("I", self._qubits)
        clone._qubit_off = array("I", self._qubit_off)
        clone._params = array("d", self._params)
        clone._param_off = array("I", self._param_off)
        clone._symbolic = dict(self._symbolic)
        clone._clbits = dict(self._clbits)
        clone._conditions = dict(self._conditions)
        return clone

    def extend(
        self,
        other: "CircuitIR",
        qubit_map: "Sequence[int] | None" = None,
        clbit_map: "Sequence[int] | None" = None,
    ) -> None:
        """Append every instruction of ``other``, remapping its bits.

        Args:
            other: The circuit to append.
            qubit_map: ``qubit_map[i]`` is where ``other``'s qubit ``i`` lands.
                Defaults to the identity.
            clbit_map: The same for classical bits.

        Raises:
            ValueError: If a map is too short for the circuit being appended.
        """
        if qubit_map is None:
            qubit_map = range(other.num_qubits)
        if len(qubit_map) != other.num_qubits:
            raise ValueError(
                f"Length of qubits mapping must match the appended circuit: got "
                f"{len(qubit_map)} entries for {other.num_qubits} qubit(s)"
            )
        if clbit_map is None:
            clbit_map = range(other.num_clbits)

        for instruction in other:
            condition = instruction.condition
            if condition is not None:
                condition = Condition(
                    clbits=tuple(clbit_map[c] for c in condition.clbits),
                    value=condition.value,
                )
            self.append(
                instruction.opcode,
                tuple(qubit_map[q] for q in instruction.qubits),
                instruction.params,
                tuple(clbit_map[c] for c in instruction.clbits),
                condition,
            )

    def inverse(self) -> "CircuitIR":
        """Return the adjoint circuit.

        Instructions are reversed and each is replaced by its adjoint, using the
        gate table: self-inverse gates stay, gates with a named adjoint opcode
        swap, and rotations negate their angles.

        Returns:
            A new circuit implementing the adjoint.

        Raises:
            NotImplementedError: If any instruction has no adjoint, such as a
                measurement or a gate whose adjoint is not in the gate set.
        """
        inverted = CircuitIR(self._num_qubits, self._num_clbits)
        for index in range(len(self._opcodes) - 1, -1, -1):
            instruction = self[index]
            opcode, params = _invert_instruction(instruction)
            inverted.append(opcode, instruction.qubits, params)
        return inverted

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    def fingerprint(self) -> bytes:
        """Return a stable digest of the circuit's full content.

        Hashing the packed buffers directly avoids the per-gate, per-bit Python
        walk a framework circuit would need, which matters because executors
        fingerprint circuits on every cached call.

        Returns:
            A 32-byte digest covering structure, angles, clbits and conditions.
        """
        cached = self._cache.get("fingerprint")
        if cached is not None:
            return cached

        digest = hashlib.blake2b(digest_size=32)
        digest.update(self._num_qubits.to_bytes(4, "little"))
        digest.update(self._num_clbits.to_bytes(4, "little"))
        digest.update(self._opcodes.tobytes())
        digest.update(self._qubits.tobytes())
        digest.update(self._qubit_off.tobytes())
        digest.update(self._params.tobytes())
        digest.update(self._param_off.tobytes())
        # srepr is unambiguous and round-trippable; str() is neither for our
        # bracketed parameter names.
        digest.update(
            repr(sorted((slot, sp.srepr(expr)) for slot, expr in self._symbolic.items())).encode()
        )
        digest.update(repr(sorted(self._clbits.items())).encode())
        digest.update(
            repr(sorted((i, c.clbits, c.value) for i, c in self._conditions.items())).encode()
        )

        result = digest.digest()
        self._cache["fingerprint"] = result
        return result

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, CircuitIR) and self.fingerprint() == other.fingerprint()

    def __hash__(self) -> int:
        return hash(self.fingerprint())

    def __repr__(self) -> str:
        return (
            f"CircuitIR(num_qubits={self._num_qubits}, num_clbits={self._num_clbits}, "
            f"instructions={len(self._opcodes)})"
        )


#: Angle reordering for gates whose adjoint permutes parameters rather than
#: simply negating them.  ``U(theta, phi, lam)`` inverts to ``U(-theta, -lam, -phi)``.
_INVERSE_PARAM_ORDER: Dict[OpCode, Tuple[int, ...]] = {OpCode.U: (0, 2, 1)}


def _invert_instruction(instruction: Instruction) -> Tuple[OpCode, Tuple[Any, ...]]:
    """Return the opcode and angles implementing an instruction's adjoint."""
    definition = instruction.definition
    if instruction.opcode is OpCode.BARRIER:
        return instruction.opcode, ()
    if definition.self_inverse:
        return instruction.opcode, instruction.params
    if definition.inverse is not None:
        return definition.inverse, instruction.params
    if definition.is_parameterized:
        order = _INVERSE_PARAM_ORDER.get(instruction.opcode)
        params = instruction.params
        if order is not None:
            params = tuple(params[i] for i in order)
        return instruction.opcode, tuple(-p for p in params)
    raise NotImplementedError(
        f"'{definition.name}' has no adjoint in the gate set; "
        "decompose the circuit to invertible primitives first"
    )


def is_symbolic_slot(value: float) -> bool:
    """Return whether a packed angle marks a symbolic overlay entry.

    Args:
        value: A value read from the packed angle column.

    Returns:
        ``True`` when the slot's real value lives in the symbolic overlay.
    """
    return math.isnan(value)


def iter_parameters(values: Iterable[Any]) -> Iterator[Parameter]:
    """Yield the parameters appearing in a sequence of angles.

    Args:
        values: Angles, each a number or a SymPy expression.

    Yields:
        Each :class:`~qc_executor.parameters.Parameter` encountered.
    """
    for value in values:
        if isinstance(value, sp.Basic):
            for symbol in value.free_symbols:
                if isinstance(symbol, Parameter):
                    yield symbol
