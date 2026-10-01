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
from typing import Any, Dict, FrozenSet, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import sympy as sp

from ..parameters import Parameter, sort_parameters, translate_expression
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

        The instruction store is copied verbatim; lowering into
        :meth:`supported_opcodes` happens in :meth:`_build_native`, so ``ir``
        means "the circuit as written" on every class.  The copy matters
        because the lowering pass returns its input unchanged when nothing
        needs rewriting -- sharing the store would let a gate appended to the
        converted circuit appear in the original.

        Args:
            circuit: The circuit to convert.

        Returns:
            ``circuit`` unchanged if it is already of this type, else a new
            instance holding a copy of its instructions.
        """
        if isinstance(circuit, cls):
            return circuit
        return cls(circuit.num_qubits, circuit.num_clbits, _ir=circuit.ir.copy())

    def _lowered_ir(self) -> CircuitIR:
        """Return the instruction store rewritten into the supported gate set.

        Backends call this at the top of :meth:`_build_native`.

        Returns:
            The lowered store, or the original when nothing needed rewriting.
        """
        # Imported here because the decomposition pass builds on this class.
        from .decompose import decompose_ir  # pylint: disable=import-outside-toplevel

        return decompose_ir(self._ir, self.supported_opcodes())

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

    @property
    def parameter_vector_names(self) -> List[str]:
        """The distinct parameter-vector names, in :attr:`parameters` order.

        Supplying all of them as ``derivative`` arguments to
        :meth:`~qc_executor.base.executor_base.ExecutorBase.expectation_value_derivatives`
        yields the full gradient of the circuit.
        """
        names: List[str] = []
        for parameter in self.parameters:
            if parameter.vector_name not in names:
                names.append(parameter.vector_name)
        return names

    #: Names of the unitary gate-appending methods every circuit offers.  Kept
    #: by hand because six of them (``i``, ``sdag``, ``sxdag``, ``tdag`` and the
    #: aliases ``cnot``/``toffoli``) have no opcode of the same name in
    #: :data:`GATE_DEFS`; ``tests/base/test_circuit_base.py`` checks the list
    #: against the class and the gate table so it cannot drift silently.
    _GATE_METHODS = frozenset(
        {
            "i", "h", "x", "y", "z", "s", "sdag", "t", "tdag", "sx", "sxdag",
            "rx", "ry", "rz", "p", "u",
            "cx", "cnot", "cy", "cz", "ch", "cs", "csx", "ecr", "swap", "iswap",
            "cp", "crx", "cry", "crz", "cu", "rxx", "ryy", "rzz", "rzx",
            "ccx", "toffoli", "cswap",
        }
    )  # fmt: skip

    @classmethod
    def available_gates(cls) -> FrozenSet[str]:
        """Return the names of the gate methods defined on this circuit class."""
        return cls._GATE_METHODS

    def _replace_ir(self, ir: CircuitIR) -> None:
        """Adopt a new instruction store, dropping any compiled native circuit.

        The native cache is keyed on the store's revision counter, which a
        fresh store restarts, so the counter alone cannot tell the two apart.
        """
        self._ir = ir
        self._native_cache = None
        self._native_revision = -1

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

    def cu(
        self, control_qubit: int, target_qubit: int, theta: Any, phi: Any, lam: Any, gamma: Any
    ) -> None:
        """Add a controlled U gate with a global phase ``gamma`` on the target.

        Same convention as Qiskit's ``CUGate``.  The IR has no opcode for it, so
        the gate is written out as its standard decomposition into ``p``, ``cx``
        and ``u``, which every backend already runs.
        """
        self.p(control_qubit, gamma + (lam + phi) / 2)
        self.p(target_qubit, (lam - phi) / 2)
        self.cx(control_qubit, target_qubit)
        self.u(target_qubit, -theta / 2, 0, -(phi + lam) / 2)
        self.cx(control_qubit, target_qubit)
        self.u(target_qubit, theta / 2, phi, 0)

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
            pauli_string: One character per qubit, qubit 0 leftmost.

        Raises:
            ValueError: If the string length does not match the qubit count,
                or a character is not one of ``I``, ``X``, ``Y``, ``Z``.
        """
        if len(pauli_string) != self.num_qubits:
            raise ValueError("Pauli string length does not match number of qubits")

        for qubit, pauli in enumerate(pauli_string):
            if pauli == "X":
                self.x(qubit)
            elif pauli == "Y":
                self.y(qubit)
            elif pauli == "Z":
                self.z(qubit)
            elif pauli != "I":
                raise ValueError(f"Unknown Pauli operator: {pauli}")

    def _apply_basis_change(self, paulis: List[str], qubits: List[int]) -> None:
        """Rotate each non-identity Pauli into the Z basis."""
        for pauli, qubit in zip(paulis, qubits):
            if pauli == "X":
                self.h(qubit)
            elif pauli == "Y":
                self.sdag(qubit)
                self.h(qubit)
            elif pauli != "Z":
                raise ValueError(f"Unknown Pauli operator: {pauli}")

    def _undo_basis_change(self, paulis: List[str], qubits: List[int]) -> None:
        """Undo :meth:`_apply_basis_change`."""
        for pauli, qubit in zip(paulis, qubits):
            if pauli == "X":
                self.h(qubit)
            elif pauli == "Y":
                self.h(qubit)
                self.s(qubit)

    def _apply_cnot_ladder(self, qubits: List[int]) -> None:
        """Apply the forward CNOT ladder for Pauli evolution."""
        if not qubits:
            return
        control = qubits[0]
        for target in qubits[1:]:
            self.cx(control, target)
            control = target

    def _undo_cnot_ladder(self, qubits: List[int]) -> None:
        """Undo the CNOT ladder after the phase rotation."""
        if not qubits:
            return
        control = qubits[-1]
        for target in reversed(qubits[:-1]):
            self.cx(target, control)
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

    @staticmethod
    def _single_pauli_term(operator: Any) -> Tuple[str, Any]:
        """Return the label and coefficient of a single-term operator.

        Accepts any :class:`~qc_executor.base.operator_base.QuantumOperatorBase`
        or anything exposing ``paulis`` (labels, qubit 0 leftmost) and
        ``coeffs`` the same way.

        Args:
            operator: The operator to inspect.

        Returns:
            ``(label, coefficient)``.

        Raises:
            TypeError: If ``operator`` does not expose Pauli labels and coefficients.
            ValueError: If the operator holds more than one Pauli string.
        """
        paulis = getattr(operator, "paulis", None)
        coeffs = getattr(operator, "coeffs", None)
        if paulis is None or coeffs is None:
            raise TypeError(f"Expected a quantum operator, got {type(operator).__name__}")
        labels, coeffs = list(paulis), list(coeffs)
        if len(coeffs) != 1 or len(labels) != 1:
            raise ValueError("Only operators with single Pauli strings are supported")
        return str(labels[0]), coeffs[0]

    def pauli_evolution(
        self,
        operator: Any,
        parameter: Any,
        working_qubits: "Sequence[int] | None" = None,
    ) -> None:
        """Apply the Pauli evolution ``exp(-i t P)`` for a single Pauli string.

        Args:
            operator: An operator holding exactly one Pauli string, a
                :class:`~qc_executor.base.operator_base.QuantumOperatorBase`.
            parameter: The evolution parameter ``t``, numeric or symbolic
                (Qiskit parameter expressions are accepted).
            working_qubits: Circuit qubit for each label position, defaulting
                to ``0..n-1``.

        Raises:
            ValueError: If the operator holds more than one Pauli string.
        """
        self.controlled_pauli_evolution(operator, parameter, working_qubits=working_qubits)

    def controlled_pauli_evolution(
        self,
        operator: Any,
        parameter: Any,
        working_qubits: "Sequence[Sequence[int]] | Sequence[int] | None" = None,
        control_qubits: "Sequence[int | None] | int | None" = None,
        control_state: "Sequence[str | None] | str | None" = None,
    ) -> None:
        """Apply one or several (optionally controlled) Pauli evolutions.

        Each operator ``P`` with coefficient ``c`` contributes ``exp(-i c t P)``.
        Several operators are applied together on disjoint working qubits,
        sharing one basis change and one CNOT-ladder layer.

        Args:
            operator: A single-term operator (see :meth:`pauli_evolution`) or a
                list of them.
            parameter: The evolution parameter, or one per operator.
            working_qubits: Circuit qubit for each label position, per
                operator.  By default label position ``i`` maps to the
                ``i``-th qubit that is neither a control nor already used.
            control_qubits: One control qubit per operator, or ``None`` for an
                uncontrolled evolution.
            control_state: ``"0"`` or ``"1"`` per operator: the control value
                that triggers the evolution.  Defaults to ``"1"``.

        Raises:
            TypeError: If ``operator`` is neither an operator nor a list.
            ValueError: If an operator has several terms or a complex
                coefficient, if a control state is invalid, or if the qubit
                assignment is inconsistent or out of range.
            NotImplementedError: If more than one control qubit is requested.
        """
        # -- Normalise every argument to one entry per operator --
        if isinstance(operator, (list, tuple)):
            operators = list(operator)
        elif hasattr(operator, "paulis"):
            operators = [operator]
        else:
            raise TypeError("Operator must be a quantum operator or a list thereof")
        count = len(operators)

        parameters = (
            list(parameter) if isinstance(parameter, (list, tuple)) else [parameter] * count
        )
        parameters = [translate_expression(value) for value in parameters]

        if working_qubits is None:
            working_lists: list = [None] * count
        elif isinstance(working_qubits, (int, np.integer)):
            working_lists = [[int(working_qubits)]] * count
        elif len(working_qubits) > 0 and isinstance(working_qubits[0], (list, tuple, range)):
            working_lists = [list(entry) for entry in working_qubits]
        else:
            working_lists = [list(working_qubits)] * count

        if control_qubits is None:
            control_list: list = [None] * count
        elif isinstance(control_qubits, (int, np.integer)):
            control_list = [int(control_qubits)] * count
        else:
            control_list = list(control_qubits)

        if control_state is None:
            state_list: list = [None] * count
        elif isinstance(control_state, str):
            state_list = [control_state] * count
        else:
            state_list = list(control_state)

        for name, entries in (
            ("parameter", parameters),
            ("working_qubits", working_lists),
            ("control_qubits", control_list),
            ("control_state", state_list),
        ):
            if len(entries) != count:
                raise ValueError(f"{name} must have one entry per operator, got {len(entries)}")
        for state in state_list:
            if state not in (None, "0", "1"):
                raise ValueError(f'control_state entries must be "0" or "1", got {state!r}')

        # -- Per-operator preprocessing --
        angles = []
        paulis_per_operator = []
        positions_per_operator = []
        for op, value in zip(operators, parameters):
            label, coeff = self._single_pauli_term(op)
            for pauli in label:
                if pauli not in "IXYZ":
                    raise ValueError(f"Unknown Pauli operator: {pauli}")
            angles.append(self._evolution_angle(coeff, value, scale=1.0))
            positions_per_operator.append([i for i, pauli in enumerate(label) if pauli != "I"])
            paulis_per_operator.append([pauli for pauli in label if pauli != "I"])

        # -- Resolve control and working qubits --
        free_qubits = set(range(self.num_qubits))
        controls: list = []
        for control in control_list:
            if control is not None:
                control = int(control)
                if control not in free_qubits and control not in controls:
                    raise ValueError(f"Control qubit {control} is out of range")
                free_qubits.discard(control)
            controls.append(control)

        used: set = set()
        working_per_operator = []
        for positions, working in zip(positions_per_operator, working_lists):
            if working is None:
                candidates = sorted(free_qubits)
                if max(positions, default=-1) >= len(candidates):
                    raise ValueError("Not enough qubits left for implementing pauli evolution!")
                qubits = [candidates[i] for i in positions]
            else:
                if max(positions, default=-1) >= len(working):
                    raise ValueError("working_qubits has fewer entries than the Pauli string")
                qubits = [int(working[i]) for i in positions]
                if any(q < 0 or q >= self.num_qubits for q in qubits):
                    raise ValueError("Working qubits are out of range for this circuit!")
            if any(q in used for q in qubits):
                raise ValueError("No distinct support qubits between the operators!")
            if any(q in controls for q in qubits):
                raise ValueError("Controlled qubits must be distinct from working qubits!")
            used.update(qubits)
            free_qubits.difference_update(qubits)
            working_per_operator.append(qubits)

        # -- Emit the gates --
        for paulis, qubits in zip(paulis_per_operator, working_per_operator):
            self._apply_basis_change(paulis, qubits)
        for qubits in working_per_operator:
            self._apply_cnot_ladder(qubits)

        for qubits, angle, control, state in zip(
            working_per_operator, angles, controls, state_list
        ):
            if control is None:
                if qubits:
                    self.rz(qubits[-1], 2 * angle)
                # An identity term without control is a global phase: nothing to do.
                continue
            if state == "0":
                self.x(control)
            if qubits:
                self.crz(control, qubits[-1], 2 * angle)
            else:
                # A controlled global phase is a Z rotation on the control.
                self.rz(control, -angle)
            if state == "0":
                self.x(control)

        for qubits in working_per_operator:
            self._undo_cnot_ladder(qubits)
        for paulis, qubits in zip(paulis_per_operator, working_per_operator):
            self._undo_basis_change(paulis, qubits)

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    def compose(
        self,
        qc: "QuantumCircuitBase",
        qubits: "Sequence[int] | None" = None,
        clbits: "Sequence[int] | None" = None,
        new_parameters: bool = True,
    ) -> "QuantumCircuitBase":
        """Append another circuit's instructions onto this one, in place.

        When the two circuits share a parameter name, their parameters are
        re-indexed into a single vector named after this circuit's first
        parameter so that repeatedly composing circuits that all use
        ``theta[0]`` never collides: this circuit's parameters keep their
        positions and the parameters of ``qc`` are appended after them (or
        merged positionally for ``new_parameters=False``).  Circuits whose
        parameter names are disjoint -- features ``x`` and weights ``p``, say --
        are composed as they are, so every name keeps its meaning.

        Args:
            qc: The circuit to append.
            qubits: Where ``qc``'s qubits land, defaulting to the identity,
                which requires equal qubit counts.
            clbits: Where ``qc``'s classical bits land.
            new_parameters: Only relevant when a parameter name is shared.
                If True (default), the parameters of ``qc`` are appended after
                the parameters of this circuit.  If False, the parameters of
                both circuits are merged positionally.

        Returns:
            This circuit, to allow chaining.

        Raises:
            TypeError: If ``qc`` is not a quantum circuit.
            ValueError: If the qubit mapping is invalid.
        """
        if not isinstance(qc, QuantumCircuitBase):
            raise TypeError(f"can only compose with a quantum circuit, got {type(qc).__name__}")
        if qubits is None:
            if self.num_qubits != qc.num_qubits:
                raise ValueError(
                    "When qubits=None, both circuits must have the same number of qubits "
                    f"(got self.num_qubits={self.num_qubits}, qc.num_qubits={qc.num_qubits})."
                )
            qubits = list(range(qc.num_qubits))
        qubits = [int(q) for q in qubits]
        if len(qubits) != qc.num_qubits:
            raise ValueError(
                "Length of qubits mapping must match the composed circuit qubit count "
                f"(got len(qubits)={len(qubits)}, qc.num_qubits={qc.num_qubits})."
            )
        if any(q < 0 or q >= self.num_qubits for q in qubits):
            raise ValueError("Qubit mapping contains indexes out of range for the target circuit.")
        if len(set(qubits)) != len(qubits):
            raise ValueError("Qubit mapping contains duplicate indices.")

        # Merge first: renaming this circuit's own parameters swaps in a new
        # instruction store, and the appended instructions must land in it.
        merged = self._merged_parameter_ir(qc, new_parameters)
        self._ir.extend(merged, qubits, clbits)
        return self

    def _merged_parameter_ir(self, qc: "QuantumCircuitBase", new_parameters: bool) -> CircuitIR:
        """Re-index both circuits' parameters into one vector before composing.

        Renames this circuit's parameters in place and returns ``qc``'s
        instruction store with its parameters renamed.  Circuits that share no
        parameter name -- including those where only one side is parameterised
        -- are left untouched: renaming them would only destroy names that
        callers bind by, such as a feature vector ``x`` composed onto weights
        ``p``.
        """
        own, other = self.parameters, qc.parameters
        if not set(own) & set(other):
            return qc.ir
        name = own[0].vector_name
        offset = len(own) if new_parameters else 0
        own_binding = {p: Parameter(name, i) for i, p in enumerate(own) if p != Parameter(name, i)}
        other_binding = {
            p: Parameter(name, offset + i)
            for i, p in enumerate(other)
            if p != Parameter(name, offset + i)
        }
        if own_binding:
            self._replace_ir(self._ir.substitute(own_binding))
        return qc.ir.substitute(other_binding) if other_binding else qc.ir

    def fixate_parameters(self, parameters: Sequence[float]) -> None:
        """Bind every free parameter in place, leaving a numeric circuit.

        Args:
            parameters: One value per parameter, in :attr:`parameters` order.

        Raises:
            ValueError: If the number of values does not match.
        """
        values = np.asarray(parameters, dtype=float).reshape(-1)
        free = self.parameters
        if len(values) != len(free):
            raise ValueError(f"Expected {len(free)} parameter values, got {len(values)}")
        self._replace_ir(self._ir.substitute(dict(zip(free, (float(v) for v in values)))))

    def assign_parameters(self, parameters: Mapping[Any, float]) -> "QuantumCircuitBase":
        """Return a copy of this circuit with parameter values substituted.

        This is pure: the receiver is unchanged.  It used to bind in place and
        return ``self``, which made the bound circuit and the original the same
        object -- that silently zeroed the Pauli-propagation gradients, because
        the parameter-shift rule compares two bindings of one circuit.

        Args:
            parameters: Values keyed by :class:`~qc_executor.parameters.Parameter`
                or by parameter name.

        Returns:
            A new circuit with the given parameters bound.
        """
        binding = {
            (key if isinstance(key, Parameter) else Parameter(str(key))): value
            for key, value in parameters.items()
        }
        return type(self)(self.num_qubits, self.num_clbits, _ir=self._ir.substitute(binding))

    def invert(self) -> "QuantumCircuitBase":
        """Return the adjoint of this circuit."""
        return type(self)(self.num_qubits, self.num_clbits, _ir=self._ir.inverse())

    def copy(self) -> "QuantumCircuitBase":
        """Return an independent copy of this circuit."""
        return type(self)(self.num_qubits, self.num_clbits, _ir=self._ir.copy())

    def clear(self) -> None:
        """Remove every instruction in place, keeping the qubit and clbit counts.

        Lets a caller rewrite a circuit it holds a reference to -- for example
        after simplifying its gate sequence -- without replacing the object.
        """
        self._replace_ir(CircuitIR(self.num_qubits, self.num_clbits))

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
        # Same type, not just any circuit: a QulacsCircuit and a generic
        # QuantumCircuit holding identical instructions are not
        # interchangeable, and the executors key their conversion caches on
        # the generic circuit, so cross-type equality would let a converted
        # circuit collide with the one it was converted from.
        return type(self) is type(other) and self._ir == other.ir

    def __str__(self) -> str:
        return self.draw()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(num_qubits={self.num_qubits}, "
            f"num_clbits={self.num_clbits}, instructions={len(self._ir)})"
        )
