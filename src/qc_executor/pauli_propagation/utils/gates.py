"""Internal gate representations for Pauli propagation.

Gates are stored in a form optimized for Heisenberg picture propagation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import sympy as sp

from .pauli_algebra import commutes as pauli_commutes
from .pauli_algebra import get_pauli, int_to_symbol, low_mask, set_pauli, symbol_to_int


class Gate(ABC):
    """Abstract base class for quantum gates."""

    def __init__(self, qubits: int | List[int], nqubits: int):
        """Initialize a gate.

        Args:
            qubits: Qubit index or list of qubit indices
            nqubits: Total number of qubits in the circuit
        """
        if isinstance(qubits, int):
            self.qubits = [qubits]
        else:
            self.qubits = list(qubits)
        self.nqubits = nqubits

    @abstractmethod
    def commutes_with(self, pauli_term: int) -> bool:
        """Check if this gate commutes with a Pauli term.

        Args:
            pauli_term: Bit-encoded Pauli string

        Returns:
            True if commutes, False if anticommutes
        """

    @abstractmethod
    def is_parametric(self) -> bool:
        """Return True if gate has a parameter (like rotation angle)."""


class PauliRotation(Gate):
    """Parametric Pauli rotation gate: exp(-i θ/2 P)

    Where P is a Pauli string. Examples:
    - RX(θ) = exp(-i θ/2 X)
    - RY(θ) = exp(-i θ/2 Y)
    - RZ(θ) = exp(-i θ/2 Z)
    - RXX(θ) = exp(-i θ/2 XX)
    """

    def __init__(
        self,
        symbols: List[str],
        qubits: int | List[int],
        nqubits: int,
        param_name: str | None = None,
        param_expr: sp.Expr | None = None,
        param_value: float | None = None,
    ):
        """Initialize a Pauli rotation gate.

        Args:
            symbols: List of Pauli symbols defining the rotation axis (e.g., ['X'], ['X', 'X'])
            qubits: Qubit index or list of indices where Paulis act
            nqubits: Total number of qubits
            param_name: Optional parameter name for compatibility
            param_expr: Optional sympy expression for parametric gates
            param_value: Optional concrete parameter value (for non-parametric gates)
        """
        super().__init__(qubits, nqubits)

        if len(symbols) != len(self.qubits):
            raise ValueError(
                f"Number of symbols ({len(symbols)}) must match "
                f"number of qubits ({len(self.qubits)})"
            )

        self.symbols = symbols
        if param_expr is None and param_name is not None:
            param_expr = sp.Symbol(param_name)
        self.param_expr = param_expr
        if param_name is not None:
            self.param_name = param_name
        elif isinstance(self.param_expr, sp.Symbol):
            self.param_name = self.param_expr.name
        else:
            self.param_name = None
        self.param_value = param_value

        # Build the Pauli generator term (what we're rotating around)
        self.generator_term = 0
        for symbol, qubit in zip(symbols, self.qubits):
            pauli_int = symbol_to_int(symbol)
            self.generator_term = set_pauli(self.generator_term, qubit, pauli_int, nqubits)

        # Precomputed symplectic components of the generator for the
        # propagation hot loop (see pauli_algebra module docstring):
        # generator_z/generator_x are the per-qubit Z/X component words and
        # generator_phase_count is popcount(x & z), the generator's constant
        # contribution to the Pauli-product phase exponent.
        mask = low_mask(nqubits)
        gen = self.generator_term
        self.generator_z = (gen >> 1) & mask
        self.generator_x = (gen ^ (gen >> 1)) & mask
        self.generator_phase_count = (self.generator_x & self.generator_z).bit_count()

    def commutes_with(self, pauli_term: int) -> bool:
        """Check if this rotation commutes with a Pauli term.

        A Pauli rotation exp(-i θ/2 P) commutes with Q if [P, Q] = 0.

        Args:
            pauli_term: Bit-encoded Pauli string

        Returns:
            True if commutes
        """
        return pauli_commutes(self.generator_term, pauli_term, self.nqubits)

    def is_parametric(self) -> bool:
        """Pauli rotations are parametric."""
        return True

    def __repr__(self) -> str:
        """String representation."""
        pauli_str = "".join(self.symbols)
        qubits_str = str(self.qubits) if len(self.qubits) > 1 else str(self.qubits[0])
        return f"PauliRotation({pauli_str}, qubits={qubits_str})"


class CliffordGate(Gate):
    """Clifford gate with explicit Pauli transformation rules.

    Clifford gates map Paulis to Paulis (up to phase). We store the transformation
    rules explicitly for efficient propagation.

    Supported gates: H, S, T, X, Y, Z, CNOT, CZ, SWAP
    """

    # Transformation rules for single-qubit Clifford gates in the
    # Heisenberg picture: input_pauli P → U† P U
    # Format: gate_name → {input_pauli → (output_pauli, phase)}
    SINGLE_QUBIT_RULES = {
        "H": {  # Hadamard: X ↔ Z, Y → -Y
            "I": ("I", 1.0),
            "X": ("Z", 1.0),
            "Y": ("Y", -1.0),
            "Z": ("X", 1.0),
        },
        "S": {  # Phase gate: S†XS = -Y, S†YS = X, Z → Z
            "I": ("I", 1.0),
            "X": ("Y", -1.0),
            "Y": ("X", 1.0),
            "Z": ("Z", 1.0),
        },
        "T": {  # T gate: NOT Clifford (T†XT = (X+Y)/√2 is no single Pauli).
            # Legacy approximate rule kept only for direct CliffordGate("T")
            # construction; the circuit builder and Qiskit converter now
            # produce the exact PauliRotation(Z, π/4) instead.
            "I": ("I", 1.0),
            "X": ("X", np.exp(-1j * np.pi / 4)),  # phase factor
            "Y": ("Y", np.exp(-1j * np.pi / 4)),
            "Z": ("Z", 1.0),
        },
        "X": {  # Pauli-X: Z → -Z, Y → -Y
            "I": ("I", 1.0),
            "X": ("X", 1.0),
            "Y": ("Y", -1.0),
            "Z": ("Z", -1.0),
        },
        "Y": {  # Pauli-Y: X → -X, Z → -Z
            "I": ("I", 1.0),
            "X": ("X", -1.0),
            "Y": ("Y", 1.0),
            "Z": ("Z", -1.0),
        },
        "Z": {  # Pauli-Z: X → -X, Y → -Y
            "I": ("I", 1.0),
            "X": ("X", -1.0),
            "Y": ("Y", -1.0),
            "Z": ("Z", 1.0),
        },
    }

    # Heisenberg transformation rules for two-qubit Clifford gates:
    # (p_a, p_b) → (p_a', p_b', phase) with Paulis encoded I=0, X=1, Y=2, Z=3.
    # Derived from the generator maps and verified against exact matrix
    # conjugation U† (P_a ⊗ P_b) U (see tests).
    # CNOT (a=control, b=target): X_c → X_c X_t, Z_c → Z_c, X_t → X_t,
    # Z_t → Z_c Z_t; CZ: X_i → X_i Z_j, Z_i → Z_i.
    TWO_QUBIT_RULES = {
        "CNOT": {
            (0, 0): (0, 0, 1.0),
            (0, 1): (0, 1, 1.0),
            (0, 2): (3, 2, 1.0),
            (0, 3): (3, 3, 1.0),
            (1, 0): (1, 1, 1.0),
            (1, 1): (1, 0, 1.0),
            (1, 2): (2, 3, 1.0),
            (1, 3): (2, 2, -1.0),
            (2, 0): (2, 1, 1.0),
            (2, 1): (2, 0, 1.0),
            (2, 2): (1, 3, -1.0),
            (2, 3): (1, 2, 1.0),
            (3, 0): (3, 0, 1.0),
            (3, 1): (3, 1, 1.0),
            (3, 2): (0, 2, 1.0),
            (3, 3): (0, 3, 1.0),
        },
        "CZ": {
            (0, 0): (0, 0, 1.0),
            (0, 1): (3, 1, 1.0),
            (0, 2): (3, 2, 1.0),
            (0, 3): (0, 3, 1.0),
            (1, 0): (1, 3, 1.0),
            (1, 1): (2, 2, 1.0),
            (1, 2): (2, 1, -1.0),
            (1, 3): (1, 0, 1.0),
            (2, 0): (2, 3, 1.0),
            (2, 1): (1, 2, -1.0),
            (2, 2): (1, 1, 1.0),
            (2, 3): (2, 0, 1.0),
            (3, 0): (3, 0, 1.0),
            (3, 1): (0, 1, 1.0),
            (3, 2): (0, 2, 1.0),
            (3, 3): (3, 3, 1.0),
        },
    }

    def __init__(self, gate_type: str, qubits: int | List[int], nqubits: int):
        """Initialize a Clifford gate.

        Args:
            gate_type: Type of gate ('H', 'S', 'CNOT', 'CZ', 'SWAP', etc.)
            qubits: Qubit index or list of indices
            nqubits: Total number of qubits
        """
        super().__init__(qubits, nqubits)
        self.gate_type = gate_type.upper()

        # Validate gate type and qubit count
        if self.gate_type in ["H", "S", "T", "X", "Y", "Z"]:
            if len(self.qubits) != 1:
                raise ValueError(f"{self.gate_type} gate requires exactly 1 qubit")
        elif self.gate_type in ["CNOT", "CX", "CZ", "SWAP"]:
            if len(self.qubits) != 2:
                raise ValueError(f"{self.gate_type} gate requires exactly 2 qubits")
        else:
            raise ValueError(f"Unknown Clifford gate type: {self.gate_type}")

        # Lazily built lookup table for transform_pauli_term:
        # (clear_mask, shifts, replacement_bits, phases). Built on first use
        # from the per-case transform methods so behavior stays identical.
        self._transform_table = None
        # Numpy view of the same table for the vectorized array engine
        self._transform_table_np = None

    def commutes_with(self, pauli_term: int) -> bool:
        """Clifford gates generally don't commute with arbitrary Paulis.

        We could compute this, but for propagation we apply the transformation regardless.
        Return False conservatively.

        Args:
            pauli_term: Bit-encoded Pauli string

        Returns:
            False (conservative)
        """
        return False

    def is_parametric(self) -> bool:
        """Clifford gates are not parametric."""
        return False

    def transform_pauli_term(self, pauli_term: int):
        """Transform a Pauli term through this Clifford gate.

        Uses the Heisenberg picture: U† P U

        Uses a lookup table over the gate's own qubit slots (4 entries for
        single-qubit gates, 16 for two-qubit gates), generated once from the
        per-case transform methods.

        Args:
            pauli_term: Bit-encoded Pauli term

        Returns:
            Tuple of (transformed_term, phase)
        """
        table = self._transform_table
        if table is None:
            table = self._build_transform_table()

        clear_mask, shifts, replacement_bits, phases = table
        pauli_term = int(pauli_term)
        if len(shifts) == 1:
            idx = (pauli_term >> shifts[0]) & 0b11
        else:
            idx = ((pauli_term >> shifts[0]) & 0b11) | (((pauli_term >> shifts[1]) & 0b11) << 2)

        return (pauli_term & clear_mask) | replacement_bits[idx], phases[idx]

    def _build_transform_table(self):
        """Build the transform lookup table from the per-case methods.

        The per-case methods only read and write the gate's own qubit slots,
        so probing them with every local Pauli combination captures the full
        transformation (behavior-preserving by construction).
        """
        if self.gate_type in ["H", "S", "T", "X", "Y", "Z"]:
            transform = self._transform_single_qubit
            shifts = (2 * self.qubits[0],)
            local_terms = [p << shifts[0] for p in range(4)]
            clear_mask = ~(0b11 << shifts[0])
        else:
            if self.gate_type in ["CNOT", "CX"]:
                transform = self._transform_cnot
            elif self.gate_type == "CZ":
                transform = self._transform_cz
            elif self.gate_type == "SWAP":
                transform = self._transform_swap
            else:
                raise ValueError(f"Transformation not implemented for {self.gate_type}")
            shifts = (2 * self.qubits[0], 2 * self.qubits[1])
            local_terms = [
                (p_a << shifts[0]) | (p_b << shifts[1]) for p_b in range(4) for p_a in range(4)
            ]
            clear_mask = ~((0b11 << shifts[0]) | (0b11 << shifts[1]))

        replacement_bits = []
        phases = []
        for probe in local_terms:
            new_term, phase = transform(probe)
            replacement_bits.append(int(new_term))
            phases.append(complex(phase))

        self._transform_table = (clear_mask, shifts, replacement_bits, phases)
        return self._transform_table

    def transform_table_numpy(self):
        """Return the transform table in numpy form for the array engine.

        Requires nqubits <= 32 so all bit patterns fit in uint64 (the array
        engine enforces this before calling).

        Returns:
            Tuple of (uint64 clear mask, shifts, uint64 replacement-bits
            array, complex128 phase array)
        """
        if self._transform_table_np is None:
            table = self._transform_table
            if table is None:
                table = self._build_transform_table()
            clear_mask, shifts, replacement_bits, phases = table
            self._transform_table_np = (
                np.uint64(clear_mask & 0xFFFFFFFFFFFFFFFF),
                shifts,
                np.array(replacement_bits, dtype=np.uint64),
                np.array(phases, dtype=np.complex128),
            )
        return self._transform_table_np

    def _transform_single_qubit(self, pauli_term: int):
        """Transform single-qubit Clifford gate."""
        qubit = self.qubits[0]
        pauli_int = get_pauli(pauli_term, qubit, self.nqubits)
        pauli_symbol = int_to_symbol(pauli_int)

        # Look up transformation rule
        new_symbol, phase = self.SINGLE_QUBIT_RULES[self.gate_type][pauli_symbol]
        new_pauli_int = symbol_to_int(new_symbol)

        # Set the transformed Pauli
        new_term = set_pauli(pauli_term, qubit, new_pauli_int, self.nqubits)
        return new_term, complex(phase)

    def _transform_cnot(self, pauli_term: int):
        """Transform through CNOT: control=qubits[0], target=qubits[1].

        Uses the TWO_QUBIT_RULES table (Heisenberg picture, U† P U).
        """
        return self._transform_two_qubit_rules("CNOT", pauli_term)

    def _transform_cz(self, pauli_term: int):
        """Transform through CZ (symmetric in its two qubits).

        Uses the TWO_QUBIT_RULES table (Heisenberg picture, U† P U).
        """
        return self._transform_two_qubit_rules("CZ", pauli_term)

    def _transform_two_qubit_rules(self, rule_name: str, pauli_term: int):
        """Apply a TWO_QUBIT_RULES entry to the gate's qubit pair."""
        qubit_a, qubit_b = self.qubits[0], self.qubits[1]
        p_a = get_pauli(pauli_term, qubit_a, self.nqubits)
        p_b = get_pauli(pauli_term, qubit_b, self.nqubits)

        new_a, new_b, phase = self.TWO_QUBIT_RULES[rule_name][(p_a, p_b)]

        new_term = set_pauli(pauli_term, qubit_a, new_a, self.nqubits)
        new_term = set_pauli(new_term, qubit_b, new_b, self.nqubits)
        return new_term, complex(phase)

    def _transform_swap(self, pauli_term: int):
        """Transform SWAP gate - exchanges Paulis on two qubits."""
        q0, q1 = self.qubits
        p0 = get_pauli(pauli_term, q0, self.nqubits)
        p1 = get_pauli(pauli_term, q1, self.nqubits)

        # Simply swap the Paulis
        new_term = set_pauli(pauli_term, q0, p1, self.nqubits)
        new_term = set_pauli(new_term, q1, p0, self.nqubits)

        return new_term, 1.0

    def __repr__(self) -> str:
        """String representation."""
        qubits_str = str(self.qubits) if len(self.qubits) > 1 else str(self.qubits[0])
        return f"CliffordGate({self.gate_type}, qubits={qubits_str})"


class LayerBarrier:  # pylint: disable=too-few-public-methods
    """Marker class for circuit layer boundaries.

    LayerBarrier is not an actual quantum gate. It's a placeholder used to mark
    the end of a circuit layer/group. When converting Qiskit circuits with
    barrier() instructions, these are converted to LayerBarrier markers.

    The propagation functions use these markers to group gates into layers:
    - Gates before the first barrier form layer 1
    - Gates between barriers form subsequent layers
    - Without barriers, each gate forms its own layer (for backward compatibility)

    After each layer is propagated, symmetry merging and truncation are applied
    (if enabled).

    Example:
        gates = [RX, RY, LayerBarrier(), CX, CZ, LayerBarrier(), RZ]
        -> Layers: [[RX, RY], [CX, CZ], [RZ]]
    """

    def __repr__(self) -> str:
        """String representation."""
        return "LayerBarrier()"
