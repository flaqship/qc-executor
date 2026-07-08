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

    # Transformation rules for single-qubit Clifford gates
    # Format: gate_name → {input_pauli → (output_pauli, phase)}
    SINGLE_QUBIT_RULES = {
        "H": {  # Hadamard: X ↔ Z, Y → -Y
            "I": ("I", 1.0),
            "X": ("Z", 1.0),
            "Y": ("Y", -1.0),
            "Z": ("X", 1.0),
        },
        "S": {  # Phase gate: X → Y, Y → -X, Z → Z
            "I": ("I", 1.0),
            "X": ("Y", 1.0),
            "Y": ("X", -1.0),
            "Z": ("Z", 1.0),
        },
        "T": {  # T gate (π/8 rotation): more complex, using rotation representation
            # T is exp(-i π/8 Z), not pure Clifford but commonly included
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
        """Transform CNOT gate: control=qubits[0], target=qubits[1]

        CNOT transformation rules (in Heisenberg picture):
        - I_c I_t → I_c I_t
        - I_c X_t → I_c X_t
        - I_c Y_t → Z_c Y_t
        - I_c Z_t → Z_c Z_t
        - X_c I_t → X_c X_t
        - X_c X_t → X_c I_t
        - X_c Y_t → Y_c Z_t
        - X_c Z_t → Y_c Y_t
        - Y_c I_t → Y_c X_t
        - Y_c X_t → Y_c I_t
        - Y_c Y_t → X_c Z_t
        - Y_c Z_t → -X_c Y_t
        - Z_c I_t → Z_c I_t
        - Z_c X_t → Z_c X_t
        - Z_c Y_t → I_c Y_t
        - Z_c Z_t → I_c Z_t
        """
        control = self.qubits[0]
        target = self.qubits[1]

        p_c = get_pauli(pauli_term, control, self.nqubits)
        p_t = get_pauli(pauli_term, target, self.nqubits)

        # CNOT transformation table: (p_c, p_t) → (p_c', p_t', phase)
        # Using XOR-based rules for CNOT
        new_term = pauli_term
        phase = 1.0

        # Simplified rules using XOR logic:
        # Control: X_c → X_c X_t (XOR with target X)
        # Target: Z_t → Z_c Z_t (XOR with control Z)
        if p_c == 1:  # Control is X
            # XOR target with X
            p_t_new = p_t ^ 1  # Toggle X bit
            new_term = set_pauli(new_term, target, p_t_new, self.nqubits)
            # Phase adjustment for Y
            if p_t == 2:  # Y → Z: -i phase, but CNOT is real, check rules
                phase *= (-1) ** ((p_c == 2 or p_t == 2) and p_c * p_t != 0)

        if p_t == 3:  # Target is Z
            # Z_t acts on control
            if p_c == 0:  # I_c Z_t → Z_c Z_t
                new_term = set_pauli(new_term, control, 3, self.nqubits)
            elif p_c == 1:  # X_c Z_t → Y_c Y_t (with phase)
                new_term = set_pauli(new_term, control, 2, self.nqubits)
                new_term = set_pauli(new_term, target, 2, self.nqubits)
            elif p_c == 2:  # Y_c Z_t → -X_c Y_t
                new_term = set_pauli(new_term, control, 1, self.nqubits)
                new_term = set_pauli(new_term, target, 2, self.nqubits)
                phase = -1.0
            # p_c == 3: Z_c Z_t → I_c Z_t
            elif p_c == 3:
                new_term = set_pauli(new_term, control, 0, self.nqubits)

        elif p_t == 2:  # Target is Y
            # Y_t = iXZ, needs special handling
            if p_c == 0:  # I_c Y_t → Z_c Y_t
                new_term = set_pauli(new_term, control, 3, self.nqubits)
            elif p_c == 1:  # X_c Y_t → Y_c Z_t
                new_term = set_pauli(new_term, control, 2, self.nqubits)
                new_term = set_pauli(new_term, target, 3, self.nqubits)
            elif p_c == 2:  # Y_c Y_t → X_c Z_t
                new_term = set_pauli(new_term, control, 1, self.nqubits)
                new_term = set_pauli(new_term, target, 3, self.nqubits)
            elif p_c == 3:  # Z_c Y_t → I_c Y_t
                new_term = set_pauli(new_term, control, 0, self.nqubits)

        elif p_t == 1:  # Target is X
            if p_c == 1:  # X_c X_t → X_c I_t
                new_term = set_pauli(new_term, target, 0, self.nqubits)
            elif p_c == 2:  # Y_c X_t → Y_c I_t
                new_term = set_pauli(new_term, target, 0, self.nqubits)

        return new_term, complex(phase)

    def _transform_cz(self, pauli_term: int):
        """Transform CZ gate.

        CZ is symmetric: both qubits are control/target.
        CZ transformation: X_i → X_i Z_j, Z → Z (identity on Z)
        """
        q0, q1 = self.qubits[0], self.qubits[1]
        p0 = get_pauli(pauli_term, q0, self.nqubits)
        p1 = get_pauli(pauli_term, q1, self.nqubits)

        new_term = pauli_term
        phase = 1.0

        # X_0 → X_0 Z_1 and X_1 → X_1 Z_0
        if p0 == 1:  # X on q0
            p1_new = p1 ^ 3 if p1 != 3 else 0  # Toggle Z
            new_term = set_pauli(new_term, q1, p1_new if p1 == 0 else (p1 ^ 3) % 4, self.nqubits)

        if p1 == 1:  # X on q1
            p0_new = p0 ^ 3 if p0 != 3 else 0
            new_term = set_pauli(new_term, q0, p0_new if p0 == 0 else (p0 ^ 3) % 4, self.nqubits)

        # Handle Y (which has both X and Z components)
        # This needs careful phase tracking
        # For simplicity, using composition rules

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
