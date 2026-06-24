"""Framework-independent quantum operator backed by Pauli strings + coefficients.

This is the operator counterpart to
:class:`~executor.abstraction.abstract_quantum_circuit.AbstractQuantumCircuit`:
where the circuit describes *what you run*, the operator describes the
**observable you measure** (the ``observable`` argument of
``expectation_value``).

An observable is stored as plain data — a list of Pauli label strings and a
matching list of coefficients. Coefficients may be plain numbers or symbolic
:class:`~executor.abstraction.abstract_parameter.Parameter` expressions (SymPy),
so parametrized observables carry no quantum-framework dependency. Each backend
converts this canonical ``(paulis, coeffs)`` data into its native operator type.

Pauli labels follow Qiskit's **little-endian** convention: the rightmost
character acts on qubit 0. For example ``"XIIZ"`` means ``Z`` on qubit 0 and
``X`` on qubit 3, with qubits 1 and 2 left untouched (identity).
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import sympy as sp

from ..base import QuantumOperatorBase
from .abstract_parameter import Parameter, free_parameters

#: A coefficient is either numeric or a symbolic parameter expression.
Coeff = Union[float, int, complex, sp.Basic]

# Single-qubit Pauli products: ``left + right`` -> ``(phase, result)``.
# Used by :meth:`AbstractQuantumOperator.compose`.
_PAULI_MUL = {
    "II": (1, "I"), "IX": (1, "X"), "IY": (1, "Y"), "IZ": (1, "Z"),
    "XI": (1, "X"), "XX": (1, "I"), "XY": (1j, "Z"), "XZ": (-1j, "Y"),
    "YI": (1, "Y"), "YX": (-1j, "Z"), "YY": (1, "I"), "YZ": (1j, "X"),
    "ZI": (1, "Z"), "ZX": (1j, "Y"), "ZY": (-1j, "X"), "ZZ": (1, "I"),
}


def _conj(c: Coeff) -> Coeff:
    """Complex conjugate that works for numbers and SymPy expressions."""
    return sp.conjugate(c) if isinstance(c, sp.Basic) else np.conjugate(c)


def _is_zero(c: Coeff) -> bool:
    """True if ``c`` is (numerically or symbolically) zero."""
    if isinstance(c, sp.Basic):
        return bool(sp.simplify(c) == 0)
    return bool(np.isclose(complex(c), 0.0))


def _y_phase(pauli: str) -> int:
    """Sign picked up under transpose/conjugate (-1 for an odd number of Y)."""
    return -1 if pauli.count("Y") % 2 else 1


class AbstractQuantumOperator(QuantumOperatorBase):
    """Observable defined independently of any quantum framework.

    Build it from Pauli label strings and matching coefficients::

        # 0.5·Z0Z1 + 0.3·X0  on 2 qubits
        H = AbstractQuantumOperator(paulis=["ZZ", "IX"], coeffs=[0.5, 0.3])

    Args:
        paulis: Pauli label strings (e.g. ``["IX", "ZZ"]``); all the same length.
        coeffs: Matching coefficients (number or SymPy expression). Defaults to 1.
        num_qubits: Only required when ``paulis`` is empty.
    """

    def __init__(
        self,
        paulis: Optional[List[str]] = None,
        coeffs: Optional[List[Coeff]] = None,
        num_qubits: Optional[int] = None,
    ):
        paulis = list(paulis) if paulis else []
        if paulis:
            width = len(paulis[0])
            if any(len(p) != width for p in paulis):
                raise ValueError("All Pauli strings must have the same length.")
            num_qubits = width if num_qubits is None else num_qubits
        elif num_qubits is None:
            raise ValueError("num_qubits is required when paulis is empty.")

        coeffs = list(coeffs) if coeffs is not None else [1.0] * len(paulis)
        if len(coeffs) != len(paulis):
            raise ValueError("Length of coeffs must match length of paulis.")

        super().__init__(num_qubits=num_qubits, paulis=paulis, coeffs=coeffs)

    @classmethod
    def from_quantum_operator(cls, operator: QuantumOperatorBase) -> "AbstractQuantumOperator":
        """Return ``operator`` unchanged if abstract, else copy its paulis/coeffs."""
        if isinstance(operator, cls):
            return operator.copy()
        return cls(paulis=list(operator.paulis), coeffs=list(operator.coeffs))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def num_paulis(self) -> int:
        return len(self._paulis)

    @property
    def parameters(self) -> List[Parameter]:
        """Return all native parameters used in the coefficients, sorted."""
        seen: dict = {}
        for c in self._coeffs:
            for param in free_parameters(c):
                seen[param] = None
        return sorted(seen, key=lambda p: (p.vector_name, p.index))

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    @property
    def is_parametrized(self) -> bool:
        return len(self.parameters) > 0

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------

    def append(self, pauli: str, coeff: Coeff = 1.0) -> "AbstractQuantumOperator":
        """Append a single Pauli term (mutates and returns ``self``)."""
        if len(pauli) != self._num_qubits:
            raise ValueError("Pauli length does not match num_qubits.")
        self._paulis.append(pauli)
        self._coeffs.append(coeff)
        return self

    def compose(self, other: QuantumOperatorBase) -> "AbstractQuantumOperator":
        """Compose with ``other`` (apply ``self`` first, then ``other``).

        Matches Qiskit's ``SparsePauliOp.compose`` (``front=False``), i.e. the
        matrix product ``other · self``. The result is not auto-simplified (call
        :meth:`simplify`). Verify against
        ``tests/pauli_propagation/test_cross_backend_parity.py``.
        """
        if other.num_qubits != self._num_qubits:
            raise ValueError("Cannot compose operators with different qubit counts.")
        paulis: List[str] = []
        coeffs: List[Coeff] = []
        for pa, ca in zip(self._paulis, self._coeffs):
            for pb, cb in zip(other.paulis, other.coeffs):
                phase: Coeff = 1
                chars = []
                for x, y in zip(pa, pb):
                    # other · self  ->  other's char on the left of the product
                    ph, r = _PAULI_MUL[y + x]
                    phase *= ph
                    chars.append(r)
                paulis.append("".join(chars))
                coeffs.append(ca * cb * phase)
        self._paulis, self._coeffs = paulis, coeffs
        return self

    def adjoint(self) -> "AbstractQuantumOperator":
        """Conjugate transpose. Each Pauli string is Hermitian, so only the
        coefficients are conjugated."""
        return self.__class__(self._paulis, [_conj(c) for c in self._coeffs], self._num_qubits)

    def transpose(self) -> "AbstractQuantumOperator":
        return self.__class__(
            self._paulis,
            [c * _y_phase(p) for p, c in zip(self._paulis, self._coeffs)],
            self._num_qubits,
        )

    def conjugate(self) -> "AbstractQuantumOperator":
        return self.__class__(
            self._paulis,
            [_conj(c) * _y_phase(p) for p, c in zip(self._paulis, self._coeffs)],
            self._num_qubits,
        )

    def simplify(self) -> "AbstractQuantumOperator":
        """Combine duplicate Pauli strings and drop (near-)zero terms."""
        combined: dict = {}
        for p, c in zip(self._paulis, self._coeffs):
            combined[p] = combined.get(p, 0) + c
        paulis, coeffs = [], []
        for p, c in combined.items():
            c = sp.simplify(c) if isinstance(c, sp.Basic) else c
            if not _is_zero(c):
                paulis.append(p)
                coeffs.append(c)
        if not paulis:  # keep a valid (zero) operator on the same register
            return self.__class__(["I" * self._num_qubits], [0.0], self._num_qubits)
        return self.__class__(paulis, coeffs, self._num_qubits)

    def apply_layout(
        self, layout: List[int], num_qubits: Optional[int] = None
    ) -> "AbstractQuantumOperator":
        """Embed the operator on a (possibly larger) register.

        ``layout[q]`` is the new index of qubit ``q``. Follows Qiskit's
        little-endian convention so the result matches ``SparsePauliOp.apply_layout``.
        """
        n = self._num_qubits
        new_n = num_qubits if num_qubits is not None else (max(layout) + 1)
        new_paulis = []
        for label in self._paulis:
            chars = ["I"] * new_n
            for q in range(n):
                chars[new_n - 1 - layout[q]] = label[n - 1 - q]
            new_paulis.append("".join(chars))
        return self.__class__(new_paulis, list(self._coeffs), new_n)

    def group_commuting(self) -> List["AbstractQuantumOperator"]:
        """Greedy *qubit-wise* commuting grouping.

        Note: Qiskit's default groups generally-commuting terms; this is the
        simpler qubit-wise variant (two strings commute if, on every qubit, the
        Paulis are equal or at least one is ``I``).
        """

        def qwise_commute(a: str, b: str) -> bool:
            return all(x == y or "I" in (x, y) for x, y in zip(a, b))

        groups: List[List[int]] = []
        for i, p in enumerate(self._paulis):
            for g in groups:
                if all(qwise_commute(p, self._paulis[j]) for j in g):
                    g.append(i)
                    break
            else:
                groups.append([i])
        return [
            self.__class__(
                [self._paulis[i] for i in g],
                [self._coeffs[i] for i in g],
                self._num_qubits,
            )
            for g in groups
        ]

    # ------------------------------------------------------------------
    # Parameters / utility
    # ------------------------------------------------------------------

    def assign_parameters(self, parameters: dict) -> None:
        """Substitute numeric values for symbolic parameters in every coefficient."""
        for i, c in enumerate(self._coeffs):
            if isinstance(c, sp.Basic):
                result = c.subs(parameters)
                if result.is_number:
                    self._coeffs[i] = float(result) if result.is_real else complex(result)
                else:
                    self._coeffs[i] = result

    def copy(self) -> "AbstractQuantumOperator":
        return self.__class__(list(self._paulis), list(self._coeffs), self._num_qubits)

    @property
    def is_unitary(self) -> bool:
        """True for a single Pauli term with unit-modulus coefficient."""
        return len(self._paulis) == 1 and _is_zero(abs(complex(self._coeffs[0])) - 1.0)

    @property
    def is_real(self) -> bool:
        return all(
            _is_zero(sp.im(c) if isinstance(c, sp.Basic) else np.imag(c)) for c in self._coeffs
        )

    @property
    def is_imaginary(self) -> bool:
        return all(
            _is_zero(sp.re(c) if isinstance(c, sp.Basic) else np.real(c)) for c in self._coeffs
        )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def _signature(self):
        return (self._num_qubits, tuple(sorted((p, str(c)) for p, c in zip(self._paulis, self._coeffs))))

    def __hash__(self):
        return hash(self._signature())

    def __eq__(self, other):
        return (
            isinstance(other, AbstractQuantumOperator)
            and self._signature() == other._signature()
        )

    def __str__(self):
        return " + ".join(f"({c}) * {p}" for c, p in zip(self._coeffs, self._paulis))

    def __repr__(self):
        return f"AbstractQuantumOperator(num_qubits={self._num_qubits}, terms={len(self._paulis)})"
