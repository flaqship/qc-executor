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

#: The only characters allowed in a Pauli label (checked at construction).
_PAULI_CHARS = frozenset("IXYZ")

# Single-qubit Pauli products: ``left + right`` -> ``(phase, result)``.
# Used by :meth:`AbstractQuantumOperator.compose`.
_PAULI_MUL = {
    "II": (1, "I"), "IX": (1, "X"), "IY": (1, "Y"), "IZ": (1, "Z"),
    "XI": (1, "X"), "XX": (1, "I"), "XY": (1j, "Z"), "XZ": (-1j, "Y"),
    "YI": (1, "Y"), "YX": (-1j, "Z"), "YY": (1, "I"), "YZ": (1j, "X"),
    "ZI": (1, "Z"), "ZX": (1j, "Y"), "ZY": (-1j, "X"), "ZZ": (1, "I"),
}


def _is_zero(c: Coeff) -> bool:
    """True if ``c`` is (numerically or symbolically) zero."""
    if isinstance(c, sp.Basic):
        return bool(sp.simplify(c) == 0)
    return bool(np.isclose(complex(c), 0.0))


class AbstractQuantumOperator(QuantumOperatorBase):
    """Observable defined independently of any quantum framework.

    Build it from Pauli label strings and matching coefficients::

        # 0.5·Z0Z1 + 0.3·X0  on 2 qubits
        H = AbstractQuantumOperator(paulis=["ZZ", "IX"], coeffs=[0.5, 0.3])

    Args:
        paulis: Pauli label strings (e.g. ``["IX", "ZZ"]``); all the same
            length, uppercase characters ``I``, ``X``, ``Y``, ``Z`` only.
        coeffs: Matching coefficients (number or SymPy expression). Defaults to 1.
        num_qubits: Only required when ``paulis`` is empty; if given alongside
            ``paulis`` it must equal the label length.
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
            for p in paulis:
                if not _PAULI_CHARS.issuperset(p):
                    raise ValueError(
                        f"Invalid Pauli label {p!r}: only the characters "
                        "I, X, Y, Z are allowed."
                    )
            if num_qubits is not None and num_qubits != width:
                raise ValueError(
                    f"num_qubits ({num_qubits}) does not match the Pauli "
                    f"string length ({width})."
                )
            num_qubits = width
        elif num_qubits is None:
            raise ValueError("num_qubits is required when paulis is empty.")

        coeffs = list(coeffs) if coeffs is not None else [1.0] * len(paulis)
        if len(coeffs) != len(paulis):
            raise ValueError("Length of coeffs must match length of paulis.")

        super().__init__(num_qubits=num_qubits, paulis=paulis, coeffs=coeffs)

    @classmethod
    def from_quantum_operator(cls, operator: QuantumOperatorBase) -> "AbstractQuantumOperator":
        """Return ``operator`` as a copy if abstract; reject anything else.

        Observables enter the library through the abstraction only (matching
        ``from_quantum_circuit``), so backend-native operators are not
        converted here.
        """
        if isinstance(operator, cls):
            return operator.copy()
        raise TypeError(
            f"Cannot build an AbstractQuantumOperator from {type(operator).__name__}. "
            "Build the observable as a qc_executor.abstraction.AbstractQuantumOperator instead."
        )

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

    def compose(self, other: QuantumOperatorBase) -> "AbstractQuantumOperator":
        """Compose with ``other`` (apply ``self`` first, then ``other``).

        Matches Qiskit's ``SparsePauliOp.compose`` (``front=False``), i.e. the
        matrix product ``other · self``, and like Qiskit returns a new operator
        (``self`` is left unchanged). The result is not auto-simplified (call
        :meth:`simplify`). Verify against
        ``tests/pauli_propagation/test_cross_backend_parity.py``.
        """
        if other.num_qubits != self._num_qubits:
            raise ValueError("Cannot compose operators with different qubit counts.")
        other_paulis, other_coeffs = other.paulis, other.coeffs
        paulis: List[str] = []
        coeffs: List[Coeff] = []
        for pa, ca in zip(self._paulis, self._coeffs):
            for pb, cb in zip(other_paulis, other_coeffs):
                phase: Coeff = 1
                chars = []
                for x, y in zip(pa, pb):
                    # other · self  ->  other's char on the left of the product
                    ph, r = _PAULI_MUL[y + x]
                    phase *= ph
                    chars.append(r)
                paulis.append("".join(chars))
                coeffs.append(ca * cb * phase)
        return self.__class__(paulis, coeffs, self._num_qubits)

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

    # ------------------------------------------------------------------
    # Parameters / utility
    # ------------------------------------------------------------------

    def assign_parameters(self, parameters: dict) -> "AbstractQuantumOperator":
        """Substitute numeric values for symbolic parameters in every coefficient.

        Mutates ``self`` (matching ``AbstractQuantumCircuit.assign_parameters``)
        and returns it so the call can also be used in expression position.
        """
        for i, c in enumerate(self._coeffs):
            if isinstance(c, sp.Basic):
                result = c.subs(parameters)
                if result.is_number:
                    self._coeffs[i] = float(result) if result.is_real else complex(result)
                else:
                    self._coeffs[i] = result
        return self

    def copy(self) -> "AbstractQuantumOperator":
        return self.__class__(list(self._paulis), list(self._coeffs), self._num_qubits)

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
