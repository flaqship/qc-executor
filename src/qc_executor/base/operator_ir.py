"""The sparse Pauli representation behind every quantum operator.

Operators are stored symplectically: two boolean matrices ``z`` and ``x`` of
shape ``(num_terms, num_qubits)`` plus a complex coefficient per term.  That
costs two bytes per qubit per term and lets composition, adjoints and commuting
checks run as vectorised NumPy rather than per-term Python.

Qubit ordering
--------------
**Qubit ``q`` is character ``q`` of a Pauli label**, i.e. qubit 0 is leftmost.
``QuantumOperator(["ZI"], [1.0])`` therefore measures Z on qubit 0.  This
matches the statevector and sampling conventions already shared by every
backend.

Only the *rendering* differs from Qiskit, which writes qubit 0 rightmost: the
symplectic ``z``/``x`` columns are indexed by qubit number on both sides, so
translation copies them across unchanged and our ``"ZI"`` simply prints as
Qiskit's ``"IZ"``.

Symbolic coefficients
---------------------
A term whose coefficient is symbolic stores ``NaN`` in the numeric column and
the expression in a sparse overlay.  ``NaN`` rather than a plausible-looking
placeholder so that reading ``coeffs`` on an unbound operator fails loudly
instead of silently returning wrong numbers.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import sympy as sp

from ..parameters import Parameter, canonicalize

__all__ = ["PauliIR", "PAULI_CHARS"]

#: Pauli characters indexed by ``x + 2 * z``.
PAULI_CHARS = np.array(["I", "X", "Z", "Y"])

#: Reverse lookup: character -> (z, x) bits.
_PAULI_BITS: Dict[str, Tuple[bool, bool]] = {
    "I": (False, False),
    "X": (False, True),
    "Z": (True, False),
    "Y": (True, True),
}


def _as_complex(value: Any) -> complex:
    """Coerce a numeric coefficient, rejecting foreign symbolic types clearly.

    Args:
        value: The coefficient to coerce.

    Returns:
        The coefficient as a complex number.

    Raises:
        TypeError: If the value is neither numeric nor a SymPy expression.
            A framework's own parameter type lands here, which is deliberate:
            the representation is framework independent and only speaks SymPy.
    """
    try:
        return complex(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "operator coefficients must be numbers or SymPy expressions built from "
            f"qc_executor Parameters, got {type(value).__name__}: {value!r}"
        ) from exc


class PauliIR:
    """A sparse weighted sum of Pauli strings.

    Args:
        num_qubits: Number of qubits the operator acts on.
        z: Boolean ``(num_terms, num_qubits)`` matrix of Z components.
        x: Boolean ``(num_terms, num_qubits)`` matrix of X components.
        coeffs: Complex coefficient per term; ``NaN`` marks a symbolic entry.
        symbolic: Symbolic coefficients keyed by term index.
    """

    __slots__ = ("_num_qubits", "_z", "_x", "_coeffs", "_symbolic", "_cache")

    def __init__(
        self,
        num_qubits: int,
        z: np.ndarray,
        x: np.ndarray,
        coeffs: np.ndarray,
        symbolic: "Mapping[int, sp.Expr] | None" = None,
    ):
        self._num_qubits = int(num_qubits)
        self._z = np.ascontiguousarray(z, dtype=bool).reshape(-1, self._num_qubits)
        self._x = np.ascontiguousarray(x, dtype=bool).reshape(-1, self._num_qubits)
        self._coeffs = np.ascontiguousarray(coeffs, dtype=np.complex128).reshape(-1)
        if len(self._z) != len(self._x) or len(self._z) != len(self._coeffs):
            raise ValueError(
                f"z, x and coeffs must agree in length, got {len(self._z)}, "
                f"{len(self._x)} and {len(self._coeffs)}"
            )
        self._symbolic: Dict[int, sp.Expr] = dict(symbolic or {})
        self._cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_labels(
        cls,
        labels: Sequence[str],
        coeffs: "Sequence[Any] | None" = None,
        num_qubits: "int | None" = None,
    ) -> "PauliIR":
        """Build an operator from Pauli labels.

        Args:
            labels: Pauli strings, qubit 0 leftmost.
            coeffs: One coefficient per label, defaulting to 1.
            num_qubits: Width, inferred from the labels when omitted.

        Returns:
            The assembled operator.

        Raises:
            ValueError: If labels disagree in length or contain a bad character.
        """
        labels = list(labels)
        if not labels:
            if num_qubits is None:
                raise ValueError("num_qubits is required when no labels are given")
            return cls.zero(num_qubits)

        widths = {len(label) for label in labels}
        if len(widths) != 1:
            raise ValueError(f"all Pauli labels must have the same length, got widths {widths}")
        width = widths.pop()
        if num_qubits is not None and num_qubits != width:
            raise ValueError(f"labels are {width} characters wide but num_qubits is {num_qubits}")

        z = np.zeros((len(labels), width), dtype=bool)
        x = np.zeros((len(labels), width), dtype=bool)
        for row, label in enumerate(labels):
            for qubit, char in enumerate(label):
                bits = _PAULI_BITS.get(char.upper())
                if bits is None:
                    raise ValueError(f"invalid Pauli character {char!r} in label {label!r}")
                z[row, qubit], x[row, qubit] = bits

        numeric = np.ones(len(labels), dtype=np.complex128)
        symbolic: Dict[int, sp.Expr] = {}
        if coeffs is not None:
            if len(coeffs) != len(labels):
                raise ValueError(f"got {len(labels)} label(s) but {len(coeffs)} coefficient(s)")
            for row, coeff in enumerate(coeffs):
                if isinstance(coeff, sp.Basic) and coeff.free_symbols:
                    symbolic[row] = canonicalize(coeff)
                    numeric[row] = np.nan
                else:
                    numeric[row] = _as_complex(coeff)

        return cls(width, z, x, numeric, symbolic)

    @classmethod
    def zero(cls, num_qubits: int) -> "PauliIR":
        """Return the all-identity operator with coefficient zero."""
        return cls(
            num_qubits,
            np.zeros((1, num_qubits), dtype=bool),
            np.zeros((1, num_qubits), dtype=bool),
            np.zeros(1, dtype=np.complex128),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_qubits(self) -> int:
        """Number of qubits the operator acts on."""
        return self._num_qubits

    @property
    def num_terms(self) -> int:
        """Number of Pauli terms."""
        return len(self._coeffs)

    @property
    def z(self) -> np.ndarray:
        """The Z component matrix, qubit ``q`` in column ``q``."""
        return self._z

    @property
    def x(self) -> np.ndarray:
        """The X component matrix, qubit ``q`` in column ``q``."""
        return self._x

    @property
    def coeffs(self) -> List[Any]:
        """Coefficients per term, symbolic entries resolved from the overlay."""
        return [
            self._symbolic[index] if index in self._symbolic else value
            for index, value in enumerate(self._coeffs)
        ]

    @property
    def numeric_coeffs(self) -> "np.ndarray | None":
        """The coefficient column, or ``None`` if any term is symbolic."""
        if self._symbolic:
            return None
        return self._coeffs

    @property
    def symbolic(self) -> Dict[int, sp.Expr]:
        """Symbolic coefficients keyed by term index."""
        return self._symbolic

    def to_labels(self) -> List[str]:
        """Return the Pauli labels, qubit 0 leftmost."""
        codes = self._x.astype(np.uint8) + 2 * self._z.astype(np.uint8)
        return ["".join(row) for row in PAULI_CHARS[codes]]

    @property
    def free_parameters(self) -> FrozenSet[Parameter]:
        """The parameters appearing in any coefficient."""
        cached = self._cache.get("free_parameters")
        if cached is None:
            found: set = set()
            for expr in self._symbolic.values():
                found.update(s for s in expr.free_symbols if isinstance(s, Parameter))
            cached = frozenset(found)
            self._cache["free_parameters"] = cached
        return cached

    @property
    def is_hermitian(self) -> bool:
        """Whether every numeric coefficient is real.

        Symbolic coefficients are assumed real, matching the ``real=True``
        assumption carried by :class:`~qc_executor.parameters.Parameter`.
        """
        finite = self._coeffs[~np.isnan(self._coeffs.real)]
        return bool(np.allclose(finite.imag, 0.0))

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------

    def substitute(self, binding: Mapping[Parameter, float]) -> "PauliIR":
        """Return a copy with parameter values substituted.

        Args:
            binding: Values to substitute.

        Returns:
            A new operator; the original is untouched.
        """
        coeffs = self._coeffs.copy()
        symbolic: Dict[int, sp.Expr] = {}
        replacements = dict(binding)
        for index, expr in self._symbolic.items():
            result = expr.xreplace(replacements)
            # xreplace on a bare Symbol returns the replacement itself, which is
            # a plain Python number rather than a SymPy object.
            if isinstance(result, sp.Basic) and result.free_symbols:
                symbolic[index] = result
            else:
                coeffs[index] = complex(result)
        return PauliIR(self._num_qubits, self._z, self._x, coeffs, symbolic)

    def adjoint(self) -> "PauliIR":
        """Return the adjoint, conjugating coefficients."""
        return PauliIR(
            self._num_qubits,
            self._z,
            self._x,
            self._coeffs.conj(),
            {index: sp.conjugate(expr) for index, expr in self._symbolic.items()},
        )

    def conjugate(self) -> "PauliIR":
        """Return the complex conjugate.

        Y is imaginary, so a term picks up a sign for every Y it contains.
        """
        y_count = np.count_nonzero(self._z & self._x, axis=1)
        signs = np.where(y_count % 2 == 0, 1.0, -1.0)
        return self._with_signs(self._coeffs.conj(), signs, sp.conjugate)

    def transpose(self) -> "PauliIR":
        """Return the transpose.

        Y is the only antisymmetric Pauli, so a term picks up a sign for every Y.
        """
        y_count = np.count_nonzero(self._z & self._x, axis=1)
        signs = np.where(y_count % 2 == 0, 1.0, -1.0)
        return self._with_signs(self._coeffs, signs, lambda expr: expr)

    def _with_signs(self, coeffs: np.ndarray, signs: np.ndarray, transform) -> "PauliIR":
        """Apply per-term signs to numeric and symbolic coefficients alike."""
        symbolic = {
            index: transform(expr) * int(signs[index]) for index, expr in self._symbolic.items()
        }
        return PauliIR(self._num_qubits, self._z, self._x, coeffs * signs, symbolic)

    def compose(self, other: "PauliIR") -> "PauliIR":
        """Return the product of two operators.

        Args:
            other: The right-hand operator.

        Returns:
            The composed operator, with one term per pair of input terms.

        Raises:
            ValueError: If the operators act on different numbers of qubits, or
                if either carries symbolic coefficients.
        """
        if self._num_qubits != other.num_qubits:
            raise ValueError(
                f"cannot compose a {self._num_qubits}-qubit operator with a "
                f"{other.num_qubits}-qubit one"
            )
        if self._symbolic or other.symbolic:
            raise NotImplementedError(
                "composing operators with symbolic coefficients is not supported; "
                "bind the parameters first"
            )

        # Pairwise over terms: (i, j) -> row i * other row j.
        z = self._z[:, None, :] ^ other.z[None, :, :]
        x = self._x[:, None, :] ^ other.x[None, :, :]
        # Writing P(z, x) = (-i)^(z.x) Z^z X^x, the product of two terms is
        #   (-i)^(z1.x1 + z2.x2) (-1)^(x1.z2) Z^z3 X^x3
        # and re-expressing that as a coefficient times P(z3, x3) leaves
        #   (-i)^(z1.x1 + z2.x2 - z3.x3 + 2 * x1.z2).
        # The 2 * x1.z2 term is the commutation cost of moving X^x1 past Z^z2.
        exponent = np.sum(
            2 * (self._x[:, None, :] & other.z[None, :, :]).astype(np.int64)
            + (self._z[:, None, :] & self._x[:, None, :]).astype(np.int64)
            + (other.z[None, :, :] & other.x[None, :, :]).astype(np.int64)
            - (z & x).astype(np.int64),
            axis=2,
        )
        phase = (-1j) ** (exponent % 4)
        coeffs = self._coeffs[:, None] * other.coeffs_array[None, :] * phase
        shape = (-1, self._num_qubits)
        return PauliIR(self._num_qubits, z.reshape(shape), x.reshape(shape), coeffs.reshape(-1))

    @property
    def coeffs_array(self) -> np.ndarray:
        """The raw complex coefficient column."""
        return self._coeffs

    def simplify(self, atol: float = 1e-12) -> "PauliIR":
        """Combine duplicate Pauli terms and drop those with zero coefficient.

        Args:
            atol: Terms whose combined coefficient is smaller than this are
                dropped.

        Returns:
            The simplified operator.

        Raises:
            NotImplementedError: If any coefficient is symbolic.
        """
        if self._symbolic:
            raise NotImplementedError(
                "simplifying operators with symbolic coefficients is not supported; "
                "bind the parameters first"
            )
        packed = np.concatenate([self._z, self._x], axis=1)
        _, first_index, inverse = np.unique(packed, axis=0, return_index=True, return_inverse=True)
        combined = np.zeros(len(first_index), dtype=np.complex128)
        np.add.at(combined, inverse.reshape(-1), self._coeffs)

        keep = np.abs(combined) > atol
        if not keep.any():
            return PauliIR.zero(self._num_qubits)
        rows = first_index[keep]
        return PauliIR(self._num_qubits, self._z[rows], self._x[rows], combined[keep])

    def apply_layout(self, layout: Sequence[int], num_qubits: "int | None" = None) -> "PauliIR":
        """Move each qubit onto a new position.

        Args:
            layout: ``layout[i]`` is the target position of source qubit ``i``.
            num_qubits: Width of the result, defaulting to the current width.

        Returns:
            The relocated operator, padded with identity on untouched qubits.

        Raises:
            ValueError: If the layout does not cover every source qubit.
        """
        if len(layout) != self._num_qubits:
            raise ValueError(
                f"layout has {len(layout)} entries but the operator has "
                f"{self._num_qubits} qubits"
            )
        width = self._num_qubits if num_qubits is None else num_qubits
        if any(not 0 <= target < width for target in layout):
            raise ValueError(f"layout targets must be within 0..{width - 1}")

        z = np.zeros((self.num_terms, width), dtype=bool)
        x = np.zeros((self.num_terms, width), dtype=bool)
        z[:, list(layout)] = self._z
        x[:, list(layout)] = self._x
        return PauliIR(width, z, x, self._coeffs, self._symbolic)

    def group_commuting(self) -> List["PauliIR"]:
        """Split the operator into groups of mutually commuting terms.

        Uses a greedy colouring, which is not guaranteed minimal but is cheap
        and deterministic.

        Returns:
            One operator per group.
        """
        groups: List[List[int]] = []
        for term in range(self.num_terms):
            for group in groups:
                if all(self._commutes(term, other) for other in group):
                    group.append(term)
                    break
            else:
                groups.append([term])

        return [
            PauliIR(
                self._num_qubits,
                self._z[rows],
                self._x[rows],
                self._coeffs[rows],
                {
                    new: self._symbolic[old]
                    for new, old in enumerate(rows)
                    if old in self._symbolic
                },
            )
            for rows in groups
        ]

    def _commutes(self, first: int, second: int) -> bool:
        """Whether two terms commute, i.e. anticommute on an even qubit count."""
        anticommuting = np.count_nonzero(
            (self._z[first] & self._x[second]) ^ (self._x[first] & self._z[second])
        )
        return anticommuting % 2 == 0

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def fingerprint(self) -> bytes:
        """Return a stable digest of the operator's content."""
        cached = self._cache.get("fingerprint")
        if cached is not None:
            return cached
        digest = hashlib.blake2b(digest_size=32)
        digest.update(self._num_qubits.to_bytes(4, "little"))
        digest.update(np.packbits(self._z, axis=None).tobytes())
        digest.update(np.packbits(self._x, axis=None).tobytes())
        digest.update(self._coeffs.tobytes())
        digest.update(repr(sorted((i, sp.srepr(e)) for i, e in self._symbolic.items())).encode())
        result = digest.digest()
        self._cache["fingerprint"] = result
        return result

    def __len__(self) -> int:
        return self.num_terms

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, PauliIR) and self.fingerprint() == other.fingerprint()

    def __hash__(self) -> int:
        return hash(self.fingerprint())

    def __repr__(self) -> str:
        return f"PauliIR(num_qubits={self._num_qubits}, num_terms={self.num_terms})"


def iter_parameters(coeffs: Iterable[Any]) -> Iterable[Parameter]:
    """Yield the parameters appearing in a sequence of coefficients.

    Args:
        coeffs: Coefficients, each numeric or a SymPy expression.

    Yields:
        Each :class:`~qc_executor.parameters.Parameter` encountered.
    """
    for coeff in coeffs:
        if isinstance(coeff, sp.Basic):
            for symbol in coeff.free_symbols:
                if isinstance(symbol, Parameter):
                    yield symbol
