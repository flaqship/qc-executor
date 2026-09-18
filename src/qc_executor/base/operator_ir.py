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

from ..parameters import Parameter, canonicalize, translate_expression

__all__ = ["PauliIR", "PAULI_CHARS"]

#: Pauli characters indexed by ``x + 2 * z``.
PAULI_CHARS = np.array(["I", "X", "Z", "Y"])


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


def _parse_labels(labels: Sequence[str], width: int) -> Tuple[np.ndarray, np.ndarray]:
    """Turn Pauli labels into symplectic ``z`` and ``x`` matrices, vectorised.

    Args:
        labels: Pauli strings of equal length ``width``.
        width: The common label length.

    Returns:
        The boolean ``(num_labels, width)`` matrices ``z`` and ``x``.

    Raises:
        ValueError: If a label contains a character other than ``I X Y Z``.
    """
    if width == 0:
        empty = np.zeros((len(labels), 0), dtype=bool)
        return empty, empty.copy()
    chars = np.char.upper(np.asarray(labels, dtype=f"U{width}")).view("U1").reshape(-1, width)
    z = (chars == "Z") | (chars == "Y")
    x = (chars == "X") | (chars == "Y")
    invalid = ~(z | x | (chars == "I"))
    if invalid.any():
        row, column = np.argwhere(invalid)[0]
        raise ValueError(
            f"invalid Pauli character {labels[row][column]!r} in label {labels[row]!r}"
        )
    return z, x


def _parity(values: np.ndarray) -> np.ndarray:
    """Return the bit parity (popcount mod 2) of each integer, as ``0``/``1``.

    Args:
        values: Non-negative ``int64`` values.

    Returns:
        An integer array of the same shape holding ``0`` or ``1``.
    """
    values = values.copy()
    for shift in (32, 16, 8, 4, 2, 1):
        values ^= values >> shift
    return values & 1


def _packed_rows(z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Pack each term's ``z`` and ``x`` bits into one fixed-width byte key.

    Args:
        z: Boolean ``(num_terms, num_qubits)`` Z components.
        x: Boolean ``(num_terms, num_qubits)`` X components.

    Returns:
        A ``(num_terms,)`` array of byte strings, comparable and hashable.
    """
    packed = np.packbits(np.concatenate([z, x], axis=1), axis=1)
    if packed.shape[1] == 0:
        return np.zeros(len(packed), dtype="S1")
    return np.ascontiguousarray(packed).view(f"S{packed.shape[1]}").ravel()


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
        self._coeffs = np.ascontiguousarray(coeffs, dtype=np.complex128).reshape(-1)
        # A zero-width operator has empty z/x rows, and numpy cannot infer the
        # row count of an empty matrix from ``-1``; take it from the coefficients.
        rows = len(self._coeffs) if self._num_qubits == 0 else -1
        self._z = np.ascontiguousarray(z, dtype=bool).reshape(rows, self._num_qubits)
        self._x = np.ascontiguousarray(x, dtype=bool).reshape(rows, self._num_qubits)
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

        z, x = _parse_labels(labels, width)

        numeric = np.ones(len(labels), dtype=np.complex128)
        symbolic: Dict[int, sp.Expr] = {}
        if coeffs is not None:
            if len(coeffs) != len(labels):
                raise ValueError(f"got {len(labels)} label(s) but {len(coeffs)} coefficient(s)")
            values = np.asarray(coeffs)
            if values.dtype != object and np.issubdtype(values.dtype, np.number):
                # Plain numbers: one vectorised cast instead of a Python loop.
                numeric = values.astype(np.complex128)
            else:
                for row, coeff in enumerate(coeffs):
                    coeff = translate_expression(coeff)
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
        if self._num_qubits == 0:
            return [""] * self.num_terms
        codes = self._x.astype(np.uint8) + 2 * self._z.astype(np.uint8)
        chars = np.ascontiguousarray(PAULI_CHARS[codes])
        # Reinterpret each row of single characters as one fixed-width string.
        return chars.view(f"U{self._num_qubits}").ravel().tolist()

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

    def scaled(self, factor: Any) -> "PauliIR":
        """Return the operator with every coefficient multiplied by ``factor``.

        Args:
            factor: A number or a SymPy expression.

        Returns:
            The scaled operator; symbolic coefficients stay symbolic.
        """
        factor = translate_expression(factor)
        if isinstance(factor, sp.Basic) and factor.free_symbols:
            return PauliIR.from_labels(
                self.to_labels(), [factor * coeff for coeff in self.coeffs], self._num_qubits
            )
        value = complex(factor)
        symbolic = {index: expr * value for index, expr in self._symbolic.items()}
        return PauliIR(self._num_qubits, self._z, self._x, self._coeffs * value, symbolic)

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

    def _require_numeric(self, operation: str) -> None:
        """Raise unless every coefficient is a plain number."""
        if self._symbolic:
            raise NotImplementedError(
                f"{operation} is not supported for operators with symbolic coefficients; "
                "bind the parameters first"
            )

    def tensor(self, other: "PauliIR") -> "PauliIR":
        """Return the tensor product with ``other`` acting on the qubits above.

        ``self`` keeps qubits ``0..n-1`` and ``other`` lands on ``n..n+m-1``, so
        the labels concatenate as ``self_label + other_label``.

        Args:
            other: The operator for the upper qubits.

        Returns:
            The product, one term per pair of input terms.

        Raises:
            NotImplementedError: If either operator carries symbolic coefficients.
        """
        self._require_numeric("tensor")
        other._require_numeric("tensor")
        rows_self = np.repeat(np.arange(self.num_terms), other.num_terms)
        rows_other = np.tile(np.arange(other.num_terms), self.num_terms)
        return PauliIR(
            self._num_qubits + other.num_qubits,
            np.hstack([self._z[rows_self], other.z[rows_other]]),
            np.hstack([self._x[rows_self], other.x[rows_other]]),
            self._coeffs[rows_self] * other.coeffs_array[rows_other],
        )

    def _term_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return per-term integer bit masks of ``z`` and ``x``, qubit 0 as the MSB."""
        weights = 1 << np.arange(self._num_qubits - 1, -1, -1, dtype=np.int64)
        return self._z.astype(np.int64) @ weights, self._x.astype(np.int64) @ weights

    def expand(self, other: "PauliIR") -> "PauliIR":
        """Return the tensor product with ``other`` acting on the qubits below."""
        return other.tensor(self)

    def power(self, exponent: int) -> "PauliIR":
        """Return the operator raised to a non-negative integer power.

        Binary exponentiation with a simplification after every product keeps
        the term count bounded.

        Args:
            exponent: The power, ``0`` giving the identity.

        Returns:
            The matrix power.
        """
        if exponent < 0:
            raise ValueError(f"exponent must be non-negative, got {exponent}")
        self._require_numeric("power")
        result = PauliIR.from_labels(["I" * self._num_qubits], [1.0], self._num_qubits)
        base = self
        while exponent:
            if exponent & 1:
                result = result.compose(base).simplify()
            exponent >>= 1
            if exponent:
                base = base.compose(base).simplify()
        return result

    def to_matrix(self, sparse: bool = False):
        """Return the operator as a matrix, qubit 0 as the most significant bit.

        The basis index convention matches the executors' statevectors and
        probability keys.  Built from the symplectic data directly: every
        term contributes one entry per row, so no Kronecker products of
        ``2 x 2`` matrices are formed.

        Args:
            sparse: Return a ``scipy.sparse.csr_matrix`` instead of a dense
                array.  SciPy is imported only in that case.

        Returns:
            The ``(2**n, 2**n)`` complex matrix.

        Raises:
            NotImplementedError: If any coefficient is symbolic.
        """
        self._require_numeric("to_matrix")
        dim = 1 << self._num_qubits
        rows = np.arange(dim, dtype=np.int64)
        z_masks, x_masks = self._term_masks()
        # P(z, x) = (-i)^(z.x) Z^z X^x maps |c> to (-1)^(popcount((c ^ x) & z)) |c ^ x>,
        # i.e. row r = c ^ x holds (-1)^(popcount(r & z)) times the phase.
        phases = (-1j) ** np.sum(self._z & self._x, axis=1)
        all_rows = np.tile(rows, self.num_terms)
        all_cols = (rows[None, :] ^ x_masks[:, None]).reshape(-1)
        signs = 1 - 2 * _parity(rows[None, :] & z_masks[:, None]).reshape(-1)
        values = np.repeat(self._coeffs * phases, dim) * signs
        if sparse:
            # Imported lazily: SciPy is not a dependency of the core package.
            from scipy.sparse import coo_matrix  # pylint: disable=import-outside-toplevel

            return coo_matrix((values, (all_rows, all_cols)), shape=(dim, dim)).tocsr()
        matrix = np.zeros((dim, dim), dtype=np.complex128)
        np.add.at(matrix, (all_rows, all_cols), values)
        return matrix

    def diagonal(self) -> np.ndarray:
        """Return the diagonal of :meth:`to_matrix` without building the matrix.

        Only the terms without ``X`` components contribute.

        Returns:
            The ``2**n`` diagonal entries, qubit 0 as the most significant bit.
        """
        self._require_numeric("diagonal")
        dim = 1 << self._num_qubits
        rows = np.arange(dim, dtype=np.int64)
        diagonal = np.zeros(dim, dtype=np.complex128)
        z_masks, _ = self._term_masks()
        for index in np.flatnonzero(~self._x.any(axis=1)):
            signs = 1 - 2 * _parity(rows & z_masks[index])
            diagonal += self._coeffs[index] * signs
        return diagonal

    def canonical_key(self, atol: float = 1e-12, decimals: int = 12) -> bytes:
        """Return a digest identifying the operator up to term order and noise.

        Terms are merged and sorted first and the coefficients rounded, so two
        operators that agree up to ``10**-decimals`` share the key.  Suited for
        de-duplicating operators, unlike :meth:`fingerprint`, which is exact.

        Args:
            atol: Coefficients below this magnitude are dropped first.
            decimals: Rounding applied to the real and imaginary parts.

        Returns:
            A 32-byte digest.
        """
        self._require_numeric("canonical_key")
        canonical = self.simplify(atol)
        digest = hashlib.blake2b(digest_size=32)
        digest.update(self._num_qubits.to_bytes(4, "little"))
        digest.update(_packed_rows(canonical.z, canonical.x).tobytes())
        # Adding zero maps -0.0 onto 0.0, so signed zeros agree after rounding.
        rounded = np.round(canonical.coeffs_array, decimals) + 0.0
        digest.update(rounded.tobytes())
        return digest.digest()

    def commutes_with(self, other: "PauliIR") -> np.ndarray:
        """Return which pairs of terms commute.

        Args:
            other: The operator to test against.

        Returns:
            A boolean ``(self.num_terms, other.num_terms)`` matrix; entry
            ``(i, j)`` is ``True`` when term ``i`` commutes with term ``j``.
        """
        if self._num_qubits != other.num_qubits:
            raise ValueError("operators must act on the same number of qubits")
        z = self._z.astype(np.int32)
        x = self._x.astype(np.int32)
        anticommuting = (z @ other.x.astype(np.int32).T + x @ other.z.astype(np.int32).T) % 2
        return anticommuting == 0

    def commutes_with_all(self, other: "PauliIR") -> np.ndarray:
        """Return, per term of this operator, whether it commutes with all of ``other``."""
        return self.commutes_with(other).all(axis=1)

    @staticmethod
    def _concatenate(operators: Sequence["PauliIR"]) -> "PauliIR":
        """Stack the terms of several operators into one, without merging.

        Args:
            operators: Operators of equal width.

        Returns:
            The combined operator; call :meth:`simplify` to merge duplicates.

        Raises:
            ValueError: If no operator is given or the widths differ.
        """
        operators = list(operators)
        if not operators:
            raise ValueError("sum needs at least one operator")
        width = operators[0].num_qubits
        if any(op.num_qubits != width for op in operators):
            raise ValueError("all operators must act on the same number of qubits")
        symbolic: Dict[int, sp.Expr] = {}
        offset = 0
        for op in operators:
            symbolic.update({offset + index: expr for index, expr in op.symbolic.items()})
            offset += op.num_terms
        return PauliIR(
            width,
            np.concatenate([op.z for op in operators]),
            np.concatenate([op.x for op in operators]),
            np.concatenate([op.coeffs_array for op in operators]),
            symbolic,
        )

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
        # Deduplicate on the bit-packed rows: comparing a few bytes per term
        # is far cheaper than sorting 2 * num_qubits booleans per term, and
        # big-endian packing keeps the same lexicographic order.
        _, first_index, inverse = np.unique(
            _packed_rows(self._z, self._x), return_index=True, return_inverse=True
        )
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

    def group_commuting(self, qubit_wise: bool = False) -> List["PauliIR"]:
        """Split the operator into groups of mutually commuting terms.

        Uses a greedy colouring, which is not guaranteed minimal but is cheap
        and deterministic.

        Args:
            qubit_wise: Require the terms of a group to commute on every qubit
                individually, so that one basis rotation measures the whole
                group.  The default allows general commutation.

        Returns:
            One operator per group.
        """
        groups = self._qubit_wise_groups() if qubit_wise else self._general_groups()
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

    def _general_groups(self) -> List[List[int]]:
        """Greedy grouping under general commutation."""
        # One integer matrix product tells for every pair whether the terms
        # anticommute (an odd number of anticommuting qubits).
        anticommute = ~self.commutes_with(self)
        groups: List[List[int]] = []
        for term in range(self.num_terms):
            for group in groups:
                if not anticommute[term, group].any():
                    group.append(term)
                    break
            else:
                groups.append([term])
        return groups

    def _qubit_wise_groups(self) -> List[List[int]]:
        """Greedy grouping under qubit-wise commutation, without a pair matrix.

        Each group keeps the Pauli it has fixed per qubit; a term joins the
        first group that agrees with it on every qubit both have touched.
        """
        occupied = self._z | self._x
        group_z = np.zeros_like(self._z)
        group_x = np.zeros_like(self._x)
        group_occupied = np.zeros_like(occupied)
        members: List[List[int]] = []
        for term in range(self.num_terms):
            count = len(members)
            conflicts = (
                group_occupied[:count]
                & occupied[term]
                & ((group_z[:count] ^ self._z[term]) | (group_x[:count] ^ self._x[term]))
            ).any(axis=1)
            fitting = np.flatnonzero(~conflicts)
            if fitting.size:
                index = int(fitting[0])
                members[index].append(term)
            else:
                index = count
                members.append([term])
            group_z[index] |= self._z[term]
            group_x[index] |= self._x[term]
            group_occupied[index] |= occupied[term]
        return members

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
