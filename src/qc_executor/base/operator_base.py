"""The quantum operator interface shared by the generic and native operators.

Like :class:`~qc_executor.base.circuit_base.QuantumCircuitBase`, this class is
concrete: it owns the sparse Pauli representation and implements the whole
operator API on top of it.  A backend subclass supplies only how to compile that
representation into its native form.
"""

from __future__ import annotations

from abc import ABC
from numbers import Number
from typing import Any, List, Mapping, Sequence

import numpy as np
import sympy as sp

from ..parameters import Parameter, sort_parameters, translate_expression
from .operator_ir import PauliIR

__all__ = ["QuantumOperatorBase"]


class QuantumOperatorBase(ABC):
    """A weighted sum of Pauli strings, independent of any quantum framework.

    Qubit ``q`` is character ``q`` of a Pauli label, so ``["ZI"]`` acts with Z on
    qubit 0.

    Args:
        paulis: Pauli labels making up the operator.
        coeffs: One coefficient per label; numbers or SymPy expressions.
        num_qubits: Width, required only when no labels are given.
        _ir: Adopt this representation instead of building one.  Used by
            conversion helpers; not part of the public construction API.
    """

    def __init__(
        self,
        paulis: "Sequence[str] | None" = None,
        coeffs: "Sequence[Any] | None" = None,
        num_qubits: "int | None" = None,
        *,
        _ir: "PauliIR | None" = None,
    ):
        if _ir is not None:
            self._ir = _ir
        elif paulis is not None:
            self._ir = PauliIR.from_labels(paulis, coeffs, num_qubits)
        elif num_qubits is not None:
            self._ir = PauliIR.zero(num_qubits)
        else:
            raise ValueError("Must provide paulis, num_qubits, or an existing representation")
        self._native_cache: Any = None
        self._native_built = False

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------

    def _build_native(self) -> Any:
        """Compile the representation into this backend's native operator."""
        raise NotImplementedError(
            f"{type(self).__name__} has no native representation; "
            "override _build_native() in a backend subclass"
        )

    @property
    def native(self) -> Any:
        """The compiled native operator, built on first use and cached."""
        if not self._native_built:
            self._native_cache = self._build_native()
            self._native_built = True
        return self._native_cache

    @classmethod
    def from_quantum_operator(
        cls, operator: "QuantumOperatorBase", **options: Any
    ) -> "QuantumOperatorBase":
        """Convert any operator into this operator type.

        Args:
            operator: The operator to convert.
            ``**options``: Backend-specific conversion options.

        Returns:
            ``operator`` unchanged if it is already of this type, else a new
            instance sharing its content.
        """
        if isinstance(operator, cls):
            return operator
        return cls(_ir=operator.ir, **options)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ir(self) -> PauliIR:
        """The underlying sparse Pauli representation."""
        return self._ir

    @property
    def num_qubits(self) -> int:
        """Number of qubits the operator acts on."""
        return self._ir.num_qubits

    @property
    def num_paulis(self) -> int:
        """Number of Pauli terms."""
        return self._ir.num_terms

    @property
    def paulis(self) -> List[str]:
        """The Pauli labels, qubit 0 leftmost."""
        return self._ir.to_labels()

    @property
    def coeffs(self) -> List[Any]:
        """The coefficients, symbolic entries kept as SymPy expressions."""
        return self._ir.coeffs

    @property
    def coeffs_array(self) -> np.ndarray:
        """The coefficients as a complex NumPy array.

        Raises:
            ValueError: If any coefficient is symbolic.
        """
        if self._ir.symbolic:
            raise ValueError("coeffs_array needs numeric coefficients; bind the parameters first")
        return self._ir.coeffs_array

    @property
    def parameters(self) -> List[Parameter]:
        """The free parameters, sorted by ``(vector_name, index)``."""
        return sort_parameters(self._ir.free_parameters)

    @property
    def num_parameters(self) -> int:
        """Number of free parameters."""
        return len(self._ir.free_parameters)

    @property
    def is_parametrized(self) -> bool:
        """Whether any coefficient is symbolic."""
        return bool(self._ir.free_parameters)

    @property
    def is_hermitian(self) -> bool:
        """Whether every numeric coefficient is real."""
        return self._ir.is_hermitian

    @property
    def is_real(self) -> bool:
        """Whether every numeric coefficient is real."""
        return self._ir.is_hermitian

    @property
    def is_imaginary(self) -> bool:
        """Whether every numeric coefficient is purely imaginary."""
        coeffs = self._ir.coeffs_array
        finite = coeffs[~np.isnan(coeffs.real)]
        return bool(np.allclose(finite.real, 0.0))

    @property
    def is_unitary(self) -> bool:
        """Whether the operator is unitary.

        Raises:
            NotImplementedError: If any coefficient is symbolic.
        """
        product = self._ir.compose(self._ir.adjoint()).simplify()
        if product.num_terms != 1:
            return False
        return product.to_labels() == ["I" * self.num_qubits] and bool(
            np.isclose(complex(product.coeffs[0]), 1.0)
        )

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------

    #: Keep NumPy scalars from treating the operator as a sequence of terms:
    #: ``np.float64(2.0) * op`` must reach :meth:`__rmul__` instead of being
    #: broadcast by numpy; the flip side is that ``np.sum`` over operators is
    #: not available, use :meth:`sum` or the built-in ``sum``.
    __array_ufunc__ = None
    #: ``theta * op`` with a SymPy symbol on the left: SymPy defers to the
    #: operand with the higher priority, so it reaches :meth:`__rmul__` too.
    _op_priority = 100.0

    def _rebuild(self, ir: PauliIR) -> "QuantumOperatorBase":
        """Wrap a new representation in this operator's type."""
        return type(self)(_ir=ir)

    def copy(self) -> "QuantumOperatorBase":
        """Return a copy of this operator."""
        return self._rebuild(self._ir)

    def adjoint(self) -> "QuantumOperatorBase":
        """Return the adjoint of the operator."""
        return self._rebuild(self._ir.adjoint())

    def transpose(self) -> "QuantumOperatorBase":
        """Return the transpose of the operator."""
        return self._rebuild(self._ir.transpose())

    def conjugate(self) -> "QuantumOperatorBase":
        """Return the complex conjugate of the operator."""
        return self._rebuild(self._ir.conjugate())

    def simplify(self, atol: float = 1e-12) -> "QuantumOperatorBase":
        """Combine duplicate terms and drop those with zero coefficient.

        Args:
            atol: Terms whose combined coefficient is smaller than this are
                dropped; ``0.0`` drops exact zeros only.
        """
        return self._rebuild(self._ir.simplify(atol))

    def tensor(self, other: "QuantumOperatorBase") -> "QuantumOperatorBase":
        """Return the tensor product with ``other`` acting on the qubits above this one's."""
        return self._rebuild(self._ir.tensor(other.ir))

    def expand(self, other: "QuantumOperatorBase") -> "QuantumOperatorBase":
        """Return the tensor product with ``other`` acting on the qubits below this one's."""
        return self._rebuild(self._ir.expand(other.ir))

    def power(self, exponent: int) -> "QuantumOperatorBase":
        """Return the operator raised to a non-negative integer power."""
        return self._rebuild(self._ir.power(exponent))

    def to_matrix(self, sparse: bool = False):
        """Return the operator as a matrix, qubit 0 as the most significant bit.

        Args:
            sparse: Return a ``scipy.sparse.csr_matrix`` instead of a dense array.
        """
        return self._ir.to_matrix(sparse=sparse)

    def diagonal(self) -> np.ndarray:
        """Return the diagonal of :meth:`to_matrix` without building the matrix."""
        return self._ir.diagonal()

    def canonical_key(self, atol: float = 1e-12, decimals: int = 12) -> bytes:
        """Return a digest identifying the operator up to term order and rounding."""
        return self._ir.canonical_key(atol, decimals)

    def commutes_with(self, other: "QuantumOperatorBase") -> np.ndarray:
        """Return the boolean ``(len(self), len(other))`` matrix of commuting term pairs."""
        return self._ir.commutes_with(other.ir)

    def commutes_with_all(self, other: "QuantumOperatorBase") -> np.ndarray:
        """Return, per term, whether it commutes with every term of ``other``."""
        return self._ir.commutes_with_all(other.ir)

    def compose(self, other: "QuantumOperatorBase") -> "QuantumOperatorBase":
        """Return the product with another operator.

        This is pure: neither operand is modified.

        Args:
            other: The right-hand operator.

        Returns:
            The composed operator.

        Raises:
            TypeError: If ``other`` is not a quantum operator.
        """
        if not isinstance(other, QuantumOperatorBase):
            raise TypeError(
                f"can only compose with a quantum operator, got {type(other).__name__}"
            )
        return self._rebuild(self._ir.compose(other.ir))

    def apply_layout(
        self, layout: Sequence[int], num_qubits: "int | None" = None
    ) -> "QuantumOperatorBase":
        """Move each qubit onto a new position.

        Args:
            layout: ``layout[i]`` is the target position of source qubit ``i``.
            num_qubits: Width of the result, defaulting to the current width.

        Returns:
            The relocated operator.
        """
        return self._rebuild(self._ir.apply_layout(layout, num_qubits))

    def append(self, pauli: str, coeff: Any = None) -> "QuantumOperatorBase":
        """Return this operator with one more Pauli term.

        Args:
            pauli: The Pauli label to add.
            coeff: Its coefficient, defaulting to 1.

        Returns:
            The extended operator.
        """
        extra = PauliIR.from_labels([pauli], [1.0 if coeff is None else coeff])
        if extra.num_qubits != self.num_qubits:
            raise ValueError(
                f"cannot append a {extra.num_qubits}-qubit term to a "
                f"{self.num_qubits}-qubit operator"
            )
        combined = PauliIR(
            self.num_qubits,
            np.concatenate([self._ir.z, extra.z]),
            np.concatenate([self._ir.x, extra.x]),
            np.concatenate([self._ir.coeffs_array, extra.coeffs_array]),
            {**self._ir.symbolic, **{k + self.num_paulis: v for k, v in extra.symbolic.items()}},
        )
        return self._rebuild(combined)

    def __add__(self, other: Any) -> "QuantumOperatorBase":
        """Return the sum with another operator, as a new operator.

        Terms are concatenated without simplification, mirroring Qiskit's
        ``SparsePauliOp.__add__``; call :meth:`simplify` to merge duplicates.

        Args:
            other: The operator to add.

        Returns:
            The sum, in this operator's type.

        Raises:
            ValueError: If the operators act on different numbers of qubits.
        """
        if not isinstance(other, QuantumOperatorBase):
            return NotImplemented
        if other.num_qubits != self.num_qubits:
            raise ValueError(
                f"cannot add a {other.num_qubits}-qubit operator to a "
                f"{self.num_qubits}-qubit operator"
            )
        other_ir = other.ir
        combined = PauliIR(
            self.num_qubits,
            np.concatenate([self._ir.z, other_ir.z]),
            np.concatenate([self._ir.x, other_ir.x]),
            np.concatenate([self._ir.coeffs_array, other_ir.coeffs_array]),
            {
                **self._ir.symbolic,
                **{k + self.num_paulis: v for k, v in other_ir.symbolic.items()},
            },
        )
        return self._rebuild(combined)

    def __neg__(self) -> "QuantumOperatorBase":
        """Return the operator with every coefficient negated."""
        return self._rebuild(self._ir.scaled(-1.0))

    def __sub__(self, other: Any) -> "QuantumOperatorBase":
        """Return the difference with another operator, as a new operator."""
        if not isinstance(other, QuantumOperatorBase):
            return NotImplemented
        return self + (-other)

    def __mul__(self, factor: Any) -> "QuantumOperatorBase":
        """Return the operator scaled by a number or SymPy expression."""
        if isinstance(factor, QuantumOperatorBase):
            return NotImplemented
        return self._rebuild(self._ir.scaled(factor))

    __rmul__ = __mul__

    def __truediv__(self, divisor: Any) -> "QuantumOperatorBase":
        """Return the operator divided by a number or SymPy expression."""
        if isinstance(divisor, QuantumOperatorBase):
            return NotImplemented
        return self._rebuild(self._ir.scaled(1 / translate_expression(divisor)))

    def __matmul__(self, other: Any) -> "QuantumOperatorBase":
        """Return the matrix product ``self · other`` (same as :meth:`compose`)."""
        if not isinstance(other, QuantumOperatorBase):
            return NotImplemented
        return self.compose(other)

    def group_commuting(self, qubit_wise: bool = False) -> List["QuantumOperatorBase"]:
        """Split the operator into groups of mutually commuting terms.

        Args:
            qubit_wise: Require commutation on every qubit individually.
        """
        return [self._rebuild(group) for group in self._ir.group_commuting(qubit_wise)]

    @classmethod
    def sum(cls, operators: Sequence["QuantumOperatorBase"]) -> "QuantumOperatorBase":
        """Return the sum of several operators in one step.

        Stacks every term once instead of chaining ``+``, then merges
        duplicates.

        Args:
            operators: Operators of equal width, at least one.

        Returns:
            The simplified sum, in this class.
        """
        return cls(_ir=PauliIR._concatenate([op.ir for op in operators]).simplify(atol=0.0))

    @classmethod
    def from_symplectic(
        cls,
        z: np.ndarray,
        x: np.ndarray,
        coeffs: np.ndarray,
        num_qubits: "int | None" = None,
    ) -> "QuantumOperatorBase":
        """Build an operator from symplectic ``z``/``x`` matrices and coefficients.

        Args:
            z: Boolean ``(num_terms, num_qubits)`` Z components.
            x: Boolean ``(num_terms, num_qubits)`` X components.
            coeffs: One numeric coefficient per term.
            num_qubits: Width, inferred from the matrices when omitted.

        Returns:
            The operator.
        """
        z = np.asarray(z, dtype=bool)
        width = z.shape[1] if num_qubits is None else num_qubits
        return cls(_ir=PauliIR(width, z, np.asarray(x, dtype=bool), np.asarray(coeffs)))

    @classmethod
    def identity(cls, num_qubits: int, coeff: Any = 1.0) -> "QuantumOperatorBase":
        """Return ``coeff`` times the identity on ``num_qubits`` qubits."""
        return cls(["I" * num_qubits], [coeff], num_qubits)

    def assign_parameters(self, parameters: Mapping[Any, float]) -> "QuantumOperatorBase":
        """Return this operator with parameter values substituted.

        Args:
            parameters: Values keyed by
                :class:`~qc_executor.parameters.Parameter` or by name.

        Returns:
            The bound operator.
        """
        binding = {
            (key if isinstance(key, Parameter) else Parameter(str(key))): value
            for key, value in parameters.items()
        }
        return self._rebuild(self._ir.substitute(binding))

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def fingerprint(self) -> bytes:
        """Return a stable digest of the operator's content."""
        return self._ir.fingerprint()

    def __len__(self) -> int:
        return self._ir.num_terms

    def __getitem__(self, index: Any) -> "QuantumOperatorBase":
        """Return one term (integer index) or a range of terms (slice) as an operator."""
        rows = np.atleast_1d(np.arange(self._ir.num_terms)[index])
        return self._rebuild(
            PauliIR(
                self.num_qubits,
                self._ir.z[rows],
                self._ir.x[rows],
                self._ir.coeffs_array[rows],
                {
                    new: self._ir.symbolic[old]
                    for new, old in enumerate(rows)
                    if old in self._ir.symbolic
                },
            )
        )

    def __iter__(self):
        """Iterate over the terms as single-term operators of this type."""
        for index in range(self._ir.num_terms):
            yield self._rebuild(
                PauliIR(
                    self.num_qubits,
                    self._ir.z[index : index + 1],
                    self._ir.x[index : index + 1],
                    self._ir.coeffs_array[index : index + 1],
                    {0: self._ir.symbolic[index]} if index in self._ir.symbolic else None,
                )
            )

    def __radd__(self, other: Any) -> "QuantumOperatorBase":
        """Support ``sum()``: ``0 + operator`` is the operator itself."""
        if isinstance(other, Number) and other == 0:
            return self.copy()
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._ir.fingerprint())

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, QuantumOperatorBase) and self._ir == other.ir

    def __str__(self) -> str:
        terms = []
        for label, coeff in zip(self.paulis, self.coeffs):
            if isinstance(coeff, sp.Basic):
                terms.append(f"({coeff}) * {label}")
            else:
                terms.append(f"({complex(coeff):g}) * {label}")
        return " + ".join(terms) if terms else "0"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(num_qubits={self.num_qubits}, "
            f"num_paulis={self.num_paulis})"
        )
