"""The quantum operator interface shared by the generic and native operators.

Like :class:`~qc_executor.base.circuit_base.QuantumCircuitBase`, this class is
concrete: it owns the sparse Pauli representation and implements the whole
operator API on top of it.  A backend subclass supplies only how to compile that
representation into its native form.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, List, Mapping, Sequence

import numpy as np
import sympy as sp

from ..parameters import Parameter, sort_parameters
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

    def simplify(self) -> "QuantumOperatorBase":
        """Combine duplicate terms and drop those with zero coefficient."""
        return self._rebuild(self._ir.simplify())

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

    def group_commuting(self) -> List["QuantumOperatorBase"]:
        """Split the operator into groups of mutually commuting terms."""
        return [self._rebuild(group) for group in self._ir.group_commuting()]

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
