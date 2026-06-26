"""Pauli propagation native operator datatype."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple, overload

import numpy as np
import sympy as sp

from qc_executor.base.operator_base import QuantumOperatorBase

from .symmetry import CompositeSymmetry, NoSymmetry
from .utils.pauli_algebra import (
    get_pauli,
    pauli_multiply,
    set_pauli,
    string_to_term,
    term_to_string,
)
from .utils.pauli_types import PauliSum

if TYPE_CHECKING:
    from .symmetry import SymmetryStrategy


class PauliPropagationOperator(QuantumOperatorBase):
    """Backend-native operator representation for Pauli propagation."""

    @overload
    @classmethod
    def from_quantum_operator(
        cls, operator: QuantumOperatorBase
    ) -> "PauliPropagationOperator": ...

    @overload
    @classmethod
    def from_quantum_operator(  # pylint: disable=arguments-differ
        cls,
        operator: QuantumOperatorBase,
        symmetry_strategy: SymmetryStrategy,
    ) -> "PauliPropagationOperator": ...

    @classmethod
    def from_quantum_operator(
        cls,
        operator: QuantumOperatorBase,
        symmetry_strategy: SymmetryStrategy | None = None,
    ) -> "PauliPropagationOperator":  # type: ignore[override]
        """Create a PauliPropagationOperator from a generic operator."""
        if isinstance(operator, cls):
            result = operator.copy()
            if symmetry_strategy is not None:
                result.symmetry = symmetry_strategy
            return result

        try:
            paulis = operator.paulis
            coeffs = [c.sympify() if hasattr(c, "sympify") else c for c in operator.coeffs]
            return cls(paulis=paulis, coeffs=coeffs, symmetry_strategy=symmetry_strategy)
        except (AttributeError, TypeError) as exc:
            raise TypeError(
                "PauliPropagationOperator.from_quantum_operator expects a generic "
                f"QuantumOperator or {cls.__name__}, got {type(operator).__name__}"
            ) from exc

    def __init__(
        self,
        paulis: List[str] | None = None,
        coeffs: List[complex | "sp.Expr"] | None = None,
        num_qubits: int | None = None,
        pauli_sum: PauliSum | None = None,
        symmetry_strategy: SymmetryStrategy | None = None,
        *,
        parametric_coeffs: Dict[int, sp.Expr] | None = None,
        parameter_symbols: Dict[str, sp.Symbol] | None = None,
    ):
        # Track symbolic coefficients separately
        self._parametric_coeffs: Dict[int, sp.Expr] = {}  # Maps term -> symbolic expr
        self._parameters: Dict[str, sp.Symbol] = {}  # Maps param name -> symbol

        if pauli_sum is not None:
            self._pauli_sum = pauli_sum.copy()
            if symmetry_strategy is not None:
                self._pauli_sum.symmetry = symmetry_strategy
            self._num_qubits = self._pauli_sum.nqubits
        elif paulis is not None:
            if len(paulis) == 0:
                if num_qubits is None:
                    raise ValueError("num_qubits is required when paulis is empty.")
                self._num_qubits = num_qubits
                self._pauli_sum = PauliSum(num_qubits, symmetry=symmetry_strategy)
            else:
                inferred_qubits = len(paulis[0])
                self._num_qubits = num_qubits if num_qubits is not None else inferred_qubits
                self._pauli_sum = PauliSum(self._num_qubits, symmetry=symmetry_strategy)

                coeff_values = coeffs if coeffs is not None else [1.0] * len(paulis)
                if len(coeff_values) != len(paulis):
                    raise ValueError("Length of coeffs must match length of paulis.")

                for pauli, coeff in zip(paulis, coeff_values):
                    # Normalize coefficients to sympy for robust symbolic handling
                    # across sympy, symengine and Qiskit parameter expression types.
                    if hasattr(coeff, "sympify"):
                        coeff_expr = sp.sympify(coeff.sympify())
                    else:
                        coeff_expr = sp.sympify(coeff)

                    # Check if coefficient is symbolic
                    if isinstance(coeff_expr, sp.Expr) and len(coeff_expr.free_symbols) > 0:
                        term = string_to_term(pauli, self._num_qubits)
                        self._parametric_coeffs[term] = coeff_expr
                        # Track parameters
                        for symbol in coeff_expr.free_symbols:
                            self._parameters[symbol.name] = symbol
                        # Add with coefficient 1.0 as placeholder
                        self._pauli_sum.add_term(pauli, 1.0)
                    else:
                        # Numeric coefficient
                        self._pauli_sum.add_term(pauli, complex(coeff_expr))
        elif num_qubits is not None:
            self._num_qubits = num_qubits
            self._pauli_sum = PauliSum(num_qubits, symmetry=symmetry_strategy)
        else:
            raise ValueError("Provide either paulis, num_qubits, or pauli_sum.")

        # Explicit parameter tracking, intended for use together with pauli_sum
        # (e.g. when deriving a new operator from an existing one).
        if parametric_coeffs is not None:
            self._parametric_coeffs = dict(parametric_coeffs)
        if parameter_symbols is not None:
            self._parameters = dict(parameter_symbols)

        super().__init__(num_qubits=self._num_qubits)

    @property
    def pauli_sum(self) -> PauliSum:
        """Return a copy of the underlying PauliSum."""
        return self._pauli_sum.copy()

    @property
    def symmetry(self):
        """Return the symmetry strategy of the underlying PauliSum."""
        return self._pauli_sum.symmetry

    @symmetry.setter
    def symmetry(self, strategy) -> None:
        self._pauli_sum.symmetry = strategy

    @property
    def has_active_symmetry(self) -> bool:
        """Return True if a non-trivial symmetry strategy is active."""
        return self._pauli_sum.has_active_symmetry

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def num_paulis(self) -> int:
        return len(self._pauli_sum)

    @property
    def paulis(self) -> List[str]:
        return [term_to_string(term, self._num_qubits) for term, _ in self._pauli_sum]

    @property
    def coeffs(self) -> List[complex]:
        return [coeff for _, coeff in self._pauli_sum]

    @property
    def is_parametrized(self) -> bool:
        return len(self._parametric_coeffs) > 0

    @property
    def parameters(self) -> List[str]:
        return list(self._parameters.keys())

    @property
    def parameter_symbols(self) -> Dict[str, sp.Symbol]:
        """Return a copy of the parameter-name to sympy-symbol mapping."""
        return dict(self._parameters)

    @property
    def parametric_coeffs(self) -> Dict[int, sp.Expr]:
        """Return a copy of the term to symbolic-coefficient mapping."""
        return dict(self._parametric_coeffs)

    @property
    def num_parameters(self) -> int:
        return len(self._parameters)

    def copy(self) -> "PauliPropagationOperator":
        """Return a deep copy of this operator."""
        return PauliPropagationOperator(
            pauli_sum=self._pauli_sum,
            parametric_coeffs=self._parametric_coeffs,
            parameter_symbols=self._parameters,
        )

    def assign_parameters(self, parameters: Dict[str, float]) -> "PauliPropagationOperator":
        """Bind symbolic parameters to concrete values.

        Args:
            parameters: Dict mapping parameter names to float values

        Returns:
            New operator with parameters substituted
        """
        # Build substitution dict for sympy
        subs_dict = {}
        for param_name, param_value in parameters.items():
            if param_name in self._parameters:
                subs_dict[self._parameters[param_name]] = param_value

        # Substitute in parametric coefficients
        new_pauli_sum = PauliSum(self._num_qubits, symmetry=self._pauli_sum.symmetry)
        new_parametric_coeffs = {}

        for term, coeff in self._pauli_sum:
            if term in self._parametric_coeffs:
                # This term has a symbolic coefficient
                symbolic_coeff = self._parametric_coeffs[term]
                substituted_coeff = symbolic_coeff.subs(subs_dict)

                if substituted_coeff.is_number:
                    # Fully evaluated, add to PauliSum with concrete value
                    new_pauli_sum.add_term(term, complex(substituted_coeff))
                else:
                    # Partially evaluated, keep as parametric
                    new_parametric_coeffs[term] = substituted_coeff
                    new_pauli_sum.add_term(term, 1.0)  # Placeholder
            else:
                # Non-parametric term, just copy
                new_pauli_sum.add_term(term, coeff)

        # Update parameter tracking - remove fully bound parameters
        remaining_params = {
            name: symbol for name, symbol in self._parameters.items() if name not in parameters
        }

        return PauliPropagationOperator(
            pauli_sum=new_pauli_sum,
            parametric_coeffs=new_parametric_coeffs,
            parameter_symbols=remaining_params,
        )

    def adjoint(self) -> "PauliPropagationOperator":
        conjugated = PauliSum(self._num_qubits, symmetry=self._pauli_sum.symmetry)
        for term, coeff in self._pauli_sum:
            conjugated.add_term(term, np.conjugate(coeff))
        return PauliPropagationOperator(
            pauli_sum=conjugated,
            parametric_coeffs=self._parametric_coeffs,
            parameter_symbols=self._parameters,
        )

    def apply_layout(self, layout: Dict[int, int]) -> "PauliPropagationOperator":
        remapped = PauliSum(self._num_qubits, symmetry=self._pauli_sum.symmetry)
        term_mapping: Dict[int, int] = {}

        for term, coeff in self._pauli_sum:
            remapped_term = 0
            for source_idx in range(self._num_qubits):
                symbol = get_pauli(term, source_idx, self._num_qubits)
                target_idx = layout.get(source_idx, source_idx)
                remapped_term = set_pauli(remapped_term, target_idx, symbol, self._num_qubits)

            term_mapping[term] = remapped_term
            remapped.add_term(remapped_term, coeff)

        remapped_coeffs = {
            term_mapping[old_term]: expr
            for old_term, expr in self._parametric_coeffs.items()
            if old_term in term_mapping
        }
        return PauliPropagationOperator(
            pauli_sum=remapped,
            parametric_coeffs=remapped_coeffs,
            parameter_symbols=self._parameters,
        )

    def compose(self, other: "QuantumOperatorBase") -> "PauliPropagationOperator":
        if not isinstance(other, PauliPropagationOperator):
            raise TypeError("compose currently supports PauliPropagationOperator only.")
        if self.num_qubits != other.num_qubits:
            raise ValueError("Cannot compose operators with different qubit counts.")

        composed_symmetry = self._compose_symmetry_with(other)
        composed_sum = PauliSum(self.num_qubits, symmetry=composed_symmetry)
        other_sum = other.pauli_sum
        for left_term, left_coeff in self._pauli_sum:
            for right_term, right_coeff in other_sum:
                result_term, phase = pauli_multiply(left_term, right_term, self.num_qubits)
                composed_sum.add_term(result_term, left_coeff * right_coeff * phase)

        return PauliPropagationOperator(pauli_sum=composed_sum)

    def append(self, pauli: str, coeff=None) -> "PauliPropagationOperator":
        new_sum = self._pauli_sum.copy()
        new_sum.add_term(pauli, 1.0 if coeff is None else coeff)
        return PauliPropagationOperator(
            pauli_sum=new_sum,
            parametric_coeffs=self._parametric_coeffs,
            parameter_symbols=self._parameters,
        )

    def simplify(self) -> "PauliPropagationOperator":
        # Terms are combined on insertion in PauliSum; copy is already simplified.
        return self.copy()

    def transpose(self) -> "PauliPropagationOperator":
        transposed = PauliSum(self.num_qubits, symmetry=self._pauli_sum.symmetry)
        for term, coeff in self._pauli_sum:
            pauli = term_to_string(term, self.num_qubits)
            y_count = pauli.count("Y")
            phase = -1 if y_count % 2 else 1
            transposed.add_term(term, coeff * phase)
        return PauliPropagationOperator(pauli_sum=transposed)

    def conjugate(self) -> "PauliPropagationOperator":
        conjugated = PauliSum(self.num_qubits, symmetry=self._pauli_sum.symmetry)
        for term, coeff in self._pauli_sum:
            conjugated.add_term(term, np.conjugate(coeff))
        return PauliPropagationOperator(pauli_sum=conjugated)

    def group_commuting(self) -> List["PauliPropagationOperator"]:
        # Minimal implementation for interface parity.
        return [self.copy()]

    def _compose_symmetry_with(self, other: "PauliPropagationOperator"):
        self_symmetry = self.symmetry
        other_symmetry = other.symmetry

        self_active = not isinstance(self_symmetry, NoSymmetry)
        other_active = not isinstance(other_symmetry, NoSymmetry)

        if self_active and other_active:
            if self_symmetry.name == other_symmetry.name:
                return self_symmetry
            return CompositeSymmetry(self_symmetry, other_symmetry)
        if self_active:
            return self_symmetry
        if other_active:
            return other_symmetry
        return NoSymmetry()

    @property
    def is_unitary(self) -> bool:
        if len(self._pauli_sum) != 1:
            return False
        _, coeff = next(iter(self._pauli_sum))
        return np.isclose(abs(coeff), 1.0)

    @property
    def is_real(self) -> bool:
        return all(np.isclose(np.imag(coeff), 0.0) for _, coeff in self._pauli_sum)

    @property
    def is_imaginary(self) -> bool:
        return all(np.isclose(np.real(coeff), 0.0) for _, coeff in self._pauli_sum)

    def _canonical_signature(self) -> Tuple:
        """Canonical signature used for hashing and equality comparisons.

        Includes the number of qubits, Pauli terms, symmetry name, and any
        parametric coefficient expressions (if present).
        """
        terms_signature = tuple(sorted(self._pauli_sum.terms.items()))
        symmetry_signature = self.symmetry.name if self.symmetry is not None else None
        parametric_signature = (
            tuple(sorted((k, str(v)) for k, v in self._parametric_coeffs.items()))
            if self._parametric_coeffs
            else ()
        )
        return (self.num_qubits, terms_signature, symmetry_signature, parametric_signature)

    def __hash__(self):
        return hash(self._canonical_signature())

    def __eq__(self, other):
        return (
            isinstance(other, PauliPropagationOperator)
            and self._canonical_signature() == other._canonical_signature()
        )

    def __str__(self):
        return (
            f"PauliPropagationOperator(num_qubits={self.num_qubits}, terms={len(self._pauli_sum)})"
        )

    def __repr__(self):
        return self.__str__()
