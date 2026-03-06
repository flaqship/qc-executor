"""Pauli propagation native observable datatype."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Sequence, overload

import numpy as np
import sympy as sp

from executor.base.operator_base import QuantumOperatorBase

from .pauli_algebra import term_to_string
from .pauli_types import PauliSum
from .symmetry import CompositeSymmetry, NoSymmetry

if TYPE_CHECKING:
    from .symmetry import SymmetryStrategy


class PauliPropagationObservable(QuantumOperatorBase):
    """Backend-native observable representation for Pauli propagation."""

    @overload
    @classmethod
    def from_quantum_operator(
        cls, operator: QuantumOperatorBase
    ) -> "PauliPropagationObservable": ...

    @overload
    @classmethod
    def from_quantum_operator(
        cls,
        operator: QuantumOperatorBase,
        symmetry_strategy: "SymmetryStrategy",
    ) -> "PauliPropagationObservable": ...

    @classmethod
    def from_quantum_operator(
        cls,
        operator: QuantumOperatorBase,
        symmetry_strategy: "SymmetryStrategy" | None = None,
    ) -> "PauliPropagationObservable":  # type: ignore[override]
        """Create a PauliPropagationObservable from a generic operator."""
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
                "PauliPropagationObservable.from_quantum_operator expects a generic "
                f"QuantumOperator or {cls.__name__}, got {type(operator).__name__}"
            ) from exc

    def __init__(
        self,
        paulis: List[str] | None = None,
        coeffs: List[complex | sp.Expr] | None = None,
        num_qubits: int | None = None,
        pauli_sum: PauliSum | None = None,
        symmetry_strategy: "SymmetryStrategy" | None = None,
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
                    # Check if coefficient is symbolic
                    if isinstance(coeff, sp.Expr) and not coeff.is_number:
                        from .pauli_algebra import string_to_term

                        term = string_to_term(pauli, self._num_qubits)
                        self._parametric_coeffs[term] = coeff
                        # Track parameters
                        for symbol in coeff.free_symbols:
                            self._parameters[symbol.name] = symbol
                        # Add with coefficient 1.0 as placeholder
                        self._pauli_sum.add_term(pauli, 1.0)
                    else:
                        # Numeric coefficient
                        self._pauli_sum.add_term(pauli, complex(coeff))
        elif num_qubits is not None:
            self._num_qubits = num_qubits
            self._pauli_sum = PauliSum(num_qubits, symmetry=symmetry_strategy)
        else:
            raise ValueError("Provide either paulis, num_qubits, or pauli_sum.")

        super().__init__(num_qubits=self._num_qubits)

    @property
    def pauli_sum(self) -> PauliSum:
        return self._pauli_sum.copy()

    @property
    def symmetry(self):
        return self._pauli_sum.symmetry

    @symmetry.setter
    def symmetry(self, strategy) -> None:
        self._pauli_sum.symmetry = strategy

    @property
    def has_active_symmetry(self) -> bool:
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
    def num_parameters(self) -> int:
        return len(self._parameters)

    def copy(self) -> "PauliPropagationObservable":
        result = PauliPropagationObservable(pauli_sum=self._pauli_sum.copy())
        result._parametric_coeffs = dict(self._parametric_coeffs)
        result._parameters = dict(self._parameters)
        return result

    def assign_parameters(self, parameters: Dict[str, float]) -> "PauliPropagationObservable":
        """Bind symbolic parameters to concrete values.

        Args:
            parameters: Dict mapping parameter names to float values

        Returns:
            New observable with parameters substituted
        """
        result = self.copy()

        # Build substitution dict for sympy
        subs_dict = {}
        for param_name, param_value in parameters.items():
            if param_name in result._parameters:
                subs_dict[result._parameters[param_name]] = param_value

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

        result._pauli_sum = new_pauli_sum
        result._parametric_coeffs = new_parametric_coeffs

        # Update parameter tracking - remove fully bound parameters
        remaining_params = {}
        for param_name, param_symbol in result._parameters.items():
            if param_name not in parameters:
                remaining_params[param_name] = param_symbol
        result._parameters = remaining_params

        return result

    def adjoint(self) -> "PauliPropagationObservable":
        result = self.copy()
        conjugated = PauliSum(self._num_qubits, symmetry=result._pauli_sum.symmetry)
        for term, coeff in result._pauli_sum:
            conjugated.add_term(term, np.conjugate(coeff))
        result._pauli_sum = conjugated
        return result

    def apply_layout(self, layout: Dict[int, int]) -> "PauliPropagationObservable":
        remapped = PauliSum(self._num_qubits, symmetry=self._pauli_sum.symmetry)

        for term, coeff in self._pauli_sum:
            term_str = term_to_string(term, self._num_qubits)
            mapped_symbols = ["I"] * self._num_qubits

            for source_idx, symbol in enumerate(term_str):
                target_idx = layout.get(source_idx, source_idx)
                mapped_symbols[target_idx] = symbol

            remapped.add_term("".join(mapped_symbols), coeff)

        return PauliPropagationObservable(pauli_sum=remapped)

    def compose(self, other: "QuantumOperatorBase") -> "PauliPropagationObservable":
        if not isinstance(other, PauliPropagationObservable):
            raise TypeError("compose currently supports PauliPropagationObservable only.")
        if self.num_qubits != other.num_qubits:
            raise ValueError("Cannot compose observables with different qubit counts.")

        composed_symmetry = self._compose_symmetry_with(other)
        composed_sum = PauliSum(self.num_qubits, symmetry=composed_symmetry)
        for left_term, left_coeff in self._pauli_sum:
            for right_term, right_coeff in other._pauli_sum:
                from .pauli_algebra import pauli_multiply

                result_term, phase = pauli_multiply(left_term, right_term, self.num_qubits)
                composed_sum.add_term(result_term, left_coeff * right_coeff * phase)

        return PauliPropagationObservable(pauli_sum=composed_sum)

    def append(self, pauli: str, coeff=None) -> "PauliPropagationObservable":
        result = self.copy()
        result._pauli_sum.add_term(pauli, 1.0 if coeff is None else coeff)
        return result

    def simplify(self) -> "PauliPropagationObservable":
        # Terms are combined on insertion in PauliSum; copy is already simplified.
        return self.copy()

    def transpose(self) -> "PauliPropagationObservable":
        transposed = PauliSum(self.num_qubits, symmetry=self._pauli_sum.symmetry)
        for term, coeff in self._pauli_sum:
            pauli = term_to_string(term, self.num_qubits)
            y_count = pauli.count("Y")
            phase = -1 if y_count % 2 else 1
            transposed.add_term(term, coeff * phase)
        return PauliPropagationObservable(pauli_sum=transposed)

    def conjugate(self) -> "PauliPropagationObservable":
        conjugated = PauliSum(self.num_qubits, symmetry=self._pauli_sum.symmetry)
        for term, coeff in self._pauli_sum:
            conjugated.add_term(term, np.conjugate(coeff))
        return PauliPropagationObservable(pauli_sum=conjugated)

    def group_commuting(self) -> List["PauliPropagationObservable"]:
        # Minimal implementation for interface parity.
        return [self.copy()]

    def _compose_symmetry_with(self, other: "PauliPropagationObservable"):
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

    def __hash__(self):
        signature: Sequence[tuple[int, complex]] = tuple(sorted(self._pauli_sum.terms.items()))
        return hash((self.num_qubits, signature))

    def __eq__(self, other):
        return (
            isinstance(other, PauliPropagationObservable)
            and self._pauli_sum.terms == other._pauli_sum.terms
        )

    def __str__(self):
        return f"PauliPropagationObservable(num_qubits={self.num_qubits}, terms={len(self._pauli_sum)})"

    def __repr__(self):
        return self.__str__()
