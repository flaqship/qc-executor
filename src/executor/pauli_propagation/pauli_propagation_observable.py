"""Pauli propagation native observable datatype."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Sequence

import numpy as np

from executor.base.operator_base import QuantumOperatorBase

from .pauli_algebra import term_to_string
from .pauli_types import PauliSum
from .symmetry import CompositeSymmetry, NoSymmetry

if TYPE_CHECKING:
    from .symmetry import SymmetryStrategy


class PauliPropagationObservable(QuantumOperatorBase):
    """Backend-native observable representation for Pauli propagation."""

    def __init__(
        self,
        paulis: List[str] | None = None,
        coeffs: List[complex] | None = None,
        num_qubits: int | None = None,
        pauli_sum: PauliSum | None = None,
        symmetry_strategy: "SymmetryStrategy" | None = None,
    ):
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
                    self._pauli_sum.add_term(pauli, coeff)
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
        return False

    @property
    def parameters(self) -> List:
        return []

    @property
    def num_parameters(self) -> int:
        return 0

    def copy(self) -> "PauliPropagationObservable":
        return PauliPropagationObservable(pauli_sum=self._pauli_sum.copy())

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
