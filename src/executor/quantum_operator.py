"""Concrete quantum operator implementation backed by a Qiskit SparsePauliOp."""

from __future__ import annotations

from typing import Any, List, Optional, cast

from qiskit.quantum_info import SparsePauliOp

from .base import QuantumOperatorBase
from .utils.qiskit_hash_functions import _observable_key


class QuantumOperator(QuantumOperatorBase):
    """Quantum operator backed by a Qiskit SparsePauliOp."""

    @classmethod
    def from_quantum_operator(cls, operator: QuantumOperatorBase) -> QuantumOperatorBase:
        """Identity conversion for generic operators."""
        return operator

    def __init__(
        self,
        paulis: Optional[List[str]] = None,
        coeffs: Optional[List[float]] = None,
        num_qubits: Optional[int] = None,
        _native_operator: Optional[SparsePauliOp] = None,
    ):
        super().__init__(num_qubits=num_qubits, paulis=paulis, coeffs=coeffs)
        if _native_operator is not None:
            self._qiskit_operator: SparsePauliOp = _native_operator
        elif paulis is not None:
            self._qiskit_operator = SparsePauliOp(paulis, coeffs=cast(Any, coeffs))
        elif num_qubits is not None:
            self._qiskit_operator = SparsePauliOp.from_list([("I" * num_qubits, 0.0)])
        else:
            raise ValueError("Must provide paulis, num_qubits, or _native_operator")

    @property
    def qiskit_operator(self) -> SparsePauliOp:
        """The underlying Qiskit SparsePauliOp."""
        return self._qiskit_operator

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits in the circuit."""
        return cast(int, self._qiskit_operator.num_qubits)

    @property
    def num_paulis(self) -> int:
        """Return the number of Paulis in the operator."""
        return len(self.paulis)

    @property
    def paulis(self) -> List[str]:
        """Return the list of Paulis."""
        return list(self._qiskit_operator.paulis.to_labels())

    @property
    def coeffs(self) -> List:
        """Return the list of coefficients."""
        return cast(Any, self._qiskit_operator.coeffs).tolist()

    @property
    def is_parametrized(self) -> bool:
        """Return True if the operator is parametrized."""
        return len(self.parameters) > 0

    @property
    def parameters(self) -> list:
        """
        Return the parameters of the operator.

        Returns:
            List of parameters.
        """
        return list(self._qiskit_operator.parameters)

    @property
    def num_parameters(self) -> int:
        """
        Return the number of parameters in the operator.

        Returns:
            Number of parameters.
        """
        return len(self.parameters)

    def copy(self) -> "QuantumOperatorBase":
        """
        Return a copy of the operator.

        Returns:
            Copy of the operator.
        """
        return_value = self.__class__(_native_operator=self._qiskit_operator.copy())
        return return_value

    def adjoint(self) -> "QuantumOperatorBase":
        """
        Return the adjoint of the operator.

        Returns:
            Adjoint of the operator.
        """
        return_value = self.__class__(_native_operator=self._qiskit_operator.adjoint())
        return return_value

    def apply_layout(self, layout: List[int]) -> "QuantumOperatorBase":
        """
        Apply a layout to the operator.

        Args:
            layout (List[int]): Layout to apply.

        Returns:
            Operator with applied layout.
        """
        return_value = self.__class__(_native_operator=self._qiskit_operator.apply_layout(layout))
        return return_value

    def compose(self, other: "QuantumOperatorBase") -> "QuantumOperatorBase":
        """
        Compose the operator with another operator.

        Args:
            other (QuantumOperatorBase): Operator to compose with.

        Returns:
            Composed operator.
        """

        if not isinstance(other, QuantumOperator):
            raise TypeError("Can only compose with a QuantumOperator instance")
        self._qiskit_operator = self._qiskit_operator.compose(other.qiskit_operator)
        return self

    def append(self, pauli: str, coeff=None) -> "QuantumOperatorBase":
        """
        Append a Pauli operator with a coefficient to the operator.

        Args:
            pauli (str): Pauli operator to append.
            coeff (float): Coefficient of the Pauli operator.
        """
        if coeff is None:
            coeff = 1.0

        self._qiskit_operator = SparsePauliOp.from_list(
            self._qiskit_operator.to_list() + [(pauli, coeff)]
        )
        return self

    def simplify(self) -> "QuantumOperatorBase":
        """
        Simplify the operator.

        Returns:
            Simplified operator.
        """
        return_value = self.__class__(_native_operator=self._qiskit_operator.simplify())
        return return_value

    def transpose(self) -> "QuantumOperatorBase":
        """
        Return the transpose of the operator.

        Returns:
            Transpose of the operator.
        """
        return_value = self.__class__(_native_operator=self._qiskit_operator.transpose())
        return return_value

    def conjugate(self) -> "QuantumOperatorBase":
        """
        Return the conjugate of the operator.

        Returns:
            Conjugate of the operator.
        """
        return_value = self.__class__(_native_operator=self._qiskit_operator.conjugate())
        return return_value

    def group_commuting(self) -> List["QuantumOperatorBase"]:
        """
        Group commuting operators.

        Returns:
            List of commuting operators.
        """
        commuting_op = self._qiskit_operator.group_commuting()
        return [self.__class__(_native_operator=operator) for operator in commuting_op]

    @property
    def is_unitary(self) -> bool:
        """
        Return True if the operator is unitary.

        Returns:
            True if the operator is unitary.
        """
        return self._qiskit_operator.is_unitary()

    @property
    def is_real(self) -> bool:
        """
        Return True if the operator is real.

        Returns:
            True if the operator is real.
        """
        raise NotImplementedError

    @property
    def is_imaginary(self) -> bool:
        """
        Return True if the operator is imaginary.

        Returns:
            True if the operator is imaginary.
        """
        raise NotImplementedError

    def __hash__(self):
        return hash(_observable_key(self._qiskit_operator))

    def __eq__(self, other):
        return (
            isinstance(other, QuantumOperator) and self._qiskit_operator == other._qiskit_operator
        )

    def __str__(self):
        """
        Return the string representation of the operator.

        Returns:
            String representation of the operator.
        """
        return str(self._qiskit_operator)

    def __repr__(self):
        """
        Return the string representation of the operator.

        Returns:
            String representation of the operator.
        """
        return str(self._qiskit_operator)
