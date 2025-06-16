import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union

from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.circuit import ParameterExpression, Parameter

from qiskit.quantum_info import SparsePauliOp

from .base import QuantumOperatorBase

from .utils.qiskit_hash_functions import _observable_key


class QuantumOperator(QuantumOperatorBase):

    def __init__(
        self, paulis: List[str] = None, coeffs: List[float] = None, num_qubits: int = None
    ):

        if paulis is not None:
            self._qiskit_operator = SparsePauliOp(paulis, coeffs=coeffs)
        elif num_qubits is not None:
            self._qiskit_operator = SparsePauliOp.from_list([("I" * num_qubits, 0.0)])

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits in the circuit."""
        return self._qiskit_operator.num_qubits

    @property
    def num_paulis(self) -> int:
        """Return the number of Paulis in the operator."""
        return len(self.paulis)

    @property
    def paulis(self) -> List[str]:
        """Return the list of Paulis."""
        return self._qiskit_operator.paulis.tolist()

    @property
    def coeffs(self) -> List:
        """Return the list of coefficients."""
        return self._qiskit_operator.coeffs.tolist()

    @property
    def is_parametrized(self) -> bool:
        """Return True if the operator is parametrized."""
        return len(self.parameters) > 0

    @property
    def parameters(self) -> List[Union[Parameter, ParameterExpression]]:
        """
        Return the parameters of the operator.

        Returns:
            List of parameters.
        """
        raise self._qiskit_operator.parameters

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
        return_value = self.__class__()
        return_value._qiskit_operator = self._qiskit_operator.copy()
        return return_value

    def adjoint(self) -> "QuantumOperatorBase":
        """
        Return the adjoint of the operator.

        Returns:
            Adjoint of the operator.
        """
        return_value = self.copy()
        return_value._qiskit_operator = self._qiskit_operator.adjoint()
        return return_value

    def apply_layout(self, layout: dict) -> "QuantumOperatorBase":
        """
        Apply a layout to the operator.

        Args:
            layout (List[int]): Layout to apply.

        Returns:
            Operator with applied layout.
        """
        return_value = self.copy()
        return_value._qiskit_operator = self._qiskit_operator.apply_layout(layout)
        return return_value

    def compose(self, other: "QuantumOperatorBase") -> "QuantumOperatorBase":
        """
        Compose the operator with another operator.

        Args:
            other (QuantumOperatorBase): Operator to compose with.

        Returns:
            Composed operator.
        """

        self._qiskit_operator = self._qiskit_operator.compose(other._qiskit_operator)

    def append(self, pauli: str, coeff=None) -> None:
        """
        Append a Pauli operator with a coefficient to the operator.

        Args:
            pauli (str): Pauli operator to append.
            coeff (float): Coefficient of the Pauli operator.
        """
        if coeff is None:
            coeff = 1.0

        self._qiskit_operator  = SparsePauliOp.from_list(
            self._qiskit_operator.to_list() + [(pauli, coeff)]
        )

    def simplify(self) -> "QuantumOperatorBase":
        """
        Simplify the operator.

        Returns:
            Simplified operator.
        """
        return_value = self.copy()
        return_value._qiskit_operator = self._qiskit_operator.simplify()
        return return_value

    def transpose(self) -> "QuantumOperatorBase":
        """
        Return the transpose of the operator.

        Returns:
            Transpose of the operator.
        """
        return_value = self.copy()
        return_value._qiskit_operator = self._qiskit_operator.transpose()
        return return_value

    def conjugate(self) -> "QuantumOperatorBase":
        """
        Return the conjugate of the operator.

        Returns:
            Conjugate of the operator.
        """
        return_value = self.copy()
        return_value._qiskit_operator = self._qiskit_operator.conjugate()
        return return_value

    def group_commuting(self) -> List["QuantumOperatorBase"]:
        """
        Group commuting operators.

        Returns:
            List of commuting operators.
        """
        commuting_op = self._qiskit_operator.group_commuting()

        return [self.__class__(paulis=op.paulis, coeffs=op.coeffs) for op in commuting_op]

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
        return isinstance(other, QuantumOperator) and self._qiskit_operator == other._qiskit_operator

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
