from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from qiskit.circuit import Parameter, ParameterExpression


class QuantumOperatorBase(ABC):
    """
    Base class for quantum circuits for different quantum frameworks.

    Args:
        num_qubits (int): Number of qubits in the circuit
    """

    def __init__(
        self, num_qubits: int = None, paulis: List[str] = None, coeffs: List[float] = None
    ):
        self._num_qubits = num_qubits

    @classmethod
    @abstractmethod
    def from_quantum_operator(cls, operator: "QuantumOperatorBase") -> "QuantumOperatorBase":
        """Create a backend-native operator from a generic quantum operator."""
        raise NotImplementedError

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits in the circuit."""
        return self._num_qubits

    @property
    def num_paulis(self) -> int:
        """Return the number of Paulis in the operator."""
        return len(self._paulis)

    @property
    def paulis(self) -> List[str]:
        """Return the list of Paulis."""
        return self._paulis

    @property
    def coeffs(self) -> List[float]:
        """Return the list of coefficients."""
        return self._coeffs

    @property
    def is_parametrized(self) -> bool:
        """Return True if the operator is parametrized."""
        raise NotImplementedError

    @property
    def parameters(self) -> List[Parameter | ParameterExpression]:
        """
        Return the parameters of the operator.

        Returns:
            List of parameters.
        """
        raise NotImplementedError

    @property
    def num_parameters(self) -> int:
        """
        Return the number of parameters in the operator.

        Returns:
            Number of parameters.
        """
        return NotImplementedError

    @abstractmethod
    def adjoint(self) -> "QuantumOperatorBase":
        """
        Return the adjoint of the operator.

        Returns:
            Adjoint of the operator.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_layout(self, layout: dict) -> "QuantumOperatorBase":
        """
        Apply a layout to the operator.

        Args:
            layout (List[int]): Layout to apply.

        Returns:
            Operator with applied layout.
        """
        raise NotImplementedError

    @abstractmethod
    def compose(self, other: "QuantumOperatorBase") -> "QuantumOperatorBase":
        """
        Compose the operator with another operator.

        Args:
            other (QuantumOperatorBase): Operator to compose with.

        Returns:
            Composed operator.
        """
        raise NotImplementedError

    @abstractmethod
    def append(self, pauli: str, coeff) -> "QuantumOperatorBase":
        """
        Append another operator to the current operator.

        Args:
            other (QuantumOperatorBase): Operator to append.

        Returns:
            Appended operator.
        """
        raise NotImplementedError

    @abstractmethod
    def simplify(self) -> "QuantumOperatorBase":
        """
        Simplify the operator.

        Returns:
            Simplified operator.
        """
        raise NotImplementedError

    @abstractmethod
    def transpose(self) -> "QuantumOperatorBase":
        """
        Return the transpose of the operator.

        Returns:
            Transpose of the operator.
        """
        raise NotImplementedError

    @abstractmethod
    def conjugate(self) -> "QuantumOperatorBase":
        """
        Return the conjugate of the operator.

        Returns:
            Conjugate of the operator.
        """
        raise NotImplementedError

    @abstractmethod
    def group_commuting(self) -> List["QuantumOperatorBase"]:
        """
        Group commuting operators.

        Returns:
            List of commuting operators.
        """
        raise NotImplementedError

    @property
    def is_unitary(self) -> bool:
        """
        Return True if the operator is unitary.

        Returns:
            True if the operator is unitary.
        """
        raise NotImplementedError

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
        raise NotImplementedError(
            "Hashing is not implemented for this class. "
            "Please implement the __hash__ method in the subclass."
        )

    def __eq__(self, other):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError
