"""Base classes for quantum operators across different quantum frameworks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class QuantumOperatorBase(ABC):
    """
    Base class for quantum circuits for different quantum frameworks.

    Args:
        num_qubits (int): Number of qubits in the circuit
    """

    def __init__(
        self,
        num_qubits: Optional[int] = None,
        paulis: Optional[List[str]] = None,
        coeffs: Optional[List[float]] = None,
    ):
        self._num_qubits = num_qubits
        self._paulis = paulis or []
        self._coeffs = coeffs or []

    @classmethod
    @abstractmethod
    def from_quantum_operator(cls, operator: "QuantumOperatorBase") -> "QuantumOperatorBase":
        """Create a backend-native operator from a generic quantum operator."""
        raise NotImplementedError

    @property
    def num_qubits(self) -> int | None:
        """Return the number of qubits in the circuit."""
        return self._num_qubits

    @property
    def num_paulis(self) -> int:
        """Return the number of Paulis in the operator."""
        return len(self._paulis)

    @property
    def paulis(self) -> List[str]:
        """Return the list of Paulis (a copy; mutating it does not affect the operator)."""
        return list(self._paulis)

    @property
    def coeffs(self) -> List[float]:
        """Return the list of coefficients (a copy; mutating it does not affect the operator)."""
        return list(self._coeffs)

    @property
    def is_parametrized(self) -> bool:
        """Return True if the operator is parametrized."""
        raise NotImplementedError

    @property
    def parameters(self) -> List[Any]:
        """
        Return the free parameters of the operator.

        The element type is backend-native: the abstraction returns its own
        SymPy-based ``Parameter`` objects, the Qiskit backend returns Qiskit
        parameter elements. Code working across backends must not assume a
        specific class.

        Returns:
            List of backend-native parameter objects.
        """
        raise NotImplementedError

    @property
    def num_parameters(self) -> int:
        """
        Return the number of parameters in the operator.

        Returns:
            Number of parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def compose(self, other: "QuantumOperatorBase") -> "QuantumOperatorBase":
        """
        Compose the operator with another operator.

        Args:
            other (QuantumOperatorBase): Operator to compose with.

        Returns:
            New composed operator; ``self`` is left unchanged.
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
