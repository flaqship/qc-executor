"""Base classes for quantum circuits across different quantum frameworks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from qiskit.circuit import ParameterExpression

from qc_executor.circuit_idendity_mixin import CircuitIdentityMixin
from qc_executor.parameters import Parameter

from .operator_base import QuantumOperatorBase


class QuantumCircuitBase(ABC, CircuitIdentityMixin):
    """
    Base class for quantum circuits for different quantum frameworks.

    Args:
        num_qubits (int): Number of qubits in the circuit
    """

    def __init__(self, num_qubits: int):
        super().__init__()
        self._num_qubits = num_qubits
        self._free_parameters = set()

    @classmethod
    @abstractmethod
    def from_quantum_circuit(cls, circuit: "QuantumCircuitBase") -> "QuantumCircuitBase":
        """Create a backend-native circuit from a generic quantum circuit."""
        raise NotImplementedError

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits in the circuit."""
        return self._num_qubits

    @property
    def parameters(self) -> List[Parameter]:
        """Return the free trainable parameters in the circuit."""
        return sorted(list(self._free_parameters), key=lambda x: x.index)

    @property
    def num_parameters(self) -> int:
        """Return the number of free trainable parameters in the circuit."""
        return len(self.parameters)

    @property
    def is_parameterized(self) -> bool:
        """Check if the wavefunction is parameterized."""
        return len(self.parameters) > 0

    @abstractmethod
    def draw(self) -> str:
        """Returns printable string representation of the circuit."""
        raise NotImplementedError

    @abstractmethod
    def h(self, qubits: int | List[int]):
        """Add hadamard gates"""
        raise NotImplementedError

    @abstractmethod
    def s(self, qubits: int | List[int]):
        """Add S gates"""
        raise NotImplementedError

    @abstractmethod
    def sdag(self, qubits: int | List[int]):
        """Add Sdag gates"""
        raise NotImplementedError

    @abstractmethod
    def t(self, qubits: int | List[int]):
        """Add T gates"""
        raise NotImplementedError

    @abstractmethod
    def tdag(self, qubits: int | List[int]):
        """Add Tdg gates"""
        raise NotImplementedError

    @abstractmethod
    def p(self, qubits: int | List[int], angle: float):
        """Add P gates"""
        raise NotImplementedError

    def cp(self, control_qubit: int, target_qubit: int, angle: float):
        """Add CP gates"""
        raise NotImplementedError

    @abstractmethod
    def x(self, qubits: int | List[int]):
        """Add X gates"""
        raise NotImplementedError

    @abstractmethod
    def y(self, qubits: int | List[int]):
        """Add Y gates"""
        raise NotImplementedError

    @abstractmethod
    def z(self, qubits: int | List[int]):
        """Add Z gates"""
        raise NotImplementedError

    @abstractmethod
    def rx(self, qubits: int | List[int], angle: float):
        """Add RX gates"""
        raise NotImplementedError

    @abstractmethod
    def ry(self, qubits: int | List[int], angle: float):
        """Add RY gates"""
        raise NotImplementedError

    @abstractmethod
    def rz(self, qubits: int | List[int], angle: float):
        """Add RZ gates"""
        raise NotImplementedError

    @abstractmethod
    def cx(self, control_qubit: int, target_qubit: int):
        """Add CNOT gates"""
        raise NotImplementedError

    @abstractmethod
    def cy(self, control_qubit: int, target_qubit: int):
        """Add CY gates"""
        raise NotImplementedError

    @abstractmethod
    def cz(self, control_qubit: int, target_qubit: int):
        """Add CZ gates"""
        raise NotImplementedError

    def cnot(self, control_qubit: int, target_qubit: int):
        """Add CNOT gates"""
        self.cx(control_qubit, target_qubit)

    def ecr(self, control_qubit: int, target_qubit: int):
        """Add ECR gates"""
        raise NotImplementedError

    @abstractmethod
    def crx(self, control_qubit: int, target_qubit: int, angle: float):
        """Add CRX gates"""
        raise NotImplementedError

    @abstractmethod
    def cry(self, control_qubit: int, target_qubit: int, angle: float):
        """Add CRX gates"""
        raise NotImplementedError

    @abstractmethod
    def crz(self, control_qubit: int, target_qubit: int, angle: float):
        """Add CRX gates"""
        raise NotImplementedError

    @abstractmethod
    def rxx(self, control_qubit: int, target_qubit: int, angle: float):
        """Add RXX gates"""
        raise NotImplementedError

    @abstractmethod
    def ryy(self, control_qubit: int, target_qubit: int, angle: float):
        """Add RYY gates"""
        raise NotImplementedError

    @abstractmethod
    def rzz(self, control_qubit: int, target_qubit: int, angle: float):
        """Add RZZ gates"""
        raise NotImplementedError

    @abstractmethod
    def rzx(self, control_qubit: int, target_qubit: int, angle: float):
        """Add RZX gates"""
        raise NotImplementedError

    @abstractmethod
    def swap(self, qubit1: int, qubit2: int):
        """Add SWAP gates"""
        raise NotImplementedError

    @abstractmethod
    def barrier(self, qubits: int | List[int]):
        """Add barrier gates"""
        raise NotImplementedError

    @abstractmethod
    def measure(self):
        """Add measure gates"""
        raise NotImplementedError

    def pauli_string(self, pauli_string: str) -> None:
        """Apply a Pauli string to the circuit.

        Args:
            pauli_string (str): Pauli string to apply to the circuit

        """
        if len(pauli_string) != self.num_qubits:
            raise ValueError("Pauli string length does not match number of qubits")

        for i, pauli in enumerate(pauli_string[::-1]):
            if pauli == "X":
                self.x(i)
            elif pauli == "Y":
                self.y(i)
            elif pauli == "Z":
                self.z(i)
            elif pauli == "I":
                pass  # Identity gate (I) can be skipped as it does nothing

    def _apply_basis_change(
        self, paulis: List[str], qubits: List[int], working_qubits: List[int]
    ) -> None:
        """Apply basis change for non-trivial Paulis."""
        for p, q in zip(paulis, qubits):
            if p == "X":
                self.h(working_qubits[q])
            elif p == "Y":
                self.sdag(working_qubits[q])
                self.h(working_qubits[q])
            elif p != "Z":
                raise ValueError(f"Unknown Pauli operator: {p}")

    def _undo_basis_change(
        self, paulis: List[str], qubits: List[int], working_qubits: List[int]
    ) -> None:
        """Undo basis change for non-trivial Paulis."""
        for p, q in zip(paulis, qubits):
            if p == "X":
                self.h(working_qubits[q])
            elif p == "Y":
                self.h(working_qubits[q])
                self.s(working_qubits[q])

    def _apply_cnot_ladder(self, qubits: List[int], working_qubits: List[int]) -> None:
        """Apply the forward CNOT ladder for Pauli evolution."""
        if not qubits:
            return
        control = qubits[0]
        for target in qubits[1:]:
            self.cx(working_qubits[control], working_qubits[target])
            control = target

    def _undo_cnot_ladder(self, qubits: List[int], working_qubits: List[int]) -> None:
        """Undo the CNOT ladder after the phase rotation."""
        if not qubits:
            return
        control = qubits[-1]
        for target in reversed(qubits[:-1]):
            self.cx(working_qubits[target], working_qubits[control])
            control = target

    def pauli_evolution(
        self,
        operator: QuantumOperatorBase,
        parameter: Parameter | float,
        working_qubits: List[int] | None = None,
    ) -> None:
        """
        Applies Pauli evolution exp(itP) where P is a Pauli operator.

        Args:
            operator (QuantumOperatorBase): The Pauli operator to evolve.
            parameter (Parameter | float): The evolution parameter.
            working_qubits (List[int]): Optional: the qubits to use as working qubits.
        """

        pauli_str = operator.paulis[0]
        coeff = operator.coeffs
        if len(coeff) != 1:
            raise ValueError("Only operators with single Pauli strings are supported")
        coeff = coeff[0]

        if not isinstance(coeff, (Parameter, ParameterExpression)):
            coeff = np.real_if_close(coeff)
            if np.iscomplexobj(coeff):
                raise ValueError("Complex coefficients are not supported")
        else:
            # the 1j fixes a bug in qiskit
            coeff = -1j * (1j * coeff)

        coeff = coeff * parameter

        qubits = [i for i, p in enumerate(pauli_str[::-1]) if p != "I"][::-1]
        paulis = [p for p in pauli_str if p != "I"]

        if working_qubits is None:
            working_qubits = list(range(len(pauli_str)))

        # Apply basis change for non-trivial Paulis
        self._apply_basis_change(paulis, qubits, working_qubits)

        if qubits:
            self._apply_cnot_ladder(qubits, working_qubits)
            # Apply phase rotation on the last qubit
            self.rz(working_qubits[qubits[-1]], 2.0 * float(np.real(coeff)))
            self._undo_cnot_ladder(qubits, working_qubits)

        # Undo basis change
        self._undo_basis_change(paulis, qubits, working_qubits)

    def controlled_pauli_evolution(
        self,
        operator: QuantumOperatorBase,
        parameter: Parameter | float,
        control_qubit: int,
        working_qubits: List[int] | None = None,
    ) -> None:
        """
        Applies controlled Pauli evolution exp(itP) where P is a Pauli operator.

        Args:
            operator (QuantumOperatorBase): The Pauli operator to evolve.
            parameter (Parameter | float): The evolution parameter.
            control_qubit (int): The qubit to control the evolution.
            working_qubits (List[int]): Optional: the qubits to use as working qubits.
        """

        pauli_str = operator.paulis[0]
        coeff = operator.coeffs

        if len(coeff) != 1:
            raise ValueError("Only operators with single Pauli strings are supported")
        coeff = coeff[0] * parameter

        if not isinstance(coeff, (Parameter, ParameterExpression)):
            coeff = np.real_if_close(coeff)
            if np.iscomplexobj(coeff):
                raise ValueError("Complex coefficients are not supported")

        qubits = [i for i, p in enumerate(pauli_str[::-1]) if p != "I"][::-1]
        paulis = [p for p in pauli_str if p != "I"]

        if len(paulis) == 0:
            self.rz(control_qubit, float(np.real(-coeff)))
            return

        if working_qubits is None:
            working_qubits = list(range(len(pauli_str) + 1))
            working_qubits.remove(control_qubit)

        # Apply basis change for non-trivial Paulis
        self._apply_basis_change(paulis, qubits, working_qubits)

        if qubits:
            self._apply_cnot_ladder(qubits, working_qubits)
            # Apply phase rotation on the last qubit
            self.crz(control_qubit, working_qubits[qubits[-1]], 2.0 * float(np.real(coeff)))
            self._undo_cnot_ladder(qubits, working_qubits)

        # Undo basis change
        self._undo_basis_change(paulis, qubits, working_qubits)

    def compose(self, qc: "QuantumCircuitBase", qubits: List[int]) -> "QuantumCircuitBase":
        """Compose two quantum circuits."""
        raise NotImplementedError

    def assign_parameters(self, parameters: dict):
        """Change parameters in the circuit.

        Args:
            parameters (np.array): parameters to assign to the circuit
        """
        raise NotImplementedError

    def invert(self) -> "QuantumCircuitBase":
        """Invert the circuit."""
        raise NotImplementedError

    def copy(self) -> "QuantumCircuitBase":
        """Return a copy of the circuit."""
        raise NotImplementedError

    def circuit_metrics(self) -> dict:
        """count number of gates in the circuit"""
        raise NotImplementedError

    def from_qasm(self, qasm: str) -> None:
        """Load the circuit from a qasm string"""
        raise NotImplementedError

    def to_qasm(self) -> str:
        """Convert the circuit to a qasm string"""
        raise NotImplementedError

    def _circuit_hash_key(self):
        return (self.num_qubits, self.draw())

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError
