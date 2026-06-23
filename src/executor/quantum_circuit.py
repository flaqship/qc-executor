"""Concrete quantum circuit implementation backed by a Qiskit circuit."""

from __future__ import annotations

from typing import List

import numpy as np
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.circuit import ParameterExpression

from executor.parameters import Parameter

from .base import QuantumCircuitBase, QuantumOperatorBase
from .utils.qiskit_hash_functions import _circuit_key


class QuantumCircuit(QuantumCircuitBase):
    """
    Base class for quantum circuits for different quantum frameworks.

    Args:
        num_qubits (int): Number of qubits in the circuit
    """

    def __init__(self, num_qubits: int, _native_circuit: QiskitQuantumCircuit | None = None):
        super().__init__(num_qubits)
        self._qiskit_circuit: QiskitQuantumCircuit = (
            _native_circuit
            if _native_circuit is not None
            else QiskitQuantumCircuit(self._num_qubits)
        )

    @classmethod
    def from_quantum_circuit(cls, circuit: QuantumCircuitBase) -> QuantumCircuitBase:
        """Identity conversion for generic circuits."""
        return circuit

    @property
    def qiskit_circuit(self) -> QiskitQuantumCircuit:
        """The underlying Qiskit circuit."""
        return self._qiskit_circuit

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits in the circuit."""
        return self._qiskit_circuit.num_qubits

    @property
    def parameters(self) -> List[Parameter]:
        """Return the free trainable parameters in the circuit."""
        return list(self._qiskit_circuit.parameters)

    @property
    def num_parameters(self) -> int:
        """Return the number of free trainable parameters in the circuit."""
        return len(self.parameters)

    @property
    def is_parameterized(self) -> bool:
        """Check if the wavefunction is parameterized."""
        return len(self.parameters) > 0

    def draw(self) -> str:
        """Returns printable string representation of the circuit."""
        return self._qiskit_circuit.draw("text")

    def h(self, qubits: int | List[int]):
        """Add hadamard gates"""
        self._qiskit_circuit.h(qubits)

    def s(self, qubits: int | List[int]):
        """Add S gates"""
        self._qiskit_circuit.s(qubits)

    def sdag(self, qubits: int | List[int]):
        """Add Sdag gates"""
        self._qiskit_circuit.sdg(qubits)

    def t(self, qubits: int | List[int]):
        """Add T gates"""
        self._qiskit_circuit.t(qubits)

    def tdag(self, qubits: int | List[int]):
        """Add Tdg gates"""
        self._qiskit_circuit.tdg(qubits)

    def p(self, qubits: int | List[int], angle: float):
        """Add P gates"""
        self._qiskit_circuit.p(angle, qubits)

    def cp(self, control_qubit: int, target_qubit: int, angle: float):
        """Add CP gates"""
        self._qiskit_circuit.cp(angle, control_qubit, target_qubit)

    def x(self, qubits: int | List[int]):
        """Add X gates"""
        self._qiskit_circuit.x(qubits)

    def y(self, qubits: int | List[int]):
        """Add Y gates"""
        self._qiskit_circuit.y(qubits)

    def z(self, qubits: int | List[int]):
        """Add Z gates"""
        self._qiskit_circuit.z(qubits)

    def rx(self, qubits: int | List[int], angle: float):
        """Add RX gates"""
        self._qiskit_circuit.rx(angle, qubits)

    def ry(self, qubits: int | List[int], angle: float):
        """Add RY gates"""
        self._qiskit_circuit.ry(angle, qubits)

    def rz(self, qubits: int | List[int], angle: float):
        """Add RZ gates"""
        self._qiskit_circuit.rz(angle, qubits)

    def cx(self, control_qubit: int, target_qubit: int):
        """Add CNOT gates"""
        self._qiskit_circuit.cx(control_qubit, target_qubit)

    def cy(self, control_qubit: int, target_qubit: int):
        """Add CY gates"""
        self._qiskit_circuit.cy(control_qubit, target_qubit)

    def cz(self, control_qubit: int, target_qubit: int):
        """Add CZ gates"""
        self._qiskit_circuit.cz(control_qubit, target_qubit)

    def cnot(self, control_qubit: int, target_qubit: int):
        """Add CNOT gates"""
        self.cx(control_qubit, target_qubit)

    def ecr(self, control_qubit: int, target_qubit: int):
        """Add ECR gates"""
        self._qiskit_circuit.ecr(control_qubit, target_qubit)

    def crx(self, control_qubit: int, target_qubit: int, angle: float):
        """Add CRX gates"""
        self._qiskit_circuit.crx(angle, control_qubit, target_qubit)

    def cry(self, control_qubit: int, target_qubit: int, angle: float):
        """Add CRY gates"""
        self._qiskit_circuit.cry(angle, control_qubit, target_qubit)

    def crz(self, control_qubit: int, target_qubit: int, angle: float):
        """Add CRZ gates"""
        self._qiskit_circuit.crz(angle, control_qubit, target_qubit)

    def rxx(self, control_qubit: int, target_qubit: int, angle: float):
        """Add RXX gates"""
        self._qiskit_circuit.rxx(angle, control_qubit, target_qubit)

    def ryy(self, control_qubit: int, target_qubit: int, angle: float):
        """Add RYY gates"""
        self._qiskit_circuit.ryy(angle, control_qubit, target_qubit)

    def rzz(self, control_qubit: int, target_qubit: int, angle: float):
        """Add RZZ gates"""
        self._qiskit_circuit.rzz(angle, control_qubit, target_qubit)

    def rzx(self, control_qubit: int, target_qubit: int, angle: float):
        """Add RZX gates"""
        self._qiskit_circuit.rzx(angle, control_qubit, target_qubit)

    def swap(self, qubit1: int, qubit2: int):
        """Add SWAP gates"""
        self._qiskit_circuit.swap(qubit1, qubit2)

    def barrier(self, qubits: int | List[int]):
        """Add barrier gates"""
        self._qiskit_circuit.barrier(qubits)

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
        """Apply pre-rotation basis change for Pauli evolution."""
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
        """Undo the pre-rotation basis change after Pauli evolution."""
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
            if isinstance(coeff, complex):
                raise ValueError("Complex coefficients are not supported")
        else:
            # the 1j fixes a bug in qiskit
            coeff = -1j * (1j * coeff)

        coeff = coeff * parameter

        qubits = [i for i, p in enumerate(pauli_str[::-1]) if p != "I"][::-1]
        paulis = [p for p in pauli_str if p != "I"]

        if working_qubits is None:
            working_qubits = list(range(len(pauli_str)))

        self._apply_basis_change(paulis, qubits, working_qubits)
        if qubits:
            self._apply_cnot_ladder(qubits, working_qubits)
            self.rz(working_qubits[qubits[-1]], 2.0 * float(np.real(coeff)))
            self._undo_cnot_ladder(qubits, working_qubits)
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
            if isinstance(coeff, complex):
                raise ValueError("Complex coefficients are not supported")

        qubits = [i for i, p in enumerate(pauli_str[::-1]) if p != "I"][::-1]
        paulis = [p for p in pauli_str if p != "I"]

        if len(paulis) == 0:
            self.rz(control_qubit, float(np.real(-coeff)))
            return

        if working_qubits is None:
            working_qubits = list(range(len(pauli_str) + 1))
            working_qubits.remove(control_qubit)

        self._apply_basis_change(paulis, qubits, working_qubits)
        if qubits:
            self._apply_cnot_ladder(qubits, working_qubits)
            self.crz(control_qubit, working_qubits[qubits[-1]], 2.0 * float(np.real(coeff)))
            self._undo_cnot_ladder(qubits, working_qubits)
        self._undo_basis_change(paulis, qubits, working_qubits)

    def compose(self, qc: QuantumCircuitBase, qubits: List[int]) -> "QuantumCircuit":
        """Compose two quantum circuits."""
        if isinstance(qc, QuantumCircuit):
            self._qiskit_circuit.compose(qc.qiskit_circuit, qubits, inplace=True)
            return self
        raise ValueError("The circuit to compose must be a QuantumCircuit object")

    def assign_parameters(self, parameters: dict):
        """Change parameters in the circuit.

        Args:
            parameters (np.array): parameters to assign to the circuit
        """
        self._qiskit_circuit.assign_parameters(parameters, inplace=True)

    def invert(self) -> "QuantumCircuit":
        """Invert the circuit."""
        return self.__class__(self._num_qubits, self._qiskit_circuit.inverse())

    def copy(self) -> "QuantumCircuit":
        """Return a copy of the circuit."""
        return self.__class__(self._num_qubits, self._qiskit_circuit.copy())

    def circuit_metrics(self) -> dict:
        """count number of gates in the circuit"""
        raise NotImplementedError

    def from_qasm(self, qasm: str) -> None:
        """Load the circuit from a qasm string"""
        raise NotImplementedError

    def to_qasm(self) -> str:
        """Convert the circuit to a qasm string"""
        raise NotImplementedError

    def _circuit_hash_key(self) -> tuple:
        # Override with precise Qiskit-based key
        return (_circuit_key(self._qiskit_circuit),)

    def __str__(self):
        return str(self._qiskit_circuit)

    def __repr__(self):
        return str(self)
