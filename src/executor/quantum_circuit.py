import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union

from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.circuit import ParameterExpression, Parameter

from qiskit import QuantumCircuit as QiskitQuantumCircuit

from .base_classes import QuantumOperatorBase

from .base_classes import QuantumCircuitBase


class QuantumCircuit(QuantumCircuitBase):
    """
    Base class for quantum circuits for different quantum frameworks.

    Args:
        num_qubits (int): Number of qubits in the circuit
    """

    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self._qiskit_circuit = QiskitQuantumCircuit(self._num_qubits)

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits in the circuit."""
        return self._qiskit_circuit.num_qubits

    @property
    def parameters(self) -> List[ParameterVectorElement]:
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
        raise NotImplementedError

    def h(self, qubits: Union[int, List[int]]):
        """Add hadamard gates"""
        self._qiskit_circuit.h(qubits)

    def s(self, qubits: Union[int, List[int]]):
        """Add S gates"""
        self._qiskit_circuit.s(qubits)

    def sdag(self, qubits: Union[int, List[int]]):
        """Add Sdag gates"""
        self._qiskit_circuit.sdg(qubits)

    def t(self, qubits: Union[int, List[int]]):
        """Add T gates"""
        self._qiskit_circuit.t(qubits)

    def tdag(self, qubits: Union[int, List[int]]):
        """Add Tdg gates"""
        self._qiskit_circuit.tdg(qubits)

    def p(self, qubits: Union[int, List[int]], angle: float):
        """Add P gates"""
        self._qiskit_circuit.p(angle, qubits)

    def cp(self, control_qubit: int, target_qubit: int, angle: float):
        """Add CP gates"""
        self._qiskit_circuit.cp(angle, control_qubit, target_qubit)

    def x(self, qubits: Union[int, List[int]]):
        """Add X gates"""
        self._qiskit_circuit.x(qubits)

    def y(self, qubits: Union[int, List[int]]):
        """Add Y gates"""
        self._qiskit_circuit.y(qubits)

    def z(self, qubits: Union[int, List[int]]):
        """Add Z gates"""
        self._qiskit_circuit.z(qubits)

    def rx(self, qubits: Union[int, List[int]], angle: float):
        """Add RX gates"""
        self._qiskit_circuit.rx(angle, qubits)

    def ry(self, qubits: Union[int, List[int]], angle: float):
        """Add RY gates"""
        self._qiskit_circuit.ry(angle, qubits)

    def rz(self, qubits: Union[int, List[int]], angle: float):
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
        """Add CRX gates"""
        self._qiskit_circuit.cry(angle, control_qubit, target_qubit)

    def crz(self, control_qubit: int, target_qubit: int, angle: float):
        """Add CRX gates"""
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

    def barrier(self, qubits: Union[int, List[int]]):
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

    def pauli_evolution(
        self,
        op: QuantumOperatorBase,
        parameter: Union[ParameterVectorElement, float],
        working_qubits: Union[List[int], None] = None,
    ) -> None:
        """
        Applies Pauli evolution exp(itP) where P is a Pauli operator.

        Args:
            op (QuantumOperatorBase): The Pauli operator to evolve.
            parameter (Union[ParameterVectorElement, float]): The evolution parameter.
            working_qubits (List[int]): Optional: the qubits to use as working qubits.
        """

        pauli_str = op.paulis[0].to_label()
        coeff = op.coeffs
        if len(coeff) != 1:
            raise ValueError("Only operators with single Pauli strings are supported")
        coeff = coeff[0]

        if not isinstance(coeff, (ParameterVectorElement, ParameterExpression)):
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

        # Apply basis change for non-trivial Paulis
        for p, q in zip(paulis, qubits):
            if p == "X":
                self.h(working_qubits[q])
            elif p == "Y":
                self.sdag(working_qubits[q])
                self.h(working_qubits[q])
            elif p != "Z":
                raise ValueError(f"Unknown Pauli operator: {p}")

        # Apply controlled chain of CNOTs
        if qubits:
            control = qubits[0]
            for target in qubits[1:]:
                self.cx(working_qubits[control], working_qubits[target])
                control = target

            # Apply phase rotation on the last qubit
            self.rz(working_qubits[qubits[-1]], 2 * coeff)

            # Undo controlled CNOT chain
            control = qubits[-1]
            for target in reversed(qubits[:-1]):
                self.cx(working_qubits[target], working_qubits[control])
                control = target

        # Undo basis change
        for p, q in zip(paulis, qubits):
            if p == "X":
                self.h(working_qubits[q])
            elif p == "Y":
                self.h(working_qubits[q])
                self.s(working_qubits[q])

    def controlled_pauli_evolution(
        self,
        op: QuantumOperatorBase,
        parameter: Union[ParameterVectorElement, float],
        control_qubit: int,
        working_qubits: Union[List[int], None] = None,
    ) -> None:
        """
        Applies controlled Pauli evolution exp(itP) where P is a Pauli operator.

        Args:
            op (QuantumOperatorBase): The Pauli operator to evolve.
            parameter (Union[ParameterVectorElement, float]): The evolution parameter.
            control_qubit (int): The qubit to control the evolution.
            working_qubits (List[int]): Optional: the qubits to use as working qubits.
        """

        pauli_str = op.paulis[0].to_label()
        coeff = op.coeffs

        if len(coeff) != 1:
            raise ValueError("Only operators with single Pauli strings are supported")
        coeff = coeff[0] * parameter

        if not isinstance(coeff, (ParameterVectorElement, ParameterExpression)):
            coeff = np.real_if_close(coeff)
            if isinstance(coeff, complex):
                raise ValueError("Complex coefficients are not supported")

        qubits = [i for i, p in enumerate(pauli_str[::-1]) if p != "I"][::-1]
        paulis = [p for p in pauli_str if p != "I"]

        if len(paulis) == 0:
            self.rz(control_qubit, -coeff)
            return None

        if working_qubits is None:
            working_qubits = list(range(len(pauli_str) + 1))
            working_qubits.remove(control_qubit)

        # Apply basis change for non-trivial Paulis
        for p, q in zip(paulis, qubits):
            if p == "X":
                self.h(working_qubits[q])
            elif p == "Y":
                self.sdag(working_qubits[q])
                self.h(working_qubits[q])
            elif p != "Z":
                raise ValueError(f"Unknown Pauli operator: {p}")

        # Apply controlled chain of CNOTs
        if qubits:
            control = qubits[0]
            for target in qubits[1:]:
                self.cx(working_qubits[control], working_qubits[target])
                control = target

            # Apply phase rotation on the last qubit
            self.crz(control_qubit, working_qubits[qubits[-1]], 2.0 * coeff)

            # Undo controlled CNOT chain
            control = qubits[-1]
            for target in reversed(qubits[:-1]):
                self.cx(working_qubits[target], working_qubits[control])
                control = target

        # Undo basis change
        for p, q in zip(paulis, qubits):
            if p == "X":
                self.h(working_qubits[q])
            elif p == "Y":
                self.h(working_qubits[q])
                self.s(working_qubits[q])

    def compose(self, qc: "QuantumCircuit", qubits: List[int]) -> "QuantumCircuit":
        """Compose two quantum circuits."""
        if isinstance(qc, QuantumCircuit):
            self._qiskit_circuit.compose(qc._qiskit_circuit, qubits, inplace=True)
        else:
            raise ValueError("The circuit to compose must be a QuantumCircuit object")

    def assign_parameters(self, parameters: dict):
        """Change parameters in the circuit.

        Args:
            parameters (np.array): parameters to assign to the circuit
        """
        self._qiskit_circuit.assign_parameters(parameters, inplace=True)

    def invert(self) -> "QuantumCircuitBase":
        """Invert the circuit."""
        """Invert the circuit."""
        return_class = self.copy()
        return_class._quantum_circuit_qiskit = return_class._quantum_circuit_qiskit.inverse()
        return return_class

    def copy(self) -> "QuantumCircuitBase":
        """Return a copy of the circuit."""
        return_class = self.__class__(self._num_qubits)
        return_class._qiskit_circuit = self._qiskit_circuit.copy()
        return return_class

    def circuit_metrics(self) -> dict:
        """ count number of gates in the circuit"""
        raise NotImplementedError

    def from_qasm(self, qasm: str) -> None:
        """Load the circuit from a qasm string"""
        raise NotImplementedError

    def to_qasm(self) -> str:
        """Convert the circuit to a qasm string"""
        raise NotImplementedError

    def __hash__(self):
        # TODO: implement a better hash function
        return hash(str(self))

    def __str__(self):
        return str(self._qiskit_circuit)

    def __repr__(self):
        return str(self)
