"""Concrete quantum circuit implementation backed by a Qiskit circuit."""

from __future__ import annotations

from typing import List

from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.circuit.parametervector import ParameterVectorElement

from .base import QuantumCircuitBase
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

    def ccx(self, control_qubit1: int, control_qubit2: int, target_qubit: int):
        """Add Toffoli (CCX) gates"""
        self._qiskit_circuit.ccx(control_qubit1, control_qubit2, target_qubit)

    def toffoli(self, control_qubit1: int, control_qubit2: int, target_qubit: int):
        """Add Toffoli (CCX) gates"""
        self.ccx(control_qubit1, control_qubit2, target_qubit)

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

    def ch(self, control_qubit: int, target_qubit: int):
        """Add CH gates"""
        self._qiskit_circuit.ch(control_qubit, target_qubit)

    def i(self, qubits: int | List[int]):
        """Add Identity gates"""
        self._qiskit_circuit.id(qubits)

    def u(self, qubits: int | List[int], theta: float, phi: float, lam: float):
        """Add U gates"""
        self._qiskit_circuit.u(theta, phi, lam, qubits)

    def cu(
        self,
        control_qubit: int,
        target_qubit: int,
        theta: float,
        phi: float,
        lam: float,
        gamma: float,
    ):
        """Add CU gates"""
        self._qiskit_circuit.cu(theta, phi, lam, gamma, control_qubit, target_qubit)

    def barrier(self, qubits: int | List[int] = None):
        """Add barrier gates"""
        self._qiskit_circuit.barrier(qubits)

    def measure(self):
        """Add measure gates"""
        raise NotImplementedError

    # pauli_string, pauli_evolution and controlled_pauli_evolution are
    # inherited from QuantumCircuitBase.

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

    def __hash__(self):
        return hash(_circuit_key(self._qiskit_circuit))

    def __str__(self):
        return str(self._qiskit_circuit)

    def __repr__(self):
        return str(self)
