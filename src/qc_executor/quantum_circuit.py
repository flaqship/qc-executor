"""Concrete quantum circuit implementation backed by a Qiskit circuit."""

from __future__ import annotations

from typing import List

import numpy as np
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement

from .base import QuantumCircuitBase
from .utils.qiskit_hash_functions import _circuit_key


class QuantumCircuit(QuantumCircuitBase):
    """
    Base class for quantum circuits for different quantum frameworks.

    Args:
        num_qubits (int): Number of qubits in the circuit
    """

    _GATE_NAMES = frozenset(
        {
            "h",
            "s",
            "sdag",
            "t",
            "tdag",
            "x",
            "y",
            "z",
            "i",
            "p",
            "rx",
            "ry",
            "rz",
            "u",
            "cx",
            "cy",
            "cz",
            "ch",
            "cnot",
            "ecr",
            "swap",
            "cswap",
            "ccx",
            "toffoli",
            "cp",
            "crx",
            "cry",
            "crz",
            "rxx",
            "ryy",
            "rzz",
            "rzx",
            "cu",
        }
    )

    def __init__(self, num_qubits: int, num_clbits: int = 0, _native_circuit: QiskitQuantumCircuit | None = None):
        super().__init__(num_qubits)
        self._qiskit_circuit: QiskitQuantumCircuit = (
            _native_circuit
            if _native_circuit is not None
            else QiskitQuantumCircuit(num_qubits, num_clbits)
        )

    @classmethod
    def from_quantum_circuit(cls, circuit: QuantumCircuitBase) -> QuantumCircuitBase:
        """Identity conversion for generic circuits."""
        return circuit

    @classmethod
    def from_qiskit(cls, circuit: QiskitQuantumCircuit) -> "QuantumCircuit":
        """Wrap a native qiskit circuit.

        Args:
            circuit (QiskitQuantumCircuit): The qiskit circuit to wrap. The
                circuit is used as-is (not copied).

        Returns:
            QuantumCircuit: The wrapping circuit.
        """
        return cls(circuit.num_qubits, _native_circuit=circuit)

    @classmethod
    def available_gates(cls) -> frozenset[str]:
        """Return the set of gate-method names defined on this circuit class."""
        return cls._GATE_NAMES

    @property
    def qiskit_circuit(self) -> QiskitQuantumCircuit:
        """The underlying Qiskit circuit."""
        return self._qiskit_circuit

    @qiskit_circuit.setter
    def qiskit_circuit(self, circuit: QiskitQuantumCircuit) -> None:
        """Replace the underlying Qiskit circuit.

        Args:
            circuit (QiskitQuantumCircuit): The new native circuit. Must act
                on the same number of qubits.

        Raises:
            ValueError: If the qubit count differs from the current circuit.
        """
        if circuit.num_qubits != self._num_qubits:
            raise ValueError(
                f"Replacement circuit must have {self._num_qubits} qubits, "
                f"got {circuit.num_qubits}."
            )
        self._qiskit_circuit = circuit

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits in the circuit."""
        return self._qiskit_circuit.num_qubits

    @property
    def num_clbits(self) -> int:
        """Return the number of classical bits in the circuit."""
        return self._qiskit_circuit.num_clbits

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

    def draw(self):
        """Return a printable text representation of the circuit."""
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

    def cswap(self, control_qubit: int, qubit1: int, qubit2: int):
        """Add a controlled-SWAP (Fredkin) gate."""
        self._qiskit_circuit.cswap(control_qubit, qubit1, qubit2)

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
        if qubits is None:
            self._qiskit_circuit.barrier()
        else:
            self._qiskit_circuit.barrier(qubits)

    def measure(self, qubits: int | List[int], clbits: int | List[int]) -> None:
        """Measure qubits into classical bits."""
        self._qiskit_circuit.measure(qubits, clbits)

    def if_test(self, clbit: int, value: int):
        """Create a classical conditional block.

        The returned context manager can be used as:

            with qc.if_test(0, 1):
                qc.x(0)
        """
        if clbit < 0 or clbit >= self.num_clbits:
            raise ValueError(
                f"Classical bit index {clbit} is out of range for "
                f"a circuit with {self.num_clbits} classical bits."
            )

        return self._qiskit_circuit.if_test(
            (self._qiskit_circuit.clbits[clbit], value)
        )

    # pauli_string, pauli_evolution and controlled_pauli_evolution are
    # inherited from QuantumCircuitBase.

    def compose(
        self,
        qc: QuantumCircuitBase,
        qubits: List[int] | None = None,
        clbits: List[int] | None = None,
        new_parameters: bool = True,
    ) -> "QuantumCircuit":
        """Compose another circuit into this one (always in place).

        Parameters of both circuits are re-indexed into a single fresh
        parameter vector so that repeatedly composing circuits that use
        identically named parameter vectors never collides. The parameters of
        ``self`` keep their positions; the parameters of ``qc`` are appended.

        Args:
            qc (QuantumCircuitBase): Circuit to compose with.
            qubits (List[int] | None): Qubit indices of ``self`` that the
                qubits of ``qc`` are mapped onto. Defaults to the identity
                mapping, which requires equal qubit counts.
            new_parameters (bool): If True (default), the parameters of ``qc``
                are appended after the parameters of ``self``. If False, the
                parameters of both circuits are merged positionally, i.e.
                parameter ``i`` of ``qc`` becomes parameter ``i`` of ``self``.

        Returns:
            QuantumCircuit: This circuit, after composition.

        Raises:
            ValueError: If ``qc`` is not a compatible circuit or the qubit
                mapping is invalid.
        """
        if not isinstance(qc, QuantumCircuit):
            raise ValueError("QuantumCircuit can only compose with QuantumCircuit objects")

        if qubits is None:
            if self.num_qubits != qc.num_qubits:
                raise ValueError(
                    "When qubits=None, both circuits must have the same number of qubits "
                    f"(got self.num_qubits={self.num_qubits}, qc.num_qubits={qc.num_qubits})."
                )
            qubits = list(range(qc.num_qubits))

        if len(qubits) != qc.num_qubits:
            raise ValueError(
                f"Qubit mapping length must equal qc.num_qubits "
                f"(got len(qubits)={len(qubits)}, qc.num_qubits={qc.num_qubits})."
            )
        if any(q < 0 or q >= self.num_qubits for q in qubits):
            raise ValueError("Qubit mapping contains indexes out of range for the target circuit.")

        own = self._qiskit_circuit
        other = qc.qiskit_circuit

        if own.parameters and other.parameters:
            # TODO: Merging squashes both circuits into a single vector named
            # after self's first parameter, so qc's parameters are renamed
            # (e.g. "y[0]" becomes "x[1]") and keyword access via the old name
            # stops working. Decide whether the original names should be kept.
            own_params = list(own.parameters)
            other_params = list(other.parameters)
            first = own_params[0]
            name = first.vector.name if isinstance(first, ParameterVectorElement) else first.name
            if new_parameters:
                merged = ParameterVector(name, len(own_params) + len(other_params))
                other_target = merged[len(own_params) :]
            else:
                merged = ParameterVector(name, max(len(own_params), len(other_params)))
                other_target = merged[: len(other_params)]
            own.assign_parameters(dict(zip(own_params, merged[: len(own_params)])), inplace=True)
            other_assigned = other.assign_parameters(
                dict(zip(other_params, other_target)), inplace=False
            )
            own.compose(other_assigned, qubits=qubits, clbits=clbits, inplace=True)
        else:
            own.compose(other, qubits=qubits, clbits=clbits, inplace=True)
        return self

    def assign_parameters(self, parameters: dict):
        """Change parameters in the circuit.

        Args:
            parameters (np.array): parameters to assign to the circuit
        """
        self._qiskit_circuit.assign_parameters(parameters, inplace=True)

    def fixate_parameters(self, parameters: np.ndarray) -> None:
        """Bind all free parameters, removing them from the circuit.

        Args:
            parameters (np.ndarray): Values to assign, in parameter order.
        """
        self._qiskit_circuit.assign_parameters(
            dict(zip(self._qiskit_circuit.parameters, parameters)), inplace=True
        )

    def invert(self) -> "QuantumCircuit":
        """Invert the circuit."""
        return self.__class__(self._num_qubits, self._qiskit_circuit.inverse())

    def copy(self) -> "QuantumCircuit":
        """Return a copy of the circuit."""
        return self.__class__(self._num_qubits, self.num_clbits, self._qiskit_circuit.copy())

    def circuit_metrics(self) -> dict:
        """count number of gates in the circuit"""
        raise NotImplementedError

    def from_qasm(self, qasm: str) -> None:
        """Load the circuit from a qasm string"""
        raise NotImplementedError

    def to_qasm(self) -> str:
        """Convert the circuit to a qasm string"""
        raise NotImplementedError

    def structural_key(self) -> tuple:
        """Return a hashable key describing the current circuit structure.

        Recomputed on every call so that in-place mutations are always
        reflected; never memoize this value.
        """
        return _circuit_key(self._qiskit_circuit)

    def __hash__(self):
        return hash(_circuit_key(self._qiskit_circuit))

    def __eq__(self, other):
        """Structural equality, consistent with the structural ``__hash__``."""
        return isinstance(other, QuantumCircuit) and _circuit_key(
            self._qiskit_circuit
        ) == _circuit_key(other._qiskit_circuit)

    def __str__(self):
        return str(self._qiskit_circuit)

    def __repr__(self):
        return str(self)
