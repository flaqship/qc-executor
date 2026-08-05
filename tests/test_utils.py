from qc_executor.base.circuit_base import QuantumCircuitBase
from qc_executor.base.operator_base import QuantumOperatorBase


class FakeOperator(QuantumOperatorBase):
    """Minimal generic operator carrying raw labels/coefficients."""

    def __init__(self, label_or_labels, coeffs):
        labels = label_or_labels if isinstance(label_or_labels, list) else [label_or_labels]
        super().__init__(num_qubits=len(labels[0]), paulis=labels, coeffs=coeffs)

    @classmethod
    def from_quantum_operator(cls, operator):
        return operator

    def adjoint(self):
        raise NotImplementedError

    def apply_layout(self, layout):
        raise NotImplementedError

    def compose(self, other):
        raise NotImplementedError

    def append(self, pauli, coeff):
        raise NotImplementedError

    def simplify(self):
        raise NotImplementedError

    def transpose(self):
        raise NotImplementedError

    def conjugate(self):
        raise NotImplementedError

    def group_commuting(self):
        raise NotImplementedError


class SpyCircuit(QuantumCircuitBase):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.ops = []

    @classmethod
    def from_quantum_circuit(cls, circuit: "QuantumCircuitBase") -> "QuantumCircuitBase":
        return circuit

    def _record(self, name, *args):
        self.ops.append((name, *args))

    def draw(self) -> str:
        return "SpyCircuit"

    def h(self, qubits):
        self._record("h", qubits)

    def s(self, qubits):
        self._record("s", qubits)

    def sdag(self, qubits):
        self._record("sdag", qubits)

    def t(self, qubits):
        self._record("t", qubits)

    def tdag(self, qubits):
        self._record("tdag", qubits)

    def p(self, qubits, angle):
        self._record("p", qubits, angle)

    def x(self, qubits):
        self._record("x", qubits)

    def y(self, qubits):
        self._record("y", qubits)

    def z(self, qubits):
        self._record("z", qubits)

    def rx(self, qubits, angle):
        self._record("rx", qubits, angle)

    def ry(self, qubits, angle):
        self._record("ry", qubits, angle)

    def rz(self, qubits, angle):
        self._record("rz", qubits, angle)

    def cx(self, control_qubit, target_qubit):
        self._record("cx", control_qubit, target_qubit)

    def cy(self, control_qubit, target_qubit):
        self._record("cy", control_qubit, target_qubit)

    def cz(self, control_qubit, target_qubit):
        self._record("cz", control_qubit, target_qubit)

    def crx(self, control_qubit, target_qubit, angle):
        self._record("crx", control_qubit, target_qubit, angle)

    def cry(self, control_qubit, target_qubit, angle):
        self._record("cry", control_qubit, target_qubit, angle)

    def crz(self, control_qubit, target_qubit, angle):
        self._record("crz", control_qubit, target_qubit, angle)

    def rxx(self, control_qubit, target_qubit, angle):
        self._record("rxx", control_qubit, target_qubit, angle)

    def ryy(self, control_qubit, target_qubit, angle):
        self._record("ryy", control_qubit, target_qubit, angle)

    def rzz(self, control_qubit, target_qubit, angle):
        self._record("rzz", control_qubit, target_qubit, angle)

    def rzx(self, control_qubit, target_qubit, angle):
        self._record("rzx", control_qubit, target_qubit, angle)

    def swap(self, qubit1, qubit2):
        self._record("swap", qubit1, qubit2)

    def ch(self, control_qubit, target_qubit):
        self._record("ch", control_qubit, target_qubit)

    def i(self, qubits):
        self._record("i", qubits)

    def u(self, qubits, theta, phi, lam):
        self._record("u", qubits, theta, phi, lam)

    def cu(self, control_qubit, target_qubit, theta, phi, lam, gamma):
        self._record("cu", control_qubit, target_qubit, theta, phi, lam, gamma)

    def barrier(self, qubits):
        self._record("barrier", qubits)

    def measure(self):
        self._record("measure")
