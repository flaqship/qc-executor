from qc_executor.base.circuit_base import QuantumCircuitBase


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

    def barrier(self, qubits):
        self._record("barrier", qubits)

    def measure(self):
        self._record("measure")
