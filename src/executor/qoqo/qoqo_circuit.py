from qoqo import Circuit
from qoqo_qasm import qasm_str_to_circuit
from qiskit.qasm2 import dumps
from qiskit.compiler import transpile

from ..utils.decompose_to_std import decompose_to_std
from ..quantum_circuit import QuantumCircuit
from .qoqo_gate import qoqo_target


class QoqoCircuit:

    def __init__(
        self,
        circuit: QuantumCircuit,
    ) -> None:
        # Transpile circuit to supported basis gates and expand blocks automatically
        self._qiskit_circuit = transpile(
            decompose_to_std(circuit._qiskit_circuit),
            target=qoqo_target,
            optimization_level=0,
        )
        self._num_qubits = self._qiskit_circuit.num_qubits

    @property
    def num_qubits(self) -> int:
        """Number of qubits of the circuit"""
        return self._num_qubits

    @property
    def qoqo_circuit(self) -> callable:
        """qoqo circuit that can be called with parameters"""
        return self._qoqo_circuit

    @property
    def parameter_names(self) -> list:
        """List of circuit parameter names"""
        return self._qoqo_gates_parameters

    @property
    def parameter_dimensions(self) -> dict:
        """Dictionary with the dimension of each circuit parameter"""
        return self._qoqo_gates_parameters_dimensions

    @property
    def hash(self) -> str:
        """Hashable object of the circuit and observable for caching"""
        return hash(str(self._qiskit_circuit))

    def get_qoqo_circuit(self) -> Circuit:
        """Builds and returns the qoqo circuit as callable function"""
        self._qoqo_circuit = qasm_str_to_circuit(dumps(self._qiskit_circuit))
        return self._qoqo_circuit

    def __call__(self, *args, **kwargs):
        return self._qoqo_circuit(*args, **kwargs)
