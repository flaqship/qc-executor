from qoqo import Circuit
from qoqo_qasm import qasm_str_to_circuit
from qiskit.qasm2 import dumps, loads
from qiskit.compiler import transpile

from ..qiskit._decompose import decompose_to_std
from ..quantum_circuit import QuantumCircuit
from .qoqo_gate import qoqo_target


class QoqoCircuit:
    """qoqo (QuEST) circuit data structure.

    Args:
        circuit (QuantumCircuit): Circuit to convert to qoqo. A generic
            :class:`~qc_executor.quantum_circuit.QuantumCircuit` is imported
            via its Qiskit representation.
        backend (Union[Backend, None]): Optional qiskit backend. If given, the
            circuit is transpiled to the gate set of the backend instead of the
            default qoqo target. Required for noise simulation, since the
            noise insertion only supports device gate sets.
            (Defaults to ``None``)
    """

    @classmethod
    def from_quantum_circuit(cls, circuit: QuantumCircuit) -> "QoqoCircuit":
        """Create a Qoqo native circuit from a generic circuit."""
        return cls(circuit)

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend=None,
    ) -> None:
        qiskit_circuit = (
            circuit._qiskit_circuit
            if hasattr(circuit, "_qiskit_circuit")
            else getattr(circuit, "qiskit_circuit", circuit)
        )
        # Transpile circuit to supported basis gates and expand blocks automatically
        self._qiskit_base_circuit = transpile(
            decompose_to_std(qiskit_circuit),
            backend=backend,
            target=None if backend is not None else qoqo_target,
            optimization_level=0,
        )
        self._num_qubits = self._qiskit_base_circuit.num_qubits
        self._noise_backend = backend
        self.is_parameterized = len(self._qiskit_base_circuit.parameters) > 0
        self._qiskit_circuit = self._qiskit_base_circuit if not self.is_parameterized else None
        self._qoqo_gates_parameters = []
        self._qoqo_gates_parameters_dimensions = {}

        for param in self._qiskit_base_circuit.parameters:
            vector_name = (
                param.vector.name if hasattr(param, "vector") else param.name.split("[")[0]
            )
            if vector_name not in self._qoqo_gates_parameters:
                self._qoqo_gates_parameters.append(vector_name)
                self._qoqo_gates_parameters_dimensions[vector_name] = len(param.vector)

    @property
    def num_qubits(self) -> int:
        """Number of qubits of the circuit"""
        return self._num_qubits

    @property
    def qoqo_circuit(self) -> Circuit:
        """qoqo circuit"""
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
        return str(self._qiskit_base_circuit)

    def circuit_metrics(self) -> dict:
        """count number of gates in the circuit"""
        if self._qoqo_circuit is None:
            raise ValueError("Cannot calculate circuit metrics for unloaded circuit.")
        return len(self._qoqo_circuit)

    def from_qasm(self, qasm: str) -> None:
        """Load the circuit from a qasm string"""
        self._qoqo_circuit = qasm_str_to_circuit(qasm)
        self._qiskit_base_circuit = loads(qasm)

    def to_qasm(self) -> str:
        """Convert the circuit to a qasm string"""
        return dumps(self._qiskit_circuit)

    def get_qoqo_circuit(self) -> Circuit:
        """Builds and returns the qoqo circuit as callable function"""
        if self.is_parameterized:
            raise ValueError("Cannot build qoqo circuit from a parameterized circuit")
        self._qoqo_circuit = qasm_str_to_circuit(dumps(self._qiskit_circuit))
        return self._qoqo_circuit

    def __call__(self, *args, **kwargs):
        return self._qoqo_circuit(*args, **kwargs)

    def assign_parameters(self, parameters: dict) -> None:
        """
        Assigns the given parameters to the circuit's parameters.

        Args:
            parameters (dict): Dictionary with parameter names as keys and their values as values.
        """
        self._qiskit_circuit = self._qiskit_base_circuit.copy()
        self._qiskit_circuit.assign_parameters(parameters, inplace=True)
        if len(self._qiskit_circuit.parameters) == 0:
            self.is_parameterized = False
