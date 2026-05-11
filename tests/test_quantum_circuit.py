"""Tests for `executor.quantum_circuit`."""

from unittest.mock import MagicMock

import pytest
from qiskit.circuit import ParameterVector

import executor.quantum_circuit as quantum_circuit_module
from executor import QuantumCircuit
from executor.parameters import Parameters


def create_mock_operator(paulis, coeffs):
    """Create a mock operator with the given paulis and coeffs."""
    operator = MagicMock()
    operator.paulis = paulis
    operator.coeffs = coeffs
    return operator


class RecordingQuantumCircuit(QuantumCircuit):
    """Quantum circuit that records gate calls for behavioural tests."""

    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.calls = []

    def _record(self, name: str, *args):
        self.calls.append((name, *args))

    def h(self, qubits):
        self._record("h", qubits)
        super().h(qubits)

    def s(self, qubits):
        self._record("s", qubits)
        super().s(qubits)

    def sdag(self, qubits):
        self._record("sdag", qubits)
        super().sdag(qubits)

    def x(self, qubits):
        self._record("x", qubits)
        super().x(qubits)

    def y(self, qubits):
        self._record("y", qubits)
        super().y(qubits)

    def z(self, qubits):
        self._record("z", qubits)
        super().z(qubits)

    def cx(self, control_qubit: int, target_qubit: int):
        self._record("cx", control_qubit, target_qubit)
        super().cx(control_qubit, target_qubit)

    def rz(self, qubits, angle):
        self._record("rz", qubits, angle)
        super().rz(qubits, angle)

    def crz(self, control_qubit: int, target_qubit: int, angle: float):
        self._record("crz", control_qubit, target_qubit, angle)
        super().crz(control_qubit, target_qubit, angle)


class TestQuantumCircuitBasics:
    def test_from_quantum_circuit_returns_same_instance(self):
        circuit = QuantumCircuit(1)

        assert QuantumCircuit.from_quantum_circuit(circuit) is circuit

    def test_properties_and_parameter_binding(self):
        params = Parameters("theta", 1)
        circuit = QuantumCircuit(1)

        assert circuit.num_qubits == 1
        assert circuit.num_parameters == 0
        assert not circuit.is_parameterized

        circuit.rx(0, params[0])

        assert circuit.is_parameterized
        assert circuit.num_parameters == 1
        assert circuit.parameters == [params[0]]

        circuit.assign_parameters({"theta[0]": 0.5})

        assert not circuit.is_parameterized
        assert circuit.num_parameters == 0

    def test_cnot_is_alias_for_cx(self):
        circuit = RecordingQuantumCircuit(2)

        circuit.cnot(0, 1)

        assert circuit.calls == [("cx", 0, 1)]


class TestQuantumCircuitPauliString:

    def test_pauli_string_applies_reverse_qubit_order(self):
        circuit = RecordingQuantumCircuit(3)

        circuit.pauli_string("XYZ")

        assert circuit.calls == [("z", 0), ("y", 1), ("x", 2)]

    def test_pauli_string_skips_identity_paulis(self):
        circuit = RecordingQuantumCircuit(3)

        circuit.pauli_string("III")

        assert circuit.calls == []

    def test_pauli_string_validates_length(self):
        circuit = QuantumCircuit(2)

        with pytest.raises(ValueError, match="length does not match number of qubits"):
            circuit.pauli_string("XYZ")


class TestQuantumCircuitPauliEvolution:

    def test_pauli_evolution_single_x_pauli(self):
        circuit = RecordingQuantumCircuit(1)
        operator = create_mock_operator(paulis=["X"], coeffs=[1.0])

        circuit.pauli_evolution(operator, 0.5)

        assert circuit.calls == [("h", 0), ("rz", 0, 1.0), ("h", 0)]

    def test_pauli_evolution_with_y_basis_change(self):
        circuit = RecordingQuantumCircuit(1)
        operator = create_mock_operator(paulis=["Y"], coeffs=[1.0])

        circuit.pauli_evolution(operator, 0.5)

        assert circuit.calls == [
            ("sdag", 0),
            ("h", 0),
            ("rz", 0, 1.0),
            ("h", 0),
            ("s", 0),
        ]

    def test_pauli_evolution_with_multi_qubit_chain(self):
        circuit = RecordingQuantumCircuit(3)
        operator = create_mock_operator(paulis=["XX"], coeffs=[1.0])

        circuit.pauli_evolution(operator, 0.5)

        assert circuit.calls == [
            ("h", 1),
            ("h", 0),
            ("cx", 1, 0),
            ("rz", 0, 1.0),
            ("cx", 1, 0),
            ("h", 1),
            ("h", 0),
        ]

    def test_pauli_evolution_with_symbolic_coefficient(self):
        theta = ParameterVector("theta", 1)
        circuit = RecordingQuantumCircuit(1)
        operator = create_mock_operator(paulis=["Z"], coeffs=[theta[0]])

        # Symbolic coefficients currently cause TypeError when converted to float
        with pytest.raises(TypeError, match="is not numeric"):
            circuit.pauli_evolution(operator, 0.5)

    def test_pauli_evolution_rejects_multi_term_operator(self):
        circuit = RecordingQuantumCircuit(1)
        operator = create_mock_operator(paulis=["X", "Z"], coeffs=[1.0, 0.5])

        with pytest.raises(ValueError, match="single Pauli strings"):
            circuit.pauli_evolution(operator, 0.5)

    def test_pauli_evolution_rejects_complex_coefficients(self, monkeypatch):
        circuit = RecordingQuantumCircuit(1)
        operator = create_mock_operator(paulis=["X"], coeffs=[1 + 1j])

        monkeypatch.setattr(quantum_circuit_module.np, "real_if_close", lambda value: value)

        with pytest.raises(ValueError, match="Complex coefficients are not supported"):
            circuit.pauli_evolution(operator, 0.5)

    def test_pauli_evolution_rejects_unknown_pauli(self):
        circuit = RecordingQuantumCircuit(1)
        operator = create_mock_operator(paulis=["A"], coeffs=[1.0])

        with pytest.raises(ValueError, match="Unknown Pauli operator: A"):
            circuit.pauli_evolution(operator, 0.5)


class TestQuantumCircuitControlledPauliEvolution:

    def test_controlled_pauli_evolution_identity_pauli(self):
        circuit = RecordingQuantumCircuit(1)
        operator = create_mock_operator(paulis=["I"], coeffs=[1.0])

        circuit.controlled_pauli_evolution(operator, 0.25, control_qubit=0)

        assert circuit.calls == [("rz", 0, -0.25)]

    def test_controlled_pauli_evolution_with_basis_change(self):
        circuit = RecordingQuantumCircuit(2)
        operator = create_mock_operator(paulis=["X"], coeffs=[1.0])

        circuit.controlled_pauli_evolution(operator, 0.5, control_qubit=1)

        assert circuit.calls == [("h", 0), ("crz", 1, 0, 1.0), ("h", 0)]

    def test_controlled_pauli_evolution_with_y_basis_change(self):
        circuit = RecordingQuantumCircuit(2)
        operator = create_mock_operator(paulis=["Y"], coeffs=[1.0])

        circuit.controlled_pauli_evolution(operator, 0.5, control_qubit=1)

        assert circuit.calls == [
            ("sdag", 0),
            ("h", 0),
            ("crz", 1, 0, 1.0),
            ("h", 0),
            ("s", 0),
        ]

    def test_controlled_pauli_evolution_with_multi_qubit_chain(self):
        circuit = RecordingQuantumCircuit(3)
        operator = create_mock_operator(paulis=["XX"], coeffs=[1.0])

        circuit.controlled_pauli_evolution(operator, 0.5, control_qubit=2)

        assert circuit.calls == [
            ("h", 1),
            ("h", 0),
            ("cx", 1, 0),
            ("crz", 2, 0, 1.0),
            ("cx", 1, 0),
            ("h", 1),
            ("h", 0),
        ]

    def test_controlled_pauli_evolution_with_symbolic_coefficient(self):
        theta = ParameterVector("theta", 1)
        circuit = RecordingQuantumCircuit(2)
        operator = create_mock_operator(paulis=["Z"], coeffs=[theta[0]])

        # Symbolic coefficients currently cause TypeError when converted to float
        with pytest.raises(TypeError, match="is not numeric"):
            circuit.controlled_pauli_evolution(operator, 0.5, control_qubit=1)

    def test_controlled_pauli_evolution_rejects_complex_coefficients(self, monkeypatch):
        circuit = RecordingQuantumCircuit(1)
        operator = create_mock_operator(paulis=["X"], coeffs=[1 + 1j])

        monkeypatch.setattr(quantum_circuit_module.np, "real_if_close", lambda value: value)

        with pytest.raises(ValueError, match="Complex coefficients are not supported"):
            circuit.controlled_pauli_evolution(operator, 0.5, control_qubit=0)

    def test_controlled_pauli_evolution_rejects_multi_term_operator(self):
        circuit = RecordingQuantumCircuit(1)
        operator = create_mock_operator(paulis=["X", "Z"], coeffs=[1.0, 0.5])

        with pytest.raises(ValueError, match="single Pauli strings"):
            circuit.controlled_pauli_evolution(operator, 0.5, control_qubit=0)

    def test_controlled_pauli_evolution_rejects_unknown_pauli(self):
        circuit = RecordingQuantumCircuit(1)
        operator = create_mock_operator(paulis=["A"], coeffs=[1.0])

        with pytest.raises(ValueError, match="Unknown Pauli operator: A"):
            circuit.controlled_pauli_evolution(operator, 0.5, control_qubit=0)


class TestQuantumCircuitOperations:

    def test_gate_methods_delegate_to_qiskit(self):
        circuit = QuantumCircuit(3)

        circuit.h(0)
        circuit.s(0)
        circuit.sdag(0)
        circuit.t(1)
        circuit.tdag(1)
        circuit.p(2, 0.1)
        circuit.cp(0, 1, 0.2)
        circuit.x(0)
        circuit.y(1)
        circuit.z(2)
        circuit.rx(0, 0.3)
        circuit.ry(1, 0.4)
        circuit.rz(2, 0.5)
        circuit.cx(0, 1)
        circuit.cy(1, 2)
        circuit.cz(0, 2)
        circuit.cnot(1, 2)
        circuit.ecr(0, 1)
        circuit.crx(0, 1, 0.6)
        circuit.cry(1, 2, 0.7)
        circuit.crz(0, 2, 0.8)
        circuit.rxx(0, 1, 0.9)
        circuit.ryy(1, 2, 1.0)
        circuit.rzz(0, 2, 1.1)
        circuit.rzx(0, 1, 1.2)
        circuit.swap(0, 2)
        circuit.barrier([0, 1, 2])

        counts = circuit._qiskit_circuit.count_ops()
        assert counts.get("h", 0) == 1
        assert counts.get("s", 0) == 1
        assert counts.get("sdg", 0) == 1
        assert counts.get("t", 0) == 1
        assert counts.get("tdg", 0) == 1
        assert counts.get("p", 0) == 1
        assert counts.get("cp", 0) == 1
        assert counts.get("x", 0) == 1
        assert counts.get("y", 0) == 1
        assert counts.get("z", 0) == 1
        assert counts.get("rx", 0) == 1
        assert counts.get("ry", 0) == 1
        assert counts.get("rz", 0) == 1
        assert counts.get("cx", 0) == 2
        assert counts.get("cy", 0) == 1
        assert counts.get("cz", 0) == 1
        assert counts.get("ecr", 0) == 1
        assert counts.get("crx", 0) == 1
        assert counts.get("cry", 0) == 1
        assert counts.get("crz", 0) == 1
        assert counts.get("rxx", 0) == 1
        assert counts.get("ryy", 0) == 1
        assert counts.get("rzz", 0) == 1
        assert counts.get("rzx", 0) == 1
        assert counts.get("swap", 0) == 1
        assert counts.get("barrier", 0) == 1

    def test_copy_creates_independent_circuit(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)

        copied = circuit.copy()
        copied.cx(0, 1)

        assert circuit is not copied
        assert circuit._qiskit_circuit is not copied._qiskit_circuit
        assert circuit._qiskit_circuit.count_ops().get("cx", 0) == 0
        assert copied._qiskit_circuit.count_ops().get("cx", 0) == 1

    def test_compose_combines_circuits(self):
        left = QuantumCircuit(2)
        left.h(0)
        right = QuantumCircuit(2)
        right.cx(0, 1)

        left.compose(right, [0, 1])

        assert left._qiskit_circuit.count_ops().get("h", 0) == 1
        assert left._qiskit_circuit.count_ops().get("cx", 0) == 1

    def test_compose_rejects_non_quantum_circuit(self):
        circuit = QuantumCircuit(1)

        with pytest.raises(ValueError, match="must be a QuantumCircuit object"):
            circuit.compose("not-a-circuit", [0])

    def test_hash_str_and_repr(self):
        circuit = QuantumCircuit(1)
        circuit.h(0)

        assert isinstance(hash(circuit), int)
        assert isinstance(str(circuit), str)
        assert repr(circuit) == str(circuit)

    def test_invert_uses_internal_qiskit_copy(self, monkeypatch):
        circuit = QuantumCircuit(1)
        circuit.h(0)

        original_copy = QuantumCircuit.copy

        def fake_copy(self):
            copied = original_copy(self)
            copied._quantum_circuit_qiskit = copied._qiskit_circuit
            return copied

        monkeypatch.setattr(QuantumCircuit, "copy", fake_copy)

        inverted = circuit.invert()

        assert isinstance(inverted, QuantumCircuit)
        assert inverted is not circuit
