"""Tests for `executor.quantum_circuit`."""

from unittest.mock import MagicMock, patch

import pytest
from qiskit.circuit import ParameterVector

import executor.quantum_circuit as quantum_circuit_module
from executor import QuantumCircuit
from executor.parameters import Parameters
from tests.test_utils import SpyCircuit


def create_mock_operator(paulis, coeffs):
    """Create a mock operator with the given paulis and coeffs."""
    operator = MagicMock()
    operator.paulis = paulis
    operator.coeffs = coeffs
    return operator


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
        circuit = SpyCircuit(2)

        circuit.cnot(0, 1)

        assert circuit.ops == [("cx", 0, 1)]


class TestQuantumCircuitPauliString:

    def test_pauli_string_applies_reverse_qubit_order(self):
        circuit = SpyCircuit(3)

        circuit.pauli_string("XYZ")

        assert circuit.ops == [("z", 0), ("y", 1), ("x", 2)]

    def test_pauli_string_skips_identity_paulis(self):
        circuit = SpyCircuit(3)

        circuit.pauli_string("III")

        assert circuit.ops == []

    def test_pauli_string_validates_length(self):
        circuit = QuantumCircuit(2)

        with pytest.raises(ValueError, match="length does not match number of qubits"):
            circuit.pauli_string("XYZ")


class TestQuantumCircuitPauliEvolution:

    def test_pauli_evolution_single_x_pauli(self):
        circuit = SpyCircuit(1)
        operator = create_mock_operator(paulis=["X"], coeffs=[1.0])

        circuit.pauli_evolution(operator, 0.5)

        assert circuit.ops == [("h", 0), ("rz", 0, 1.0), ("h", 0)]

    def test_pauli_evolution_with_y_basis_change(self):
        circuit = SpyCircuit(1)
        operator = create_mock_operator(paulis=["Y"], coeffs=[1.0])

        circuit.pauli_evolution(operator, 0.5)

        assert circuit.ops == [
            ("sdag", 0),
            ("h", 0),
            ("rz", 0, 1.0),
            ("h", 0),
            ("s", 0),
        ]

    def test_pauli_evolution_with_multi_qubit_chain(self):
        circuit = SpyCircuit(3)
        operator = create_mock_operator(paulis=["XX"], coeffs=[1.0])

        circuit.pauli_evolution(operator, 0.5)

        assert circuit.ops == [
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
        circuit = SpyCircuit(1)
        operator = create_mock_operator(paulis=["Z"], coeffs=[theta[0]])

        # Symbolic coefficients currently cause TypeError when converted to float
        with pytest.raises(TypeError, match="is not numeric"):
            circuit.pauli_evolution(operator, 0.5)

    def test_pauli_evolution_rejects_multi_term_operator(self):
        circuit = SpyCircuit(1)
        operator = create_mock_operator(paulis=["X", "Z"], coeffs=[1.0, 0.5])

        with pytest.raises(ValueError, match="single Pauli strings"):
            circuit.pauli_evolution(operator, 0.5)

    def test_pauli_evolution_rejects_complex_coefficients(self, monkeypatch):
        circuit = SpyCircuit(1)
        operator = create_mock_operator(paulis=["X"], coeffs=[1 + 1j])

        monkeypatch.setattr(quantum_circuit_module.np, "real_if_close", lambda value: value)

        with pytest.raises(ValueError, match="Complex coefficients are not supported"):
            circuit.pauli_evolution(operator, 0.5)

    def test_pauli_evolution_rejects_unknown_pauli(self):
        circuit = SpyCircuit(1)
        operator = create_mock_operator(paulis=["A"], coeffs=[1.0])

        with pytest.raises(ValueError, match="Unknown Pauli operator: A"):
            circuit.pauli_evolution(operator, 0.5)


class TestQuantumCircuitControlledPauliEvolution:

    def test_controlled_pauli_evolution_identity_pauli(self):
        circuit = SpyCircuit(1)
        operator = create_mock_operator(paulis=["I"], coeffs=[1.0])

        circuit.controlled_pauli_evolution(operator, 0.25, control_qubit=0)

        assert circuit.ops == [("rz", 0, -0.25)]

    def test_controlled_pauli_evolution_with_basis_change(self):
        circuit = SpyCircuit(2)
        operator = create_mock_operator(paulis=["X"], coeffs=[1.0])

        circuit.controlled_pauli_evolution(operator, 0.5, control_qubit=1)

        assert circuit.ops == [("h", 0), ("crz", 1, 0, 1.0), ("h", 0)]

    def test_controlled_pauli_evolution_with_y_basis_change(self):
        circuit = SpyCircuit(2)
        operator = create_mock_operator(paulis=["Y"], coeffs=[1.0])

        circuit.controlled_pauli_evolution(operator, 0.5, control_qubit=1)

        assert circuit.ops == [
            ("sdag", 0),
            ("h", 0),
            ("crz", 1, 0, 1.0),
            ("h", 0),
            ("s", 0),
        ]

    def test_controlled_pauli_evolution_with_multi_qubit_chain(self):
        circuit = SpyCircuit(3)
        operator = create_mock_operator(paulis=["XX"], coeffs=[1.0])

        circuit.controlled_pauli_evolution(operator, 0.5, control_qubit=2)

        assert circuit.ops == [
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
        circuit = SpyCircuit(2)
        operator = create_mock_operator(paulis=["Z"], coeffs=[theta[0]])

        # Symbolic coefficients currently cause TypeError when converted to float
        with pytest.raises(TypeError, match="is not numeric"):
            circuit.controlled_pauli_evolution(operator, 0.5, control_qubit=1)

    def test_controlled_pauli_evolution_rejects_complex_coefficients(self, monkeypatch):
        circuit = SpyCircuit(1)
        operator = create_mock_operator(paulis=["X"], coeffs=[1 + 1j])

        monkeypatch.setattr(quantum_circuit_module.np, "real_if_close", lambda value: value)

        with pytest.raises(ValueError, match="Complex coefficients are not supported"):
            circuit.controlled_pauli_evolution(operator, 0.5, control_qubit=0)

    def test_controlled_pauli_evolution_rejects_multi_term_operator(self):
        circuit = SpyCircuit(1)
        operator = create_mock_operator(paulis=["X", "Z"], coeffs=[1.0, 0.5])

        with pytest.raises(ValueError, match="single Pauli strings"):
            circuit.controlled_pauli_evolution(operator, 0.5, control_qubit=0)

    def test_controlled_pauli_evolution_rejects_unknown_pauli(self):
        circuit = SpyCircuit(1)
        operator = create_mock_operator(paulis=["A"], coeffs=[1.0])

        with pytest.raises(ValueError, match="Unknown Pauli operator: A"):
            circuit.controlled_pauli_evolution(operator, 0.5, control_qubit=0)


class TestQuantumCircuitOperations:

    GATE_CASES = [
        ("h", (0,), "h", (0,)),
        ("s", (0,), "s", (0,)),
        ("sdag", (0,), "sdg", (0,)),
        ("t", (1,), "t", (1,)),
        ("tdag", (1,), "tdg", (1,)),
        ("p", (2, 0.1), "p", (0.1, 2)),  # Qiskit: p(θ, qubit)
        ("cp", (0, 1, 0.2), "cp", (0.2, 0, 1)),  # Qiskit: cp(θ, c, t)
        ("x", (0,), "x", (0,)),
        ("y", (1,), "y", (1,)),
        ("z", (2,), "z", (2,)),
        ("rx", (0, 0.3), "rx", (0.3, 0)),  # Qiskit: rx(θ, qubit)
        ("ry", (1, 0.4), "ry", (0.4, 1)),
        ("rz", (2, 0.5), "rz", (0.5, 2)),
        ("cx", (0, 1), "cx", (0, 1)),
        ("cy", (1, 2), "cy", (1, 2)),
        ("cz", (0, 2), "cz", (0, 2)),
        ("cnot", (1, 2), "cx", (1, 2)),
        ("ecr", (0, 1), "ecr", (0, 1)),
        ("crx", (0, 1, 0.6), "crx", (0.6, 0, 1)),  # Qiskit: crx(θ, c, t)
        ("cry", (1, 2, 0.7), "cry", (0.7, 1, 2)),
        ("crz", (0, 2, 0.8), "crz", (0.8, 0, 2)),
        ("rxx", (0, 1, 0.9), "rxx", (0.9, 0, 1)),  # Qiskit: rxx(θ, q0, q1)
        ("ryy", (1, 2, 1.0), "ryy", (1.0, 1, 2)),
        ("rzz", (0, 2, 1.1), "rzz", (1.1, 0, 2)),
        ("rzx", (0, 1, 1.2), "rzx", (1.2, 0, 1)),
        ("swap", (0, 2), "swap", (0, 2)),
        ("barrier", ([0, 1, 2],), "barrier", ([0, 1, 2],)),
    ]

    @pytest.mark.parametrize(
        "method, args, expected_gate, expected_args", GATE_CASES, ids=[c[0] for c in GATE_CASES]
    )
    def test_gate_delegates_to_qiskit(self, method, args, expected_gate, expected_args):
        mock_qiskit = MagicMock()

        with patch("executor.quantum_circuit.QiskitQuantumCircuit", return_value=mock_qiskit):
            circuit = QuantumCircuit(3)
            getattr(circuit, method)(*args)

        getattr(mock_qiskit, expected_gate).assert_called_once_with(*expected_args)

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
