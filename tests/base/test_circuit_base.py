import pytest

from qc_executor import QuantumCircuit
from qc_executor.parameters import Parameters
from tests.test_utils import SpyCircuit


class FakePauli:
    def __init__(self, label: str):
        self._label = label

    def to_label(self) -> str:
        return self._label


class FakeOperator:
    def __init__(self, label: str, coeffs):
        self.paulis = [label]
        self.coeffs = coeffs


class TestQuantumCircuitBasePropertiesAndAliases:
    def test_parameter_ordering_and_count(self):
        circuit = QuantumCircuit(1)
        x = Parameters("x", 3)
        for index in (2, 0, 1):
            circuit.rx(0, x[index])

        assert circuit.parameters == [x[0], x[1], x[2]]
        assert circuit.num_parameters == 3
        assert circuit.is_parameterized is True

    def test_cnot_alias_calls_cx(self):
        circuit = SpyCircuit(2)
        circuit.cnot(0, 1)
        assert circuit.ops == [("cx", 0, 1)]

    def test_available_gates_lists_the_gate_methods(self):
        gates = QuantumCircuit.available_gates()

        assert {"h", "cx", "rx", "crz"} <= gates
        assert "compose" not in gates

    def test_fixate_parameters_rejects_a_wrong_value_count(self):
        circuit = QuantumCircuit(1)
        x = Parameters("x", 2)
        circuit.rx(0, x[0])
        circuit.ry(0, x[1])

        with pytest.raises(ValueError, match="Expected 2 parameter values, got 1"):
            circuit.fixate_parameters([0.1])

    def test_clear_removes_every_instruction_but_keeps_the_width(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.rx(1, Parameters("x", 1)[0])

        circuit.clear()

        assert circuit.num_qubits == 2
        assert circuit.circuit_metrics() == {}
        assert circuit.num_parameters == 0

    def test_compose_reindexes_a_sparse_parameter_vector_from_zero(self):
        p = Parameters("p", 3)
        first = QuantumCircuit(1)
        first.rx(0, p[2])
        second = QuantumCircuit(1)
        second.ry(0, p[2])

        first.compose(second)

        assert first.parameters == [p[0], p[1]]


class TestPauliString:
    def test_pauli_string_reads_qubit_zero_leftmost(self):
        circuit = SpyCircuit(3)
        circuit.pauli_string("XYZ")

        assert circuit.ops == [("x", 0), ("y", 1), ("z", 2)]

    def test_pauli_string_length_mismatch_raises(self):
        circuit = SpyCircuit(2)

        with pytest.raises(ValueError, match="Pauli string length"):
            circuit.pauli_string("X")

    def test_pauli_string_unknown_character_raises(self):
        circuit = SpyCircuit(2)

        with pytest.raises(ValueError, match="Unknown Pauli operator: A"):
            circuit.pauli_string("XA")

    def test_pauli_string_identity_only_has_no_effect(self):
        circuit = SpyCircuit(2)
        circuit.pauli_string("II")

        assert not circuit.ops


class TestPauliEvolution:
    def test_pauli_evolution_with_z_applies_single_rz(self):
        circuit = SpyCircuit(1)
        op = FakeOperator("Z", [0.5])

        circuit.pauli_evolution(op, 2.0)

        assert circuit.ops == [("rz", 0, 2.0)]

    def test_pauli_evolution_with_y_applies_basis_change_and_inverse(self):
        circuit = SpyCircuit(1)
        op = FakeOperator("Y", [1.0])

        circuit.pauli_evolution(op, 0.25)

        assert circuit.ops == [
            ("sdag", 0),
            ("h", 0),
            ("rz", 0, 0.5),
            ("h", 0),
            ("s", 0),
        ]

    def test_pauli_evolution_complex_coeff_raises(self):
        circuit = SpyCircuit(1)
        op = FakeOperator("Z", [1 + 1j])

        with pytest.raises(ValueError, match="Complex coefficients are not supported"):
            circuit.pauli_evolution(op, 1.0)

    def test_pauli_evolution_multi_term_coeffs_raises(self):
        circuit = SpyCircuit(1)
        op = FakeOperator("Z", [1.0, 2.0])

        with pytest.raises(ValueError, match="single Pauli strings"):
            circuit.pauli_evolution(op, 1.0)

    def test_pauli_evolution_unknown_pauli_raises(self):
        circuit = SpyCircuit(1)
        op = FakeOperator("A", [1.0])

        with pytest.raises(ValueError, match="Unknown Pauli operator"):
            circuit.pauli_evolution(op, 1.0)

    def test_pauli_evolution_all_identity_has_no_effect(self):
        circuit = SpyCircuit(3)
        op = FakeOperator("III", [1.0])

        circuit.pauli_evolution(op, 1.0)

        assert not circuit.ops

    def test_pauli_evolution_respects_explicit_working_qubits(self):
        circuit = SpyCircuit(3)
        op = FakeOperator("XZI", [0.5])

        # Label position i acts on working_qubits[i]: X on 2, Z on 0.
        circuit.pauli_evolution(op, 2.0, working_qubits=[2, 0, 1])

        assert circuit.ops == [("h", 2), ("cx", 2, 0), ("rz", 0, 2.0), ("cx", 2, 0), ("h", 2)]

    def test_pauli_evolution_with_parameterized_coeff_runs(self):
        coeff = 1.0
        circuit = SpyCircuit(1)
        op = FakeOperator("Z", [coeff])

        circuit.pauli_evolution(op, 2.0)

        assert len(circuit.ops) == 1
        assert circuit.ops[0][0] == "rz"
        assert circuit.ops[0][1] == 0


class TestControlledPauliEvolution:
    def test_controlled_pauli_evolution_identity_only_rotates_control(self):
        circuit = SpyCircuit(1)
        op = FakeOperator("I", [0.5])

        assert circuit.controlled_pauli_evolution(op, 4.0, control_qubits=0) is None
        assert circuit.ops == [("rz", 0, -2.0)]

    def test_controlled_pauli_evolution_nontrivial_uses_crz(self):
        circuit = SpyCircuit(2)
        op = FakeOperator("Z", [0.5])

        circuit.controlled_pauli_evolution(op, 2.0, control_qubits=1)

        assert circuit.ops == [("crz", 1, 0, 2.0)]

    def test_controlled_pauli_evolution_unknown_pauli_raises(self):
        circuit = SpyCircuit(2)
        op = FakeOperator("A", [1.0])

        with pytest.raises(ValueError, match="Unknown Pauli operator"):
            circuit.controlled_pauli_evolution(op, 1.0, control_qubits=0)

    def test_controlled_pauli_evolution_complex_coeff_raises(self):
        circuit = SpyCircuit(1)
        op = FakeOperator("Z", [1 + 1j])

        with pytest.raises(ValueError, match="Complex coefficients are not supported"):
            circuit.controlled_pauli_evolution(op, 1.0, control_qubits=0)

    def test_controlled_pauli_evolution_multi_term_coeffs_raises(self):
        circuit = SpyCircuit(1)
        op = FakeOperator("Z", [1.0, 2.0])

        with pytest.raises(ValueError, match="single Pauli strings"):
            circuit.controlled_pauli_evolution(op, 1.0, control_qubits=0)

    def test_controlled_pauli_evolution_yx_chain_default_working_qubits(self):
        circuit = SpyCircuit(3)
        op = FakeOperator("YX", [0.5])

        # Y on qubit 0, X on qubit 1; the free qubits 0 and 1 are used in order.
        circuit.controlled_pauli_evolution(op, 2.0, control_qubits=2)

        assert circuit.ops == [
            ("sdag", 0),
            ("h", 0),
            ("h", 1),
            ("cx", 0, 1),
            ("crz", 2, 1, 2.0),
            ("cx", 0, 1),
            ("h", 0),
            ("s", 0),
            ("h", 1),
        ]


class TestControlledPauliEvolutionValidation:
    def test_a_single_working_qubit_can_be_given_as_an_int(self):
        circuit = SpyCircuit(2)
        op = FakeOperator("Z", [0.5])

        circuit.controlled_pauli_evolution(op, 2.0, working_qubits=1)

        assert circuit.ops == [("rz", 1, 2.0)]

    def test_non_operator_raises(self):
        circuit = SpyCircuit(1)

        with pytest.raises(TypeError, match="quantum operator or a list thereof"):
            circuit.controlled_pauli_evolution("Z", 1.0)

    def test_non_operator_inside_a_list_raises(self):
        circuit = SpyCircuit(1)

        with pytest.raises(TypeError, match="Expected a quantum operator, got str"):
            circuit.controlled_pauli_evolution(["Z"], 1.0)

    def test_one_parameter_per_operator_is_required(self):
        circuit = SpyCircuit(2)
        ops = [FakeOperator("Z", [1.0]), FakeOperator("Z", [1.0])]

        with pytest.raises(ValueError, match="parameter must have one entry per operator, got 3"):
            circuit.controlled_pauli_evolution(ops, [1.0, 2.0, 3.0])

    def test_out_of_range_control_qubit_raises(self):
        circuit = SpyCircuit(2)
        op = FakeOperator("Z", [1.0])

        with pytest.raises(ValueError, match="Control qubit 5 is out of range"):
            circuit.controlled_pauli_evolution(op, 1.0, control_qubits=5)

    def test_too_few_free_qubits_raise(self):
        circuit = SpyCircuit(2)
        op = FakeOperator("ZZ", [1.0])

        # The control takes one of the two qubits, leaving one for a two-qubit string.
        with pytest.raises(ValueError, match="Not enough qubits left"):
            circuit.controlled_pauli_evolution(op, 1.0, control_qubits=0)

    def test_working_qubits_shorter_than_the_label_raise(self):
        circuit = SpyCircuit(3)
        op = FakeOperator("ZZ", [1.0])

        with pytest.raises(ValueError, match="fewer entries than the Pauli string"):
            circuit.controlled_pauli_evolution(op, 1.0, working_qubits=[0])

    def test_working_qubit_equal_to_the_control_raises(self):
        circuit = SpyCircuit(2)
        op = FakeOperator("Z", [1.0])

        with pytest.raises(ValueError, match="Controlled qubits must be distinct"):
            circuit.controlled_pauli_evolution(op, 1.0, working_qubits=[0], control_qubits=0)
