import pytest
from qiskit.circuit import ParameterVector

from tests.test_utils import FakeOperator, SpyCircuit


class TestQuantumCircuitBasePropertiesAndAliases:
    def test_parameter_ordering_and_count(self):
        circuit = SpyCircuit(1)
        x = ParameterVector("x", 3)
        circuit._free_parameters = {x[2], x[0], x[1]}

        assert circuit.parameters == [x[0], x[1], x[2]]
        assert circuit.num_parameters == 3
        assert circuit.is_parameterized is True

    def test_cnot_alias_calls_cx(self):
        circuit = SpyCircuit(2)
        circuit.cnot(0, 1)
        assert circuit.ops == [("cx", 0, 1)]


class TestPauliString:
    def test_pauli_string_applies_in_qubit_order(self):
        circuit = SpyCircuit(3)
        circuit.pauli_string("XYZ")

        assert circuit.ops == [("x", 0), ("y", 1), ("z", 2)]

    def test_pauli_string_length_mismatch_raises(self):
        circuit = SpyCircuit(2)

        with pytest.raises(ValueError, match="Pauli string length"):
            circuit.pauli_string("X")

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

        circuit.pauli_evolution(op, 2.0, working_qubits=[2, 0, 1])

        # "XZI" is big-endian: X on qubit 0 -> working qubit 2,
        # Z on qubit 1 -> working qubit 0.
        assert ("h", 2) in circuit.ops
        assert ("rz", 0, 2.0) in circuit.ops
        assert ("cx", 2, 0) in circuit.ops

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

        circuit.controlled_pauli_evolution(op, 2.0, control_qubits=0)

        assert circuit.ops == [("crz", 0, 1, 2.0)]

    def test_controlled_pauli_evolution_default_working_qubits_skip_control(self):
        """Without working_qubits, label positions map to the free qubits."""
        circuit = SpyCircuit(3)
        op = FakeOperator("YX", [0.5])

        # Control sits at the highest index, so the free qubits are 0 and 1:
        # label position 0 ("Y") -> qubit 0, position 1 ("X") -> qubit 1.
        circuit.controlled_pauli_evolution(op, 2.0, control_qubits=2)

        assert ("sdag", 0) in circuit.ops
        assert ("h", 0) in circuit.ops
        assert ("h", 1) in circuit.ops
        assert ("cx", 0, 1) in circuit.ops
        assert ("crz", 2, 1, 2.0) in circuit.ops

    def test_controlled_pauli_evolution_default_working_qubits_low_control(self):
        """A control at index 0 shifts the operator onto the higher qubits."""
        circuit = SpyCircuit(3)
        op = FakeOperator("ZZ", [0.5])

        circuit.controlled_pauli_evolution(op, 2.0, control_qubits=0)

        assert ("cx", 1, 2) in circuit.ops
        assert ("crz", 0, 2, 2.0) in circuit.ops

    def test_controlled_pauli_evolution_out_of_range_working_qubits_raise(self):
        circuit = SpyCircuit(2)
        op = FakeOperator("ZZ", [0.5])

        with pytest.raises(ValueError, match="out of range"):
            circuit.controlled_pauli_evolution(op, 1.0, working_qubits=[0, 5])

    def test_controlled_pauli_evolution_invalid_control_state_raises(self):
        circuit = SpyCircuit(2)
        op = FakeOperator("Z", [0.5])

        with pytest.raises(ValueError, match="control_state entries must be"):
            circuit.controlled_pauli_evolution(op, 1.0, control_qubits=0, control_state="o")

    def test_controlled_pauli_evolution_list_form_with_control_states(self):
        """Each operator gets its own control qubit and control state."""
        circuit = SpyCircuit(4)
        ops = [FakeOperator("Z", [0.5]), FakeOperator("Z", [0.5])]

        circuit.controlled_pauli_evolution(
            ops,
            [2.0, 2.0],
            working_qubits=[[1], [3]],
            control_qubits=[0, 2],
            control_state=["0", "1"],
        )

        # control_state "0" brackets the controlled rotation with X gates
        assert circuit.ops == [
            ("x", 0),
            ("crz", 0, 1, 2.0),
            ("x", 0),
            ("crz", 2, 3, 2.0),
        ]

    def test_controlled_pauli_evolution_unknown_pauli_raises(self):
        circuit = SpyCircuit(2)
        op = FakeOperator("A", [1.0])

        with pytest.raises(ValueError, match="Unknown Pauli operator"):
            circuit.controlled_pauli_evolution(op, 1.0, control_qubits=0)

    def test_controlled_pauli_evolution_complex_coeff_raises(self):
        circuit = SpyCircuit(2)
        op = FakeOperator("Z", [1 + 1j])

        with pytest.raises(ValueError, match="Complex coefficients are not supported"):
            circuit.controlled_pauli_evolution(op, 1.0, control_qubits=0)

    def test_controlled_pauli_evolution_multi_term_coeffs_raises(self):
        circuit = SpyCircuit(2)
        op = FakeOperator("Z", [1.0, 2.0])

        with pytest.raises(ValueError, match="single Pauli strings"):
            circuit.controlled_pauli_evolution(op, 1.0, control_qubits=0)

    def test_controlled_pauli_evolution_yx_chain_explicit_working_qubits(self):
        circuit = SpyCircuit(3)
        op = FakeOperator("YX", [0.5])

        circuit.controlled_pauli_evolution(op, 2.0, working_qubits=[0, 1], control_qubits=2)

        # "YX" is big-endian: Y on qubit 0, X on qubit 1.
        assert ("sdag", 0) in circuit.ops
        assert ("h", 0) in circuit.ops
        assert ("h", 1) in circuit.ops
        assert ("cx", 0, 1) in circuit.ops
        assert ("crz", 2, 1, 2.0) in circuit.ops
        assert ("s", 0) in circuit.ops

    def test_controlled_pauli_evolution_control_state_zero_conjugates_with_x(self):
        circuit = SpyCircuit(2)
        op = FakeOperator("Z", [0.5])

        circuit.controlled_pauli_evolution(
            op, 2.0, working_qubits=[1], control_qubits=0, control_state="0"
        )

        assert circuit.ops == [("x", 0), ("crz", 0, 1, 2.0), ("x", 0)]

    def test_controlled_pauli_evolution_list_of_operators_disjoint_qubits(self):
        circuit = SpyCircuit(4)
        ops = [FakeOperator("ZI", [1.0]), FakeOperator("IZ", [1.0])]

        circuit.controlled_pauli_evolution(ops, [0.5, 0.5], working_qubits=[[2, 3], [0, 1]])

        # Big-endian labels: first operator "ZI" acts on qubit 0 of
        # [2, 3] -> qubit 2; second operator "IZ" acts on qubit 1 of
        # [0, 1] -> qubit 1.
        assert ("rz", 2, 1.0) in circuit.ops
        assert ("rz", 1, 1.0) in circuit.ops

    def test_controlled_pauli_evolution_overlapping_working_qubits_raise(self):
        circuit = SpyCircuit(2)
        ops = [FakeOperator("Z", [1.0]), FakeOperator("Z", [1.0])]

        with pytest.raises(ValueError, match="No distinct support qubits"):
            circuit.controlled_pauli_evolution(ops, [0.5, 0.5], working_qubits=[[0], [0]])
