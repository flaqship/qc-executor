"""Tests carried over from ``integration-support`` for behaviour sQUlearn relies on.

Merged in with ``integration-support``; the tests both branches share live in
``test_circuit_base.py``, which is kept exactly as on ``integration-support-ir``.
"""

import pytest

from tests.integration_support_helpers import FakeOperator, SpyCircuit


class RecordingComposeCircuit(SpyCircuit):
    def _backend_specific_compose(self, qc, qubits, clbits, new_parameters):
        # pylint: disable-next=attribute-defined-outside-init
        self.compose_args = (
            qc,
            qubits,
            clbits,
            new_parameters,
        )
        return self


class TestComposeValidation:
    def test_identity_mapping_requires_equal_qubit_counts(self):
        with pytest.raises(ValueError, match="same number of qubits"):
            RecordingComposeCircuit(3).compose(SpyCircuit(2))

    def test_mapping_length_must_match(self):
        with pytest.raises(ValueError, match="Length of qubits mapping"):
            RecordingComposeCircuit(2).compose(SpyCircuit(1), qubits=[0, 1])

    def test_mapping_indices_must_be_in_range(self):
        with pytest.raises(ValueError, match="out of range"):
            RecordingComposeCircuit(2).compose(SpyCircuit(2), qubits=[0, 5])

    def test_mapping_indices_must_be_unique(self):
        with pytest.raises(ValueError, match="duplicate"):
            RecordingComposeCircuit(2).compose(SpyCircuit(2), qubits=[1, 1])


class TestPauliString:
    def test_pauli_string_applies_in_qubit_order(self):
        circuit = SpyCircuit(3)
        circuit.pauli_string("XYZ")

        assert circuit.ops == [("x", 0), ("y", 1), ("z", 2)]


class TestControlledPauliEvolution:
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
