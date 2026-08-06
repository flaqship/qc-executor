"""Tests for `qc_executor.quantum_circuit`."""

from unittest.mock import MagicMock

import pytest
from qiskit.circuit import Parameter as QiskitParameter

from qc_executor import QuantumCircuit
from qc_executor.base.circuit_ir import Condition
from qc_executor.parameters import Parameters
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

    def test_toffoli_is_alias_for_ccx(self):
        reference = QuantumCircuit(3)
        reference.ccx(0, 1, 2)

        circuit = QuantumCircuit(3)
        circuit.toffoli(0, 1, 2)

        toffoli_ops = [instr.operation.name for instr in circuit.qiskit_circuit.data]
        ccx_ops = [instr.operation.name for instr in reference.qiskit_circuit.data]
        assert toffoli_ops == ccx_ops == ["ccx"]


class TestQuantumCircuitPauliString:

    def test_pauli_string_applies_reverse_qubit_order(self):
        circuit = SpyCircuit(3)

        circuit.pauli_string("XYZ")

        assert circuit.ops == [("z", 0), ("y", 1), ("x", 2)]

    def test_pauli_string_skips_identity_paulis(self):
        circuit = SpyCircuit(3)

        circuit.pauli_string("III")

        assert not circuit.ops

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
        """A symbolic coefficient must stay symbolic instead of forcing a float."""
        theta = Parameters("theta", 1)
        circuit = SpyCircuit(1)
        operator = create_mock_operator(paulis=["Z"], coeffs=[theta[0]])

        circuit.pauli_evolution(operator, 0.5)

        ((name, qubit, angle),) = circuit.ops
        assert (name, qubit) == ("rz", 0)
        assert angle.free_symbols == {theta[0]}
        # angle == 2 * coeff * parameter, so theta[0] = 2.0 gives 2.0
        assert float(angle.subs({theta[0]: 2.0})) == pytest.approx(2.0)

    def test_pauli_evolution_with_symbolic_parameter(self):
        """The evolution parameter itself may be symbolic."""
        theta = Parameters("theta", 1)
        circuit = SpyCircuit(1)
        operator = create_mock_operator(paulis=["Z"], coeffs=[0.5])

        circuit.pauli_evolution(operator, theta[0])

        ((_, _, angle),) = circuit.ops
        assert float(angle.subs({theta[0]: 3.0})) == pytest.approx(3.0)

    def test_pauli_evolution_rejects_multi_term_operator(self):
        circuit = SpyCircuit(1)
        operator = create_mock_operator(paulis=["X", "Z"], coeffs=[1.0, 0.5])

        with pytest.raises(ValueError, match="single Pauli strings"):
            circuit.pauli_evolution(operator, 0.5)

    def test_pauli_evolution_rejects_complex_coefficients(self):
        circuit = SpyCircuit(1)
        operator = create_mock_operator(paulis=["X"], coeffs=[1 + 1j])

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
        """A symbolic coefficient must stay symbolic instead of forcing a float."""
        theta = Parameters("theta", 1)
        circuit = SpyCircuit(2)
        operator = create_mock_operator(paulis=["Z"], coeffs=[theta[0]])

        circuit.controlled_pauli_evolution(operator, 0.5, control_qubit=1)

        ((name, control, target, angle),) = circuit.ops
        assert (name, control, target) == ("crz", 1, 0)
        assert float(angle.subs({theta[0]: 2.0})) == pytest.approx(2.0)

    def test_controlled_pauli_evolution_identity_with_symbolic_coefficient(self):
        """The all-identity shortcut must also keep symbolic coefficients."""
        theta = Parameters("theta", 1)
        circuit = SpyCircuit(1)
        operator = create_mock_operator(paulis=["I"], coeffs=[theta[0]])

        circuit.controlled_pauli_evolution(operator, 0.25, control_qubit=0)

        ((name, qubit, angle),) = circuit.ops
        assert (name, qubit) == ("rz", 0)
        assert float(angle.subs({theta[0]: 4.0})) == pytest.approx(-1.0)

    def test_controlled_pauli_evolution_rejects_complex_coefficients(self):
        circuit = SpyCircuit(1)
        operator = create_mock_operator(paulis=["X"], coeffs=[1 + 1j])

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

    #: (builder method, args) -> expected gate name, qubits and params in the IR
    GATE_CASES = [
        ("h", (0,), "h", (0,), ()),
        ("s", (0,), "s", (0,), ()),
        ("sdag", (0,), "sdg", (0,), ()),
        ("t", (1,), "t", (1,), ()),
        ("tdag", (1,), "tdg", (1,), ()),
        ("sx", (0,), "sx", (0,), ()),
        ("sxdag", (0,), "sxdg", (0,), ()),
        ("i", (0,), "id", (0,), ()),
        ("p", (2, 0.1), "p", (2,), (0.1,)),
        ("cp", (0, 1, 0.2), "cp", (0, 1), (0.2,)),
        ("x", (0,), "x", (0,), ()),
        ("y", (1,), "y", (1,), ()),
        ("z", (2,), "z", (2,), ()),
        ("rx", (0, 0.3), "rx", (0,), (0.3,)),
        ("ry", (1, 0.4), "ry", (1,), (0.4,)),
        ("rz", (2, 0.5), "rz", (2,), (0.5,)),
        ("u", (0, 0.1, 0.2, 0.3), "u", (0,), (0.1, 0.2, 0.3)),
        ("cx", (0, 1), "cx", (0, 1), ()),
        ("cy", (1, 2), "cy", (1, 2), ()),
        ("cz", (0, 2), "cz", (0, 2), ()),
        ("ch", (0, 1), "ch", (0, 1), ()),
        ("cs", (0, 1), "cs", (0, 1), ()),
        ("csx", (0, 1), "csx", (0, 1), ()),
        ("cnot", (1, 2), "cx", (1, 2), ()),
        ("ecr", (0, 1), "ecr", (0, 1), ()),
        ("swap", (0, 2), "swap", (0, 2), ()),
        ("iswap", (0, 2), "iswap", (0, 2), ()),
        ("crx", (0, 1, 0.6), "crx", (0, 1), (0.6,)),
        ("cry", (1, 2, 0.7), "cry", (1, 2), (0.7,)),
        ("crz", (0, 2, 0.8), "crz", (0, 2), (0.8,)),
        ("rxx", (0, 1, 0.9), "rxx", (0, 1), (0.9,)),
        ("ryy", (1, 2, 1.0), "ryy", (1, 2), (1.0,)),
        ("rzz", (0, 2, 1.1), "rzz", (0, 2), (1.1,)),
        ("rzx", (0, 1, 1.2), "rzx", (0, 1), (1.2,)),
        ("ccx", (0, 1, 2), "ccx", (0, 1, 2), ()),
        ("toffoli", (0, 1, 2), "ccx", (0, 1, 2), ()),
        ("cswap", (0, 1, 2), "cswap", (0, 1, 2), ()),
        ("barrier", ([0, 1, 2],), "barrier", (0, 1, 2), ()),
        ("reset", (1,), "reset", (1,), ()),
    ]

    @pytest.mark.parametrize(
        "method, args, name, qubits, params", GATE_CASES, ids=[c[0] for c in GATE_CASES]
    )
    def test_gate_appends_the_expected_instruction(self, method, args, name, qubits, params):
        circuit = QuantumCircuit(3)

        getattr(circuit, method)(*args)

        assert len(circuit) == 1
        instruction = circuit[0]
        assert instruction.name == name
        assert instruction.qubits == qubits
        assert instruction.params == params

    def test_single_qubit_gates_broadcast_over_a_sequence(self):
        circuit = QuantumCircuit(3)

        circuit.h([0, 2])

        assert [(i.name, i.qubits) for i in circuit] == [("h", (0,)), ("h", (2,))]

    def test_rotation_broadcasts_the_same_angle(self):
        circuit = QuantumCircuit(2)

        circuit.rx([0, 1], 0.25)

        assert [(i.qubits, i.params) for i in circuit] == [((0,), (0.25,)), ((1,), (0.25,))]

    def test_barrier_defaults_to_every_qubit(self):
        circuit = QuantumCircuit(3)

        circuit.barrier()

        assert circuit[0].qubits == (0, 1, 2)

    def test_out_of_range_qubit_is_rejected(self):
        circuit = QuantumCircuit(2)

        with pytest.raises(ValueError, match="out of range"):
            circuit.h(5)

    def test_repeated_qubits_are_rejected(self):
        circuit = QuantumCircuit(2)

        with pytest.raises(ValueError, match="repeated qubit indices"):
            circuit.cx(1, 1)

    def test_foreign_parameter_types_are_rejected_clearly(self):
        """A framework's own parameter type must not enter the IR silently."""
        circuit = QuantumCircuit(1)

        with pytest.raises(TypeError, match="must be a number or a SymPy expression"):
            circuit.rx(0, QiskitParameter("theta"))

    def test_copy_creates_independent_circuit(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)

        copied = circuit.copy()
        copied.cx(0, 1)

        assert circuit is not copied
        assert circuit.circuit_metrics().get("cx", 0) == 0
        assert copied.circuit_metrics().get("cx", 0) == 1

    def test_compose_combines_circuits(self):
        left = QuantumCircuit(2)
        left.h(0)
        right = QuantumCircuit(2)
        right.cx(0, 1)

        left.compose(right, [0, 1])

        assert left.circuit_metrics() == {"h": 1, "cx": 1}

    def test_compose_remaps_qubits(self):
        left = QuantumCircuit(3)
        right = QuantumCircuit(2)
        right.cx(0, 1)

        left.compose(right, [2, 0])

        assert left[0].qubits == (2, 0)

    def test_compose_rejects_non_quantum_circuit(self):
        circuit = QuantumCircuit(1)

        with pytest.raises(TypeError, match="can only compose with a quantum circuit"):
            circuit.compose("not-a-circuit", [0])

    def test_hash_and_equality_follow_content(self):
        first = QuantumCircuit(1)
        first.h(0)
        second = QuantumCircuit(1)
        second.h(0)
        different = QuantumCircuit(1)
        different.x(0)

        assert first == second
        assert hash(first) == hash(second)
        assert first != different
        assert hash(first) != hash(different)

    def test_hash_changes_when_an_angle_changes(self):
        first = QuantumCircuit(1)
        first.rx(0, 0.1)
        second = QuantumCircuit(1)
        second.rx(0, 0.2)

        assert hash(first) != hash(second)

    def test_str_draws_and_repr_summarises(self):
        circuit = QuantumCircuit(1)
        circuit.h(0)

        assert str(circuit) == "H        [0]"
        assert repr(circuit) == "QuantumCircuit(num_qubits=1, num_clbits=0, instructions=1)"

    def test_invert_reverses_and_adjoints(self):
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.s(0)
        circuit.rx(0, 0.3)

        inverted = circuit.invert()

        assert isinstance(inverted, QuantumCircuit)
        assert inverted is not circuit
        assert [(i.name, i.params) for i in inverted] == [
            ("rx", (-0.3,)),
            ("sdg", ()),
            ("h", ()),
        ]

    def test_invert_rejects_gates_without_an_adjoint(self):
        circuit = QuantumCircuit(2)
        circuit.iswap(0, 1)

        with pytest.raises(NotImplementedError, match="no adjoint in the gate set"):
            circuit.invert()

    def test_circuit_metrics_counts_gate_names(self):
        circuit = QuantumCircuit(2)
        circuit.h([0, 1])
        circuit.cx(0, 1)

        assert circuit.circuit_metrics() == {"h": 2, "cx": 1}


class TestQuantumCircuitParameters:
    def test_assign_parameters_accepts_parameter_objects(self):
        theta = Parameters("theta", 1)
        circuit = QuantumCircuit(1)
        circuit.rx(0, theta[0])

        circuit.assign_parameters({theta[0]: 0.5})

        assert not circuit.is_parameterized
        assert circuit[0].params == (0.5,)

    def test_assign_parameters_accepts_names(self):
        theta = Parameters("theta", 1)
        circuit = QuantumCircuit(1)
        circuit.rx(0, theta[0])

        circuit.assign_parameters({"theta[0]": 0.5})

        assert circuit[0].params == (0.5,)

    def test_partial_assignment_keeps_the_rest_symbolic(self):
        x = Parameters("x", 2)
        circuit = QuantumCircuit(1)
        circuit.rx(0, x[0] + x[1])

        circuit.assign_parameters({x[0]: 1.0})

        assert circuit.parameters == [x[1]]

    def test_parameters_are_sorted_numerically(self):
        x = Parameters("x", 12)
        circuit = QuantumCircuit(1)
        for index in (10, 1, 9):
            circuit.rx(0, x[index])

        assert [p.name for p in circuit.parameters] == ["x[1]", "x[9]", "x[10]"]


class TestQuantumCircuitMidCircuit:
    def test_measure_allocates_classical_bits(self):
        circuit = QuantumCircuit(2)

        circuit.measure(0)

        assert circuit.num_clbits == 1
        assert circuit[0].clbits == (0,)

    def test_measure_all_covers_every_qubit(self):
        circuit = QuantumCircuit(3)

        circuit.measure_all()

        assert circuit.num_clbits == 3
        assert [i.clbits for i in circuit] == [(0,), (1,), (2,)]

    def test_measure_into_explicit_clbits(self):
        circuit = QuantumCircuit(2, 2)

        circuit.measure([0, 1], [1, 0])

        assert [(i.qubits, i.clbits) for i in circuit] == [((0,), (1,)), ((1,), (0,))]

    def test_measure_rejects_mismatched_widths(self):
        circuit = QuantumCircuit(2, 2)

        with pytest.raises(ValueError, match="one classical bit per qubit"):
            circuit.measure([0, 1], [0])

    def test_conditional_scope_gates_appended_instructions(self):
        circuit = QuantumCircuit(2, 1)
        circuit.measure(0, 0)

        with circuit.if_(0, 1):
            circuit.x(1)
        circuit.h(1)

        assert circuit[1].condition == Condition(clbits=(0,), value=1)
        assert circuit[2].condition is None

    def test_condition_on_an_unknown_clbit_is_rejected(self):
        circuit = QuantumCircuit(1)

        with pytest.raises(ValueError, match="out of range"):
            circuit.if_(3, 1)

    def test_nested_conditions_are_rejected(self):
        circuit = QuantumCircuit(1, 2)

        with pytest.raises(RuntimeError, match="Nested classical conditions"):
            with circuit.if_(0, 1):
                with circuit.if_(1, 1):
                    circuit.x(0)

    def test_reset_defaults_to_every_qubit(self):
        circuit = QuantumCircuit(2)

        circuit.reset()

        assert [(i.name, i.qubits) for i in circuit] == [("reset", (0,)), ("reset", (1,))]


class TestQuantumCircuitNative:
    def test_generic_circuit_has_no_native_representation(self):
        circuit = QuantumCircuit(1)
        circuit.h(0)

        with pytest.raises(NotImplementedError, match="no native representation"):
            _ = circuit.native

    def test_qiskit_bridge_emits_matching_instructions(self):
        x = Parameters("x", 1)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.ryy(0, 1, 2 * x[0])

        native = circuit.qiskit_circuit

        assert [instruction.operation.name for instruction in native.data] == ["h", "ryy"]
        assert sorted(p.name for p in native.parameters) == ["x[0]"]

    def test_qiskit_bridge_reuses_parameter_identity(self):
        """Qiskit compares parameters by UUID, so repeated builds must agree."""
        x = Parameters("x", 1)
        circuit = QuantumCircuit(1)
        circuit.rx(0, x[0])

        assert set(circuit.qiskit_circuit.parameters) == set(circuit.qiskit_circuit.parameters)
