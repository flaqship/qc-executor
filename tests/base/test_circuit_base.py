from qiskit.circuit import ParameterVector
import pytest

from executor.base.circuit_base import QuantumCircuitBase


class FakePauli:
    def __init__(self, label: str):
        self._label = label

    def to_label(self) -> str:
        return self._label


class FakeOperator:
    def __init__(self, label: str, coeffs):
        self.paulis = [FakePauli(label)]
        self.coeffs = coeffs


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

    def test_cp_alias_calls_p(self):
        circuit = SpyCircuit(1)
        circuit.cp(0, 0.25)
        assert circuit.ops == [("p", 0, 0.25)]


class TestPauliString:
    def test_pauli_string_applies_reversed_qubit_order(self):
        circuit = SpyCircuit(3)
        circuit.pauli_string("XYZ")

        assert circuit.ops == [("z", 0), ("y", 1), ("x", 2)]

    def test_pauli_string_length_mismatch_raises(self):
        circuit = SpyCircuit(2)

        import pytest

        with pytest.raises(ValueError, match="Pauli string length"):
            circuit.pauli_string("X")

    def test_pauli_string_identity_only_has_no_effect(self):
        circuit = SpyCircuit(2)
        circuit.pauli_string("II")

        assert circuit.ops == []


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
        import pytest

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

        assert circuit.ops == []

    def test_pauli_evolution_respects_explicit_working_qubits(self):
        circuit = SpyCircuit(3)
        op = FakeOperator("XZI", [0.5])

        circuit.pauli_evolution(op, 2.0, working_qubits=[2, 0, 1])

        assert ("h", 1) in circuit.ops
        assert ("rz", 0, 2.0) in circuit.ops
        assert ("cx", 1, 0) in circuit.ops

    def test_pauli_evolution_with_parameterized_coeff_runs(self):
        coeff = ParameterVector("c", 1)[0]
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

        returned = circuit.controlled_pauli_evolution(op, 4.0, control_qubit=0)

        assert returned is None
        assert circuit.ops == [("rz", 0, -2.0)]

    def test_controlled_pauli_evolution_nontrivial_uses_crz(self):
        circuit = SpyCircuit(2)
        op = FakeOperator("Z", [0.5])

        circuit.controlled_pauli_evolution(op, 2.0, control_qubit=1)

        assert circuit.ops == [("crz", 1, 0, 2.0)]

    def test_controlled_pauli_evolution_unknown_pauli_raises(self):
        circuit = SpyCircuit(2)
        op = FakeOperator("A", [1.0])

        with pytest.raises(ValueError, match="Unknown Pauli operator"):
            circuit.controlled_pauli_evolution(op, 1.0, control_qubit=0)

    def test_controlled_pauli_evolution_complex_coeff_raises(self):
        circuit = SpyCircuit(1)
        op = FakeOperator("Z", [1 + 1j])

        with pytest.raises(ValueError, match="Complex coefficients are not supported"):
            circuit.controlled_pauli_evolution(op, 1.0, control_qubit=0)

    def test_controlled_pauli_evolution_multi_term_coeffs_raises(self):
        circuit = SpyCircuit(1)
        op = FakeOperator("Z", [1.0, 2.0])

        with pytest.raises(ValueError, match="single Pauli strings"):
            circuit.controlled_pauli_evolution(op, 1.0, control_qubit=0)

    def test_controlled_pauli_evolution_yx_chain_default_working_qubits(self):
        circuit = SpyCircuit(3)
        op = FakeOperator("YX", [0.5])

        circuit.controlled_pauli_evolution(op, 2.0, control_qubit=2)

        assert ("sdag", 1) in circuit.ops
        assert ("h", 1) in circuit.ops
        assert ("h", 0) in circuit.ops
        assert ("cx", 1, 0) in circuit.ops
        assert ("crz", 2, 0, 2.0) in circuit.ops
        assert ("s", 1) in circuit.ops
