"""End-to-end tests for mid-circuit measurement, reset and classical conditioning.

The pieces are built in earlier work packages: the IR carries ``MEASURE``,
``RESET`` and a :class:`~qc_executor.base.circuit_ir.Condition` per instruction,
the Qiskit bridge emits ``if_test`` blocks and the PennyLane backend uses
``qml.cond``.  These tests exercise them together and pin the per-backend
contract:

============  ===========  ========  =========
backend       measure      reset     condition
============  ===========  ========  =========
qiskit        yes          yes       yes
pennylane     yes          yes       yes
qulacs        raises       raises    raises
pauli_prop.   raises       raises    raises
============  ===========  ========  =========

Backends that cannot represent an operation raise ``NotImplementedError``.  For
Pauli propagation that is not a limitation to work around but the definition of
the method: it propagates operators in the Heisenberg picture, where there is no
measurement outcome to branch on.
"""

from __future__ import annotations

import numpy as np
import pytest

from qc_executor import Executor, QuantumCircuit, QuantumOperator
from qc_executor.base.decompose import UnsupportedGateError, decompose_ir
from qc_executor.base.gate_set import OpCode
from tests.conftest import INSTALLED_BACKENDS, parametrize_backends, requires_backends


def _expectation(executor, circuit: QuantumCircuit, label: str) -> float:
    """Evaluate one Pauli observable and return it as a real scalar."""
    observable = QuantumOperator([label], [1.0])
    value = executor.expectation_value(circuit, observable)
    return float(np.real(np.asarray(value).reshape(-1)[0]))


def _teleportation(theta: float) -> QuantumCircuit:
    """Build the standard three-qubit teleportation circuit.

    Qubit 0 carries the state ``cos(theta/2)|0> + sin(theta/2)|1>``, qubits 1
    and 2 share a Bell pair, and the two corrections on qubit 2 are conditioned
    on the measurement outcomes.  Qubit 2 ends up holding the original state,
    which is only true if both conditions are honoured -- an unconditional or a
    skipped correction gives a visibly different answer.

    Args:
        theta: Polar angle of the state being teleported.

    Returns:
        The circuit, with two classical bits.
    """
    circuit = QuantumCircuit(3, 2)
    circuit.ry(0, theta)
    circuit.h(1)
    circuit.cx(1, 2)
    circuit.cx(0, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    with circuit.if_(1, 1):
        circuit.x(2)
    with circuit.if_(0, 1):
        circuit.z(2)
    return circuit


class TestConditionSurvivesLowering:
    """A conditioned gate must stay conditioned after decomposition.

    Rewrite rules build their replacements from scratch, so before this was
    fixed a conditioned gate that needed lowering came out of the pass
    unconditional -- and then ran on every shot.  The failure is silent: the
    circuit still executes and still returns numbers.
    """

    def test_every_replacement_inherits_the_condition(self):
        circuit = QuantumCircuit(2, 1)
        circuit.measure(0, 0)
        with circuit.if_(0, 1):
            circuit.cy(0, 1)  # lowered into sdg, cx, s

        lowered = decompose_ir(
            circuit.ir, frozenset({OpCode.H, OpCode.CX, OpCode.S, OpCode.SDG, OpCode.MEASURE})
        )

        gates = [i for i in lowered if i.opcode is not OpCode.MEASURE]
        assert len(gates) == 3
        assert all(i.condition == circuit.ir[1].condition for i in gates)

    def test_a_multi_pass_lowering_keeps_the_condition(self):
        # CRX lowers to CRZ, which lowers again: the condition has to survive
        # every pass, not just the first.
        circuit = QuantumCircuit(2, 1)
        circuit.measure(0, 0)
        with circuit.if_(0, 1):
            circuit.crx(0, 1, 0.4)

        lowered = decompose_ir(
            circuit.ir, frozenset({OpCode.H, OpCode.CX, OpCode.RZ, OpCode.MEASURE})
        )

        gates = [i for i in lowered if i.opcode is not OpCode.MEASURE]
        assert gates
        assert all(i.condition is not None for i in gates)

    def test_unconditional_gates_stay_unconditional(self):
        circuit = QuantumCircuit(2)
        circuit.cy(0, 1)

        lowered = decompose_ir(circuit.ir, frozenset({OpCode.CX, OpCode.S, OpCode.SDG}))

        assert all(instruction.condition is None for instruction in lowered)


@requires_backends("qiskit")
class TestQiskitMidCircuit:
    """Qiskit represents all three operations natively."""

    def test_measurement_and_conditions_reach_the_native_circuit(self):
        from qc_executor.qiskit.qiskit_circuit import QiskitCircuit  # noqa: PLC0415

        native = QiskitCircuit.from_quantum_circuit(_teleportation(0.7)).qiskit_circuit
        names = [instruction.operation.name for instruction in native.data]

        assert names.count("measure") == 2
        assert names.count("if_else") == 2

    def test_reset_reaches_the_native_circuit(self):
        from qc_executor.qiskit.qiskit_circuit import QiskitCircuit  # noqa: PLC0415

        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.reset(0)

        native = QiskitCircuit.from_quantum_circuit(circuit).qiskit_circuit

        assert [instruction.operation.name for instruction in native.data] == ["x", "reset"]

    def test_reset_returns_the_qubit_to_zero(self):
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.reset(0)

        assert _expectation(Executor.create("qiskit"), circuit, "Z") == pytest.approx(1.0)

    def test_teleportation(self):
        """Aer is required: the local primitives reject control flow."""
        pytest.importorskip("qiskit_aer")
        from qiskit_aer import AerSimulator  # noqa: PLC0415

        theta = 0.7
        executor = Executor.create(AerSimulator(), shots=20000, seed=17)
        circuit = _teleportation(theta)

        assert _expectation(executor, circuit, "IIZ") == pytest.approx(np.cos(theta), abs=0.03)
        assert _expectation(executor, circuit, "IIX") == pytest.approx(np.sin(theta), abs=0.03)

    def test_sampling_a_circuit_that_brings_its_own_classical_register(self):
        """Counts used to be read from a hard-coded ``meas`` register.

        ``measure_all`` names its register ``meas``, so only circuits this
        executor measured itself could be sampled; a circuit carrying its own
        mid-circuit measurements brought a register named ``c`` and failed.
        """
        circuit = QuantumCircuit(2, 2)
        circuit.x(0)
        circuit.measure(0, 0)
        circuit.measure(1, 1)

        counts = Executor.create("qiskit", shots=128).sample(circuit)

        # Classical bit 0 is written leftmost, matching the qubit ordering.
        assert counts == {"10": 128}

    def test_sampling_still_works_without_measurements(self):
        circuit = QuantumCircuit(2)
        circuit.x(0)

        counts = Executor.create("qiskit", shots=128).sample(circuit)

        assert counts == {"10": 128}


@requires_backends("pennylane")
class TestPennyLaneMidCircuit:
    """PennyLane covers all three through ``qml.measure`` and ``qml.cond``."""

    def test_reset_returns_the_qubit_to_zero(self):
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.reset(0)

        assert _expectation(Executor.create("pennylane"), circuit, "Z") == pytest.approx(1.0)

    def test_teleportation(self):
        theta = 0.7
        executor = Executor.create("pennylane")
        circuit = _teleportation(theta)

        assert _expectation(executor, circuit, "IIZ") == pytest.approx(np.cos(theta), abs=1e-8)
        assert _expectation(executor, circuit, "IIX") == pytest.approx(np.sin(theta), abs=1e-8)

    def test_teleportation_agrees_with_the_state_it_teleports(self):
        """Both halves of the correction matter, so vary the angle."""
        executor = Executor.create("pennylane")
        for theta in (0.0, 0.4, 1.3, np.pi / 2):
            assert _expectation(executor, _teleportation(theta), "IIZ") == pytest.approx(
                np.cos(theta), abs=1e-8
            )

    def test_a_conditioned_gate_that_needs_lowering(self):
        """Exercises the lowering path with a live condition attached."""
        circuit = QuantumCircuit(2, 1)
        circuit.x(0)
        circuit.measure(0, 0)
        with circuit.if_(0, 1):
            circuit.cy(0, 1)

        # CY on |1>|0> gives |1> (i|1>), so qubit 1 flips to |1>.
        assert _expectation(Executor.create("pennylane"), circuit, "IZ") == pytest.approx(-1.0)


@parametrize_backends(["qulacs", "pauli_propagation"])
class TestUnsupportedBackends:
    """Qulacs and Pauli propagation reject all three, and say why."""

    def test_measurement_raises(self, backend_name):
        circuit = QuantumCircuit(2, 1)
        circuit.h(0)
        circuit.measure(0, 0)

        with pytest.raises(NotImplementedError, match="mid-circuit measurement"):
            Executor.create(backend_name).expectation_value(
                circuit, QuantumOperator(["ZI"], [1.0])
            )

    def test_reset_raises(self, backend_name):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.reset(0)

        with pytest.raises(NotImplementedError, match="reset"):
            Executor.create(backend_name).expectation_value(
                circuit, QuantumOperator(["ZI"], [1.0])
            )

    def test_a_conditioned_gate_raises_rather_than_firing(self, backend_name):
        """The dangerous case: every gate is supported, only the condition is not.

        Nothing here needs lowering, so the circuit compiles happily and the
        conditioned X would simply be applied on every shot.
        """
        circuit = QuantumCircuit(2, 1)
        circuit.h(0)
        with circuit.if_(0, 1):
            circuit.x(1)

        with pytest.raises(NotImplementedError):
            Executor.create(backend_name).expectation_value(
                circuit, QuantumOperator(["ZI"], [1.0])
            )


class TestUnsupportedErrorMessages:
    """Non-unitary opcodes get their own message.

    "no decomposition rule is registered" invites the reader to go and write
    one; no rule can exist for a measurement.
    """

    def test_measurement_is_not_described_as_a_missing_rule(self):
        circuit = QuantumCircuit(1, 1)
        circuit.measure(0, 0)

        with pytest.raises(UnsupportedGateError) as excinfo:
            decompose_ir(circuit.ir, frozenset({OpCode.H}))

        assert "mid-circuit measurement is not supported" in str(excinfo.value)
        assert "decomposition rule" not in str(excinfo.value)

    def test_reset_is_not_described_as_a_missing_rule(self):
        circuit = QuantumCircuit(1)
        circuit.reset(0)

        with pytest.raises(UnsupportedGateError, match="mid-circuit reset is not supported"):
            decompose_ir(circuit.ir, frozenset({OpCode.H}))


class TestStatevectorRejectsCollapse:
    """``statevector`` refuses circuits whose final state is a random sample.

    Qiskit happily simulates a reset by drawing an outcome, so repeated calls
    returned *different* vectors for the same circuit -- and the result cache
    then froze whichever one came first.
    """

    @pytest.mark.parametrize("backend", INSTALLED_BACKENDS)
    @pytest.mark.parametrize("operation", ["measure", "reset"])
    def test_measure_and_reset_are_rejected(self, backend, operation):
        circuit = QuantumCircuit(2, 1)
        circuit.h(0)
        if operation == "measure":
            circuit.measure(0, 0)
        else:
            circuit.reset(0)

        with pytest.raises(NotImplementedError, match="statevector is not defined"):
            Executor.create(backend).statevector(circuit)

    @requires_backends("qiskit")
    def test_a_unitary_circuit_is_unaffected(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        statevector = np.asarray(Executor.create("qiskit").statevector(circuit)).reshape(-1)

        assert np.allclose(statevector, [2**-0.5, 0.0, 0.0, 2**-0.5])

    @requires_backends("qiskit")
    def test_a_list_of_circuits_is_checked_too(self):
        good = QuantumCircuit(1)
        good.h(0)
        bad = QuantumCircuit(1, 1)
        bad.measure(0, 0)

        with pytest.raises(NotImplementedError, match="statevector is not defined"):
            Executor.create("qiskit").statevector([good, bad])

    @requires_backends("pennylane")
    def test_a_native_circuit_is_checked_too(self):
        from qc_executor.pennylane.pennylane_circuit import PennyLaneCircuit  # noqa: PLC0415

        circuit = QuantumCircuit(1, 1)
        circuit.measure(0, 0)

        with pytest.raises(NotImplementedError, match="statevector is not defined"):
            Executor.create("pennylane").statevector(
                PennyLaneCircuit.from_quantum_circuit(circuit)
            )
