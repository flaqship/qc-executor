"""Tests for the Pauli-propagation circuit now that it is built on the shared IR.

This backend used to convert circuits through Qiskit and rejected several gates
outright.  It now records instructions in the framework-independent IR and
lowers whatever the engine cannot execute, so those gates work and no Qiskit
code is involved.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import qc_executor.pauli_propagation
from qc_executor import Executor, QuantumCircuit, QuantumOperator
from qc_executor.base.gate_set import GATE_DEFS, OpCode
from qc_executor.parameters import Parameters
from qc_executor.pauli_propagation import PauliPropagationCircuit, PauliPropagationOperator
from qc_executor.pauli_propagation.utils.gates import CliffordGate, LayerBarrier, PauliRotation


def _reference(circuit: QuantumCircuit, observable: QuantumOperator, **values) -> float:
    """Expectation value from a statevector backend, for cross-checking."""
    executor = Executor.create("qiskit")
    return float(np.real(executor.expectation_value(circuit, observable, **values)))


class TestNoQiskitInvolved:
    def test_the_backend_package_does_not_reference_qiskit(self):
        """The converter that made this backend depend on Qiskit is gone."""
        root = pathlib.Path(qc_executor.pauli_propagation.__file__).parent
        offenders = [
            path.relative_to(root).as_posix()
            for path in root.rglob("*.py")
            if "qiskit" in path.read_text(encoding="utf-8")
        ]

        assert not offenders, f"Qiskit still referenced in: {offenders}"


class TestGateCompilation:
    def test_cliffords_and_rotations_map_onto_engine_gates(self):
        circuit = PauliPropagationCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.rz(1, 0.3)
        circuit.barrier()

        gates = circuit.gates

        assert [type(g).__name__ for g in gates] == [
            "CliffordGate",
            "CliffordGate",
            "PauliRotation",
            "LayerBarrier",
        ]
        assert gates[1].gate_type == "CNOT"
        assert gates[2].symbols == ["Z"]

    def test_two_qubit_rotation_generators_come_from_the_gate_table(self):
        circuit = PauliPropagationCircuit(2)
        circuit.ryy(0, 1, 0.4)

        gate = circuit.gates[0]

        assert isinstance(gate, PauliRotation)
        assert gate.symbols == list(GATE_DEFS[OpCode.RYY].pauli_rotation)
        assert gate.qubits == [0, 1]

    def test_symbolic_and_numeric_angles_are_split(self):
        theta = Parameters("theta", 1)
        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, 2 * theta[0])
        circuit.ry(0, 0.5)

        symbolic, numeric = circuit.gates

        assert symbolic.param_expr == 2 * theta[0]
        assert symbolic.param_value is None
        assert numeric.param_expr is None
        assert numeric.param_value == pytest.approx(0.5)

    def test_the_compiled_gate_list_is_cached(self):
        circuit = PauliPropagationCircuit(1)
        circuit.h(0)

        assert circuit.native is circuit.native

    def test_mutating_the_circuit_invalidates_the_cache(self):
        circuit = PauliPropagationCircuit(1)
        circuit.h(0)
        before = len(circuit.gates)

        circuit.x(0)

        assert len(circuit.gates) == before + 1


class TestGatesGainedFromLowering:
    """Gates this backend used to reject now lower into its own basis."""

    @pytest.mark.parametrize(
        "build",
        [
            lambda c: c.crx(0, 1, 0.3),
            lambda c: c.cry(0, 1, 0.3),
            lambda c: c.crz(0, 1, 0.3),
            lambda c: c.rzx(0, 1, 0.3),
            lambda c: c.ecr(0, 1),
            lambda c: c.cy(0, 1),
            lambda c: c.cp(0, 1, 0.3),
            lambda c: c.iswap(0, 1),
            lambda c: c.ch(0, 1),
            lambda c: c.sdag(0),
            lambda c: c.tdag(0),
            lambda c: c.u(0, 0.1, 0.2, 0.3),
            lambda c: c.sx(0),
        ],
        ids=[
            "crx",
            "cry",
            "crz",
            "rzx",
            "ecr",
            "cy",
            "cp",
            "iswap",
            "ch",
            "sdag",
            "tdag",
            "u",
            "sx",
        ],
    )
    def test_gate_compiles_into_the_engine_basis(self, build):
        circuit = PauliPropagationCircuit(2)
        build(circuit)

        gates = circuit.gates

        assert gates, "expected the gate to be lowered rather than rejected"
        assert all(isinstance(g, (CliffordGate, PauliRotation, LayerBarrier)) for g in gates)


class TestParameters:
    def test_parameters_are_the_shared_parameter_type(self):
        theta = Parameters("theta", 2)
        circuit = PauliPropagationCircuit(2)
        circuit.rx(0, theta[0])
        circuit.ry(1, theta[1])

        assert circuit.parameters == [theta[0], theta[1]]
        assert circuit.parameter_names == ["theta[0]", "theta[1]"]

    def test_assign_parameters_does_not_consume_the_original(self):
        """The parameter-shift rule compares bound against unbound."""
        theta = Parameters("theta", 1)
        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, theta[0])

        bound = circuit.assign_parameters({theta[0]: 0.5})

        assert bound is not circuit
        assert circuit.is_parameterized, "binding must not mutate the receiver"
        assert not bound.is_parameterized

    def test_replace_gate_returns_a_new_circuit(self):
        theta = Parameters("theta", 1)
        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, theta[0])
        shifted = PauliRotation(["X"], 0, 1, param_value=0.25)

        replaced = circuit.replace_gate(0, shifted)

        assert replaced is not circuit
        assert replaced.gates[0].param_value == pytest.approx(0.25)
        assert circuit.gates[0].param_expr == theta[0]

    def test_replace_gate_validates_the_index(self):
        circuit = PauliPropagationCircuit(1)
        circuit.h(0)

        with pytest.raises(IndexError, match="out of range"):
            circuit.replace_gate(5, LayerBarrier())


class TestEquivalenceWithAStatevectorBackend:
    @pytest.mark.parametrize("angle", [0.0, 0.37, np.pi / 2])
    def test_parameterised_circuit_matches_qiskit(self, angle):
        theta = Parameters("theta", 1)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.ryy(0, 1, theta[0])
        observable = QuantumOperator(["ZI", "IZ"], [1.0, -0.5])

        executor = Executor.create("pauli_propagation")
        result = float(np.real(executor.expectation_value(circuit, observable, theta=[angle])))

        assert result == pytest.approx(_reference(circuit, observable, theta=[angle]), abs=1e-9)

    def test_native_operator_still_works(self):
        circuit = PauliPropagationCircuit(1)
        circuit.h(0)
        observable = PauliPropagationOperator(["X"], [1.0])

        executor = Executor.create("pauli_propagation")

        assert executor.expectation_value(circuit, observable) == pytest.approx(1.0)


class TestGateListConstruction:
    """The circuit can also adopt an engine gate list directly.

    This path exists for the parameter-shift rule, which displaces one gate's
    angle to something no instruction in the original store describes.
    """

    def _symbolic(self):
        theta = Parameters("theta", 1)
        gates = [
            CliffordGate("H", 0, 1),
            PauliRotation(["X"], 0, 1, param_expr=2 * theta[0]),
        ]
        return theta, PauliPropagationCircuit(1, gates=gates)

    def test_adopted_gates_are_used_verbatim(self):
        _, circuit = self._symbolic()

        assert [type(g).__name__ for g in circuit.gates] == [
            "CliffordGate",
            "PauliRotation",
        ]

    def test_parameters_are_read_off_the_gate_list(self):
        theta, circuit = self._symbolic()

        assert circuit.parameters == [theta[0]]
        assert circuit.parameter_names == ["theta[0]"]
        assert circuit.num_parameters == 1
        assert circuit.is_parameterized

    def test_binding_an_adopted_gate_list_is_pure(self):
        theta, circuit = self._symbolic()

        bound = circuit.assign_parameters({theta[0]: 0.25})

        assert bound is not circuit
        assert circuit.is_parameterized
        assert not bound.is_parameterized
        assert bound.gates[1].param_value == pytest.approx(0.5)

    def test_partial_binding_keeps_the_rest_symbolic(self):
        theta = Parameters("theta", 2)
        circuit = PauliPropagationCircuit(
            1, gates=[PauliRotation(["X"], 0, 1, param_expr=theta[0] + theta[1])]
        )

        bound = circuit.assign_parameters({theta[0]: 1.0})

        assert bound.parameters == [theta[1]]

    def test_copy_is_independent(self):
        _, circuit = self._symbolic()

        copied = circuit.copy()

        assert copied is not circuit
        assert copied.gates[0] is not circuit.gates[0]
        assert copied.gates[0].gate_type == circuit.gates[0].gate_type

    def test_barriers_survive_a_copy(self):
        circuit = PauliPropagationCircuit(1, gates=[LayerBarrier()])

        assert isinstance(circuit.copy().gates[0], LayerBarrier)

    def test_hashing_reflects_the_gate_list(self):
        _, first = self._symbolic()
        _, second = self._symbolic()
        other = PauliPropagationCircuit(1, gates=[CliffordGate("X", 0, 1)])

        assert hash(first) == hash(second)
        assert hash(first) != hash(other)

    def test_draw_lists_the_gates(self):
        _, circuit = self._symbolic()

        assert "CliffordGate" in circuit.draw()

    def test_circuit_metrics_counts_gates(self):
        _, circuit = self._symbolic()

        metrics = circuit.circuit_metrics()

        assert metrics["num_gates"] == 2
        assert metrics["num_parameters"] == 1

    def test_a_gate_list_needs_a_qubit_count(self):
        with pytest.raises(ValueError, match="num_qubits is required"):
            PauliPropagationCircuit(gates=[CliffordGate("X", 0, 1)])
