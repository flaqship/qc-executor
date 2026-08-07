"""Tests for the PennyLane circuit now that it is built on the shared IR.

This backend used to reach ``circuit.qiskit_circuit``, run
``qiskit.transpile(target=pennylane_target)`` and walk ``circuit.data``.  It now
lowers the framework-independent IR and walks the instruction store.

The classical-conditioning path is the notable gain: it read
``Instruction.condition``, which Qiskit removed in 2.0, so conditional gates had
silently stopped being applied.  Reading the condition off the IR makes it work
again, and the tests below would fail if a conditioned gate were skipped.
"""

from __future__ import annotations

import numpy as np
import pytest

from qc_executor import Executor, QuantumCircuit, QuantumOperator
from qc_executor.base.circuit_ir import Condition, Instruction
from qc_executor.base.gate_set import OpCode
from qc_executor.parameters import Parameters
from qc_executor.pennylane.pennylane_circuit import PennyLaneCircuit


def _expectation(circuit: QuantumCircuit, label: str) -> float:
    """Run a single Pauli observable on the PennyLane backend."""
    executor = Executor.create("pennylane")
    observable = QuantumOperator([label], [1.0])
    return float(
        np.real(np.asarray(executor.expectation_value(circuit, observable)).reshape(-1)[0])
    )


class TestConditionalGates:
    """Feed-forward: a gate applied only when a measurement says so."""

    def test_a_met_condition_applies_the_gate(self):
        # Qubit 0 is |1>, so measuring it yields 1 and the X on qubit 1 fires.
        circuit = QuantumCircuit(2, 1)
        circuit.x(0)
        circuit.measure(0, 0)
        with circuit.if_(0, 1):
            circuit.x(1)

        assert _expectation(circuit, "IZ") == pytest.approx(-1.0, abs=1e-8)

    def test_an_unmet_condition_leaves_the_gate_out(self):
        # Qubit 0 is |0>, so the measurement yields 0 and the X must not fire.
        circuit = QuantumCircuit(2, 1)
        circuit.measure(0, 0)
        with circuit.if_(0, 1):
            circuit.x(1)

        assert _expectation(circuit, "IZ") == pytest.approx(1.0, abs=1e-8)

    def test_conditioning_on_a_zero_value(self):
        circuit = QuantumCircuit(2, 1)
        circuit.measure(0, 0)
        with circuit.if_(0, 0):
            circuit.x(1)

        assert _expectation(circuit, "IZ") == pytest.approx(-1.0, abs=1e-8)

    def test_the_condition_survives_compilation(self):
        circuit = QuantumCircuit(2, 1)
        circuit.measure(0, 0)
        with circuit.if_(0, 1):
            circuit.x(1)

        native = PennyLaneCircuit(circuit)

        assert (0, 1) in native._pennylane_conditions


class TestConditionExtraction:
    def test_unconditional_instructions_report_none(self):
        assert PennyLaneCircuit._get_gate_condition(Instruction(OpCode.H, (0,))) is None

    def test_a_single_bit_condition_reports_a_bare_index(self):
        instruction = Instruction(OpCode.X, (0,), condition=Condition((2,), 1))

        assert PennyLaneCircuit._get_gate_condition(instruction) == (2, 1)

    def test_a_multi_bit_condition_reports_a_list(self):
        instruction = Instruction(OpCode.X, (0,), condition=Condition((0, 1), 3))

        assert PennyLaneCircuit._get_gate_condition(instruction) == ([0, 1], 3)


class TestMidCircuitMeasurement:
    def test_measurement_is_recorded_with_its_classical_bit(self):
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.measure(0, 1)

        native = PennyLaneCircuit(circuit)

        assert ("measure", [1]) in native._pennylane_gates


class TestLowering:
    @pytest.mark.parametrize(
        "name, build",
        [
            ("sxdag", lambda c: c.sxdag(0)),
            ("crz", lambda c: c.crz(0, 1, 0.4)),
            ("ryy", lambda c: c.ryy(0, 1, 0.4)),
            ("ccx", lambda c: c.ccx(0, 1, 2)),
        ],
    )
    def test_gates_agree_with_a_statevector_reference(self, name, build):
        assert name  # keeps the parametrize id meaningful
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.ry(1, 0.3)
        circuit.rx(2, 0.2)
        build(circuit)
        observable = QuantumOperator(
            ["ZII", "IZI", "IIZ", "XII", "IXI", "YII"], [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]
        )

        pennylane = float(
            np.real(Executor.create("pennylane").expectation_value(circuit, observable))
        )
        reference = float(
            np.real(Executor.create("qiskit").expectation_value(circuit, observable))
        )

        assert pennylane == pytest.approx(reference, abs=1e-8)

    def test_barriers_are_skipped(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.barrier()
        circuit.cx(0, 1)

        native = PennyLaneCircuit(circuit)

        assert len(native._pennylane_gates) == 2


class TestSymbolicAngles:
    def test_parameter_names_and_dimensions(self):
        x = Parameters("x", 2)
        circuit = QuantumCircuit(2)
        circuit.rx(0, x[0])
        circuit.ry(1, 2 * x[1])

        native = PennyLaneCircuit(circuit)

        assert native.parameter_names == ["x"]
        assert native.parameter_dimensions == {"x": 2}

    @pytest.mark.parametrize(
        "angle_factory, label",
        [
            (lambda x, p: x[0], "bare"),
            (lambda x, p: 2 * x[0], "scaled"),
            (lambda x, p: p[0] * x[0], "product"),
        ],
    )
    def test_gradients_match_a_statevector_reference(self, angle_factory, label):
        assert label
        x, p = Parameters("x", 1), Parameters("p", 1)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.ryy(0, 1, angle_factory(x, p))
        observable = QuantumOperator(["IZ"], [1.0])
        values = {"x": [0.8], "p": [0.5]}

        pennylane = np.asarray(
            Executor.create("pennylane").expectation_value_derivatives(
                circuit, observable, "x", **values
            )
        ).reshape(-1)
        reference = np.asarray(
            Executor.create("qiskit").expectation_value_derivatives(
                circuit, observable, "x", **values
            )
        ).reshape(-1)

        assert np.allclose(pennylane, reference, atol=1e-8)


class TestNoQiskitTranspile:
    def test_the_circuit_module_no_longer_imports_qiskit(self):
        import pathlib  # noqa: PLC0415

        import qc_executor.pennylane.pennylane_circuit as module  # noqa: PLC0415

        source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        used = [
            line
            for line in source.splitlines()
            if line.startswith(("import qiskit", "from qiskit"))
            or ("transpile(" in line and not line.lstrip().startswith("#"))
        ]

        assert not used, f"Qiskit still used in pennylane_circuit.py: {used}"


class TestObservableCoefficients:
    """Numeric coefficients must weight their Pauli terms.

    The no-free-parameter path built an unweighted ``qml.sum`` of the Pauli
    words, so any observable whose coefficients were not all 1 silently returned
    the all-ones value.
    """

    @pytest.mark.parametrize(
        "labels, coeffs",
        [
            (["IIZ"], [0.5]),
            (["XII"], [0.3]),
            (["IIZ", "XII"], [0.5, 0.3]),
            (["ZII", "IZI", "IIZ"], [1.0, 0.7, 0.5]),
            (["IIZ"], [-1.5]),
        ],
    )
    def test_coefficients_are_applied(self, labels, coeffs):
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.ry(1, 0.3)
        circuit.rx(2, 0.2)
        circuit.cx(0, 1)
        observable = QuantumOperator(labels, coeffs)

        pennylane = float(
            np.real(Executor.create("pennylane").expectation_value(circuit, observable))
        )
        reference = float(
            np.real(Executor.create("qiskit").expectation_value(circuit, observable))
        )

        assert pennylane == pytest.approx(reference, abs=1e-8)

    def test_scaling_an_observable_scales_the_result(self):
        """The clearest statement of the bug: doubling the coefficient doubles it."""
        circuit = QuantumCircuit(1)
        circuit.ry(0, 0.7)
        executor = Executor.create("pennylane")

        single = float(np.real(executor.expectation_value(circuit, QuantumOperator(["Z"], [1.0]))))
        double = float(np.real(executor.expectation_value(circuit, QuantumOperator(["Z"], [2.0]))))

        assert double == pytest.approx(2 * single, abs=1e-8)
