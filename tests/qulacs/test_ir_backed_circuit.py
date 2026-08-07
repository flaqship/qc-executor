"""Tests for the Qulacs circuit now that it is built on the shared IR.

This backend used to reach ``circuit.qiskit_circuit``, run
``qiskit.transpile(target=qulacs_target)`` and walk ``circuit.data``.  It now
lowers the framework-independent IR with the shared decomposition pass and walks
that, and takes angle derivatives with ``sympy.diff`` rather than Qiskit's
``ParameterExpression.gradient``.
"""

from __future__ import annotations

import numpy as np
import pytest

from qc_executor import Executor, QuantumCircuit, QuantumOperator
from qc_executor.base.gate_set import OpCode
from qc_executor.parameters import Parameters
from qc_executor.qulacs import QulacsCircuit

#: Gates outside the Qulacs basis, which therefore exercise the lowering pass.
LOWERED_GATES = {
    "cy": lambda c: c.cy(0, 1),
    "ch": lambda c: c.ch(0, 1),
    "cs": lambda c: c.cs(0, 1),
    "csx": lambda c: c.csx(0, 1),
    "iswap": lambda c: c.iswap(0, 1),
    "ecr": lambda c: c.ecr(0, 1),
    "cp": lambda c: c.cp(0, 1, 0.4),
    "crx": lambda c: c.crx(0, 1, 0.4),
    "cry": lambda c: c.cry(0, 1, 0.4),
    "crz": lambda c: c.crz(0, 1, 0.4),
    "rxx": lambda c: c.rxx(0, 1, 0.4),
    "ryy": lambda c: c.ryy(0, 1, 0.4),
    "rzz": lambda c: c.rzz(0, 1, 0.4),
    "rzx": lambda c: c.rzx(0, 1, 0.4),
    "u": lambda c: c.u(0, 0.1, 0.2, 0.3),
    "sx": lambda c: c.sx(0),
    "cswap": lambda c: c.cswap(0, 1, 2),
}

OBSERVABLE = QuantumOperator(
    ["ZII", "IZI", "IIZ", "XII", "IXI", "YII"], [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]
)


def _entangled(build) -> QuantumCircuit:
    """A three-qubit circuit in a general state, plus the gate under test."""
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.ry(1, 0.3)
    circuit.rx(2, 0.2)
    build(circuit)
    return circuit


class TestLowering:
    @pytest.mark.parametrize("name", sorted(LOWERED_GATES))
    def test_gates_outside_the_basis_agree_with_a_statevector_reference(self, name):
        circuit = _entangled(LOWERED_GATES[name])

        qulacs = float(np.real(Executor.create("qulacs").expectation_value(circuit, OBSERVABLE)))
        reference = float(
            np.real(Executor.create("qiskit").expectation_value(circuit, OBSERVABLE))
        )

        assert qulacs == pytest.approx(reference, abs=1e-8)

    def test_the_compiled_circuit_uses_only_supported_opcodes(self):
        circuit = _entangled(LOWERED_GATES["crz"])

        native = QulacsCircuit(circuit)

        assert {
            OpCode(op) for op, _, _ in native._ir.iter_ops()
        } <= QulacsCircuit.supported_opcodes()

    def test_barriers_are_skipped(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.barrier()
        circuit.cx(0, 1)

        native = QulacsCircuit(circuit)

        assert native._operation_list == ["h", "cx"]


class TestSymbolicAngles:
    def test_parameters_are_the_shared_parameter_type(self):
        x = Parameters("x", 2)
        circuit = QuantumCircuit(2)
        circuit.rx(0, x[0])
        circuit.ry(1, 2 * x[1])

        native = QulacsCircuit(circuit)

        assert native.free_parameters == {x[0], x[1]}
        assert native.parameter_dimensions == {"x": 2}

    @pytest.mark.parametrize(
        "angle_factory, label",
        [
            (lambda x, p: x[0], "bare"),
            (lambda x, p: 2 * x[0], "scaled"),
            (lambda x, p: p[0] * x[0], "product"),
            (lambda x, p: 2 * p[0] * x[0], "scaled_product"),
            (lambda x, p: x[0] + p[0], "sum"),
        ],
    )
    def test_gradients_match_a_statevector_reference(self, angle_factory, label):
        """Exercises the sympy.diff chain rule that replaced Qiskit's gradient()."""
        assert label
        x, p = Parameters("x", 1), Parameters("p", 1)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.ryy(0, 1, angle_factory(x, p))
        observable = QuantumOperator(["IZ"], [1.0])
        values = {"x": [0.8], "p": [0.5]}

        qulacs = np.asarray(
            Executor.create("qulacs").expectation_value_derivatives(
                circuit, observable, "x", **values
            )
        ).reshape(-1)
        reference = np.asarray(
            Executor.create("qiskit").expectation_value_derivatives(
                circuit, observable, "x", **values
            )
        ).reshape(-1)

        assert np.allclose(qulacs, reference, atol=1e-8)

    def test_a_constant_derivative_still_produces_a_callable(self):
        """A linear angle differentiates to a number, not an expression."""
        x = Parameters("x", 1)
        circuit = QuantumCircuit(1)
        circuit.rx(0, 3 * x[0])

        native = QulacsCircuit(circuit)

        assert native._func_grad_list[0][0]() == pytest.approx(-3.0)


class TestNoQiskitTranspile:
    def test_the_circuit_module_no_longer_imports_qiskit(self):
        import pathlib  # noqa: PLC0415

        import qc_executor.qulacs.qulacs_circuit as module  # noqa: PLC0415

        source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        imports = [
            line
            for line in source.splitlines()
            if line.startswith(("import qiskit", "from qiskit"))
            or ("transpile(" in line and not line.lstrip().startswith("#"))
        ]

        assert not imports, f"Qiskit still used in qulacs_circuit.py: {imports}"
