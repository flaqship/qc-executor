"""Tests for PauliPropagationCircuit."""

import numpy as np
import pytest
import sympy as sp
from qiskit.circuit import Parameter

from qc_executor import QuantumCircuit
from qc_executor.parameters import Parameters
from qc_executor.pauli_propagation import PauliPropagationCircuit
from qc_executor.pauli_propagation.utils.gates import (
    CliffordGate,
    Gate,
    LayerBarrier,
    PauliRotation,
)


class TestPauliPropagationCircuitBasics:
    def test_basic_gate_storage(self):
        circuit = PauliPropagationCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.barrier([0, 1])

        gates = circuit.gates
        assert len(gates) == 3
        assert isinstance(gates[0], CliffordGate)
        assert isinstance(gates[1], CliffordGate)
        assert isinstance(gates[2], LayerBarrier)

    def test_draw_returns_multiline_representation(self):
        circuit = PauliPropagationCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        drawn = circuit.draw()
        assert "CliffordGate(H" in drawn
        assert "CliffordGate(CNOT" in drawn
        assert "\n" in drawn

    def test_circuit_metrics(self):
        circuit = PauliPropagationCircuit(2)
        circuit.x(0)
        circuit.rxx(0, 1, 0.3)

        metrics = circuit.circuit_metrics()
        assert metrics["num_qubits"] == 2
        assert metrics["num_gates"] == 2
        assert metrics["num_parameters"] == 0


class TestPauliPropagationCircuitConversion:
    def test_from_quantum_circuit_identity(self):
        circuit = PauliPropagationCircuit(1)

        converted = PauliPropagationCircuit.from_quantum_circuit(circuit)
        assert converted is circuit

    def test_from_quantum_circuit_rejects_invalid_type(self):
        with pytest.raises(TypeError, match="expects a generic QuantumCircuit"):
            PauliPropagationCircuit.from_quantum_circuit(object())

    def test_from_quantum_circuit_converts_generic_circuit(self):
        params = Parameters("theta", 1)
        circuit = QuantumCircuit(1)
        circuit.rx(0, params[0])

        converted = PauliPropagationCircuit.from_quantum_circuit(circuit)
        assert isinstance(converted, PauliPropagationCircuit)
        assert converted.num_qubits == 1
        assert converted.is_parameterized
        assert converted.parameters == ["theta[0]"]

    def test_from_quantum_circuit_handles_non_parametric_gates(self):
        circuit = QuantumCircuit(1)
        circuit.h(0)

        converted = PauliPropagationCircuit.from_quantum_circuit(circuit)

        assert isinstance(converted, PauliPropagationCircuit)
        assert not converted.is_parameterized
        assert not converted.parameters


class TestPauliPropagationCircuitParameters:

    def test_parameter_tracking(self):
        params = Parameters("theta", 2)
        circuit = PauliPropagationCircuit(2)
        circuit.rx(0, params[0])
        circuit.rzz(0, 1, params[1])

        assert circuit.is_parameterized
        assert set(circuit.parameters) == {"theta[0]", "theta[1]"}

    def test_assign_parameters(self):
        params = Parameters("theta", 1)
        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, params[0])

        assigned = circuit.assign_parameters({"theta[0]": np.pi / 2})
        assert isinstance(assigned.gates[0], PauliRotation)
        assert assigned.gates[0].param_name is None
        assert np.isclose(assigned.gates[0].param_value, np.pi / 2)

    def test_extract_parameter_handles_qiskit_and_sympy_types(self, monkeypatch):
        symbol = Parameter("theta")
        expr_symbolic, value_symbolic = PauliPropagationCircuit._extract_parameter(symbol)
        assert expr_symbolic is not None
        assert str(expr_symbolic) == "theta"
        assert value_symbolic is None

        class DummyQiskitParameter:
            parameters = set()

        monkeypatch.setattr(
            "qc_executor.pauli_propagation.pauli_propagation_circuit._param_is_constant",
            lambda _: True,
        )
        monkeypatch.setattr(
            "qc_executor.pauli_propagation.pauli_propagation_circuit._param_to_float",
            lambda _: 0.25,
        )
        expr_constant, value_constant = PauliPropagationCircuit._extract_parameter(
            DummyQiskitParameter()
        )
        assert expr_constant is None
        assert np.isclose(value_constant, 0.25)

        expr_sympy_num, value_sympy_num = PauliPropagationCircuit._extract_parameter(sp.Float(0.5))
        assert expr_sympy_num is None
        assert np.isclose(value_sympy_num, 0.5)

        x = sp.Symbol("x")
        expr_sympy_symbolic, value_sympy_symbolic = PauliPropagationCircuit._extract_parameter(
            x + 1
        )
        assert expr_sympy_symbolic == x + 1
        assert value_sympy_symbolic is None

    def test_extract_parameter_rejects_unsupported_type(self):
        with pytest.raises(TypeError, match="Unsupported parameter type"):
            PauliPropagationCircuit._extract_parameter("theta")

    def test_assign_parameters_partial_keeps_symbolic_expression(self):
        theta = Parameter("theta")
        phi = Parameter("phi")
        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, theta + phi)

        assigned = circuit.assign_parameters({"theta": 0.2})

        assert assigned.is_parameterized
        assert set(assigned.parameters) == {"phi"}
        assert isinstance(assigned.gates[0], PauliRotation)
        assert assigned.gates[0].param_expr == sp.Symbol("phi") + 0.2
        assert assigned.gates[0].param_value is None

    def test_assign_parameters_leaves_non_parametric_gates_unchanged(self):
        circuit = PauliPropagationCircuit(1)
        circuit.x(0)
        circuit.rz(0, 0.25)

        assigned = circuit.assign_parameters({"theta": 0.5})

        assert isinstance(assigned.gates[0], CliffordGate)
        assert isinstance(assigned.gates[1], PauliRotation)
        assert assigned.gates[1].param_expr is None
        assert np.isclose(assigned.gates[1].param_value, 0.25)


class TestPauliPropagationCircuitGates:

    def test_single_qubit_helpers_and_phase_helpers(self):
        circuit = PauliPropagationCircuit(2)
        circuit.s([0, 1])
        circuit.t([0, 1])
        circuit.y([0, 1])
        circuit.z([0, 1])
        circuit.sdag(0)
        circuit.tdag(1)
        circuit.p(0, 0.125)

        assert len(circuit.gates) == 11
        assert isinstance(circuit.gates[0], CliffordGate)
        assert circuit.gates[0].gate_type == "S"
        assert isinstance(circuit.gates[2], CliffordGate)
        assert circuit.gates[2].gate_type == "T"
        assert isinstance(circuit.gates[4], CliffordGate)
        assert circuit.gates[4].gate_type == "Y"
        assert isinstance(circuit.gates[6], CliffordGate)
        assert circuit.gates[6].gate_type == "Z"
        assert isinstance(circuit.gates[8], PauliRotation)
        assert circuit.gates[8].symbols == ["Z"]
        assert np.isclose(circuit.gates[8].param_value, -np.pi / 2)
        assert isinstance(circuit.gates[9], PauliRotation)
        assert np.isclose(circuit.gates[9].param_value, -np.pi / 4)
        assert isinstance(circuit.gates[10], PauliRotation)
        assert np.isclose(circuit.gates[10].param_value, 0.125)

    def test_single_qubit_helpers_accept_int_input(self):
        circuit = PauliPropagationCircuit(1)
        circuit.t(0)
        circuit.y(0)
        circuit.z(0)
        circuit.ry(0, 0.2)
        circuit.barrier(0)

        assert len(circuit.gates) == 5
        assert isinstance(circuit.gates[0], CliffordGate)
        assert circuit.gates[0].gate_type == "T"
        assert isinstance(circuit.gates[1], CliffordGate)
        assert circuit.gates[1].gate_type == "Y"
        assert isinstance(circuit.gates[2], CliffordGate)
        assert circuit.gates[2].gate_type == "Z"
        assert isinstance(circuit.gates[3], PauliRotation)
        assert circuit.gates[3].symbols == ["Y"]
        assert np.isclose(circuit.gates[3].param_value, 0.2)
        assert isinstance(circuit.gates[4], LayerBarrier)

    def test_controlled_and_two_qubit_gates(self):
        circuit = PauliPropagationCircuit(2)
        circuit.cy(0, 1)
        circuit.cz(0, 1)
        circuit.swap(0, 1)
        circuit.ryy(0, 1, 0.3)

        gates = circuit.gates
        assert len(gates) == 6
        assert isinstance(gates[0], PauliRotation)
        assert gates[0].symbols == ["Z"]
        assert np.isclose(gates[0].param_value, -np.pi / 2)
        assert isinstance(gates[1], CliffordGate)
        assert gates[1].gate_type == "CNOT"
        assert isinstance(gates[2], CliffordGate)
        assert gates[2].gate_type == "S"
        assert isinstance(gates[3], CliffordGate)
        assert gates[3].gate_type == "CZ"
        assert isinstance(gates[4], CliffordGate)
        assert gates[4].gate_type == "SWAP"
        assert isinstance(gates[5], PauliRotation)
        assert gates[5].symbols == ["Y", "Y"]

    def test_not_supported_gates_and_measure_raise(self):
        circuit = PauliPropagationCircuit(2)

        with pytest.raises(NotImplementedError, match="CRX"):
            circuit.crx(0, 1, 0.1)
        with pytest.raises(NotImplementedError, match="CRY"):
            circuit.cry(0, 1, 0.1)
        with pytest.raises(NotImplementedError, match="CRZ"):
            circuit.crz(0, 1, 0.1)
        with pytest.raises(NotImplementedError, match="RZX"):
            circuit.rzx(0, 1, 0.1)
        with pytest.raises(NotImplementedError, match="Measurement"):
            circuit.measure()


class TestPauliPropagationCircuitComposition:

    def test_compose_is_not_supported(self):
        """PP circuits share no qiskit representation, so the base merge rejects them."""
        first = PauliPropagationCircuit(2)
        second = PauliPropagationCircuit(2)

        with pytest.raises(NotImplementedError, match="share no qiskit representation"):
            first.compose(second)

    def test_compose_still_validates_the_arguments(self):
        circuit = PauliPropagationCircuit(2)

        with pytest.raises(TypeError, match="expects a QuantumCircuitBase"):
            circuit.compose("invalid", [0, 1])
        with pytest.raises(ValueError, match="Length of qubits mapping"):
            circuit.compose(PauliPropagationCircuit(1), [0, 1])


class TestPauliPropagationCircuitUtilityMethods:
    def test_copy_invert_and_hash_ignore_unknown_gate_types(self):
        class DummyGate(Gate):
            def commutes_with(self, pauli_term: int) -> bool:
                return False

            def is_parametric(self) -> bool:
                return False

        source = PauliPropagationCircuit(1)
        source._gates = [DummyGate(0, source.num_qubits)]

        copied = source.copy()
        assert not copied.gates

        inverted = source.invert()
        assert not inverted.gates

        assert isinstance(hash(source), int)

    def test_invert_copy_hash_and_string_interfaces(self):
        theta = Parameter("theta")
        circuit = PauliPropagationCircuit(2)
        circuit.x(0)
        circuit.rx(1, theta)
        circuit.rzz(0, 1, 0.3)
        circuit.barrier([0, 1])

        copied = circuit.copy()
        assert copied is not circuit
        assert len(copied.gates) == len(circuit.gates)
        assert isinstance(copied.gates[-1], LayerBarrier)

        inverted = circuit.invert()
        assert len(inverted.gates) == 4
        assert isinstance(inverted.gates[0], LayerBarrier)
        assert isinstance(inverted.gates[1], PauliRotation)
        assert np.isclose(inverted.gates[1].param_value, -0.3)
        assert isinstance(inverted.gates[2], PauliRotation)
        assert str(inverted.gates[2].param_expr) == "-theta"
        assert isinstance(inverted.gates[3], CliffordGate)
        assert inverted.gates[3].gate_type == "X"

        assert isinstance(hash(circuit), int)
        assert str(circuit) == repr(circuit)
        assert "PauliPropagationCircuit" in str(circuit)
