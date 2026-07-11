"""Tests for PauliPropagationCircuit."""

import numpy as np

from qc_executor.abstraction import ParameterVector
from qc_executor.pauli_propagation import PauliPropagationCircuit
from qc_executor.pauli_propagation.utils.gates import CliffordGate, LayerBarrier, PauliRotation


class TestPauliPropagationCircuit:
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

    def test_parameter_tracking(self):
        params = ParameterVector("theta", 2)
        circuit = PauliPropagationCircuit(2)
        circuit.rx(0, params[0])
        circuit.rzz(0, 1, params[1])

        assert circuit.is_parameterized
        assert set(circuit.parameters) == {"theta[0]", "theta[1]"}

    def test_assign_parameters(self):
        params = ParameterVector("theta", 1)
        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, params[0])

        assigned = circuit.assign_parameters({"theta[0]": np.pi / 2})
        assert isinstance(assigned.gates[0], PauliRotation)
        assert assigned.gates[0].param_name is None
        assert np.isclose(assigned.gates[0].param_value, np.pi / 2)

    def test_circuit_metrics(self):
        circuit = PauliPropagationCircuit(2)
        circuit.x(0)
        circuit.rxx(0, 1, 0.3)

        metrics = circuit.circuit_metrics()
        assert metrics["num_qubits"] == 2
        assert metrics["num_gates"] == 2
        assert metrics["num_parameters"] == 0
