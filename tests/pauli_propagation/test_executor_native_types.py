"""Tests for PauliPropagationExecutor with backend-native types."""

import numpy as np
import pytest

from qc_executor.parameters import Parameters
from qc_executor.pauli_propagation import (
    PauliPropagationCircuit,
    PauliPropagationExecutor,
    PauliPropagationOperator,
)
from qc_executor.pauli_propagation.symmetry import PermutationSymmetry


class TestPauliPropagationExecutorNativeTypes:
    def test_expectation_value_single(self):
        circuit = PauliPropagationCircuit(1)
        circuit.h(0)
        observable = PauliPropagationOperator(["X"], [1.0])

        executor = PauliPropagationExecutor()
        value = executor.expectation_value(circuit, observable)

        assert np.isclose(value, 1.0, atol=1e-10)

    def test_expectation_value_batch(self):
        circuit_0 = PauliPropagationCircuit(1)
        circuit_1 = PauliPropagationCircuit(1)
        circuit_1.x(0)
        observable = PauliPropagationOperator(["Z"], [1.0])

        executor = PauliPropagationExecutor()
        values = executor.expectation_value([circuit_0, circuit_1], observable)

        assert values.shape == (2,)
        assert np.isclose(values[0], 1.0, atol=1e-10)
        assert np.isclose(values[1], -1.0, atol=1e-10)

    def test_expectation_value_parametric(self):
        params = Parameters("theta", 1)
        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, params[0])
        observable = PauliPropagationOperator(["Z"], [1.0])

        executor = PauliPropagationExecutor()
        value_0 = executor.expectation_value(circuit, observable, **{"theta[0]": 0.0})
        value_pi = executor.expectation_value(circuit, observable, **{"theta[0]": np.pi})

        assert np.isclose(value_0, 1.0, atol=1e-10)
        assert np.isclose(value_pi, -1.0, atol=1e-10)

    def test_statevector(self):
        circuit = PauliPropagationCircuit(1)
        circuit.h(0)

        executor = PauliPropagationExecutor()
        state = executor.statevector(circuit)

        expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
        assert np.allclose(state, expected, atol=1e-10)

    def test_sample(self):
        circuit = PauliPropagationCircuit(1)
        circuit.x(0)

        executor = PauliPropagationExecutor(shots=100, seed=7)
        samples = executor.sample(circuit)

        assert samples == {"1": 100}

    def test_reject_non_native_types(self):
        executor = PauliPropagationExecutor()
        observable = PauliPropagationOperator(["Z"], [1.0])

        with pytest.raises(TypeError, match="PauliPropagationCircuit"):
            executor.expectation_value("not a circuit", observable)

    def test_executor_uses_observable_symmetry_when_present(self, monkeypatch):
        circuit = PauliPropagationCircuit(1)
        observable = PauliPropagationOperator(
            ["I"], [1.0], symmetry_strategy=PermutationSymmetry()
        )
        executor = PauliPropagationExecutor()

        captured = {}

        def fake_propagate(gates, obs, params, max_weight, truncate_threshold):
            captured["symmetry_name"] = obs.symmetry.name
            return obs

        monkeypatch.setattr(
            "qc_executor.pauli_propagation.pauli_propagation_executor.propagate", fake_propagate
        )

        value = executor.expectation_value(circuit, observable)

        assert np.isclose(value, 1.0)
        assert captured["symmetry_name"] == "permutation"

    def test_executor_falls_back_to_executor_symmetry(self, monkeypatch):
        circuit = PauliPropagationCircuit(1)
        observable = PauliPropagationOperator(["I"], [1.0])
        executor = PauliPropagationExecutor(symmetry_strategy=PermutationSymmetry())

        captured = {}

        def fake_propagate(gates, obs, params, max_weight, truncate_threshold):
            captured["symmetry_name"] = obs.symmetry.name
            return obs

        monkeypatch.setattr(
            "qc_executor.pauli_propagation.pauli_propagation_executor.propagate", fake_propagate
        )

        value = executor.expectation_value(circuit, observable)

        assert np.isclose(value, 1.0)
        assert captured["symmetry_name"] == "permutation"
