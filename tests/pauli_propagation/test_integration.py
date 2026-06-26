"""Integration tests for PauliPropagation strict native API."""

import numpy as np
import pytest
import sympy as sp

from qc_executor.base import ExecutorBase
from qc_executor.factory import Executor
from qc_executor.parameters import Parameters
from qc_executor.pauli_propagation import (
    PauliPropagationCircuit,
    PauliPropagationExecutor,
    PauliPropagationOperator,
)
from qc_executor.pauli_propagation.utils.pauli_types import PauliString, PauliSum


class TestImportPaths:
    def test_import_executor_class(self):
        assert PauliPropagationExecutor is not None

    def test_import_pauli_types(self):
        assert PauliSum is not None
        assert PauliString is not None

    def test_import_native_types(self):
        assert PauliPropagationCircuit is not None
        assert PauliPropagationOperator is not None


class TestInheritance:
    def test_isinstance_executor_base(self):
        executor = PauliPropagationExecutor()
        assert isinstance(executor, ExecutorBase)

    def test_remote_property(self):
        executor = PauliPropagationExecutor()
        assert executor.remote is False

    def test_shots_property(self):
        executor = PauliPropagationExecutor(shots=500)
        assert executor.shots == 500

    def test_default_shots_none(self):
        executor = PauliPropagationExecutor()
        assert executor.shots is None


class TestNativeTypes:
    def test_expectation_value_native_types(self):
        executor = PauliPropagationExecutor()

        circuit = PauliPropagationCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        observable = PauliPropagationOperator(["ZZ"], [1.0])

        result = executor.expectation_value(circuit, observable)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_parametric_expectation_value(self):
        executor = PauliPropagationExecutor()

        p = Parameters("theta", 1)
        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, p[0])
        observable = PauliPropagationOperator(["Z"], [1.0])

        result0 = executor.expectation_value(circuit, observable, **{"theta[0]": 0.0})
        result_pi = executor.expectation_value(circuit, observable, **{"theta[0]": np.pi})

        assert np.isclose(result0, 1.0, atol=1e-10)
        assert np.isclose(result_pi, -1.0, atol=1e-10)

    def test_batch_observables(self):
        executor = PauliPropagationExecutor()

        circuit = PauliPropagationCircuit(1)
        circuit.h(0)

        op_x = PauliPropagationOperator(["X"], [1.0])
        op_z = PauliPropagationOperator(["Z"], [1.0])

        results = executor.expectation_value(circuit, [op_x, op_z])
        assert len(results) == 2
        assert np.isclose(results[0], 1.0, atol=1e-10)
        assert np.isclose(results[1], 0.0, atol=1e-10)

    def test_sample_native(self):
        executor = PauliPropagationExecutor(shots=1000, seed=42)

        circuit = PauliPropagationCircuit(1)
        circuit.x(0)

        counts = executor.sample(circuit)
        assert counts.get("1", 0) == 1000

    def test_statevector_native(self):
        executor = PauliPropagationExecutor()

        circuit = PauliPropagationCircuit(1)
        circuit.h(0)

        statevector = executor.statevector(circuit)
        expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
        assert np.allclose(statevector, expected, atol=1e-10)


class TestDerivativesNative:
    def test_derivative_with_string_param(self):
        executor = PauliPropagationExecutor()

        p = Parameters("theta", 1)
        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, p[0])
        observable = PauliPropagationOperator(["Z"], [1.0])

        theta_val = np.pi / 4
        grad = executor.expectation_value_derivatives(
            circuit,
            observable,
            "theta[0]",
            **{"theta[0]": theta_val},
        )

        eps = 1e-5
        f_plus = executor.expectation_value(circuit, observable, **{"theta[0]": theta_val + eps})
        f_minus = executor.expectation_value(circuit, observable, **{"theta[0]": theta_val - eps})
        fd_grad = (f_plus - f_minus) / (2 * eps)

        assert np.isclose(grad, fd_grad, atol=1e-6)

    def test_derivative_with_composite_gate_expression_matches_expected_values(self):
        executor = PauliPropagationExecutor()

        x = Parameters("x", 1)
        p = Parameters("p", 2)
        circuit = PauliPropagationCircuit(2)
        circuit.h(0)
        circuit.ryy(0, 1, p[0] * x[0])

        observable = PauliPropagationOperator(
            ["ZI", "IZ"], [sp.Symbol("p_obs[0]"), sp.Symbol("p_obs[1]")]
        )

        gradients = executor.expectation_value_derivatives(
            circuit,
            observable,
            "p",
            "p_obs",
            "x",
            x=[0.1],
            p=[0.3],
            p_obs=[0.5, 0.6],
        )

        assert np.allclose(gradients["p"], np.array([-0.001799730012149741]), atol=1e-10)
        assert np.allclose(gradients["p_obs"], np.array([0.0, 0.9995500337489875]), atol=1e-10)
        assert np.allclose(gradients["x"], np.array([-0.005399190036449223]), atol=1e-10)


class TestFactoryIntegration:
    def test_backend_available(self):
        assert "pauli_propagation" in Executor.available_backends()

    def test_factory_create(self):
        executor = Executor.create("pauli_propagation")
        assert isinstance(executor, PauliPropagationExecutor)

    def test_factory_executor_works(self):
        executor = Executor.create("pauli_propagation")
        circuit = PauliPropagationCircuit(1)
        observable = PauliPropagationOperator(["Z"], [1.0])

        value = executor.expectation_value(circuit, observable)
        assert np.isclose(value, 1.0, atol=1e-10)


class TestStrictInputContract:
    def test_reject_legacy_quantumcircuit(self):
        executor = PauliPropagationExecutor()
        observable = PauliPropagationOperator(["Z"], [1.0])

        with pytest.raises(TypeError, match="PauliPropagationCircuit"):
            from qc_executor import QuantumCircuit

            legacy = QuantumCircuit(1)
            executor.expectation_value(legacy, observable)

    def test_reject_legacy_quantumoperator(self):
        executor = PauliPropagationExecutor()
        circuit = PauliPropagationCircuit(1)

        with pytest.raises(TypeError, match="PauliPropagationOperator"):
            from qc_executor import QuantumOperator

            legacy = QuantumOperator(["Z"], [1.0])
            executor.expectation_value(circuit, legacy)
