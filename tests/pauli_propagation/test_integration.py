"""Integration tests for PauliPropagation strict native API."""

import numpy as np
import pytest

from executor.base import ExecutorBase
from executor.factory import Executor
from executor.parameters import Parameters
from executor.pauli_propagation import (
    PauliPropagationCircuit,
    PauliPropagationExecutor,
    PauliPropagationObservable,
    PauliString,
    PauliSum,
)


class TestImportPaths:
    def test_import_executor_class(self):
        assert PauliPropagationExecutor is not None

    def test_import_pauli_types(self):
        assert PauliSum is not None
        assert PauliString is not None

    def test_import_native_types(self):
        assert PauliPropagationCircuit is not None
        assert PauliPropagationObservable is not None


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

        operator = PauliPropagationObservable(["ZZ"], [1.0])

        result = executor.expectation_value(circuit, operator)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_parametric_expectation_value(self):
        executor = PauliPropagationExecutor()

        p = Parameters("theta", 1)
        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, p[0])
        operator = PauliPropagationObservable(["Z"], [1.0])

        result0 = executor.expectation_value(circuit, operator, **{"theta[0]": 0.0})
        result_pi = executor.expectation_value(circuit, operator, **{"theta[0]": np.pi})

        assert np.isclose(result0, 1.0, atol=1e-10)
        assert np.isclose(result_pi, -1.0, atol=1e-10)

    def test_batch_operators(self):
        executor = PauliPropagationExecutor()

        circuit = PauliPropagationCircuit(1)
        circuit.h(0)

        op_x = PauliPropagationObservable(["X"], [1.0])
        op_z = PauliPropagationObservable(["Z"], [1.0])

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
        operator = PauliPropagationObservable(["Z"], [1.0])

        theta_val = np.pi / 4
        grad = executor.expectation_value_derivatives(
            circuit,
            operator,
            "theta[0]",
            **{"theta[0]": theta_val},
        )

        eps = 1e-5
        f_plus = executor.expectation_value(circuit, operator, **{"theta[0]": theta_val + eps})
        f_minus = executor.expectation_value(circuit, operator, **{"theta[0]": theta_val - eps})
        fd_grad = (f_plus - f_minus) / (2 * eps)

        assert np.isclose(grad, fd_grad, atol=1e-6)


class TestFactoryIntegration:
    def test_backend_available(self):
        assert "pauli_propagation" in Executor.available_backends()

    def test_factory_create(self):
        executor = Executor.create("pauli_propagation")
        assert isinstance(executor, PauliPropagationExecutor)

    def test_factory_executor_works(self):
        executor = Executor.create("pauli_propagation")
        circuit = PauliPropagationCircuit(1)
        observable = PauliPropagationObservable(["Z"], [1.0])

        value = executor.expectation_value(circuit, observable)
        assert np.isclose(value, 1.0, atol=1e-10)


class TestStrictInputContract:
    def test_reject_legacy_quantumcircuit(self):
        executor = PauliPropagationExecutor()
        observable = PauliPropagationObservable(["Z"], [1.0])

        with pytest.raises(TypeError, match="PauliPropagationCircuit"):
            from executor import QuantumCircuit

            legacy = QuantumCircuit(1)
            executor.expectation_value(legacy, observable)

    def test_reject_legacy_quantumoperator(self):
        executor = PauliPropagationExecutor()
        circuit = PauliPropagationCircuit(1)

        with pytest.raises(TypeError, match="PauliPropagationObservable"):
            from executor import QuantumOperator

            legacy = QuantumOperator(["Z"], [1.0])
            executor.expectation_value(circuit, legacy)
