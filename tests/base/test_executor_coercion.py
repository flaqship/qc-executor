"""Tests for the executor's generic-or-native input coercion.

Every public execution method accepts a generic circuit/operator or one already
native to the backend.  Generic inputs are converted; native ones pass straight
through.  Before this, each backend unwrapped inputs itself and
PauliPropagation rejected generic inputs outright.
"""

from __future__ import annotations

import numpy as np
import pytest

from qc_executor import Executor, QuantumCircuit, QuantumOperator
from qc_executor.base.executor_base import ExecutorBase
from tests.conftest import INSTALLED_BACKENDS


class NativeCircuit(QuantumCircuit):
    """Stand-in for a backend's native circuit type."""


class NativeOperator(QuantumOperator):
    """Stand-in for a backend's native operator type."""


class RecordingExecutor(ExecutorBase):
    """Records what reached the abstract hooks, and how often."""

    _native_circuit_class = NativeCircuit
    _native_operator_class = NativeOperator

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen: list = []
        self.conversions = {"circuit": 0, "operator": 0}

    @property
    def remote(self) -> bool:
        return False

    def _expectation_value(self, circuit, observable, **parameters):
        self.seen.append((circuit, observable))
        return 0.0

    def _expectation_value_derivatives(self, circuit, observable, *derivative, **parameters):
        self.seen.append((circuit, observable))
        return 0.0

    def _sample(self, circuit, **parameters):
        self.seen.append((circuit, None))
        return {}

    def _statevector(self, circuit, **parameters):
        self.seen.append((circuit, None))
        return np.zeros(2)

    def _transpile_circuit(self, circuit):
        self.conversions["circuit"] += 1
        return NativeCircuit(circuit.num_qubits, _ir=circuit.ir)

    def _transpile_operator(self, operator, **_options):
        self.conversions["operator"] += 1
        return NativeOperator(_ir=operator.ir)

    @classmethod
    def get_accepted_backend_types(cls) -> list[type]:
        return []


def _generic():
    """A one-qubit generic circuit and observable."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    return circuit, QuantumOperator(["Z"], [1.0])


class TestCoercion:
    @pytest.mark.parametrize(
        "call",
        [
            lambda ex, c, o: ex.expectation_value(c, o),
            lambda ex, c, o: ex.expectation_value_derivatives(c, o, "x"),
        ],
        ids=["expectation_value", "derivatives"],
    )
    def test_generic_inputs_reach_the_backend_as_native(self, call):
        executor = RecordingExecutor()
        circuit, observable = _generic()

        call(executor, circuit, observable)

        seen_circuit, seen_observable = executor.seen[-1]
        assert isinstance(seen_circuit, NativeCircuit)
        assert isinstance(seen_observable, NativeOperator)

    @pytest.mark.parametrize(
        "call",
        [lambda ex, c: ex.sample(c), lambda ex, c: ex.statevector(c)],
        ids=["sample", "statevector"],
    )
    def test_circuit_only_methods_coerce_too(self, call):
        executor = RecordingExecutor()
        circuit, _ = _generic()

        call(executor, circuit)

        assert isinstance(executor.seen[-1][0], NativeCircuit)

    def test_native_inputs_pass_through_untouched(self):
        executor = RecordingExecutor()
        circuit = NativeCircuit(1)
        observable = NativeOperator(["Z"], [1.0])

        executor.expectation_value(circuit, observable)

        assert executor.seen[-1] == (circuit, observable)
        assert executor.conversions == {"circuit": 0, "operator": 0}

    def test_lists_are_coerced_elementwise(self):
        executor = RecordingExecutor()
        circuit, observable = _generic()

        executor.expectation_value([circuit, NativeCircuit(1)], [observable])

        seen_circuits, seen_observables = executor.seen[-1]
        assert all(isinstance(c, NativeCircuit) for c in seen_circuits)
        assert all(isinstance(o, NativeOperator) for o in seen_observables)
        assert executor.conversions["circuit"] == 1, "the native circuit needed no conversion"

    def test_generic_and_native_share_one_cache_entry(self):
        """Coercion happens before the key is built, so both spellings hit."""
        executor = RecordingExecutor(caching=True)
        circuit, observable = _generic()

        executor.expectation_value(circuit, observable)
        calls_after_first = len(executor.seen)
        executor.expectation_value(
            executor.transpile_circuit(circuit), executor.transpile_operator(observable)
        )

        assert len(executor.seen) == calls_after_first


class TestTranspileOptions:
    def test_options_are_forwarded_to_the_backend_hook(self):
        received = {}

        class OptionExecutor(RecordingExecutor):
            def _transpile_operator(self, operator, **options):
                received.update(options)
                return NativeOperator(_ir=operator.ir)

        _, observable = _generic()
        OptionExecutor().transpile_operator(observable, flavour="spicy")

        assert received == {"flavour": "spicy"}

    def test_options_are_part_of_the_cache_key(self):
        calls = []

        class OptionExecutor(RecordingExecutor):
            def _transpile_operator(self, operator, **options):
                calls.append(options)
                return NativeOperator(_ir=operator.ir)

        executor = OptionExecutor(caching=True)
        _, observable = _generic()

        executor.transpile_operator(observable, flavour="a")
        executor.transpile_operator(observable, flavour="b")
        executor.transpile_operator(observable, flavour="a")

        assert calls == [{"flavour": "a"}, {"flavour": "b"}]

    def test_options_bypass_the_native_pass_through(self):
        """A native operator must still be re-transpiled when options are given."""
        executor = RecordingExecutor()
        operator = NativeOperator(["Z"], [1.0])

        executor._coerce_operator(operator, flavour="spicy")

        assert executor.conversions["operator"] == 1


class TestEveryBackendAcceptsBothForms:
    @pytest.mark.parametrize("backend", INSTALLED_BACKENDS)
    def test_generic_and_native_agree(self, backend):
        executor = Executor.create(backend, seed=0)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        observable = QuantumOperator(["ZZ"], [1.0])

        from_generic = executor.expectation_value(circuit, observable)
        from_native = executor.expectation_value(
            executor.transpile_circuit(circuit), executor.transpile_operator(observable)
        )

        assert np.isclose(float(np.real(from_generic)), float(np.real(from_native)), atol=1e-10)

    @pytest.mark.parametrize("backend", INSTALLED_BACKENDS)
    def test_mixing_generic_and_native_works(self, backend):
        executor = Executor.create(backend, seed=0)
        circuit = QuantumCircuit(1)
        observable = QuantumOperator(["Z"], [1.0])

        result = executor.expectation_value(executor.transpile_circuit(circuit), observable)

        assert np.isclose(float(np.real(result)), 1.0, atol=1e-10)
