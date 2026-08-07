"""Tests for the process-parallel execution helpers.

expectation_task is the shared serial/parallel code path; when the executor
runs with n_jobs > 1 it executes inside worker processes where coverage
cannot observe it, so it is unit-tested directly here in-process.
"""

import numpy as np
import sympy as sp

from qc_executor.pauli_propagation import (
    PauliPropagationCircuit,
    PauliPropagationExecutor,
    PauliPropagationOperator,
)
from qc_executor.pauli_propagation.symmetry import NoSymmetry, PermutationSymmetry
from qc_executor.pauli_propagation.utils.parallel import (
    expectation_task,
    expectation_task_star,
)
from qc_executor.pauli_propagation.utils.truncation import TruncationStats


def _bell_setup():
    circuit = PauliPropagationCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    observable = PauliPropagationOperator(["ZZ"], [1.0])
    return circuit, observable


class TestExpectationTask:
    """Direct in-process tests of the worker function."""

    def test_basic_expectation_no_truncation(self):
        circuit, observable = _bell_setup()

        value, stats = expectation_task(circuit, observable, {}, None, None, NoSymmetry())

        assert np.isclose(value.real, 1.0)
        assert stats is None

    def test_truncation_returns_stats(self):
        circuit, observable = _bell_setup()

        value, stats = expectation_task(circuit, observable, {}, 1e-10, 4, NoSymmetry())

        assert np.isclose(value.real, 1.0)
        assert isinstance(stats, TruncationStats)

    def test_parametric_gate_binding(self):
        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, sp.Symbol("theta[0]"))
        observable = PauliPropagationOperator(["Z"], [1.0])

        value, _ = expectation_task(
            circuit, observable, {"theta[0]": np.pi}, None, None, NoSymmetry()
        )

        assert np.isclose(value.real, -1.0)

    def test_parametrized_observable_coefficients(self):
        circuit = PauliPropagationCircuit(1)
        observable = PauliPropagationOperator(["Z"], [sp.Symbol("a")], num_qubits=1)
        assert observable.is_parametrized

        value, _ = expectation_task(circuit, observable, {"a": 0.5}, None, None, NoSymmetry())

        assert np.isclose(value.real, 0.5)

    def test_executor_symmetry_fallback_is_applied(self):
        """Observables without their own symmetry get the executor-level one."""
        nqubits = 2
        circuit = PauliPropagationCircuit(nqubits)
        observable = PauliPropagationOperator(["ZI", "IZ"], [0.5, 0.5], num_qubits=nqubits)

        value, _ = expectation_task(circuit, observable, {}, None, None, PermutationSymmetry())

        # ZI and IZ merge to one canonical Z term with coefficient 1.0;
        # on |00> the expectation is 1.0 either way
        assert np.isclose(value.real, 1.0)

    def test_star_adapter_matches_direct_call(self):
        circuit, observable = _bell_setup()
        args = (circuit, observable, {}, None, None, NoSymmetry())

        assert expectation_task_star(args) == expectation_task(*args)

    def test_matches_executor_result(self):
        """The task must agree with the executor's public API end to end."""
        circuit = PauliPropagationCircuit(2)
        circuit.h(0)
        circuit.rzz(0, 1, 0.7)
        circuit.s(1)
        observable = PauliPropagationOperator(["ZZ", "XI"], [0.8, 0.2])

        executor = PauliPropagationExecutor(truncate_threshold=1e-12)
        expected = executor.expectation_value(circuit, observable)

        value, stats = expectation_task(circuit, observable, {}, 1e-12, None, NoSymmetry())

        assert np.isclose(value.real, expected)
        assert isinstance(stats, TruncationStats)
