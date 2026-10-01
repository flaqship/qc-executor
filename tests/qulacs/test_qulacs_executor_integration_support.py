"""Tests carried over from ``integration-support`` for behaviour sQUlearn relies on.

Merged in with ``integration-support``; the tests both branches share live in
``test_qulacs_executor.py``, which is kept exactly as on ``integration-support-ir``.
"""

import numpy as np
import pytest

from qc_executor import QuantumCircuit, QuantumOperator
from qc_executor.parameters import Parameters
from qc_executor.qulacs import QulacsExecutor


def _build_circuit(num_qubits, operations):
    qc = QuantumCircuit(num_qubits)
    for gate_name, gate_args in operations:
        getattr(qc, gate_name)(*gate_args)
    return qc


class TestQulacsExecutorExpectationAndStatevector:
    def test_expectation_value_batch_of_parameter_sets(self):
        """Passing several parameter sets at once returns one result per set,
        matching a loop over individual calls - previously this raised a raw
        numpy reshape error."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])
        executor = QulacsExecutor()

        x_values = [0.1, 0.5, 1.0]
        batched = executor.expectation_value(qc, op, x=[[v] for v in x_values])
        individually = [executor.expectation_value(qc, op, x=[v]) for v in x_values]

        assert np.shape(batched) == (3,)
        assert np.allclose(batched, individually, atol=1e-10)

    def test_expectation_value_single_set_unaffected_by_batching_support(self):
        """A single parameter set still returns a bare scalar."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])
        executor = QulacsExecutor()

        result = executor.expectation_value(qc, op, x=[0.3])

        assert np.shape(result) == ()

    def test_statevector_batch_of_parameter_sets(self):
        """Passing several parameter sets returns a leading batch axis of
        statevectors instead of the previous "cannot reshape" error."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        executor = QulacsExecutor()

        x_values = [0.1, 0.5, 1.0]
        batched = executor.statevector(qc, x=[[v] for v in x_values])
        individually = np.array([executor.statevector(qc, x=[v]) for v in x_values])

        assert batched.shape == (3, 2)
        assert np.allclose(batched, individually, atol=1e-10)


class TestQulacsExecutorDerivatives:
    def test_derivatives_batch_of_parameter_sets_single_todo(self):
        """Passing several parameter sets returns one derivative per set.

        Previously this silently evaluated only the first parameter set
        (circuit_parameters.append(param_values[0])) and returned a
        plausible-looking but wrong single result."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])
        executor = QulacsExecutor()

        x_values = [0.1, 0.5, 1.0]
        batched = executor.expectation_value_derivatives(qc, op, "x", x=[[v] for v in x_values])
        individually = [
            executor.expectation_value_derivatives(qc, op, "x", x=[v]) for v in x_values
        ]

        assert np.shape(batched) == (3, 1)
        assert np.allclose(batched, individually, atol=1e-10)

    def test_derivatives_batch_of_parameter_sets_multiple_todo(self):
        """The dict form (several requested derivatives) also batches: each
        value is stacked with its own leading batch axis."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])
        executor = QulacsExecutor()

        x_values = [0.1, 0.5, 1.0]
        batched = executor.expectation_value_derivatives(
            qc, op, "expectation_value", "x", x=[[v] for v in x_values]
        )
        expected_f = [executor.expectation_value(qc, op, x=[v]) for v in x_values]
        expected_dx = [
            executor.expectation_value_derivatives(qc, op, "x", x=[v]) for v in x_values
        ]

        assert np.allclose(batched["expectation_value"], expected_f, atol=1e-10)
        assert np.allclose(batched["x"], expected_dx, atol=1e-10)

    def test_derivatives_disagreeing_batch_sizes_raise(self):
        """Two batched named parameters with different lengths are rejected
        up front with a clear error instead of a confusing downstream one."""
        x = Parameters("x", 1)
        p_obs = Parameters("p_obs", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        op = QuantumOperator(["Z"], [p_obs[0]])
        executor = QulacsExecutor()

        with pytest.raises(ValueError, match="must share the same batch size"):
            executor.expectation_value_derivatives(
                qc, op, "x", x=[[0.1], [0.2], [0.3]], p_obs=[[1.0], [2.0]]
            )
