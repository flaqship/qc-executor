"""Tests carried over from ``integration-support`` for behaviour sQUlearn relies on.

Merged in with ``integration-support``; the tests both branches share live in
``test_pennylane_executor.py``, which is kept exactly as on ``integration-support-ir``.
"""

import numpy as np
import pytest

from qc_executor import QuantumCircuit, QuantumOperator
from qc_executor.parameters import Parameters
from qc_executor.pennylane.pennylane_executor import PennyLaneExecutor


def _build_circuit(num_qubits, operations):
    """
    Helper function to build a quantum circuit from a list of operations.

    Args:
        num_qubits (int): Number of qubits in the circuit
        operations (list): List of tuples (gate_name, gate_args)
                          Example: [("h", [0]), ("cx", [0, 1])]

    Returns:
        QuantumCircuit: The constructed quantum circuit
    """
    qc = QuantumCircuit(num_qubits)
    for gate_name, gate_args in operations:
        getattr(qc, gate_name)(*gate_args)
    return qc


class TestPennylaneExpectationValue:
    """Test suite for PennyLane executor expectation values."""

    def test_expectation_value_batch_of_parameter_sets(self):
        """Passing several parameter sets at once returns one result per set,
        matching a loop over individual calls - previously this raised
        NotImplementedError (or, before that guard was even reached, crashed
        with an inhomogeneous-shape error from a generator that was silently
        exhausted after the first parameter set)."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [1.0])
        executor = PennyLaneExecutor()

        x_values = [0.1, 0.5, 1.0]
        batched = executor.expectation_value(qc, operator, x=[[v] for v in x_values])
        individually = [executor.expectation_value(qc, operator, x=[v]) for v in x_values]

        assert np.shape(batched) == (3,)
        assert np.allclose(batched, individually, atol=1e-10)

    def test_expectation_value_single_set_unaffected_by_batching_support(self):
        """A single parameter set still returns a bare scalar, not a
        length-1 batch - the common case is unchanged by batch support."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [1.0])
        executor = PennyLaneExecutor()

        result = executor.expectation_value(qc, operator, x=[0.3])

        assert np.shape(result) == ()

    def test_expectation_value_batch_with_parametric_observable(self):
        """A batched circuit parameter alongside a parametric observable
        exercises the fix to the per-observable parameter-tuple generator,
        which used to be exhausted after the first circuit parameter set."""
        x = Parameters("x", 1)
        p_obs = Parameters("p_obs", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [p_obs[0]])
        executor = PennyLaneExecutor()

        x_values = [0.1, 0.5, 1.0]
        batched = executor.expectation_value(qc, operator, x=[[v] for v in x_values], p_obs=[2.0])
        individually = [
            executor.expectation_value(qc, operator, x=[v], p_obs=[2.0]) for v in x_values
        ]

        assert np.allclose(batched, individually, atol=1e-10)


class TestPennylaneStatevector:
    """Test suite for PennyLane executor statevector."""

    def test_statevector_batch_of_parameter_sets(self):
        """Passing several parameter sets returns a leading batch axis of
        statevectors instead of the previous "cannot reshape" error."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("rx", [0, x[0]])])
        executor = PennyLaneExecutor()

        x_values = [0.1, 0.5, 1.0]
        batched = executor.statevector(qc, x=[[v] for v in x_values])
        individually = np.array([executor.statevector(qc, x=[v]) for v in x_values])

        assert batched.shape == (3, 2)
        assert np.allclose(batched, individually, atol=1e-10)


class TestPennylaneDerivatives:
    """Test suite for PennyLane executor derivatives."""

    def test_derivatives_batch_of_parameter_sets_single_todo(self):
        """Passing several parameter sets returns one derivative per set.

        Previously this silently evaluated only the first parameter set
        (params.append(pv[0])) and returned a plausible-looking but wrong
        single result, discarding the rest without any warning."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [1.0])
        executor = PennyLaneExecutor()

        x_values = [0.1, 0.5, 1.0]
        batched = executor.expectation_value_derivatives(
            qc, operator, "x", x=[[v] for v in x_values]
        )
        individually = [
            executor.expectation_value_derivatives(qc, operator, "x", x=[v]) for v in x_values
        ]

        assert np.shape(batched) == (3, 1)
        assert np.allclose(batched, individually, atol=1e-10)

    def test_derivatives_batch_of_parameter_sets_multiple_todo(self):
        """The dict form (several requested derivatives) also batches:
        each value is stacked with its own leading batch axis."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [1.0])
        executor = PennyLaneExecutor()

        x_values = [0.1, 0.5, 1.0]
        batched = executor.expectation_value_derivatives(
            qc, operator, "expectation_value", "x", x=[[v] for v in x_values]
        )
        expected_f = [executor.expectation_value(qc, operator, x=[v]) for v in x_values]
        expected_dx = [
            executor.expectation_value_derivatives(qc, operator, "x", x=[v]) for v in x_values
        ]

        assert np.allclose(batched["expectation_value"], expected_f, atol=1e-10)
        assert np.allclose(batched["x"], expected_dx, atol=1e-10)

    def test_derivatives_disagreeing_batch_sizes_raise(self):
        """Two batched named parameters with different lengths are rejected
        up front with a clear error instead of a confusing downstream one."""
        x = Parameters("x", 1)
        p_obs = Parameters("p_obs", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [p_obs[0]])
        executor = PennyLaneExecutor()

        with pytest.raises(ValueError, match="must share the same batch size"):
            executor.expectation_value_derivatives(
                qc, operator, "x", x=[[0.1], [0.2], [0.3]], p_obs=[[1.0], [2.0]]
            )
