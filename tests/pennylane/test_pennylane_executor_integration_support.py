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

    def test_expectation_value_batch_uses_pennylane_broadcasting(self):
        """A parameter batch is one PennyLane device execution, not a Python loop."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [1.0])
        executor = PennyLaneExecutor()

        with executor._device.tracker:
            result = executor.expectation_value(qc, operator, x=[[0.1], [0.5], [1.0]])

        assert np.shape(result) == (3,)
        assert executor._device.tracker.totals["simulations"] == 1

    def test_expectation_value_batch_preserves_cartesian_parameter_groups(self):
        """Multiple vector-valued parameters retain their Cartesian semantics."""
        x = Parameters("x", 1)
        y = Parameters("y", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]]), ("rz", [0, y[0]])])
        operator = QuantumOperator(["X"], [1.0])
        executor = PennyLaneExecutor()

        batched = executor.expectation_value(qc, operator, x=[[0.1], [0.5]], y=[[0.2], [0.7]])
        individually = np.array(
            [
                [
                    executor.expectation_value(qc, operator, x=[first], y=[second])
                    for second in [0.2, 0.7]
                ]
                for first in [0.1, 0.5]
            ]
        )

        assert np.shape(batched) == (2, 2)
        assert np.allclose(batched, individually, atol=1e-10)

    def test_expectation_value_multiple_parameter_sweeps(self):
        x = Parameters("x", 1)
        p = Parameters("p", 1)
        qc = _build_circuit(1, [("ry", [0, p[0] * x[0]])])
        operator = QuantumOperator(["Z"], [1.0])
        executor = PennyLaneExecutor()
        parameter_values = {"x": [[0.1], [0.3], [0.5]], "p": [[0.3], [0.7]]}

        grid = executor.expectation_value(qc, operator, **parameter_values)

        expected = np.array(
            [
                [
                    executor.expectation_value(qc, operator, p=[pv], x=[xv])
                    for xv in (0.1, 0.3, 0.5)
                ]
                for pv in (0.3, 0.7)
            ]
        )
        assert grid.shape == (2, 3)
        np.testing.assert_allclose(grid, expected, atol=1e-10)

    @pytest.mark.parametrize(
        "circuit_list, observable_list", [(True, False), (False, True), (True, True)]
    )
    def test_expectation_value_lists_preserve_sweep_axes(self, circuit_list, observable_list):
        x = Parameters("x", 1)
        p = Parameters("p", 1)
        qc = _build_circuit(1, [("ry", [0, p[0] * x[0]])])
        operator = QuantumOperator(["Z"], [1.0])
        executor = PennyLaneExecutor()
        parameters = {"x": [[0.1], [0.3], [0.5]], "p": [[0.3], [0.7]]}
        expected = executor.expectation_value(qc, operator, **parameters)

        result = executor.expectation_value(
            [qc, qc] if circuit_list else qc,
            [operator, operator] if observable_list else operator,
            **parameters,
        )

        leading_shape = (2,) * (circuit_list + observable_list)
        assert result.shape == (*leading_shape, 2, 3)
        np.testing.assert_allclose(result, np.broadcast_to(expected, result.shape))

    def test_expectation_value_lists_reject_mismatched_sweep_shapes(self):
        x = Parameters("x", 1)
        first = _build_circuit(1, [("ry", [0, x[0]])])
        second = _build_circuit(1, [("h", [0])])
        operator = QuantumOperator(["Z"], [1.0])

        with pytest.raises(ValueError, match="matching parameter sweep shapes"):
            PennyLaneExecutor().expectation_value([first, second], operator, x=[[0.1], [0.2]])

    def test_expectation_value_with_observable_sweep(self):
        x = Parameters("x", 1)
        coefficient = Parameters("coefficient", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [coefficient[0]])
        executor = PennyLaneExecutor()

        result = executor.expectation_value(
            qc, operator, x=[[0.1], [0.5], [1.0]], coefficient=[[1.0], [2.0]]
        )

        assert result.shape == (3, 2)
        np.testing.assert_allclose(result, np.cos([0.1, 0.5, 1.0])[:, None] * [1.0, 2.0])

    def test_expectation_value_single_and_empty_batches(self):
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [1.0])
        executor = PennyLaneExecutor()

        assert np.shape(executor.expectation_value(qc, operator, x=[0.1])) == ()
        assert executor.expectation_value(qc, operator, x=[[0.1], [0.5]]).shape == (2,)
        assert executor.expectation_value(qc, operator, x=np.empty((0, 1))).shape == (0,)

    def test_expectation_value_empty_observable_batch(self):
        x = Parameters("x", 1)
        coefficient = Parameters("coefficient", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [coefficient[0]])

        result = PennyLaneExecutor().expectation_value(
            qc, operator, x=[[0.1], [0.5]], coefficient=np.empty((0, 1))
        )

        assert result.shape == (2, 0)

    def test_expectation_value_vector_parameter_rows(self):
        angles = Parameters("angles", 2)
        scale = Parameters("scale", 1)
        qc = _build_circuit(1, [("ry", [0, angles[0] * scale[0]]), ("rz", [0, angles[1]])])
        operator = QuantumOperator(["X"], [1.0])
        executor = PennyLaneExecutor()
        angles_values = [[0.1, 0.2], [0.5, 0.7]]
        scale_values = [[1.0], [2.0], [3.0]]

        grid = executor.expectation_value(qc, operator, angles=angles_values, scale=scale_values)
        expected = [
            [
                executor.expectation_value(qc, operator, angles=angle, scale=scale_value)
                for scale_value in scale_values
            ]
            for angle in angles_values
        ]
        assert grid.shape == (2, 3)
        np.testing.assert_allclose(grid, expected, atol=1e-10)

    def test_expectation_value_shot_based_shape(self):
        x = Parameters("x", 1)
        p = Parameters("p", 1)
        qc = _build_circuit(1, [("ry", [0, p[0] * x[0]])])
        operator = QuantumOperator(["Z"], [1.0])
        executor = PennyLaneExecutor(shots=1000, seed=7)

        result = executor.expectation_value(
            qc, operator, p=[[0.3], [0.7]], x=[[0.1], [0.3], [0.5]]
        )

        assert result.shape == (2, 3)
        assert np.all(np.abs(result) <= 1.0)

    def test_expectation_value_shared_parameter_name(self):
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [x[0]])
        executor = PennyLaneExecutor()

        grid = executor.expectation_value(qc, operator, x=[[0.2], [0.7]])

        assert grid.shape == (2, 2)
        np.testing.assert_allclose(grid, np.cos([0.2, 0.7])[:, None] * [0.2, 0.7])

    def test_expectation_value_no_parameters(self):
        qc = _build_circuit(1, [("h", [0])])
        operator = QuantumOperator(["Z"], [1.0])

        result = PennyLaneExecutor().expectation_value(qc, operator)

        assert np.shape(result) == ()
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_expectation_value_empty_parameter_batch(self):
        """An empty batch remains an empty result without invoking the device."""
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        operator = QuantumOperator(["Z"], [1.0])

        result = PennyLaneExecutor().expectation_value(qc, operator, x=np.empty((0, 1)))

        assert np.shape(result) == (0,)

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
