"""Tests carried over from ``integration-support`` for behaviour sQUlearn relies on.

Merged in with ``integration-support``; the tests both branches share live in
``test_qulacs_executor.py``, which is kept exactly as on ``integration-support-ir``.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

from qc_executor import QuantumCircuit, QuantumOperator
from qc_executor.parameters import Parameters
from qc_executor.qulacs import QulacsExecutor
from qc_executor.qulacs.qulacs_executor import _initialize_qulacs_worker


def _build_circuit(num_qubits, operations):
    qc = QuantumCircuit(num_qubits)
    for gate_name, gate_args in operations:
        getattr(qc, gate_name)(*gate_args)
    return qc


def test_batch_benchmark_json_output():
    script = Path(__file__).resolve().parents[2] / "examples" / "qulacs" / "benchmark_batches.py"
    output = subprocess.run(
        [
            sys.executable,
            str(script),
            "--batch-size",
            "2",
            "--qubits",
            "2",
            "--layers",
            "1",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.stdout)
    assert result["shape"] == [2]
    assert len(result["values"]) == 2
    assert result["wall_seconds"] > 0
    assert len(result["wall_samples_seconds"]) == 1
    assert result["peak_rss_self"] > 0


def test_batch_benchmark_cartesian_json_output():
    script = Path(__file__).resolve().parents[2] / "examples" / "qulacs" / "benchmark_batches.py"
    output = subprocess.run(
        [
            sys.executable,
            str(script),
            "--grid-size",
            "2",
            "--qubits",
            "2",
            "--layers",
            "1",
            "--warmups",
            "1",
            "--repeats",
            "1",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.stdout)
    assert result["shape"] == [2, 2]
    assert result["grid_size"] == 2
    assert result["batch_size"] == 4
    assert result["warmups"] == 1
    assert len(result["wall_samples_seconds"]) == 1
    assert len(result["values"]) == 4


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

    def test_expectation_value_cartesian_sweeps(self):
        x = Parameters("x", 1)
        p = Parameters("p", 1)
        coefficient = Parameters("coefficient", 1)
        qc = _build_circuit(1, [("ry", [0, x[0] * p[0]])])
        op = QuantumOperator(["Z"], [coefficient[0]])
        executor = QulacsExecutor()

        result = executor.expectation_value(
            qc, op, x=[[0.1], [0.3], [0.5]], p=[[0.2], [0.4]], coefficient=[[1.0], [2.0]]
        )

        expected = np.cos(np.outer([0.2, 0.4], [0.1, 0.3, 0.5]))[:, :, None] * [1.0, 2.0]
        assert result.shape == (2, 3, 2)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_expectation_value_parallel_matches_sequential(self):
        x = Parameters("x", 1)
        p = Parameters("p", 1)
        coefficient = Parameters("coefficient", 1)
        qc = _build_circuit(1, [("ry", [0, x[0] * p[0]])])
        op = QuantumOperator(["Z"], [coefficient[0]])
        values = {"x": [[0.1], [0.3], [0.5]], "p": [[0.2], [0.4]], "coefficient": [[1.0], [2.0]]}

        sequential = QulacsExecutor().expectation_value(qc, op, **values)
        with patch(
            "qc_executor.qulacs.qulacs_executor.ProcessPoolExecutor", wraps=ProcessPoolExecutor
        ) as pool:
            parallel = QulacsExecutor(max_workers=2, parallel_min_batch_size=2).expectation_value(
                qc, op, **values
            )

        assert parallel.shape == (2, 3, 2)
        np.testing.assert_allclose(parallel, sequential, atol=1e-10)
        assert pool.call_count == 1
        assert pool.call_args.kwargs["max_workers"] == 2

    def test_expectation_value_parallel_matches_other_backends(self):
        pytest.importorskip("pennylane")
        pytest.importorskip("qiskit")
        from qc_executor.pennylane import PennyLaneExecutor
        from qc_executor.qiskit import QiskitExecutor

        x = Parameters("x", 2)
        p = Parameters("p", 1)
        coefficient = Parameters("coefficient", 1)
        qc = _build_circuit(1, [("ry", [0, x[0] * p[0]]), ("rz", [0, x[1]])])
        op = QuantumOperator(["X"], [coefficient[0]])
        values = {
            "p": [[0.2], [0.4]],
            "x": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            "coefficient": [[1.0], [2.0], [3.0]],
        }

        result = QulacsExecutor(max_workers=2, parallel_min_batch_size=2).expectation_value(
            qc, op, **values
        )
        assert result.shape == (2, 3, 3)
        for executor in (PennyLaneExecutor(), QiskitExecutor(backend="statevector")):
            np.testing.assert_allclose(
                result, executor.expectation_value(qc, op, **values), atol=1e-10
            )

    def test_expectation_value_parallel_threshold_and_empty_batch(self):
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])
        executor = QulacsExecutor(max_workers=2, parallel_min_batch_size=3)

        with patch("qc_executor.qulacs.qulacs_executor.ProcessPoolExecutor") as pool:
            result = executor.expectation_value(qc, op, x=[[0.1], [0.5]])
            empty = executor.expectation_value(qc, op, x=np.empty((0, 1)))

        pool.assert_not_called()
        np.testing.assert_allclose(result, np.cos([0.1, 0.5]))
        assert empty.shape == (0,)

    @pytest.mark.parametrize(
        "name,value",
        [
            ("max_workers", 0),
            ("max_workers", True),
            ("parallel_min_batch_size", 1),
            ("parallel_min_batch_size", 1.5),
        ],
    )
    def test_invalid_parallel_configuration(self, name, value):
        with pytest.raises(ValueError, match=name):
            QulacsExecutor(**{name: value})

    def test_parallel_worker_limits_native_thread_pools(self):
        from threadpoolctl import threadpool_info

        qc = QulacsExecutor().transpile_circuit(_build_circuit(1, [("h", [0])]))
        with ProcessPoolExecutor(
            max_workers=1,
            mp_context=get_context("spawn"),
            initializer=_initialize_qulacs_worker,
            initargs=((qc.num_qubits, qc.num_clbits, qc.ir), []),
        ) as pool:
            libraries = pool.submit(threadpool_info).result()

        assert all(lib["num_threads"] == 1 for lib in libraries)

    def test_expectation_value_parallel_circuit_and_observable_lists(self):
        x = Parameters("x", 1)
        circuits = [
            _build_circuit(1, [("ry", [0, x[0]])]),
            _build_circuit(1, [("rx", [0, x[0]])]),
        ]
        operators = [QuantumOperator(["Z"], [1.0]), QuantumOperator(["X"], [1.0])]
        values = {"x": [[0.1], [0.5], [1.0]]}
        sequential = QulacsExecutor().expectation_value(circuits, operators, **values)
        parallel = QulacsExecutor(max_workers=2, parallel_min_batch_size=2).expectation_value(
            circuits, operators, **values
        )

        assert parallel.shape == (2, 2, 3)
        np.testing.assert_allclose(parallel, sequential, atol=1e-10)

    def test_expectation_value_rejects_mismatched_list_sweep_shapes(self):
        x = Parameters("x", 1)
        circuits = [_build_circuit(1, [("ry", [0, x[0]])]), _build_circuit(1, [("h", [0])])]
        op = QuantumOperator(["Z"], [1.0])

        with pytest.raises(ValueError, match="matching parameter sweep shapes"):
            QulacsExecutor().expectation_value(circuits, op, x=[[0.1], [0.5]])

    def test_parallel_unsupported_circuit_raises_directly(self):
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("reset", [0]), ("ry", [0, x[0]])])
        op = QuantumOperator(["Z"], [1.0])
        executor = QulacsExecutor(max_workers=2, parallel_min_batch_size=2)

        with pytest.raises(NotImplementedError, match="reset"):
            executor.expectation_value(qc, op, x=[[0.1], [0.5]])

    def test_parallel_same_name_in_circuit_and_observable(self):
        x = Parameters("x", 1)
        qc = _build_circuit(1, [("ry", [0, x[0]])])
        op = QuantumOperator(["Z"], [x[0]])
        executor = QulacsExecutor(max_workers=2, parallel_min_batch_size=2)

        result = executor.expectation_value(qc, op, x=[[0.2], [0.7]])

        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, np.cos([0.2, 0.7])[:, None] * [0.2, 0.7])

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
