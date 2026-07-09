"""Performance benchmarks for Pauli propagation.

Skipped by default (like the benchmarks in test_symmetry.py) to avoid
flakiness from scheduler noise and hardware differences. Set
RUN_BENCHMARKS=1 to enable. Timings are printed so before/after comparisons
can be made across commits; assertions are deliberately loose sanity bounds.
"""

import os
import random
import time

import numpy as np
import pytest

from qc_executor.parameters import Parameters
from qc_executor.pauli_propagation import (
    PauliPropagationCircuit,
    PauliPropagationExecutor,
    PauliPropagationOperator,
)

_RUN_BENCHMARKS = os.environ.get("RUN_BENCHMARKS", "0") == "1"
_benchmark_skip = pytest.mark.skipif(
    not _RUN_BENCHMARKS,
    reason="Benchmark tests are skipped by default; set RUN_BENCHMARKS=1 to enable",
)


def _time_call(func, *args, n_runs=3, **kwargs):
    """Run func once for warmup, then time n_runs executions; return best."""
    func(*args, **kwargs)
    timings = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        timings.append(time.perf_counter() - start)
    return min(timings), result


def _clifford_circuit(nqubits: int, ngates: int, seed: int) -> PauliPropagationCircuit:
    rng = random.Random(seed)
    circuit = PauliPropagationCircuit(nqubits)
    for _ in range(ngates):
        name = rng.choice(["h", "s", "x", "z", "cx", "swap"])
        if name in ("cx", "swap"):
            q0, q1 = rng.sample(range(nqubits), 2)
            getattr(circuit, name)(q0, q1)
        else:
            getattr(circuit, name)(rng.randrange(nqubits))
    return circuit


def _rotation_circuit(nqubits: int, ngates: int, seed: int) -> PauliPropagationCircuit:
    rng = random.Random(seed)
    circuit = PauliPropagationCircuit(nqubits)
    for _ in range(ngates):
        name = rng.choice(["rx", "ry", "rz", "rzz"])
        if name == "rzz":
            q0, q1 = rng.sample(range(nqubits), 2)
            circuit.rzz(q0, q1, rng.uniform(-np.pi, np.pi))
        else:
            getattr(circuit, name)(rng.randrange(nqubits), rng.uniform(-np.pi, np.pi))
    return circuit


@_benchmark_skip
@pytest.mark.benchmark
class TestPropagationBenchmarks:
    """Wall-clock benchmarks for the main Pauli propagation workloads."""

    def test_clifford_heavy(self):
        """20 qubits, 1000 random Clifford gates, sum-of-Z observable."""
        nqubits = 20
        circuit = _clifford_circuit(nqubits, 1000, seed=1)
        paulis = ["I" * q + "Z" + "I" * (nqubits - q - 1) for q in range(nqubits)]
        observable = PauliPropagationOperator(paulis, [1.0] * nqubits, num_qubits=nqubits)
        executor = PauliPropagationExecutor()

        elapsed, result = _time_call(executor.expectation_value, circuit, observable)

        print(f"\n[bench] clifford_heavy_20q_1000g: {elapsed * 1e3:.2f} ms (result={result:.6f})")
        assert elapsed < 60

    def test_rotation_heavy_truncated(self):
        """12 qubits, 80 random rotations, coefficient + weight truncation."""
        nqubits = 12
        circuit = _rotation_circuit(nqubits, 80, seed=2)
        observable = PauliPropagationOperator(
            ["Z" + "I" * (nqubits - 1)], [1.0], num_qubits=nqubits
        )
        executor = PauliPropagationExecutor(truncate_threshold=1e-8, max_weight=6)

        elapsed, result = _time_call(executor.expectation_value, circuit, observable)

        print(f"\n[bench] rotation_heavy_12q_80g: {elapsed * 1e3:.2f} ms (result={result:.6f})")
        assert elapsed < 120

    def test_multi_observable(self):
        """One 10-qubit rotation circuit, 20 single-Z observables."""
        nqubits = 10
        circuit = _rotation_circuit(nqubits, 40, seed=3)
        observables = [
            PauliPropagationOperator(
                ["I" * q + "Z" + "I" * (nqubits - q - 1)], [1.0], num_qubits=nqubits
            )
            for q in range(nqubits)
        ] * 2
        executor = PauliPropagationExecutor(truncate_threshold=1e-8, max_weight=5)

        elapsed, results = _time_call(executor.expectation_value, circuit, observables)

        print(f"\n[bench] multi_observable_10q_x20: {elapsed * 1e3:.2f} ms")
        assert len(results) == 20
        assert elapsed < 120

    def test_gradient_workload(self):
        """Parameter-shift gradient over 10 parameters on 6 qubits."""
        nqubits = 6
        nparams = 10
        params = Parameters("theta", nparams)
        rng = random.Random(4)
        circuit = PauliPropagationCircuit(nqubits)
        for i in range(nparams):
            circuit.rx(i % nqubits, params[i])
            if i % 2 == 0 and nqubits >= 2:
                q0, q1 = rng.sample(range(nqubits), 2)
                circuit.cx(q0, q1)
        observable = PauliPropagationOperator(["Z" * nqubits], [1.0])
        executor = PauliPropagationExecutor(truncate_threshold=1e-10)
        values = {"theta": [rng.uniform(-np.pi, np.pi) for _ in range(nparams)]}

        elapsed, gradient = _time_call(
            executor.expectation_value_derivatives, circuit, observable, "theta", **values
        )

        print(f"\n[bench] gradient_6q_10params: {elapsed * 1e3:.2f} ms")
        assert len(np.atleast_1d(gradient)) == nparams
        assert elapsed < 300


@_benchmark_skip
@pytest.mark.benchmark
class TestParallelBenchmarks:
    """n_jobs scaling on independent circuit evaluations.

    Note: on Windows each worker process pays spawn + package import cost,
    so parallelism only wins for workloads far heavier than the startup
    overhead. Timings are printed for comparison, not asserted.
    """

    def test_n_jobs_scaling(self):
        # Four heavy (~seconds each) independent evaluations of the same
        # circuit; heavy per-task cost is required to amortize worker startup
        nqubits = 16
        circuits = [_rotation_circuit(nqubits, 160, seed=0) for _ in range(4)]
        observable = PauliPropagationOperator(
            ["Z" + "I" * (nqubits - 1)], [1.0], num_qubits=nqubits
        )

        serial = PauliPropagationExecutor(truncate_threshold=1e-8)
        elapsed_serial, results_serial = _time_call(
            serial.expectation_value, circuits, observable, n_runs=1
        )

        parallel = PauliPropagationExecutor(truncate_threshold=1e-8, n_jobs=4)
        elapsed_parallel, results_parallel = _time_call(
            parallel.expectation_value, circuits, observable, n_runs=1
        )

        print(
            f"\n[bench] n_jobs_scaling_4x16q: serial={elapsed_serial * 1e3:.0f} ms, "
            f"n_jobs=4={elapsed_parallel * 1e3:.0f} ms"
        )
        assert np.allclose(results_serial, results_parallel)
