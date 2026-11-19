"""Tests for batched and process-parallel expectation evaluation (n_jobs)."""

import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from unittest.mock import patch

import numpy as np
import pytest

from qc_executor import QuantumCircuit, QuantumOperator
from qc_executor.parameters import Parameters
from qc_executor.pauli_propagation import (
    PauliPropagationCircuit,
    PauliPropagationExecutor,
    PauliPropagationOperator,
)
from qc_executor.pauli_propagation.symmetry import NoSymmetry, PermutationSymmetry
from qc_executor.pauli_propagation.utils.pauli_types import PauliSum


def _rotation_circuit(nqubits, ngates, seed):
    rng = random.Random(seed)
    circuit = PauliPropagationCircuit(nqubits)
    for _ in range(ngates):
        name = rng.choice(["rx", "ry", "rz", "rzz", "cx", "h"])
        if name in ("rzz", "cx"):
            q0, q1 = rng.sample(range(nqubits), 2)
            if name == "rzz":
                circuit.rzz(q0, q1, rng.uniform(-np.pi, np.pi))
            else:
                circuit.cx(q0, q1)
        elif name == "h":
            circuit.h(rng.randrange(nqubits))
        else:
            getattr(circuit, name)(rng.randrange(nqubits), rng.uniform(-np.pi, np.pi))
    return circuit


def _observables(nqubits, count):
    return [
        PauliPropagationOperator(
            ["I" * (q % nqubits) + "Z" + "I" * (nqubits - (q % nqubits) - 1)],
            [1.0],
            num_qubits=nqubits,
        )
        for q in range(count)
    ]


class TestNJobsValidation:
    def test_default_serial(self):
        assert PauliPropagationExecutor().n_jobs == 1

    def test_minus_one_allowed(self):
        assert PauliPropagationExecutor(n_jobs=-1).n_jobs == -1

    @pytest.mark.parametrize("bad", [0, -2, 1.5, "2"])
    def test_invalid_rejected(self, bad):
        with pytest.raises(ValueError, match="n_jobs"):
            PauliPropagationExecutor(n_jobs=bad)


class TestBatchPathEquivalence:
    """Multi-observable / multi-circuit results must equal per-call results."""

    def test_multi_observable_matches_individual_calls(self):
        nqubits = 5
        circuit = _rotation_circuit(nqubits, 20, seed=1)
        observables = _observables(nqubits, 5)
        executor = PauliPropagationExecutor(truncate_threshold=1e-10)

        batched = executor.expectation_value(circuit, observables)
        individual = [executor.expectation_value(circuit, obs) for obs in observables]

        assert np.allclose(batched, individual)

    def test_multi_circuit_multi_observable_order(self):
        nqubits = 4
        circuits = [_rotation_circuit(nqubits, 15, seed=s) for s in (2, 3)]
        observables = _observables(nqubits, 3)
        executor = PauliPropagationExecutor()

        batched = executor.expectation_value(circuits, observables)

        # Circuit-major ordering
        expected = [
            [executor.expectation_value(circ, obs) for obs in observables] for circ in circuits
        ]
        assert np.allclose(batched, expected)

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_cartesian_parameter_sweeps(self, n_jobs):
        p = Parameters("p", 1)
        x = Parameters("x", 1)
        circuit = PauliPropagationCircuit(1)
        circuit.ry(0, p[0] * x[0])
        observable = PauliPropagationOperator(["Z"], [1.0])

        result = PauliPropagationExecutor(n_jobs=n_jobs).expectation_value(
            circuit, observable, p=[[0.2], [0.4]], x=[[0.1], [0.3], [0.5]]
        )

        assert np.shape(result) == (2, 3)
        np.testing.assert_allclose(result, np.cos(np.outer([0.2, 0.4], [0.1, 0.3, 0.5])))

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_cartesian_observable_and_shared_parameters(self, n_jobs):
        x = Parameters("x", 1)
        coefficient = Parameters("coefficient", 1)
        circuit = PauliPropagationCircuit(1)
        circuit.ry(0, x[0])
        executor = PauliPropagationExecutor(n_jobs=n_jobs)

        result = executor.expectation_value(
            circuit,
            PauliPropagationOperator(["Z"], [coefficient[0]]),
            x=[[0.1], [0.5], [1.0]],
            coefficient=[[1.0], [2.0]],
        )
        assert np.shape(result) == (3, 2)
        np.testing.assert_allclose(result, np.cos([0.1, 0.5, 1.0])[:, None] * [1.0, 2.0])

        shared = executor.expectation_value(
            circuit, PauliPropagationOperator(["Z"], [x[0]]), x=[[0.2], [0.7]]
        )
        assert np.shape(shared) == (2, 2)
        np.testing.assert_allclose(shared, np.cos([0.2, 0.7])[:, None] * [0.2, 0.7])

    def test_parameter_batch_uses_worker_pool(self):
        x = Parameters("x", 1)
        circuit = PauliPropagationCircuit(1)
        circuit.ry(0, x[0])
        observable = PauliPropagationOperator(["Z"], [1.0])
        executor = PauliPropagationExecutor(n_jobs=2)

        with patch(
            "qc_executor.pauli_propagation.pauli_propagation_executor.ProcessPoolExecutor",
            wraps=ProcessPoolExecutor,
        ) as pool:
            result = executor.expectation_value(circuit, observable, x=[[0.1], [0.5], [1.0]])

        pool.assert_called_once()
        assert pool.call_args.kwargs["max_workers"] == 2
        np.testing.assert_allclose(result, np.cos([0.1, 0.5, 1.0]), atol=1e-10)

    def test_parameter_batch_with_spawn_workers(self):
        x = Parameters("x", 1)
        circuit = PauliPropagationCircuit(1)
        circuit.ry(0, x[0])
        observable = PauliPropagationOperator(["Z"], [1.0])
        executor = PauliPropagationExecutor(n_jobs=2)

        def spawn_pool(**kwargs):
            return ProcessPoolExecutor(mp_context=get_context("spawn"), **kwargs)

        with patch(
            "qc_executor.pauli_propagation.pauli_propagation_executor.ProcessPoolExecutor",
            side_effect=spawn_pool,
        ):
            result = executor.expectation_value(circuit, observable, x=[[0.1], [0.5]])

        np.testing.assert_allclose(result, np.cos([0.1, 0.5]), atol=1e-10)

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_parameter_batch_preserves_truncation_statistics(self, n_jobs):
        x = Parameters("x", 1)
        circuit = PauliPropagationCircuit(1)
        circuit.ry(0, x[0])
        observable = PauliPropagationOperator(["Z"], [1.0])
        executor = PauliPropagationExecutor(n_jobs=n_jobs, truncate_threshold=1e-10)

        result = executor.expectation_value(circuit, observable, x=[[0.1], [0.5]])

        np.testing.assert_allclose(result, np.cos([0.1, 0.5]), atol=1e-10)
        assert executor.get_truncation_stats() is not None

    def test_cartesian_vector_rows_match_other_backends(self):
        pennylane = pytest.importorskip("qc_executor.pennylane")
        qiskit = pytest.importorskip("qc_executor.qiskit")
        qulacs = pytest.importorskip("qc_executor.qulacs")

        x = Parameters("x", 2)
        p = Parameters("p", 1)
        coefficient = Parameters("coefficient", 1)
        circuit = QuantumCircuit(1)
        circuit.ry(0, p[0] * x[0])
        circuit.rz(0, x[1])
        observable = QuantumOperator(["X"], [coefficient[0]])
        values = {
            "p": [[0.2], [0.4]],
            "x": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            "coefficient": [[1.0], [2.0]],
        }

        result = PauliPropagationExecutor(n_jobs=2).expectation_value(
            circuit, observable, **values
        )
        assert np.shape(result) == (2, 3, 2)
        for executor in (
            pennylane.PennyLaneExecutor(),
            qiskit.QiskitExecutor(),
            qulacs.QulacsExecutor(),
        ):
            np.testing.assert_allclose(
                result, executor.expectation_value(circuit, observable, **values), atol=1e-10
            )

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_lists_with_circuit_and_observable_sweeps(self, n_jobs):
        x = Parameters("x", 1)
        coefficient = Parameters("coefficient", 1)
        first = PauliPropagationCircuit(1)
        first.ry(0, x[0])
        second = PauliPropagationCircuit(1)
        second.rx(0, x[0])
        circuits = [first, second]
        observables = [
            PauliPropagationOperator(["Z"], [coefficient[0]]),
            PauliPropagationOperator(["X"], [coefficient[0]]),
        ]
        executor = PauliPropagationExecutor(n_jobs=n_jobs)
        result = executor.expectation_value(
            circuits, observables, x=[[0.1], [0.5]], coefficient=[[1.0], [2.0], [3.0]]
        )
        expected = np.array(
            [
                [
                    [
                        [
                            executor.expectation_value(circ, obs, x=[angle], coefficient=[scale])
                            for scale in [1.0, 2.0, 3.0]
                        ]
                        for angle in [0.1, 0.5]
                    ]
                    for obs in observables
                ]
                for circ in circuits
            ]
        )

        assert np.shape(result) == (2, 2, 2, 3)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_empty_observable_sweep(self):
        x = Parameters("x", 1)
        coefficient = Parameters("coefficient", 1)
        circuit = PauliPropagationCircuit(1)
        circuit.ry(0, x[0])
        observable = PauliPropagationOperator(["Z"], [coefficient[0]])

        result = PauliPropagationExecutor().expectation_value(
            circuit, observable, x=[[0.1], [0.5]], coefficient=np.empty((0, 1))
        )

        assert np.shape(result) == (2, 0)

    def test_empty_batch_singleton_lists_and_mismatched_shapes(self):
        x = Parameters("x", 1)
        circuit = PauliPropagationCircuit(1)
        circuit.ry(0, x[0])
        observable = PauliPropagationOperator(["Z"], [1.0])
        executor = PauliPropagationExecutor()

        assert np.shape(executor.expectation_value(circuit, observable, x=np.empty((0, 1)))) == (
            0,
        )
        assert np.shape(executor.expectation_value([circuit], [observable], x=[[0.1], [0.5]])) == (
            1,
            1,
            2,
        )
        with pytest.raises(ValueError, match="matching parameter sweep shapes"):
            executor.expectation_value(
                [circuit, PauliPropagationCircuit(1)], observable, x=[[0.1], [0.5]]
            )

    def test_missing_parameter_has_clear_error(self):
        x = Parameters("x", 1)
        circuit = PauliPropagationCircuit(1)
        circuit.ry(0, x[0])

        with pytest.raises(ValueError, match="Parameter 'x' not found"):
            PauliPropagationExecutor().expectation_value(
                circuit, PauliPropagationOperator(["Z"], [1.0])
            )

    def test_truncation_stats_still_reported(self):
        nqubits = 4
        circuit = _rotation_circuit(nqubits, 15, seed=4)
        observables = _observables(nqubits, 3)
        executor = PauliPropagationExecutor(truncate_threshold=1e-8)

        executor.expectation_value(circuit, observables)

        assert executor.get_truncation_stats() is not None


class TestPicklability:
    """Everything shipped to spawn workers must survive a pickle round-trip."""

    def test_circuit_pickles(self):
        params = Parameters("theta", 2)
        circuit = PauliPropagationCircuit(3)
        circuit.rx(0, params[0])
        circuit.cx(0, 1)
        circuit.rzz(1, 2, params[1])
        restored = pickle.loads(pickle.dumps(circuit))
        assert restored.num_qubits == 3
        assert len(restored.gates) == len(circuit.gates)

    def test_operator_pickles(self):
        operator = PauliPropagationOperator(["ZZI", "IXY"], [1.0, 0.5])
        restored = pickle.loads(pickle.dumps(operator))
        assert restored.num_qubits == 3

    def test_pauli_sum_and_symmetries_pickle(self):
        psum = PauliSum(3, symmetry=PermutationSymmetry())
        psum.add_term("ZZI", 0.7)
        restored = pickle.loads(pickle.dumps(psum))
        assert restored.terms == psum.terms
        assert isinstance(pickle.loads(pickle.dumps(NoSymmetry())), NoSymmetry)

    def test_gate_with_built_tables_pickles(self):
        circuit = PauliPropagationCircuit(2)
        circuit.cx(0, 1)
        gate = circuit.gates[0]
        gate.transform_pauli_term(0b0101)  # force table construction
        gate.transform_table_numpy()
        restored = pickle.loads(pickle.dumps(gate))
        assert restored.transform_pauli_term(0b0101) == gate.transform_pauli_term(0b0101)


@pytest.mark.slow
class TestNJobsEquivalence:
    """n_jobs > 1 must produce identical results to serial execution.

    Kept small: each spawn-based pool costs worker startup + package import.
    """

    def test_expectation_values_match(self):
        nqubits = 5
        circuits = [_rotation_circuit(nqubits, 15, seed=s) for s in (5, 6)]
        observables = _observables(nqubits, 3)

        serial = PauliPropagationExecutor(truncate_threshold=1e-10)
        parallel = PauliPropagationExecutor(truncate_threshold=1e-10, n_jobs=2)

        serial_results = serial.expectation_value(circuits, observables)
        parallel_results = parallel.expectation_value(circuits, observables)

        assert np.allclose(serial_results, parallel_results)
        # Stats semantics preserved (last pair's stats)
        assert serial.get_truncation_stats() is not None
        assert parallel.get_truncation_stats() is not None

    def test_gradients_match(self):
        nqubits = 3
        nparams = 4
        params = Parameters("theta", nparams)
        circuit = PauliPropagationCircuit(nqubits)
        for i in range(nparams):
            circuit.rx(i % nqubits, params[i])
            circuit.cx(i % nqubits, (i + 1) % nqubits)
        observable = PauliPropagationOperator(["Z" * nqubits], [1.0])
        rng = random.Random(7)
        values = {"theta": [rng.uniform(-np.pi, np.pi) for _ in range(nparams)]}

        serial = PauliPropagationExecutor()
        parallel = PauliPropagationExecutor(n_jobs=2)

        grad_serial = serial.expectation_value_derivatives(circuit, observable, "theta", **values)
        grad_parallel = parallel.expectation_value_derivatives(
            circuit, observable, "theta", **values
        )

        assert np.allclose(np.atleast_1d(grad_serial), np.atleast_1d(grad_parallel))


class TestInternalHelpers:
    """Coverage of executor helpers that back the parallel machinery."""

    def test_resolved_n_jobs_minus_one_uses_all_cores(self):
        import os

        executor = PauliPropagationExecutor(n_jobs=-1)
        assert executor._resolved_n_jobs() == (os.cpu_count() or 1)

    def test_resolved_n_jobs_passthrough(self):
        assert PauliPropagationExecutor(n_jobs=3)._resolved_n_jobs() == 3

    def test_compute_single_expectation_compat(self):
        """The kept-for-compat single-pair method matches the public API."""
        circuit = PauliPropagationCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        observable = PauliPropagationOperator(["ZZ"], [1.0])

        executor = PauliPropagationExecutor(truncate_threshold=1e-10)
        value = executor._compute_single_expectation(circuit, observable, {})

        assert np.isclose(value.real, executor.expectation_value(circuit, observable))
        # The truncation stats of the final pass are recorded
        assert executor.last_truncation_stats is not None
