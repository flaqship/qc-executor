"""Cross-backend parity checks for pauli_propagation, qulacs, and pennylane.

These tests verify that all three backends produce numerically consistent outputs
for the same small circuit/operator setup when configured deterministically.
"""

from __future__ import annotations

import numpy as np
import pytest

from qc_executor import Executor, Parameters, QuantumCircuit, QuantumOperator

# Skip module when optional backends are missing in the active test environment.
pytest.importorskip("qulacs")
pytest.importorskip("pennylane")


def _build_problem():
    x = Parameters("x", 1)
    p = Parameters("p", 2)
    p_obs = Parameters("p_obs", 2)

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.ryy(0, 1, p[0] * x[0])

    operator = QuantumOperator(["ZI", "IZ"], [p_obs[0], p_obs[1]])
    return qc, operator


def _scalar_result(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(np.real_if_close(arr))
    if arr.size == 1:
        return float(np.real_if_close(arr.reshape(-1)[0]))
    raise ValueError(f"Expected scalar-like result, got shape {arr.shape}")


def _counts_dict(sample_result):
    if isinstance(sample_result, list):
        if len(sample_result) != 1:
            raise ValueError(f"Expected single-sample result, got length {len(sample_result)}")
        sample_result = sample_result[0]
    if not isinstance(sample_result, dict):
        raise TypeError(f"Expected dict counts, got {type(sample_result).__name__}")
    return sample_result


def _normalize_counts(counts):
    total = float(sum(counts.values()))
    if total <= 0:
        raise ValueError("Counts total must be positive")
    return {k: v / total for k, v in counts.items()}


def _tv_distance(a_probs, b_probs):
    keys = set(a_probs.keys()) | set(b_probs.keys())
    return 0.5 * sum(abs(a_probs.get(k, 0.0) - b_probs.get(k, 0.0)) for k in keys)


def _align_global_phase(vec, ref):
    vec = np.asarray(vec, dtype=complex).reshape(-1)
    ref = np.asarray(ref, dtype=complex).reshape(-1)

    idx = int(np.argmax(np.abs(ref)))
    if np.abs(vec[idx]) < 1e-12:
        return vec
    phase = ref[idx] / vec[idx]
    return vec * phase


class TestCrossBackendParity:
    def test_expectation_value_parity(self):
        qc, operator = _build_problem()

        kwargs = {"x": [0.1], "p": [0.3], "p_obs": [0.5, 0.6]}
        backends = ("pauli_propagation", "qulacs", "pennylane")

        values = {}
        for backend in backends:
            executor = Executor.create(backend, seed=0)
            native_circuit = executor.transpile_circuit(qc)
            test_observable = (
                executor.transpile_operator(operator)
                if backend == "pauli_propagation"
                else operator
            )
            value = executor.expectation_value(native_circuit, test_observable, **kwargs)
            values[backend] = _scalar_result(value)

        assert np.isclose(values["pauli_propagation"], values["qulacs"], atol=1e-8)
        assert np.isclose(values["pauli_propagation"], values["pennylane"], atol=1e-8)

    def test_derivative_parity(self):
        qc, operator = _build_problem()

        kwargs = {"x": [0.1], "p": [0.3], "p_obs": [0.5, 0.6]}
        backends = ("pauli_propagation", "qulacs", "pennylane")

        derivatives = {}
        for backend in backends:
            executor = Executor.create(backend, seed=0)
            native_circuit = executor.transpile_circuit(qc)
            test_observable = (
                executor.transpile_operator(operator)
                if backend == "pauli_propagation"
                else operator
            )
            result = executor.expectation_value_derivatives(
                native_circuit,
                test_observable,
                "x",
                "p",
                "p_obs",
                **kwargs,
            )
            derivatives[backend] = result

        for key in ("x", "p", "p_obs"):
            # Backends agree numerically but not on trailing singleton axes;
            # compare the flattened gradients.
            # TODO: Decide whether the derivative return shape is part of the
            # executor contract. If it is, fix the deviating backend and
            # assert the shape here instead of flattening it away.
            pp_val = np.asarray(derivatives["pauli_propagation"][key], dtype=float).reshape(-1)
            qulacs_val = np.asarray(derivatives["qulacs"][key], dtype=float).reshape(-1)
            pennylane_val = np.asarray(derivatives["pennylane"][key], dtype=float).reshape(-1)

            assert np.allclose(pp_val, qulacs_val, atol=1e-7)
            assert np.allclose(pp_val, pennylane_val, atol=1e-7)

    def test_statevector_parity_up_to_global_phase(self):
        qc, _ = _build_problem()

        kwargs = {"x": [0.8], "p": [0.5]}
        backends = ("pauli_propagation", "qulacs", "pennylane")

        vectors = {}
        for backend in backends:
            executor = Executor.create(backend, seed=0)
            native_circuit = executor.transpile_circuit(qc)
            vec = executor.statevector(native_circuit, **kwargs)
            vectors[backend] = np.asarray(vec, dtype=complex).reshape(-1)

        pp_ref = vectors["pauli_propagation"]
        qulacs_aligned = _align_global_phase(vectors["qulacs"], pp_ref)
        pennylane_aligned = _align_global_phase(vectors["pennylane"], pp_ref)

        assert np.allclose(qulacs_aligned, pp_ref, atol=1e-8)
        assert np.allclose(pennylane_aligned, pp_ref, atol=1e-8)

    def test_sample_distribution_parity(self):
        qc, _ = _build_problem()

        kwargs = {"x": [0.8], "p": [0.5]}
        backends = ("pauli_propagation", "qulacs", "pennylane")

        probs = {}
        for backend in backends:
            executor = Executor.create(backend, seed=0, shots=10000)
            native_circuit = executor.transpile_circuit(qc)
            sample = executor.sample(native_circuit, **kwargs)
            counts = _counts_dict(sample)
            probs[backend] = _normalize_counts(counts)

        # Compare empirical distributions using total variation distance.
        assert _tv_distance(probs["pauli_propagation"], probs["qulacs"]) < 0.08
        assert _tv_distance(probs["pauli_propagation"], probs["pennylane"]) < 0.08
