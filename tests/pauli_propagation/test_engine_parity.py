"""Parity tests pinning Pauli propagation behavior against a frozen reference.

These tests guard the performance-optimization work: the production
implementation may change internally (bit-trick primitives, lookup tables,
vectorized engines), but its observable behavior must stay equal to the
frozen snapshot in ``reference_propagation.py`` and, where the gate set is
free of known transformation bugs, to a direct statevector simulation.
"""

import random

import numpy as np
import pytest

from qc_executor.pauli_propagation import (
    PauliPropagationCircuit,
    PauliPropagationExecutor,
    PauliPropagationOperator,
)
from qc_executor.pauli_propagation.symmetry import PermutationSymmetry
from qc_executor.pauli_propagation.utils.pauli_algebra import pauli_to_matrix
from qc_executor.pauli_propagation.utils.pauli_types import PauliSum
from qc_executor.pauli_propagation.utils.propagation import propagate
from qc_executor.pauli_propagation.utils.state_overlap import overlap_with_zero

from .reference_propagation import ref_overlap_with_zero, ref_propagate

# Gate pools. CNOT and CZ carry known transformation bugs for Y operands
# (see plan notes); they are exercised separately so the strict statevector
# parity tests stay meaningful.
_SAFE_SINGLE = ["h", "s", "x", "y", "z"]
_ROTATIONS_1Q = ["rx", "ry", "rz"]
_ROTATIONS_2Q = ["rxx", "ryy", "rzz"]


def _random_circuit(
    nqubits: int,
    ngates: int,
    seed: int,
    include_cnot_cz: bool = False,
    rotations_only: bool = False,
) -> PauliPropagationCircuit:
    """Build a seeded random circuit with concrete rotation angles."""
    rng = random.Random(seed)
    circuit = PauliPropagationCircuit(nqubits)

    pool = list(_ROTATIONS_1Q)
    if not rotations_only:
        pool += _SAFE_SINGLE + ["swap"]
    if nqubits >= 2:
        pool += _ROTATIONS_2Q
        if include_cnot_cz:
            pool += ["cx", "cz"]

    for _ in range(ngates):
        name = rng.choice(pool)
        if name in _SAFE_SINGLE:
            getattr(circuit, name)(rng.randrange(nqubits))
        elif name in _ROTATIONS_1Q:
            getattr(circuit, name)(rng.randrange(nqubits), rng.uniform(-np.pi, np.pi))
        elif name in _ROTATIONS_2Q:
            q0, q1 = rng.sample(range(nqubits), 2)
            getattr(circuit, name)(q0, q1, rng.uniform(-np.pi, np.pi))
        else:  # swap / cx / cz
            q0, q1 = rng.sample(range(nqubits), 2)
            getattr(circuit, name)(q0, q1)

    return circuit


def _random_observable(nqubits: int, nterms: int, seed: int) -> PauliSum:
    """Build a seeded random observable with weight-1..3 Pauli terms."""
    rng = random.Random(seed)
    psum = PauliSum(nqubits)
    for _ in range(nterms):
        weight = rng.randint(1, min(3, nqubits))
        qubits = rng.sample(range(nqubits), weight)
        term = 0
        for q in qubits:
            term |= rng.randint(1, 3) << (2 * q)
        psum.add_term(term, rng.uniform(-1.0, 1.0))
    return psum


def _assert_terms_close(actual: PauliSum, expected: PauliSum, rtol=1e-9, atol=1e-12):
    assert set(actual.terms.keys()) == set(expected.terms.keys())
    for term, coeff in expected.terms.items():
        assert np.isclose(
            actual.terms[term], coeff, rtol=rtol, atol=atol
        ), f"Coefficient mismatch for term {term}: {actual.terms[term]} vs {coeff}"


def _statevector_expectation(circuit: PauliPropagationCircuit, psum: PauliSum) -> float:
    executor = PauliPropagationExecutor()
    state = executor._simulate_statevector(circuit, {})  # pylint: disable=protected-access
    matrix = sum(coeff * pauli_to_matrix(term, psum.nqubits) for term, coeff in psum.terms.items())
    return float(np.real(np.conj(state) @ (matrix @ state)))


class TestReferenceParity:
    """Production propagate() must match the frozen reference exactly."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_full_terms_parity_mixed_gates(self, seed):
        nqubits = 5 + seed % 3
        circuit = _random_circuit(nqubits, 25, seed, include_cnot_cz=True)
        observable = _random_observable(nqubits, 4, seed + 100)

        produced = propagate(circuit.gates, observable.copy())
        expected = ref_propagate(circuit.gates, observable.copy())

        _assert_terms_close(produced, expected)

    @pytest.mark.parametrize("seed", [10, 11, 12])
    def test_full_terms_parity_with_truncation(self, seed):
        nqubits = 6
        circuit = _random_circuit(nqubits, 30, seed, include_cnot_cz=True)
        observable = _random_observable(nqubits, 3, seed + 100)

        produced = propagate(
            circuit.gates, observable.copy(), max_weight=3, truncate_threshold=1e-6
        )
        expected = ref_propagate(
            circuit.gates, observable.copy(), max_weight=3, truncate_threshold=1e-6
        )

        _assert_terms_close(produced, expected)

    @pytest.mark.parametrize("seed", [20, 21])
    def test_overlap_parity(self, seed):
        nqubits = 6
        circuit = _random_circuit(nqubits, 25, seed, include_cnot_cz=True)
        observable = _random_observable(nqubits, 4, seed + 100)

        produced = propagate(circuit.gates, observable.copy())
        expected = ref_propagate(circuit.gates, observable.copy())

        assert np.isclose(
            overlap_with_zero(produced), ref_overlap_with_zero(expected), rtol=1e-9, atol=1e-12
        )

    @pytest.mark.parametrize("seed", [30, 31])
    def test_symmetry_status_quo(self, seed):
        """Pin the current symmetry-merging semantics.

        Today the input observable is merged once before propagation, and the
        advertised per-layer merge never fires afterwards (the per-gate
        helpers return PauliSums carrying default NoSymmetry). The frozen
        reference replicates exactly that; production must agree.
        """
        nqubits = 5
        circuit = _random_circuit(nqubits, 20, seed)
        observable = _random_observable(nqubits, 5, seed + 100)
        observable.symmetry = PermutationSymmetry()

        produced = propagate(circuit.gates, observable.copy())
        expected = ref_propagate(circuit.gates, observable.copy())

        _assert_terms_close(produced, expected)


class TestStatevectorParity:
    """Expectation values must match direct statevector simulation.

    Restricted to gates whose Heisenberg transforms are correct today
    (rotations, H/S/X/Y/Z, SWAP). CNOT/CZ circuits are exercised in a
    non-strict xfail test: they carry known phase bugs for Y operands that
    this performance work must preserve, not fix.
    """

    @pytest.mark.parametrize("seed", [40, 41, 42, 43])
    def test_safe_gates_match_statevector(self, seed):
        nqubits = 4 + seed % 2
        circuit = _random_circuit(nqubits, 20, seed)
        observable = _random_observable(nqubits, 4, seed + 100)

        propagated = propagate(circuit.gates, observable.copy())
        pp_value = float(np.real(overlap_with_zero(propagated)))
        sv_value = _statevector_expectation(circuit, observable)

        assert np.isclose(pp_value, sv_value, atol=1e-8)

    @pytest.mark.parametrize("seed", [50, 51, 52])
    @pytest.mark.xfail(
        strict=False,
        reason="Known CNOT/CZ transformation bugs for Y operands (pre-existing)",
    )
    def test_cnot_cz_gates_match_statevector(self, seed):
        nqubits = 4
        circuit = _random_circuit(nqubits, 20, seed, include_cnot_cz=True)
        observable = _random_observable(nqubits, 4, seed + 100)

        propagated = propagate(circuit.gates, observable.copy())
        pp_value = float(np.real(overlap_with_zero(propagated)))
        sv_value = _statevector_expectation(circuit, observable)

        assert np.isclose(pp_value, sv_value, atol=1e-8)


class TestExecutorParity:
    """End-to-end executor results must match the frozen reference."""

    @pytest.mark.parametrize("seed", [60, 61])
    def test_executor_expectation_matches_reference(self, seed):
        nqubits = 5
        circuit = _random_circuit(nqubits, 20, seed, include_cnot_cz=True)
        observable = _random_observable(nqubits, 4, seed + 100)

        paulis, coeffs = [], []
        from qc_executor.pauli_propagation.utils.pauli_algebra import term_to_string

        for term, coeff in observable.terms.items():
            paulis.append(term_to_string(term, nqubits))
            coeffs.append(complex(coeff).real)
        operator = PauliPropagationOperator(paulis, coeffs, num_qubits=nqubits)

        executor = PauliPropagationExecutor()
        result = executor.expectation_value(circuit, operator)

        expected = ref_overlap_with_zero(ref_propagate(circuit.gates, operator.pauli_sum))
        assert np.isclose(result, float(np.real(expected)), rtol=1e-9, atol=1e-12)
