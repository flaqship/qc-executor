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

# Gate pools. All gates now carry exact Heisenberg transforms (t() builds an
# RZ(π/4) rotation); CNOT/CZ are pooled separately purely to keep seeds
# stable across circuit-generation variants.
_SAFE_SINGLE = ["h", "s", "t", "x", "y", "z"]
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
    def test_symmetry_merging_per_layer(self, seed):
        """Symmetry merging fires on the input AND after every layer.

        The symmetry strategy survives gate application (it used to be lost
        after the first gate), so the advertised per-layer merge actually
        runs. The reference implements the same semantics independently.
        """
        nqubits = 5
        circuit = _random_circuit(nqubits, 20, seed)
        observable = _random_observable(nqubits, 5, seed + 100)
        observable.symmetry = PermutationSymmetry()

        produced = propagate(circuit.gates, observable.copy())
        expected = ref_propagate(circuit.gates, observable.copy())

        _assert_terms_close(produced, expected)
        # The strategy must survive propagation on the returned sum
        assert isinstance(produced.symmetry, PermutationSymmetry)


class TestStatevectorParity:
    """Expectation values must match direct statevector simulation."""

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
    def test_cnot_cz_gates_match_statevector(self, seed):
        """CNOT/CZ conjugation phases were fixed; parity is now strict."""
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


class TestArrayEngineParity:
    """Dict engine vs numpy array engine must agree."""

    @staticmethod
    def _propagate_with(monkeypatch, use_arrays, gates, observable, **kwargs):
        from qc_executor.pauli_propagation.utils import propagation

        monkeypatch.setattr(propagation, "USE_ARRAY_ENGINE", use_arrays)
        if use_arrays:
            # Force the array engine from the very first gate
            monkeypatch.setattr(propagation, "_ARRAY_ENGINE_MIN_TERMS", 1)
        return propagate(gates, observable.copy(), **kwargs)

    @pytest.mark.parametrize("seed", [70, 71, 72, 73])
    def test_engines_agree_mixed_gates(self, seed, monkeypatch):
        nqubits = 5 + seed % 3
        circuit = _random_circuit(nqubits, 30, seed, include_cnot_cz=True)
        observable = _random_observable(nqubits, 4, seed + 100)

        via_dict = self._propagate_with(monkeypatch, False, circuit.gates, observable)
        via_arrays = self._propagate_with(monkeypatch, True, circuit.gates, observable)

        _assert_terms_close(via_arrays, via_dict)

    @pytest.mark.parametrize("seed", [80, 81])
    def test_engines_agree_with_truncation(self, seed, monkeypatch):
        nqubits = 8
        circuit = _random_circuit(nqubits, 40, seed, include_cnot_cz=True)
        observable = _random_observable(nqubits, 3, seed + 100)

        kwargs = {"max_weight": 4, "truncate_threshold": 1e-8}
        via_dict = self._propagate_with(monkeypatch, False, circuit.gates, observable, **kwargs)
        via_arrays = self._propagate_with(monkeypatch, True, circuit.gates, observable, **kwargs)

        _assert_terms_close(via_arrays, via_dict)

    @pytest.mark.parametrize("seed", [90])
    def test_engines_agree_with_symmetry(self, seed, monkeypatch):
        nqubits = 6
        circuit = _random_circuit(nqubits, 25, seed)
        observable = _random_observable(nqubits, 5, seed + 100)
        observable.symmetry = PermutationSymmetry()

        via_dict = self._propagate_with(monkeypatch, False, circuit.gates, observable)
        via_arrays = self._propagate_with(monkeypatch, True, circuit.gates, observable)

        _assert_terms_close(via_arrays, via_dict)

    @pytest.mark.parametrize("nqubits", [31, 32, 33])
    def test_dispatch_boundary(self, nqubits):
        """31/32 qubits go through uint64 arrays, 33 through big-int dicts.

        The observable acts on the top qubit, so at 32 qubits bit 63 is in
        use. Results must match the frozen reference either way.
        """
        top = nqubits - 1
        circuit = PauliPropagationCircuit(nqubits)
        circuit.rx(top, 0.7)
        circuit.rz(top, -0.3)
        circuit.rzz(top - 1, top, 1.1)
        circuit.h(top - 1)
        circuit.rx(top, 0.4)

        observable = PauliSum(nqubits)
        observable.add_term(3 << (2 * top), 1.0)  # Z on top qubit
        observable.add_term((3 << (2 * top)) | (1 << (2 * (top - 1))), 0.5)  # X(top-1) Z(top)

        produced = propagate(circuit.gates, observable.copy())
        expected = ref_propagate(circuit.gates, observable.copy())

        _assert_terms_close(produced, expected)
        assert all(isinstance(t, int) for t in produced.terms)
        assert all(isinstance(c, complex) for c in produced.terms.values())

    def test_min_terms_switchover(self, monkeypatch):
        """Default config: dict engine below threshold, arrays above; results equal."""
        nqubits = 10
        circuit = _random_circuit(nqubits, 45, seed=95, rotations_only=True)
        observable = _random_observable(nqubits, 2, seed=195)

        default = propagate(circuit.gates, observable.copy())
        via_dict = self._propagate_with(monkeypatch, False, circuit.gates, observable)

        _assert_terms_close(default, via_dict)


class TestPopcountSwar:
    """The SWAR popcount fallback must match int.bit_count exactly."""

    def test_swar_matches_bit_count(self):
        from qc_executor.pauli_propagation.utils.array_engine import popcount_swar

        rng = random.Random(11)
        values = [0, 1, 2**64 - 1, 2**63, 0x5555555555555555] + [
            rng.getrandbits(64) for _ in range(500)
        ]
        arr = np.array(values, dtype=np.uint64)
        counts = popcount_swar(arr)
        for value, count in zip(values, counts.tolist()):
            assert count == value.bit_count(), value
