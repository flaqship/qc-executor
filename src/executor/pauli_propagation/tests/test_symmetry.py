"""Tests for symmetry module."""

import os
import time

import numpy as np
import pytest

from executor.pauli_propagation.pauli_types import PauliString, PauliSum
from executor.pauli_propagation.symmetry import (CompositeSymmetry, NoSymmetry,
                                                 PermutationSymmetry)

# Benchmark tests are skipped by default to avoid flakiness due to scheduler noise
# and hardware differences. Set RUN_BENCHMARKS=1 to enable them.
_RUN_BENCHMARKS = os.environ.get("RUN_BENCHMARKS", "0") == "1"
_benchmark_skip = pytest.mark.skipif(
    not _RUN_BENCHMARKS,
    reason="Benchmark tests are skipped by default; set RUN_BENCHMARKS=1 to enable",
)


class TestNoSymmetry:
    """Test NoSymmetry strategy."""

    def test_canonical_identity(self):
        """NoSymmetry should return input unchanged."""
        sym = NoSymmetry()
        assert sym.canonical_representative(0b11100100, 4) == 0b11100100
        assert sym.canonical_representative(0b00000000, 4) == 0b00000000
        assert sym.canonical_representative(0xFFFFFFFF, 16) == 0xFFFFFFFF

    def test_name(self):
        """NoSymmetry should have descriptive name."""
        sym = NoSymmetry()
        assert "no" in sym.name.lower() or "identity" in sym.name.lower()


class TestPermutationSymmetry:
    """Test PermutationSymmetry strategy."""

    def test_all_identical_paulis(self):
        """All-X, all-Y, all-Z should be canonical (lexicographically first)."""
        sym = PermutationSymmetry()
        nqubits = 4

        # XXXX = 01010101 in binary (2 bits per X)
        assert sym.canonical_representative(0b01010101, nqubits) == 0b01010101

        # YYYY = 10101010
        assert sym.canonical_representative(0b10101010, nqubits) == 0b10101010

        # ZZZZ = 11111111
        assert sym.canonical_representative(0b11111111, nqubits) == 0b11111111

        # IIII = 00000000
        assert sym.canonical_representative(0b00000000, nqubits) == 0b00000000

    def test_two_qubit_permutations(self):
        """2-qubit case: IX and XI should map to same canonical form."""
        sym = PermutationSymmetry()
        nqubits = 2

        # In little-endian encoding (2 bits per qubit):
        # 0b0001 = bits[1:0]=01 (X), bits[3:2]=00 (I) → XI (qubit 0 = X, qubit 1 = I)
        # 0b0100 = bits[1:0]=00 (I), bits[3:2]=01 (X) → IX (qubit 0 = I, qubit 1 = X)
        # Canonical form of multiset {I, X} is IX (sorted: I first, then X)
        # This corresponds to 0b0100 (IX in encoding)

        xi_term = 0b0001  # XI
        ix_term = 0b0100  # IX

        xi_canonical = sym.canonical_representative(xi_term, nqubits)
        ix_canonical = sym.canonical_representative(ix_term, nqubits)

        # Both should map to IX (the sorted canonical form)
        assert xi_canonical == ix_canonical
        assert xi_canonical == 0b0100  # IX is canonical (I before X in sorted order)

    def test_three_qubit_multiset(self):
        """3-qubit: XIZ, IXZ, ZIX, etc. should all map to same canonical."""
        sym = PermutationSymmetry()
        nqubits = 3

        # Multiset {I, X, Z} has 3! = 6 permutations
        # Canonical form is IXZ (sorted order)

        # IXZ = I(q0) X(q1) Z(q2) = 0b110100 = 0x34
        # XIZ = X(q0) I(q1) Z(q2) = 0b110001 = 0x31
        # ZIX = Z(q0) I(q1) X(q2) = 0b010011 = 0x13
        # IZX = I(q0) Z(q1) X(q2) = 0b011100 = 0x1C
        # XZI = X(q0) Z(q1) I(q2) = 0b001101 = 0x0D
        # ZXI = Z(q0) X(q1) I(q2) = 0b000111 = 0x07

        ixz = 0b110100
        xiz = 0b110001
        zix = 0b010011
        izx = 0b011100
        xzi = 0b001101
        zxi = 0b000111

        canonical = sym.canonical_representative(ixz, nqubits)

        assert sym.canonical_representative(xiz, nqubits) == canonical
        assert sym.canonical_representative(zix, nqubits) == canonical
        assert sym.canonical_representative(izx, nqubits) == canonical
        assert sym.canonical_representative(xzi, nqubits) == canonical
        assert sym.canonical_representative(zxi, nqubits) == canonical

        # Canonical should be IXZ (I=00, X=01, Z=11 sorted)
        assert canonical == ixz

    def test_repeated_paulis(self):
        """Terms with repeated Paulis (e.g., XXY, XYX, YXX) should merge."""
        sym = PermutationSymmetry()
        nqubits = 3

        # XXY = X(q0) X(q1) Y(q2) = 0b100101
        # XYX = X(q0) Y(q1) X(q2) = 0b011001
        # YXX = Y(q0) X(q1) X(q2) = 0b010110

        xxy = 0b100101
        xyx = 0b011001
        yxx = 0b010110

        canonical = sym.canonical_representative(xxy, nqubits)

        assert sym.canonical_representative(xyx, nqubits) == canonical
        assert sym.canonical_representative(yxx, nqubits) == canonical

        # Canonical form should be XXY (sorted: X, X, Y)
        assert canonical == xxy

    def test_all_identities(self):
        """All-identity should be unchanged."""
        sym = PermutationSymmetry()
        nqubits = 5
        assert sym.canonical_representative(0b0000000000, nqubits) == 0b0000000000

    def test_single_non_identity(self):
        """Single non-identity Pauli should be canonical at first position."""
        sym = PermutationSymmetry()
        nqubits = 3

        # IIX = 0b010000
        # IXI = 0b000100
        # XII = 0b000001
        # All should map to IIX (sorted: I, I, X)

        iix = 0b010000
        ixi = 0b000100
        xii = 0b000001

        canonical = sym.canonical_representative(iix, nqubits)

        assert sym.canonical_representative(ixi, nqubits) == canonical
        assert sym.canonical_representative(xii, nqubits) == canonical
        assert canonical == iix

    def test_large_system(self):
        """Verify correctness on larger systems (10 qubits)."""
        sym = PermutationSymmetry()
        nqubits = 10

        # Create two permutations of the same multiset
        # Multiset: 5 I's, 3 X's, 1 Y, 1 Z
        # Permutation 1: IIIIIXXX YZ
        # Permutation 2: XIIIIXXY IZ

        # Build term1: IIIIIXXXYZ (qubits 0-9)
        # I=00, X=01, Y=10, Z=11
        term1_str = "IIIIIXXXYZ"
        term1 = sum(
            ({"I": 0, "X": 1, "Y": 2, "Z": 3}[char]) << (2 * i) for i, char in enumerate(term1_str)
        )

        # Build term2: XIIIIXXIYZ (different permutation)
        term2_str = "XIIIXXIYIZ"
        term2 = sum(
            ({"I": 0, "X": 1, "Y": 2, "Z": 3}[char]) << (2 * i) for i, char in enumerate(term2_str)
        )

        # Both should map to same canonical (sorted: 5 I's, 3 X's, 1 Y, 1 Z)
        canonical1 = sym.canonical_representative(term1, nqubits)
        canonical2 = sym.canonical_representative(term2, nqubits)

        assert canonical1 == canonical2

    def test_name(self):
        """PermutationSymmetry should have descriptive name."""
        sym = PermutationSymmetry()
        name = sym.name
        assert "permutation" in name.lower()


class TestCompositeSymmetry:
    """Test CompositeSymmetry strategy composition."""

    def test_empty_composition(self):
        """Empty composition should act like NoSymmetry."""
        sym = CompositeSymmetry()  # Empty strategies
        assert sym.canonical_representative(0b11001100, 4) == 0b11001100

    def test_single_strategy(self):
        """Single strategy should behave identically to that strategy."""
        perm_sym = PermutationSymmetry()
        comp_sym = CompositeSymmetry(perm_sym)
        nqubits = 3

        test_term = 0b110001  # XIZ
        assert comp_sym.canonical_representative(
            test_term, nqubits
        ) == perm_sym.canonical_representative(test_term, nqubits)

    def test_composition_order(self):
        """Composition should apply strategies in order."""

        # Create two dummy strategies that modify the term
        class IncrementSymmetry:
            """Test strategy that increments the term."""

            def canonical_representative(self, term: int, nqubits: int) -> int:
                return term + 1

            @property
            def name(self) -> str:
                return "increment"

        class DoubleSymmetry:
            """Test strategy that doubles the term."""

            def canonical_representative(self, term: int, nqubits: int) -> int:
                return term * 2

            @property
            def name(self) -> str:
                return "double"

        # Order 1: Increment then Double
        comp1 = CompositeSymmetry(IncrementSymmetry(), DoubleSymmetry())
        # Input 5 -> Increment -> 6 -> Double -> 12
        assert comp1.canonical_representative(5, 1) == 12

        # Order 2: Double then Increment
        comp2 = CompositeSymmetry(DoubleSymmetry(), IncrementSymmetry())
        # Input 5 -> Double -> 10 -> Increment -> 11
        assert comp2.canonical_representative(5, 1) == 11

    def test_name(self):
        """CompositeSymmetry should describe all composed strategies."""
        perm = PermutationSymmetry()
        nosym = NoSymmetry()
        comp = CompositeSymmetry(perm, nosym)

        name = comp.name
        assert "composite" in name.lower()
        assert "permutation" in name.lower()


class TestPauliSumSymmetryIntegration:
    """Test symmetry integration with PauliSum."""

    def test_pauli_sum_default_no_symmetry(self):
        """PauliSum should default to NoSymmetry."""
        ps = PauliSum(nqubits=3)
        assert ps.symmetry is not None
        assert isinstance(ps.symmetry, NoSymmetry)
        assert not ps.has_active_symmetry

    def test_pauli_sum_with_permutation_symmetry(self):
        """PauliSum should accept PermutationSymmetry."""
        sym = PermutationSymmetry()
        ps = PauliSum(nqubits=3, symmetry=sym)

        assert ps.symmetry is sym
        assert ps.has_active_symmetry

    def test_pauli_sum_copy_preserves_symmetry(self):
        """Copying PauliSum should preserve symmetry."""
        sym = PermutationSymmetry()
        ps1 = PauliSum(nqubits=3, symmetry=sym)
        ps1.add_term("XYZ", 1.0)

        ps2 = ps1.copy()
        assert ps2.symmetry is ps1.symmetry
        assert ps2.has_active_symmetry

    def test_manual_merging_with_symmetry(self):
        """Manually merging equivalent terms should work with symmetry."""
        sym = PermutationSymmetry()
        ps = PauliSum(nqubits=3, symmetry=sym)

        # Add three permutations of IXZ
        ps.add_term("IXZ", 1.0)
        ps.add_term("XIZ", 2.0)
        ps.add_term("ZIX", 3.0)

        # Before merging: 3 terms
        assert len(ps) == 3

        # Manually merge using symmetry
        from executor.pauli_propagation.propagation import \
            _apply_symmetry_merging

        _apply_symmetry_merging(ps)

        # After merging: 1 term with coefficient sum
        assert len(ps) == 1
        # Coefficient should be 1 + 2 + 3 = 6
        _, coeff = list(ps)[0]
        assert abs(coeff - 6.0) < 1e-12


class TestPropagationSymmetryIntegration:
    """Test symmetry integration with propagation functions."""

    def test_propagate_with_symmetry(self):
        """Propagate should apply symmetry merging if enabled."""
        # Simplified test: verify symmetry merging is callable
        # Full integration tests would require constructing realistic gate sequences
        from executor.pauli_propagation.propagation import propagate

        sym = PermutationSymmetry()
        observable = PauliSum(nqubits=2, symmetry=sym)
        observable.add_term("ZI", 1.0)
        observable.add_term("IZ", 1.0)

        # With no gates, propagate returns input (with merging applied)
        result = propagate([], observable, parameters={})

        # After merging, ZI and IZ (both weight-1 Z terms) should merge
        # into a single canonical term under permutation symmetry
        assert len(result) == 1

    def test_batch_propagate_with_symmetry(self):
        """Batch propagate should apply symmetry merging to all observables."""
        from executor.pauli_propagation.propagation import batch_propagate

        sym = PermutationSymmetry()

        obs1 = PauliSum(nqubits=2, symmetry=sym)
        obs1.add_term("ZI", 1.0)
        obs1.add_term("IZ", 1.0)

        obs2 = PauliSum(nqubits=2, symmetry=sym)
        obs2.add_term("XI", 1.0)
        obs2.add_term("IX", 1.0)

        observables = [obs1, obs2]

        # Empty gate list: just returns observables with merging applied
        results = batch_propagate([], observables, parameters={})

        assert len(results) == 2
        # Both observables should have been merged
        assert all(isinstance(r, PauliSum) for r in results)
        assert all(len(r) == 1 for r in results)  # Each should have 1 canonical term


class TestExecutorSymmetryIntegration:
    """Test symmetry integration with PauliPropagationExecutor."""

    def test_executor_default_no_symmetry(self):
        """Executor should default to NoSymmetry."""
        from executor.pauli_propagation.executor import \
            PauliPropagationExecutor

        executor = PauliPropagationExecutor()
        assert isinstance(executor.symmetry_strategy, NoSymmetry)

    def test_executor_with_permutation_symmetry(self):
        """Executor should accept PermutationSymmetry."""
        from executor.pauli_propagation.executor import \
            PauliPropagationExecutor

        sym = PermutationSymmetry()
        executor = PauliPropagationExecutor(symmetry_strategy=sym)

        assert executor.symmetry_strategy is sym

    def test_executor_expectation_with_symmetry(self):
        """Executor should apply symmetry during expectation value computation."""
        pytest.importorskip("qiskit")
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp

        from executor.pauli_propagation.executor import \
            PauliPropagationExecutor

        # Create simple circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        # Create symmetric observable (Z0 + Z1)
        operator = SparsePauliOp.from_list([("ZI", 1.0), ("IZ", 1.0)])

        # Execute with and without symmetry
        executor_no_sym = PauliPropagationExecutor()
        executor_with_sym = PauliPropagationExecutor(symmetry_strategy=PermutationSymmetry())

        result_no_sym = executor_no_sym.expectation_value(qc, operator)
        result_with_sym = executor_with_sym.expectation_value(qc, operator)

        # Results should be identical (symmetry doesn't change correctness)
        assert abs(result_no_sym - result_with_sym) < 1e-12


class TestSymmetryPerformance:
    """Performance benchmarks for symmetry strategies."""

    @pytest.mark.benchmark
    @_benchmark_skip
    def test_nosymmetry_overhead(self):
        """Measure overhead of NoSymmetry (should be negligible)."""
        sym = NoSymmetry()
        test_term = 0x123456
        nqubits = 12

        # Warm-up
        for _ in range(100):
            sym.canonical_representative(test_term, nqubits)

        # Benchmark
        start = time.perf_counter()
        for _ in range(10000):
            sym.canonical_representative(test_term, nqubits)
        elapsed = time.perf_counter() - start

        # Should be extremely fast (< 1 ms for 10k calls)
        assert elapsed < 0.001

    @pytest.mark.benchmark
    @_benchmark_skip
    def test_permutation_symmetry_scaling(self):
        """Measure PermutationSymmetry scaling with number of qubits."""
        nqubits_list = [4, 8, 16, 32, 64]
        times = []

        for nqubits in nqubits_list:
            sym = PermutationSymmetry()
            # Create test term with mixed Paulis
            test_term = sum((i % 4) << (2 * i) for i in range(nqubits))

            # Benchmark
            start = time.perf_counter()
            for _ in range(1000):
                sym.canonical_representative(test_term, nqubits)
            elapsed = time.perf_counter() - start

            times.append(elapsed)

        # Verify O(n) scaling: time should grow roughly linearly
        # Ratio of times should be ~ ratio of nqubits
        # For 64 vs 4 qubits (16x larger), time should be < 20x (allowing overhead)
        ratio = times[-1] / times[0]
        assert ratio < 20  # O(n) with reasonable constant

    @pytest.mark.benchmark
    def test_merging_reduces_terms(self):
        """Verify that symmetry merging reduces term count on realistic examples."""
        from executor.pauli_propagation.propagation import \
            _apply_symmetry_merging

        nqubits = 10
        sym = PermutationSymmetry()

        # Create PauliSum with many equivalent terms (permutations)
        ps = PauliSum(nqubits, symmetry=sym)

        # Add unique permutations of a multiset without enumerating all permutations
        # Base multiset: 5 I's, 3 X's, 1 Y, 1 Z
        import random

        random.seed(42)
        chars = list("IIIIIXXXYZ")
        unique_perms = set()
        while len(unique_perms) < 100:
            random.shuffle(chars)
            unique_perms.add("".join(chars))

        random_perms = list(unique_perms)
        for perm_str in random_perms:
            ps.add_term(perm_str, 1.0)

        # Before merging: PauliSum may already have fewer than 100 terms because
        # PauliSum.add_term merges duplicate strings immediately.
        initial_count = len(ps)
        assert 1 <= initial_count <= len(random_perms)

        # Apply merging
        _apply_symmetry_merging(ps)

        # After merging: should have only 1 term (all are equivalent under S_n)
        assert len(ps) == 1

        # Coefficient should be sum of all coefficients; this equals the
        # number of times we called add_term.
        _, coeff = list(ps)[0]
        expected_coeff = len(random_perms)
        assert abs(coeff - expected_coeff) < 1e-10
