"""Unit tests for the vectorized array engine.

End-to-end behavior is covered by the engine parity suite
(tests/pauli_propagation/test_engine_parity.py); these tests target the
module's building blocks and error paths directly.
"""

import numpy as np
import pytest

from qc_executor.pauli_propagation.utils import array_engine
from qc_executor.pauli_propagation.utils.gates import (
    CliffordGate,
    LayerBarrier,
    PauliRotation,
)
from qc_executor.pauli_propagation.utils.pauli_types import PauliSum


class TestPopcountDispatch:
    """popcount_u64 must give identical results on both implementations."""

    def test_fallback_matches_fast_path(self, monkeypatch):
        values = np.array([0, 1, 2**63, 2**64 - 1, 0x5555555555555555], dtype=np.uint64)

        fast = array_engine.popcount_u64(values)
        monkeypatch.setattr(array_engine, "_HAS_BITWISE_COUNT", False)
        fallback = array_engine.popcount_u64(values)

        np.testing.assert_array_equal(fast, fallback)
        np.testing.assert_array_equal(fallback, [0, 1, 1, 64, 32])


class TestApplyGate:
    """Dispatch and error paths of apply_gate."""

    def test_rotation_without_angle_raises(self):
        gate = PauliRotation(["X"], 0, nqubits=1)
        terms = np.array([0b11], dtype=np.uint64)
        coeffs = np.array([1.0 + 0.0j])

        with pytest.raises(ValueError, match="requires parameter value"):
            array_engine.apply_gate(terms, coeffs, gate, None)

    def test_unknown_gate_type_raises(self):
        terms = np.array([0b11], dtype=np.uint64)
        coeffs = np.array([1.0 + 0.0j])

        with pytest.raises(TypeError, match="Unknown gate type"):
            array_engine.apply_gate(terms, coeffs, LayerBarrier(), None)

    def test_clifford_prunes_tiny_coefficients(self):
        """apply_clifford replicates add_term's magnitude pruning."""
        gate = CliffordGate("H", 0, nqubits=1)
        terms = np.array([0b01, 0b11], dtype=np.uint64)  # X, Z
        coeffs = np.array([1e-20 + 0.0j, 1.0 + 0.0j])

        new_terms, new_coeffs = array_engine.apply_clifford(terms, coeffs, gate)

        # H maps X->Z (dropped, tiny) and Z->X (kept)
        assert new_terms.tolist() == [0b01]
        assert np.isclose(new_coeffs[0], 1.0)


class TestOverlapZeroArrays:
    """Vectorized zero-state overlap."""

    def test_only_iz_terms_contribute(self):
        nqubits = 2
        terms = np.array(
            [
                0b0000,  # II
                0b0011,  # Z on qubit 0
                0b0001,  # X on qubit 0 (no contribution)
                0b1011,  # Z0 Y1 (no contribution)
            ],
            dtype=np.uint64,
        )
        coeffs = np.array([0.25, 0.5, 3.0, 4.0], dtype=np.complex128)

        value = array_engine.overlap_zero_arrays(terms, coeffs, nqubits)

        assert np.isclose(value, 0.75)
        assert isinstance(value, complex)

    def test_round_trip_matches_dict_overlap(self):
        from qc_executor.pauli_propagation.utils.state_overlap import overlap_with_zero

        psum = PauliSum(3)
        psum.add_term("ZIZ", 0.7)
        psum.add_term("XYI", -0.3)
        psum.add_term("III", 0.1)

        terms, coeffs = array_engine.psum_to_arrays(psum)
        assert np.isclose(
            array_engine.overlap_zero_arrays(terms, coeffs, 3), overlap_with_zero(psum)
        )


class TestTruncateArrays:
    """Vectorized truncation criteria."""

    def test_keep_all_returns_inputs_unchanged(self):
        terms = np.array([0b11], dtype=np.uint64)
        coeffs = np.array([1.0 + 0.0j])

        new_terms, new_coeffs = array_engine.truncate_arrays(terms, coeffs, 1, 1e-10, None)

        assert new_terms is terms
        assert new_coeffs is coeffs

    def test_weight_and_coefficient_filtering(self):
        nqubits = 3
        terms = np.array([0b000011, 0b111111, 0b000001], dtype=np.uint64)  # Z, ZZZ, X
        coeffs = np.array([1.0, 1.0, 1e-12], dtype=np.complex128)

        new_terms, new_coeffs = array_engine.truncate_arrays(terms, coeffs, nqubits, 1e-6, 2)

        # ZZZ removed by weight, X removed by magnitude
        assert new_terms.tolist() == [0b000011]
        assert np.isclose(new_coeffs[0], 1.0)
