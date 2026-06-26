"""Tests for propagation module."""

import numpy as np
import pytest

from qc_executor.pauli_propagation.utils.gates import (
    CliffordGate,
    Gate,
    LayerBarrier,
    PauliRotation,
)
from qc_executor.pauli_propagation.utils.pauli_types import PauliSum
from qc_executor.pauli_propagation.utils.propagation import (
    PropagationCache,
    batch_propagate,
    propagate,
    propagate_single_gate,
)


class TestPropagationCache:
    """Test PropagationCache."""

    def test_init(self):
        """Test initialization."""
        cache = PropagationCache(3)
        assert cache.nqubits == 3
        assert len(cache.mainsum) == 0
        assert len(cache.auxsum) == 0

    def test_clear(self):
        """Test clearing cache."""
        cache = PropagationCache(2)
        cache.mainsum.add_term("XY", 1.0)
        cache.auxsum.add_term("ZZ", 2.0)

        cache.clear()
        assert len(cache.mainsum) == 0
        assert len(cache.auxsum) == 0


class TestPropagatePauliRotation:
    """Test propagation through Pauli rotation gates."""

    def test_rz_commutes_with_z(self):
        """RZ(θ) doesn't change Z observable."""
        rz = PauliRotation(["Z"], 0, nqubits=1, param_name="theta")
        observable = PauliSum(1)
        observable.add_term("Z", 1.0)

        result = propagate_single_gate(rz, observable, param_value=np.pi / 4)

        # Z commutes with RZ, so it passes through unchanged
        assert len(result) == 1
        assert np.isclose(result.get_coeff("Z"), 1.0)

    def test_rx_on_z_observable(self):
        """RX(θ) rotates Z observable: Z → cos(θ)Z + sin(θ)Y."""
        rx = PauliRotation(["X"], 0, nqubits=1)
        observable = PauliSum(1)
        observable.add_term("Z", 1.0)

        theta = np.pi / 3
        result = propagate_single_gate(rx, observable, param_value=theta)

        # Should get cos(θ) Z + sin(θ) Y
        assert len(result) == 2
        assert np.isclose(result.get_coeff("Z"), np.cos(theta))
        assert np.isclose(result.get_coeff("Y"), np.sin(theta))

    def test_ry_on_x_observable(self):
        """RY(θ) rotates X observable: X → cos(θ)X - sin(θ)Z."""
        ry = PauliRotation(["Y"], 0, nqubits=1)
        observable = PauliSum(1)
        observable.add_term("X", 1.0)

        theta = np.pi / 6
        result = propagate_single_gate(ry, observable, param_value=theta)

        # Should get cos(θ) X - sin(θ) Z (check sign)
        assert len(result) == 2
        assert np.isclose(result.get_coeff("X"), np.cos(theta))

    def test_rx_pi_flips_z_to_minus_z(self):
        """RX(π) flips Z → -Z."""
        rx = PauliRotation(["X"], 0, nqubits=1)
        observable = PauliSum(1)
        observable.add_term("Z", 1.0)

        result = propagate_single_gate(rx, observable, param_value=np.pi)

        # cos(π) = -1, sin(π) ≈ 0
        assert len(result) >= 1
        assert np.isclose(result.get_coeff("Z"), -1.0, atol=1e-10)

    def test_rx_pi_over_2_on_z(self):
        """RX(π/2) rotates Z → Y."""
        rx = PauliRotation(["X"], 0, nqubits=1)
        observable = PauliSum(1)
        observable.add_term("Z", 1.0)

        result = propagate_single_gate(rx, observable, param_value=np.pi / 2)

        # cos(π/2) ≈ 0, sin(π/2) = 1
        # Should get Y (check coefficient sign)
        # Z component should vanish
        z_coeff = result.get_coeff("Z")
        assert np.isclose(z_coeff, 0.0, atol=1e-10)

    def test_parametric_gate_requires_parameter(self):
        """Parametric gates require parameter value."""
        rx = PauliRotation(["X"], 0, nqubits=1)
        observable = PauliSum(1)
        observable.add_term("Z", 1.0)

        with pytest.raises(ValueError, match="requires parameter"):
            propagate_single_gate(rx, observable, param_value=None)

    def test_unknown_gate_type_raises(self):
        """Unknown gate types are rejected."""

        class DummyGate(Gate):
            def commutes_with(self, pauli_term: int) -> bool:
                return True

            def is_parametric(self) -> bool:
                return False

        dummy = DummyGate(0, nqubits=1)
        observable = PauliSum(1)
        observable.add_term("Z", 1.0)

        with pytest.raises(TypeError, match="Unknown gate type"):
            propagate_single_gate(dummy, observable)


class TestPropagateClifford:
    """Test propagation through Clifford gates."""

    def test_hadamard_on_x(self):
        """H transforms X → Z."""
        h = CliffordGate("H", 0, nqubits=1)
        observable = PauliSum(1)
        observable.add_term("X", 2.0)

        result = propagate_single_gate(h, observable)

        assert len(result) == 1
        assert np.isclose(result.get_coeff("Z"), 2.0)

    def test_hadamard_on_z(self):
        """H transforms Z → X."""
        h = CliffordGate("H", 0, nqubits=1)
        observable = PauliSum(1)
        observable.add_term("Z", 3.0)

        result = propagate_single_gate(h, observable)

        assert len(result) == 1
        assert np.isclose(result.get_coeff("X"), 3.0)

    def test_hadamard_on_sum(self):
        """H transforms sum of Paulis."""
        h = CliffordGate("H", 0, nqubits=1)
        observable = PauliSum(1)
        observable.add_term("X", 1.0)
        observable.add_term("Z", 1.0)

        result = propagate_single_gate(h, observable)

        # X → Z, Z → X, so we get Z + X = X + Z
        assert len(result) == 2
        assert np.isclose(result.get_coeff("X"), 1.0)
        assert np.isclose(result.get_coeff("Z"), 1.0)

    def test_s_gate_on_x(self):
        """S transforms X → Y."""
        s = CliffordGate("S", 0, nqubits=1)
        observable = PauliSum(1)
        observable.add_term("X", 1.0)

        result = propagate_single_gate(s, observable)

        assert len(result) == 1
        assert np.isclose(result.get_coeff("Y"), 1.0)

    def test_swap_gate(self):
        """SWAP exchanges observables on two qubits."""
        swap = CliffordGate("SWAP", [0, 1], nqubits=2)
        observable = PauliSum(2)
        observable.add_term("XY", 1.0)

        result = propagate_single_gate(swap, observable)

        # XY → YX
        assert len(result) == 1
        assert np.isclose(result.get_coeff("YX"), 1.0)


class TestPropagate:
    """Test full propagation through gate sequences."""

    def test_empty_gate_list(self):
        """Empty gate list returns observable unchanged."""
        observable = PauliSum(2)
        observable.add_term("XY", 1.0)

        result = propagate([], observable)

        assert len(result) == 1
        assert np.isclose(result.get_coeff("XY"), 1.0)

    def test_single_gate_propagation(self):
        """Single gate propagation."""
        h = CliffordGate("H", 0, nqubits=2)
        observable = PauliSum(2)
        observable.add_term("XI", 1.0)

        result = propagate([h], observable)

        # H on qubit 0: XI → ZI
        assert len(result) == 1
        assert np.isclose(result.get_coeff("ZI"), 1.0)

    def test_two_gate_sequence(self):
        """Two gate sequence (Heisenberg: apply in reverse)."""
        h1 = CliffordGate("H", 0, nqubits=2)
        h2 = CliffordGate("H", 0, nqubits=2)
        observable = PauliSum(2)
        observable.add_term("XI", 1.0)

        result = propagate([h1, h2], observable)

        # H twice returns to original (H² = I)
        # In Heisenberg: H2† H1† X H1 H2 = H2† H1† X H1 H2
        # = H2† Z H2 = X
        assert len(result) == 1
        assert np.isclose(result.get_coeff("XI"), 1.0)

    def test_propagate_with_parameters(self):
        """Propagate with parametric gates."""
        rx = PauliRotation(["X"], 0, nqubits=1, param_name="angle")
        observable = PauliSum(1)
        observable.add_term("Z", 1.0)

        result = propagate([rx], observable, parameters={"angle": 0.0})

        # RX(0) doesn't change anything
        assert len(result) == 1
        assert np.isclose(result.get_coeff("Z"), 1.0)

    def test_mixed_gate_types(self):
        """Mix of parametric and non-parametric gates."""
        rx = PauliRotation(["X"], 0, nqubits=1, param_name="theta")
        h = CliffordGate("H", 0, nqubits=1)
        observable = PauliSum(1)
        observable.add_term("Z", 1.0)

        # Circuit: H, then RX
        # Heisenberg: apply RX first (to observable), then H
        # RX(0) on Z: Z → Z
        # H on Z: Z → X
        result = propagate([h, rx], observable, parameters={"theta": 0.0})

        assert len(result) == 1
        assert np.isclose(result.get_coeff("X"), 1.0)

    def test_propagate_with_barriers_uses_layer_splitting(self):
        """Barriers split layers and still preserve expected propagation."""
        h1 = CliffordGate("H", 0, nqubits=1)
        h2 = CliffordGate("H", 0, nqubits=1)
        observable = PauliSum(1)
        observable.add_term("X", 1.0)

        result = propagate([h1, LayerBarrier(), h2], observable)

        # H twice gives identity action, independent of barrier split.
        assert len(result) == 1
        assert np.isclose(result.get_coeff("X"), 1.0)

    def test_param_resolution_fallback_param_name_after_expr_error(self):
        """If symbolic eval fails, resolver falls back to param_name."""

        class DummySymbol:
            def __init__(self, name: str):
                self.name = name

            def __hash__(self):
                return hash(self.name)

            def __eq__(self, other):
                return isinstance(other, DummySymbol) and self.name == other.name

        class FailingExpr:
            free_symbols = {DummySymbol("bad")}

            def subs(self, _):
                raise ValueError("cannot evaluate")

        rx = PauliRotation(["X"], 0, nqubits=1, param_name="phi")
        rx.param_expr = FailingExpr()

        observable = PauliSum(1)
        observable.add_term("Z", 1.0)

        result = propagate([rx], observable, parameters={"bad": 1.0, "phi": np.pi / 2})

        # Fallback to phi=pi/2 yields Z -> Y.
        assert np.isclose(result.get_coeff("Z"), 0.0, atol=1e-10)
        assert np.isclose(result.get_coeff("Y"), 1.0, atol=1e-10)

    def test_param_resolution_fallback_generic_theta(self):
        """Generic 'theta' parameter is used when no named parameter is set."""
        rx = PauliRotation(["X"], 0, nqubits=1)
        observable = PauliSum(1)
        observable.add_term("Z", 1.0)

        result = propagate([rx], observable, parameters={"theta": np.pi / 2})

        assert np.isclose(result.get_coeff("Z"), 0.0, atol=1e-10)
        assert np.isclose(result.get_coeff("Y"), 1.0, atol=1e-10)


class TestBatchPropagate:
    """Tests for batch_propagate: must match N individual propagate() calls."""

    def test_empty_list_returns_empty(self):
        """batch_propagate with no observables returns empty list."""
        h = CliffordGate("H", 0, nqubits=1)
        result = batch_propagate([h], [])
        assert result == []

    def test_single_observable_matches_propagate(self):
        """batch_propagate with one observable is identical to propagate()."""
        h = CliffordGate("H", 0, nqubits=1)
        obs = PauliSum(1)
        obs.add_term("Z", 1.0)

        single = propagate([h], obs.copy())
        batch = batch_propagate([h], [obs.copy()])

        assert len(batch) == 1
        assert len(batch[0]) == len(single)
        for term, coeff in single:
            assert np.isclose(batch[0].get_coeff(term), coeff)

    def test_multiple_observables_match_individual_propagate(self):
        """batch_propagate results must match individual propagate() calls."""
        rx = PauliRotation(["X"], 0, nqubits=1, param_name="theta")
        h = CliffordGate("H", 0, nqubits=1)
        gates = [h, rx]
        params = {"theta": np.pi / 4}

        obs_z = PauliSum(1)
        obs_z.add_term("Z", 1.0)

        obs_x = PauliSum(1)
        obs_x.add_term("X", 1.0)

        obs_zx = PauliSum(1)
        obs_zx.add_term("Z", 0.5)
        obs_zx.add_term("X", 0.5)

        observables = [obs_z.copy(), obs_x.copy(), obs_zx.copy()]
        expected = [
            propagate(gates, obs_z.copy(), params),
            propagate(gates, obs_x.copy(), params),
            propagate(gates, obs_zx.copy(), params),
        ]

        batch_results = batch_propagate(gates, observables, params)

        assert len(batch_results) == 3
        for i, (expected_ps, batch_ps) in enumerate(zip(expected, batch_results)):
            assert len(batch_ps) == len(expected_ps), f"Term count mismatch at observable {i}"
            for term, coeff in expected_ps:
                assert np.isclose(
                    batch_ps.get_coeff(term), coeff
                ), f"Mismatch at observable {i}, term {term}"

    def test_does_not_mutate_input_observables(self):
        """batch_propagate must not modify the input PauliSums."""
        h = CliffordGate("H", 0, nqubits=1)
        obs = PauliSum(1)
        obs.add_term("Z", 1.0)
        original_coeff = obs.get_coeff("Z")

        batch_propagate([h], [obs])

        assert np.isclose(obs.get_coeff("Z"), original_coeff)

    def test_parametric_gates_correct_values(self):
        """Parametric gates in batch_propagate use correct parameter values."""
        rz = PauliRotation(["Z"], 0, nqubits=1, param_name="phi")
        obs_x = PauliSum(1)
        obs_x.add_term("X", 1.0)
        params = {"phi": np.pi / 3}

        single = propagate([rz], obs_x.copy(), params)
        batch = batch_propagate([rz], [obs_x.copy()], params)

        assert len(batch[0]) == len(single)
        for term, coeff in single:
            assert np.isclose(batch[0].get_coeff(term), coeff)

    def test_batch_propagate_with_truncation(self):
        """batch_propagate applies truncation branch when configured."""
        rx = PauliRotation(["X"], 0, nqubits=1, param_name="theta")

        obs = PauliSum(1)
        obs.add_term("Z", 1.0)

        result = batch_propagate(
            [rx],
            [obs],
            parameters={"theta": np.pi / 3},
            truncate_threshold=0.8,
        )[0]

        # cos(pi/3)=0.5 is truncated, sin(pi/3)=sqrt(3)/2 remains.
        assert len(result) == 1
        assert np.isclose(result.get_coeff("Z"), 0.0)
        assert np.isclose(result.get_coeff("Y"), np.sin(np.pi / 3))
