"""Tests for PauliPropagationObservable."""

import numpy as np

from executor.pauli_propagation import PauliPropagationObservable
from executor.pauli_propagation.symmetry import PermutationSymmetry


class TestPauliPropagationObservable:
    def test_create_from_paulis(self):
        observable = PauliPropagationObservable(["ZI", "IZ"], [0.5, 0.5])

        assert observable.num_qubits == 2
        assert observable.num_paulis == 2
        assert set(observable.paulis) == {"ZI", "IZ"}

    def test_append(self):
        observable = PauliPropagationObservable(num_qubits=2)
        updated = observable.append("ZZ", 1.0)

        assert updated.num_paulis == 1
        assert updated.coeffs[0] == 1.0

    def test_apply_layout(self):
        observable = PauliPropagationObservable(["ZI"], [1.0])
        remapped = observable.apply_layout({0: 1, 1: 0})

        assert remapped.paulis == ["IZ"]

    def test_properties(self):
        observable = PauliPropagationObservable(["Z"], [1.0 + 0.0j])

        assert observable.is_real
        assert not observable.is_imaginary
        assert observable.is_unitary

        complex_observable = PauliPropagationObservable(["Z"], [1.0j])
        assert complex_observable.is_imaginary
        assert not complex_observable.is_real

    def test_adjoint(self):
        observable = PauliPropagationObservable(["XX"], [1.0 + 2.0j])
        adjoint = observable.adjoint()

        assert np.isclose(adjoint.coeffs[0], 1.0 - 2.0j)

    def test_symmetry_strategy_is_retained(self):
        symmetry = PermutationSymmetry()
        observable = PauliPropagationObservable(
            ["ZI", "IZ"], [0.5, 0.5], symmetry_strategy=symmetry
        )

        assert observable.symmetry.name == "permutation"
        assert observable.has_active_symmetry

        mapped = observable.apply_layout({0: 1, 1: 0})
        assert mapped.symmetry.name == "permutation"
