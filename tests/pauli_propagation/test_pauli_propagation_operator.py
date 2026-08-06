"""Tests for PauliPropagationOperator."""

import numpy as np
import pytest
import sympy as sp

from qc_executor.parameters import Parameter
from qc_executor.pauli_propagation import PauliPropagationOperator
from qc_executor.pauli_propagation.symmetry import (
    CompositeSymmetry,
    NoSymmetry,
    PermutationSymmetry,
    SymmetryStrategy,
)
from qc_executor.pauli_propagation.utils.pauli_types import PauliSum
from qc_executor.quantum_operator import QuantumOperator


class DummySymmetry(SymmetryStrategy):
    def canonical_representative(self, term: int, nqubits: int) -> int:
        return term

    @property
    def name(self) -> str:
        return "dummy"


class TestPauliPropagationOperatorConstruction:
    def test_create_from_paulis(self):
        observable = PauliPropagationOperator(["ZI", "IZ"], [0.5, 0.5])

        assert observable.num_qubits == 2
        assert observable.num_paulis == 2
        assert set(observable.paulis) == {"ZI", "IZ"}

    def test_init_from_pauli_sum_with_symmetry_override(self):
        pauli_sum = PauliSum(1)
        pauli_sum.add_term("Z", 1.0)

        observable = PauliPropagationOperator(
            pauli_sum=pauli_sum,
            symmetry_strategy=PermutationSymmetry(),
        )

        assert observable.num_qubits == 1
        assert observable.symmetry.name == "permutation"

    def test_init_empty_paulis_requires_num_qubits(self):
        with pytest.raises(ValueError, match="num_qubits is required"):
            PauliPropagationOperator(paulis=[])

    def test_init_empty_paulis_with_num_qubits(self):
        observable = PauliPropagationOperator(paulis=[], num_qubits=3)

        assert observable.num_qubits == 3
        assert observable.num_paulis == 0

    def test_init_rejects_mismatched_coeff_lengths(self):
        with pytest.raises(ValueError, match="Length of coeffs"):
            PauliPropagationOperator(paulis=["Z", "X"], coeffs=[1.0])

    def test_init_requires_any_input(self):
        with pytest.raises(ValueError, match="Provide either paulis"):
            PauliPropagationOperator()

    def test_pauli_sum_property_returns_copy(self):
        observable = PauliPropagationOperator(["Z"], [1.0])

        pauli_sum_copy = observable.pauli_sum
        pauli_sum_copy.add_term("X", 2.0)

        assert observable.num_paulis == 1


class TestPauliPropagationOperatorConversion:
    def test_from_quantum_operator_native_copy(self):
        source = PauliPropagationOperator(["Z"], [1.0])

        converted = PauliPropagationOperator.from_quantum_operator(source)

        assert converted is not source
        assert converted == source

    def test_from_quantum_operator_native_copy_with_symmetry_override(self):
        source = PauliPropagationOperator(["Z"], [1.0])

        converted = PauliPropagationOperator.from_quantum_operator(
            source,
            symmetry_strategy=PermutationSymmetry(),
        )

        assert converted is not source
        assert converted.symmetry.name == "permutation"

    def test_from_quantum_operator_with_generic_operator(self):
        op = QuantumOperator(["Z"], [Parameter("theta", 0)])

        converted = PauliPropagationOperator.from_quantum_operator(op)

        assert isinstance(converted, PauliPropagationOperator)
        assert converted.is_parametrized
        assert converted.parameters == ["theta[0]"]
        assert converted.paulis == ["Z"]
        assert converted.num_parameters == 1

    def test_from_quantum_operator_rejects_invalid_input(self):
        with pytest.raises(TypeError, match="expects a generic QuantumOperator"):
            PauliPropagationOperator.from_quantum_operator(object())


class TestPauliPropagationOperatorParameters:
    def test_parameter_tracking(self):
        theta = sp.Symbol("theta")
        observable = PauliPropagationOperator(["Z"], [theta])

        assert observable.is_parametrized
        assert observable.parameters == ["theta"]
        assert observable.num_parameters == 1

    def test_assign_parameters_full_binding(self):
        theta = sp.Symbol("theta")
        observable = PauliPropagationOperator(["Z"], [theta])

        assigned = observable.assign_parameters({"theta": 0.75})

        assert not assigned.is_parametrized
        assert assigned.num_parameters == 0
        assert np.isclose(assigned.coeffs[0], 0.75)

    def test_assign_parameters_partial_binding_and_nonparam_copy(self):
        theta = sp.Symbol("theta")
        phi = sp.Symbol("phi")
        observable = PauliPropagationOperator(["Z", "X"], [theta + phi, 2.0])

        assigned = observable.assign_parameters({"theta": 0.25, "unused": 1.0})

        assert assigned.is_parametrized
        assert assigned.parameters == ["phi"]
        assert assigned.num_parameters == 1
        assert np.isclose(assigned.coeffs[1], 2.0)
        assert np.isclose(assigned.coeffs[0], 1.0)


class TestPauliPropagationOperatorAlgebra:
    def test_append(self):
        observable = PauliPropagationOperator(num_qubits=2)
        updated = observable.append("ZZ", 1.0)

        assert updated.num_paulis == 1
        assert updated.coeffs[0] == 1.0

    def test_append_default_coeff(self):
        observable = PauliPropagationOperator(num_qubits=1)
        updated = observable.append("Z")

        assert updated.coeffs[0] == 1.0

    def test_compose_rejects_invalid_other_type(self):
        observable = PauliPropagationOperator(["Z"], [1.0])

        with pytest.raises(TypeError, match="PauliPropagationOperator only"):
            observable.compose("invalid")

    def test_compose_rejects_different_qubit_counts(self):
        left = PauliPropagationOperator(["Z"], [1.0])
        right = PauliPropagationOperator(["ZI"], [1.0])

        with pytest.raises(ValueError, match="different qubit counts"):
            left.compose(right)

    def test_compose_success(self):
        left = PauliPropagationOperator(["Z"], [2.0])
        right = PauliPropagationOperator(["Z"], [0.5])

        composed = left.compose(right)

        assert composed.paulis == ["I"]
        assert np.isclose(composed.coeffs[0], 1.0)

    def test_simplify_returns_copy(self):
        observable = PauliPropagationOperator(["Z"], [1.0])

        simplified = observable.simplify()

        assert simplified == observable
        assert simplified is not observable

    def test_transpose_flips_sign_for_odd_number_of_y(self):
        observable = PauliPropagationOperator(["Y"], [2.0])

        transposed = observable.transpose()

        assert transposed.paulis == ["Y"]
        assert np.isclose(transposed.coeffs[0], -2.0)

    def test_conjugate(self):
        observable = PauliPropagationOperator(["XX"], [1.0 + 2.0j])

        conjugated = observable.conjugate()

        assert np.isclose(conjugated.coeffs[0], 1.0 - 2.0j)

    def test_group_commuting_returns_single_copy(self):
        observable = PauliPropagationOperator(["Z"], [1.0])

        groups = observable.group_commuting()

        assert len(groups) == 1
        assert groups[0] == observable
        assert groups[0] is not observable

    def test_apply_layout(self):
        observable = PauliPropagationOperator(["ZI"], [1.0])
        remapped = observable.apply_layout({0: 1, 1: 0})

        assert remapped.paulis == ["IZ"]

    def test_apply_layout_keeps_parametric_mapping(self):
        theta = sp.Symbol("theta")
        observable = PauliPropagationOperator(["ZI"], [theta])

        remapped = observable.apply_layout({0: 1, 1: 0})

        assert remapped.paulis == ["IZ"]
        assert remapped.is_parametrized
        assert remapped.parameters == ["theta"]

    def test_adjoint(self):
        observable = PauliPropagationOperator(["XX"], [1.0 + 2.0j])
        adjoint = observable.adjoint()

        assert np.isclose(adjoint.coeffs[0], 1.0 - 2.0j)


class TestPauliPropagationOperatorSymmetry:
    def test_compose_symmetry_both_active_same(self):
        left = PauliPropagationOperator(["Z"], [1.0], symmetry_strategy=PermutationSymmetry())
        right = PauliPropagationOperator(["X"], [1.0], symmetry_strategy=PermutationSymmetry())

        composed = left.compose(right)

        assert composed.symmetry.name == "permutation"

    def test_compose_symmetry_both_active_different(self):
        left = PauliPropagationOperator(["Z"], [1.0], symmetry_strategy=PermutationSymmetry())
        right = PauliPropagationOperator(["X"], [1.0], symmetry_strategy=DummySymmetry())

        composed = left.compose(right)

        assert isinstance(composed.symmetry, CompositeSymmetry)
        assert composed.symmetry.name == "composite(permutation + dummy)"

    def test_compose_symmetry_self_active_only(self):
        left = PauliPropagationOperator(["Z"], [1.0], symmetry_strategy=PermutationSymmetry())
        right = PauliPropagationOperator(["X"], [1.0])

        composed = left.compose(right)

        assert composed.symmetry.name == "permutation"

    def test_compose_symmetry_other_active_only(self):
        left = PauliPropagationOperator(["Z"], [1.0])
        right = PauliPropagationOperator(["X"], [1.0], symmetry_strategy=PermutationSymmetry())

        composed = left.compose(right)

        assert composed.symmetry.name == "permutation"

    def test_compose_symmetry_none_active(self):
        left = PauliPropagationOperator(["Z"], [1.0])
        right = PauliPropagationOperator(["X"], [1.0])

        composed = left.compose(right)

        assert isinstance(composed.symmetry, NoSymmetry)

    def test_symmetry_strategy_is_retained(self):
        symmetry = PermutationSymmetry()
        observable = PauliPropagationOperator(["ZI", "IZ"], [0.5, 0.5], symmetry_strategy=symmetry)

        assert observable.symmetry.name == "permutation"
        assert observable.has_active_symmetry

        mapped = observable.apply_layout({0: 1, 1: 0})
        assert mapped.symmetry.name == "permutation"


class TestPauliPropagationOperatorProperties:
    def test_properties(self):
        observable = PauliPropagationOperator(["Z"], [1.0 + 0.0j])

        assert observable.is_real
        assert not observable.is_imaginary
        assert observable.is_unitary

        complex_observable = PauliPropagationOperator(["Z"], [1.0j])
        assert complex_observable.is_imaginary
        assert not complex_observable.is_real

    def test_is_unitary_false_for_multiple_terms(self):
        observable = PauliPropagationOperator(["Z", "X"], [1.0, 0.5])

        assert not observable.is_unitary

    def test_canonical_signature_includes_parametric_terms(self):
        theta = sp.Symbol("theta")
        parametric = PauliPropagationOperator(["Z"], [theta])
        numeric = PauliPropagationOperator(["Z"], [1.0])

        param_sig = parametric._canonical_signature()
        num_sig = numeric._canonical_signature()

        assert len(param_sig) == 4
        assert len(num_sig) == 4
        assert param_sig[-1]
        assert not num_sig[-1]

    def test_hash_eq_str_and_repr(self):
        left = PauliPropagationOperator(["Z"], [1.0])
        right = PauliPropagationOperator(["Z"], [1.0])
        other = PauliPropagationOperator(["X"], [1.0])

        assert hash(left) == hash(right)
        assert left == right
        assert left != other
        assert left != object()
        assert "PauliPropagationOperator" in str(left)
        assert repr(left) == str(left)
