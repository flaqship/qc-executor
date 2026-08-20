"""Tests for the sparse Pauli representation."""

from __future__ import annotations

import functools
import itertools

import numpy as np
import pytest
import sympy as sp

from qc_executor.base.operator_ir import PauliIR
from qc_executor.parameters import Parameter, Parameters

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]).astype(complex),
}


def dense(operator: PauliIR) -> np.ndarray:
    """Build the dense matrix, qubit 0 as the leading tensor factor."""
    total = np.zeros((2**operator.num_qubits,) * 2, dtype=complex)
    for label, coeff in zip(operator.to_labels(), operator.coeffs):
        total = total + complex(coeff) * functools.reduce(
            np.kron, [_PAULI[char] for char in label]
        )
    return total


def labels(num_qubits: int):
    """All Pauli labels of the given width."""
    return ["".join(t) for t in itertools.product("IXYZ", repeat=num_qubits)]


class TestConstruction:
    def test_labels_round_trip(self):
        for label in labels(2):
            assert PauliIR.from_labels([label]).to_labels() == [label]

    def test_lowercase_labels_are_accepted(self):
        assert PauliIR.from_labels(["zi"]).to_labels() == ["ZI"]

    def test_default_coefficients_are_one(self):
        assert np.allclose(PauliIR.from_labels(["Z", "X"]).coeffs_array, [1, 1])

    def test_zero_operator(self):
        operator = PauliIR.zero(3)

        assert operator.to_labels() == ["III"]
        assert np.allclose(operator.coeffs_array, [0])

    def test_empty_labels_need_a_width(self):
        with pytest.raises(ValueError, match="num_qubits is required"):
            PauliIR.from_labels([])

    def test_empty_labels_with_a_width_give_zero(self):
        assert PauliIR.from_labels([], num_qubits=2).to_labels() == ["II"]

    def test_mismatched_widths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            PauliIR.from_labels(["Z", "ZZ"])

    def test_declared_width_must_match(self):
        with pytest.raises(ValueError, match="but num_qubits is"):
            PauliIR.from_labels(["ZZ"], num_qubits=3)

    def test_invalid_character(self):
        with pytest.raises(ValueError, match="invalid Pauli character"):
            PauliIR.from_labels(["Q"])

    def test_coefficient_count_must_match(self):
        with pytest.raises(ValueError, match="1 label"):
            PauliIR.from_labels(["Z"], [1.0, 2.0])

    def test_inconsistent_arrays_are_rejected(self):
        with pytest.raises(ValueError, match="must agree in length"):
            PauliIR(1, np.zeros((2, 1), bool), np.zeros((1, 1), bool), np.ones(1, complex))

    def test_foreign_coefficient_types_report_clearly(self):
        with pytest.raises(TypeError, match="must be numbers or SymPy expressions"):
            PauliIR.from_labels(["Z"], [object()])


class TestQubitConvention:
    def test_column_q_is_qubit_q(self):
        operator = PauliIR.from_labels(["ZI"])

        assert operator.z[0, 0] and not operator.z[0, 1]

    def test_dense_matrix_places_qubit_zero_first(self):
        assert np.allclose(dense(PauliIR.from_labels(["ZI"])), np.kron(_PAULI["Z"], _PAULI["I"]))


class TestAlgebra:
    @pytest.mark.parametrize("num_qubits", [1, 2])
    def test_compose_matches_the_dense_product(self, num_qubits):
        """Pauli product phases are easy to get wrong, so check every pair."""
        for left_label, right_label in itertools.product(labels(num_qubits), repeat=2):
            left = PauliIR.from_labels([left_label])
            right = PauliIR.from_labels([right_label])

            assert np.allclose(
                dense(left.compose(right)), dense(left) @ dense(right)
            ), f"{left_label} * {right_label}"

    def test_compose_handles_multiple_terms(self):
        left = PauliIR.from_labels(["XI", "ZZ"], [0.5, -1.5])
        right = PauliIR.from_labels(["YY", "IZ"], [2.0, 1j])

        assert np.allclose(dense(left.compose(right)), dense(left) @ dense(right))

    def test_compose_rejects_mismatched_widths(self):
        with pytest.raises(ValueError, match="cannot compose"):
            PauliIR.from_labels(["Z"]).compose(PauliIR.from_labels(["ZZ"]))

    @pytest.mark.parametrize(
        "method, reference",
        [
            ("adjoint", lambda m: m.conj().T),
            ("transpose", lambda m: m.T),
            ("conjugate", lambda m: m.conj()),
        ],
    )
    def test_matrix_operations_match_a_dense_reference(self, method, reference):
        for label in labels(2):
            operator = PauliIR.from_labels([label], [1 + 2j])

            assert np.allclose(
                dense(getattr(operator, method)()), reference(dense(operator))
            ), f"{method} on {label}"

    def test_simplify_combines_and_drops_terms(self):
        simplified = PauliIR.from_labels(["Z", "Z", "X"], [1.0, -0.5, 0.0]).simplify()

        assert simplified.to_labels() == ["Z"]
        assert np.allclose(simplified.coeffs_array, [0.5])

    def test_simplify_of_a_fully_cancelling_operator(self):
        simplified = PauliIR.from_labels(["Z", "Z"], [1.0, -1.0]).simplify()

        assert simplified.to_labels() == ["I"]
        assert np.allclose(simplified.coeffs_array, [0.0])

    def test_apply_layout_permutes(self):
        assert PauliIR.from_labels(["ZI"]).apply_layout([1, 0]).to_labels() == ["IZ"]

    def test_apply_layout_widens(self):
        relocated = PauliIR.from_labels(["ZX"]).apply_layout([0, 2], num_qubits=3)

        assert relocated.to_labels() == ["ZIX"]

    def test_apply_layout_validates_length(self):
        with pytest.raises(ValueError, match="layout has 1 entries"):
            PauliIR.from_labels(["ZI"]).apply_layout([0])

    def test_apply_layout_validates_range(self):
        with pytest.raises(ValueError, match="layout targets"):
            PauliIR.from_labels(["ZI"]).apply_layout([0, 9])

    def test_group_commuting_groups_only_commuting_terms(self):
        operator = PauliIR.from_labels(["ZI", "IZ", "XX", "XI"])

        groups = operator.group_commuting()

        for group in groups:
            for first, second in itertools.combinations(group.to_labels(), 2):
                left = PauliIR.from_labels([first])
                right = PauliIR.from_labels([second])
                assert np.allclose(
                    dense(left) @ dense(right), dense(right) @ dense(left)
                ), f"{first} and {second} do not commute"

    def test_group_commuting_preserves_every_term(self):
        operator = PauliIR.from_labels(["ZI", "IZ", "XX"], [1.0, 2.0, 3.0])

        groups = operator.group_commuting()

        assert sorted(label for group in groups for label in group.to_labels()) == sorted(
            operator.to_labels()
        )

    def test_group_commuting_carries_symbolic_coefficients(self):
        p = Parameters("p", 1)
        operator = PauliIR.from_labels(["ZI", "XX"], [p[0], 1.0])

        groups = operator.group_commuting()

        assert any(group.symbolic for group in groups)


class TestSymbolicCoefficients:
    def test_symbolic_coefficients_are_stored_in_the_overlay(self):
        p = Parameters("p", 1)
        operator = PauliIR.from_labels(["Z"], [2 * p[0]])

        assert operator.symbolic == {0: 2 * p[0]}
        assert np.isnan(operator.coeffs_array[0].real)

    def test_numeric_view_is_unavailable_while_symbolic(self):
        p = Parameters("p", 1)

        assert PauliIR.from_labels(["Z"], [p[0]]).numeric_coeffs is None

    def test_numeric_view_is_available_otherwise(self):
        assert PauliIR.from_labels(["Z"], [1.0]).numeric_coeffs is not None

    def test_free_parameters(self):
        p = Parameters("p", 2)
        operator = PauliIR.from_labels(["ZI", "IZ"], [p[0], p[1] * 2])

        assert operator.free_parameters == frozenset({p[0], p[1]})

    def test_substitute_binds_into_the_numeric_column(self):
        p = Parameters("p", 1)
        bound = PauliIR.from_labels(["Z"], [2 * p[0]]).substitute({p[0]: 0.5})

        assert np.allclose(bound.coeffs_array, [1.0])
        assert not bound.symbolic

    def test_substitute_a_bare_symbol(self):
        p = Parameters("p", 1)
        bound = PauliIR.from_labels(["Z"], [p[0]]).substitute({p[0]: 0.75})

        assert np.allclose(bound.coeffs_array, [0.75])

    def test_partial_substitution_keeps_the_rest_symbolic(self):
        p = Parameters("p", 2)
        bound = PauliIR.from_labels(["Z"], [p[0] + p[1]]).substitute({p[0]: 1.0})

        assert bound.free_parameters == frozenset({p[1]})

    def test_foreign_symbols_are_canonicalized(self):
        operator = PauliIR.from_labels(["Z"], [sp.Symbol("p[0]") * 2])

        assert operator.free_parameters == frozenset({Parameter("p", 0)})

    def test_symbolic_operators_reject_composition(self):
        p = Parameters("p", 1)
        with pytest.raises(NotImplementedError, match="bind the parameters first"):
            PauliIR.from_labels(["Z"], [p[0]]).compose(PauliIR.from_labels(["X"]))

    def test_symbolic_operators_reject_simplification(self):
        p = Parameters("p", 1)
        with pytest.raises(NotImplementedError, match="bind the parameters first"):
            PauliIR.from_labels(["Z"], [p[0]]).simplify()

    def test_adjoint_conjugates_symbolic_coefficients(self):
        p = Parameters("p", 1)
        adjoint = PauliIR.from_labels(["Z"], [p[0]]).adjoint()

        # Parameters are real, so conjugation is the identity on them.
        assert adjoint.symbolic[0] == p[0]

    def test_symbolic_coefficients_are_assumed_hermitian(self):
        p = Parameters("p", 1)

        assert PauliIR.from_labels(["Z"], [p[0]]).is_hermitian


class TestFlags:
    def test_hermitian_for_real_coefficients(self):
        assert PauliIR.from_labels(["Z"], [1.0]).is_hermitian

    def test_not_hermitian_for_complex_coefficients(self):
        assert not PauliIR.from_labels(["Z"], [1j]).is_hermitian


class TestIdentity:
    def test_identical_operators_agree(self):
        left = PauliIR.from_labels(["ZI"], [1.0])
        right = PauliIR.from_labels(["ZI"], [1.0])

        assert left == right
        assert hash(left) == hash(right)

    @pytest.mark.parametrize(
        "other",
        [
            PauliIR.from_labels(["IZ"], [1.0]),
            PauliIR.from_labels(["ZI"], [2.0]),
            PauliIR.from_labels(["ZII"], [1.0]),
        ],
        ids=["different_label", "different_coeff", "different_width"],
    )
    def test_differing_operators_disagree(self, other):
        assert PauliIR.from_labels(["ZI"], [1.0]) != other

    def test_symbolic_coefficients_distinguish_operators(self):
        p = Parameters("p", 2)

        assert PauliIR.from_labels(["Z"], [p[0]]) != PauliIR.from_labels(["Z"], [p[1]])

    def test_not_equal_to_other_types(self):
        assert PauliIR.from_labels(["Z"]) != "not-an-operator"

    def test_len_is_the_term_count(self):
        assert len(PauliIR.from_labels(["ZI", "IZ"])) == 2

    def test_repr_summarises(self):
        assert repr(PauliIR.from_labels(["ZI", "IZ"])) == "PauliIR(num_qubits=2, num_terms=2)"
