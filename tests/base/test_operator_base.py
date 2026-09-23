"""Tests for the concrete operator base shared by generic and native operators."""

from __future__ import annotations

import numpy as np
import pytest

from qc_executor.base.operator_base import QuantumOperatorBase
from qc_executor.base.operator_ir import PauliIR
from qc_executor.parameters import Parameters


class NativeOperator(QuantumOperatorBase):
    """A minimal backend operator: it only has to say how to compile itself."""

    def _build_native(self):
        return [(label, complex(coeff)) for label, coeff in zip(self.paulis, self.coeffs)]


class TestBackendContract:
    def test_a_backend_only_implements_build_native(self):
        operator = NativeOperator(["ZI", "IZ"], [1.0, -1.0])

        assert operator.native == [("ZI", 1 + 0j), ("IZ", -1 + 0j)]

    def test_the_native_form_is_cached(self):
        operator = NativeOperator(["Z"], [1.0])

        assert operator.native is operator.native

    def test_the_base_has_no_native_form(self):
        class Plain(QuantumOperatorBase):
            """An operator type that declares no native representation."""

        with pytest.raises(NotImplementedError, match="no native representation"):
            _ = Plain(["Z"], [1.0]).native

    def test_from_quantum_operator_converts_between_types(self):
        source = NativeOperator(["ZI"], [1.0])

        class Other(NativeOperator):
            """A second backend type, to exercise conversion."""

        converted = Other.from_quantum_operator(source)

        assert isinstance(converted, Other)
        assert converted.paulis == source.paulis

    def test_from_quantum_operator_passes_through_its_own_type(self):
        operator = NativeOperator(["Z"], [1.0])

        assert NativeOperator.from_quantum_operator(operator) is operator

    def test_derived_operations_keep_the_subclass_type(self):
        operator = NativeOperator(["Z"], [1.0])

        for derived in (operator.adjoint(), operator.copy(), operator.simplify()):
            assert isinstance(derived, NativeOperator)


class TestSharedProperties:
    def test_properties_read_from_the_representation(self):
        operator = NativeOperator(["ZI", "IZ"], [1.0, -1.0])

        assert operator.num_qubits == 2
        assert operator.paulis == ["ZI", "IZ"]
        assert np.allclose(np.asarray(operator.coeffs, dtype=complex), [1.0, -1.0])
        assert operator.num_paulis == 2
        assert len(operator) == 2

    def test_an_adopted_representation_is_used_directly(self):
        ir = PauliIR.from_labels(["XY"], [2.0])

        operator = NativeOperator(_ir=ir)

        assert operator.ir is ir
        assert operator.paulis == ["XY"]


class TestSymbolicCoefficients:
    def test_coeffs_array_returns_numeric_coefficients(self):
        operator = NativeOperator(["ZI", "IZ"], [1.0, -2.0])

        coeffs = operator.coeffs_array

        assert coeffs.dtype == complex
        assert np.allclose(coeffs, [1.0, -2.0])

    def test_coeffs_array_rejects_symbolic_coefficients(self):
        p = Parameters("p", 1)
        operator = NativeOperator(["ZI", "IZ"], [p[0], 1.0])

        with pytest.raises(ValueError, match="bind the parameters first"):
            _ = operator.coeffs_array

    def test_coeffs_array_is_available_after_binding(self):
        p = Parameters("p", 1)
        operator = NativeOperator(["Z"], [2 * p[0]])

        bound = operator.assign_parameters({p[0]: 0.5})

        assert np.allclose(bound.coeffs_array, [1.0])


class TestAlgebra:
    def test_add_concatenates_terms_without_simplifying(self):
        total = NativeOperator(["ZI"], [1.0]) + NativeOperator(["ZI", "IX"], [2.0, 3.0])

        assert isinstance(total, NativeOperator)
        assert total.paulis == ["ZI", "ZI", "IX"]
        assert np.allclose(total.coeffs_array, [1.0, 2.0, 3.0])

    def test_add_keeps_symbolic_coefficients_on_the_right_terms(self):
        p = Parameters("p", 2)
        left = NativeOperator(["Z", "X"], [p[0], 2.0])
        right = NativeOperator(["Y"], [p[1]])

        total = left + right

        assert total.coeffs[0] == p[0]
        assert complex(total.coeffs[1]) == 2.0
        assert total.coeffs[2] == p[1]

    def test_add_rejects_mismatched_widths(self):
        with pytest.raises(ValueError, match="cannot add a 1-qubit operator to a 2-qubit"):
            _ = NativeOperator(["ZI"], [1.0]) + NativeOperator(["Z"], [1.0])

    def test_add_rejects_non_operators(self):
        with pytest.raises(TypeError):
            _ = NativeOperator(["Z"], [1.0]) + 1.0

    def test_neg_negates_every_coefficient(self):
        negated = -NativeOperator(["Z", "X"], [1.0, -2.0])

        assert np.allclose(negated.coeffs_array, [-1.0, 2.0])

    def test_sub_appends_the_negated_terms(self):
        p = Parameters("p", 1)

        difference = NativeOperator(["Z"], [1.0]) - NativeOperator(["X"], [p[0]])

        assert difference.paulis == ["Z", "X"]
        assert complex(difference.coeffs[0]) == 1.0
        assert (difference.coeffs[1] + p[0]).simplify() == 0

    def test_sub_of_an_operator_from_itself_simplifies_to_zero(self):
        operator = NativeOperator(["ZI", "IX"], [1.0, 2.0])

        difference = (operator - operator).simplify()

        assert difference.paulis == ["II"]
        assert np.allclose(difference.coeffs_array, [0.0])

    def test_sub_rejects_non_operators(self):
        with pytest.raises(TypeError):
            _ = NativeOperator(["Z"], [1.0]) - 1.0

    @pytest.mark.parametrize("factor", [2, 2.0, np.float64(2.0), 2 + 0j])
    def test_mul_scales_by_a_number_from_either_side(self, factor):
        operator = NativeOperator(["Z", "X"], [1.0, -1.0])

        for scaled in (operator * factor, factor * operator):
            assert isinstance(scaled, NativeOperator)
            assert np.allclose(scaled.coeffs_array, [2.0, -2.0])

    def test_mul_scales_by_a_symbol_from_either_side(self):
        p = Parameters("p", 2)
        operator = NativeOperator(["Z", "X"], [p[0], 2.0])

        for scaled in (operator * p[1], p[1] * operator):
            assert scaled.coeffs == [p[0] * p[1], 2.0 * p[1]]
            assert scaled.parameters == [p[0], p[1]]

    def test_mul_of_two_operators_is_rejected(self):
        with pytest.raises(TypeError):
            _ = NativeOperator(["Z"], [1.0]) * NativeOperator(["X"], [1.0])

    def test_truediv_divides_by_a_number(self):
        halved = NativeOperator(["Z", "X"], [1.0, -4.0]) / 2

        assert isinstance(halved, NativeOperator)
        assert np.allclose(halved.coeffs_array, [0.5, -2.0])

    def test_truediv_divides_by_a_symbol(self):
        p = Parameters("p", 2)

        divided = NativeOperator(["Z", "X"], [p[0], 2.0]) / p[1]

        assert divided.coeffs == [p[0] / p[1], 2.0 / p[1]]

    def test_truediv_by_an_operator_is_rejected(self):
        with pytest.raises(TypeError):
            _ = NativeOperator(["Z"], [1.0]) / NativeOperator(["X"], [1.0])

    def test_matmul_is_compose(self):
        x = NativeOperator(["X"], [1.0])
        y = NativeOperator(["Y"], [1.0])

        product = x @ y

        assert isinstance(product, NativeOperator)
        assert product == x.compose(y)
        assert product.paulis == ["Z"]
        assert np.allclose(product.coeffs_array, [1j])

    def test_matmul_rejects_non_operators(self):
        with pytest.raises(TypeError):
            _ = NativeOperator(["X"], [1.0]) @ 2.0

    def test_tensor_places_the_other_operator_on_higher_qubits(self):
        product = NativeOperator(["ZI"], [1.0]).tensor(NativeOperator(["X"], [2.0]))

        assert isinstance(product, NativeOperator)
        assert product.paulis == ["ZIX"]
        assert np.allclose(product.coeffs_array, [2.0])

    def test_expand_places_the_other_operator_on_lower_qubits(self):
        product = NativeOperator(["ZI"], [1.0]).expand(NativeOperator(["X"], [2.0]))

        assert product.paulis == ["XZI"]

    def test_power_matches_repeated_composition(self):
        operator = NativeOperator(["X", "Z"], [1.0, 1.0])

        squared = operator.power(2).simplify()

        assert squared.paulis == ["I"]
        assert np.allclose(squared.coeffs_array, [2.0])

    def test_sum_merges_duplicates_across_operators(self):
        total = NativeOperator.sum(
            [
                NativeOperator(["Z"], [1.0]),
                NativeOperator(["Z"], [2.0]),
                NativeOperator(["X"], [1.0]),
            ]
        )

        assert isinstance(total, NativeOperator)
        assert dict(zip(total.paulis, total.coeffs_array)) == {"Z": 3.0, "X": 1.0}


class TestMatrixViews:
    def test_to_matrix_is_dense_by_default(self):
        matrix = NativeOperator(["Z"], [1.0]).to_matrix()

        assert isinstance(matrix, np.ndarray)
        assert np.allclose(matrix, np.diag([1.0, -1.0]))

    def test_to_matrix_can_be_sparse(self):
        sparse = pytest.importorskip("scipy.sparse")

        matrix = NativeOperator(["X"], [1.0]).to_matrix(sparse=True)

        assert sparse.issparse(matrix)
        assert np.allclose(matrix.toarray(), [[0.0, 1.0], [1.0, 0.0]])

    def test_diagonal_matches_the_dense_matrix(self):
        operator = NativeOperator(["ZI", "XI", "IZ"], [1.0, 2.0, 0.5])

        assert np.allclose(operator.diagonal(), np.diag(operator.to_matrix()))

    def test_canonical_key_ignores_term_order(self):
        first = NativeOperator(["Z", "X"], [1.0, 2.0])
        second = NativeOperator(["X", "Z"], [2.0, 1.0])

        assert first.canonical_key() == second.canonical_key()
        assert first.canonical_key() != NativeOperator(["Z", "X"], [1.0, 3.0]).canonical_key()


class TestConstructionHelpers:
    def test_identity_has_a_single_identity_term(self):
        identity = NativeOperator.identity(2)

        assert isinstance(identity, NativeOperator)
        assert identity.paulis == ["II"]
        assert np.allclose(identity.coeffs_array, [1.0])

    def test_identity_takes_a_symbolic_coefficient(self):
        p = Parameters("p", 1)

        identity = NativeOperator.identity(3, p[0])

        assert identity.paulis == ["III"]
        assert identity.coeffs == [p[0]]

    def test_from_symplectic_reads_rows_as_terms_and_columns_as_qubits(self):
        z = [[True, False], [False, False], [True, True]]
        x = [[False, False], [False, True], [True, False]]

        operator = NativeOperator.from_symplectic(z, x, [1.0, 2.0, 3.0])

        assert isinstance(operator, NativeOperator)
        assert operator.num_qubits == 2
        assert operator.paulis == ["ZI", "IX", "YZ"]

    def test_from_symplectic_accepts_integer_matrices_and_an_explicit_width(self):
        operator = NativeOperator.from_symplectic(np.zeros((1, 3)), np.zeros((1, 3)), [2.0], 3)

        assert operator.paulis == ["III"]
        assert np.allclose(operator.coeffs_array, [2.0])

    def test_from_symplectic_round_trips_through_the_representation(self):
        source = NativeOperator(["XYZ", "IZY"], [1.0, -0.5j])

        rebuilt = NativeOperator.from_symplectic(source.ir.z, source.ir.x, source.coeffs_array)

        assert rebuilt == source


class TestTermAccess:
    def test_getitem_with_an_integer_returns_one_term(self):
        operator = NativeOperator(["ZI", "IX", "YY"], [1.0, 2.0, 3.0])

        term = operator[1]

        assert isinstance(term, NativeOperator)
        assert term.paulis == ["IX"]
        assert np.allclose(term.coeffs_array, [2.0])

    def test_getitem_accepts_negative_indices(self):
        operator = NativeOperator(["ZI", "IX", "YY"], [1.0, 2.0, 3.0])

        assert operator[-1].paulis == ["YY"]

    def test_getitem_with_a_slice_returns_a_range_of_terms(self):
        operator = NativeOperator(["ZI", "IX", "YY"], [1.0, 2.0, 3.0])

        assert operator[1:].paulis == ["IX", "YY"]
        assert operator[::-1].paulis == ["YY", "IX", "ZI"]

    def test_getitem_remaps_symbolic_coefficients(self):
        p = Parameters("p", 2)
        operator = NativeOperator(["Z", "X", "Y"], [p[0], 2.0, p[1]])

        reversed_terms = operator[::-1]

        assert reversed_terms.coeffs[0] == p[1]
        assert complex(reversed_terms.coeffs[1]) == 2.0
        assert reversed_terms.coeffs[2] == p[0]
        assert operator[1].parameters == []

    def test_getitem_out_of_range_raises(self):
        with pytest.raises(IndexError):
            _ = NativeOperator(["Z"], [1.0])[1]

    def test_iteration_yields_single_term_operators(self):
        p = Parameters("p", 1)
        operator = NativeOperator(["ZI", "IX"], [p[0], 2.0])

        terms = list(operator)

        assert all(isinstance(term, NativeOperator) for term in terms)
        assert [term.paulis for term in terms] == [["ZI"], ["IX"]]
        assert terms[0].coeffs == [p[0]]
        assert np.allclose(terms[1].coeffs_array, [2.0])

    def test_iterating_the_zero_operator_yields_its_identity_term(self):
        terms = list(NativeOperator(num_qubits=2))

        assert [term.paulis for term in terms] == [["II"]]
        assert np.allclose(terms[0].coeffs_array, [0.0])

    def test_builtin_sum_reassembles_the_terms(self):
        operator = NativeOperator(["ZI", "IX"], [1.0, 2.0])

        total = sum(operator)

        assert isinstance(total, NativeOperator)
        assert total == operator

    def test_radd_with_zero_returns_a_copy(self):
        operator = NativeOperator(["Z"], [1.0])

        result = 0 + operator

        assert result == operator
        assert result is not operator

    def test_radd_with_a_nonzero_number_is_rejected(self):
        with pytest.raises(TypeError):
            _ = 1 + NativeOperator(["Z"], [1.0])


class TestUnitaryAndCommutation:
    @pytest.mark.parametrize(
        "paulis, coeffs",
        [
            (["X"], [1.0]),
            (["ZZ"], [-1j]),
            (["X", "Z"], [1 / np.sqrt(2), 1 / np.sqrt(2)]),
        ],
    )
    def test_unitary_operators_are_recognised(self, paulis, coeffs):
        assert NativeOperator(paulis, coeffs).is_unitary

    @pytest.mark.parametrize(
        "paulis, coeffs",
        [
            (["X"], [2.0]),  # identity product, wrong scale
            (["X", "Z"], [1.0, 1.0]),  # identity product, wrong scale
            (["X", "Y"], [1.0, 1j]),  # product has more than one term
        ],
    )
    def test_non_unitary_operators_are_rejected(self, paulis, coeffs):
        assert not NativeOperator(paulis, coeffs).is_unitary

    def test_is_unitary_rejects_symbolic_coefficients(self):
        p = Parameters("p", 1)

        with pytest.raises(NotImplementedError, match="bind the parameters first"):
            _ = NativeOperator(["X"], [p[0]]).is_unitary

    def test_commutes_with_returns_the_pairwise_matrix(self):
        operator = NativeOperator(["ZI", "XI", "IX"], [1.0, 1.0, 1.0])
        other = NativeOperator(["ZZ", "IZ"], [1.0, 1.0])

        matrix = operator.commutes_with(other)

        assert matrix.shape == (3, 2)
        assert matrix.tolist() == [[True, True], [False, True], [False, False]]

    def test_commutes_with_all_reduces_over_the_other_operator(self):
        operator = NativeOperator(["ZI", "XI", "IX"], [1.0, 1.0, 1.0])
        other = NativeOperator(["ZZ", "IZ"], [1.0, 1.0])

        assert operator.commutes_with_all(other).tolist() == [True, False, False]
