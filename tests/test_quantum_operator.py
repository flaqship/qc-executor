"""Tests for `qc_executor.quantum_operator`."""

from __future__ import annotations

import functools

import numpy as np
import pytest
import sympy as sp

from qc_executor import QuantumOperator
from qc_executor.parameters import Parameters

#: Single-qubit Pauli matrices, for building dense references.
_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]).astype(complex),
}


def dense(operator: QuantumOperator) -> np.ndarray:
    """Build the dense matrix, qubit 0 as the leading tensor factor."""
    total = np.zeros((2**operator.num_qubits,) * 2, dtype=complex)
    for label, coeff in zip(operator.paulis, operator.coeffs):
        total = total + complex(coeff) * functools.reduce(
            np.kron, [_PAULI[char] for char in label]
        )
    return total


class TestQuantumOperatorConstruction:
    def test_from_quantum_operator_returns_same_instance(self):
        operator = QuantumOperator(["Z"], [1.0])

        assert QuantumOperator.from_quantum_operator(operator) is operator

    def test_init_with_paulis_and_coeffs_sets_properties(self):
        operator = QuantumOperator(["ZI", "IZ"], [0.5, -0.25])

        assert operator.num_qubits == 2
        assert operator.num_paulis == 2
        assert operator.paulis == ["ZI", "IZ"]
        assert np.allclose(np.asarray(operator.coeffs, dtype=complex), [0.5, -0.25])

    def test_init_with_num_qubits_creates_zero_identity(self):
        operator = QuantumOperator(num_qubits=3)

        assert operator.num_qubits == 3
        assert operator.num_paulis == 1
        assert operator.paulis == ["III"]
        assert np.allclose(np.asarray(operator.coeffs, dtype=complex), [0.0])

    def test_init_without_paulis_or_width_is_rejected(self):
        with pytest.raises(ValueError, match="Must provide paulis"):
            QuantumOperator()

    def test_default_coefficients_are_one(self):
        assert np.allclose(np.asarray(QuantumOperator(["Z", "X"]).coeffs, dtype=complex), [1, 1])

    def test_labels_of_differing_width_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            QuantumOperator(["Z", "ZZ"], [1.0, 1.0])

    def test_invalid_pauli_character_is_rejected(self):
        with pytest.raises(ValueError, match="invalid Pauli character"):
            QuantumOperator(["A"], [1.0])

    def test_coefficient_count_must_match(self):
        with pytest.raises(ValueError, match="1 label"):
            QuantumOperator(["Z"], [1.0, 2.0])


class TestQubitConvention:
    def test_qubit_zero_is_the_leftmost_character(self):
        """Pinned convention: label position q is qubit q."""
        operator = QuantumOperator(["ZI"], [1.0])

        # Z on qubit 0 is diag(1, -1) tensored with identity on qubit 1.
        expected = np.kron(_PAULI["Z"], _PAULI["I"])
        assert np.allclose(dense(operator), expected)

    def test_labels_round_trip(self):
        labels = ["XYZ", "IIZ", "ZII"]
        assert QuantumOperator(labels, [1.0] * 3).paulis == labels


class TestQuantumOperatorProperties:
    def test_numeric_operator_is_not_parameterized(self):
        operator = QuantumOperator(["Z"], [1.0])

        assert not operator.is_parametrized
        assert operator.num_parameters == 0
        assert not operator.parameters

    def test_symbolic_coefficients_are_reported(self):
        p = Parameters("p", 2)
        operator = QuantumOperator(["ZI", "IZ"], [p[0], 2 * p[1]])

        assert operator.is_parametrized
        assert operator.num_parameters == 2
        assert operator.parameters == [p[0], p[1]]

    def test_symbolic_coefficients_are_returned_as_expressions(self):
        p = Parameters("p", 1)
        operator = QuantumOperator(["Z"], [2 * p[0]])

        assert operator.coeffs[0] == 2 * p[0]

    def test_unbound_symbolic_coefficients_do_not_masquerade_as_numbers(self):
        """A NaN placeholder makes a misread loud instead of silently wrong."""
        p = Parameters("p", 1)
        operator = QuantumOperator(["Z"], [p[0]])

        assert isinstance(operator.coeffs[0], sp.Expr)
        assert operator.ir.numeric_coeffs is None

    def test_hermitian_and_real_flags(self):
        assert QuantumOperator(["Z"], [1.0]).is_hermitian
        assert QuantumOperator(["Z"], [1.0]).is_real
        assert not QuantumOperator(["Z"], [1j]).is_hermitian

    def test_imaginary_flag(self):
        assert QuantumOperator(["Z"], [2j]).is_imaginary
        assert not QuantumOperator(["Z"], [1.0]).is_imaginary


class TestQuantumOperatorAlgebra:
    def test_append_returns_a_new_operator(self):
        operator = QuantumOperator(["Z"], [1.0])

        extended = operator.append("X")

        assert operator.paulis == ["Z"], "append must not mutate its receiver"
        assert extended.paulis == ["Z", "X"]
        assert np.allclose(np.asarray(extended.coeffs, dtype=complex), [1.0, 1.0])

    def test_append_with_explicit_coefficient(self):
        extended = QuantumOperator(["Z"], [1.0]).append("X", coeff=0.2)

        assert np.allclose(np.asarray(extended.coeffs, dtype=complex), [1.0, 0.2])

    def test_append_rejects_a_mismatched_width(self):
        with pytest.raises(ValueError, match="cannot append"):
            QuantumOperator(["Z"], [1.0]).append("ZZ")

    def test_compose_is_pure_and_matches_the_dense_product(self):
        left = QuantumOperator(["ZI"], [1.0])
        right = QuantumOperator(["XY"], [0.5])

        composed = left.compose(right)

        assert left.paulis == ["ZI"], "compose must not mutate its receiver"
        assert np.allclose(dense(composed), dense(left) @ dense(right))

    def test_compose_rejects_non_operators(self):
        with pytest.raises(TypeError, match="can only compose with a quantum operator"):
            QuantumOperator(["Z"], [1.0]).compose("not-an-operator")

    def test_compose_rejects_mismatched_widths(self):
        with pytest.raises(ValueError, match="cannot compose"):
            QuantumOperator(["Z"], [1.0]).compose(QuantumOperator(["ZZ"], [1.0]))

    @pytest.mark.parametrize(
        "method, reference",
        [
            ("adjoint", lambda m: m.conj().T),
            ("transpose", lambda m: m.T),
            ("conjugate", lambda m: m.conj()),
        ],
    )
    def test_matrix_operations_match_a_dense_reference(self, method, reference):
        operator = QuantumOperator(["XY", "ZI"], [1 + 2j, -0.5])

        result = getattr(operator, method)()

        assert np.allclose(dense(result), reference(dense(operator)))

    def test_simplify_combines_equal_paulis(self):
        simplified = QuantumOperator(["Z", "Z"], [1.0, -0.5]).simplify()

        assert simplified.paulis == ["Z"]
        assert np.allclose(np.asarray(simplified.coeffs, dtype=complex), [0.5])

    def test_simplify_drops_cancelled_terms(self):
        simplified = QuantumOperator(["Z", "Z"], [1.0, -1.0]).simplify()

        assert simplified.paulis == ["I"]
        assert np.allclose(np.asarray(simplified.coeffs, dtype=complex), [0.0])

    def test_apply_layout_moves_qubits(self):
        relocated = QuantumOperator(["ZI"], [1.0]).apply_layout([1, 0])

        assert relocated.paulis == ["IZ"]

    def test_apply_layout_can_widen(self):
        relocated = QuantumOperator(["ZI"], [1.0]).apply_layout([0, 2], num_qubits=3)

        assert relocated.paulis == ["ZII"]

    def test_apply_layout_validates_length(self):
        with pytest.raises(ValueError, match="layout has 1 entries"):
            QuantumOperator(["ZI"], [1.0]).apply_layout([0])

    def test_apply_layout_validates_targets(self):
        with pytest.raises(ValueError, match="layout targets"):
            QuantumOperator(["ZI"], [1.0]).apply_layout([0, 5])

    def test_group_commuting_returns_commuting_groups(self):
        operator = QuantumOperator(["ZI", "IZ", "XX"], [1.0, 1.0, 0.5])

        groups = operator.group_commuting()

        assert all(isinstance(group, QuantumOperator) for group in groups)
        assert sum(group.num_paulis for group in groups) == 3

    def test_copy_is_independent(self):
        operator = QuantumOperator(["ZI", "IZ"], [1.0, 0.5])

        copied = operator.copy().append("ZZ", coeff=0.25)

        assert operator.num_paulis == 2
        assert copied.num_paulis == 3


class TestSymbolicOperators:
    def test_assign_parameters_binds_coefficients(self):
        p = Parameters("p", 2)
        operator = QuantumOperator(["ZI", "IZ"], [p[0], 2 * p[1]])

        bound = operator.assign_parameters({p[0]: 0.5, p[1]: 1.5})

        assert not bound.is_parametrized
        assert np.allclose(np.asarray(bound.coeffs, dtype=complex), [0.5, 3.0])

    def test_assign_parameters_accepts_names(self):
        p = Parameters("p", 1)
        bound = QuantumOperator(["Z"], [p[0]]).assign_parameters({"p[0]": 2.0})

        assert np.allclose(np.asarray(bound.coeffs, dtype=complex), [2.0])

    def test_partial_binding_keeps_the_rest_symbolic(self):
        p = Parameters("p", 2)
        operator = QuantumOperator(["ZI", "IZ"], [p[0], p[1]])

        bound = operator.assign_parameters({p[0]: 0.5})

        assert bound.parameters == [p[1]]

    def test_symbolic_operators_reject_composition(self):
        p = Parameters("p", 1)
        with pytest.raises(NotImplementedError, match="bind the parameters first"):
            QuantumOperator(["Z"], [p[0]]).compose(QuantumOperator(["X"], [1.0]))

    def test_symbolic_operators_reject_simplification(self):
        p = Parameters("p", 1)
        with pytest.raises(NotImplementedError, match="bind the parameters first"):
            QuantumOperator(["Z"], [p[0]]).simplify()


class TestQuantumOperatorIdentity:
    def test_is_unitary_for_a_single_pauli(self):
        assert QuantumOperator(["Z"], [1.0]).is_unitary

    def test_is_not_unitary_for_a_scaled_pauli(self):
        assert not QuantumOperator(["Z"], [2.0]).is_unitary

    def test_is_not_unitary_for_a_sum(self):
        assert not QuantumOperator(["Z", "X"], [1.0, 1.0]).is_unitary

    def test_hash_and_equality_follow_content(self):
        left = QuantumOperator(["ZI", "IZ"], [1.0, 0.5])
        right = QuantumOperator(["ZI", "IZ"], [1.0, 0.5])
        other = QuantumOperator(["ZI"], [1.0])

        assert left == right
        assert hash(left) == hash(right)
        assert left != other
        assert left != "not-an-operator"

    def test_coefficients_are_part_of_the_identity(self):
        assert QuantumOperator(["Z"], [1.0]) != QuantumOperator(["Z"], [2.0])

    def test_str_lists_the_terms(self):
        assert "Z" in str(QuantumOperator(["Z"], [1.0]))

    def test_repr_summarises(self):
        operator = QuantumOperator(["ZI", "IZ"], [1.0, 0.5])

        assert repr(operator) == "QuantumOperator(num_qubits=2, num_paulis=2)"

    def test_len_is_the_term_count(self):
        assert len(QuantumOperator(["ZI", "IZ"], [1.0, 0.5])) == 2


class TestQiskitBridge:
    def test_translation_preserves_the_physical_qubit(self):
        """Qiskit renders qubit 0 rightmost, so our 'ZI' becomes its 'IZ'."""
        native = QuantumOperator(["ZI"], [1.0]).qiskit_operator

        assert native.paulis.to_labels() == ["IZ"]

    def test_round_trip_through_qiskit(self):
        from qc_executor.qiskit._ir_bridge import sparse_pauli_op_to_pauli_ir

        operator = QuantumOperator(["XYZ", "ZII"], [1.5, -0.5])

        restored = sparse_pauli_op_to_pauli_ir(operator.qiskit_operator)

        assert restored.to_labels() == operator.paulis
        assert np.allclose(restored.coeffs_array, [1.5, -0.5])

    def test_symbolic_coefficients_survive_translation(self):
        p = Parameters("p", 1)
        native = QuantumOperator(["Z"], [2 * p[0]]).qiskit_operator

        assert sorted(param.name for param in native.parameters) == ["p[0]"]

    def test_generic_operator_has_no_native_representation(self):
        with pytest.raises(NotImplementedError, match="no native representation"):
            _ = QuantumOperator(["Z"], [1.0]).native
