"""Tests for `qc_executor.quantum_operator`."""

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qc_executor import QuantumOperator


class TestQuantumOperatorConstruction:
    def test_from_quantum_operator_returns_same_instance(self):
        operator = QuantumOperator(["Z"], [1.0])

        assert QuantumOperator.from_quantum_operator(operator) is operator

    def test_init_with_paulis_and_coeffs_sets_properties(self):
        operator = QuantumOperator(["ZI", "IZ"], [0.5, -0.25])

        assert operator.num_qubits == 2
        assert operator.num_paulis == 2
        assert operator.paulis == ["ZI", "IZ"]
        assert np.allclose(operator.coeffs, [0.5, -0.25])

    def test_init_with_num_qubits_creates_zero_identity(self):
        operator = QuantumOperator(num_qubits=3)

        assert operator.num_qubits == 3
        assert operator.num_paulis == 1
        assert operator.paulis == ["III"]
        assert np.allclose(operator.coeffs, [0.0])


class TestQuantumOperatorProperties:
    def test_is_parameterized(self):
        operator = QuantumOperator(["Z"], [1.0])

        assert not operator.is_parametrized

    def test_num_parameters(self):
        # QuantumOperator is built on SparsePauliOp, which has no parameters,
        # so this is always 0.
        operator = QuantumOperator(["Z"], [1.0])

        assert operator.num_parameters == 0

    def test_parameters(self):
        # QuantumOperator is built on SparsePauliOp, which has no parameters,
        # so this is always an empty list.
        operator = QuantumOperator(["Z"], [1.0])

        assert not operator.parameters


class TestQuantumOperatorMutations:
    def test_append_with_default_coefficient(self):
        operator = QuantumOperator(["Z"], [1.0])

        operator.append("X")

        assert operator.paulis == ["Z", "X"]
        assert np.allclose(operator.coeffs, [1.0, 1.0])

    def test_append_with_explicit_coefficient(self):
        operator = QuantumOperator(["Z"], [1.0])

        operator.append("X", coeff=0.2)

        assert operator.paulis == ["Z", "X"]
        assert np.allclose(operator.coeffs, [1.0, 0.2])

    def test_compose_updates_operator_in_place(self):
        left = QuantumOperator(["Z"], [1.0])
        right = QuantumOperator(["X"], [1.0])

        left.compose(right)

        expected = SparsePauliOp(["Z"], coeffs=[1.0]).compose(SparsePauliOp(["X"], coeffs=[1.0]))
        assert left._qiskit_operator == expected


class TestQuantumOperatorDerivedOperators:
    def test_copy_returns_independent_object(self):
        operator = QuantumOperator(["ZI", "IZ"], [1.0, 0.5])

        copied = operator.copy()
        copied.append("ZZ", coeff=0.25)

        assert isinstance(copied, QuantumOperator)
        assert copied is not operator
        assert copied._qiskit_operator is not operator._qiskit_operator
        assert operator.num_paulis == 2
        assert copied.num_paulis == 3

    def test_adjoint_matches_qiskit(self):
        operator = QuantumOperator(["X"], [1 + 2j])

        result = operator.adjoint()

        assert result._qiskit_operator == operator._qiskit_operator.adjoint()

    def test_apply_layout_matches_qiskit(self):
        operator = QuantumOperator(["ZI"], [1.0])
        layout = [1, 0]

        result = operator.apply_layout(layout)

        assert result._qiskit_operator == operator._qiskit_operator.apply_layout(layout)

    def test_simplify_combines_equal_paulis(self):
        operator = QuantumOperator(["Z", "Z"], [1.0, -0.5])

        simplified = operator.simplify()

        assert simplified.paulis == ["Z"]
        assert np.allclose(simplified.coeffs, [0.5])

    def test_transpose_matches_qiskit(self):
        operator = QuantumOperator(["XY"], [1.0])

        result = operator.transpose()

        assert result._qiskit_operator == operator._qiskit_operator.transpose()

    def test_conjugate_matches_qiskit(self):
        operator = QuantumOperator(["X"], [1 + 1j])

        result = operator.conjugate()

        assert result._qiskit_operator == operator._qiskit_operator.conjugate()

    def test_group_commuting_returns_quantum_operators(self):
        operator = QuantumOperator(["ZI", "IZ", "XX"], [1.0, 1.0, 0.5])

        groups = operator.group_commuting()

        assert isinstance(groups, list)
        assert groups
        assert all(isinstance(group, QuantumOperator) for group in groups)


class TestQuantumOperatorPropertiesAndRepresentation:
    def test_is_unitary_for_pauli_operator(self):
        operator = QuantumOperator(["Z"], [1.0])

        assert operator.is_unitary

    def test_hash_equality_and_repr(self):
        left = QuantumOperator(["ZI", "IZ"], [1.0, 0.5])
        right = QuantumOperator(["ZI", "IZ"], [1.0, 0.5])
        other = QuantumOperator(["ZI"], [1.0])

        assert left == right
        assert left != other
        assert hash(left) == hash(right)
        assert isinstance(str(left), str)
        assert repr(left) == str(left)
