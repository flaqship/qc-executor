"""Tests for the Qulacs operator and the batch that evaluates several at once.

``QulacsOperator`` used to accept either one operator or a list of them, with
nested internal state for the list case.  It is now one observable and a
``QuantumOperatorBase``, so the whole inherited algebra works on it; evaluating
several against one circuit is ``QulacsObservableBatch``, which is what the
multi-observable gradient path uses.
"""

import numpy as np
import pytest
from qulacs import GeneralQuantumOperator, PauliOperator  # pylint: disable=no-name-in-module

from qc_executor import QuantumOperator
from qc_executor.base.operator_base import QuantumOperatorBase
from qc_executor.parameters import Parameters
from qc_executor.qulacs import QulacsOperator
from qc_executor.qulacs.qulacs_operator import QulacsObservableBatch


class TestQulacsOperatorProperties:
    def test_it_is_a_quantum_operator(self):
        assert issubclass(QulacsOperator, QuantumOperatorBase)

    def test_from_quantum_operator_single(self):
        """Test creation from a single generic operator."""
        op = QuantumOperator(["Z"], [1.0])

        qulacs_op = QulacsOperator.from_quantum_operator(op)

        assert isinstance(qulacs_op, QulacsOperator)
        assert qulacs_op.num_qubits == 1
        assert len(qulacs_op.parameter_names) == 0
        assert not qulacs_op.parameter_dimensions

    def test_it_can_be_built_directly(self):
        """Construction takes labels and coefficients, like any operator."""
        qulacs_op = QulacsOperator(["ZI", "IX"], [1.0, 0.5])

        assert qulacs_op.paulis == ["ZI", "IX"]
        assert qulacs_op.num_qubits == 2

    def test_hash_is_derived_from_the_operator_content(self):
        """The cache key comes from the representation's fingerprint."""
        same = QulacsOperator(["Z"], [1.0])
        also_same = QulacsOperator(["Z"], [1.0])
        different = QulacsOperator(["Z"], [2.0])

        assert same.fingerprint() == also_same.fingerprint()
        assert same.fingerprint() != different.fingerprint()
        assert hash(same) == hash(also_same)


class TestInheritedAlgebra:
    """The point of subclassing: the operator algebra works on the native type."""

    @pytest.mark.parametrize(
        "operation", ["adjoint", "transpose", "conjugate", "simplify", "copy"]
    )
    def test_unary_operations_match_the_generic_operator(self, operation):
        labels, coeffs = ["ZI", "IY"], [1.0, 0.5j]

        native = getattr(QulacsOperator(labels, coeffs), operation)()
        generic = getattr(QuantumOperator(labels, coeffs), operation)()

        assert isinstance(native, QulacsOperator)
        assert native.paulis == generic.paulis
        assert native.coeffs == pytest.approx(generic.coeffs)

    def test_compose_matches_the_generic_operator(self):
        native = QulacsOperator(["ZI"], [1.0]).compose(QulacsOperator(["IX"], [0.5]))
        generic = QuantumOperator(["ZI"], [1.0]).compose(QuantumOperator(["IX"], [0.5]))

        assert native.paulis == generic.paulis
        assert native.coeffs == pytest.approx(generic.coeffs)

    def test_apply_layout_matches_the_generic_operator(self):
        native = QulacsOperator(["ZII", "IXI"], [1.0, 0.5]).apply_layout([2, 0, 1])
        generic = QuantumOperator(["ZII", "IXI"], [1.0, 0.5]).apply_layout([2, 0, 1])

        assert native.paulis == generic.paulis


class TestQulacsOperatorCompilation:
    def test_numeric_coefficients_compile_to_terms(self):
        qulacs_op = QulacsOperator(["ZI", "IX"], [1.5, -0.25])

        assert qulacs_op.terms == ["Z 0 I 1 ", "I 0 X 1 "]
        assert qulacs_op.parameter_names == []

    def test_parameter_expressions_are_collected(self):
        x = Parameters("x", 2)
        qulacs_op = QulacsOperator(["Z", "X"], [x[0], 2.0 * x[0] + x[1] * x[0]])

        assert "x" in qulacs_op.parameter_names
        assert qulacs_op.parameter_dimensions["x"] == 2
        assert qulacs_op.free_parameters == {x[0], x[1]}

    def test_building_an_operator_evaluates_the_coefficients(self):
        x = Parameters("x", 1)
        qulacs_op = QulacsOperator(["Z"], [2.0 * x[0]])

        assert isinstance(qulacs_op.build_operator([0.5]), GeneralQuantumOperator)


class TestQulacsObservableBatch:
    def test_a_batch_reports_the_union_of_parameters(self):
        x = Parameters("x", 1)
        batch = QulacsObservableBatch(
            [QulacsOperator(["Z"], [x[0]]), QulacsOperator(["X"], [1.0])]
        )

        assert len(batch) == 2
        assert batch.num_qubits == 1
        assert batch.parameter_dimensions == {"x": 1}
        assert batch.free_parameters == {x[0]}

    def test_an_empty_batch_is_rejected(self):
        with pytest.raises(ValueError, match="at least one operator"):
            QulacsObservableBatch([])

    def test_mismatched_widths_are_rejected(self):
        with pytest.raises(ValueError, match="same number of qubits"):
            QulacsObservableBatch([QulacsOperator(["Z"], [1.0]), QulacsOperator(["ZI"], [1.0])])

    def test_split_arguments_gives_each_operator_its_own_values(self):
        """Each observable was lambdified against its own symbols."""
        x, y = Parameters("x", 1), Parameters("y", 2)
        batch = QulacsObservableBatch(
            [QulacsOperator(["Z"], [x[0]]), QulacsOperator(["X", "Y"], [y[0], y[1]])]
        )

        assert batch.split_arguments([0.1, 0.2, 0.3]) == [[0.1], [0.2, 0.3]]

    def test_get_operator_func_builds_one_operator_per_observable(self):
        x = Parameters("x", 2)
        batch = QulacsObservableBatch(
            [QulacsOperator(["Z"], [x[0]]), QulacsOperator(["X"], [x[1]])]
        )

        operators = batch.get_operator_func()(0.5, -0.25)

        assert len(operators) == 2
        assert all(isinstance(o, GeneralQuantumOperator) for o in operators)

    def test_gradient_terms_select_the_dependent_paulis(self):
        x = Parameters("x", 2)
        batch = QulacsObservableBatch([QulacsOperator(["Z", "X"], [x[0], x[1]])])

        terms = batch.get_operators_for_gradient(x[0])()

        assert len(terms) == 1
        assert len(terms[0]) == 1
        assert isinstance(terms[0][0], PauliOperator)

    def test_gradient_terms_are_empty_without_parameters(self):
        x = Parameters("x", 1)
        batch = QulacsObservableBatch([QulacsOperator(["Z"], [x[0]])])

        assert batch.get_operators_for_gradient()() == [[]]

    def test_the_outer_jacobian_differentiates_the_coefficients(self):
        x = Parameters("x", 2)
        batch = QulacsObservableBatch(
            [QulacsOperator(["Z", "X"], [x[0], 2.0 * x[0] + x[1] * x[0]])]
        )

        jacobians = batch.get_gradient_outer_jacobian_operators_new(x[0])([0.5, 0.25])

        assert len(jacobians) == 1
        assert jacobians[0].shape == (2, 1)
        assert np.isclose(jacobians[0][0, 0], 1.0)
        assert np.isclose(jacobians[0][1, 0], 2.25)

    def test_the_outer_jacobian_handles_several_parameters(self):
        x = Parameters("x", 2)
        batch = QulacsObservableBatch([QulacsOperator(["Z"], [2.0 * x[0] + x[1] * x[0]])])

        jacobians = batch.get_gradient_outer_jacobian_operators_new([x[0], x[1]])([0.5, 0.25])

        assert jacobians[0].shape == (1, 2)
        assert np.isclose(np.max(jacobians[0][0]), 2.25)

    def test_the_outer_jacobian_is_empty_without_parameters(self):
        x = Parameters("x", 1)
        batch = QulacsObservableBatch([QulacsOperator(["Z"], [x[0]])])

        assert batch.get_gradient_outer_jacobian_operators_new()([0.1])[0].shape == (0, 0)

    def test_a_batch_spans_several_observables_for_gradients(self):
        x = Parameters("x", 1)
        batch = QulacsObservableBatch(
            [QulacsOperator(["Z"], [x[0]]), QulacsOperator(["X"], [1.0])]
        )

        terms = batch.get_operators_for_gradient([x[0]])()

        assert len(terms) == 2
        assert len(terms[0]) == 1
        assert len(terms[1]) == 0

    def test_repr_names_the_size(self):
        assert "1 observables" in repr(QulacsObservableBatch([QulacsOperator(["Z"], [1.0])]))

    def test_iteration_and_indexing(self):
        first, second = QulacsOperator(["Z"], [1.0]), QulacsOperator(["X"], [1.0])
        batch = QulacsObservableBatch([first, second])

        assert list(batch) == [first, second]
        assert batch[1] is second
        assert batch.operators == [first, second]
