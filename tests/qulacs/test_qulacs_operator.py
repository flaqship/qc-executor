import numpy as np
import pytest
from qulacs import GeneralQuantumOperator, PauliOperator  # pylint: disable=no-name-in-module

from qc_executor import QuantumOperator
from qc_executor.parameters import Parameters
from qc_executor.qulacs import QulacsOperator


class TestQulacsOperatorProperties:
    def test_from_quantum_operator_single(self):
        """Test creation from a single generic operator."""
        op = QuantumOperator(["Z"], [1.0])

        qulacs_op = QulacsOperator.from_quantum_operator(op)

        assert isinstance(qulacs_op, QulacsOperator)
        assert qulacs_op.num_qubits == 1
        assert len(qulacs_op.parameter_names) == 0
        assert not qulacs_op.parameter_dimensions

    def test_from_quantum_operator_list(self):
        """Test creation from a list of generic operators."""
        op1 = QuantumOperator(["Z"], [1.0])
        op2 = QuantumOperator(["X"], [1.0])

        qulacs_op = QulacsOperator.from_quantum_operator([op1, op2])

        assert isinstance(qulacs_op, QulacsOperator)
        assert qulacs_op.multiple_operators is True
        assert qulacs_op.num_qubits == 1

    def test_hash_is_derived_from_the_operator_content(self):
        """The cache key comes from the representation's fingerprint."""
        same = QulacsOperator(QuantumOperator(["Z"], [1.0]))
        also_same = QulacsOperator(QuantumOperator(["Z"], [1.0]))
        different = QulacsOperator(QuantumOperator(["Z"], [2.0]))

        assert same.hash == also_same.hash
        assert same.hash != different.hash


class TestQulacsOperatorInitialization:
    def test_init_unsupported_type_raises(self):
        """Test unsupported constructor input type."""
        with pytest.raises(ValueError, match="Unsupported operator type"):
            QulacsOperator(object())

    def test_init_list_with_invalid_element_raises(self):
        """Test list constructor validation for non-operator elements."""
        op = QuantumOperator(["Z"], [1.0])

        with pytest.raises(ValueError, match="Unsupported operator type"):
            QulacsOperator([op, object()])

    def test_init_empty_list_raises(self):
        """An empty list has no width to report."""
        with pytest.raises(ValueError, match="Unsupported operator type"):
            QulacsOperator([])


class TestQulacsOperatorBuildInstructions:
    def test_build_instructions_with_numeric_coefficients(self):
        """Test instruction build for non-parameterized operator coefficients."""
        op = QuantumOperator(["ZI", "IX"], [1.5, -0.25])
        qulacs_op = QulacsOperator(op)

        assert qulacs_op.multiple_operators is False
        assert len(qulacs_op.new_operators) == 1
        assert len(qulacs_op.new_operators[0]) == 2
        assert qulacs_op.new_operators_used_parameters[0][0] == []
        assert qulacs_op.new_operators_used_parameters[0][1] == []

    def test_build_instructions_with_parameter_expressions(self):
        """Test instruction build for parameterized coefficients."""
        x = Parameters("x", 2)
        op = QuantumOperator(["Z", "X"], [x[0], 2.0 * x[0] + x[1] * x[0]])
        qulacs_op = QulacsOperator(op)

        assert "x" in qulacs_op.parameter_names
        assert qulacs_op.parameter_dimensions["x"] == 2
        assert qulacs_op._free_parameters == {x[0], x[1]}

    def test_build_instructions_multiple_operator_objects(self):
        """Test instruction build with a list of operators."""
        x = Parameters("x", 1)
        op1 = QuantumOperator(["Z"], [x[0]])
        op2 = QuantumOperator(["X"], [1.0])

        qulacs_op = QulacsOperator([op1, op2])

        assert qulacs_op.multiple_operators is True
        assert len(qulacs_op.new_operators) == 2
        assert len(qulacs_op.new_operators_coeff) == 2


class TestQulacsOperatorRuntimeFunctions:
    def test_get_operator_func_returns_callable_and_qulacs_objects(self):
        """Test operator callable creation and returned object types."""
        x = Parameters("x", 2)
        op = QuantumOperator(["Z", "X"], [x[0], x[1]])
        qulacs_op = QulacsOperator(op)

        operator_func = qulacs_op.get_operator_func()
        operators = operator_func(0.5, -0.25)

        assert callable(operator_func)
        assert isinstance(operators, list)
        assert len(operators) == 1
        assert isinstance(operators[0], GeneralQuantumOperator)

    def test_get_gradient_outer_jacobian_with_single_parameter_element(self):
        """Test gradient parameter normalization from single element to list."""
        x = Parameters("x", 2)
        op = QuantumOperator(["Z", "X"], [x[0], 2.0 * x[0] + x[1] * x[0]])
        qulacs_op = QulacsOperator(op)

        jacobian_func = qulacs_op.get_gradient_outer_jacobian_operators_new(x[0])
        jacobians = jacobian_func([0.5, 0.25])

        assert isinstance(jacobians, list)
        assert len(jacobians) == 1
        assert jacobians[0].shape == (2, 1)
        assert np.isclose(jacobians[0][0, 0], 1.0)
        assert np.isclose(jacobians[0][1, 0], 2.25)

    def test_get_gradient_outer_jacobian_with_multiple_parameters(self):
        """Test outer Jacobian values for multiple gradient parameters."""
        x = Parameters("x", 2)
        op = QuantumOperator(["Z"], [2.0 * x[0] + x[1] * x[0]])
        qulacs_op = QulacsOperator(op)

        jacobian_func = qulacs_op.get_gradient_outer_jacobian_operators_new([x[0], x[1]])
        jacobians = jacobian_func([0.5, 0.25])

        assert len(jacobians) == 1
        assert jacobians[0].shape == (1, 2)
        row = jacobians[0][0]
        assert np.isclose(np.max(row), 2.25)
        assert np.min(row) > 0.0

    def test_get_gradient_outer_jacobian_without_gradient_parameters(self):
        """Test Jacobian behavior when no gradient parameters are requested."""
        x = Parameters("x", 1)
        op = QuantumOperator(["Z"], [x[0]])
        qulacs_op = QulacsOperator(op)

        jacobian_func = qulacs_op.get_gradient_outer_jacobian_operators_new()
        jacobians = jacobian_func([0.1])

        assert len(jacobians) == 1
        assert jacobians[0].shape == (0, 0)

    def test_get_operators_for_gradient_with_single_parameter_element(self):
        """Test gradient operator creation with single parameter input."""
        x = Parameters("x", 2)
        op = QuantumOperator(["Z", "X"], [x[0], x[1]])
        qulacs_op = QulacsOperator(op)

        operators_func = qulacs_op.get_operators_for_gradient(x[0])
        gradient_operators = operators_func()

        assert isinstance(gradient_operators, list)
        assert len(gradient_operators) == 1
        assert len(gradient_operators[0]) == 1
        assert isinstance(gradient_operators[0][0], PauliOperator)

    def test_get_operators_for_gradient_without_parameters(self):
        """Test gradient operator creation with no selected parameters."""
        x = Parameters("x", 1)
        op = QuantumOperator(["Z"], [x[0]])
        qulacs_op = QulacsOperator(op)

        operators_func = qulacs_op.get_operators_for_gradient()
        gradient_operators = operators_func()

        assert isinstance(gradient_operators, list)
        assert len(gradient_operators) == 1
        assert gradient_operators[0] == []

    def test_get_operators_for_gradient_multiple_operators(self):
        """Test gradient operator extraction for multiple operator objects."""
        x = Parameters("x", 1)
        op1 = QuantumOperator(["Z"], [x[0]])
        op2 = QuantumOperator(["X"], [1.0])
        qulacs_op = QulacsOperator([op1, op2])

        operators_func = qulacs_op.get_operators_for_gradient([x[0]])
        gradient_operators = operators_func()

        assert len(gradient_operators) == 2
        assert len(gradient_operators[0]) == 1
        assert len(gradient_operators[1]) == 0
