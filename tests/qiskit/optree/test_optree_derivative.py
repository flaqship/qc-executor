import numpy as np
import pytest
from packaging import version
from qiskit import __version__ as qiskit_version
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from executor.qiskit.optree import OpTree
from executor.qiskit.optree.optree import (
    OpTreeList,
    OpTreeNodeBase,
    OpTreeOperator,
    OpTreeSum,
    OpTreeValue,
)
from executor.qiskit.optree.optree_derivative import (
    _circuit_parameter_shift,
    _differentiate_copy,
    _differentiate_inplace,
    _operator_differentiation,
)

QISKIT_SMALLER_2_0 = version.parse(qiskit_version) < version.parse("2.0.0")


class TestOpTreeDerivative:
    """Test class for OpTree derivatives."""

    def test_derivative(self):
        """Function for comparing analytical and numerical derivatives"""

        p = ParameterVector("p", 1)

        qc = QuantumCircuit(2)
        qc.rx(2.0 * p[0], 0)
        qc.rx(10.0 * p[0], 1)
        qc.cx(0, 1)

        operator = SparsePauliOp(["IZ", "ZI"])

        p_val = np.arange(-0.5, 0.5, 0.01)
        p_array = [{p[0]: p_} for p_ in p_val]

        if QISKIT_SMALLER_2_0:
            from qiskit.primitives import Estimator

            estimator = Estimator()
        else:
            from qiskit.primitives import StatevectorEstimator

            estimator = StatevectorEstimator(default_precision=0.0)

        val = OpTree.evaluate.evaluate_with_estimator(qc, operator, p_array, {}, estimator)
        qc_d = OpTree.derivative.differentiate(qc, p[0])
        val_d = OpTree.evaluate.evaluate_with_estimator(qc_d, operator, p_array, {}, estimator)
        qc_dd = OpTree.derivative.differentiate(qc_d, p[0])
        val_dd = OpTree.evaluate.evaluate_with_estimator(qc_dd, operator, p_array, {}, estimator)

        # Compare numerical and analytical derivatives
        assert np.linalg.norm(np.abs(np.gradient(val, p_val)[1:-1] - val_d[1:-1])) < 0.15
        assert np.linalg.norm(np.abs(np.gradient(val_d, p_val)[2:-2] - val_dd[2:-2])) < 1.5

    def test_qc_gradient(self):
        """Function for testing derivatives of the circuit"""

        # set-up of the expectation value
        p = ParameterVector("p", 4)
        x = ParameterVector("x", 1)
        qc = QuantumCircuit(2)
        qc.rx(p[0] * x[0], 0)
        qc.rx(p[1] * x[0], 1)
        qc.ry(p[2], 0)
        qc.ry(p[3], 1)
        qc.rxx(p[0] * x[0], 0, 1)
        operator = SparsePauliOp(["IZ", "ZI"])
        dictionary = {x[0]: 0.5, p[0]: 1.5, p[1]: 2.5, p[2]: 0.5, p[3]: 0.25}

        if QISKIT_SMALLER_2_0:
            from qiskit.primitives import Estimator

            estimator = Estimator()
        else:
            from qiskit.primitives import StatevectorEstimator

            estimator = StatevectorEstimator(default_precision=0.0)

        # Compare the gradient w.r.t the parameters p to precomputed values
        qc_grad = OpTree.derivative.differentiate(qc, p)
        qc_grad_v2 = OpTree.derivative.differentiate_v2(qc, p)
        reference_grad = np.array([-0.59681901, -0.31954279, -0.67203245, -0.19903458])
        assert np.allclose(
            OpTree.evaluate.evaluate_with_estimator(qc_grad, operator, dictionary, {}, estimator),
            reference_grad,
        )
        assert np.allclose(
            OpTree.evaluate.evaluate_with_estimator(
                qc_grad_v2, operator, dictionary, {}, estimator
            ),
            reference_grad,
        )

        # Compare the gradient w.r.t x to precomputed values
        qc_dx = OpTree.derivative.differentiate(qc, x)
        qc_dx_v2 = OpTree.derivative.differentiate_v2(qc, x)
        reference_dx = np.array([-3.38817095])

        assert np.allclose(
            OpTree.evaluate.evaluate_with_estimator(qc_dx, operator, dictionary, {}, estimator),
            reference_dx,
        )
        assert np.allclose(
            OpTree.evaluate.evaluate_with_estimator(qc_dx_v2, operator, dictionary, {}, estimator),
            reference_dx,
        )

    def test_operator_gradient(self):
        """Function for testing derivatives of the operator"""

        p = ParameterVector("p", 4)
        dictionary_p = {p[0]: 1.5, p[1]: 2.5, p[2]: 0.5, p[3]: 0.25}

        operator = SparsePauliOp(["IZ", "ZI", "IX", "XI"], [p[0], p[1], p[2], p[3]])
        operator = operator.power(2)  # square operator for a more complicated operator
        # trivial circuit
        qc = QuantumCircuit(2)
        qc.h([0, 1])

        if QISKIT_SMALLER_2_0:
            from qiskit.primitives import Estimator

            estimator = Estimator()
        else:
            from qiskit.primitives import StatevectorEstimator

            estimator = StatevectorEstimator(default_precision=0.0)

        # Check if the gradient reproduces the correct values
        op_grad = OpTree.derivative.differentiate(operator, p)
        op_grad_v2 = OpTree.derivative.differentiate_v2(operator, p)
        reference_values = np.array([3.0, 5.0, 1.5, 1.5])
        assert np.allclose(
            OpTree.evaluate.evaluate_with_estimator(qc, op_grad, {}, dictionary_p, estimator),
            reference_values,
        )
        assert np.allclose(
            OpTree.evaluate.evaluate_with_estimator(qc, op_grad_v2, {}, dictionary_p, estimator),
            reference_values,
        )

        if QISKIT_SMALLER_2_0:
            from qiskit.primitives import Sampler

            sampler = Sampler()
        else:
            from qiskit.primitives import StatevectorSampler

            sampler = StatevectorSampler(seed=0, default_shots=10000)
            reference_values = np.array(
                [
                    3.028,
                    5.0154,
                    1.49,
                    1.494,
                ]
            )

        # Check if gradient works with a derivative of the z-basis transformed operator
        operator_z = OpTree.evaluate.transform_to_zbasis(operator)
        op_grad_z = OpTree.derivative.differentiate(operator_z, p)
        op_grad_z_v2 = OpTree.derivative.differentiate_v2(operator_z, p)
        assert np.allclose(
            OpTree.evaluate.evaluate_with_sampler(qc, op_grad_z, {}, dictionary_p, sampler),
            reference_values,
        )
        assert np.allclose(
            OpTree.evaluate.evaluate_with_sampler(qc, op_grad_z_v2, {}, dictionary_p, sampler),
            reference_values,
        )

        # Check if gradient works with a z-basis transformed operator
        assert np.allclose(
            OpTree.evaluate.evaluate_with_sampler(
                qc, OpTree.evaluate.transform_to_zbasis(op_grad), {}, dictionary_p, sampler
            ),
            reference_values,
        )
        assert np.allclose(
            OpTree.evaluate.evaluate_with_sampler(
                qc, OpTree.evaluate.transform_to_zbasis(op_grad_v2), {}, dictionary_p, sampler
            ),
            reference_values,
        )

    def test_nonlinear_parameter_error(self):
        """Test that non-linear parameters raise an error"""

        p = ParameterVector("p", 1)

        # Test with arccos (non-linear function)
        qc_arccos = QuantumCircuit(1)
        qc_arccos.rx(np.arccos(p[0]), 0)

        with pytest.raises(ValueError, match="Parameter shift rule cannot be applied"):
            OpTree.derivative.differentiate(qc_arccos, p[0])

        # Test with sin (non-linear function)
        qc_sin = QuantumCircuit(1)
        qc_sin.rx(np.sin(p[0]), 0)

        with pytest.raises(ValueError, match="Parameter shift rule cannot be applied"):
            OpTree.derivative.differentiate(qc_sin, p[0])

        # Test with quadratic (non-linear)
        qc_quad = QuantumCircuit(1)
        qc_quad.rx(p[0] ** 2, 0)

        with pytest.raises(ValueError, match="Parameter shift rule cannot be applied"):
            OpTree.derivative.differentiate(qc_quad, p[0])

        # Test that linear parameters still work
        qc_linear = QuantumCircuit(1)
        qc_linear.rx(2.0 * p[0] + 1.0, 0)

        # This should not raise an error
        qc_linear_d = OpTree.derivative.differentiate(qc_linear, p[0])
        assert qc_linear_d is not None


class TestOpTreeDerivativeHelpers:

    class TestCircuitParameterShift:
        """Test class for the circuit parameter shift helper function"""

        def test_param_in_instruction_empty_params(self):
            """Test with instruction having no parameters"""

            # Create a mock instruction with empty params
            p = ParameterVector("p", 1)

            # Access the nested function through closure inspection
            # We'll test by creating a simple circuit and inspecting behavior
            qc = QuantumCircuit(1)
            qc.h(0)  # H gate has no params

            # Circuit with no parameters should return 0.0
            result = _circuit_parameter_shift(qc, p[0])
            assert isinstance(result, OpTreeValue)
            assert result.value == 0.0

        def test_param_in_instruction_parameter_expression_match(self):
            """Test with instruction having matching ParameterExpression"""

            p = ParameterVector("p", 1)

            # Circuit with parameterized rotation
            qc = QuantumCircuit(1)
            qc.rx(p[0], 0)

            # Should return non-zero OpTreeSum since parameter is in circuit
            result = _circuit_parameter_shift(qc, p[0])
            assert isinstance(result, OpTreeSum)
            assert len(result.children) > 0

        def test_circuit_parameter_shift_return_zero_for_optree_value(self):
            """Test that _circuit_parameter_shift returns 0.0 for an OpTreeValue"""

            p = ParameterVector("p", 1)

            # Create an OpTreeValue and test that the function returns 0.0
            optree_value = OpTreeValue(5.0)
            result = _circuit_parameter_shift(optree_value, p[0])
            assert isinstance(result, OpTreeValue)
            assert result.value == 0.0

        def test_circuit_parameter_shift_raises_for_non_circuit_input(self):
            """Test that _circuit_parameter_shift raises an error for non-circuit input"""

            p = ParameterVector("p", 1)

            # Create a non-circuit input (e.g., a string) and test that it raises an error
            with pytest.raises(
                ValueError, match="element must be a CircuitTreeLeaf or a QuantumCircuit"
            ):
                _circuit_parameter_shift("not a circuit", p[0])

    class TestOperatorDifferentiation:
        """Test class for operator differentiation helper function"""

        def test_operator_differentiation_returns_zero_for_optree_value(self):
            """Test that operator differentiation returns 0.0 for an OpTreeValue"""

            p = ParameterVector("p", 1)

            # Create an OpTreeValue and test that the function returns 0.0
            optree_value = OpTreeValue(5.0)
            result = _operator_differentiation(optree_value, p[0])
            assert isinstance(result, OpTreeValue)
            assert result.value == 0.0

        def test_operator_differentiation_returns_optree_operator_for_optree_opertor_input(self):
            """Test that operator differentiation returns an OpTreeOperator for an OpTreeOperator input"""

            p = ParameterVector("p", 1)

            optree_operator = OpTreeOperator(SparsePauliOp(["I"], [p[0]]))
            result = _operator_differentiation(optree_operator, p[0])
            assert isinstance(result, OpTreeOperator)
            assert result.operator == SparsePauliOp(["I"], [1.0])

    class TestDifferentiateInplace:
        """Test class for in-place differentiation"""

        def test_differentiate_inplace_modifies_optree_recursive(self):
            """Test that in-place differentiation modifies the original OpTree"""

            p = ParameterVector("p", 1)

            # Create a simple OpTree with a parameterized operator
            optree = OpTreeList([OpTreeList([OpTreeOperator(SparsePauliOp(["I"], [p[0]]))])])

            # Differentiate in-place
            _differentiate_inplace(optree, p[0])

            # Check that the original optree has been modified to contain the derivative
            assert isinstance(optree, OpTreeList)
            assert len(optree.children) == 1
            assert isinstance(optree.children[0].children[0], OpTreeOperator)
            assert optree.children[0].children[0].operator == SparsePauliOp(["I"], [1.0])

        def test_differentiate_inplace_optree_value(self):
            """Test that in-place differentiation modifies the original OpTreeValue"""

            p = ParameterVector("p", 1)

            # Create a simple OpTree with a parameterized operator
            optree = OpTreeList([OpTreeValue(1)])

            # Differentiate in-place
            _differentiate_inplace(optree, p[0])

            # Check that the original optree has been modified to contain the derivative
            assert isinstance(optree, OpTreeList)
            assert len(optree.children) == 1
            assert isinstance(optree.children[0], OpTreeValue)
            assert optree.children[0].value == 0.0

        def test_differentiate_inplace_parameter_expression_factor(self):
            """Test that in-place differentiation modifies the original OpTreeSum when the factor is a ParameterExpression"""
            p = ParameterVector("p", 1)

            tree = OpTreeSum([OpTreeValue(2.0)], [p[0]])

            _differentiate_inplace(tree, p[0])

            assert tree.factor[0] == 1.0
            assert isinstance(tree.children[0], OpTreeSum)

            inner = tree.children[0]
            assert inner.factor[0] == 1.0
            assert inner.factor[1] == p[0]

        def test_differentiate_inplace_parameter_expression_grad_fac(self):
            """Test that in-place differentiation modifies the original OpTreeSum when the grad_fac is a ParameterExpression"""
            p = ParameterVector("p", 2)

            tree = OpTreeSum([OpTreeValue(2.0)], [p[0] * p[1]])

            _differentiate_inplace(tree, p[0])

            assert tree.factor[0] == 1.0
            assert isinstance(tree.children[0], OpTreeSum)

            inner = tree.children[0]
            assert inner.factor[0] == p[1]
            assert inner.factor[1] == p[0] * p[1]

        def test_differentiate_inplace_raises_for_non_sum_or_list(self):
            """Test that in-place differentiation raises an error for non-sum or non-list input"""

            p = ParameterVector("p", 1)

            # Create a non-sum/list input (e.g., an OpTreeOperator) and test that it raises an error
            with pytest.raises(
                ValueError, match="tree_node must be a OpTreeNodeSum or a OpTreeNodeList"
            ):
                _differentiate_inplace(OpTreeOperator(SparsePauliOp(["I"], [p[0]])), p[0])

    class TestDifferentiateCopy:
        def test_factor_is_parameter_expression_and_grad_is_zero(self):
            p = ParameterVector("p", 2)

            tree = OpTreeSum([OpTreeValue(2.0)], [p[1]])
            result = _differentiate_copy(tree, p[0])

            assert isinstance(result, OpTreeSum)
            assert result.factor[0] == p[1]
            assert isinstance(result.children[0], OpTreeValue)
            assert result.children[0].value == 0.0

        def test_factor_is_parameter_expression_and_grad_is_nonzero_float(self):
            p = ParameterVector("p", 2)

            tree = OpTreeSum([OpTreeValue(2.0)], [p[0] + p[1]])
            result = _differentiate_copy(tree, p[0])

            assert isinstance(result, OpTreeSum)
            assert result.factor[0] == 1.0
            assert isinstance(result.children[0], OpTreeSum)
            assert result.children[0].factor[0] == 1.0
            assert result.children[0].factor[1] == p[0] + p[1]

        def test_factor_is_parameter_expression_and_grad_is_parameter_expression(self):
            p = ParameterVector("p", 2)

            tree = OpTreeSum([OpTreeValue(2.0)], [p[0] * p[1]])
            result = _differentiate_copy(tree, p[0])

            assert isinstance(result, OpTreeSum)
            assert result.factor[0] == 1.0
            assert isinstance(result.children[0], OpTreeSum)
            assert result.children[0].factor[0] == p[1]
            assert result.children[0].factor[1] == p[0] * p[1]

        def test_factor_is_not_parameter_expression(self):
            p = ParameterVector("p", 1)

            tree = OpTreeList([OpTreeValue(2.0)], [2.5])
            result = _differentiate_copy(tree, p[0])

            assert isinstance(result, OpTreeList)
            assert result.factor[0] == 2.5
            assert isinstance(result.children[0], OpTreeValue)
            assert result.children[0].value == 0.0

        def test_differentiate_copy_raises_for_non_sum_or_list(self):
            """Test that copy differentiation raises an error for non-sum or non-list input"""

            class OpTreeNodeContainer(OpTreeNodeBase):
                def __init__(self, children):
                    super().__init__(children)

            p = ParameterVector("p", 1)

            # Create a non-sum/list input and test that it raises an error
            with pytest.raises(
                ValueError, match="element must be a CircuitTreeSum or a CircuitTreeList"
            ):
                _differentiate_copy(OpTreeNodeContainer(SparsePauliOp(["I"], [p[0]])), p[0])

        def test_differentiate_copy_raises_for_unsupported_element_type(self):
            """Test that copy differentiation correctly handles nested sums"""

            p = ParameterVector("p", 1)

            with pytest.raises(ValueError, match="Unsupported element type"):
                _differentiate_copy("unsupported", p[0])
