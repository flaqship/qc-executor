"""Unit tests for core OpTree structures and helpers."""

from types import SimpleNamespace

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qc_executor.parameters import Parameters
from qc_executor.qiskit.optree import (
    OpTree,
    OpTreeCircuit,
    OpTreeContainer,
    OpTreeExpectationValue,
    OpTreeList,
    OpTreeMeasuredOperator,
    OpTreeOperator,
    OpTreeSum,
    OpTreeValue,
)
from qc_executor.qiskit.optree.optree import OpTreeNodeBase, _simplify_operator


class _DummyNode(OpTreeNodeBase):
    """Helper pseudo-node for unsupported OpTree node branches."""

    def __init__(self, children=None, factor=None, operation=None):
        super().__init__(children, factor, operation)


class TestOpTreeNodeBase:
    """Tests for node classes (`OpTreeList` / `OpTreeSum`)."""

    def test_init_append_remove_and_properties(self):
        """Check initialization defaults, append and remove behavior."""
        c0 = OpTreeValue(1.0)
        c1 = OpTreeValue(2.0)
        node = OpTreeList([c0])

        assert node.children == [c0]
        assert node.factor == [1.0]
        assert node.operation == [None]

        node.append(c1, factor=2.5, operation=float)
        assert node.children == [c0, c1]
        assert node.factor == [1.0, 2.5]
        assert node.operation == [None, float]

        node.remove(0)
        assert node.children == [c1]
        node.remove([])
        assert node.children == [c1]

    def test_init_raises_for_mismatching_list_lengths(self):
        """Check list length validation in constructor."""
        with pytest.raises(ValueError, match="factor_list"):
            OpTreeSum([OpTreeValue(1.0)], factor_list=[1.0, 2.0])

        with pytest.raises(ValueError, match="operation_list"):
            OpTreeSum([OpTreeValue(1.0)], operation_list=[None, None])

    def test_remove_raises_for_too_many_indices(self):
        """Check remove validation against number of children."""
        node = OpTreeList([OpTreeValue(1.0)])
        with pytest.raises(ValueError, match="must not be larger"):
            node.remove([0, 1])

    def test_node_copy_and_equality(self):
        """Check deep copy and equality semantics for nodes."""
        v0 = OpTreeValue(1.0)
        v1 = OpTreeValue(2.0)
        node = OpTreeSum([v0, v1], factor_list=[3.0, 4.0], operation_list=[None, abs])

        node_copy = node.copy()
        assert node_copy == node
        assert node_copy is not node
        assert node_copy.children is not node.children
        assert node_copy.factor is not node.factor
        assert node_copy.operation is not node.operation

    def test_node_string_representation(self):
        """Check string conversion for list and sum nodes."""
        node_list = OpTreeList([OpTreeValue(1.0), OpTreeValue(2.0)], [2.0, 3.0])
        node_sum = OpTreeSum([OpTreeValue(1.0), OpTreeValue(2.0)], [2.0, 3.0])

        assert str(node_list) == "[2.0*1.0, 3.0*2.0]"
        assert str(node_sum) == "(2.0*1.0 + 3.0*2.0)"

    def test_node_string_representation_with_quantum_circuit_child(self):
        """Check `__str__` branch for `QuantumCircuit` children."""
        qc = QuantumCircuit(1)
        qc.x(0)

        node_list = OpTreeList([qc], [2.0])
        node_sum = OpTreeSum([qc], [3.0])

        assert "2.0*" in str(node_list)
        assert "3.0*" in str(node_sum)

    @pytest.mark.parametrize(
        "left,right",
        [
            (OpTreeList([OpTreeValue(1.0)]), OpTreeList([OpTreeValue(1.0), OpTreeValue(2.0)])),
            (OpTreeList([OpTreeValue(1.0)], [1.0]), OpTreeList([OpTreeValue(1.0)], [2.0])),
            (OpTreeList([OpTreeValue(1.0)]), OpTreeList([OpTreeValue(2.0)])),
        ],
    )
    def test_node_equality_mismatch(self, left, right):
        """Check common mismatch branches in `OpTreeNodeBase.__eq__`."""
        assert left != right

    def test_node_equality_type_mismatch(self):
        """Check `__eq__` with unrelated object type."""
        assert OpTreeList([OpTreeValue(1.0)]) != 42

    def test_node_equality_factor_and_operation_mismatch(self):
        """Check branch where factor and operation differ simultaneously."""
        child = OpTreeValue(1.0)
        left = OpTreeList([child], factor_list=[2.0], operation_list=[abs])
        right = OpTreeList([child], factor_list=[3.0], operation_list=[None])

        assert left != right

    def test_node_equality_factor_set_length_mismatch(self):
        """Trigger the `len(fac_set_self) != len(fac_set_other)` branch."""
        a = OpTreeValue(1.0)
        b = OpTreeValue(2.0)

        left = OpTreeList([a, b], factor_list=[1.0, 1.0])
        right = OpTreeList([a, b], factor_list=[1.0, 2.0])

        assert left != right

    def test_node_equality_slow_check_both_factor_and_operation_differ(self):
        """Trigger the slow-check branch where both factor and operation differ."""
        a = OpTreeValue(1.0)
        b = OpTreeValue(2.0)

        left = OpTreeList([a, b], factor_list=[1.0, 2.0], operation_list=[None, None])
        right = OpTreeList([a, b], factor_list=[2.0, 1.0], operation_list=[abs, None])

        assert left != right


class TestOpTreeLeafs:
    """Tests for leaf classes and their helper methods."""

    def test_circuit_leaf_copy_and_hash_equality(self):
        """Check circuit leaf copy behavior and equality via hash."""
        qc = QuantumCircuit(1)
        qc.h(0)

        leaf = OpTreeCircuit(qc)
        leaf_copy = leaf.copy()

        assert leaf == leaf_copy
        assert leaf.hashvalue == leaf_copy.hashvalue
        assert leaf.circuit is not leaf_copy.circuit

    def test_operator_leaf_setter_updates_hash(self):
        """Check that operator setter updates cached hashvalue."""
        op = OpTreeOperator(SparsePauliOp(["Z"], [1.0]))
        old_hash = op.hashvalue

        op.operator = SparsePauliOp(["X"], [1.0])
        assert op.hashvalue != old_hash

    def test_expectation_value_and_measured_operator(self):
        """Check expectation and measured-operator leaf behavior."""
        qc = QuantumCircuit(1)
        qc.h(0)
        op = SparsePauliOp(["Z"], [1.0])

        exp_value = OpTreeExpectationValue(qc, op)
        measured = OpTreeMeasuredOperator(qc, op)
        base_circuit = QuantumCircuit(1)
        measured_exp = measured.measure_circuit(base_circuit)

        assert isinstance(measured_exp, OpTreeExpectationValue)
        assert exp_value.operator == op
        assert isinstance(exp_value.hashvalue, tuple)

    @pytest.mark.parametrize(
        "factory,other",
        [
            (OpTreeCircuit(QuantumCircuit(1)), "bad"),
            (OpTreeOperator(SparsePauliOp(["Z"], [1.0])), "bad"),
            (
                OpTreeExpectationValue(QuantumCircuit(1), SparsePauliOp(["Z"], [1.0])),
                "bad",
            ),
        ],
    )
    def test_leaf_equality_false_for_other_types(self, factory, other):
        """Check leaf `__eq__` false branch for non-matching types."""
        assert factory != other

    def test_leaf_string_and_copy_methods(self):
        """Check `__str__`/`copy` paths for leaf types."""
        qc = QuantumCircuit(1)
        qc.h(0)
        op = SparsePauliOp(["Z"], [1.0])

        circuit_leaf = OpTreeCircuit(qc)
        operator_leaf = OpTreeOperator(op)
        expectation = OpTreeExpectationValue(qc, op)
        measured = OpTreeMeasuredOperator(qc, op)

        assert "H" in str(circuit_leaf)
        assert "SparsePauliOp" in str(operator_leaf)
        assert "with observable" in str(expectation)
        assert isinstance(measured.copy(), OpTreeMeasuredOperator)

    def test_operator_and_expectation_eq_false_against_other_leafs(self):
        """Extra eq checks for operator/expectation mismatch branches."""
        qc = QuantumCircuit(1)
        qc.h(0)
        op = SparsePauliOp(["Z"], [1.0])

        op_leaf = OpTreeOperator(op)
        circ_leaf = OpTreeCircuit(qc)
        expectation = OpTreeExpectationValue(qc, op)

        assert (op_leaf == circ_leaf) is False
        assert (expectation == op_leaf) is False

    def test_container_eq_returns_none_for_other_types(self):
        """Directly call `__eq__` to exercise the non-instance path (returns None)."""
        container = OpTreeContainer({"a": 1})
        assert container.__eq__(123) is False

    def test_op_tree_operator_equality_true_for_same_hash(self):
        """Ensure `OpTreeOperator.__eq__` returns True for identical operators."""
        op = SparsePauliOp(["Z", "X"], [1.0, 2.0])
        leaf1 = OpTreeOperator(op)
        leaf2 = OpTreeOperator(op.copy())

        assert leaf1 == leaf2

    def test_expectation_equality_and_copy(self):
        """Cover OpTreeExpectationValue equality (True branch) and `copy()`."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        op = SparsePauliOp(["ZI", "IZ"], [1.0, 2.0])

        ev1 = OpTreeExpectationValue(qc, op)
        # construct with copies to exercise deep-equality via hashes
        ev2 = OpTreeExpectationValue(qc.copy(), op.copy())

        assert ev1 == ev2

        ev_copy = ev1.copy()
        assert ev_copy == ev1
        assert ev_copy is not ev1

    def test_measure_circuit_accepts_optree_circuit(self):
        """Check `measure_circuit` branch when input is `OpTreeCircuit`."""
        qc = QuantumCircuit(1)
        qc.h(0)
        measured = OpTreeMeasuredOperator(qc, SparsePauliOp(["Z"], [1.0]))

        with pytest.raises(AttributeError, match="compose"):
            measured.measure_circuit(OpTreeCircuit(QuantumCircuit(1)))

    def test_expectation_value_raises_for_invalid_inputs(self):
        """Check input validation of expectation-value construction."""
        with pytest.raises(ValueError, match="Wrong format of the given circuit"):
            OpTreeExpectationValue("bad", SparsePauliOp(["Z"], [1.0]))

        with pytest.raises(ValueError, match="Wrong format of the given operator"):
            OpTreeExpectationValue(QuantumCircuit(1), "bad")

    def test_container_and_value_copy_equality(self):
        """Check container/value equality and copy behavior."""
        container = OpTreeContainer({"a": [1, 2]})
        container_copy = container.copy()
        value = OpTreeValue(3.14)

        assert container == container_copy
        assert container.item is not container_copy.item
        assert value == OpTreeValue(3.14)
        assert value.copy() == value

    def test_container_str(self):
        """Check `__str__` path for `OpTreeContainer`."""
        container = OpTreeContainer({"x": 1})
        s = str(container)
        assert "'x'" in s or "x" in s


class TestOpTreeHelpers:
    """Tests for static OpTree helper functions."""

    def test_hash_helpers_return_stable_tuples(self):
        """Check that hash helper functions return tuples."""
        qc = QuantumCircuit(1)
        qc.h(0)
        op = SparsePauliOp(["Z"], [1.0])

        assert isinstance(OpTree.hash_circuit(qc), tuple)
        assert isinstance(OpTree.hash_operator(op), tuple)

    def test_tree_statistics_and_first_leaf(self):
        """Check leaf count, depth, nested-list count and first-leaf traversal."""
        tree = OpTreeSum(
            [
                OpTreeList([OpTreeValue(1.0), OpTreeValue(2.0)]),
                OpTreeValue(3.0),
            ]
        )

        assert OpTree.get_number_of_leafs(tree) == 3
        assert OpTree.get_tree_depth(tree) == 2
        assert OpTree.get_num_nested_lists(tree) == 1
        assert isinstance(OpTree.get_first_leaf(tree), OpTreeValue)

    def test_simplify_operator_merges_duplicate_paulis(self):
        """Check duplicate Pauli terms are merged correctly."""
        operator = SparsePauliOp(["Z", "Z", "X"], [0.5, 1.5, 2.0])

        simplified = _simplify_operator(operator)
        assert simplified is not None
        assert len(simplified.paulis) == 2

        z_coeff = 0.0
        x_coeff = 0.0
        for pauli, coeff in zip(simplified.paulis, simplified.coeffs):
            if str(pauli) == "Z":
                z_coeff = coeff
            elif str(pauli) == "X":
                x_coeff = coeff

        assert np.isclose(z_coeff, 2.0)
        assert np.isclose(x_coeff, 2.0)

    def test_simplify_operator_keeps_leaf_type(self):
        """Check `_simplify_operator` preserves `OpTreeOperator` input type."""
        leaf = OpTreeOperator(SparsePauliOp(["Z", "Z"], [1.0, 2.0]))
        simplified = _simplify_operator(leaf)

        assert isinstance(simplified, OpTreeOperator)
        assert len(simplified.operator.paulis) == 1

    def test_simplify_operator_returns_none_for_empty_like_input(self):
        """Check `_simplify_operator` none-branch with empty-like operator object."""

        class EmptyLikeOperator:
            paulis = []
            coeffs = []

        assert _simplify_operator(EmptyLikeOperator()) is None

    @pytest.mark.parametrize(
        "leaf,expected",
        [
            (QuantumCircuit(1), QuantumCircuit),
            (SparsePauliOp(["Z"], [1.0]), SparsePauliOp),
        ],
    )
    def test_get_first_leaf_for_non_node_inputs(self, leaf, expected):
        """Check `get_first_leaf` passthrough branch for non-node inputs."""
        first = OpTree.get_first_leaf(leaf)
        assert isinstance(first, expected)


class TestOpTreeCompositionAndSimplify:
    """Tests for tree composition (`gen_expectation_tree`) and `simplify`."""

    def test_gen_expectation_tree_for_leaf_operator(self):
        """Check composition from circuit and operator leaves."""
        qc = QuantumCircuit(1)
        qc.h(0)
        op = SparsePauliOp(["Z"], [1.0])

        exp_tree = OpTree.gen_expectation_tree(qc, op)
        assert isinstance(exp_tree, OpTreeExpectationValue)

    def test_gen_expectation_tree_for_node_operator(self):
        """Check composition with operator tree produces matching node type."""
        qc = QuantumCircuit(1)
        qc.h(0)
        op_tree = OpTreeList([SparsePauliOp(["Z"], [1.0]), SparsePauliOp(["X"], [1.0])])

        exp_tree = OpTree.gen_expectation_tree(qc, op_tree)
        assert isinstance(exp_tree, OpTreeList)
        assert len(exp_tree.children) == 2
        assert all(isinstance(child, OpTreeExpectationValue) for child in exp_tree.children)

    def test_gen_expectation_tree_raises_on_wrong_type(self):
        """Check invalid inputs raise proper type validation errors."""
        with pytest.raises(ValueError, match="circuit_tree"):
            OpTree.gen_expectation_tree("bad", SparsePauliOp(["Z"], [1.0]))

    def test_gen_expectation_tree_with_sum_circuit_tree(self):
        """Check branch where `circuit_tree` is an `OpTreeSum`."""
        circuit_tree = OpTreeSum([QuantumCircuit(1), QuantumCircuit(1)], [1.0, 2.0])
        result = OpTree.gen_expectation_tree(circuit_tree, SparsePauliOp(["Z"], [1.0]))
        assert isinstance(result, OpTreeSum)
        assert len(result.children) == 2

    def test_gen_expectation_tree_raises_for_unsupported_node_types(self):
        """Check unsupported internal node types in both tree positions."""
        bad_circuit_tree = _DummyNode([QuantumCircuit(1)], [1.0], [None])
        bad_operator_tree = _DummyNode([SparsePauliOp(["Z"], [1.0])], [1.0], [None])

        with pytest.raises(ValueError, match="wrong type of circuit_tree"):
            OpTree.gen_expectation_tree(bad_circuit_tree, SparsePauliOp(["Z"], [1.0]))

        with pytest.raises(ValueError, match="CircuitTreeSum or a CircuitTreeList"):
            OpTree.gen_expectation_tree(QuantumCircuit(1), bad_operator_tree)

        with pytest.raises(ValueError, match="wrong type of operator_tree"):
            OpTree.gen_expectation_tree(QuantumCircuit(1), 123)

    def test_simplify_merges_nested_sum_and_equal_children(self):
        """Check simplify merges nested sums and duplicate branches."""
        shared = OpTreeValue(1.0)
        nested = OpTreeSum([shared], [3.0])
        tree = OpTreeSum([shared, nested], [2.0, 5.0])

        simplified = OpTree.simplify(tree)

        assert isinstance(simplified, OpTreeSum)
        assert len(simplified.children) == 1
        assert simplified.factor[0] == 17.0

    def test_simplify_returns_copy_for_empty_node_and_leaf(self):
        """Check simplify returns deep copies for empty nodes and plain leaves."""
        empty_sum = OpTreeSum()
        value_leaf = OpTreeValue(7.0)

        simplified_empty = OpTree.simplify(empty_sum)
        simplified_leaf = OpTree.simplify(value_leaf)

        assert isinstance(simplified_empty, OpTreeSum)
        assert simplified_empty is not empty_sum
        assert simplified_leaf == value_leaf
        assert simplified_leaf is not value_leaf

    def test_simplify_with_list_node_and_operator_leaf(self):
        """Check `simplify` branch for list nodes and operator leaves."""
        list_tree = OpTreeList([OpTreeValue(1.0), OpTreeValue(2.0)], [1.0, 2.0])
        simplified_list = OpTree.simplify(list_tree)
        assert isinstance(simplified_list, OpTreeList)

        op = SparsePauliOp(["Z", "Z"], [1.0, 2.0])
        simplified_op = OpTree.simplify(op)
        assert isinstance(simplified_op, SparsePauliOp)

    @pytest.mark.parametrize(
        "op_parent,op_child,expected",
        [
            (None, lambda value: value + 1, 4),
            (lambda value: 2 * value, None, 6),
            (lambda value: 2 * value, lambda value: value + 1, 8),
        ],
    )
    def test_simplify_combine_operations(self, op_parent, op_child, expected):
        """Check operation-combination branches in nested-sum merge."""
        child_leaf = OpTreeValue(1.0)

        nested = OpTreeSum([child_leaf], [1.0], [op_child])
        tree = OpTreeSum([nested], [1.0], [op_parent])
        simplified = OpTree.simplify(tree)

        operation = simplified.operation[0]
        assert callable(operation)
        assert operation(3) == expected

    def test_simplify_raises_for_unsupported_node_type(self):
        """Check `simplify` error branch for unsupported node subclass."""
        bad = _DummyNode([OpTreeValue(1.0)], [1.0], [None])

        with pytest.raises(ValueError, match="CircuitTreeSum or a CircuitTreeList"):
            OpTree.simplify(bad)


class TestOpTreeAssignParameters:
    """Tests for `OpTree.assign_parameters` branches."""

    def test_assign_parameters_on_tree_and_qc(self):
        """Check parameter assignment for OpTree node and bare circuit."""
        p = Parameters("p", 1)

        qc = QuantumCircuit(1)
        qc.rx(p[0], 0)

        node = OpTreeList([qc], [2.0 * p[0]])
        assigned_node = OpTree.assign_parameters(node, {p[0]: 0.5})

        assert isinstance(assigned_node, OpTreeList)
        assert assigned_node.factor[0] == 1.0
        assert len(assigned_node.children[0].parameters) == 0

        assigned_qc = OpTree.assign_parameters(qc, {p[0]: 0.5})
        assert len(assigned_qc.parameters) == 0

    def test_assign_parameters_on_expectation_and_operator(self):
        """Check parameter assignment for expectation values and operators."""
        p = Parameters("p", 1)

        qc = QuantumCircuit(1)
        qc.rx(p[0], 0)
        op = SparsePauliOp(["Z"], [p[0]])

        exp = OpTreeExpectationValue(qc, op)
        assigned_exp = OpTree.assign_parameters(exp, {p[0]: 0.25})
        assert isinstance(assigned_exp, OpTreeExpectationValue)
        assert len(assigned_exp.circuit.parameters) == 0

        op_leaf = OpTreeOperator(op)
        assigned_op_leaf = OpTree.assign_parameters(op_leaf, {p[0]: 0.75})
        assert isinstance(assigned_op_leaf, OpTreeOperator)

        assigned_op = OpTree.assign_parameters(op, {p[0]: 0.75})
        assert isinstance(assigned_op, SparsePauliOp)

    def test_assign_parameters_inplace_error_cases_and_invalid_type(self):
        """Check inplace error paths and invalid element type handling."""
        p = Parameters("p", 1)

        qc = QuantumCircuit(1)
        qc.rx(p[0], 0)
        with pytest.raises(ValueError, match="Cannot assign parameters inplace"):
            OpTree.assign_parameters(qc, {p[0]: 0.5}, inplace=True)

        op = SparsePauliOp(["Z"], [p[0]])
        with pytest.raises(ValueError, match="Cannot assign parameters inplace"):
            OpTree.assign_parameters(op, {p[0]: 0.5}, inplace=True)

        with pytest.raises(ValueError, match="OpTreeNodeBase"):
            OpTree.assign_parameters(1234, {p[0]: 0.5})

    def test_assign_parameters_inplace_on_node(self):
        """Check inplace assignment branch for node factors/circuits."""
        p = Parameters("p", 1)
        tree = OpTreeList([])
        tree._factor_list = [2.0 * p[0]]
        result = OpTree.assign_parameters(tree, {p[0]: 0.5}, inplace=True)

        assert result is None
        assert tree.factor[0] == 1.0

    def test_assign_parameters_on_sum_and_unsupported_node(self):
        """Check sum branch and unsupported-node error branch."""
        p = Parameters("p", 1)
        qc = QuantumCircuit(1)
        qc.rx(p[0], 0)

        sum_tree = OpTreeSum([qc], [2.0])
        assigned = OpTree.assign_parameters(sum_tree, {p[0]: 0.25})
        assert isinstance(assigned, OpTreeSum)
        assert assigned.factor[0] == 2.0

        bad = _DummyNode([qc], [2.0], [None])
        with pytest.raises(ValueError, match="CircuitTreeSum or a CircuitTreeList"):
            OpTree.assign_parameters(bad, {p[0]: 0.25})

    def test_assign_parameters_on_optree_circuit_branches(self):
        """Check `OpTreeCircuit` assignment branches (copy/inplace)."""
        p = Parameters("p", 1)
        qc = QuantumCircuit(1)
        qc.rx(p[0], 0)
        leaf = OpTreeCircuit(qc)

        assigned = OpTree.assign_parameters(leaf, {p[0]: 0.5})
        assert isinstance(assigned, OpTreeCircuit)
        assert len(assigned.circuit.parameters) == 0
        # copy branch must not mutate the original leaf
        assert len(leaf.circuit.parameters) == 1

        # inplace branch returns None and binds the parameter on the original leaf
        result = OpTree.assign_parameters(leaf, {p[0]: 0.5}, inplace=True)
        assert result is None
        assert len(leaf.circuit.parameters) == 0

    def test_assign_parameters_inplace_on_expectation_and_operator(self):
        """Check inplace branches for expectation/measured and operator leaves."""
        p = Parameters("p", 1)
        qc = QuantumCircuit(1)
        qc.rx(p[0], 0)
        op = SparsePauliOp(["Z"], [p[0]])

        expectation = OpTreeExpectationValue(qc, op)
        expectation._circuit = SimpleNamespace(circuit=qc)
        result = OpTree.assign_parameters(expectation, {p[0]: 0.5}, inplace=True)
        assert result is None

        operator_leaf = OpTreeOperator(op)
        op_result = OpTree.assign_parameters(operator_leaf, {p[0]: 0.5}, inplace=True)
        assert op_result is None
        assert len(operator_leaf.operator.parameters) == 0

    def test_assign_parameters_inplace_with_optree_circuit_child(self):
        """Cover the inplace node branch that recurses into `OpTreeCircuit` children."""
        p = Parameters("p", 1)
        qc = QuantumCircuit(1)
        qc.rx(p[0], 0)

        # create an OpTreeList with an OpTreeCircuit child and a parameter expression factor
        child = OpTreeCircuit(qc)
        tree = OpTreeList([child], factor_list=[2.0 * p[0]])

        result = OpTree.assign_parameters(tree, {p[0]: 0.5}, inplace=True)

        # inplace assignment returns None and mutates the tree in place: the child
        # circuit's parameter is bound and the parameter-expression factor is reduced
        # to a plain float (2.0 * 0.5).
        assert result is None
        assert len(child.circuit.parameters) == 0
        assert tree.factor[0] == pytest.approx(1.0)

    def test_assign_parameters_inplace_on_bare_leaves_raises(self):
        """Inplace assignment on bare QuantumCircuit / SparsePauliOp must raise."""
        p = Parameters("p", 1)
        qc = QuantumCircuit(1)
        qc.rx(p[0], 0)

        with pytest.raises(ValueError, match="bare QuantumCircuit"):
            OpTree.assign_parameters(qc, {p[0]: 0.5}, inplace=True)

        op = SparsePauliOp(["Z"], [p[0]])
        with pytest.raises(ValueError, match="bare SparsePauliOp"):
            OpTree.assign_parameters(op, {p[0]: 0.5}, inplace=True)
