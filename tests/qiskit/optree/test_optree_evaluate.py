"""Tests for OpTree evaluation"""

from types import SimpleNamespace
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from packaging import version
from qiskit import __version__ as qiskit_version
from qiskit.circuit import ClassicalRegister, Parameter, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

import qc_executor.qiskit.optree.optree_evaluate as oe
from qc_executor.parameters import Parameters
from qc_executor.qiskit.optree import OpTree, OpTreeList, OpTreeSum
from qc_executor.qiskit.optree.optree import (
    OpTreeContainer,
    OpTreeExpectationValue,
    OpTreeMeasuredOperator,
    OpTreeNodeBase,
    OpTreeOperator,
    OpTreeValue,
)

QISKIT_SMALLER_2_0 = version.parse(qiskit_version) < version.parse("2.0.0")


class UnknownNode(OpTreeNodeBase):
    pass


def make_mock_ev(hashvalue="ev_hash", circuit_hashvalue="circ_hash"):
    """Minimales Mock-OpTreeExpectationValue mit parameter-freiem Operator."""
    operator = SparsePauliOp("Z")  # echtes Objekt, keine Parameter

    circuit = MagicMock()
    circuit.parameters = []

    ev = MagicMock(spec=OpTreeExpectationValue)
    ev.operator = operator
    ev.circuit = circuit
    ev._circuit = MagicMock()
    ev._circuit.hashvalue = circuit_hashvalue
    ev.hashvalue = hashvalue
    return ev


def make_mock_bitarray_result(exp_val=1.0):
    """Mock für ein einzelnes BitArray-Result (primitives v2)."""
    result = MagicMock()
    result.expectation_values.return_value = exp_val
    return result


def make_mock_sampler_result(quasi_dists):
    """Mock für SamplerResult (primitives v1)."""
    result = MagicMock(spec=oe.SamplerResult)
    result.quasi_dists = quasi_dists
    return result


def make_quasi_dist(bitstring_probs):
    """Mock für ein quasi_dist-Objekt mit binary_probabilities()."""
    qd = MagicMock()
    qd.binary_probabilities.return_value = bitstring_probs
    return qd


class TestOpTreeEvaluation:
    """Test class for OpTree evaluation"""

    @pytest.fixture(scope="module")
    def _create_random_circuits(self) -> OpTreeList:
        """Creates the random circuits used in the tests"""
        circuit1 = QuantumCircuit(2)
        circuit1.h(0)
        circuit1.cx(0, 1)
        circuit1.rz(0.5, 0)
        circuit1.ry(0.5, 1)
        circuit2 = QuantumCircuit(2)
        circuit2.h(0)
        circuit2.cp(0.5, 0, 1)
        circuit2.s(0)
        circuit2.t(1)
        circuit2.h(0)
        circuit2.h(1)
        return OpTreeList([circuit1, circuit2])

    @pytest.fixture(scope="module")
    def _create_param_circuits(self) -> Tuple[OpTreeList, List[dict]]:
        p = Parameters("p", 2)
        circuit1 = QuantumCircuit(2)
        circuit1.rx(p[0], 0)
        circuit1.rx(p[1], 1)
        circuit2 = QuantumCircuit(2)
        circuit2.ry(p[0], 0)
        circuit2.ry(p[1], 1)
        dictionary1 = {p[0]: 0.25, p[1]: 0.5}
        dictionary2 = {p[0]: 0.33, p[1]: 0.44}
        return OpTreeList([circuit1, circuit2]), [dictionary1, dictionary2]

    @pytest.fixture(scope="module")
    def _create_observable_z(self) -> Tuple[OpTreeSum, List[dict]]:
        """Creates the Z-based operators used in the tests"""
        x = Parameters("x", 2)
        observable1 = SparsePauliOp(["IZ", "ZI"], [x[0], x[1]])
        observable2 = SparsePauliOp(["II", "ZZ"], [x[0], x[1]])
        observable = OpTreeSum([observable1, observable2])
        dictionary1 = {x[0]: 1.0, x[1]: 0.5}
        dictionary2 = {x[0]: 0.3, x[1]: 0.2}
        return observable, [dictionary1, dictionary2]

    @pytest.fixture(scope="module")
    def _create_observable_xy(self) -> Tuple[OpTreeSum, dict]:
        """Creates the XY-based operators used in the tests"""
        x = Parameters("x", 2)
        observable1 = SparsePauliOp(["XY", "YX"], [x[0], x[1]])
        observable2 = SparsePauliOp(["ZZ", "YY"], [x[0], x[1]])
        observable = OpTreeSum([observable1, observable2])
        dictionary = {x[0]: 1.0, x[1]: 0.5}
        return observable, dictionary

    def test_estimator_z(self, _create_random_circuits, _create_observable_z):
        """Tests the estimator with Z basis operators

        Args:
            _create_random_circuits (Tuple[OpTreeList, List[dict]]): The circuits and dictionaries.
            _create_observable_z (Tuple[OpTreeSum, List[dict]]): The operators and dictionaries.
        """

        if QISKIT_SMALLER_2_0:
            from qiskit.primitives import Estimator

            estimator = Estimator()
        else:
            from qiskit.primitives import StatevectorEstimator

            estimator = StatevectorEstimator(default_precision=0)

        reference_values = np.array([1.43879128, 1.0])

        # Check functionality of estimator evaluation
        val = OpTree.evaluate.evaluate_with_estimator(
            _create_random_circuits,
            _create_observable_z[0],
            {},
            _create_observable_z[1][0],
            estimator=estimator,
        )
        assert np.allclose(val, reference_values)

        # Check functionality of estimator tree evaluation
        expectation_tree = OpTree.gen_expectation_tree(
            _create_random_circuits, _create_observable_z[0]
        )
        val = OpTree.evaluate.evaluate_tree_with_estimator(
            expectation_tree, _create_observable_z[1][0], estimator=estimator
        )
        assert np.allclose(val, reference_values)

    def test_sampler_z(self, _create_random_circuits, _create_observable_z):
        """Tests the sampler with Z basis operators

        Args:
            _create_random_circuits (Tuple[OpTreeList, List[dict]]): The circuits and dictionaries.
            _create_observable_z (Tuple[OpTreeSum, List[dict]]): The operators and dictionaries.
        """

        if QISKIT_SMALLER_2_0:
            from qiskit.primitives import Sampler

            sampler = Sampler()
            reference_values = np.array([1.43879128, 1.0])
        else:
            from qiskit.primitives import StatevectorSampler

            sampler = StatevectorSampler(seed=0, default_shots=5000)
            # StatevectorSampler does only support sampling, not statevectors
            reference_values = np.array([1.4524, 1.0102])

        # Check functionality of sampler evaluation
        val = OpTree.evaluate.evaluate_with_sampler(
            _create_random_circuits,
            _create_observable_z[0],
            {},
            _create_observable_z[1][0],
            sampler,
        )
        assert np.allclose(val, reference_values)

        # Check functionality of sampler tree evaluation
        expectation_tree = OpTree.gen_expectation_tree(
            _create_random_circuits, _create_observable_z[0]
        )
        val = OpTree.evaluate.evaluate_tree_with_sampler(
            expectation_tree, _create_observable_z[1][0], sampler
        )
        assert np.allclose(val, reference_values)

    def test_estimator_xy(self, _create_random_circuits, _create_observable_xy):
        """
        Tests the estimator with XY/YX basis operators

        Args:
            _create_random_circuits (Tuple[OpTreeList, List[dict]]): The circuits and dictionaries.
            _create_observable_xy (Tuple[OpTreeSum, dict]): The operators and dictionary.
        """

        if QISKIT_SMALLER_2_0:
            from qiskit.primitives import Estimator

            estimator = Estimator()
        else:
            from qiskit.primitives import StatevectorEstimator

            estimator = StatevectorEstimator(default_precision=0)

        reference_values = np.array([1.09923954, -1.0])

        # Check functionality of estimator evaluation
        val = OpTree.evaluate.evaluate_with_estimator(
            _create_random_circuits,
            _create_observable_xy[0],
            {},
            _create_observable_xy[1],
            estimator,
        )
        assert np.allclose(val, reference_values)

        # Check functionality of estimator tree evaluation
        expectation_tree = OpTree.gen_expectation_tree(
            _create_random_circuits, _create_observable_xy[0]
        )
        val = OpTree.evaluate.evaluate_tree_with_estimator(
            expectation_tree, _create_observable_xy[1], estimator
        )
        assert np.allclose(val, reference_values)

    def test_sampler_xy(self, _create_random_circuits, _create_observable_xy):
        """Tests the estimator with Z basis operators

        Args:
            _create_random_circuits (Tuple[OpTreeList, List[dict]]): The circuits and dictionaries.
            _create_observable_xy (Tuple[OpTreeSum, dict]): The operators and dictionary.
        """

        if QISKIT_SMALLER_2_0:
            from qiskit.primitives import Sampler

            sampler = Sampler()
            reference_values = np.array([1.09923954, -1.0])
        else:
            from qiskit.primitives import StatevectorSampler

            sampler = StatevectorSampler(seed=0, default_shots=5000)
            # StatevectorSampler does only support sampling, not statevectors
            reference_values = np.array([1.1204, -0.9738])

        # Check functionality of evaluation
        with pytest.raises(ValueError):
            OpTree.evaluate.evaluate_with_sampler(
                _create_random_circuits,
                _create_observable_xy[0],
                {},
                _create_observable_xy[1],
                sampler,
            )
        op_in_z_base = OpTree.evaluate.transform_to_zbasis(_create_observable_xy[0])
        val = OpTree.evaluate.evaluate_with_sampler(
            _create_random_circuits, op_in_z_base, {}, _create_observable_xy[1], sampler
        )
        assert np.allclose(val, reference_values)

        # Check functionality of tree evaluation
        expectation_tree = OpTree.gen_expectation_tree(
            _create_random_circuits, _create_observable_xy[0]
        )
        with pytest.raises(ValueError):
            OpTree.evaluate.evaluate_tree_with_sampler(
                expectation_tree, _create_observable_xy[1], sampler
            )
        expectation_tree_in_z_base = OpTree.evaluate.transform_to_zbasis(expectation_tree)
        val = OpTree.evaluate.evaluate_tree_with_sampler(
            expectation_tree_in_z_base, _create_observable_xy[1], sampler
        )
        assert np.allclose(val, reference_values)

    def test_estimator_multi_dict(self, _create_param_circuits, _create_observable_z):
        """
        Checks the functionality of the estimator with multiple dictionaries.

        Args:
            _create_param_circuits (Tuple[OpTreeList, List[dict]]): The circuits and dictionaries.
            _create_observable_z (Tuple[OpTreeSum, List[dict]]): The operators and dictionaries.
        """

        if QISKIT_SMALLER_2_0:
            from qiskit.primitives import Estimator

            estimator = Estimator()
        else:
            from qiskit.primitives import StatevectorEstimator

            estimator = StatevectorEstimator(default_precision=0)

        reference_values = np.array(
            [
                [[2.83285403, 2.83285403], [0.93625037, 0.93625037]],
                [[2.82638487, 2.82638487], [0.93594971, 0.93594971]],
            ]
        )
        val = OpTree.evaluate.evaluate_with_estimator(
            _create_param_circuits[0],
            _create_observable_z[0],
            _create_param_circuits[1],
            _create_observable_z[1],
            estimator,
        )
        assert np.allclose(val, reference_values)

        reference_values = np.array([[2.83285403, 2.83285403], [0.93594971, 0.93594971]])
        val = OpTree.evaluate.evaluate_with_estimator(
            _create_param_circuits[0],
            _create_observable_z[0],
            _create_param_circuits[1],
            _create_observable_z[1],
            estimator,
            dictionaries_combined=True,
        )
        assert np.allclose(val, reference_values)

    def test_estimator_dictionaries_combined_mismatched_lengths_raises(self):
        circ = QuantumCircuit(1)
        operator = SparsePauliOp("Z")

        dict_circ = [{}, {}]
        dict_op = [{}]

        with pytest.raises(
            ValueError, match="The length of the circuit and operator dictionary must be the same"
        ):
            oe.OpTreeEvaluate.evaluate_with_estimator(
                circ, operator, dict_circ, dict_op, estimator=None, dictionaries_combined=True
            )

    def test_estimator_with_unknown_estimator_type_raises(self):
        circ = QuantumCircuit(1)
        operator = SparsePauliOp("Z")

        with pytest.raises(ValueError, match="Unknown estimator type!"):
            oe.OpTreeEvaluate.evaluate_with_estimator(circ, operator, {}, {}, estimator=object())

        with pytest.raises(ValueError, match="Unknown estimator type!"):
            oe.OpTreeEvaluate.evaluate_tree_with_estimator(make_mock_ev(), {}, estimator=object())

    def test_estimator_with_invalid_element_type_raises(self):
        invalid_circuit_tree = UnknownNode([OpTreeContainer(0)], [1.0])

        # Patch the internal builders so evaluate_with_estimator uses our invalid tree
        with (
            patch.object(
                oe,
                "_build_circuit_list",
                return_value=([QuantumCircuit(1)], [np.array([])], invalid_circuit_tree),
            ),
            patch.object(
                oe, "_build_operator_list", return_value=([SparsePauliOp("Z")], OpTreeContainer(0))
            ),
        ):
            with pytest.raises(
                ValueError, match="element must be a CircuitTreeSum or a CircuitTreeList"
            ):
                oe.OpTreeEvaluate.evaluate_with_estimator(
                    QuantumCircuit(1), SparsePauliOp("Z"), {}, {}, estimator=None
                )

    def test_estimator_v1_branch_calls_run_and_uses_result_values(self):
        circuit_list = [QuantumCircuit(1)]
        parameter_list = [np.array([])]
        circuit_tree = OpTreeValue(0)
        operator_list = [SparsePauliOp("Z")]
        operator_tree = OpTreeContainer(0)

        class DummyEstimator(oe.BaseEstimatorV1):
            def __init__(self):
                self.called_with = None

            def _run(self, circuits, observables, parameter_values, **run_options):
                raise NotImplementedError

            def run(self, circuits, operators, parameters):
                self.called_with = (circuits, operators, parameters)
                return SimpleNamespace(result=lambda: SimpleNamespace(values=np.array([42.0])))

        est = DummyEstimator()
        with (
            patch.object(
                oe,
                "_build_circuit_list",
                return_value=(circuit_list, parameter_list, circuit_tree),
            ),
            patch.object(oe, "_build_operator_list", return_value=(operator_list, operator_tree)),
            patch.object(oe, "_add_offset_to_tree", return_value=OpTreeValue(0)),
        ):
            res = oe.OpTreeEvaluate.evaluate_with_estimator(
                QuantumCircuit(1), SparsePauliOp("Z"), {}, {}, estimator=est
            )

            res_tree = oe.OpTreeEvaluate.evaluate_tree_with_estimator(
                make_mock_ev(), {}, estimator=est
            )

        assert isinstance(est.called_with, tuple)
        assert res == 0
        assert res_tree == 0

    def test_evaluate_tree_with_estimator_multiple_dict_returns_list(self):
        with patch.object(
            oe,
            "_build_expectation_list",
            side_effect=[
                ([], [], [], [], OpTreeValue(3.14)),
                ([], [], [], [], OpTreeValue(2.0)),
            ],
        ):
            res = oe.OpTreeEvaluate.evaluate_tree_with_estimator(
                expectation_tree=make_mock_ev(), dictionary=[{}, {}], estimator=None
            )

        assert isinstance(res, np.ndarray)
        assert np.allclose(res, np.array([3.14, 2.0]))

    def test_sampler_multi_dict(self, _create_param_circuits, _create_observable_z):
        """
        Checks the functionality of the sampler with multiple dictionaries.

        Args:
            _create_param_circuits (Tuple[OpTreeList, List[dict]]): The circuits and dictionaries.
            _create_observable_z (Tuple[OpTreeSum, List[dict]]): The operators and dictionaries.

        """

        if QISKIT_SMALLER_2_0:
            from qiskit.primitives import Sampler

            sampler = Sampler()
            reference_values = np.array(
                [
                    [[2.83285403, 2.83285403], [0.93625037, 0.93625037]],
                    [[2.82638487, 2.82638487], [0.93594971, 0.93594971]],
                ]
            )
            reference_values2 = np.array([[2.83285403, 2.83285403], [0.93594971, 0.93594971]])
        else:
            from qiskit.primitives import StatevectorSampler

            sampler = StatevectorSampler(seed=0, default_shots=5000)
            # StatevectorSampler does only support sampling, not statevectors
            reference_values = np.array(
                [[[2.8344, 2.8344], [0.9368, 0.9368]], [[2.8264, 2.8264], [0.93608, 0.93608]]]
            )
            reference_values2 = np.array([[2.8344, 2.8344], [0.93608, 0.93608]])

        val = OpTree.evaluate.evaluate_with_sampler(
            _create_param_circuits[0],
            _create_observable_z[0],
            _create_param_circuits[1],
            _create_observable_z[1],
            sampler,
        )
        assert np.allclose(val, reference_values)

        val = OpTree.evaluate.evaluate_with_sampler(
            _create_param_circuits[0],
            _create_observable_z[0],
            _create_param_circuits[1],
            _create_observable_z[1],
            sampler,
            dictionaries_combined=True,
        )
        assert np.allclose(val, reference_values2)

    def test_sampler_dictionaries_combined_mismatched_lengths_raises(self):
        circ = QuantumCircuit(1)
        operator = SparsePauliOp("Z")

        dict_circ = [{}, {}]
        dict_op = [{}]

        with pytest.raises(
            ValueError, match="The length of the circuit and operator dictionary must be the same"
        ):
            oe.OpTreeEvaluate.evaluate_with_sampler(
                circ, operator, dict_circ, dict_op, sampler=None, dictionaries_combined=True
            )

    def test_sampler_with_unknown_sampler_type_raises(self):
        circ = QuantumCircuit(1)
        operator = SparsePauliOp("Z")

        with pytest.raises(ValueError, match="Unknown sampler type!"):
            oe.OpTreeEvaluate.evaluate_with_sampler(circ, operator, {}, {}, sampler=object())

        with pytest.raises(ValueError, match="Unknown sampler type!"):
            oe.OpTreeEvaluate.evaluate_tree_with_sampler(
                OpTreeExpectationValue(circ, operator), {}, sampler=object()
            )

    def test_sampler_with_base_sampler_v1_branch(self):
        circ = QuantumCircuit(1)
        operator = SparsePauliOp("Z")

        class DummySamplerV1(oe.BaseSamplerV1):
            def _run(self, circuits, parameter_values, **run_options):
                raise NotImplementedError

            def run(self, circuits, parameter_list):
                return SimpleNamespace(result=lambda: "SAMPLER_RESULT_OBJ")

        sampler = DummySamplerV1()

        with patch.object(
            oe,
            "_evaluate_expectation_from_sampler",
            return_value=[np.array([3.14])],
        ) as mock_eval:
            val = oe.OpTreeEvaluate.evaluate_with_sampler(circ, operator, {}, {}, sampler=sampler)
            val_tree = oe.OpTreeEvaluate.evaluate_tree_with_sampler(
                OpTreeExpectationValue(circ, operator), {}, sampler=sampler
            )
        mock_eval.assert_called()
        assert np.allclose(val, [3.14])
        assert np.allclose(val_tree, [3.14])

    def test_sampler_with_no_circuits_returns_empty_array(self):
        circ = OpTreeValue(0.0)
        operator = OpTreeValue(0.0)

        res = oe.OpTreeEvaluate.evaluate_with_sampler(
            circ, operator, dictionary_circuit={}, dictionary_operator={}, sampler=None
        )

        assert isinstance(res, np.ndarray)
        assert res.size == 0

    def test_sampler_no_circuits_but_operator_tree_is_still_evaluated(self):
        circ = OpTreeValue(0.0)
        operator = SparsePauliOp("Z")

        with patch.object(
            oe,
            "_evaluate_index_tree",
            side_effect=[np.array([1.0]), np.array([1.0])],
        ) as mock_eval:
            result = oe.OpTreeEvaluate.evaluate_with_sampler(
                circ,
                operator,
                {},
                {},
                sampler=object(),
            )

        assert mock_eval.call_count == 2
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1.0]))

    def test_sampler_operator_measurement_list_mismatch_raises(self):
        if QISKIT_SMALLER_2_0:
            from qiskit.primitives import Sampler as Sampler
        else:
            from qiskit.primitives import StatevectorSampler as Sampler

        circ = QuantumCircuit(1)
        operator = SparsePauliOp("Z")

        with patch.object(
            oe,
            "_build_measurement_list",
            return_value=([circ], [[1]]),
        ):
            with pytest.raises(
                ValueError, match="Operator measurement list does not match operator list!"
            ):
                oe.OpTreeEvaluate.evaluate_with_sampler(
                    circ,
                    operator,
                    {},
                    {},
                    sampler=Sampler(),
                )

    def test_sampler_multiple_operator_dict_single_circuit_dict_returns_array(self):
        circ = QuantumCircuit(1)
        operator = SparsePauliOp("Z")

        if QISKIT_SMALLER_2_0:
            from qiskit.primitives import Sampler as Sampler

        else:
            from qiskit.primitives import StatevectorSampler as Sampler

        result = oe.OpTreeEvaluate.evaluate_with_sampler(
            circ,
            operator,
            {},
            [{}, {}],
            sampler=Sampler(),
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert np.allclose(result, [1.0, 1.0])

    def test_evaluate_tree_with_sampler_multiple_dict_wraps_tree_in_optreelist(self):
        if QISKIT_SMALLER_2_0:
            from qiskit.primitives import Sampler as Sampler
        else:
            from qiskit.primitives import StatevectorSampler as Sampler

        with (
            patch.object(
                oe,
                "_build_expectation_list",
                side_effect=[
                    ([], [], [], [], OpTreeValue(1.0)),
                    ([], [], [], [], OpTreeValue(2.0)),
                ],
            ),
            patch.object(
                oe,
                "_evaluate_expectation_from_sampler",
                return_value=np.array([1.0, 2.0]),
            ),
            patch.object(
                oe,
                "_evaluate_index_tree",
                return_value=np.array([1.0, 2.0]),
            ) as mock_eval,
        ):
            result = oe.OpTreeEvaluate.evaluate_tree_with_sampler(
                expectation_tree=OpTreeExpectationValue(QuantumCircuit(1), SparsePauliOp("Z")),
                dictionary=[{}, {}],
                sampler=Sampler(),
            )

        assert isinstance(result, np.ndarray)
        assert isinstance(mock_eval.call_args.args[0], OpTreeList)
        assert mock_eval.call_args.args[0].children[0].value == 1.0
        assert mock_eval.call_args.args[0].children[1].value == 2.0

    def test_evaluate_with_estimator_no_circuits_returns_evaluated_tree(self):
        with (
            patch.object(
                oe,
                "_build_circuit_list",
                return_value=([], [], OpTreeContainer(0)),
            ),
            patch.object(
                oe,
                "_build_operator_list",
                return_value=([], OpTreeValue(7.5)),
            ),
            patch.object(oe, "_evaluate_index_tree", return_value=7.5) as mock_eval,
        ):
            result = oe.OpTreeEvaluate.evaluate_with_estimator(
                circuit=QuantumCircuit(1),
                operator=SparsePauliOp("Z"),
                dictionary_circuit={},
                dictionary_operator={},
                estimator=object(),
            )

        assert result == 7.5
        mock_eval.assert_called_once()
        assert mock_eval.call_args.args[1] == []


class TestOpTreeEvaluateHelpers:
    """Test class for edge cases in OpTree evaluation helper functions"""

    class TestCheckTreeForMatrixCompatibility:
        def test_check_tree_for_matrix_compatibility(self):
            compatibleList = OpTreeList(
                [
                    OpTreeList([OpTreeContainer(0), OpTreeContainer(1)]),
                    OpTreeList([OpTreeContainer(2), OpTreeContainer(3)]),
                ]
            )
            incompatibleList = OpTreeList(
                [
                    OpTreeContainer(0),
                    OpTreeList([OpTreeContainer(1), OpTreeContainer(2)]),
                ]
            )

            compatibleSum = OpTreeSum(
                [
                    OpTreeSum([OpTreeContainer(0), OpTreeContainer(1)]),
                    OpTreeSum([OpTreeContainer(2), OpTreeContainer(3)]),
                ]
            )
            incompatibleSum = OpTreeSum(
                [
                    OpTreeContainer(0),
                    OpTreeList([OpTreeContainer(1), OpTreeContainer(2)]),
                ]
            )

            assert oe._check_tree_for_matrix_compatibility(compatibleList) is True
            assert oe._check_tree_for_matrix_compatibility(incompatibleList) is False
            assert oe._check_tree_for_matrix_compatibility(compatibleSum) is True
            assert oe._check_tree_for_matrix_compatibility(incompatibleSum) is False
            assert oe._check_tree_for_matrix_compatibility(object()) is False

    class TestEvaluateIndexTree:
        def test_evaluate_index_tree_datatype_object(self):
            tree = OpTreeList([OpTreeContainer(0), OpTreeContainer(1)], [1.0, 2.0])

            result = oe._evaluate_index_tree(tree, np.array([10, 20]), datatype="object")

            assert result.dtype == object
            np.testing.assert_array_equal(result, [10, 40])

        def test_evaluate_index_tree_datatype_float(self):
            tree = OpTreeList([OpTreeContainer(0), OpTreeContainer(1)], [1.0, 2.0])

            result = oe._evaluate_index_tree(tree, np.array([10, 20]), datatype="float")

            assert result.dtype == float
            np.testing.assert_array_equal(result, [10.0, 40.0])

        def test_evaluate_index_tree_float_to_object_fallback(self):
            inner_short = OpTreeList([OpTreeContainer(0), OpTreeContainer(1)], [1.0, 1.0])

            inner_long = OpTreeList(
                [OpTreeContainer(0), OpTreeContainer(1), OpTreeContainer(2)], [1.0, 1.0, 1.0]
            )

            outer = OpTreeList([inner_short, inner_long], [1.0, 1.0])

            result = oe._evaluate_index_tree(outer, np.array([10, 20, 30]), datatype="float")

            assert result.dtype == object
            np.testing.assert_array_equal(result[0], [10, 20])
            np.testing.assert_array_equal(result[1], [10, 20, 30])

        def test_evaluate_index_tree_raises_for_invalid_datatype(self):
            tree = OpTreeList([OpTreeContainer(0), OpTreeContainer(1)], [1.0, 2.0])

            with pytest.raises(ValueError):
                oe._evaluate_index_tree(tree, np.array([10, 20]), datatype="invalid")

        def test_evaluate_index_tree_raises_for_non_float_factors(self):
            invalid_factors = OpTreeNodeBase([OpTreeValue(0), OpTreeValue(1)], [0, object()])
            with pytest.raises(ValueError):
                oe._evaluate_index_tree(invalid_factors, np.array([10, 20]))

        def test_evaluate_index_tree_raises_for_unknown_node(self):
            tree = UnknownNode([OpTreeContainer(0)], [1.0])

            with pytest.raises(ValueError, match="OpTreeNodeSum or a OpTreeNodeList"):
                oe._evaluate_index_tree(tree, np.array([10]), datatype="float")

            with pytest.raises(ValueError, match="OpTreeNode or a OpTreeLeafContainer"):
                oe._evaluate_index_tree(object(), np.array([10]), datatype="float")

    class TestBuildCircuitList:
        def test_build_circuit_list_raises_for_invalid_nodes_and_leafs(self):
            with pytest.raises(ValueError, match="OpTreeNodeSum or a OpTreeNodeList"):
                oe._build_circuit_list(UnknownNode([OpTreeValue(0)], [1.0]), dict())

            with pytest.raises(ValueError, match="CircuitTreeLeaf or a QuantumCircuit"):
                oe._build_circuit_list(OpTreeNodeBase([OpTreeContainer(0)], [1.0]), dict())

    class TestBuildOperatorList:
        def test_build_operator_list_binds_parameter_expression(self):
            param = Parameter("phi")

            element = OpTreeList([OpTreeValue(1)], [param])
            dictionary = {param: 2.0}

            operator_list, index_tree = oe._build_operator_list(element, dictionary)

            assert operator_list == []
            assert index_tree.factor[0] == 2.0

        def test_build_operator_list_binds_parameter_expression_arithmetic(self):
            param = Parameter("phi")

            element = OpTreeList([OpTreeValue(1)], [2 * param + 1])
            dictionary = {param: 3.0}

            operator_list, index_tree = oe._build_operator_list(element, dictionary)

            assert operator_list == []
            assert index_tree.factor[0] == 7.0

        def test_build_operator_list_raises_for_invalid_nodes_and_leafs(self):
            with pytest.raises(ValueError, match="OpTreeNodeSum or a OpTreeNodeList"):
                oe._build_operator_list(UnknownNode([OpTreeValue(0)], [1.0]), dict())

            with pytest.raises(ValueError, match="OpTreeLeafOperator or a SparsePauliOp"):
                oe._build_operator_list(OpTreeNodeBase([OpTreeContainer(0)], [1.0]), dict())

        def test_build_operator_list_raises_if_unassigned_parameters(self):
            param = Parameter("phi")
            unresolved = Parameter("psi")
            operator = SparsePauliOp("Z", coeffs=[param])

            element = OpTreeOperator(operator)
            dictionary = {param: unresolved}

            with pytest.raises(ValueError, match="Not all parameters are assigned"):
                oe._build_operator_list(element, dictionary)

        def test_build_operator_list_deduplicates_operators(self):
            operator = SparsePauliOp("Z")
            op_leaf = OpTreeOperator(operator)

            element = OpTreeList([op_leaf, op_leaf], [1.0, 1.0])

            operator_list, index_tree = oe._build_operator_list(element, dictionary={})

            assert len(operator_list) == 1
            assert index_tree.children[0].item == index_tree.children[1].item

        def test_build_operator_list_no_deduplication(self):
            operator = SparsePauliOp("Z")
            op_leaf = OpTreeOperator(operator)

            element = OpTreeList([op_leaf, op_leaf], [1.0, 1.0])

            operator_list, index_tree = oe._build_operator_list(
                element, dictionary={}, detect_operator_duplicates=False
            )

            assert len(operator_list) == 2
            assert index_tree.children[0].item == 0
            assert index_tree.children[1].item == 1

    class TestBuildMeasurementList:
        def test_build_measurement_list_duplicate_measured_operator_is_skipped(self):
            circuit = QuantumCircuit(1)
            circuit.measure_all()
            op = OpTreeMeasuredOperator(circuit, SparsePauliOp("Z"))
            element = OpTreeList([op, op], [1.0, 1.0])

            _, operator_measurement_list = oe._build_measurement_list(
                element, detect_operator_duplicates=True
            )

            assert sum(len(ops) for ops in operator_measurement_list) == 1

        def test_build_measurement_list_circuit_missing_measurement_raises(self):
            circuit = QuantumCircuit(1)
            op = OpTreeMeasuredOperator(circuit, SparsePauliOp("Z"))
            element = OpTreeList([op], [1.0])

            with pytest.raises(ValueError, match="Circuit missing a measurement"):
                oe._build_measurement_list(element)

        def test_build_measurement_list_optree_operator_leaf_uses_hashvalue(self):
            op = OpTreeOperator(SparsePauliOp("Z"))
            element = OpTreeList([op], [1.0])

            measurement_circuits, operator_measurement_list = oe._build_measurement_list(element)

            assert measurement_circuits == [None]
            assert operator_measurement_list == [[0]]

        def test_build_measurement_list_duplicate_optree_operator_is_skipped(self):
            op = OpTreeOperator(SparsePauliOp("Z"))
            element = OpTreeList([op, op], [1.0, 1.0])

            _, operator_measurement_list = oe._build_measurement_list(
                element, detect_operator_duplicates=True
            )

            assert sum(len(ops) for ops in operator_measurement_list) == 1

        def test_build_measurement_list_unknown_element_type_raises(self):
            with pytest.raises(ValueError, match="Wrong OpTree type detected"):
                oe._build_measurement_list(object())

    class TestBuildExpectationList:
        def test_build_expectation_list_binds_parameter_expression_in_factor(self):
            param = Parameter("phi")
            ev = make_mock_ev()

            element = OpTreeList([ev], [2 * param + 1])

            *_, index_tree = oe._build_expectation_list(element, dictionary={param: 3.0})

            assert index_tree.factor[0] == pytest.approx(7.0)

        def test_build_expectation_list_unknown_node_raises(self):
            ev = make_mock_ev()
            element = UnknownNode([ev], [1.0])

            with pytest.raises(ValueError, match="CircuitTreeSum or a CircuitTreeList"):
                oe._build_expectation_list(element, dictionary={})

        def test_build_expectation_list_optree_value_passthrough(self):
            value = OpTreeValue(42.0)
            element = OpTreeList([value], [1.0])

            circuit_list, operator_list, parameter_list, _, index_tree = (
                oe._build_expectation_list(element, dictionary={})
            )

            assert circuit_list == []
            assert operator_list == []
            assert isinstance(index_tree.children[0], OpTreeValue)
            assert index_tree.children[0].value == 42.0

        def test_build_expectation_list_deduplicates_expectation_values(self):
            ev = make_mock_ev(hashvalue="same_hash")
            element = OpTreeList([ev, ev], [1.0, 1.0])

            circuit_list, operator_list, *_ = oe._build_expectation_list(
                element, dictionary={}, detect_expectation_duplicates=True
            )

            assert len(operator_list) == 1
            assert len(circuit_list) == 1

        def test_build_expectation_list_raises_if_unassigned_operator_parameters(self):
            param = Parameter("phi")
            unresolved = Parameter("psi")

            operator = SparsePauliOp("Z", coeffs=[param])
            ev = MagicMock(spec=OpTreeExpectationValue)
            ev.operator = operator
            ev.circuit = MagicMock()
            ev.circuit.parameters = []
            ev._circuit = MagicMock()
            ev._circuit.hashvalue = "circ_hash"
            ev.hashvalue = "ev_hash"

            with pytest.raises(
                ValueError, match="Not all parameters are assigned in the operator"
            ):
                oe._build_expectation_list(ev, dictionary={param: unresolved})

        def test_build_expectation_list_unknown_element_raises(self):
            with pytest.raises(ValueError, match="OpTreeNode or a OpTreeLeafContainer"):
                oe._build_expectation_list(object(), dictionary={})

    class TestAddOffsetToTree:
        def test_add_offset_unknown_node_raises(self):
            element = UnknownNode([OpTreeContainer(0)], [1.0])

            with pytest.raises(ValueError, match="CircuitTreeSum or a CircuitTreeList"):
                oe._add_offset_to_tree(element, offset=1)

        def test_add_offset_non_integer_container_raises(self):
            element = OpTreeContainer(item="not_an_int")

            with pytest.raises(ValueError, match="Offset can only be added to integer leafs"):
                oe._add_offset_to_tree(element, offset=1)

        def test_add_offset_optree_value_passthrough(self):
            element = OpTreeValue(42.0)

            result = oe._add_offset_to_tree(element, offset=5)

            assert result is element

        def test_add_offset_unknown_element_raises(self):
            with pytest.raises(ValueError, match="OpTreeNode or a OpTreeLeafContainer"):
                oe._add_offset_to_tree(object(), offset=1)

    class TestEvaluateExpectationFromSampler:

        def test_evaluate_expectation_sampler_result_v1_branch(self):
            operator = [SparsePauliOp("Z")]
            quasi_dist = make_quasi_dist({"0": 0.6, "1": 0.4})
            results = make_mock_sampler_result([quasi_dist])

            with patch.object(
                oe, "_pauli_expval_with_variance", return_value=(np.array([1.0]), None)
            ):
                result = oe._evaluate_expectation_from_sampler(
                    operator, results, operator_measurement_list=[[0]]
                )

            assert isinstance(result, np.ndarray)

        def test_evaluate_expectation_none_measurement_list(self):
            operator = [SparsePauliOp("Z"), SparsePauliOp("I")]
            result_mock = make_mock_bitarray_result(exp_val=0.5)
            results = [result_mock, result_mock]

            result = oe._evaluate_expectation_from_sampler(
                operator, results, operator_measurement_list=None
            )

            assert len(result) == 4

        def test_evaluate_expectation_wrong_depth_raises(self):
            operator = [SparsePauliOp("Z")]
            results = [make_mock_bitarray_result()]

            with pytest.raises(ValueError, match="Wrong depth of operator_measurement_list"):
                oe._evaluate_expectation_from_sampler(
                    operator, results, operator_measurement_list=[0]
                )

        def test_evaluate_expectation_primitives_v2_basic(self):
            operator = [SparsePauliOp("Z")]
            result_mock = make_mock_bitarray_result(exp_val=0.8)
            results = [result_mock]

            result = oe._evaluate_expectation_from_sampler(
                operator, results, operator_measurement_list=[[0]]
            )

            np.testing.assert_array_almost_equal(result, [0.8])
            result_mock.expectation_values.assert_called_once_with(operator[0])

        def test_evaluate_expectation_primitives_v2_empty_observable_is_skipped(self):
            """'Empty observable was detected.' wird geschluckt, exp_val bleibt 0."""
            operator = [SparsePauliOp("Z")]
            result_mock = MagicMock()
            result_mock.expectation_values.side_effect = ValueError(
                "Empty observable was detected."
            )
            results = [result_mock]

            result = oe._evaluate_expectation_from_sampler(
                operator, results, operator_measurement_list=[[0]]
            )

            np.testing.assert_array_almost_equal(result, [0.0])

        def test_evaluate_expectation_primitives_v2_other_valueerror_reraises(self):
            """Andere ValueErrors werden nicht geschluckt."""
            operator = [SparsePauliOp("Z")]
            result_mock = MagicMock()
            result_mock.expectation_values.side_effect = ValueError("Something else went wrong")
            results = [result_mock]

            with pytest.raises(ValueError, match="Something else went wrong"):
                oe._evaluate_expectation_from_sampler(
                    operator, results, operator_measurement_list=[[0]]
                )

        def test_evaluate_expectation_v1_uses_quasi_dists(self):
            operator = [SparsePauliOp("Z")]
            quasi_dist = make_quasi_dist({"0": 0.7, "1": 0.3})
            results = make_mock_sampler_result([quasi_dist])

            with patch.object(
                oe, "_pauli_expval_with_variance", return_value=(np.array([0.4]), None)
            ) as mock_pauli:
                result = oe._evaluate_expectation_from_sampler(
                    operator, results, operator_measurement_list=[[0]]
                )

            mock_pauli.assert_called_once()
            # 0.4 * coeff (1.0+0j) = 0.4
            np.testing.assert_equal(result, [0.4])

    class TestTransformOperatorToZBasis:

        def test_transform_to_zbasis_unwraps_optree_operator(self):
            op = SparsePauliOp("Z")
            op_leaf = OpTreeOperator(op)

            result_wrapped = oe._transform_operator_to_zbasis(op_leaf)
            result_direct = oe._transform_operator_to_zbasis(op)

            assert result_wrapped == result_direct

        def test_transform_to_zbasis_no_changes_needed(self):
            op = SparsePauliOp.from_list([("ZI", 1.0), ("IZ", 0.5)])

            result = oe._transform_operator_to_zbasis(op)

            assert result == op

        def test_transform_to_zbasis_no_grouping_multiple_terms(self):
            op = SparsePauliOp.from_list([("X", 1.0), ("Y", 0.5)])

            result = oe._transform_operator_to_zbasis(op, abelian_grouping=False)

            assert isinstance(result, OpTreeSum)
            assert len(result.children) == 2
            assert all(isinstance(c, OpTreeMeasuredOperator) for c in result.children)

        def test_transform_to_zbasis_no_grouping_single_term(self):
            op = SparsePauliOp.from_list([("X", 1.0)])

            result = oe._transform_operator_to_zbasis(op, abelian_grouping=False)

            assert isinstance(result, OpTreeMeasuredOperator)

        def test_transform_to_zbasis_single_group_returns_leaf(self):

            op = SparsePauliOp.from_list([("X", 1.0), ("X", 0.5)])

            result = oe._transform_operator_to_zbasis(op, abelian_grouping=True)

            assert isinstance(result, OpTreeMeasuredOperator)

        def test_transform_to_zbasis_invalid_type_raises(self):
            with pytest.raises(ValueError, match="Wrong type of Optree Element"):
                oe.OpTreeEvaluate.transform_to_zbasis(123)

        def test_transform_to_zbasis_unknown_node_raises(self):
            with pytest.raises(
                ValueError, match="element must be a CircuitTreeSum or a CircuitTreeList"
            ):
                oe.OpTreeEvaluate.transform_to_zbasis(UnknownNode())

        def test_transform_to_zbasis_sparsepauliop_calls_helper(self):
            op = OpTreeOperator(SparsePauliOp("X"))

            with patch.object(
                oe, "_transform_operator_to_zbasis", return_value="ZB_OP"
            ) as mock_transform:
                result = oe.OpTreeEvaluate.transform_to_zbasis(op, abelian_grouping=False)

            assert result == "ZB_OP"
            mock_transform.assert_called

    class TestMeasureAllUnmeasured:

        def test_measure_all_unmeasured_conditional_gate_reorders_clbits(self):
            circ = QuantumCircuit(2, 1)
            circ.measure(0, 0)
            # use a single Clbit for the condition to avoid ClassicalRegister objects
            # in the measure arguments which some qiskit versions expose as registers
            with circ.if_test((circ.clbits[0], 1)):
                circ.x(1)

            result = oe._measure_all_unmeasured(circ)

            assert isinstance(result, QuantumCircuit)
            assert result.num_clbits > 0

        def test_measure_all_unmeasured_duplicate_measurement_raises(self):
            circ = QuantumCircuit(2, 2)
            circ.measure(0, 0)
            circ.measure(0, 1)

            with pytest.raises(ValueError, match="multiple measurements on the same qubit"):
                oe._measure_all_unmeasured(circ)

        def test_measure_all_unmeasured_no_existing_measurements_returns_early(self):
            circ = QuantumCircuit(2)
            circ.add_register(ClassicalRegister(1, "c"))
            circ.x(0)

            result = oe._measure_all_unmeasured(circ)

            assert isinstance(result, QuantumCircuit)
            measured_qubits = [
                circ.find_bit(qargs[0])[0]
                for inst, qargs, cargs in result.data
                if inst.name == "measure"
            ]
            assert set(measured_qubits) == {0, 1}
