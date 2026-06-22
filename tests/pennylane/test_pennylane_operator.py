"""
Test suite for PennyLane operator conversion.

This module tests the PennyLaneOperator class which converts Qiskit quantum
operators (SparsePauliOp) to PennyLane format, including:
- Non-parametric operators (constant coefficients)
- Parametric operators (parameter vectors)
- Parameter expressions
- Operator properties and methods
"""

from typing import Callable
from unittest.mock import MagicMock

import numpy as np
import pennylane as qml
import pytest
from qiskit.quantum_info import SparsePauliOp

from executor import QuantumOperator
from executor.base.operator_base import QuantumOperatorBase
from executor.parameters import Parameters
from executor.pennylane.pennylane_operator import PennyLaneOperator


def _make_quantum_operator_base(qiskit_op):
    """Helper function to create a MagicMock that simulates a QuantumOperatorBase with a given qiskit_operator."""
    mock = MagicMock(spec=QuantumOperatorBase)
    mock.qiskit_operator = qiskit_op
    return mock


class TestPennyLaneOperator:
    """Test suite for PennyLane operator conversion."""

    # Non-Parametric Operator Tests

    def test_init_single_operator(self):
        """isinstance(operator, QuantumOperatorBase) -> True."""
        sparse_op = SparsePauliOp(["ZZ", "XX"], coeffs=[1.0, 0.5])
        op_base = _make_quantum_operator_base(sparse_op)
        pl_op = PennyLaneOperator(op_base)

        assert pl_op._qiskit_operator is sparse_op
        assert pl_op._num_qubits == 2
        assert isinstance(pl_op._pennylane_words, list)
        assert len(pl_op._pennylane_words) == 2
        assert all(not isinstance(w, list) for w in pl_op._pennylane_words)

    def test_init_list_of_operators(self):
        """Test that initializing with a list of QuantumOperatorBase objects works correctly."""
        sparse_op1 = SparsePauliOp(["ZZ"], coeffs=[1.0])
        sparse_op2 = SparsePauliOp(["XX"], coeffs=[0.5])
        op_base1 = _make_quantum_operator_base(sparse_op1)
        op_base2 = _make_quantum_operator_base(sparse_op2)

        pl_op = PennyLaneOperator([op_base1, op_base2])

        assert pl_op._qiskit_operator == [sparse_op1, sparse_op2]
        assert pl_op._num_qubits == 2
        assert isinstance(pl_op._pennylane_words, list)
        assert len(pl_op._pennylane_words) == 2

    def test_init_list_with_invalid_element_raises(self):
        """Test that initializing with a list containing an invalid element raises an error."""
        sparse_op1 = SparsePauliOp(["ZZ"], coeffs=[1.0])
        op_base1 = _make_quantum_operator_base(sparse_op1)
        invalid_element = "not_a_quantum_operator"

        with pytest.raises(ValueError, match="Unsupported operator type"):
            PennyLaneOperator([op_base1, invalid_element])

    def test_init_unsupported_type_raises(self):
        """Test that initializing with an unsupported type raises an error."""
        with pytest.raises(ValueError, match="Unsupported operator type"):
            PennyLaneOperator("not_an_operator")

    def test_single_qubit_z_operator(self):
        """Test single-qubit Z operator."""
        operator = QuantumOperator(["Z"], [1.0])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0
        assert len(plo.parameter_dimensions) == 0

    def test_single_qubit_x_operator(self):
        """Test single-qubit X operator."""
        operator = QuantumOperator(["X"], [1.0])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_single_qubit_y_operator(self):
        """Test single-qubit Y operator."""
        operator = QuantumOperator(["Y"], [1.0])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_two_qubit_zi_iz_operator(self):
        """Test two-qubit ZI + IZ operator."""
        operator = QuantumOperator(["ZI", "IZ"], [1.0, 1.0])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_two_qubit_xi_ix_operator(self):
        """Test two-qubit XI + IX operator."""
        operator = QuantumOperator(["XI", "IX"], [1.0, 1.0])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_two_qubit_zz_operator(self):
        """Test two-qubit ZZ operator."""
        operator = QuantumOperator(["ZZ"], [1.0])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_two_qubit_xx_operator(self):
        """Test two-qubit XX operator."""
        operator = QuantumOperator(["XX"], [1.0])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_multi_term_operator(self):
        """Test operator with multiple Pauli terms."""
        operator = QuantumOperator(["XX", "YY", "ZZ"], [0.5, 0.3, 0.2])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_identity_operator(self):
        """Test identity operator."""
        operator = QuantumOperator(["II"], [1.0])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_three_qubit_operator(self):
        """Test three-qubit operator."""
        operator = QuantumOperator(["ZZZ", "XXX"], [1.0, 0.5])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    @pytest.mark.parametrize("coeff", [0.0, 0.5, 1.0, -1.0, 2.5])
    def test_operator_with_various_coefficients(self, coeff):
        """Test operator with various coefficient values."""
        operator = QuantumOperator(["Z"], [coeff])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    # Parametric Operator Tests

    def test_operator_single_parameter(self):
        """Test operator with a single parameter."""
        theta = Parameters("theta", 1)
        operator = QuantumOperator(["Z"], [theta[0]])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert "theta" in plo.parameter_names
        assert plo.parameter_dimensions["theta"] == 1

    def test_operator_two_parameters_same_vector(self):
        """Test operator with two parameters from the same vector."""
        theta = Parameters("theta", 2)
        operator = QuantumOperator(["ZI", "IZ"], [theta[0], theta[1]])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert "theta" in plo.parameter_names
        assert plo.parameter_dimensions["theta"] == 2

    def test_operator_three_parameters_same_vector(self):
        """Test operator with three parameters from the same vector."""
        alpha = Parameters("alpha", 3)
        operator = QuantumOperator(["ZI", "IZ", "ZZ"], [alpha[0], alpha[1], alpha[2]])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert "alpha" in plo.parameter_names
        assert plo.parameter_dimensions["alpha"] == 3

    def test_operator_multiple_parameter_vectors(self):
        """Test operator with multiple different parameter vectors."""
        pop1 = Parameters("pop1", 1)
        pop2 = Parameters("pop2", 1)
        operator = QuantumOperator(["ZI", "IZ"], [pop1[0], pop2[0]])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert "pop1" in plo.parameter_names
        assert "pop2" in plo.parameter_names
        assert plo.parameter_dimensions["pop1"] == 1
        assert plo.parameter_dimensions["pop2"] == 1

    def test_operator_three_parameter_vectors(self):
        """Test operator with three different parameter vectors."""
        a = Parameters("a", 1)
        b = Parameters("b", 1)
        c = Parameters("c", 1)
        operator = QuantumOperator(["X", "Y", "Z"], [a[0], b[0], c[0]])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert "a" in plo.parameter_names
        assert "b" in plo.parameter_names
        assert "c" in plo.parameter_names

    # Parameter Expression Tests

    def test_operator_with_parameter_multiplication(self):
        """Test operator with parameter expression: 2 * theta[0]."""
        theta = Parameters("theta", 1)
        operator = QuantumOperator(["Z"], [2 * theta[0]])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert "theta" in plo.parameter_names

    def test_operator_with_parameter_expressions(self):
        """Test operator with multiple parameter expressions."""
        theta = Parameters("theta", 2)
        operator = QuantumOperator(["ZI", "IZ"], [theta[0] * 2, theta[1] * 0.5])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert "theta" in plo.parameter_names
        assert plo.parameter_dimensions["theta"] == 2

    def test_operator_with_parameter_addition(self):
        """Test operator with parameter addition expression."""
        theta = Parameters("theta", 2)
        operator = QuantumOperator(["ZI", "IZ"], [theta[0] + theta[1], theta[0]])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert "theta" in plo.parameter_names


class TestPennyLaneOperatorPropertiesAndMethods:

    def test_parameter_names_property_empty(self):
        """Test parameter_names property for non-parametric operator."""
        operator = QuantumOperator(["Z"], [1.0])
        plo = PennyLaneOperator(operator)

        assert isinstance(plo.parameter_names, list)
        assert len(plo.parameter_names) == 0

    def test_parameter_names_property_with_params(self):
        """Test parameter_names property for parametric operator."""
        theta = Parameters("theta", 2)
        operator = QuantumOperator(["ZI", "IZ"], [theta[0], theta[1]])
        plo = PennyLaneOperator(operator)

        assert isinstance(plo.parameter_names, list)
        assert "theta" in plo.parameter_names

    def test_parameter_dimensions_property(self):
        """Test parameter_dimensions property."""
        theta = Parameters("theta", 3)
        operator = QuantumOperator(["X", "Y", "Z"], [theta[0], theta[1], theta[2]])
        plo = PennyLaneOperator(operator)

        assert isinstance(plo.parameter_dimensions, dict)
        assert plo.parameter_dimensions["theta"] == 3

    def test_hash_property(self):
        """Test that hash property returns a valid integer."""
        operator = QuantumOperator(["ZI", "IZ"], [1.0, 1.0])
        plo = PennyLaneOperator(operator)

        hash_value = plo.hash
        assert isinstance(hash_value, int)

    def test_hash_consistency(self):
        """Test that hash remains consistent for the same operator."""
        operator = QuantumOperator(["ZI", "IZ"], [1.0, 1.0])
        plo = PennyLaneOperator(operator)

        hash1 = plo.hash
        hash2 = plo.hash
        assert hash1 == hash2


class TestBuildPennylaneObservable:

    def test_single_op_no_params_nonempty_words(self):
        """Test that build_pennylane_observable returns a valid function for a non-parametric operator with non-empty Pauli words."""
        sparse_op = SparsePauliOp(["ZZ", "XX"], coeffs=[1.0, 0.5])
        op_base = _make_quantum_operator_base(sparse_op)
        pl_op = PennyLaneOperator(op_base)

        observable_fn = pl_op.build_pennylane_observable()

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            return observable_fn()

        result = circuit()
        assert isinstance(observable_fn, Callable)
        assert result is not None

    def test_single_op_no_params_empty_words(self):
        """Test that build_pennylane_observable returns a function that evaluates to 0 for a non-parametric operator with empty Pauli words."""
        sparse_op = SparsePauliOp(["II"], coeffs=[1.0])
        op_base = _make_quantum_operator_base(sparse_op)
        pl_op = PennyLaneOperator(op_base)
        pl_op._pennylane_words = []

        observable_fn = pl_op.build_pennylane_observable()
        result = observable_fn()

        assert isinstance(observable_fn, Callable)
        assert result == 0.0

    def test_single_op_with_params(self):
        """Test that build_pennylane_observable returns a valid function for a parametric operator."""
        theta_vec = Parameters("theta", 1)
        sparse_op = SparsePauliOp(["ZZ", "XX"], coeffs=[theta_vec[0], 0.5])
        op_base = _make_quantum_operator_base(sparse_op)
        pl_op = PennyLaneOperator(op_base)

        observable_fn = pl_op.build_pennylane_observable()

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(param):
            return observable_fn([1.0])

        result = circuit([1.0])
        assert isinstance(observable_fn, Callable)
        assert result is not None

    # --- List branch ---

    def test_list_no_params_nonempty_words(self):
        """Test that build_pennylane_observable returns a valid function for a list of non-parametric operators with non-empty Pauli words."""
        sparse_op1 = SparsePauliOp(["ZZ"], coeffs=[1.0])
        sparse_op2 = SparsePauliOp(["XX"], coeffs=[0.5])
        op_base1 = _make_quantum_operator_base(sparse_op1)
        op_base2 = _make_quantum_operator_base(sparse_op2)

        pl_op = PennyLaneOperator([op_base1, op_base2])

        observable_fn = pl_op.build_pennylane_observable()

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            return observable_fn()

        result = circuit()
        assert isinstance(observable_fn, Callable)
        assert result.shape == (2,)

    def test_list_no_params_empty_words_for_one_operator(self):
        """Test that build_pennylane_observable returns a function that evaluates to 0 for the operator with empty Pauli words in a list of non-parametric operators."""
        sparse_op1 = SparsePauliOp(["ZZ"], coeffs=[1.0])
        sparse_op2 = SparsePauliOp(["XX"], coeffs=[0.5])
        op_base1 = _make_quantum_operator_base(sparse_op1)
        op_base2 = _make_quantum_operator_base(sparse_op2)

        pl_op = PennyLaneOperator([op_base1, op_base2])
        pl_op._pennylane_words[1] = []

        observable_fn = pl_op.build_pennylane_observable()

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            return observable_fn()

        result = circuit()

        result_array = np.asarray(result)
        assert isinstance(observable_fn, Callable)
        assert result_array.size == 1
        assert float(result_array.flatten()[0]) == 1.0

    def test_list_with_params(self):
        """Test that build_pennylane_observable returns a valid function for a list of parametric operators."""
        theta_vec = Parameters("theta", 1)
        sparse_op1 = SparsePauliOp(["ZZ"], coeffs=[theta_vec[0]])
        sparse_op2 = SparsePauliOp(["XX"], coeffs=[0.5])
        op_base1 = _make_quantum_operator_base(sparse_op1)
        op_base2 = _make_quantum_operator_base(sparse_op2)

        pl_op = PennyLaneOperator([op_base1, op_base2])

        observable_fn = pl_op.build_pennylane_observable()

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(param):
            return observable_fn([1.0])

        result = circuit([1.0])
        assert isinstance(observable_fn, Callable)
        assert result.shape == (2,)

    def test_list_mixed_params_and_no_params_per_op(self):
        """Test that build_pennylane_observable returns a valid function for a list of mixed parametric and non-parametric operators."""
        theta_vec = Parameters("theta", 1)
        sparse_op1 = SparsePauliOp(["ZZ"], coeffs=[theta_vec[0]])
        sparse_op2 = SparsePauliOp(["XX"], coeffs=[0.5])
        op_base1 = _make_quantum_operator_base(sparse_op1)
        op_base2 = _make_quantum_operator_base(sparse_op2)

        pl_op = PennyLaneOperator([op_base1, op_base2])

        observable_fn = pl_op.build_pennylane_observable()

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(param):
            return observable_fn([1.0])

        result = circuit([1.0])
        assert isinstance(observable_fn, Callable)
        assert result.shape == (2,)
