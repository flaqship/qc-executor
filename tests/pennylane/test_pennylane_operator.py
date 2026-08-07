"""
Test suite for PennyLane operator conversion.

This module tests the PennyLaneOperator class, which compiles the shared sparse
Pauli representation into PennyLane Pauli words and coefficient callables.  It
used to read a Qiskit ``SparsePauliOp``, so these tests built their inputs from
Qiskit objects; they now use real operators throughout.  Coverage includes:
- Non-parametric operators (constant coefficients)
- Parametric operators (parameter vectors)
- Parameter expressions
- Operator properties and methods
"""

from typing import Callable

import numpy as np
import pennylane as qml
import pytest

from qc_executor import QuantumOperator
from qc_executor.parameters import Parameters
from qc_executor.pennylane.pennylane_operator import PennyLaneOperator


class TestPennyLaneOperator:
    """Test suite for PennyLane operator conversion."""

    # Non-Parametric Operator Tests

    def test_init_single_operator(self):
        """A single operator compiles to one flat list of Pauli words."""
        pl_op = PennyLaneOperator(QuantumOperator(["ZZ", "XX"], [1.0, 0.5]))

        assert pl_op._islist is False
        assert pl_op.num_qubits == 2
        assert isinstance(pl_op._pennylane_words, list)
        assert len(pl_op._pennylane_words) == 2
        assert all(not isinstance(w, list) for w in pl_op._pennylane_words)

    def test_init_list_of_operators(self):
        """A list of operators compiles to one word list per operator."""
        pl_op = PennyLaneOperator([QuantumOperator(["ZZ"], [1.0]), QuantumOperator(["XX"], [0.5])])

        assert pl_op._islist is True
        assert pl_op.num_qubits == 2
        assert len(pl_op._pennylane_words) == 2
        assert all(len(words) == 1 for words in pl_op._pennylane_words)

    def test_init_list_with_invalid_element_raises(self):
        """Test that initializing with a list containing an invalid element raises an error."""
        with pytest.raises(ValueError, match="Unsupported operator type"):
            PennyLaneOperator([QuantumOperator(["ZZ"], [1.0]), "not_a_quantum_operator"])

    def test_init_empty_list_raises(self):
        """An empty list has no width to report."""
        with pytest.raises(ValueError, match="Unsupported operator type"):
            PennyLaneOperator([])

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

    def test_hash_is_derived_from_the_operator_content(self):
        """The cache key comes from the representation's fingerprint."""
        same = PennyLaneOperator(QuantumOperator(["ZI", "IZ"], [1.0, 1.0]))
        also_same = PennyLaneOperator(QuantumOperator(["ZI", "IZ"], [1.0, 1.0]))
        different = PennyLaneOperator(QuantumOperator(["ZI", "IZ"], [1.0, 2.0]))

        assert same.hash == also_same.hash
        assert same.hash != different.hash

    def test_hash_consistency(self):
        """Test that hash remains consistent for the same operator."""
        operator = QuantumOperator(["ZI", "IZ"], [1.0, 1.0])
        plo = PennyLaneOperator(operator)

        hash1 = plo.hash
        hash2 = plo.hash
        assert hash1 == hash2


class TestBuildPennylaneObservable:
    """The observable callable, checked by value.

    A Bell state makes every reference exact: ``<ZZ> = 1`` and ``<XX> = 1``, so
    the coefficients are the only thing left that can be wrong.  Asserting the
    number rather than "not None" is what catches coefficients being dropped.
    """

    @staticmethod
    def _evaluate(observable_fn, *args):
        """Measure the observable on a two-qubit Bell state."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return observable_fn(*args)

        return circuit()

    def test_single_op_no_params(self):
        pl_op = PennyLaneOperator(QuantumOperator(["ZZ", "XX"], [1.0, 0.5]))

        observable_fn = pl_op.build_pennylane_observable()

        assert isinstance(observable_fn, Callable)
        assert float(self._evaluate(observable_fn)) == pytest.approx(1.5, abs=1e-8)

    def test_coefficients_are_weights_not_decoration(self):
        """Doubling a coefficient must double the result."""
        single = PennyLaneOperator(QuantumOperator(["ZZ"], [1.0])).build_pennylane_observable()
        double = PennyLaneOperator(QuantumOperator(["ZZ"], [2.0])).build_pennylane_observable()

        assert float(self._evaluate(double)) == pytest.approx(
            2 * float(self._evaluate(single)), abs=1e-8
        )

    def test_single_op_no_params_empty_words(self):
        """An operator with no Pauli words measures nothing."""
        pl_op = PennyLaneOperator(QuantumOperator(["II"], [1.0]))
        pl_op._pennylane_words = []

        observable_fn = pl_op.build_pennylane_observable()

        assert observable_fn() == 0.0

    def test_single_op_with_params(self):
        theta = Parameters("theta", 1)
        pl_op = PennyLaneOperator(QuantumOperator(["ZZ", "XX"], [theta[0], 0.5]))

        observable_fn = pl_op.build_pennylane_observable()

        assert float(self._evaluate(observable_fn, [2.0])) == pytest.approx(2.5, abs=1e-8)

    def test_a_symbolic_expression_is_evaluated(self):
        theta = Parameters("theta", 2)
        pl_op = PennyLaneOperator(QuantumOperator(["ZZ"], [2 * theta[0] + theta[1] * theta[0]]))

        observable_fn = pl_op.build_pennylane_observable()

        # 2*0.5 + 0.25*0.5 = 1.125, times <ZZ> = 1.
        assert float(self._evaluate(observable_fn, [0.5, 0.25])) == pytest.approx(1.125, abs=1e-8)

    # --- List branch ---

    def test_list_no_params(self):
        pl_op = PennyLaneOperator([QuantumOperator(["ZZ"], [1.0]), QuantumOperator(["XX"], [0.5])])

        observable_fn = pl_op.build_pennylane_observable()
        result = np.asarray(self._evaluate(observable_fn))

        assert result.shape == (2,)
        assert result == pytest.approx([1.0, 0.5], abs=1e-8)

    def test_list_empty_words_for_one_operator(self):
        """An empty operator in a list contributes a plain zero."""
        pl_op = PennyLaneOperator([QuantumOperator(["ZZ"], [1.0]), QuantumOperator(["XX"], [0.5])])
        pl_op._pennylane_words[1] = []

        observable_fn = pl_op.build_pennylane_observable()
        result = np.asarray(self._evaluate(observable_fn))

        assert float(result.flatten()[0]) == pytest.approx(1.0, abs=1e-8)

    def test_list_with_params(self):
        theta = Parameters("theta", 1)
        pl_op = PennyLaneOperator(
            [QuantumOperator(["ZZ"], [theta[0]]), QuantumOperator(["XX"], [0.5])]
        )

        observable_fn = pl_op.build_pennylane_observable()
        result = np.asarray(self._evaluate(observable_fn, [3.0]))

        assert result.shape == (2,)
        assert result == pytest.approx([3.0, 0.5], abs=1e-8)


class TestUnsupportedCoefficients:
    def test_an_imaginary_coefficient_is_rejected(self):
        """PennyLane observables must be Hermitian, so the weight has to be real."""
        with pytest.raises(ValueError, match="Imaginary part"):
            PennyLaneOperator(QuantumOperator(["Z"], [1.0 + 2.0j]))
