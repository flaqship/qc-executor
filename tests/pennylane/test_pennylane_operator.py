"""
Test suite for PennyLane operator conversion.

This module tests the PennyLaneOperator class which converts Qiskit quantum
operators (SparsePauliOp) to PennyLane format, including:
- Non-parametric operators (constant coefficients)
- Parametric operators (parameter vectors)
- Parameter expressions
- Operator properties and methods
"""

import pytest

from qc_executor.abstraction import AbstractQuantumOperator, ParameterVector
from qc_executor.pennylane.pennylane_operator import PennyLaneOperator


class TestPennyLaneOperator:
    """Test suite for PennyLane operator conversion."""

    # Non-Parametric Operator Tests

    def test_single_qubit_z_operator(self):
        """Test single-qubit Z operator."""
        operator = AbstractQuantumOperator(["Z"], [1.0])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0
        assert len(plo.parameter_dimensions) == 0

    def test_single_qubit_x_operator(self):
        """Test single-qubit X operator."""
        operator = AbstractQuantumOperator(["X"], [1.0])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_single_qubit_y_operator(self):
        """Test single-qubit Y operator."""
        operator = AbstractQuantumOperator(["Y"], [1.0])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_two_qubit_zi_iz_operator(self):
        """Test two-qubit ZI + IZ operator."""
        operator = AbstractQuantumOperator(["ZI", "IZ"], [1.0, 1.0])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_two_qubit_xi_ix_operator(self):
        """Test two-qubit XI + IX operator."""
        operator = AbstractQuantumOperator(["XI", "IX"], [1.0, 1.0])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_two_qubit_zz_operator(self):
        """Test two-qubit ZZ operator."""
        operator = AbstractQuantumOperator(["ZZ"], [1.0])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_two_qubit_xx_operator(self):
        """Test two-qubit XX operator."""
        operator = AbstractQuantumOperator(["XX"], [1.0])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_multi_term_operator(self):
        """Test operator with multiple Pauli terms."""
        operator = AbstractQuantumOperator(["XX", "YY", "ZZ"], [0.5, 0.3, 0.2])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_identity_operator(self):
        """Test identity operator."""
        operator = AbstractQuantumOperator(["II"], [1.0])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_three_qubit_operator(self):
        """Test three-qubit operator."""
        operator = AbstractQuantumOperator(["ZZZ", "XXX"], [1.0, 0.5])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    @pytest.mark.parametrize("coeff", [0.0, 0.5, 1.0, -1.0, 2.5])
    def test_operator_with_various_coefficients(self, coeff):
        """Test operator with various coefficient values."""
        operator = AbstractQuantumOperator(["Z"], [coeff])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    # Parametric Operator Tests

    def test_operator_single_parameter(self):
        """Test operator with a single parameter."""
        theta = ParameterVector("theta", 1)
        operator = AbstractQuantumOperator(["Z"], [theta[0]])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert "theta" in plo.parameter_names
        assert plo.parameter_dimensions["theta"] == 1

    def test_operator_two_parameters_same_vector(self):
        """Test operator with two parameters from the same vector."""
        theta = ParameterVector("theta", 2)
        operator = AbstractQuantumOperator(["ZI", "IZ"], [theta[0], theta[1]])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert "theta" in plo.parameter_names
        assert plo.parameter_dimensions["theta"] == 2

    def test_operator_three_parameters_same_vector(self):
        """Test operator with three parameters from the same vector."""
        alpha = ParameterVector("alpha", 3)
        operator = AbstractQuantumOperator(["ZI", "IZ", "ZZ"], [alpha[0], alpha[1], alpha[2]])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert "alpha" in plo.parameter_names
        assert plo.parameter_dimensions["alpha"] == 3

    def test_operator_multiple_parameter_vectors(self):
        """Test operator with multiple different parameter vectors."""
        pop1 = ParameterVector("pop1", 1)
        pop2 = ParameterVector("pop2", 1)
        operator = AbstractQuantumOperator(["ZI", "IZ"], [pop1[0], pop2[0]])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert "pop1" in plo.parameter_names
        assert "pop2" in plo.parameter_names
        assert plo.parameter_dimensions["pop1"] == 1
        assert plo.parameter_dimensions["pop2"] == 1

    def test_operator_three_parameter_vectors(self):
        """Test operator with three different parameter vectors."""
        a = ParameterVector("a", 1)
        b = ParameterVector("b", 1)
        c = ParameterVector("c", 1)
        operator = AbstractQuantumOperator(["X", "Y", "Z"], [a[0], b[0], c[0]])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert "a" in plo.parameter_names
        assert "b" in plo.parameter_names
        assert "c" in plo.parameter_names

    # Parameter Expression Tests

    def test_operator_with_parameter_multiplication(self):
        """Test operator with parameter expression: 2 * theta[0]."""
        theta = ParameterVector("theta", 1)
        operator = AbstractQuantumOperator(["Z"], [2 * theta[0]])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert "theta" in plo.parameter_names

    def test_operator_with_parameter_expressions(self):
        """Test operator with multiple parameter expressions."""
        theta = ParameterVector("theta", 2)
        operator = AbstractQuantumOperator(["ZI", "IZ"], [theta[0] * 2, theta[1] * 0.5])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert "theta" in plo.parameter_names
        assert plo.parameter_dimensions["theta"] == 2

    def test_operator_with_parameter_addition(self):
        """Test operator with parameter addition expression."""
        theta = ParameterVector("theta", 2)
        operator = AbstractQuantumOperator(["ZI", "IZ"], [theta[0] + theta[1], theta[0]])
        plo = PennyLaneOperator(operator)

        assert plo is not None
        assert "theta" in plo.parameter_names

    # Property and Method Tests

    def test_parameter_names_property_empty(self):
        """Test parameter_names property for non-parametric operator."""
        operator = AbstractQuantumOperator(["Z"], [1.0])
        plo = PennyLaneOperator(operator)

        assert isinstance(plo.parameter_names, list)
        assert len(plo.parameter_names) == 0

    def test_parameter_names_property_with_params(self):
        """Test parameter_names property for parametric operator."""
        theta = ParameterVector("theta", 2)
        operator = AbstractQuantumOperator(["ZI", "IZ"], [theta[0], theta[1]])
        plo = PennyLaneOperator(operator)

        assert isinstance(plo.parameter_names, list)
        assert "theta" in plo.parameter_names

    def test_parameter_dimensions_property(self):
        """Test parameter_dimensions property."""
        theta = ParameterVector("theta", 3)
        operator = AbstractQuantumOperator(["X", "Y", "Z"], [theta[0], theta[1], theta[2]])
        plo = PennyLaneOperator(operator)

        assert isinstance(plo.parameter_dimensions, dict)
        assert plo.parameter_dimensions["theta"] == 3

    def test_hash_property(self):
        """Test that hash property returns a valid integer."""
        operator = AbstractQuantumOperator(["ZI", "IZ"], [1.0, 1.0])
        plo = PennyLaneOperator(operator)

        hash_value = plo.hash
        assert isinstance(hash_value, int)

    def test_hash_consistency(self):
        """Test that hash remains consistent for the same operator."""
        operator = AbstractQuantumOperator(["ZI", "IZ"], [1.0, 1.0])
        plo = PennyLaneOperator(operator)

        hash1 = plo.hash
        hash2 = plo.hash
        assert hash1 == hash2

    def test_build_pennylane_observable_method(self):
        """Test that build_pennylane_observable() returns callable."""
        operator = AbstractQuantumOperator(["ZI", "IZ"], [1.0, 1.0])
        plo = PennyLaneOperator(operator)

        pennylane_obs = plo.build_pennylane_observable()
        assert callable(pennylane_obs)

    def test_build_pennylane_observable_with_params(self):
        """Test that build_pennylane_observable() works with parameters."""
        theta = ParameterVector("theta", 2)
        operator = AbstractQuantumOperator(["ZI", "IZ"], [theta[0], theta[1]])
        plo = PennyLaneOperator(operator)

        pennylane_obs = plo.build_pennylane_observable()
        assert callable(pennylane_obs)
