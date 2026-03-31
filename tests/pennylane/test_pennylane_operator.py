"""
Test suite for PennyLane observable conversion.

This module tests the PennyLaneOperator class which converts Qiskit quantum
operators (SparsePauliOp) to PennyLane format, including:
- Non-parametric observables (constant coefficients)
- Parametric observables (parameter vectors)
- Parameter expressions
- Observable properties and methods
"""

import pytest
from qiskit.circuit import ParameterVector

from executor import QuantumOperator
from executor.pennylane.pennylane_operator import PennyLaneOperator


class TestPennyLaneOperator:
    """Test suite for PennyLane observable conversion."""

    # Non-Parametric Observable Tests

    def test_single_qubit_z_observable(self):
        """Test single-qubit Z observable."""
        op = QuantumOperator(["Z"], [1.0])
        plo = PennyLaneOperator(op)

        assert plo is not None
        assert len(plo.parameter_names) == 0
        assert len(plo.parameter_dimensions) == 0

    def test_single_qubit_x_observable(self):
        """Test single-qubit X observable."""
        op = QuantumOperator(["X"], [1.0])
        plo = PennyLaneOperator(op)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_single_qubit_y_observable(self):
        """Test single-qubit Y observable."""
        op = QuantumOperator(["Y"], [1.0])
        plo = PennyLaneOperator(op)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_two_qubit_zi_iz_observable(self):
        """Test two-qubit ZI + IZ observable."""
        op = QuantumOperator(["ZI", "IZ"], [1.0, 1.0])
        plo = PennyLaneOperator(op)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_two_qubit_xi_ix_observable(self):
        """Test two-qubit XI + IX observable."""
        op = QuantumOperator(["XI", "IX"], [1.0, 1.0])
        plo = PennyLaneOperator(op)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_two_qubit_zz_observable(self):
        """Test two-qubit ZZ observable."""
        op = QuantumOperator(["ZZ"], [1.0])
        plo = PennyLaneOperator(op)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_two_qubit_xx_observable(self):
        """Test two-qubit XX observable."""
        op = QuantumOperator(["XX"], [1.0])
        plo = PennyLaneOperator(op)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_multi_term_observable(self):
        """Test observable with multiple Pauli terms."""
        op = QuantumOperator(["XX", "YY", "ZZ"], [0.5, 0.3, 0.2])
        plo = PennyLaneOperator(op)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_identity_observable(self):
        """Test identity observable."""
        op = QuantumOperator(["II"], [1.0])
        plo = PennyLaneOperator(op)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_three_qubit_observable(self):
        """Test three-qubit observable."""
        op = QuantumOperator(["ZZZ", "XXX"], [1.0, 0.5])
        plo = PennyLaneOperator(op)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    @pytest.mark.parametrize("coeff", [0.0, 0.5, 1.0, -1.0, 2.5])
    def test_observable_with_various_coefficients(self, coeff):
        """Test observable with various coefficient values."""
        op = QuantumOperator(["Z"], [coeff])
        plo = PennyLaneOperator(op)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    # Parametric Observable Tests

    def test_observable_single_parameter(self):
        """Test observable with a single parameter."""
        theta = ParameterVector("theta", 1)
        op = QuantumOperator(["Z"], [theta[0]])
        plo = PennyLaneOperator(op)

        assert plo is not None
        assert "theta" in plo.parameter_names
        assert plo.parameter_dimensions["theta"] == 1

    def test_observable_two_parameters_same_vector(self):
        """Test observable with two parameters from the same vector."""
        theta = ParameterVector("theta", 2)
        op = QuantumOperator(["ZI", "IZ"], [theta[0], theta[1]])
        plo = PennyLaneOperator(op)

        assert plo is not None
        assert "theta" in plo.parameter_names
        assert plo.parameter_dimensions["theta"] == 2

    def test_observable_three_parameters_same_vector(self):
        """Test observable with three parameters from the same vector."""
        alpha = ParameterVector("alpha", 3)
        op = QuantumOperator(["ZI", "IZ", "ZZ"], [alpha[0], alpha[1], alpha[2]])
        plo = PennyLaneOperator(op)

        assert plo is not None
        assert "alpha" in plo.parameter_names
        assert plo.parameter_dimensions["alpha"] == 3

    def test_observable_multiple_parameter_vectors(self):
        """Test observable with multiple different parameter vectors."""
        pop1 = ParameterVector("pop1", 1)
        pop2 = ParameterVector("pop2", 1)
        op = QuantumOperator(["ZI", "IZ"], [pop1[0], pop2[0]])
        plo = PennyLaneOperator(op)

        assert plo is not None
        assert "pop1" in plo.parameter_names
        assert "pop2" in plo.parameter_names
        assert plo.parameter_dimensions["pop1"] == 1
        assert plo.parameter_dimensions["pop2"] == 1

    def test_observable_three_parameter_vectors(self):
        """Test observable with three different parameter vectors."""
        a = ParameterVector("a", 1)
        b = ParameterVector("b", 1)
        c = ParameterVector("c", 1)
        op = QuantumOperator(["X", "Y", "Z"], [a[0], b[0], c[0]])
        plo = PennyLaneOperator(op)

        assert plo is not None
        assert "a" in plo.parameter_names
        assert "b" in plo.parameter_names
        assert "c" in plo.parameter_names

    # Parameter Expression Tests

    def test_observable_with_parameter_multiplication(self):
        """Test observable with parameter expression: 2 * theta[0]."""
        theta = ParameterVector("theta", 1)
        op = QuantumOperator(["Z"], [2 * theta[0]])
        plo = PennyLaneOperator(op)

        assert plo is not None
        assert "theta" in plo.parameter_names

    def test_observable_with_parameter_expressions(self):
        """Test observable with multiple parameter expressions."""
        theta = ParameterVector("theta", 2)
        op = QuantumOperator(["ZI", "IZ"], [theta[0] * 2, theta[1] * 0.5])
        plo = PennyLaneOperator(op)

        assert plo is not None
        assert "theta" in plo.parameter_names
        assert plo.parameter_dimensions["theta"] == 2

    def test_observable_with_parameter_addition(self):
        """Test observable with parameter addition expression."""
        theta = ParameterVector("theta", 2)
        op = QuantumOperator(["ZI", "IZ"], [theta[0] + theta[1], theta[0]])
        plo = PennyLaneOperator(op)

        assert plo is not None
        assert "theta" in plo.parameter_names

    # Property and Method Tests

    def test_parameter_names_property_empty(self):
        """Test parameter_names property for non-parametric observable."""
        op = QuantumOperator(["Z"], [1.0])
        plo = PennyLaneOperator(op)

        assert isinstance(plo.parameter_names, list)
        assert len(plo.parameter_names) == 0

    def test_parameter_names_property_with_params(self):
        """Test parameter_names property for parametric observable."""
        theta = ParameterVector("theta", 2)
        op = QuantumOperator(["ZI", "IZ"], [theta[0], theta[1]])
        plo = PennyLaneOperator(op)

        assert isinstance(plo.parameter_names, list)
        assert "theta" in plo.parameter_names

    def test_parameter_dimensions_property(self):
        """Test parameter_dimensions property."""
        theta = ParameterVector("theta", 3)
        op = QuantumOperator(["X", "Y", "Z"], [theta[0], theta[1], theta[2]])
        plo = PennyLaneOperator(op)

        assert isinstance(plo.parameter_dimensions, dict)
        assert plo.parameter_dimensions["theta"] == 3

    def test_hash_property(self):
        """Test that hash property returns a valid integer."""
        op = QuantumOperator(["ZI", "IZ"], [1.0, 1.0])
        plo = PennyLaneOperator(op)

        hash_value = plo.hash
        assert isinstance(hash_value, int)

    def test_hash_consistency(self):
        """Test that hash remains consistent for the same observable."""
        op = QuantumOperator(["ZI", "IZ"], [1.0, 1.0])
        plo = PennyLaneOperator(op)

        hash1 = plo.hash
        hash2 = plo.hash
        assert hash1 == hash2

    def test_build_pennylane_observable_method(self):
        """Test that build_pennylane_observable() returns callable."""
        op = QuantumOperator(["ZI", "IZ"], [1.0, 1.0])
        plo = PennyLaneOperator(op)

        pennylane_obs = plo.build_pennylane_observable()
        assert callable(pennylane_obs)

    def test_build_pennylane_observable_with_params(self):
        """Test that build_pennylane_observable() works with parameters."""
        theta = ParameterVector("theta", 2)
        op = QuantumOperator(["ZI", "IZ"], [theta[0], theta[1]])
        plo = PennyLaneOperator(op)

        pennylane_obs = plo.build_pennylane_observable()
        assert callable(pennylane_obs)
