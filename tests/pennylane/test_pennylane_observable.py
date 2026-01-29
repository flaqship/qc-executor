from qiskit.circuit import ParameterVector

from executor import QuantumOperator
from executor.pennylane.pennylane_observable import PennyLaneObservable


class TestPennyLaneObservable:
    """Test suite for PennyLane observable conversion."""

    def test_simple_observable_initialization(self):
        """Test basic observable initialization."""
        op = QuantumOperator(["ZI", "IZ"], [1.0, 1.0])
        plo = PennyLaneObservable(op)

        assert plo is not None
        assert len(plo.parameter_names) == 0

    def test_observable_with_parameters(self):
        """Test observable with parameters."""
        pop = ParameterVector('pop', 2)
        op = QuantumOperator(["ZI", "IZ"], [pop[0], pop[1]])
        plo = PennyLaneObservable(op)

        assert 'pop' in plo.parameter_names
        assert plo.parameter_dimensions['pop'] == 2

    def test_observable_with_single_parameter(self):
        """Test observable with single parameter."""
        pop = ParameterVector('pop', 1)
        op = QuantumOperator(["ZZ"], [pop[0]])
        plo = PennyLaneObservable(op)

        assert 'pop' in plo.parameter_names
        assert plo.parameter_dimensions['pop'] == 1

    def test_observable_with_multiple_paulis(self):
        """Test observable with multiple Pauli terms."""
        op = QuantumOperator(["XX", "YY", "ZZ"], [0.5, 0.3, 0.2])
        plo = PennyLaneObservable(op)

        assert plo is not None

    def test_observable_with_identity(self):
        """Test observable with identity operator."""
        op = QuantumOperator(["II"], [1.0])
        plo = PennyLaneObservable(op)

        assert plo is not None

    def test_observable_build_pennylane_observable(self):
        """Test building executable PennyLane observable."""
        op = QuantumOperator(["ZI", "IZ"], [1.0, 1.0])
        plo = PennyLaneObservable(op)

        pennylane_obs = plo.build_pennylane_observable()
        assert callable(pennylane_obs)

    def test_observable_hash_property(self):
        """Test that observable has a hash property."""
        op = QuantumOperator(["ZI"], [1.0])
        plo = PennyLaneObservable(op)

        hash_value = plo.hash
        assert hash_value is not None
        assert isinstance(hash_value, int)

    def test_observable_parameter_names_property(self):
        """Test parameter_names property."""
        pop = ParameterVector('pop', 2)
        op = QuantumOperator(["ZI", "IZ"], [pop[0], pop[1]])
        plo = PennyLaneObservable(op)

        assert 'pop' in plo.parameter_names

    def test_observable_parameter_dimensions_property(self):
        """Test parameter_dimensions property."""
        pop = ParameterVector('pop', 3)
        op = QuantumOperator(["ZI", "IZ", "XI"], [pop[0], pop[1], pop[2]])
        plo = PennyLaneObservable(op)

        assert plo.parameter_dimensions['pop'] == 3

    def test_observable_with_x_paulis(self):
        """Test observable with X Pauli operators."""
        op = QuantumOperator(["XI", "IX"], [1.0, 1.0])
        plo = PennyLaneObservable(op)

        assert plo is not None

    def test_observable_with_y_paulis(self):
        """Test observable with Y Pauli operators."""
        op = QuantumOperator(["YI", "IY"], [1.0, 1.0])
        plo = PennyLaneObservable(op)

        assert plo is not None

    def test_observable_with_z_paulis(self):
        """Test observable with Z Pauli operators."""
        op = QuantumOperator(["ZI", "IZ"], [1.0, 1.0])
        plo = PennyLaneObservable(op)

        assert plo is not None

    def test_observable_with_mixed_paulis(self):
        """Test observable with mixed Pauli operators."""
        op = QuantumOperator(["XY", "YZ", "ZX"], [1.0, 0.5, 0.3])
        plo = PennyLaneObservable(op)

        assert plo is not None

    def test_observable_with_parameter_expressions(self):
        """Test observable with parameter expressions."""
        pop = ParameterVector('pop', 1)
        op = QuantumOperator(["ZI"], [2 * pop[0]])
        plo = PennyLaneObservable(op)

        assert 'pop' in plo.parameter_names

    def test_observable_with_multiple_parameter_vectors(self):
        """Test observable with multiple parameter vectors."""
        pop1 = ParameterVector('pop1', 1)
        pop2 = ParameterVector('pop2', 1)
        op = QuantumOperator(["ZI", "IZ"], [pop1[0], pop2[0]])
        plo = PennyLaneObservable(op)

        assert 'pop1' in plo.parameter_names
        assert 'pop2' in plo.parameter_names

    def test_observable_with_three_qubit_pauli(self):
        """Test observable with three-qubit Pauli operators."""
        op = QuantumOperator(["ZZZ", "XXX"], [1.0, 0.5])
        plo = PennyLaneObservable(op)

        assert plo is not None

    def test_observable_with_zero_coefficient(self):
        """Test observable with zero coefficient."""
        op = QuantumOperator(["ZI", "IZ"], [0.0, 1.0])
        plo = PennyLaneObservable(op)

        assert plo is not None

    def test_observable_with_negative_coefficient(self):
        """Test observable with negative coefficient."""
        op = QuantumOperator(["ZI", "IZ"], [-1.0, 1.0])
        plo = PennyLaneObservable(op)

        assert plo is not None

    def test_observable_list_initialization(self):
        """Test observable initialization with list of operators."""
        op1 = QuantumOperator(["ZI"], [1.0])
        op2 = QuantumOperator(["IZ"], [1.0])
        plo = PennyLaneObservable([op1, op2])

        assert plo is not None

    def test_observable_with_real_coefficients(self):
        """Test observable with real coefficients."""
        op = QuantumOperator(["ZI", "IZ"], [0.7, 0.3])
        plo = PennyLaneObservable(op)

        assert plo is not None

    def test_observable_single_pauli(self):
        """Test observable with single Pauli term."""
        op = QuantumOperator(["Z"], [1.0])
        plo = PennyLaneObservable(op)

        assert plo is not None

    def test_observable_without_parameters(self):
        """Test observable without parameters."""
        op = QuantumOperator(["ZZ", "XX"], [1.0, 0.5])
        plo = PennyLaneObservable(op)

        assert len(plo.parameter_names) == 0
        assert len(plo.parameter_dimensions) == 0
