import numpy as np
import pytest
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

from executor import QuantumOperator
from executor.qiskit import QiskitOperator


def _make_parametrized_observable(num_qubits=2, vec_name="theta", length=3):
    """
    Create a parametrized SparsePauliOp with a ParameterVector.
    """
    vec = ParameterVector(vec_name, length)

    # One Pauli term per parameter (coefficients are parameters)
    paulis = ["I" * num_qubits] * length
    coeffs = list(vec)

    operator = SparsePauliOp(paulis, coeffs)
    return operator


class TestQiskitOperator:

    def test_single_qubit_z_observable(self):
        """Test single-qubit Z observable."""
        operator = QuantumOperator(["Z"], [1.0])
        qiskit_op = QiskitOperator(operator)

        assert qiskit_op is not None
        assert len(qiskit_op.parameter_names) == 0
        assert len(qiskit_op.parameter_dimensions) == 0

    def test_single_qubit_x_observable(self):
        """Test single-qubit X observable."""
        operator = QuantumOperator(["X"], [1.0])
        qiskit_op = QiskitOperator(operator)

        assert qiskit_op is not None
        assert len(qiskit_op.parameter_names) == 0

    def test_single_qubit_y_observable(self):
        """Test single-qubit Y observable."""
        operator = QuantumOperator(["Y"], [1.0])
        qiskit_op = QiskitOperator(operator)

        assert qiskit_op is not None
        assert len(qiskit_op.parameter_names) == 0

    def test_two_qubit_zi_iz_observable(self):
        """Test two-qubit ZI + IZ observable."""
        operator = QuantumOperator(["ZI", "IZ"], [1.0, 1.0])
        qiskit_op = QiskitOperator(operator)

        assert qiskit_op is not None
        assert len(qiskit_op.parameter_names) == 0

    def test_two_qubit_xi_ix_observable(self):
        """Test two-qubit XI + IX observable."""
        operator = QuantumOperator(["XI", "IX"], [1.0, 1.0])
        qiskit_op = QiskitOperator(operator)

        assert qiskit_op is not None
        assert len(qiskit_op.parameter_names) == 0

    def test_two_qubit_zz_observable(self):
        """Test two-qubit ZZ observable."""
        operator = QuantumOperator(["ZZ"], [1.0])
        qiskit_op = QiskitOperator(operator)

        assert qiskit_op is not None
        assert len(qiskit_op.parameter_names) == 0

    def test_two_qubit_xx_observable(self):
        """Test two-qubit XX observable."""
        operator = QuantumOperator(["XX"], [1.0])
        qiskit_op = QiskitOperator(operator)

        assert qiskit_op is not None
        assert len(qiskit_op.parameter_names) == 0

    def test_multi_term_observable(self):
        """Test observable with multiple Pauli terms."""
        operator = QuantumOperator(["XX", "YY", "ZZ"], [0.5, 0.3, 0.2])
        qiskit_op = QiskitOperator(operator)

        assert qiskit_op is not None
        assert len(qiskit_op.parameter_names) == 0

    def test_identity_observable(self):
        """Test identity observable."""
        operator = QuantumOperator(["II"], [1.0])
        qiskit_op = QiskitOperator(operator)

        assert qiskit_op is not None
        assert len(qiskit_op.parameter_names) == 0

    def test_three_qubit_observable(self):
        """Test three-qubit observable."""
        operator = QuantumOperator(["ZZZ", "XXX"], [1.0, 0.5])
        qiskit_op = QiskitOperator(operator)

        assert qiskit_op is not None
        assert len(qiskit_op.parameter_names) == 0

    @pytest.mark.parametrize("coeff", [0.0, 0.5, 1.0, -1.0, 2.5])
    def test_observable_with_various_coefficients(self, coeff):
        """Test observable with various coefficient values."""
        operator = QuantumOperator(["Z"], [coeff])
        qiskit_op = QiskitOperator(operator)

        assert qiskit_op is not None
        assert len(qiskit_op.parameter_names) == 0

    def test_observable_single_parameter(self):
        """Test observable with a single parameter."""
        theta = ParameterVector("theta", 1)
        operator = QuantumOperator(["Z"], [theta[0]])
        qiskit_op = QiskitOperator(operator)

        assert qiskit_op is not None
        assert "theta" in qiskit_op.parameter_names
        assert qiskit_op.parameter_dimensions["theta"] == 1

    def test_observable_two_parameters_same_vector(self):
        """Test observable with two parameters from the same vector."""
        theta = ParameterVector("theta", 2)
        operator = QuantumOperator(["ZI", "IZ"], [theta[0], theta[1]])
        qiskit_op = QiskitOperator(operator)

        assert qiskit_op is not None
        assert "theta" in qiskit_op.parameter_names
        assert qiskit_op.parameter_dimensions["theta"] == 2

    def test_observable_three_parameters_same_vector(self):
        """Test observable with three parameters from the same vector."""
        alpha = ParameterVector("alpha", 3)
        operator = QuantumOperator(["ZI", "IZ", "ZZ"], [alpha[0], alpha[1], alpha[2]])
        qiskit_op = QiskitOperator(operator)

        assert qiskit_op is not None
        assert "alpha" in qiskit_op.parameter_names
        assert qiskit_op.parameter_dimensions["alpha"] == 3

    def test_observable_multiple_parameter_vectors(self):
        """Test observable with multiple different parameter vectors."""
        pop1 = ParameterVector("pop1", 1)
        pop2 = ParameterVector("pop2", 1)
        operator = QuantumOperator(["ZI", "IZ"], [pop1[0], pop2[0]])
        qiskit_op = QiskitOperator(operator)

        assert qiskit_op is not None
        assert "pop1" in qiskit_op.parameter_names
        assert "pop2" in qiskit_op.parameter_names
        assert qiskit_op.parameter_dimensions["pop1"] == 1
        assert qiskit_op.parameter_dimensions["pop2"] == 1

    def test_observable_three_parameter_vectors(self):
        """Test observable with three different parameter vectors."""
        a = ParameterVector("a", 1)
        b = ParameterVector("b", 1)
        c = ParameterVector("c", 1)
        operator = QuantumOperator(["X", "Y", "Z"], [a[0], b[0], c[0]])
        qiskit_op = QiskitOperator(operator)

        assert qiskit_op is not None
        assert "a" in qiskit_op.parameter_names
        assert "b" in qiskit_op.parameter_names
        assert "c" in qiskit_op.parameter_names

    def test_observable_with_parameter_multiplication(self):
        """Test observable with parameter expression: 2 * theta[0]."""
        theta = ParameterVector("theta", 1)
        operator = QuantumOperator(["Z"], [2 * theta[0]])
        qiskit_op = QiskitOperator(operator)

        assert qiskit_op is not None
        assert "theta" in qiskit_op.parameter_names

    def test_observable_with_parameter_expressions(self):
        """Test observable with multiple parameter expressions."""
        theta = ParameterVector("theta", 2)
        operator = QuantumOperator(["ZI", "IZ"], [theta[0] * 2, theta[1] * 0.5])
        qiskit_op = QiskitOperator(operator)

        assert qiskit_op is not None
        assert "theta" in qiskit_op.parameter_names
        assert qiskit_op.parameter_dimensions["theta"] == 2

    def test_observable_with_parameter_addition(self):
        """Test observable with parameter addition expression."""
        theta = ParameterVector("theta", 2)
        operator = QuantumOperator(["ZI", "IZ"], [theta[0] + theta[1], theta[0]])
        qiskit_op = QiskitOperator(operator)

        assert qiskit_op is not None
        assert "theta" in qiskit_op.parameter_names

    def test_parameter_names_property_empty(self):
        """Test parameter_names property for non-parametric observable."""
        operator = QuantumOperator(["Z"], [1.0])
        qiskit_op = QiskitOperator(operator)

        assert isinstance(qiskit_op.parameter_names, list)
        assert len(qiskit_op.parameter_names) == 0

    def test_parameter_names_property_with_params(self):
        """Test parameter_names property for parametric observable."""
        theta = ParameterVector("theta", 2)
        operator = QuantumOperator(["ZI", "IZ"], [theta[0], theta[1]])
        qiskit_op = QiskitOperator(operator)

        assert isinstance(qiskit_op.parameter_names, list)
        assert "theta" in qiskit_op.parameter_names

    def test_parameter_dimensions_property(self):
        """Test parameter_dimensions property."""
        theta = ParameterVector("theta", 3)
        operator = QuantumOperator(["X", "Y", "Z"], [theta[0], theta[1], theta[2]])
        qiskit_op = QiskitOperator(operator)

        assert isinstance(qiskit_op.parameter_dimensions, dict)
        assert qiskit_op.parameter_dimensions["theta"] == 3

    def test_hash_property(self):
        """Test that hash property returns a valid integer."""
        operator = QuantumOperator(["ZI", "IZ"], [1.0, 1.0])
        qiskit_op = QiskitOperator(operator)

        hash_value = qiskit_op.hash
        assert isinstance(hash_value, int)

    def test_hash_consistency(self):
        """Test that hash remains consistent for the same observable."""
        operator = QuantumOperator(["ZI", "IZ"], [1.0, 1.0])
        qiskit_op = QiskitOperator(operator)

        hash1 = qiskit_op.hash
        hash2 = qiskit_op.hash
        assert hash1 == hash2

    @pytest.mark.parametrize(
        "vec_name, vec_len, param_values, expected_unbound_count",
        [
            ("v", 3, {"v": 0.5}, 2),  # scalar binds first
            ("w", 3, {"w": [0.1, 0.2, 0.3]}, 0),  # list binds all
            ("arr", 4, {"arr": np.array([1.0, 2.0])}, 2),  # numpy array partial bind
            ("more", 3, {"more": [1, 2, 3, 4]}, 0),  # extra values ignored
        ],
    )
    def test_bind_parameters_various_cases(
        self, vec_name, vec_len, param_values, expected_unbound_count
    ):
        """Test bind_parameters with various parameter value formats."""
        operator = _make_parametrized_observable(vec_name=vec_name, length=vec_len)
        qiskit_op = QiskitOperator(operator)
        bound = qiskit_op.bind_parameters(param_values)
        assert len(bound.parameters) == expected_unbound_count

    @pytest.mark.parametrize(
        "param_values, expect_no_binding",
        [
            ({}, True),  # empty dict
            ({"does_not_exist": [1, 2]}, True),  # unknown vector
            ({"a": [0.3], "nope": [1.0]}, False),  # mixed case
        ],
    )
    def test_bind_parameters_identity_and_mixed_cases(self, param_values, expect_no_binding):
        """Test bind_parameters for identity and mixed binding cases."""
        operator = _make_parametrized_observable(vec_name="a", length=2)
        qiskit_op = QiskitOperator(operator)

        original_params = set(operator.parameters)
        original_str = str(operator)

        ret = qiskit_op.bind_parameters(param_values)

        if expect_no_binding:
            assert set(ret.parameters) == original_params
            assert str(ret) == original_str
        else:
            assert len(ret.parameters) < len(original_params)
