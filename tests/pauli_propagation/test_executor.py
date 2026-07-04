"""Tests for PauliPropagationExecutor (strict native API)."""

import numpy as np
import pytest

from qc_executor.parameters import Parameters
from qc_executor.pauli_propagation import (
    PauliPropagationCircuit,
    PauliPropagationExecutor,
    PauliPropagationOperator,
)
from qc_executor.pauli_propagation.pauli_propagation_executor import (
    _create_projector_observable,
)
from qc_executor.pauli_propagation.utils.pauli_algebra import string_to_term


class TestPauliPropagationExecutor:
    def test_get_accepted_backend_types(self):
        assert PauliPropagationExecutor.get_accepted_backend_types() == []

    def test_get_accepted_backend_aliases(self):
        assert PauliPropagationExecutor.get_accepted_backend_aliases() == []

    def test_init(self):
        executor = PauliPropagationExecutor(
            shots=1000,
            seed=42,
            truncate_threshold=1e-10,
            max_weight=5,
        )
        assert executor.shots == 1000
        assert executor.remote is False
        assert executor.truncate_threshold == 1e-10
        assert executor.max_weight == 5

    def test_expectation_value_identity_circuit(self):
        executor = PauliPropagationExecutor()
        circuit = PauliPropagationCircuit(2)
        observable = PauliPropagationOperator(["ZZ"], [1.0])

        result = executor.expectation_value(circuit, observable)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_expectation_value_hadamard_x(self):
        executor = PauliPropagationExecutor()
        circuit = PauliPropagationCircuit(1)
        circuit.h(0)
        observable = PauliPropagationOperator(["X"], [1.0])

        result = executor.expectation_value(circuit, observable)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_expectation_value_rotation_gate(self):
        executor = PauliPropagationExecutor()
        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, np.pi / 2)
        observable = PauliPropagationOperator(["Z"], [1.0])

        result = executor.expectation_value(circuit, observable)
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_expectation_value_cnot_bell_state(self):
        executor = PauliPropagationExecutor()
        circuit = PauliPropagationCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        observable = PauliPropagationOperator(["ZZ"], [1.0])

        result = executor.expectation_value(circuit, observable)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_expectation_value_parametric_circuit(self):
        executor = PauliPropagationExecutor()
        theta = Parameters("theta", 1)

        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, theta[0])
        observable = PauliPropagationOperator(["Z"], [1.0])

        result = executor.expectation_value(circuit, observable, **{"theta[0]": 0.0})
        assert np.isclose(result, 1.0, atol=1e-10)

        result = executor.expectation_value(circuit, observable, **{"theta[0]": np.pi})
        assert np.isclose(result, -1.0, atol=1e-10)

    def test_truncation_statistics(self):
        executor = PauliPropagationExecutor(truncate_threshold=0.1)
        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, 0.1)
        observable = PauliPropagationOperator(["Z"], [1.0])

        executor.expectation_value(circuit, observable)
        stats = executor.get_truncation_stats()
        assert stats is not None
        assert stats.coeff_norm_total > 0

    def test_batch_execution_multiple_circuits(self):
        executor = PauliPropagationExecutor()

        circuit1 = PauliPropagationCircuit(1)
        circuit2 = PauliPropagationCircuit(1)
        circuit2.x(0)

        observable = PauliPropagationOperator(["Z"], [1.0])

        results = executor.expectation_value([circuit1, circuit2], observable)

        assert len(results) == 2
        assert np.isclose(results[0], 1.0, atol=1e-10)
        assert np.isclose(results[1], -1.0, atol=1e-10)


class TestBatchExpectationValue:
    def test_multi_observable_matches_single_calls(self):
        executor = PauliPropagationExecutor()

        circuit = PauliPropagationCircuit(1)
        circuit.h(0)

        op_x = PauliPropagationOperator(["X"], [1.0])
        op_z = PauliPropagationOperator(["Z"], [1.0])

        batch = executor.expectation_value(circuit, [op_x, op_z])
        single_x = executor.expectation_value(circuit, op_x)
        single_z = executor.expectation_value(circuit, op_z)

        assert np.isclose(batch[0], single_x, atol=1e-10)
        assert np.isclose(batch[1], single_z, atol=1e-10)

    def test_multi_circuit_multi_observable_ordering(self):
        executor = PauliPropagationExecutor()

        c0 = PauliPropagationCircuit(1)
        c1 = PauliPropagationCircuit(1)
        c1.x(0)

        op_z = PauliPropagationOperator(["Z"], [1.0])
        op_x = PauliPropagationOperator(["X"], [1.0])

        results = executor.expectation_value([c0, c1], [op_z, op_x])

        expected = np.array([1.0, 0.0, -1.0, 0.0])
        assert np.allclose(results, expected, atol=1e-10)

    def test_single_observable_still_returns_float(self):
        executor = PauliPropagationExecutor()
        circuit = PauliPropagationCircuit(1)
        observable = PauliPropagationOperator(["Z"], [1.0])

        result = executor.expectation_value(circuit, observable)
        assert isinstance(result, float)


class TestStrictInputs:
    def test_rejects_non_native_circuit(self):
        executor = PauliPropagationExecutor()
        observable = PauliPropagationOperator(["Z"], [1.0])

        with pytest.raises(TypeError, match="PauliPropagationCircuit"):
            executor.expectation_value("not a circuit", observable)

    def test_rejects_non_native_observable(self):
        executor = PauliPropagationExecutor()
        circuit = PauliPropagationCircuit(1)

        with pytest.raises(TypeError, match="PauliPropagationOperator"):
            executor.expectation_value(circuit, "not an observable")


class TestParameterNormalization:
    def test_expectation_value_with_list_parameters(self):
        """Test that parameters can be passed as lists like x=[0.1], p=[0.3]."""
        executor = PauliPropagationExecutor()
        theta = Parameters("theta", 1)

        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, theta[0])
        observable = PauliPropagationOperator(["Z"], [1.0])

        # Test list format: theta=[0.0]
        result = executor.expectation_value(circuit, observable, theta=[0.0])
        assert np.isclose(result, 1.0, atol=1e-10)

        # Test list format: theta=[pi]
        result = executor.expectation_value(circuit, observable, theta=[np.pi])
        assert np.isclose(result, -1.0, atol=1e-10)

    def test_expectation_value_with_multiple_list_parameters(self):
        """Test multiple parameters in list format."""
        executor = PauliPropagationExecutor()
        # Use simple Parameters that match what bind_parameters expects
        theta = Parameters("theta", 2)

        circuit = PauliPropagationCircuit(2)
        circuit.rx(0, theta[0])
        circuit.ry(1, theta[1])

        observable = PauliPropagationOperator(["ZI", "IZ"], [1.0, 1.0])

        # Test with list parameters in correct format
        result = executor.expectation_value(circuit, observable, theta=[0.0, 0.0])
        assert np.isfinite(result)

    def test_expectation_value_with_indexed_parameters(self):
        """Test that indexed format still works (backward compatibility)."""
        executor = PauliPropagationExecutor()
        theta = Parameters("theta", 1)

        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, theta[0])
        observable = PauliPropagationOperator(["Z"], [1.0])

        # Test indexed format: {"theta[0]": 0.0}
        result = executor.expectation_value(circuit, observable, **{"theta[0]": 0.0})
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_expectation_value_derivatives_with_list_parameters(self):
        """Test derivatives with list parameters."""
        executor = PauliPropagationExecutor()
        theta = Parameters("theta", 1)

        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, theta[0])
        observable = PauliPropagationOperator(["X"], [1.0])

        # Test with list parameters
        result = executor.expectation_value_derivatives(circuit, observable, "theta", theta=[0.0])
        assert isinstance(result, (float, np.ndarray))

    def test_statevector_with_list_parameters(self):
        """Test statevector with list parameters."""
        executor = PauliPropagationExecutor()
        theta = Parameters("theta", 1)

        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, theta[0])

        # Test with list parameters
        result = executor.statevector(circuit, theta=[0.0])
        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_sample_with_list_parameters(self):
        """Test sampling with list parameters."""
        executor = PauliPropagationExecutor(shots=100, seed=42)
        theta = Parameters("theta", 1)

        circuit = PauliPropagationCircuit(1)
        circuit.h(0)
        circuit.rx(0, theta[0])

        # Test with list parameters
        result = executor.sample(circuit, theta=[0.0])
        assert isinstance(result, dict)
        assert sum(result.values()) == 100  # Total shots


class TestProjectorObservableOrdering:
    """Regression tests for qubit-ordering convention in _create_projector_observable.

    Convention: qubit 0 is the leftmost character in both bitstrings and Pauli strings.
    """

    def test_single_qubit_projector_0(self):
        """Projector onto |0> should be (I + Z)/2."""
        psum = _create_projector_observable("0", 1)
        terms = dict(psum)

        i_term = string_to_term("I", 1)
        z_term = string_to_term("Z", 1)
        assert np.isclose(terms.get(i_term, 0), 0.5, atol=1e-10)
        assert np.isclose(terms.get(z_term, 0), 0.5, atol=1e-10)

    def test_single_qubit_projector_1(self):
        """Projector onto |1> should be (I - Z)/2."""
        psum = _create_projector_observable("1", 1)
        terms = dict(psum)

        i_term = string_to_term("I", 1)
        z_term = string_to_term("Z", 1)
        assert np.isclose(terms.get(i_term, 0), 0.5, atol=1e-10)
        assert np.isclose(terms.get(z_term, 0), -0.5, atol=1e-10)

    def test_two_qubit_projector_ordering(self):
        """Regression test: qubit 0 is leftmost.

        For bitstring "10" (qubit 0 = 1, qubit 1 = 0):
        Projector = [(I - Z_0)/2] ⊗ [(I + Z_1)/2]
        The II term must be 0.25 and the IZ term must be +0.25 (from qubit 1)
        while the ZI term must be -0.25 (from qubit 0).
        If the convention were reversed ("qubit 0 rightmost"), the signs of
        the ZI and IZ terms would be swapped, distinguishing the two conventions.
        """
        psum = _create_projector_observable("10", 2)
        terms = dict(psum)

        ii = string_to_term("II", 2)
        zi = string_to_term("ZI", 2)  # Z on qubit 0, I on qubit 1
        iz = string_to_term("IZ", 2)  # I on qubit 0, Z on qubit 1
        zz = string_to_term("ZZ", 2)

        assert np.isclose(terms.get(ii, 0), 0.25, atol=1e-10), "II coefficient"
        assert np.isclose(terms.get(zi, 0), -0.25, atol=1e-10), "ZI coefficient (qubit 0 = 1)"
        assert np.isclose(terms.get(iz, 0), 0.25, atol=1e-10), "IZ coefficient (qubit 1 = 0)"
        assert np.isclose(terms.get(zz, 0), -0.25, atol=1e-10), "ZZ coefficient"
