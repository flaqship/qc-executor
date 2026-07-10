"""Tests for PauliPropagationExecutor (strict native API)."""

from typing import Dict

import numpy as np
import pytest
import sympy as sp

from qc_executor import QuantumCircuit, QuantumOperator
from qc_executor.parameters import Parameters
from qc_executor.pauli_propagation import (
    PauliPropagationCircuit,
    PauliPropagationExecutor,
    PauliPropagationOperator,
)
from qc_executor.pauli_propagation import pauli_propagation_executor as ppe
from qc_executor.pauli_propagation.pauli_propagation_executor import _create_projector_observable
from qc_executor.pauli_propagation.utils.gates import CliffordGate, PauliRotation
from qc_executor.pauli_propagation.utils.pauli_algebra import string_to_term


class TestPauliPropagationExecutor:
    def test_get_accepted_backend_types(self):
        assert not PauliPropagationExecutor.get_accepted_backend_types()

    def test_get_accepted_backend_aliases(self):
        assert not PauliPropagationExecutor.get_accepted_backend_aliases()

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

    def test_expectation_value_parametric_observable_coefficients(self):
        executor = PauliPropagationExecutor()
        alpha = sp.Symbol("alpha")

        circuit = PauliPropagationCircuit(1)
        observable = PauliPropagationOperator(["Z"], [alpha])

        result = executor.expectation_value(circuit, observable, alpha=1.0)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_batch_execution_multiple_circuits(self):
        executor = PauliPropagationExecutor()

        circuit1 = PauliPropagationCircuit(1)
        circuit2 = PauliPropagationCircuit(1)
        circuit2.x(0)

        observable = PauliPropagationOperator(["Z"], [1.0])

        results = executor.expectation_value([circuit1, circuit2], observable)

        assert isinstance(results, np.ndarray)
        results = np.asarray(results)
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

        assert isinstance(batch, np.ndarray)
        batch = np.asarray(batch)
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


class TestExecutorHelperFunctions:
    def test_as_list(self):
        assert ppe._as_list(3) == [3]
        assert ppe._as_list([1, 2]) == [1, 2]

    def test_normalize_parameters_variants_and_errors(self):
        assert not ppe._normalize_parameters({})

        normalized = ppe._normalize_parameters(
            {
                "x": [0.1, 0.2],
                "y": (1, 2),
                "z": np.float64(0.3),
                "a[0]": "a[0]",
            }
        )

        assert np.isclose(normalized["x[0]"], 0.1)
        assert np.isclose(normalized["y[1]"], 2.0)
        assert np.isclose(normalized["z"], 0.3)
        assert normalized["a[0]"] == "a[0]"

        with pytest.raises(TypeError, match="invalid type"):
            ppe._normalize_parameters({"x": [object()]})

        with pytest.raises(TypeError, match="invalid value type"):
            ppe._normalize_parameters({"x": object()})

    def test_evaluate_symbolic_expression_success_and_errors(self):
        x = sp.Symbol("x")
        assert np.isclose(ppe._evaluate_symbolic_expression(x + 1, {"x": 2.0}), 3.0)

        with pytest.raises(ValueError, match="Missing parameter value"):
            ppe._evaluate_symbolic_expression(x + 1, {})

        y = sp.Symbol("y")
        with pytest.raises(ValueError, match="could not be fully evaluated"):
            ppe._evaluate_symbolic_expression(x + y, {"x": 1.0, "y": sp.Symbol("z")})

    def test_derivative_param_to_name_variants(self):
        class Named:
            name = "theta"

        class Unnamed:
            def __str__(self):
                return "unnamed"

        assert ppe._derivative_param_to_name("x") == "x"
        assert ppe._derivative_param_to_name(Named()) == "theta"
        assert ppe._derivative_param_to_name(Unnamed()) == "unnamed"

    def test_create_projector_observable_validation_and_pauli_branches(self, monkeypatch):
        with pytest.raises(ValueError, match="doesn't match nqubits"):
            _create_projector_observable("01", 1)

        # term_to_string is imported into the executor module's namespace, so it must
        # be patched there (not on its source module) for the patch to take effect.
        target = "qc_executor.pauli_propagation.pauli_propagation_executor.term_to_string"

        # X branch: I -> Z component picks up X -> Y with phase 1j.
        monkeypatch.setattr(target, lambda term, nqubits: "X")
        result = _create_projector_observable("0", 1)
        assert result.get_coeff("I") == pytest.approx(0.5)
        assert result.get_coeff("Y") == pytest.approx(0.5j)

        # Y branch: X -> Y -> X with phase -1j.
        monkeypatch.setattr(target, lambda term, nqubits: "Y")
        result = _create_projector_observable("0", 1)
        assert result.get_coeff("I") == pytest.approx(0.5)
        assert result.get_coeff("X") == pytest.approx(-0.5j)

        # Z branch: Z -> I with phase 1, doubling the existing I component.
        monkeypatch.setattr(target, lambda term, nqubits: "Z")
        result = _create_projector_observable("0", 1)
        assert result.get_coeff("I") == pytest.approx(1.0)

        # Unknown Pauli symbol raises.
        monkeypatch.setattr(target, lambda term, nqubits: "A")
        with pytest.raises(ValueError, match="Unknown Pauli"):
            _create_projector_observable("0", 1)


class TestExecutorDerivativeBranches:
    def test_derivatives_accept_list_and_iterable_derivative_params(self):
        executor = PauliPropagationExecutor()
        theta = Parameters("theta", 1)
        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, theta[0])
        observable = PauliPropagationOperator(["Z"], [1.0])

        result_list = executor.expectation_value_derivatives(
            circuit, observable, ["theta[0]"], theta=[0.0]
        )
        result_iterable = executor.expectation_value_derivatives(
            circuit,
            observable,
            Parameters("theta", 1),
            theta=[0.0],
        )

        assert isinstance(result_list, float)
        assert isinstance(result_iterable, float)

    def test_derivatives_with_generic_inputs_trigger_transpile_paths(self):
        executor = PauliPropagationExecutor()
        theta = Parameters("theta", 1)

        circuit = QuantumCircuit(1)
        circuit.rx(0, theta[0])
        observable = QuantumOperator(["Z"], [1.0], 1)

        derivative = executor.expectation_value_derivatives(
            circuit,
            observable,
            "theta",
            theta=[0.0],
        )

        assert isinstance(derivative, (float, np.ndarray))

    def test_derivatives_indexed_parameter_paths_and_grouped_output(self):
        executor = PauliPropagationExecutor()
        theta = Parameters("theta", 2)

        circuit = PauliPropagationCircuit(2)
        circuit.rx(0, theta[0])
        circuit.ry(1, theta[1])
        observable = PauliPropagationOperator(["ZI"], [1.0])

        result = executor.expectation_value_derivatives(
            circuit,
            observable,
            "theta[0]",
            "theta[1]",
            theta=[0.0, 0.0],
        )

        assert isinstance(result, dict)
        assert "theta" in result
        assert isinstance(result["theta"], np.ndarray)

    def test_derivatives_raise_for_missing_parameter(self):
        executor = PauliPropagationExecutor()
        theta = Parameters("theta", 1)
        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, theta[0])
        observable = PauliPropagationOperator(["Z"], [1.0])

        with pytest.raises(ValueError, match="not found in provided values"):
            executor.expectation_value_derivatives(circuit, observable, "theta")

    def test_derivatives_non_numeric_index_uses_base_parameter_values(self):
        executor = PauliPropagationExecutor()
        circuit = PauliPropagationCircuit(1)
        observable = PauliPropagationOperator(["Z"], [1.0])

        derivative = executor._expectation_value_derivatives(
            circuit,
            observable,
            "theta[a]",
            theta=[0.2],
        )

        assert np.isclose(derivative, 0.0)

    def test_derivatives_cover_indexed_name_pass_branch(self):
        class ToggleContainsStr(str):
            def __new__(cls, value):
                obj = str.__new__(cls, value)
                obj._contains_calls = 0
                return obj

            def __contains__(self, item):
                self._contains_calls += 1
                if self._contains_calls in (1, 2):
                    return False
                return super().__contains__(item)

        executor = PauliPropagationExecutor()
        circuit = PauliPropagationCircuit(1)
        observable = PauliPropagationOperator(["Z"], [1.0])
        weird_param = ToggleContainsStr("theta[0]")

        derivative = executor._expectation_value_derivatives(
            circuit,
            observable,
            weird_param,
            **{"theta[0]": 0.2},
        )

        assert np.isclose(derivative, 0.0)

    def test_derivatives_cover_circuit_continue_branches(self):
        executor = PauliPropagationExecutor()
        theta = sp.Symbol("theta")
        phi = sp.Symbol("phi")

        circuit = PauliPropagationCircuit(1)
        circuit.h(0)
        circuit.rx(0, 0.1)
        circuit.rx(0, phi)
        circuit.rx(0, sp.sin(theta) ** 2 + sp.cos(theta) ** 2)
        observable = PauliPropagationOperator(["Z"], [1.0])

        result = executor.expectation_value_derivatives(
            circuit,
            observable,
            "theta",
            theta=0.3,
            phi=0.2,
        )

        assert np.isfinite(result)

    def test_derivatives_cover_param_symbol_fallback_search(self):
        class WeirdDict(dict):
            def __contains__(self, key):
                return False

        executor = PauliPropagationExecutor()
        alpha = sp.Symbol("alpha")
        observable = PauliPropagationOperator(["Z"], [alpha])
        observable._parameters = WeirdDict(observable._parameters)
        circuit = PauliPropagationCircuit(1)

        derivative = executor.expectation_value_derivatives(
            circuit,
            observable,
            "alpha",
            alpha=0.4,
        )

        assert np.isfinite(derivative)

    def test_derivatives_cover_effective_param_name_lookup(self):
        executor = PauliPropagationExecutor()
        theta0 = sp.Symbol("theta[0]")
        observable = PauliPropagationOperator(["Z"], [theta0])
        circuit = PauliPropagationCircuit(1)

        derivative = executor.expectation_value_derivatives(
            circuit,
            observable,
            "theta",
            theta=[0.3],
        )

        assert np.isfinite(derivative)

    def test_derivatives_cover_param_name_lookup(self):
        executor = PauliPropagationExecutor()
        theta = sp.Symbol("theta")
        observable = PauliPropagationOperator(["Z"], [theta])
        circuit = PauliPropagationCircuit(1)

        derivative = executor.expectation_value_derivatives(
            circuit,
            observable,
            "theta",
            theta=[0.3, 0.4],
        )

        assert isinstance(derivative, np.ndarray)

    def test_derivatives_cover_base_name_lookup(self):
        class BaseOnlyContainsDict(dict):
            def __contains__(self, key):
                return key == "theta"

        executor = PauliPropagationExecutor()
        theta = sp.Symbol("theta")
        theta_indexed = sp.Symbol("theta[1]")
        observable = PauliPropagationOperator(["Z"], [theta])
        observable._parameters = BaseOnlyContainsDict({"theta": theta, "theta[1]": theta_indexed})
        circuit = PauliPropagationCircuit(1)
        executor.expectation_value = lambda *args, **kwargs: 1.0

        derivative = executor._expectation_value_derivatives(
            circuit,
            observable,
            "theta[1]",
            **{"theta[1]": [0.0, 0.4]},
        )

        assert np.isfinite(derivative)

    def test_derivatives_cover_effective_symbol_base_name_fallback(self):
        class FlakyGetDict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._calls = {}

            def get(self, key, default=None):
                self._calls[key] = self._calls.get(key, 0) + 1
                if self._calls[key] == 1:
                    return None
                return super().get(key, default)

        executor = PauliPropagationExecutor()
        theta = sp.Symbol("theta")
        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, theta)
        circuit._parameters = FlakyGetDict(circuit._parameters)
        observable = PauliPropagationOperator(["Z"], [1.0])

        derivative = executor.expectation_value_derivatives(
            circuit,
            observable,
            "theta",
            theta=0.0,
        )

        assert np.isfinite(derivative)

    def test_derivatives_single_parameter_returns_array_for_multiple_values(self):
        executor = PauliPropagationExecutor()
        theta = Parameters("theta", 2)
        circuit = PauliPropagationCircuit(2)
        circuit.rx(0, theta[0])
        circuit.ry(1, theta[1])
        observable = PauliPropagationOperator(["ZI"], [1.0])

        derivative = executor.expectation_value_derivatives(
            circuit,
            observable,
            "theta",
            theta=[0.0, 0.1],
        )

        assert isinstance(derivative, np.ndarray)

    def test_derivatives_single_parameter_scalar_return_branch(self, monkeypatch):
        executor = PauliPropagationExecutor()
        theta = Parameters("theta", 1)
        circuit = PauliPropagationCircuit(1)
        circuit.rx(0, theta[0])
        observable = PauliPropagationOperator(["Z"], [1.0])

        real_array = ppe.np.array
        monkeypatch.setattr(
            ppe.np,
            "array",
            lambda values: (
                values[0] if isinstance(values, list) and len(values) == 1 else real_array(values)
            ),
        )

        derivative = executor.expectation_value_derivatives(
            circuit,
            observable,
            "theta[0]",
            **{"theta[0]": 0.0},
        )

        assert isinstance(derivative, float)


class TestExecutorSimulationInternals:
    def test_shots_setter(self):
        executor = PauliPropagationExecutor(shots=10)
        executor.shots = 25
        assert executor.shots == 25

    def test_apply_two_qubit_gate_rejects_same_qubit(self):
        state = np.array([1, 0, 0, 0], dtype=complex)
        matrix = np.eye(4, dtype=complex)

        with pytest.raises(ValueError, match="distinct qubits"):
            PauliPropagationExecutor._apply_two_qubit_gate(state, matrix, 0, 0, 2)

    def test_resolve_angle_rejects_missing_expr_and_value(self):
        gate = PauliRotation(["X"], 0, 1, param_expr=None, param_value=None)

        with pytest.raises(ValueError, match="neither param_expr nor param_value"):
            PauliPropagationExecutor._resolve_angle(gate, {})

    def test_simulate_statevector_covers_gate_dispatch_and_errors(self):
        executor = PauliPropagationExecutor()

        circuit = PauliPropagationCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cz(0, 1)
        circuit.swap(0, 1)
        circuit.rxx(0, 1, 0.2)
        circuit.barrier([0, 1])
        state = executor._simulate_statevector(circuit, {})
        assert state.shape == (4,)

        bad_clifford = PauliPropagationCircuit(1)
        bad_clifford._gates.append(CliffordGate("X", 0, 1))
        bad_clifford._gates[0].gate_type = "ECR"
        with pytest.raises(ValueError, match="Unsupported Clifford gate type"):
            executor._simulate_statevector(bad_clifford, {})

        bad_rotation = PauliPropagationCircuit(3)
        bad_rotation._gates.append(PauliRotation(["X", "Y", "Z"], [0, 1, 2], 3, param_value=0.1))
        with pytest.raises(ValueError, match="Only 1- and 2-qubit Pauli rotations"):
            executor._simulate_statevector(bad_rotation, {})

        bad_object = PauliPropagationCircuit(1)
        bad_object._gates.append(object())
        with pytest.raises(TypeError, match="Unsupported gate object"):
            executor._simulate_statevector(bad_object, {})

    def test_sample_and_statevector_input_validation_and_warning(self):
        executor = PauliPropagationExecutor(shots=16, seed=7)

        with pytest.raises(TypeError, match="PauliPropagationCircuit"):
            executor.sample("not a circuit")

        with pytest.raises(TypeError, match="PauliPropagationCircuit"):
            executor.statevector("not a circuit")

        large = PauliPropagationCircuit(16)
        with pytest.warns(RuntimeWarning, match="memory-intensive"):
            sv = executor.statevector(large)
        assert sv.shape == (2**16,)


class TestExecutorTranspileAndCaching:
    def test_transpile_circuit(self):
        executor = PauliPropagationExecutor()
        circuit = PauliPropagationCircuit(1)

        transpiled = executor._transpile_circuit(circuit)

        assert isinstance(transpiled, PauliPropagationCircuit)

    def test_transpile_operator_list(self):
        executor = PauliPropagationExecutor()
        operators = [
            PauliPropagationOperator(["Z"], [1.0]),
            PauliPropagationOperator(["X"], [1.0]),
        ]

        transpiled = executor.transpile_operator(operators)

        assert isinstance(transpiled, list)
        assert len(transpiled) == 2
        assert all(isinstance(op, PauliPropagationOperator) for op in transpiled)

    def test_transpile_operator_cached_hit_and_wrapper(self):
        executor = PauliPropagationExecutor(caching=True)
        operator = PauliPropagationOperator(["Z"], [1.0])

        first = executor.transpile_operator(operator)
        second = executor.transpile_operator(operator)
        direct = executor._transpile_operator(operator)

        assert first is second
        assert isinstance(direct, PauliPropagationOperator)


def empty_circuit(nqubits: int = 2) -> PauliPropagationCircuit:
    """A circuit with no gates (observable is left unchanged by propagation)."""
    return PauliPropagationCircuit(nqubits)


def make_observable(terms: Dict[str, complex], nqubits: int = 2) -> PauliPropagationOperator:
    """Build an operator from a {pauli_string: coeff} dictionary."""
    paulis = list(terms.keys())
    coeffs = list(terms.values())
    return PauliPropagationOperator(paulis=paulis, coeffs=coeffs, num_qubits=nqubits)


class TestTruncationByThreshold:

    def test_no_truncation_without_settings(self):
        """Without any truncation settings, last_truncation_stats stays None."""
        executor = PauliPropagationExecutor()
        circuit = empty_circuit()
        observable = make_observable({"II": 0.5, "ZZ": 0.01})

        executor.expectation_value(circuit, observable)

        assert executor.get_truncation_stats() is None

    def test_small_coefficient_is_removed(self):
        """A term below the threshold must be counted as removed."""
        threshold = 0.1
        executor = PauliPropagationExecutor(truncate_threshold=threshold)
        circuit = empty_circuit()
        observable = make_observable({"II": 0.5, "ZZ": 0.05})

        executor.expectation_value(circuit, observable)

        stats = executor.get_truncation_stats()
        assert stats is not None
        assert stats.terms_removed == 1
        assert stats.terms_remaining == 1

    def test_all_terms_above_threshold_are_kept(self):
        executor = PauliPropagationExecutor(truncate_threshold=1e-6)
        obs = make_observable({"II": 0.5, "ZZ": 0.3, "XX": 0.2})

        executor.expectation_value(empty_circuit(), obs)

        stats = executor.get_truncation_stats()
        assert stats.terms_removed == 0
        assert stats.terms_remaining == 3

    def test_all_terms_below_threshold_are_removed(self):
        executor = PauliPropagationExecutor(truncate_threshold=1.0)
        obs = make_observable({"II": 0.1, "ZZ": 0.05})

        executor.expectation_value(empty_circuit(), obs)

        stats = executor.get_truncation_stats()
        assert stats.terms_removed == 2
        assert stats.terms_remaining == 0


class TestTruncationByMaxWeight:

    def test_terms_above_max_weight_are_removed(self):
        executor = PauliPropagationExecutor(max_weight=1)
        circuit = empty_circuit()

        observable = make_observable({"II": 0.5, "ZI": 0.3, "ZZ": 0.2})

        executor.expectation_value(circuit, observable)

        stats = executor.get_truncation_stats()
        assert stats is not None
        assert stats.terms_removed == 1
        assert stats.terms_remaining == 2

    def test_max_weight_equal_to_nqubits_removes_nothing(self):
        """When max_weight == nqubits no term can exceed it."""
        nq = 3
        executor = PauliPropagationExecutor(max_weight=nq)
        obs = make_observable({"III": 0.5, "ZII": 0.2, "ZZI": 0.2, "ZZZ": 0.1}, nqubits=nq)

        executor.expectation_value(empty_circuit(nq), obs)

        stats = executor.get_truncation_stats()
        assert stats.terms_removed == 0
        assert stats.terms_remaining == 4


class TestCombinedTruncation:

    def test_term_failing_only_threshold_is_removed(self):
        """Weight-1 term with tiny coeff: weight OK but coeff too small → removed."""
        executor = PauliPropagationExecutor(truncate_threshold=0.1, max_weight=2)
        obs = make_observable({"II": 0.5, "ZI": 0.01})

        executor.expectation_value(empty_circuit(), obs)

        stats = executor.get_truncation_stats()
        assert stats.terms_removed == 1

    def test_term_failing_only_weight_is_removed(self):
        """Weight-2 term with large coeff: coeff OK but weight too high → removed."""
        executor = PauliPropagationExecutor(truncate_threshold=0.01, max_weight=1)
        obs = make_observable({"II": 0.5, "ZZ": 0.9})

        executor.expectation_value(empty_circuit(), obs)

        stats = executor.get_truncation_stats()
        assert stats.terms_removed == 1

    def test_both_criteria_applied_simultaneously(self):
        """
        Terms:
          II  w=0 coeff=0.5  → kept
          ZI  w=1 coeff=0.05 → removed (coeff below threshold)
          ZZ  w=2 coeff=0.4  → removed (weight too high)
          XI  w=1 coeff=0.3  → kept
        """
        executor = PauliPropagationExecutor(truncate_threshold=0.1, max_weight=1)
        obs = make_observable({"II": 0.5, "ZI": 0.05, "ZZ": 0.4, "XI": 0.3})

        executor.expectation_value(empty_circuit(), obs)

        stats = executor.get_truncation_stats()
        assert stats.terms_removed == 2
        assert stats.terms_remaining == 2
