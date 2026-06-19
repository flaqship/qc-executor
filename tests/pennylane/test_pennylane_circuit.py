"""
Test suite for PennyLane circuit conversion.

This module tests the PennyLaneCircuit class which converts Qiskit quantum circuits
to PennyLane format, including:
- Non-parametric gates (H, X, Y, Z, CNOT, etc.)
- Parametric gates (RX, RY, RZ, etc.)
- Parameter expressions
- Circuit properties and methods
"""

from unittest.mock import MagicMock

import numpy as np
import pennylane as qml
import pytest
from qiskit.circuit import ParameterVector
from qiskit.circuit import QuantumCircuit as QiskitQuantumCircuit

from executor import QuantumCircuit
from executor.pennylane.pennylane_circuit import PennyLaneCircuit
from executor.pennylane.pennylane_executor import PennyLaneExecutor


def _build_conditioned_gate(qc, gate_name, qubit, clbits_or_creg, val):
    """Helper function to build a circuit with a conditioned gate."""
    getattr(qc, gate_name)(qubit)
    op = qc.data[-1].operation.to_mutable()
    op.condition = (clbits_or_creg, val)
    qc.data[-1] = qc.data[-1].replace(operation=op)
    return qc


class TestPennyLaneCircuit:
    """Test suite for PennyLane circuit conversion."""

    # Basic Circuit Tests (Non-Parametric Gates)

    def test_empty_circuit(self):
        """Test conversion of an empty circuit."""
        qc = QuantumCircuit(2)
        plc = PennyLaneCircuit(qc)

        assert plc.num_qubits == 2
        assert len(plc.parameter_names) == 0
        assert isinstance(plc.hash, int)

    def test_single_hadamard_gate(self):
        """Test circuit with a single Hadamard gate."""
        qc = QuantumCircuit(1)
        qc.h(0)

        plc = PennyLaneCircuit(qc)
        assert plc.num_qubits == 1
        assert len(plc.parameter_names) == 0

    def test_bell_state_circuit(self):
        """Test Bell state preparation: H(0), CNOT(0,1)."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        plc = PennyLaneCircuit(qc)
        assert plc.num_qubits == 2
        assert len(plc.parameter_names) == 0

    @pytest.mark.parametrize("gate_name", ["x", "y", "z", "s", "t"])
    def test_single_qubit_gates(self, gate_name):
        """Test individual single-qubit Pauli and phase gates."""
        qc = QuantumCircuit(1)
        getattr(qc, gate_name)(0)

        plc = PennyLaneCircuit(qc)
        assert plc.num_qubits == 1
        assert len(plc.parameter_names) == 0

    @pytest.mark.parametrize(
        "gate_name,num_qubits",
        [
            ("cx", 2),  # CNOT
            ("cy", 2),  # Controlled-Y
            ("cz", 2),  # Controlled-Z
            ("swap", 2),  # SWAP
        ],
    )
    def test_two_qubit_gates(self, gate_name, num_qubits):
        """Test two-qubit gates."""
        qc = QuantumCircuit(num_qubits)
        getattr(qc, gate_name)(0, 1)

        plc = PennyLaneCircuit(qc)
        assert plc.num_qubits == num_qubits
        assert len(plc.parameter_names) == 0

    def test_multi_qubit_chain(self):
        """Test a chain of CNOT gates across multiple qubits."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)

        plc = PennyLaneCircuit(qc)
        assert plc.num_qubits == 4

    # Parametric Gates Tests

    @pytest.mark.parametrize(
        "theta,gate_name",
        [
            (np.pi / 4, "rx"),
            (np.pi / 2, "ry"),
            (np.pi, "rz"),
            (0.5, "rx"),
            (2 * np.pi, "ry"),
        ],
    )
    def test_single_qubit_rotation_gates_with_float(self, theta, gate_name):
        """Test single-qubit rotation gates with float angle."""
        qc = QuantumCircuit(1)
        getattr(qc, gate_name)(0, theta)

        plc = PennyLaneCircuit(qc)
        assert plc.num_qubits == 1
        # Float parameters are not tracked as parameters
        assert len(plc.parameter_names) == 0

    @pytest.mark.parametrize(
        "theta,gate_name",
        [
            (np.pi / 4, "crx"),
            (np.pi / 2, "cry"),
            (np.pi, "crz"),
        ],
    )
    def test_controlled_rotation_gates_with_float(self, theta, gate_name):
        """Test controlled rotation gates with float angle."""
        qc = QuantumCircuit(2)
        getattr(qc, gate_name)(0, 1, theta)

        plc = PennyLaneCircuit(qc)
        assert plc.num_qubits == 2
        assert len(plc.parameter_names) == 0

    @pytest.mark.parametrize("theta", [np.pi / 4, np.pi / 2, np.pi, 0.25])
    def test_phase_gate_with_float(self, theta):
        """Test phase shift gate with float angle."""
        qc = QuantumCircuit(1)
        qc.p(0, theta)

        plc = PennyLaneCircuit(qc)
        assert plc.num_qubits == 1
        assert len(plc.parameter_names) == 0

    @pytest.mark.parametrize("theta", [np.pi / 4, np.pi / 2])
    def test_controlled_phase_gate_with_float(self, theta):
        """Test controlled phase shift gate with float angle."""
        qc = QuantumCircuit(2)
        qc.cp(0, 1, theta)

        plc = PennyLaneCircuit(qc)
        assert plc.num_qubits == 2
        assert len(plc.parameter_names) == 0

    # Parameter Vector Tests

    def test_single_parameter(self):
        """Test circuit with a single parameter."""
        x = ParameterVector("x", 1)
        qc = QuantumCircuit(1)
        qc.rx(0, x[0])

        plc = PennyLaneCircuit(qc)
        assert "x" in plc.parameter_names
        assert plc.parameter_dimensions["x"] == 1

    def test_multiple_parameters_same_vector(self):
        """Test circuit with multiple parameters from the same vector."""
        x = ParameterVector("x", 3)
        qc = QuantumCircuit(3)
        qc.rx(0, x[0])
        qc.ry(1, x[1])
        qc.rz(2, x[2])

        plc = PennyLaneCircuit(qc)
        assert "x" in plc.parameter_names
        assert plc.parameter_dimensions["x"] == 3

    def test_multiple_parameter_vectors(self):
        """Test circuit with multiple different parameter vectors."""
        x = ParameterVector("x", 2)
        y = ParameterVector("y", 1)
        qc = QuantumCircuit(3)
        qc.rx(0, x[0])
        qc.ry(1, x[1])
        qc.rz(2, y[0])

        plc = PennyLaneCircuit(qc)
        assert "x" in plc.parameter_names
        assert "y" in plc.parameter_names
        assert plc.parameter_dimensions["x"] == 2
        assert plc.parameter_dimensions["y"] == 1

    def test_parametric_two_qubit_gates(self):
        """Test two-qubit gates with parameters."""
        theta = ParameterVector("theta", 3)
        qc = QuantumCircuit(2)
        qc.crx(0, 1, theta[0])
        qc.cry(0, 1, theta[1])
        qc.crz(0, 1, theta[2])

        plc = PennyLaneCircuit(qc)
        assert "theta" in plc.parameter_names
        assert plc.parameter_dimensions["theta"] == 3

    # Parameter Expression Tests

    def test_parameter_arithmetic_multiplication(self):
        """Test parameter expression with multiplication: 2 * x[0]."""
        x = ParameterVector("x", 1)
        qc = QuantumCircuit(1)
        qc.rx(0, 2 * x[0])

        plc = PennyLaneCircuit(qc)
        assert "x" in plc.parameter_names

    def test_parameter_arithmetic_addition(self):
        """Test parameter expression with addition: x[0] + 0.5."""
        x = ParameterVector("x", 1)
        qc = QuantumCircuit(1)
        qc.rx(0, x[0] + 0.5)

        plc = PennyLaneCircuit(qc)
        assert "x" in plc.parameter_names

    def test_parameter_arithmetic_subtraction(self):
        """Test parameter expression with subtraction: x[0] - 0.2."""
        x = ParameterVector("x", 1)
        qc = QuantumCircuit(1)
        qc.rx(0, x[0] - 0.2)

        plc = PennyLaneCircuit(qc)
        assert "x" in plc.parameter_names

    def test_parameter_arithmetic_division(self):
        """Test parameter expression with division: x[0] / 2."""
        x = ParameterVector("x", 1)
        qc = QuantumCircuit(1)
        qc.rx(0, x[0] / 2)

        plc = PennyLaneCircuit(qc)
        assert "x" in plc.parameter_names

    def test_parameter_multiplication_between_vectors(self):
        """Test parameter expression multiplying two different vectors: x[0] * y[0]."""
        x = ParameterVector("x", 1)
        y = ParameterVector("y", 1)
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.crx(0, 1, x[0] * y[0])

        plc = PennyLaneCircuit(qc)
        assert "x" in plc.parameter_names
        assert "y" in plc.parameter_names

    def test_complex_parameter_expression(self):
        """Test complex parameter expression: 2 * x[0] - 1."""
        x = ParameterVector("x", 1)
        qc = QuantumCircuit(1)
        qc.rx(0, 2 * x[0] - 1)

        plc = PennyLaneCircuit(qc)
        assert "x" in plc.parameter_names


class TestPennyLanePropertiesAndMethods:

    def test_num_qubits_property(self):
        """Test that num_qubits property returns correct value."""
        qc = QuantumCircuit(5)
        plc = PennyLaneCircuit(qc)
        assert plc.num_qubits == 5

    def test_parameter_names_property_empty(self):
        """Test parameter_names property for non-parametric circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        plc = PennyLaneCircuit(qc)

        assert isinstance(plc.parameter_names, list)
        assert len(plc.parameter_names) == 0

    def test_parameter_names_property_with_params(self):
        """Test parameter_names property for parametric circuit."""
        x = ParameterVector("x", 2)
        y = ParameterVector("y", 1)
        qc = QuantumCircuit(2)
        qc.rx(0, x[0])
        qc.ry(1, x[1])
        qc.rz(0, y[0])
        plc = PennyLaneCircuit(qc)

        assert "x" in plc.parameter_names
        assert "y" in plc.parameter_names

    def test_parameter_dimensions_property(self):
        """Test parameter_dimensions property."""
        x = ParameterVector("x", 3)
        qc = QuantumCircuit(3)
        qc.rx(0, x[0])
        qc.rx(1, x[1])
        qc.rx(2, x[2])
        plc = PennyLaneCircuit(qc)

        assert isinstance(plc.parameter_dimensions, dict)
        assert plc.parameter_dimensions["x"] == 3

    def test_hash_property(self):
        """Test that hash property returns a valid integer."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        plc = PennyLaneCircuit(qc)

        hash_value = plc.hash
        assert isinstance(hash_value, int)

    def test_hash_consistency(self):
        """Test that hash remains consistent for the same circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        plc = PennyLaneCircuit(qc)

        hash1 = plc.hash
        hash2 = plc.hash
        assert hash1 == hash2

    def test_build_pennylane_circuit_method(self):
        """Test that build_pennylane_circuit() returns callable."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        plc = PennyLaneCircuit(qc)

        pennylane_circuit = plc.build_pennylane_circuit()
        assert callable(pennylane_circuit)

    def test_get_pennylane_circuit_property(self):
        """Test that get_pennylane_circuit() returns callable and sets property."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        plc = PennyLaneCircuit(qc)

        pennylane_circuit = plc.pennylane_circuit
        assert callable(pennylane_circuit)
        assert hasattr(plc, "_pennylane_circuit")

    def test_call_method(self):
        """Test that PennyLaneCircuit instance is callable after building."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        plc = PennyLaneCircuit(qc)

        plc._pennylane_circuit = plc.build_pennylane_circuit()
        assert callable(plc)

    def test_call_forwards_args_and_kwargs(self):
        """Test that calling PennyLaneCircuit forwards args and kwargs to the internal circuit."""
        qc = QuantumCircuit(1)
        pl_circuit = PennyLaneCircuit(qc)

        called_with = {}

        def fake_pennylane_circuit(*args, **kwargs):
            called_with["args"] = args
            called_with["kwargs"] = kwargs
            return "result"

        pl_circuit._pennylane_circuit = fake_pennylane_circuit

        result = pl_circuit([1, 2], foo="bar")

        assert called_with["args"] == ([1, 2],)
        assert called_with["kwargs"] == {"foo": "bar"}
        assert result == "result"

    def test_no_condition_returns_none(self):
        """Test that _get_gate_condition returns None for gates without conditions."""
        qc = QiskitQuantumCircuit(1)
        qc.h(0)
        gate_op = qc.data[0]

        pl_circuit = PennyLaneCircuit.__new__(PennyLaneCircuit)  # ohne __init__
        result = pl_circuit._get_gate_condition(qc, gate_op)

        assert result is None

    def test_single_clbit_condition(self):
        """Test that _get_gate_condition correctly identifies a single clbit condition."""
        qc = QiskitQuantumCircuit(1, 1)
        qc = _build_conditioned_gate(qc, "h", 0, qc.clbits[0], 1)
        gate_op = qc.data[-1]

        pl_circuit = PennyLaneCircuit.__new__(PennyLaneCircuit)
        result = pl_circuit._get_gate_condition(qc, gate_op)

        assert result == (0, 1)

    def test_multi_clbit_condition(self):
        """Test that _get_gate_condition correctly identifies a multi-clbit condition."""
        qc = QiskitQuantumCircuit(1, 2)
        qc = _build_conditioned_gate(qc, "h", 0, qc.cregs[0], 2)
        gate_op = qc.data[-1]

        pl_circuit = PennyLaneCircuit.__new__(PennyLaneCircuit)
        result = pl_circuit._get_gate_condition(qc, gate_op)

        bit_indices, val = result
        assert bit_indices == [0, 1]
        assert val == 2

    def test_int_condition_match_executes_gate(self):
        """Test that _apply_conditional_gate executes the gate when integer condition matches measurements."""
        pl_circuit = PennyLaneCircuit.__new__(PennyLaneCircuit)
        circuit_gate = MagicMock()
        measurements = [1, 0]
        condition = (0, 1)

        pl_circuit._apply_conditional_gate(
            circuit_gate, (0.5,), condition, measurements, wires=[0]
        )

        circuit_gate.assert_called_once_with(0.5, wires=[0])

    def test_int_condition_no_match_skips_gate(self):
        """Test that _apply_conditional_gate skips the gate when integer condition does not match measurements."""
        pl_circuit = PennyLaneCircuit.__new__(PennyLaneCircuit)
        circuit_gate = MagicMock()
        measurements = [0, 0]
        condition = (0, 1)

        pl_circuit._apply_conditional_gate(
            circuit_gate, (0.5,), condition, measurements, wires=[0]
        )

        circuit_gate.assert_not_called()

    def test_int_condition_no_param_executes_gate(self):
        """Test that _apply_conditional_gate executes the gate without parameters when integer condition matches measurements."""
        pl_circuit = PennyLaneCircuit.__new__(PennyLaneCircuit)
        circuit_gate = MagicMock()
        measurements = [1]
        condition = (0, 1)

        pl_circuit._apply_conditional_gate(circuit_gate, None, condition, measurements, wires=[0])

        circuit_gate.assert_called_once_with(wires=[0])

    def test_multi_bit_condition_computes_value(self):
        """Test that _apply_conditional_gate correctly computes the integer value from multiple bits and executes the gate if it matches the condition."""
        pl_circuit = PennyLaneCircuit.__new__(PennyLaneCircuit)
        circuit_gate = MagicMock()
        measurements = [1, 1]
        condition = ([0, 1], 3)

        pl_circuit._apply_conditional_gate(circuit_gate, None, condition, measurements, wires=[2])

        circuit_gate.assert_called_once_with(wires=[2])

    def test_mid_circuit_measurement_condition_with_param(self, monkeypatch):
        """Test that _apply_conditional_gate correctly uses qml.cond for mid-circuit measurement conditions with parameters."""
        pl_circuit = PennyLaneCircuit.__new__(PennyLaneCircuit)
        circuit_gate = MagicMock()
        condition = (0, 1)

        mock_measurement_value = MagicMock()
        mock_measurement_value.__eq__ = MagicMock(return_value="comparison_result")
        measurements = [mock_measurement_value]

        mock_cond_fn = MagicMock()
        mock_qml_cond = MagicMock(return_value=mock_cond_fn)
        monkeypatch.setattr(qml, "cond", mock_qml_cond)

        pl_circuit._apply_conditional_gate(
            circuit_gate, (0.5,), condition, measurements, wires=[0]
        )

        mock_qml_cond.assert_called_once_with("comparison_result", circuit_gate)
        mock_cond_fn.assert_called_once_with(0.5, wires=[0])

    def test_mid_circuit_measurement_condition_no_param(self, monkeypatch):
        """Test that _apply_conditional_gate correctly uses qml.cond for mid-circuit measurement conditions without parameters."""
        pl_circuit = PennyLaneCircuit.__new__(PennyLaneCircuit)
        circuit_gate = MagicMock()
        condition = (0, 1)

        mock_measurement_value = MagicMock()
        mock_measurement_value.__eq__ = MagicMock(return_value="comparison_result")
        measurements = [mock_measurement_value]

        mock_cond_fn = MagicMock()
        mock_qml_cond = MagicMock(return_value=mock_cond_fn)
        monkeypatch.setattr(qml, "cond", mock_qml_cond)

        pl_circuit._apply_conditional_gate(circuit_gate, None, condition, measurements, wires=[0])

        mock_cond_fn.assert_called_once_with(wires=[0])


class TestTranspileCircuitPennyLane:
    def setup_method(self):
        self.executor = PennyLaneExecutor()

    def test_returns_pennylane_circuit(self):
        """Test that transpile_circuit returns a PennyLaneCircuit."""
        qc = QuantumCircuit(2)
        result = self.executor.transpile_circuit(qc)
        assert isinstance(result, PennyLaneCircuit)

    def test_empty_circuit(self):
        """Test transpile_circuit with an empty circuit."""
        qc = QuantumCircuit(2)
        result = self.executor.transpile_circuit(qc)
        assert result.num_qubits == 2

    def test_single_gate_circuit(self):
        """Test transpile_circuit with a single Hadamard gate."""
        qc = QuantumCircuit(1)
        qc.h(0)
        result = self.executor.transpile_circuit(qc)
        assert result.num_qubits == 1

    def test_bell_state_circuit(self):
        """Test transpile_circuit preserves Bell state circuit structure."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = self.executor.transpile_circuit(qc)
        assert result.num_qubits == 2

    def test_parameterized_circuit_preserves_parameters(self):
        """Test that transpile_circuit preserves circuit parameters."""
        x = ParameterVector("x", 2)
        qc = QuantumCircuit(2)
        qc.rx(0, x[0])
        qc.ry(1, x[1])
        result = self.executor.transpile_circuit(qc)
        assert "x" in result.parameter_names
        assert result.parameter_dimensions["x"] == 2

    def test_circuit_is_callable(self):
        """Test that the resulting PennyLaneCircuit can build a callable circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = self.executor.transpile_circuit(qc)
        circuit_func = result.build_pennylane_circuit()
        assert callable(circuit_func)
