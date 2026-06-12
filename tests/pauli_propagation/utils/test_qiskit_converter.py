"""Tests for Qiskit circuit conversion."""

import builtins
import importlib

import numpy as np
import pytest

# Try to import Qiskit
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from executor.pauli_propagation.utils.gates import CliffordGate, PauliRotation

# Skip all tests if Qiskit is not available
pytestmark = pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")


@pytest.fixture
def qiskit_converter():
    """Import qiskit_converter module."""
    from executor.pauli_propagation.utils import qiskit_converter as converter_module

    return converter_module


class TestConvertSingleGate:
    """Test conversion of single gates."""

    def test_convert_rx_gate(self, qiskit_converter):
        """Test RX gate conversion."""
        qc = QuantumCircuit(1)
        qc.rx(np.pi / 2, 0)

        gates = qiskit_converter.convert_circuit(qc, use_cache=False)
        assert len(gates) == 1
        assert isinstance(gates[0], PauliRotation)
        assert gates[0].symbols == ["X"]
        assert gates[0].qubits == [0]

    def test_convert_ry_gate(self, qiskit_converter):
        """Test RY gate conversion."""
        qc = QuantumCircuit(1)
        qc.ry(0.5, 0)

        gates = qiskit_converter.convert_circuit(qc, use_cache=False)
        assert len(gates) == 1
        assert isinstance(gates[0], PauliRotation)
        assert gates[0].symbols == ["Y"]

    def test_convert_rz_gate(self, qiskit_converter):
        """Test RZ gate conversion."""
        qc = QuantumCircuit(1)
        qc.rz(1.0, 0)

        gates = qiskit_converter.convert_circuit(qc, use_cache=False)
        assert len(gates) == 1
        assert isinstance(gates[0], PauliRotation)
        assert gates[0].symbols == ["Z"]

    def test_convert_hadamard(self, qiskit_converter):
        """Test Hadamard gate conversion."""
        qc = QuantumCircuit(1)
        qc.h(0)

        gates = qiskit_converter.convert_circuit(qc, use_cache=False)
        assert len(gates) == 1
        assert isinstance(gates[0], CliffordGate)
        assert gates[0].gate_type == "H"

    def test_convert_s_gate(self, qiskit_converter):
        """Test S gate conversion."""
        qc = QuantumCircuit(1)
        qc.s(0)

        gates = qiskit_converter.convert_circuit(qc, use_cache=False)
        assert len(gates) == 1
        assert isinstance(gates[0], CliffordGate)
        assert gates[0].gate_type == "S"

    def test_convert_cnot(self, qiskit_converter):
        """Test CNOT gate conversion."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)

        gates = qiskit_converter.convert_circuit(qc, use_cache=False)
        assert len(gates) == 1
        assert isinstance(gates[0], CliffordGate)
        assert gates[0].gate_type == "CNOT"
        assert gates[0].qubits == [0, 1]

    def test_convert_cz(self, qiskit_converter):
        """Test CZ gate conversion."""
        qc = QuantumCircuit(2)
        qc.cz(0, 1)

        gates = qiskit_converter.convert_circuit(qc, use_cache=False)
        assert len(gates) == 1
        assert isinstance(gates[0], CliffordGate)
        assert gates[0].gate_type == "CZ"

    def test_convert_swap(self, qiskit_converter):
        """Test SWAP gate conversion."""
        qc = QuantumCircuit(2)
        qc.swap(0, 1)

        gates = qiskit_converter.convert_circuit(qc, use_cache=False)
        assert len(gates) == 1
        assert isinstance(gates[0], CliffordGate)
        assert gates[0].gate_type == "SWAP"

    def test_convert_rxx_gate(self, qiskit_converter):
        """Test RXX gate conversion."""
        qc = QuantumCircuit(2)
        qc.rxx(0.5, 0, 1)

        gates = qiskit_converter.convert_circuit(qc, use_cache=False)
        assert len(gates) == 1
        assert isinstance(gates[0], PauliRotation)
        assert gates[0].symbols == ["X", "X"]
        assert gates[0].qubits == [0, 1]

    def test_convert_rzz_gate(self, qiskit_converter):
        """Test RZZ gate conversion."""
        qc = QuantumCircuit(2)
        qc.rzz(0.5, 0, 1)

        gates = qiskit_converter.convert_circuit(qc, use_cache=False)
        assert len(gates) == 1
        assert isinstance(gates[0], PauliRotation)
        assert gates[0].symbols == ["Z", "Z"]
        assert gates[0].qubits == [0, 1]

    def test_identity_gate_skipped(self, qiskit_converter):
        """Test that identity gate is skipped."""
        qc = QuantumCircuit(1)
        qc.id(0)

        gates = qiskit_converter.convert_circuit(qc, use_cache=False)
        assert gates == []

    def test_barrier_skipped(self, qiskit_converter):
        """Test that barriers are converted to LayerBarrier markers."""
        from executor.pauli_propagation.utils.gates import LayerBarrier

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier()
        qc.cx(0, 1)

        gates = qiskit_converter.convert_circuit(qc, use_cache=False)
        # Now barriers are converted to LayerBarrier markers instead of being skipped
        assert len(gates) == 3  # H, LayerBarrier, CNOT
        assert isinstance(gates[0], CliffordGate)  # H
        assert isinstance(gates[1], LayerBarrier)  # Barrier marker
        assert isinstance(gates[2], CliffordGate)  # CNOT


class TestParametricCircuits:
    """Test conversion of parametric circuits."""

    def test_parametric_rx(self, qiskit_converter):
        """Test parametric RX gate."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        gates = qiskit_converter.convert_circuit(qc, use_cache=False)
        assert len(gates) == 1
        assert gates[0].param_name == "theta"

    def test_multiple_parameters(self, qiskit_converter):
        """Test circuit with multiple parameters."""
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc = QuantumCircuit(2)
        qc.rx(theta, 0)
        qc.ry(phi, 1)

        gates = qiskit_converter.convert_circuit(qc, use_cache=False)
        assert len(gates) == 2
        assert gates[0].param_name == "theta"
        assert gates[1].param_name == "phi"

    def test_bind_parameters(self, qiskit_converter):
        """Test parameter binding."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        gates = qiskit_converter.convert_circuit(qc, use_cache=False)
        param_dict = qiskit_converter.bind_parameters(gates, {"theta": 0.5})

        assert "theta" in param_dict
        assert param_dict["theta"] == 0.5

    def test_bind_parameters_missing(self, qiskit_converter):
        """Test error on missing parameters."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        gates = qiskit_converter.convert_circuit(qc, use_cache=False)

        with pytest.raises(ValueError, match="Missing parameter"):
            qiskit_converter.bind_parameters(gates, {})

    def test_extract_parameter_constant_expression(self, qiskit_converter):
        """Test extracting a constant ParameterExpression."""
        from qiskit.circuit import ParameterExpression

        constant_expr = ParameterExpression({}, "0.25")

        param_expr, param_value = qiskit_converter._extract_parameter(constant_expr)
        assert param_expr is None
        assert param_value == pytest.approx(0.25)


class TestCircuitConversion:
    """Test full circuit conversion."""

    def test_simple_circuit(self, qiskit_converter):
        """Test conversion of simple circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.5, 1)

        gates = qiskit_converter.convert_circuit(qc, use_cache=False)
        assert len(gates) == 3
        assert isinstance(gates[0], CliffordGate)  # H
        assert isinstance(gates[1], CliffordGate)  # CNOT
        assert isinstance(gates[2], PauliRotation)  # RZ

    def test_multi_qubit_circuit(self, qiskit_converter):
        """Test multi-qubit circuit."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.cx(0, 1)
        qc.cx(1, 2)

        gates = qiskit_converter.convert_circuit(qc, use_cache=False)
        assert len(gates) == 5

        # Check qubit assignments
        assert gates[0].qubits == [0]
        assert gates[1].qubits == [1]
        assert gates[2].qubits == [2]
        assert gates[3].qubits == [0, 1]
        assert gates[4].qubits == [1, 2]


class TestCaching:
    """Test circuit conversion caching."""

    def test_caching_enabled(self, qiskit_converter):
        """Test that caching works."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        # Clear cache first
        qiskit_converter.clear_cache()

        # First conversion
        gates1 = qiskit_converter.convert_circuit(qc, use_cache=True)

        # Second conversion should use cache
        gates2 = qiskit_converter.convert_circuit(qc, use_cache=True)

        # Should be the same objects
        assert gates1 is gates2

    def test_caching_disabled(self, qiskit_converter):
        """Test that caching can be disabled."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        qiskit_converter.clear_cache()

        gates1 = qiskit_converter.convert_circuit(qc, use_cache=False)
        gates2 = qiskit_converter.convert_circuit(qc, use_cache=False)

        # Should be different objects
        assert gates1 is not gates2

    def test_clear_cache(self, qiskit_converter):
        """Test cache clearing."""
        qc = QuantumCircuit(1)
        qc.h(0)

        qiskit_converter.clear_cache()
        gates1 = qiskit_converter.convert_circuit(qc, use_cache=True)

        qiskit_converter.clear_cache()
        gates2 = qiskit_converter.convert_circuit(qc, use_cache=True)

        # After clearing, should be different objects
        assert gates1 is not gates2


class TestUnsupportedGates:
    """Test handling of unsupported gates."""

    def test_unsupported_gate_error(self, qiskit_converter):
        """Test error on unsupported gate."""
        from qiskit.circuit import Gate as QiskitGate

        qc = QuantumCircuit(1)
        qc.append(QiskitGate("my_unsupported_gate", 1, []), [0])

        with pytest.raises(ValueError, match="Unsupported gate"):
            qiskit_converter.convert_circuit(qc, use_cache=False)


class TestImportAndAvailabilityPaths:
    """Test import and availability error paths."""

    def test_import_sets_qiskit_unavailable_on_import_error(self):
        """Test module sets availability flag to False when qiskit import fails."""
        from executor.pauli_propagation.utils import qiskit_converter as converter_module

        original_import = builtins.__import__

        def mocked_import(name, globals_=None, locals_=None, fromlist=(), level=0):
            if name == "qiskit.circuit":
                raise ImportError("mocked qiskit import error")
            return original_import(name, globals_, locals_, fromlist, level)

        builtins.__import__ = mocked_import
        try:
            reloaded = importlib.reload(converter_module)
            assert reloaded.QISKIT_AVAILABLE is False
        finally:
            builtins.__import__ = original_import
            importlib.reload(converter_module)

    def test_convert_circuit_raises_when_qiskit_unavailable(self, qiskit_converter, monkeypatch):
        """Test convert_circuit raises if qiskit is unavailable."""
        monkeypatch.setattr(qiskit_converter, "QISKIT_AVAILABLE", False)

        with pytest.raises(ImportError, match="Qiskit is required for circuit conversion"):
            qiskit_converter.convert_circuit(object())

    def test_get_hash_raises_when_qiskit_unavailable(self, qiskit_converter, monkeypatch):
        """Test cache hash computation raises if qiskit is unavailable."""
        monkeypatch.setattr(qiskit_converter, "QISKIT_AVAILABLE", False)

        cache = qiskit_converter.CircuitConversionCache()
        with pytest.raises(ImportError, match="Qiskit is required for circuit conversion"):
            cache.get_hash(object())

    def test_extract_parameter_returns_none_when_qiskit_unavailable(
        self, qiskit_converter, monkeypatch
    ):
        """Test parameter extraction fallback when qiskit is unavailable."""
        monkeypatch.setattr(qiskit_converter, "QISKIT_AVAILABLE", False)

        param_expr, param_value = qiskit_converter._extract_parameter(0.5)
        assert param_expr is None
        assert param_value is None


class TestBindParametersEdgeCases:
    """Test bind_parameters edge cases and validation paths."""

    def test_bind_parameters_expands_list_values(self, qiskit_converter):
        """Test list/tuple values are expanded to indexed parameter keys."""
        result = qiskit_converter.bind_parameters([], {"theta": [0.1, 0.2], "phi": (0.3,)})

        assert result["theta[0]"] == pytest.approx(0.1)
        assert result["theta[1]"] == pytest.approx(0.2)
        assert result["phi[0]"] == pytest.approx(0.3)

    def test_bind_parameters_raises_for_invalid_list_item_type(self, qiskit_converter):
        """Test invalid list item type raises TypeError."""
        with pytest.raises(TypeError, match="invalid type"):
            qiskit_converter.bind_parameters([], {"theta": [0.1, "bad"]})

    def test_bind_parameters_raises_for_invalid_parameter_value_type(self, qiskit_converter):
        """Test invalid top-level parameter value type raises TypeError."""
        with pytest.raises(TypeError, match="invalid value type"):
            qiskit_converter.bind_parameters([], {"theta": {"not": "numeric"}})

    def test_bind_parameters_skips_layer_barrier(self, qiskit_converter):
        """Test LayerBarrier objects are skipped during parameter collection."""
        from executor.pauli_propagation.utils.gates import LayerBarrier

        result = qiskit_converter.bind_parameters([LayerBarrier()], {})
        assert result == {}

    def test_bind_parameters_raises_for_unexpected_object(self, qiskit_converter):
        """Test unexpected objects in gates list raise TypeError."""
        with pytest.raises(TypeError, match="Unexpected object in gates list"):
            qiskit_converter.bind_parameters([object()], {})

    def test_bind_parameters_uses_gate_param_value_when_missing(self, qiskit_converter):
        """Test missing parameter name gets filled from gate.param_value."""
        gate = PauliRotation(["X"], 0, 1, param_name="theta", param_value=0.7)
        gate.param_expr = None

        result = qiskit_converter.bind_parameters([gate], {})
        assert result["theta"] == pytest.approx(0.7)

    def test_bind_parameters_marks_missing_param_name_without_value(self, qiskit_converter):
        """Test missing named parameter without value raises ValueError."""
        gate = PauliRotation(["X"], 0, 1, param_name="theta")
        gate.param_expr = None
        gate.param_value = None

        with pytest.raises(ValueError, match="Missing parameter values"):
            qiskit_converter.bind_parameters([gate], {})
