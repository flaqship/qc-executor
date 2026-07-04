"""Tests for Qiskit circuit conversion."""

import numpy as np
import pytest

# Try to import Qiskit
try:
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit import Parameter

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from qc_executor.pauli_propagation.utils.gates import CliffordGate, PauliRotation

# Skip all tests if Qiskit is not available
pytestmark = pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")


@pytest.fixture
def qiskit_converter():
    """Import qiskit_converter module."""
    from qc_executor.pauli_propagation.utils import qiskit_converter as converter_module

    return converter_module


class TestConvertSingleGate:
    """Test conversion of single gates."""

    def test_convert_rx_gate(self, qiskit_converter):
        """Test RX gate conversion."""
        from qiskit.circuit.library import RXGate

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

    def test_barrier_skipped(self, qiskit_converter):
        """Test that barriers are converted to LayerBarrier markers."""
        from qc_executor.pauli_propagation.utils.gates import LayerBarrier

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
        qc = QuantumCircuit(1)
        # Try to add an unsupported gate (if we can construct one)
        # For now, skip this test as all common gates are supported
        pytest.skip("No unsupported gates to test with")
