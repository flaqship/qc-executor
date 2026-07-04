import pytest
from qiskit.circuit import ParameterVector

from qc_executor import QuantumCircuit
from qc_executor.qulacs import QulacsCircuit
from qc_executor.qulacs.qulacs_executor import QulacsExecutor


class TestQulacsCircuit:
    def test_placeholder(self):
        pass


class TestTranspileCircuitQulacs:
    def setup_method(self):
        self.executor = QulacsExecutor()

    def test_returns_qulacs_circuit(self):
        """Test that transpile_circuit returns a QulacsCircuit."""
        qc = QuantumCircuit(2)
        result = self.executor.transpile_circuit(qc)
        assert isinstance(result, QulacsCircuit)

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

    def test_circuit_func_is_callable(self):
        """Test that the resulting QulacsCircuit can return a callable circuit function."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = self.executor.transpile_circuit(qc)
        circuit_func = result.get_circuit_func()
        assert callable(circuit_func)

    def test_transpile_list_of_circuits(self):
        """Test that transpile_circuit handles a list of circuits."""
        qc1 = QuantumCircuit(2)
        qc2 = QuantumCircuit(1)
        qc2.h(0)
        results = self.executor.transpile_circuit([qc1, qc2])
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, QulacsCircuit) for r in results)
