"""Tests carried over from ``integration-support`` for behaviour sQUlearn relies on.

Merged in with ``integration-support``; the tests both branches share live in
``test_quantum_circuit.py``, which is kept exactly as on ``integration-support-ir``.
"""

from qiskit.circuit import ParameterVector

from qc_executor import QuantumCircuit
from tests.integration_support_helpers import FakeOperator, SpyCircuit


def create_mock_operator(paulis, coeffs):
    """Create a generic operator with the given pauli labels and coeffs."""
    if len(paulis) > 1:
        # Multi-term operator: keep one label per coefficient.
        return FakeOperator(list(paulis), coeffs)
    return FakeOperator(paulis[0], coeffs)


class TestQuantumCircuitBasics:
    def test_parameter_vector_names_in_parameter_order(self):
        circuit = QuantumCircuit(2)
        x = ParameterVector("x", 2)
        y = ParameterVector("y", 1)
        circuit.rx(0, x[0])
        circuit.ry(1, x[1])
        circuit.rz(0, y[0])

        # qiskit orders parameters alphabetically: x[0], x[1], y[0]
        assert circuit.parameter_vector_names == ["x", "y"]


class TestQuantumCircuitPauliString:

    def test_pauli_string_applies_in_qubit_order(self):
        circuit = SpyCircuit(3)

        circuit.pauli_string("XYZ")

        assert circuit.ops == [("x", 0), ("y", 1), ("z", 2)]
