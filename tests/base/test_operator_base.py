from qc_executor.base.operator_base import QuantumOperatorBase


class ConcreteOperator(QuantumOperatorBase):
    def __init__(self):
        super().__init__(num_qubits=2)
        self._paulis = ["ZI", "IZ"]
        self._coeffs = [1.0, -1.0]

    @classmethod
    def from_quantum_operator(cls, operator: "QuantumOperatorBase") -> "QuantumOperatorBase":
        return operator

    def compose(self, other: "QuantumOperatorBase"):
        return self

    def simplify(self):
        return self


class TestOperatorBaseContract:
    def test_base_properties_expose_internal_fields(self):
        op = ConcreteOperator()

        assert op.num_qubits == 2
        assert op.paulis == ["ZI", "IZ"]
        assert op.coeffs == [1.0, -1.0]
        assert op.num_paulis == 2
