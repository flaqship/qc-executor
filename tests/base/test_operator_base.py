import executor.base as base_module
from executor.base.operator_base import QuantumOperatorBase


class ConcreteOperator(QuantumOperatorBase):
    def __init__(self):
        super().__init__(num_qubits=2)
        self._paulis = ["ZI", "IZ"]
        self._coeffs = [1.0, -1.0]

    @classmethod
    def from_quantum_operator(cls, operator: "QuantumOperatorBase") -> "QuantumOperatorBase":
        return operator

    def adjoint(self):
        return self

    def apply_layout(self, layout: dict):
        return self

    def compose(self, other: "QuantumOperatorBase"):
        return self

    def append(self, pauli: str, coeff):
        self._paulis.append(pauli)
        self._coeffs.append(coeff)
        return self

    def simplify(self):
        return self

    def transpose(self):
        return self

    def conjugate(self):
        return self

    def group_commuting(self):
        return [self]


class TestOperatorBaseContract:
    def test_base_properties_expose_internal_fields(self):
        op = ConcreteOperator()

        assert op.num_qubits == 2
        assert op.paulis == ["ZI", "IZ"]
        assert op.coeffs == [1.0, -1.0]
        assert op.num_paulis == 2


class TestBaseModuleApi:
    def test_base_module_all_exports(self):
        expected = {"ExecutorBase", "QuantumCircuitBase", "QuantumOperatorBase"}
        assert set(base_module.__all__) == expected

    def test_base_module_exports_point_to_classes(self):
        from executor.base.circuit_base import QuantumCircuitBase
        from executor.base.executor_base import ExecutorBase
        from executor.base.operator_base import QuantumOperatorBase

        assert base_module.ExecutorBase is ExecutorBase
        assert base_module.QuantumCircuitBase is QuantumCircuitBase
        assert base_module.QuantumOperatorBase is QuantumOperatorBase

    def test_parameters_base_module_is_importable(self):
        import executor.base.parameters_base as parameters_base

        assert parameters_base is not None
