from types import SimpleNamespace

from qiskit import QuantumCircuit

from qc_executor.qiskit._decompose import decompose_to_std


class FakeCircuit:
    def __init__(self, names, decompose_returns=None):
        self.data = [SimpleNamespace(operation=SimpleNamespace(name=name)) for name in names]
        self._decompose_returns = decompose_returns or []
        self.calls = []

    def decompose(self, names):
        self.calls.append(list(names))
        if self._decompose_returns:
            return self._decompose_returns.pop(0)
        return self


def test_decompose_to_std_default_gate_list_returns_circuit():
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)

    result = decompose_to_std(circuit)

    assert isinstance(result, QuantumCircuit)


def test_decompose_to_std_recursive_decomposition_path():
    c3 = FakeCircuit(names=["cx"])
    c2 = FakeCircuit(names=["rz", "cx"], decompose_returns=[c3])
    c1 = FakeCircuit(names=["custom", "cx"], decompose_returns=[c2])

    result = decompose_to_std(c1, gate_list=["cx", "measure"])

    assert result is c3
    assert c1.calls == [["custom"]]
    assert c2.calls == [["rz"]]
    assert c3.calls == [[]]


def test_decompose_to_std_no_decomposition_needed():
    c1 = FakeCircuit(names=["cx", "measure"])

    result = decompose_to_std(c1, gate_list=["cx", "measure"])

    assert result is c1
    assert c1.calls == [[]]
