import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

import qc_executor.utils.qiskit_hash_functions as qh


def test_bits_key_contains_indices_and_register_metadata():
    if not hasattr(qh, "_bits_key"):
        return

    circuit = QuantumCircuit(2, 1)
    result = qh._bits_key((circuit.qubits[1],), circuit)

    assert isinstance(result, tuple)
    assert result[0][0] == 1


def test_format_params_variants():
    if not hasattr(qh, "_format_params"):
        return

    array_value = np.array([1, 2, 3])
    circuit_value = QuantumCircuit(1)

    assert isinstance(qh._format_params(array_value), bytes)
    assert isinstance(qh._format_params(circuit_value), tuple)
    assert qh._format_params([1, 2]) == (1, 2)
    assert qh._format_params("abc") == ("a", "b", "c")
    assert qh._format_params(7) == 7


def test_circuit_key_functional_true_and_false():
    circuit = QuantumCircuit(2, 1, name="my-circuit")
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure(1, 0)

    key_functional = qh._circuit_key(circuit, functional=True)
    key_full = qh._circuit_key(circuit, functional=False)

    assert isinstance(key_functional, tuple)
    assert isinstance(key_full, tuple)
    assert key_full[0] == "my-circuit"


def test_observable_key_contains_binary_payloads():
    observable = SparsePauliOp(["ZI"], coeffs=np.array([1.0 + 1.0j]))

    key = qh._observable_key(observable)

    assert isinstance(key, tuple)
    assert len(key) == 4
    assert all(isinstance(part, bytes) for part in key)
