"""Behavioral tests for QuantumCircuit features against qiskit references.

Migrated from fqsim's circuit-wrapper test suite: gate parity, the
parameter-merging compose, fixate_parameters, Pauli evolution against
qiskit's PauliEvolutionGate, control states, and native-circuit access.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit as QiskitCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector

from qc_executor import QuantumCircuit


def _statevector(circuit: QuantumCircuit, values=None) -> np.ndarray:
    """Return the qiskit-ordered statevector of a circuit."""
    qiskit_circuit = circuit.qiskit_circuit
    if values is not None:
        qiskit_circuit = qiskit_circuit.assign_parameters(
            dict(zip(qiskit_circuit.parameters, values))
        )
    return Statevector(qiskit_circuit).data


def _plus_plus() -> QiskitCircuit:
    """Return a two-qubit qiskit circuit preparing the |++> state."""
    circuit = QiskitCircuit(2)
    circuit.h([0, 1])
    return circuit


def test_gate_parity_with_qiskit_reference():
    """All primitive gates must match an equivalent raw qiskit circuit."""
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.x(1)
    circuit.y(2)
    circuit.z(0)
    circuit.s(1)
    circuit.sdag(2)
    circuit.rx(0, 0.1)
    circuit.ry(1, 0.2)
    circuit.rz(2, 0.3)
    circuit.cx(0, 1)
    circuit.cnot(1, 2)
    circuit.cy(0, 2)
    circuit.cz(1, 2)
    circuit.crx(0, 1, 0.4)
    circuit.cry(1, 2, 0.5)
    circuit.crz(2, 0, 0.6)
    circuit.swap(0, 1)
    circuit.cswap(0, 1, 2)

    reference = QiskitCircuit(3)
    reference.h(0)
    reference.x(1)
    reference.y(2)
    reference.z(0)
    reference.s(1)
    reference.sdg(2)
    reference.rx(0.1, 0)
    reference.ry(0.2, 1)
    reference.rz(0.3, 2)
    reference.cx(0, 1)
    reference.cx(1, 2)
    reference.cy(0, 2)
    reference.cz(1, 2)
    reference.crx(0.4, 0, 1)
    reference.cry(0.5, 1, 2)
    reference.crz(0.6, 2, 0)
    reference.swap(0, 1)
    reference.cswap(0, 1, 2)

    np.testing.assert_allclose(_statevector(circuit), Statevector(reference).data, atol=1e-12)


def test_compose_reindexes_identically_named_vectors():
    """Composing many one-parameter circuits must grow the parameter count."""
    num_blocks = 5
    ansatz = QuantumCircuit(2)
    for _ in range(num_blocks):
        block = QuantumCircuit(2)
        theta = ParameterVector("theta", 1)
        block.ry(0, theta[0])
        block.cx(0, 1)
        ansatz.compose(block)

    assert ansatz.num_parameters == num_blocks

    # The i-th parameter must drive the i-th block (order contract)
    values = np.linspace(0.1, 0.5, num_blocks)
    reference = QiskitCircuit(2)
    for value in values:
        reference.ry(value, 0)
        reference.cx(0, 1)
    np.testing.assert_allclose(
        _statevector(ansatz, values), Statevector(reference).data, atol=1e-12
    )


def test_compose_appends_parameters_after_own():
    """Parameters of the composed circuit must come after existing ones."""
    first = QuantumCircuit(1)
    theta = ParameterVector("theta", 1)
    first.rx(0, theta[0])

    second = QuantumCircuit(1)
    phi = ParameterVector("theta", 1)
    second.ry(0, phi[0])

    first.compose(second)
    assert first.num_parameters == 2

    reference = QiskitCircuit(1)
    reference.rx(0.3, 0)
    reference.ry(0.7, 0)
    np.testing.assert_allclose(
        _statevector(first, [0.3, 0.7]), Statevector(reference).data, atol=1e-12
    )


def test_compose_merge_parameters():
    """new_parameters=False merges parameters positionally."""
    first = QuantumCircuit(1)
    theta = ParameterVector("theta", 1)
    first.rx(0, theta[0])

    second = QuantumCircuit(1)
    phi = ParameterVector("theta", 1)
    second.ry(0, phi[0])

    first.compose(second, new_parameters=False)
    assert first.num_parameters == 1

    reference = QiskitCircuit(1)
    reference.rx(0.4, 0)
    reference.ry(0.4, 0)
    np.testing.assert_allclose(_statevector(first, [0.4]), Statevector(reference).data, atol=1e-12)


def test_compose_with_qubit_mapping():
    """Composition onto a qubit subset must respect the mapping."""
    big = QuantumCircuit(3)
    small = QuantumCircuit(2)
    small.x(0)
    small.cx(0, 1)
    big.compose(small, qubits=[2, 1])

    reference = QiskitCircuit(3)
    reference.x(2)
    reference.cx(2, 1)
    np.testing.assert_allclose(_statevector(big), Statevector(reference).data, atol=1e-12)


def test_compose_rejects_mismatched_qubits():
    """Composition without a mapping requires equal qubit counts."""
    with pytest.raises(ValueError):
        QuantumCircuit(3).compose(QuantumCircuit(2))


def test_fixate_parameters():
    """Fixating parameters removes them and binds the values."""
    circuit = QuantumCircuit(1)
    theta = ParameterVector("theta", 2)
    circuit.rx(0, theta[0])
    circuit.ry(0, theta[1])

    circuit.fixate_parameters(np.array([0.2, 0.9]))
    assert circuit.num_parameters == 0
    assert not circuit.is_parameterized

    reference = QiskitCircuit(1)
    reference.rx(0.2, 0)
    reference.ry(0.9, 0)
    np.testing.assert_allclose(_statevector(circuit), Statevector(reference).data, atol=1e-12)


def test_pauli_evolution_matches_qiskit_evolution_gate():
    """Pauli evolution must match qiskit's PauliEvolutionGate convention."""
    operator = SparsePauliOp(["XY"], [0.5])
    circuit = QuantumCircuit(2)
    circuit.h([0, 1])
    circuit.pauli_evolution(operator, 0.37)

    qiskit_reference = _plus_plus()
    qiskit_reference.append(PauliEvolutionGate(operator, time=0.37), [0, 1])
    np.testing.assert_allclose(
        _statevector(circuit), Statevector(qiskit_reference).data, atol=1e-12
    )


def test_controlled_pauli_evolution_control_state_zero():
    """control_state="0" must trigger the evolution when the control is |0>."""
    operator = SparsePauliOp(["Z"], [1.0])
    time = 0.81

    for control_state, control_value in (("0", 0), ("1", 1)):
        circuit = QuantumCircuit(2)
        if control_value:
            circuit.x(0)
        circuit.h(1)
        circuit.controlled_pauli_evolution(
            operator,
            time,
            working_qubits=[1],
            control_qubits=0,
            control_state=control_state,
        )

        reference = QiskitCircuit(2)
        if control_value:
            reference.x(0)
        reference.h(1)
        # Evolution must fire because the control matches the control state
        reference.append(PauliEvolutionGate(operator, time=time), [1])
        state = _statevector(circuit)
        reference_state = Statevector(reference).data
        overlap = np.abs(np.vdot(state, reference_state))
        assert overlap == pytest.approx(
            1.0, abs=1e-9
        ), f"control_state={control_state} did not apply the evolution"


def test_controlled_pauli_evolution_control_state_mismatch_is_inert():
    """A control that does not match control_state must leave the state alone."""
    operator = SparsePauliOp(["Z"], [1.0])
    time = 0.81

    for control_state, control_value in (("0", 1), ("1", 0)):
        circuit = QuantumCircuit(2)
        if control_value:
            circuit.x(0)
        circuit.h(1)
        circuit.controlled_pauli_evolution(
            operator,
            time,
            working_qubits=[1],
            control_qubits=0,
            control_state=control_state,
        )

        reference = QiskitCircuit(2)
        if control_value:
            reference.x(0)
        reference.h(1)
        # No evolution: the control does not match the requested control state
        state = _statevector(circuit)
        reference_state = Statevector(reference).data
        overlap = np.abs(np.vdot(state, reference_state))
        assert overlap == pytest.approx(1.0, abs=1e-9), (
            f"control_state={control_state} applied the evolution even though "
            f"the control was |{control_value}>"
        )


def test_draw_and_qiskit_circuit_access():
    """draw() must render and qiskit_circuit must expose the native circuit."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    assert "q" in str(circuit.draw())
    assert isinstance(circuit.qiskit_circuit, QiskitCircuit)


def test_from_qiskit_wraps_native_circuit():
    """from_qiskit must wrap a native circuit without copying it."""
    native = QiskitCircuit(2)
    native.h(0)

    circuit = QuantumCircuit.from_qiskit(native)
    assert circuit.num_qubits == 2
    assert circuit.qiskit_circuit is native


def test_qiskit_circuit_setter_validates_qubit_count():
    """The qiskit_circuit setter must reject a different qubit count."""
    circuit = QuantumCircuit(2)
    replacement = QiskitCircuit(2)
    replacement.x(0)
    circuit.qiskit_circuit = replacement
    assert circuit.qiskit_circuit is replacement

    with pytest.raises(ValueError, match="must have 2 qubits"):
        circuit.qiskit_circuit = QiskitCircuit(3)


def test_structural_equality_and_hash():
    """Structurally identical circuits must be equal; mutation changes both."""
    first = QuantumCircuit(1)
    first.h(0)
    second = QuantumCircuit(1)
    second.h(0)

    assert first == second
    assert hash(first) == hash(second)

    second.x(0)
    assert first != second
