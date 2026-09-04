"""Cross-backend contract tests for the public executor API.

These tests pin down the conventions every backend must follow: big-endian
bit ordering (qubit 0 is the most significant bit and the leftmost bitstring
character), big-endian public Pauli labels (leftmost character acts on qubit
0) while native qiskit ``SparsePauliOp`` inputs keep their little-endian
qiskit meaning, the derivative return format and its numeric parameter
ordering, and index-ordered gradients.
"""

import numpy as np
import pytest
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

from qc_executor import Executor, QuantumCircuit, QuantumOperator

BACKENDS = ["qulacs", "qiskit", "pennylane"]


@pytest.fixture(scope="module", params=BACKENDS)
def executor(request):
    """Provide an executor for every supported backend."""
    return Executor.create(request.param)


def _ansatz(num_qubits: int = 2, num_parameters: int = 2) -> QuantumCircuit:
    """Build a small parameterized test ansatz."""
    circuit = QuantumCircuit(num_qubits)
    theta = ParameterVector("theta", num_parameters)
    circuit.h(0)
    for index in range(num_parameters):
        circuit.ry(index % num_qubits, theta[index])
    circuit.cx(0, 1)
    return circuit


def _reference_expectation(circuit: QuantumCircuit, operator: SparsePauliOp, values) -> complex:
    """Compute a reference expectation value via qiskit statevectors."""
    bound = circuit.qiskit_circuit.assign_parameters(
        dict(zip(circuit.qiskit_circuit.parameters, values))
    )
    return Statevector(bound).expectation_value(operator)


def _to_big_endian(statevector: np.ndarray) -> np.ndarray:
    """Permute qiskit little-endian amplitudes into the public big-endian order."""
    n = int(np.log2(len(statevector)))
    perm = [int(format(i, f"0{n}b")[::-1], 2) for i in range(len(statevector))]
    return statevector[perm]


def test_native_pauli_label_convention(executor):
    """Native SparsePauliOp labels keep their qiskit meaning on every backend.

    For the state X(qubit 0)|00>, the observable "ZI" (Z on qubit 1 in
    qiskit convention) must give +1 and "IZ" (Z on qubit 0) must give -1.
    """
    circuit = QuantumCircuit(2)
    circuit.x(0)

    z_on_qubit_1 = QuantumOperator(_native_operator=SparsePauliOp(["ZI"], [1.0]))
    z_on_qubit_0 = QuantumOperator(_native_operator=SparsePauliOp(["IZ"], [1.0]))

    assert executor.expectation_value(circuit, z_on_qubit_1) == pytest.approx(1.0, abs=1e-10)
    assert executor.expectation_value(circuit, z_on_qubit_0) == pytest.approx(-1.0, abs=1e-10)


def test_public_pauli_label_convention(executor):
    """Public labels are big-endian: the leftmost character acts on qubit 0.

    For the state X(qubit 0)|00>, the observable "ZI" (Z on qubit 0) must
    give -1 and "IZ" (Z on qubit 1) must give +1.
    """
    circuit = QuantumCircuit(2)
    circuit.x(0)

    assert executor.expectation_value(circuit, QuantumOperator(["ZI"], [1.0])) == pytest.approx(
        -1.0, abs=1e-10
    )
    assert executor.expectation_value(circuit, QuantumOperator(["IZ"], [1.0])) == pytest.approx(
        1.0, abs=1e-10
    )


def test_observable_coefficients_are_applied(executor):
    """Numeric coefficients must scale the expectation value on every backend."""
    circuit = QuantumCircuit(2)
    circuit.x(0)

    result = executor.expectation_value(circuit, QuantumOperator(["ZI"], [0.5]))
    assert result == pytest.approx(-0.5, abs=1e-10)


def test_expectation_value_matches_reference(executor):
    """Expectation values must match the qiskit statevector reference."""
    circuit = _ansatz()
    operator = SparsePauliOp(["ZZ", "XI"], [1.0, 0.5])
    values = [0.3, 0.8]

    result = executor.expectation_value(
        circuit, QuantumOperator(_native_operator=operator), theta=values
    )
    reference = _reference_expectation(circuit, operator, values)
    assert np.real(result) == pytest.approx(np.real(reference), abs=1e-8)


def test_derivatives_return_format(executor):
    """One requested name returns an array, several return a dict per name."""
    circuit = _ansatz()
    operator = QuantumOperator(_native_operator=SparsePauliOp(["ZZ"], [1.0]))
    values = [0.3, 0.8]

    single = executor.expectation_value_derivatives(circuit, operator, "theta", theta=values)
    assert not isinstance(single, dict)
    assert np.asarray(single).reshape(-1).shape == (2,)

    p_obs = ParameterVector("p_obs", 1)
    parametrized = QuantumOperator(["ZZ"], [p_obs[0]])
    multiple = executor.expectation_value_derivatives(
        circuit, parametrized, "theta", "p_obs", theta=values, p_obs=[0.5]
    )
    assert isinstance(multiple, dict)
    assert set(multiple) == {"theta", "p_obs"}


def test_derivatives_use_numeric_parameter_order(executor):
    """Derivatives must be in numeric parameter order beyond 10 parameters.

    Guards against lexicographic internal sorting (theta[10] < theta[2]).
    """
    num_parameters = 13
    circuit = QuantumCircuit(num_parameters)
    theta = ParameterVector("theta", num_parameters)
    for index in range(num_parameters):
        circuit.ry(index, theta[index])
    operator = QuantumOperator(["Z" * num_parameters], [1.0])
    values = np.linspace(0.1, 1.3, num_parameters)

    derivatives = np.asarray(
        executor.expectation_value_derivatives(circuit, operator, "theta", theta=values),
        dtype=float,
    ).reshape(-1)

    # d/d theta_k of prod_i cos(theta_i) = -sin(theta_k) prod_{i != k} cos(theta_i)
    product = np.prod(np.cos(values))
    expected = np.array([-np.sin(v) * product / np.cos(v) for v in values])
    np.testing.assert_allclose(derivatives, expected, atol=1e-7)


def test_derivatives_match_finite_differences(executor):
    """Full gradients via the vector name must match finite differences."""
    circuit = _ansatz()
    operator = SparsePauliOp(["ZZ", "XI"], [1.0, 0.5])
    values = [0.3, 0.8]

    gradient = np.asarray(
        executor.expectation_value_derivatives(
            circuit, QuantumOperator(_native_operator=operator), "theta", theta=values
        ),
        dtype=float,
    ).reshape(-1)
    assert gradient.shape == (2,)

    step = 1e-6
    for index in range(len(values)):
        shift = np.zeros(len(values))
        shift[index] = step
        upper = np.real(_reference_expectation(circuit, operator, np.array(values) + shift))
        lower = np.real(_reference_expectation(circuit, operator, np.array(values) - shift))
        finite_difference = (upper - lower) / (2 * step)
        assert gradient[index] == pytest.approx(finite_difference, abs=1e-5)


def test_derivatives_ordering_with_many_parameters(executor):
    """Derivatives must be in numeric parameter order beyond 10 parameters."""
    num_parameters = 13
    circuit = QuantumCircuit(num_parameters)
    theta = ParameterVector("theta", num_parameters)
    for index in range(num_parameters):
        circuit.ry(index, theta[index])
    operator = QuantumOperator(["Z" * num_parameters], [1.0])
    values = np.linspace(0.1, 1.3, num_parameters)

    gradient = np.asarray(
        executor.expectation_value_derivatives(circuit, operator, "theta", theta=values),
        dtype=float,
    ).reshape(-1)

    product = np.prod(np.cos(values))
    expected = np.array([-np.sin(v) * product / np.cos(v) for v in values])
    np.testing.assert_allclose(gradient, expected, atol=1e-7)


def test_derivatives_observable_list(executor):
    """A list of observables must return one derivative row per observable."""
    circuit = _ansatz()
    operator = QuantumOperator(["ZZ"], [1.0])
    scaled = QuantumOperator(["ZZ"], [4.0])
    values = [0.3, 0.8]

    gradient = np.asarray(
        executor.expectation_value_derivatives(circuit, [operator, scaled], "theta", theta=values),
        dtype=float,
    ).reshape(2, 2)
    np.testing.assert_allclose(gradient[1], 4 * gradient[0], atol=1e-8)


def test_derivatives_circuit_observable_cross_product(executor):
    """Lists of circuits and observables must be evaluated combinatorially."""
    first = _ansatz()
    second = _ansatz(num_parameters=2)
    second.rz(0, 0.4)
    operator = QuantumOperator(["ZZ"], [1.0])
    scaled = QuantumOperator(["ZZ"], [4.0])
    values = [0.3, 0.8]

    combined = np.asarray(
        executor.expectation_value_derivatives(
            [first, second], [operator, scaled], "theta", theta=values
        ),
        dtype=float,
    )
    assert combined.shape[:2] == (2, 2)

    for i, circuit in enumerate([first, second]):
        for j, single_operator in enumerate([operator, scaled]):
            reference = np.asarray(
                executor.expectation_value_derivatives(
                    circuit, single_operator, "theta", theta=values
                ),
                dtype=float,
            )
            np.testing.assert_allclose(combined[i, j], reference.reshape(combined[i, j].shape))

    # A circuit list with a single observable adds only the circuit axis.
    circuit_axis_only = np.asarray(
        executor.expectation_value_derivatives([first, second], operator, "theta", theta=values),
        dtype=float,
    )
    np.testing.assert_allclose(circuit_axis_only, combined[:, 0])


def test_probabilities_bit_ordering(executor):
    """Probability keys must use big-endian basis-state indices."""
    circuit = QuantumCircuit(2)
    circuit.x(0)  # q0 = 1 is the most significant bit: index 0b10 = 2

    probabilities = executor.probabilities(circuit)
    assert set(probabilities.keys()) == {2}
    assert probabilities[2] == pytest.approx(1.0, abs=1e-10)


def test_statevector_bit_ordering(executor):
    """Statevectors must be returned in big-endian ordering (qubit 0 = MSB)."""
    circuit = QuantumCircuit(2)
    circuit.x(0)

    state = executor.statevector(circuit)
    assert np.abs(state[2]) == pytest.approx(1.0, abs=1e-10)


def test_statevector_matches_reference(executor):
    """A parameterized statevector must match the qiskit reference."""
    circuit = _ansatz()
    values = [0.5, 1.2]

    state = executor.statevector(circuit, theta=values)
    bound = circuit.qiskit_circuit.assign_parameters(
        dict(zip(circuit.qiskit_circuit.parameters, values))
    )
    reference = _to_big_endian(Statevector(bound).data)
    # States may differ by a global phase
    overlap = np.abs(np.vdot(np.asarray(state).reshape(-1), reference))
    assert overlap == pytest.approx(1.0, abs=1e-8)


def test_expectation_value_batch_of_parameter_sets(executor):
    """A batch of parameter sets - the same leading-axis convention already
    used for circuit/observable lists - gives one result per set, matching a
    per-set reference, on every backend (Qiskit, PennyLane, Qulacs alike)."""
    circuit = _ansatz()
    operator = SparsePauliOp(["ZZ", "XI"], [1.0, 0.5])
    value_sets = [[0.3, 0.8], [0.1, 0.2], [1.0, -0.5]]

    batched = executor.expectation_value(
        circuit, QuantumOperator(_native_operator=operator), theta=value_sets
    )
    reference = [np.real(_reference_expectation(circuit, operator, v)) for v in value_sets]

    assert np.shape(batched) == (3,)
    np.testing.assert_allclose(np.real(batched), reference, atol=1e-8)


def test_expectation_value_single_set_unaffected_by_batching_support(executor):
    """A single parameter set still returns a bare scalar on every backend -
    the common case is unaffected by batch support."""
    circuit = _ansatz()
    operator = QuantumOperator(_native_operator=SparsePauliOp(["ZZ"], [1.0]))

    result = executor.expectation_value(circuit, operator, theta=[0.3, 0.8])

    assert np.shape(result) == ()


def test_statevector_batch_of_parameter_sets(executor):
    """A batch of parameter sets gives one statevector per set, matching a
    per-set qiskit reference, on every backend."""
    circuit = _ansatz()
    value_sets = [[0.3, 0.8], [0.1, 0.2]]

    batched = executor.statevector(circuit, theta=value_sets)
    assert np.shape(batched)[0] == 2

    for i, values in enumerate(value_sets):
        bound = circuit.qiskit_circuit.assign_parameters(
            dict(zip(circuit.qiskit_circuit.parameters, values))
        )
        reference = _to_big_endian(Statevector(bound).data)
        overlap = np.abs(np.vdot(np.asarray(batched[i]).reshape(-1), reference))
        assert overlap == pytest.approx(1.0, abs=1e-8)


def test_create_passes_through_prebuilt_executor():
    """An already constructed executor must be returned unchanged."""
    prebuilt = Executor.create("qulacs")
    assert Executor.create(prebuilt) is prebuilt

    with pytest.raises(ValueError, match="already"):
        Executor.create(prebuilt, shots=100)
