"""Cross-backend lock on the Pauli-label qubit convention.

Qubit ``q`` is character ``q`` of a Pauli label, so ``"ZI"`` measures Z on
qubit 0.  Before this was pinned the backends disagreed: Qiskit read labels
right-to-left while PauliPropagation, PennyLane and Qulacs read them
left-to-right, so ``QuantumOperator(["ZI"], [1.0])`` silently measured a
different qubit depending on which backend ran it.

These tests check every installed backend against values derived from the
circuit, not against each other, so agreeing on a wrong answer still fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from qc_executor import Executor, QuantumCircuit, QuantumOperator
from tests.conftest import INSTALLED_BACKENDS

#: Circuit under test: X on qubit 0 only, so qubit 0 is |1> and qubit 1 is |0>.
#: Hence <Z on qubit 0> = -1 and <Z on qubit 1> = +1.
_EXPECTED = {
    "ZI": -1.0,
    "IZ": +1.0,
    "ZZ": -1.0,
    "II": +1.0,
    "XI": 0.0,
    "IX": 0.0,
}


def _scalar(value) -> float:
    """Reduce a backend result to a single real number."""
    return float(np.real(np.asarray(value).reshape(-1)[0]))


def _expectation(backend: str, label: str) -> float:
    """Run ``<label>`` on the reference circuit with ``backend``."""
    executor = Executor.create(backend, seed=0)
    circuit = QuantumCircuit(2)
    circuit.x(0)
    native_circuit = executor.transpile_circuit(circuit)
    native_operator = executor.transpile_operator(QuantumOperator([label], [1.0]))
    return _scalar(executor.expectation_value(native_circuit, native_operator))


class TestPauliLabelConvention:
    @pytest.mark.parametrize("backend", INSTALLED_BACKENDS)
    @pytest.mark.parametrize("label", sorted(_EXPECTED), ids=sorted(_EXPECTED))
    def test_qubit_zero_is_the_leftmost_character(self, backend, label):
        assert _expectation(backend, label) == pytest.approx(_EXPECTED[label], abs=1e-8)

    @pytest.mark.parametrize("label", sorted(_EXPECTED), ids=sorted(_EXPECTED))
    def test_all_backends_agree(self, label):
        values = {backend: _expectation(backend, label) for backend in INSTALLED_BACKENDS}

        assert len(set(np.round(list(values.values()), 8))) == 1, values


class TestStatevectorAndSampleConvention:
    """The label convention has to match how states are indexed and sampled."""

    @pytest.mark.parametrize("backend", INSTALLED_BACKENDS)
    def test_statevector_puts_qubit_zero_in_the_most_significant_bit(self, backend):
        executor = Executor.create(backend, seed=0)
        circuit = QuantumCircuit(2)
        circuit.x(0)

        state = np.asarray(executor.statevector(executor.transpile_circuit(circuit))).reshape(-1)

        # X on qubit 0 alone gives |10>, i.e. index 2.
        assert int(np.argmax(np.abs(state))) == 2

    @pytest.mark.parametrize("backend", INSTALLED_BACKENDS)
    def test_sampling_reports_qubit_zero_leftmost(self, backend):
        executor = Executor.create(backend, seed=0, shots=256)
        circuit = QuantumCircuit(2)
        circuit.x(0)

        counts = executor.sample(executor.transpile_circuit(circuit))
        if isinstance(counts, list):
            counts = counts[0]

        assert max(counts, key=counts.get) == "10"
