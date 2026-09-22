"""Tests carried over from ``integration-support`` for behaviour sQUlearn relies on.

Merged in with ``integration-support``; the tests both branches share live in
``test_executor_base.py``, which is kept exactly as on ``integration-support-ir``.
"""

import logging

import numpy as np
import pytest

from qc_executor import factory as factory_module
from qc_executor.base.executor_base import ExecutorBase


class DummyExecutor(ExecutorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = {
            "expectation": 0,
            "derivatives": 0,
            "sample": 0,
            "statevector": 0,
            "transpile_circuit": 0,
            "transpile_operator": 0,
        }

    @property
    def remote(self) -> bool:
        return False

    @ExecutorBase.shots.setter
    def shots(self, value: int | None) -> None:
        self._shots = value

    def _expectation_value(self, circuit, observable, **parameters):
        self.calls["expectation"] += 1
        self.last_expectation_args = (circuit, observable, parameters)
        return float(self.calls["expectation"])

    def _expectation_value_derivatives(self, circuit, observable, *derivative, **parameters):
        self.calls["derivatives"] += 1
        return ("grad", circuit, observable, derivative, parameters)

    def _sample(self, circuit, **parameters):
        self.calls["sample"] += 1
        return {"circuit": circuit, "params": parameters, "shots": self._shots}

    def _statevector(self, circuit, **parameters):
        self.calls["statevector"] += 1
        return np.array([1.0, 0.0])

    def _transpile_circuit(self, circuit):
        self.calls["transpile_circuit"] += 1
        return f"tc:{circuit}"

    def _transpile_operator(self, operator):
        self.calls["transpile_operator"] += 1
        return f"to:{operator}"

    @classmethod
    def get_accepted_backend_types(cls) -> list[type]:
        return []


class ProbabilitiesExecutor(DummyExecutor):
    """Dummy executor with a controllable statevector and counts distribution."""

    def _statevector(self, circuit, **parameters):
        self.last_statevector_parameters = parameters
        # |amplitude|^2 gives probabilities 0.99, 0.01, 1e-6 and 0.0.
        return np.array([np.sqrt(0.99), 0.1, 1e-3, 0.0])

    def _sample(self, circuit, **parameters):
        # Same distribution as the statevector above, as counts out of 1000001.
        return {"00": 990_000, "01": 10_000, "10": 1, "11": 0}


class TestExecutorBaseProbabilities:
    def test_default_drops_only_exact_zeros(self):
        ex = ProbabilitiesExecutor()

        probabilities = ex.probabilities("c0")

        assert set(probabilities) == {0, 1, 2}
        assert probabilities[0] == pytest.approx(0.99)
        assert probabilities[2] == pytest.approx(1e-6)

    def test_cutoff_prunes_small_entries(self):
        ex = ProbabilitiesExecutor()

        probabilities = ex.probabilities("c0", cutoff=1e-3)

        assert set(probabilities) == {0, 1}

    def test_cutoff_applies_to_the_sampled_branch(self):
        ex = ProbabilitiesExecutor(shots=1_000_001)

        default = ex.probabilities("c0")
        pruned = ex.probabilities("c0", cutoff=1e-3)

        # The zero-count entry is pruned just like a vanishing amplitude is,
        # so switching shots on or off reports the same set of entries.
        assert set(default) == {0, 1, 2}
        assert set(pruned) == {0, 1}

    def test_cutoff_does_not_shadow_a_circuit_parameter(self):
        ex = ProbabilitiesExecutor()

        ex.probabilities("c0", cutoff=1e-3, x=[0.1])

        assert ex.last_statevector_parameters == {"x": [0.1]}


class BatchProbabilitiesExecutor(DummyExecutor):
    """Dummy executor whose statevector/sample can return a genuine batch.

    The sample() convention mirrors the PennyLane backend: a list of dicts,
    one per parameter set, even when there is only one set - exercising the
    single-vs-batch unwrapping in probabilities() independently of
    ProbabilitiesExecutor's bare-dict (Qulacs-style) convention above.
    """

    def _statevector(self, circuit, **parameters):
        if parameters.get("batch"):
            return np.array([[1.0, 0.0], [0.0, 1.0]])
        return np.array([1.0, 0.0])

    def _sample(self, circuit, **parameters):
        if parameters.get("batch"):
            return [{"0": 100}, {"1": 100}]
        return [{"0": 100}]


class TestExecutorBaseProbabilitiesBatching:
    def test_exact_single_set_returns_a_bare_mapping(self):
        ex = BatchProbabilitiesExecutor()

        assert ex.probabilities("c0") == {0: 1.0}

    def test_exact_batch_returns_a_list_of_mappings(self):
        ex = BatchProbabilitiesExecutor()

        result = ex.probabilities("c0", batch=True)

        assert result == [{0: 1.0}, {1: 1.0}]

    def test_sampled_single_set_from_list_convention_unwraps_to_a_bare_mapping(self):
        """A backend that always wraps sample() results in a list (even for
        one parameter set) still yields a bare mapping here - the length-1
        list is unwrapped, matching what a backend returning a bare dict
        directly (ProbabilitiesExecutor above) already produces."""
        ex = BatchProbabilitiesExecutor(shots=100)

        assert ex.probabilities("c0") == {0: 1.0}

    def test_sampled_batch_returns_a_list_of_mappings(self):
        ex = BatchProbabilitiesExecutor(shots=100)

        result = ex.probabilities("c0", batch=True)

        assert result == [{0: 1.0}, {1: 1.0}]
