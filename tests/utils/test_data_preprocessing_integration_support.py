"""Tests carried over from ``integration-support`` for behaviour sQUlearn relies on.

Merged in with ``integration-support``; the tests both branches share live in
``test_data_preprocessing.py``, which is kept exactly as on ``integration-support-ir``.
"""

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

import qc_executor.utils.data_preprocessing as dp


class TestResolveParameterBatchSize:
    def test_all_singleton_resolves_to_one(self):
        assert dp.resolve_parameter_batch_size([1, 1, 1]) == 1

    def test_no_parameters_resolves_to_one(self):
        assert dp.resolve_parameter_batch_size([]) == 1

    def test_one_batched_parameter_resolves_to_its_size(self):
        assert dp.resolve_parameter_batch_size([1, 5, 1]) == 5

    def test_agreeing_batched_parameters_resolve_to_the_shared_size(self):
        assert dp.resolve_parameter_batch_size([3, 1, 3, 3]) == 3

    def test_disagreeing_batch_sizes_raise(self):
        with pytest.raises(ValueError, match="must share the same batch size"):
            dp.resolve_parameter_batch_size([3, 5])

    def test_accepts_a_generator(self):
        assert dp.resolve_parameter_batch_size(n for n in [1, 4, 1]) == 4
