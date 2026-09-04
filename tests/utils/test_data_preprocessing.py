import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

import qc_executor.utils.data_preprocessing as dp


class TestAdjustFeaturesAndParameters:
    def test_adjust_features_scalar(self):
        x, multiple = dp.adjust_features(0.5, 1)
        np.testing.assert_array_equal(x, np.array([[0.5]], dtype=np.float64))
        assert not multiple

    def test_adjust_features_1d_multiple(self):
        x, multiple = dp.adjust_features(np.array([1.0, 2.0]), 1)
        np.testing.assert_array_equal(x, np.array([[1.0], [2.0]], dtype=np.float64))
        assert multiple

    def test_adjust_features_single_vector_for_multi_dim(self):
        x, multiple = dp.adjust_features(np.array([1.0, 2.0]), 2)
        np.testing.assert_array_equal(x, np.array([[1.0, 2.0]], dtype=np.float64))
        assert not multiple

    def test_adjust_features_2d(self):
        x, multiple = dp.adjust_features(np.array([[1.0, 2.0], [3.0, 4.0]]), 2)
        np.testing.assert_array_equal(x, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
        assert multiple

    def test_adjust_parameters_single_array_not_multiple(self):
        x, multiple = dp.adjust_parameters(np.array([1.0]), 1)
        np.testing.assert_array_equal(x, np.array([[1.0]], dtype=np.float64))
        assert not multiple

    def test_adjust_parameters_1d_multiple_when_length_gt_1(self):
        x, multiple = dp.adjust_parameters(np.array([1.0, 2.0]), 1)
        np.testing.assert_array_equal(x, np.array([[1.0], [2.0]], dtype=np.float64))
        assert multiple

    def test_adjust_input_raises_for_empty_input(self):
        with pytest.raises(ValueError, match="Wrong format"):
            dp.adjust_features(np.array([]), 1)

    def test_adjust_input_raises_for_wrong_1d_length(self):
        with pytest.raises(ValueError, match="Wrong format"):
            dp.adjust_features(np.array([1.0, 2.0]), 3)

    def test_adjust_input_raises_for_wrong_2d_length(self):
        with pytest.raises(ValueError, match="Wrong format"):
            dp.adjust_features(np.array([[1.0, 2.0], [3.0, 4.0]]), 3)

    def test_adjust_input_raises_for_high_dimensional_array(self):
        with pytest.raises(ValueError, match="Wrong format"):
            dp.adjust_features(np.zeros((2, 2, 2)), 2)


class TestConvertToFloat64:
    def test_convert_list_to_float64(self):
        result = dp.convert_to_float64([1, 2, 3])
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0], dtype=np.float64))

    def test_convert_real_if_close_complex(self):
        values = np.array([1.0 + 1e-15j, 2.0 - 1e-15j], dtype=np.complex128)
        result = dp.convert_to_float64(values)
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, np.array([1.0, 2.0], dtype=np.float64))

    def test_convert_raises_for_non_real_complex(self):
        values = np.array([1.0 + 1j], dtype=np.complex128)
        with pytest.raises(ValueError, match="Only real values"):
            dp.convert_to_float64(values)


class TestToTuple:
    def test_scalar_flatten_true(self):
        assert dp.to_tuple(5) == (5,)

    def test_1d_array_flatten_true(self):
        assert dp.to_tuple(np.array([1, 2, 3])) == (1, 2, 3)

    def test_nested_flatten_true(self):
        assert dp.to_tuple([[1, 2], np.array([3, 4])]) == (1, 2, 3, 4)

    def test_nested_flatten_false(self):
        result = dp.to_tuple([[1, 2], np.array([3, 4])], flatten=False)
        assert result == ((1, 2), (3, 4))

    def test_scalar_flatten_false(self):
        assert dp.to_tuple("x", flatten=False) == ("x",)


class TestEnsureComplexCoeffs:
    def test_casts_coeffs_for_affected_qiskit_versions(self, monkeypatch):
        base_operator = SparsePauliOp(["Z"], coeffs=np.array([1.0], dtype=np.complex128))

        class DummyOp:
            def __init__(self, paulis):
                self.paulis = paulis
                self.coeffs = np.array([1.0], dtype=np.float64)

        operator = DummyOp(base_operator.paulis)
        monkeypatch.setattr(dp, "qiskit_version", "2.1.5")

        result = dp.ensure_complex_coeffs(operator)

        assert isinstance(result, SparsePauliOp)
        assert result.coeffs.dtype == np.dtype("complex128")

    def test_returns_same_operator_outside_affected_range(self, monkeypatch):
        operator = SparsePauliOp(["Z"], coeffs=np.array([1.0], dtype=np.float64))
        monkeypatch.setattr(dp, "qiskit_version", "2.2.0")

        result = dp.ensure_complex_coeffs(operator)

        assert result is operator

    def test_returns_same_operator_when_already_complex(self, monkeypatch):
        operator = SparsePauliOp(["Z"], coeffs=np.array([1.0 + 0j], dtype=np.complex128))
        monkeypatch.setattr(dp, "qiskit_version", "2.1.1")

        result = dp.ensure_complex_coeffs(operator)

        assert result is operator


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
