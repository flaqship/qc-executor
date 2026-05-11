import logging

import numpy as np
import pytest

from executor.base.executor_base import ExecutorBase


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
        return ("ev", circuit, observable, parameters)

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


class TestExecutorBaseInternals:
    def test_normalize_parameter_values_indexed_to_vector_raises(self):
        with pytest.raises(ValueError, match="Incomplete indexed parameters for 'x'"):
            ExecutorBase._normalize_parameter_values(
                **{"x[0]": 0.1, "x[2]": 0.3, "y": [1.0], "z": 7}
            )

    def test_normalize_parameter_values_indexed_to_vector(self):
        normalized = ExecutorBase._normalize_parameter_values(
            **{"x[0]": 0.1, "x[1]": 0.2, "x[2]": 0.3, "y": [1.0], "z": 7}
        )

        assert list(normalized.keys()) == ["y", "z", "x"]
        assert normalized["x"] == [0.1, 0.2, 0.3]
        assert normalized["y"] == [1.0]
        assert normalized["z"] == 7

    def test_normalize_parameter_values_mixed_forms_raise(self):
        with pytest.raises(ValueError, match="Cannot mix vector and indexed parameter forms"):
            ExecutorBase._normalize_parameter_values(**{"x": [9.9], "x[0]": 1.0, "x[1]": 2.0})

    def test_make_result_key_independent_of_kwarg_order(self):
        key1 = ExecutorBase._make_result_key("m", 1, 2, a=3, b=4)
        key2 = ExecutorBase._make_result_key("m", 1, 2, b=4, a=3)
        assert key1 == key2

    def test_make_result_key_distinguishes_array_memory_layout(self):
        arr_c = np.array([[1, 2], [3, 4]], order="C")
        arr_f = np.array([[1, 2], [3, 4]], order="F")

        key_c = ExecutorBase._make_result_key("m", arr_c)
        key_f = ExecutorBase._make_result_key("m", arr_f)

        assert key_c != key_f

    def test_make_result_key_uses_identity_for_unhashable_objects(self):
        obj_a = {"x": 1}
        obj_b = {"x": 1}

        key_a = ExecutorBase._make_result_key("m", obj_a)
        key_b = ExecutorBase._make_result_key("m", obj_b)

        assert key_a != key_b


class TestExecutorBaseCachingAndDelegation:
    def test_expectation_value_uses_cache(self):
        ex = DummyExecutor(caching=True)

        first = ex.expectation_value("c0", "o0", **{"x[0]": 0.1, "x[1]": 0.2})
        second = ex.expectation_value("c0", "o0", x=[0.1, 0.2])

        assert first == second
        assert ex.calls["expectation"] == 1

    def test_expectation_value_derivatives_uses_cache_and_derivative_key(self):
        ex = DummyExecutor(caching=True)

        r1 = ex.expectation_value_derivatives("c0", "o0", "theta", **{"x[0]": 1.0})
        r2 = ex.expectation_value_derivatives("c0", "o0", "theta", x=[1.0])
        r3 = ex.expectation_value_derivatives("c0", "o0", "phi", x=[1.0])

        assert r1 == r2
        assert r1 != r3
        assert ex.calls["derivatives"] == 2

    def test_sample_cache_includes_shots(self):
        ex = DummyExecutor(caching=True, shots=100)

        r1 = ex.sample("c0")
        r2 = ex.sample("c0")
        ex.shots = 200
        r3 = ex.sample("c0")

        assert r1 == r2
        assert r3["shots"] == 200
        assert ex.calls["sample"] == 2

    def test_bounded_cache_evicts_oldest_entry(self):
        ex = DummyExecutor(caching=True, max_cache_size=2)

        ex.expectation_value("c1", "o")
        ex.expectation_value("c2", "o")
        ex.expectation_value("c1", "o")

        assert ex.calls["expectation"] == 2

        ex.expectation_value("c3", "o")
        assert ex.calls["expectation"] == 3

        ex.expectation_value("c2", "o")
        assert ex.calls["expectation"] == 3

        ex.expectation_value("c1", "o")

        assert ex.calls["expectation"] == 4

        key_c1 = ExecutorBase._make_result_key("expectation_value", "c1", "o")
        key_c2 = ExecutorBase._make_result_key("expectation_value", "c2", "o")
        key_c3 = ExecutorBase._make_result_key("expectation_value", "c3", "o")

        assert ex._result_cache is not None
        assert len(ex._result_cache) == 2
        assert key_c1 in ex._result_cache
        assert key_c3 in ex._result_cache
        assert key_c2 not in ex._result_cache

    def test_transpile_circuit_list_caches_per_item(self):
        ex = DummyExecutor(caching=True)

        out = ex.transpile_circuit(["a", "b", "a"])
        out_again = ex.transpile_circuit(["a", "b", "a"])
        out_with_new = ex.transpile_circuit(["a", "c", "a"])

        assert out == ["tc:a", "tc:b", "tc:a"]
        assert out_again == ["tc:a", "tc:b", "tc:a"]
        assert out_with_new == ["tc:a", "tc:c", "tc:a"]
        assert ex.calls["transpile_circuit"] == 3

    def test_transpile_operator_list_caches_per_item(self):
        ex = DummyExecutor(caching=True)

        out = ex.transpile_operator(["x", "y", "x"])
        out_again = ex.transpile_operator(["x", "y", "x"])
        out_with_new = ex.transpile_operator(["x", "z", "x"])

        assert out == ["to:x", "to:y", "to:x"]
        assert out_again == ["to:x", "to:y", "to:x"]
        assert out_with_new == ["to:x", "to:z", "to:x"]
        assert ex.calls["transpile_operator"] == 3

    def test_transpile_circuit_single_value_uses_cache(self):
        ex = DummyExecutor(caching=True)

        out = ex.transpile_circuit("a")
        out_again = ex.transpile_circuit("a")
        out_new = ex.transpile_circuit("b")

        assert out == "tc:a"
        assert out_again == "tc:a"
        assert out_new == "tc:b"
        assert ex.calls["transpile_circuit"] == 2

    def test_transpile_operator_single_value_uses_cache(self):
        ex = DummyExecutor(caching=True)

        out = ex.transpile_operator("x")
        out_again = ex.transpile_operator("x")
        out_new = ex.transpile_operator("y")

        assert out == "to:x"
        assert out_again == "to:x"
        assert out_new == "to:y"
        assert ex.calls["transpile_operator"] == 2

    def test_transpile_operator_single_value_without_cache_calls_backend_each_time(self):
        ex = DummyExecutor(caching=False)

        first = ex.transpile_operator("x")
        second = ex.transpile_operator("x")

        assert first == "to:x"
        assert second == "to:x"
        assert ex.calls["transpile_operator"] == 2

    def test_switch_backend_forwards_config_and_overrides(self, monkeypatch):
        import executor.factory as factory_module

        captured = {}

        def fake_create(cls, backend, **kwargs):
            captured["backend"] = backend
            captured["kwargs"] = kwargs
            return "new-executor"

        monkeypatch.setattr(factory_module.Executor, "create", classmethod(fake_create))

        ex = DummyExecutor(shots=123, seed=77, caching=True, max_cache_size=9, log_level="INFO")
        out = ex.switch_backend("qiskit", shots=999)

        assert out == "new-executor"
        assert captured["backend"] == "qiskit"
        assert captured["kwargs"]["shots"] == 999
        assert captured["kwargs"]["seed"] == 77
        assert captured["kwargs"]["caching"] is True
        assert captured["kwargs"]["max_cache_size"] == 9


class TestExecutorBaseLoggingAndValidation:
    def test_invalid_log_level_raises(self):
        with pytest.raises(ValueError, match="Invalid log_level"):
            DummyExecutor(log_level="TRACE")

    def test_file_handler_not_duplicated_for_same_logger_and_file(self, tmp_path):
        log_file = str(tmp_path / "executor_base.log")
        logger_name = f"{DummyExecutor.__module__}.{DummyExecutor.__qualname__}"
        logger = logging.getLogger(logger_name)

        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        first = DummyExecutor(log_file=log_file, log_level="INFO")
        second = DummyExecutor(log_file=log_file, log_level="DEBUG")

        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert file_handlers[0].level == logging.DEBUG

        for inst in (first, second):
            for handler in inst._logger.handlers[:]:
                handler.close()
                inst._logger.removeHandler(handler)

    def test_get_accepted_backend_aliases_default_empty(self):
        assert DummyExecutor.get_accepted_backend_aliases() == []

    def test_make_cache_size_validation(self):
        with pytest.raises(ValueError, match="max_size must be None or a positive integer"):
            DummyExecutor(caching=True, max_cache_size=0)
