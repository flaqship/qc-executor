import builtins
import importlib
import sys
from types import ModuleType

import executor.utils.qiskit_compat as qc


class DummyParam:
    def __init__(self, *, symbol_expr=None, sympy_result=None, parameters=(), float_value=0.0):
        self._symbol_expr = symbol_expr
        self._sympy_result = sympy_result
        self.parameters = set(parameters)
        self._float_value = float_value

    def sympify(self):
        return self._sympy_result

    def __float__(self):
        return float(self._float_value)


def test_param_to_sympy_uses_private_symbol_expr_for_old_versions(monkeypatch):
    monkeypatch.setattr(qc, "QISKIT_SMALLER_2_0", True)
    param = DummyParam(symbol_expr="x + 1")

    result = qc._param_to_sympy(param)

    assert str(result) == "x + 1"


def test_param_to_sympy_uses_public_sympify_for_new_versions(monkeypatch):
    monkeypatch.setattr(qc, "QISKIT_SMALLER_2_0", False)
    param = DummyParam(sympy_result="already_sympy")

    result = qc._param_to_sympy(param)

    assert result == "already_sympy"


def test_param_is_constant_true_and_false():
    assert qc._param_is_constant(DummyParam(parameters=()))
    assert not qc._param_is_constant(DummyParam(parameters=("x",)))


def test_param_to_float():
    assert qc._param_to_float(DummyParam(float_value=2.5)) == 2.5


def test_param_free_symbols():
    param = DummyParam(parameters=("a", "b"))
    assert qc._param_free_symbols(param) == {"a", "b"}


def test_runtime_version_flags_when_ibm_runtime_installed(monkeypatch):
    fake_runtime = ModuleType("qiskit_ibm_runtime")
    fake_runtime.__version__ = "0.22.0"
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", fake_runtime)

    reloaded = importlib.reload(qc)

    assert reloaded.QISKIT_RUNTIME_AVAILABLE is True
    assert reloaded.QISKIT_RUNTIME_SMALLER_0_21 is False
    assert reloaded.QISKIT_RUNTIME_SMALLER_0_23 is True
    assert reloaded.QISKIT_RUNTIME_SMALLER_0_28 is True

    importlib.reload(qc)


def test_runtime_version_flags_when_ibm_runtime_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "qiskit_ibm_runtime":
            raise ImportError("forced for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "qiskit_ibm_runtime", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    reloaded = importlib.reload(qc)

    assert reloaded.QISKIT_RUNTIME_AVAILABLE is False
    assert reloaded.QISKIT_RUNTIME_SMALLER_0_21 is None
    assert reloaded.QISKIT_RUNTIME_SMALLER_0_23 is None
    assert reloaded.QISKIT_RUNTIME_SMALLER_0_28 is None

    importlib.reload(qc)
