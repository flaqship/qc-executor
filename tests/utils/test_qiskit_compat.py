import builtins
import importlib
import sys
from types import ModuleType

import pytest

import executor.utils.qiskit_compat as qc

# Test data for Qiskit version testing
QISKIT_VERSIONS = [
    ("1.0.0", {"SMALLER_1_2": True, "SMALLER_2_0": True}),
    ("1.1.0", {"SMALLER_1_2": True, "SMALLER_2_0": True}),
    ("1.2.0", {"SMALLER_1_2": False, "SMALLER_2_0": True}),
    ("1.3.0", {"SMALLER_1_2": False, "SMALLER_2_0": True}),
    ("2.0.0", {"SMALLER_1_2": False, "SMALLER_2_0": False}),
    ("2.1.0", {"SMALLER_1_2": False, "SMALLER_2_0": False}),
]

# Test data for IBM Runtime version testing
RUNTIME_VERSIONS_INSTALLED = [
    (
        "0.20.0",
        {"AVAILABLE": True, "SMALLER_0_21": True, "SMALLER_0_23": True, "SMALLER_0_28": True},
    ),
    (
        "0.21.0",
        {"AVAILABLE": True, "SMALLER_0_21": False, "SMALLER_0_23": True, "SMALLER_0_28": True},
    ),
    (
        "0.22.0",
        {"AVAILABLE": True, "SMALLER_0_21": False, "SMALLER_0_23": True, "SMALLER_0_28": True},
    ),
    (
        "0.23.0",
        {"AVAILABLE": True, "SMALLER_0_21": False, "SMALLER_0_23": False, "SMALLER_0_28": True},
    ),
    (
        "0.27.0",
        {"AVAILABLE": True, "SMALLER_0_21": False, "SMALLER_0_23": False, "SMALLER_0_28": True},
    ),
    (
        "0.28.0",
        {"AVAILABLE": True, "SMALLER_0_21": False, "SMALLER_0_23": False, "SMALLER_0_28": False},
    ),
    (
        "0.29.0",
        {"AVAILABLE": True, "SMALLER_0_21": False, "SMALLER_0_23": False, "SMALLER_0_28": False},
    ),
]


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


@pytest.mark.parametrize("version_string,expected_flags", RUNTIME_VERSIONS_INSTALLED)
def test_runtime_version_flags_when_ibm_runtime_installed(
    monkeypatch, version_string, expected_flags
):
    """Test that version flags are correctly set based on qiskit-ibm-runtime version."""
    fake_runtime = ModuleType("qiskit_ibm_runtime")
    fake_runtime.__version__ = version_string
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", fake_runtime)

    reloaded = importlib.reload(qc)

    assert reloaded.QISKIT_RUNTIME_AVAILABLE is expected_flags["AVAILABLE"]
    assert reloaded.QISKIT_RUNTIME_SMALLER_0_21 is expected_flags["SMALLER_0_21"]
    assert reloaded.QISKIT_RUNTIME_SMALLER_0_23 is expected_flags["SMALLER_0_23"]
    assert reloaded.QISKIT_RUNTIME_SMALLER_0_28 is expected_flags["SMALLER_0_28"]

    importlib.reload(qc)


def test_runtime_version_flags_when_ibm_runtime_missing(monkeypatch):
    """Test that version flags are None when qiskit-ibm-runtime is not installed."""
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


@pytest.mark.parametrize("version_string,expected_flags", QISKIT_VERSIONS)
def test_qiskit_version_flags(monkeypatch, version_string, expected_flags):
    """Test that Qiskit version flags are correctly set based on qiskit version."""
    fake_qiskit = sys.modules.get("qiskit")
    if fake_qiskit:
        monkeypatch.setattr(fake_qiskit, "__version__", version_string)

    reloaded = importlib.reload(qc)

    assert reloaded.QISKIT_SMALLER_1_2 is expected_flags["SMALLER_1_2"]
    assert reloaded.QISKIT_SMALLER_2_0 is expected_flags["SMALLER_2_0"]

    importlib.reload(qc)
