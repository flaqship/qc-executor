"""Tests for the create_executor factory function."""

import importlib
import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest

# Import factory module directly to avoid triggering executor/__init__.py's
# eager backend imports (which may fail if specific backend versions are not installed).
_spec = importlib.util.spec_from_file_location(
    "executor.factory",
    str(Path(__file__).resolve().parent.parent / "src" / "executor" / "factory.py"),
)
_factory_mod = importlib.util.module_from_spec(_spec)
sys.modules["executor.factory"] = _factory_mod
_spec.loader.exec_module(_factory_mod)
create_executor = _factory_mod.create_executor


class TestCreateExecutorValidation:
    """Tests for input validation (no real backends needed)."""

    def test_unsupported_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown backend 'foo'"):
            create_executor("foo")

    def test_value_error_lists_supported_backends(self):
        with pytest.raises(ValueError, match="pennylane") as exc_info:
            create_executor("unknown")
        msg = str(exc_info.value)
        assert "qiskit" in msg
        assert "qulacs" in msg

    def test_backend_name_is_case_insensitive(self):
        """Passing upper/mixed case should not raise ValueError."""
        fake_module = mock.MagicMock()
        with mock.patch("importlib.import_module", return_value=fake_module):
            for name in ("PennyLane", "QISKIT", "Qulacs"):
                create_executor(name)  # no ValueError raised


class TestCreateExecutorLazyImport:
    """Tests that verify lazy-import behaviour using mocks."""

    def test_pennylane_backend_returns_correct_class(self):
        sentinel = object()
        fake_cls = mock.MagicMock(return_value=sentinel)
        fake_module = mock.MagicMock()
        fake_module.PennylaneExecutor = fake_cls

        with mock.patch("importlib.import_module", return_value=fake_module) as imp:
            result = create_executor("pennylane", shots=1024, seed=42)

        imp.assert_called_once_with("executor.pennylane.pennylane_executor")
        fake_cls.assert_called_once_with(shots=1024, seed=42)
        assert result is sentinel

    def test_qiskit_backend_returns_correct_class(self):
        sentinel = object()
        fake_cls = mock.MagicMock(return_value=sentinel)
        fake_module = mock.MagicMock()
        fake_module.QiskitExecutor = fake_cls

        with mock.patch("importlib.import_module", return_value=fake_module) as imp:
            result = create_executor("qiskit", shots=512)

        imp.assert_called_once_with("executor.qiskit.executor_qiskit")
        fake_cls.assert_called_once_with(shots=512)
        assert result is sentinel

    def test_qulacs_backend_returns_correct_class(self):
        sentinel = object()
        fake_cls = mock.MagicMock(return_value=sentinel)
        fake_module = mock.MagicMock()
        fake_module.QulacsExecutor = fake_cls

        with mock.patch("importlib.import_module", return_value=fake_module) as imp:
            result = create_executor("qulacs")

        imp.assert_called_once_with("executor.qulacs.qulacs_executor")
        fake_cls.assert_called_once_with()
        assert result is sentinel

    def test_missing_library_raises_import_error_with_hint(self):
        with mock.patch(
            "importlib.import_module", side_effect=ImportError("No module named 'pennylane'")
        ):
            with pytest.raises(ImportError, match="pip install pennylane"):
                create_executor("pennylane")

    def test_missing_qiskit_raises_import_error_with_hint(self):
        with mock.patch(
            "importlib.import_module", side_effect=ImportError("No module named 'qiskit'")
        ):
            with pytest.raises(ImportError, match="pip install qiskit"):
                create_executor("qiskit")

    def test_missing_qulacs_raises_import_error_with_hint(self):
        with mock.patch(
            "importlib.import_module", side_effect=ImportError("No module named 'qulacs'")
        ):
            with pytest.raises(ImportError, match="pip install qulacs"):
                create_executor("qulacs")

    def test_kwargs_forwarded_to_constructor(self):
        fake_cls = mock.MagicMock()
        fake_module = mock.MagicMock()
        fake_module.QulacsExecutor = fake_cls

        with mock.patch("importlib.import_module", return_value=fake_module):
            create_executor("qulacs", shots=100, seed=7, caching=True)

        fake_cls.assert_called_once_with(shots=100, seed=7, caching=True)
