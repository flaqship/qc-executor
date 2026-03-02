"""Tests for the create_executor factory function."""

from unittest.mock import patch, MagicMock

import pytest

from executor._factory import create_executor


class TestCreateExecutor:
    """Tests for create_executor()."""

    def test_unsupported_backend_raises_value_error(self):
        """Passing an unknown backend string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend 'bogus'"):
            create_executor("bogus")

    def test_unsupported_backend_lists_options(self):
        """The ValueError message lists all supported backends."""
        with pytest.raises(ValueError, match="pennylane.*qiskit.*qulacs"):
            create_executor("unknown")

    def test_case_insensitive_backend(self):
        """Backend matching should be case-insensitive."""
        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.PennyLaneExecutor = mock_cls

        with patch("importlib.import_module", return_value=mock_module):
            create_executor("PennyLane", shots=1024)

        mock_cls.assert_called_once_with(shots=1024)

    def test_pennylane_backend(self):
        """Factory returns a PennyLaneExecutor for 'pennylane'."""
        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.PennyLaneExecutor = mock_cls

        with patch("importlib.import_module", return_value=mock_module) as mock_import:
            result = create_executor("pennylane", shots=512, seed=42)

        mock_import.assert_called_once_with("executor.pennylane.pennylane_executor")
        mock_cls.assert_called_once_with(shots=512, seed=42)
        assert result is mock_cls.return_value

    def test_qiskit_backend(self):
        """Factory returns a QiskitExecutor for 'qiskit'."""
        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.QiskitExecutor = mock_cls

        with patch("importlib.import_module", return_value=mock_module) as mock_import:
            result = create_executor("qiskit", shots=256)

        mock_import.assert_called_once_with("executor.qiskit.qiskit_executor")
        mock_cls.assert_called_once_with(shots=256)
        assert result is mock_cls.return_value

    def test_qulacs_backend(self):
        """Factory returns a QulacsExecutor for 'qulacs'."""
        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.QulacsExecutor = mock_cls

        with patch("importlib.import_module", return_value=mock_module) as mock_import:
            result = create_executor("qulacs")

        mock_import.assert_called_once_with("executor.qulacs.qulacs_executor")
        mock_cls.assert_called_once_with()
        assert result is mock_cls.return_value

    def test_missing_dependency_raises_import_error(self):
        """If the backend library is missing, ImportError with install hint is raised."""
        with patch(
            "importlib.import_module", side_effect=ImportError("No module named 'pennylane'")
        ):
            with pytest.raises(ImportError, match="pip install pennylane"):
                create_executor("pennylane")

    def test_missing_dependency_preserves_cause(self):
        """The original ImportError is chained as __cause__."""
        original = ImportError("No module named 'qulacs'")
        with patch("importlib.import_module", side_effect=original):
            with pytest.raises(ImportError) as exc_info:
                create_executor("qulacs")
            assert exc_info.value.__cause__ is original

    def test_kwargs_forwarded(self):
        """Extra kwargs are forwarded to the executor constructor."""
        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.QiskitExecutor = mock_cls

        with patch("importlib.import_module", return_value=mock_module):
            create_executor("qiskit", shots=100, seed=7, caching=True)

        mock_cls.assert_called_once_with(shots=100, seed=7, caching=True)

    def test_no_eager_backend_import(self):
        """Importing the factory module itself does not import any backend."""
        import sys

        # Remove cached backend modules to check they aren't re-imported
        removed = {}
        for mod_name in list(sys.modules):
            if mod_name.startswith(("executor.pennylane.", "executor.qiskit.", "executor.qulacs.")):
                removed[mod_name] = sys.modules.pop(mod_name)

        try:
            # Re-import the factory module
            import importlib
            importlib.reload(__import__("executor._factory"))

            # Backend executor modules should not be in sys.modules
            for mod_name in (
                "executor.pennylane.pennylane_executor",
                "executor.qiskit.qiskit_executor",
                "executor.qulacs.qulacs_executor",
            ):
                assert mod_name not in sys.modules, (
                    f"{mod_name} was eagerly imported by the factory module"
                )
        finally:
            # Restore removed modules
            sys.modules.update(removed)
