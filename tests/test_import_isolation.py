"""Guards on what ``import qc_executor`` pulls into ``sys.modules``.

The core package is meant to be framework independent: importing it must not
drag in a quantum framework.  These checks run in a subprocess because the
pytest process has already imported everything by the time a test executes.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

_PROBE = (
    "import json, sys; "
    "import qc_executor; "
    "json.dump(sorted(name for name in sys.modules if '.' not in name), sys.stdout)"
)


def _top_level_modules_after_import() -> set[str]:
    """Import ``qc_executor`` in a clean interpreter and report loaded modules.

    Returns:
        The set of top-level module names present in ``sys.modules`` afterwards.

    Raises:
        AssertionError: If the subprocess fails to import the package.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        completed.returncode == 0
    ), f"importing qc_executor failed in a clean interpreter:\n{completed.stderr}"
    return set(json.loads(completed.stdout))


class TestImportIsolation:
    def test_package_imports_in_a_clean_interpreter(self):
        modules = _top_level_modules_after_import()
        assert "qc_executor" in modules

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "qiskit is still a core dependency; the qiskit backend is imported eagerly by "
            "qc_executor/__init__.py. Remove this marker once qiskit becomes an optional extra."
        ),
    )
    def test_importing_qc_executor_does_not_import_qiskit(self):
        assert "qiskit" not in _top_level_modules_after_import()

    def test_missing_optional_backends_degrade_gracefully(self):
        """Backends whose extra is absent must be ``None``, not an ImportError."""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib.util as u, qc_executor as q; "
                "print(all((getattr(q, n) is not None) == (u.find_spec(n) is not None) "
                "for n in ('pennylane', 'qulacs')))",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "True"
