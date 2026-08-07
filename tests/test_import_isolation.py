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

    @pytest.mark.parametrize("framework", ["qiskit", "pennylane", "qulacs"])
    def test_importing_qc_executor_does_not_import_a_framework(self, framework):
        """The headline property: the core package is framework independent.

        Backends resolve through the module ``__getattr__``, so none of them --
        Qiskit included -- is imported until something asks for it.
        """
        assert framework not in _top_level_modules_after_import()

    def test_a_backend_is_still_reachable_by_attribute(self):
        """Laziness must not make the backends unreachable."""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, qc_executor; "
                "assert 'qiskit' not in sys.modules; "
                "assert qc_executor.qiskit is not None; "
                "assert 'qiskit' in sys.modules; "
                "print('ok')",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "ok"

    def test_an_unknown_attribute_still_raises(self):
        import qc_executor  # noqa: PLC0415

        with pytest.raises(AttributeError, match="no attribute 'not_a_backend'"):
            _ = qc_executor.not_a_backend

    def test_the_resolved_backend_is_cached(self, monkeypatch):
        """A resolved backend lands in the module globals, so ``__getattr__`` runs once.

        By the time the suite runs, importing a backend's submodule has already
        bound it on the package, so the attribute has to be removed to reach the
        lazy path at all.
        """
        import qc_executor  # noqa: PLC0415

        monkeypatch.delattr(qc_executor, "pauli_propagation", raising=False)

        resolved = qc_executor.pauli_propagation

        assert resolved is not None
        assert qc_executor.__dict__["pauli_propagation"] is resolved

    def test_a_backend_whose_dependency_is_missing_resolves_to_none(self, monkeypatch):
        """Not installing an extra is not an error; the attribute is just ``None``."""
        import importlib  # noqa: PLC0415

        import qc_executor  # noqa: PLC0415

        def refuse(name, package=None):
            raise ImportError(f"No module named {name!r}")

        monkeypatch.delattr(qc_executor, "qulacs", raising=False)
        monkeypatch.setattr(importlib, "import_module", refuse)

        assert qc_executor.qulacs is None

    def test_dir_lists_the_backends(self):
        import qc_executor  # noqa: PLC0415

        assert {"qiskit", "pennylane", "qulacs", "pauli_propagation"} <= set(dir(qc_executor))

    @pytest.mark.parametrize("plugin", ["pennylane", "qulacs", "pauli_propagation"])
    def test_a_backend_plugin_does_not_import_qiskit(self, plugin):
        """No plugin but the Qiskit one may reach for Qiskit.

        Every non-Qiskit backend used to build its circuit by transpiling a
        Qiskit one and its operator by reading a ``SparsePauliOp``, so all three
        pulled Qiskit in.  Source is scanned rather than ``sys.modules``
        inspected because importing any submodule runs the package ``__init__``,
        which still imports the Qiskit backend eagerly.
        """
        import pathlib  # noqa: PLC0415

        import qc_executor  # noqa: PLC0415

        root = pathlib.Path(qc_executor.__file__).parent / plugin
        offenders = {
            f"{path.relative_to(root)}:{number}": line.strip()
            for path in sorted(root.rglob("*.py"))
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if line.startswith(("import qiskit", "from qiskit"))
        }

        assert not offenders, f"{plugin} still imports qiskit: {offenders}"

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
