"""Shared pytest fixtures and helpers for the ``qc_executor`` test suite.

The suite runs against a variable set of backends: ``pauli_propagation`` is pure
Python and always available, while ``qiskit``, ``pennylane`` and ``qulacs`` are
optional extras.  The helpers here let a test declare which backends it needs
without repeating :func:`pytest.importorskip` calls in every module.
"""

from __future__ import annotations

import importlib.util
from typing import Iterable

import pytest

#: All backend names registered by the package, in a stable order.
ALL_BACKENDS = ("qiskit", "pennylane", "qulacs", "qrisp", "pauli_propagation")

#: Third-party distribution each backend needs; ``None`` means "pure Python".
_BACKEND_REQUIREMENTS: dict[str, str | None] = {
    "qiskit": "qiskit",
    "pennylane": "pennylane",
    "qulacs": "qulacs",
    "qrisp": "qrisp",
    "pauli_propagation": None,
}


def _is_installed(backend: str) -> bool:
    """Return True if ``backend``'s third-party requirement is importable."""
    requirement = _BACKEND_REQUIREMENTS[backend]
    if requirement is None:
        return True
    return importlib.util.find_spec(requirement) is not None


#: Backends usable in the active environment, computed once at collection time.
INSTALLED_BACKENDS = tuple(name for name in ALL_BACKENDS if _is_installed(name))


def requires_backends(*backends: str):
    """Return a ``skipif`` marker for tests needing specific backends.

    Args:
        *backends: Backend names, e.g. ``"qulacs"``.

    Returns:
        A :func:`pytest.mark.skipif` marker that skips when any named backend is
        unavailable.
    """
    missing = [name for name in backends if name not in INSTALLED_BACKENDS]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"backend(s) not installed: {', '.join(missing)}",
    )


def parametrize_backends(backends: Iterable[str] | None = None):
    """Return a ``parametrize`` decorator over the installed backends.

    Args:
        backends: Restrict to these backend names.  Defaults to all of them.

    Returns:
        A :func:`pytest.mark.parametrize` decorator binding ``backend_name``.
    """
    candidates = tuple(backends) if backends is not None else ALL_BACKENDS
    selected = [name for name in candidates if name in INSTALLED_BACKENDS]
    return pytest.mark.parametrize("backend_name", selected, ids=selected)


@pytest.fixture(scope="session")
def installed_backends() -> tuple[str, ...]:
    """The backends available in the active test environment."""
    return INSTALLED_BACKENDS
