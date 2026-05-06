"""
Qiskit version-compatibility helpers.

This module centralises all version checks and wrappers around private / changed
Qiskit APIs so that the rest of the code-base stays free of ``try / except``
blocks and version-gated imports.

Supported Qiskit versions: >= 1.0  (including 2.x).
"""

from __future__ import annotations

from typing import Any, Protocol

from packaging import version
from qiskit import __version__ as qiskit_version
from sympy import sympify as _sympify


class _ParameterExpression(Protocol):
    """Structural interface for ``qiskit.circuit.ParameterExpression``.

    Defined as a Protocol to work around qiskit 2.x PyO3-backed stubs that
    expose ``ParameterExpression`` as a variable assignment rather than a class,
    which Pylance rejects as a valid type form.
    """

    @property
    def parameters(self) -> frozenset:
        """Free Parameter objects in the expression."""
        raise NotImplementedError

    def sympify(self) -> Any:
        """Convert to a sympy expression."""
        raise NotImplementedError

    def __float__(self) -> float:
        """Extract the numeric value of a constant expression."""
        raise NotImplementedError


# ── Qiskit version flags ──────────────────────────────────────────────────
QISKIT_SMALLER_1_2 = version.parse(qiskit_version) < version.parse("1.2.0")
QISKIT_SMALLER_2_0 = version.parse(qiskit_version) < version.parse("2.0.0")

# ── qiskit-ibm-runtime version flags ──────────────────────────────────────
# These are only meaningful when qiskit-ibm-runtime is installed.
# The flags are set to ``None`` when the package is not available and
# should be treated as "feature not available".
try:
    from qiskit_ibm_runtime import (
        __version__ as _ibm_runtime_version,
    )

    QISKIT_RUNTIME_AVAILABLE = True
    QISKIT_RUNTIME_SMALLER_0_21 = version.parse(_ibm_runtime_version) < version.parse("0.21.0")
    QISKIT_RUNTIME_SMALLER_0_23 = version.parse(_ibm_runtime_version) < version.parse("0.23.0")
    QISKIT_RUNTIME_SMALLER_0_28 = version.parse(_ibm_runtime_version) < version.parse("0.28.0")
except ImportError:
    QISKIT_RUNTIME_AVAILABLE = False
    QISKIT_RUNTIME_SMALLER_0_21 = None
    QISKIT_RUNTIME_SMALLER_0_23 = None
    QISKIT_RUNTIME_SMALLER_0_28 = None


# ── ParameterExpression helpers ────────────────────────────────────────────


def _param_to_sympy(param: _ParameterExpression):
    """Convert a ``ParameterExpression`` to a *sympy* expression.

    * Qiskit >= 2.0 exposes :py:meth:`ParameterExpression.sympify`.
    * Qiskit < 2.0 stores the expression in the private ``_symbol_expr``.
    """
    if QISKIT_SMALLER_2_0:
        return _sympify(getattr(param, "_symbol_expr"))
    return param.sympify()


def _param_is_constant(param: _ParameterExpression) -> bool:
    """Return ``True`` when *param* contains no free symbols.

    In Qiskit 1.x this was checked via ``param._symbol_expr is None``;
    the public API ``param.parameters`` is stable across all versions.
    """
    return len(param.parameters) == 0


def _param_to_float(param: _ParameterExpression) -> float:
    """Extract the numeric value from a *constant* ``ParameterExpression``.

    Replaces the private ``param._coeff`` accessor.
    """
    return float(param)


def _param_free_symbols(param: _ParameterExpression):
    """Return the free ``Parameter`` objects inside *param*.

    * Qiskit < 2.0 exposed ``param._parameter_symbols.keys()``.
    * The public ``param.parameters`` property works in all versions.
    """
    return param.parameters
