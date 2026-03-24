"""
Qiskit version-compatibility helpers.

This module centralises all version checks and wrappers around private / changed
Qiskit APIs so that the rest of the code-base stays free of ``try / except``
blocks and version-gated imports.

Supported Qiskit versions: >= 1.0  (including 2.x).
"""

from __future__ import annotations

from packaging import version
from qiskit import __version__ as qiskit_version
from qiskit.circuit import ParameterExpression
from sympy import sympify as _sympify

# ── version flags ──────────────────────────────────────────────────────────
QISKIT_SMALLER_1_2 = version.parse(qiskit_version) < version.parse("1.2.0")
QISKIT_SMALLER_2_0 = version.parse(qiskit_version) < version.parse("2.0.0")


# ── ParameterExpression helpers ────────────────────────────────────────────


def _param_to_sympy(param: ParameterExpression):
    """Convert a ``ParameterExpression`` to a *sympy* expression.

    * Qiskit >= 2.0 exposes :py:meth:`ParameterExpression.sympify`.
    * Qiskit < 2.0 stores the expression in the private ``_symbol_expr``.
    """
    if hasattr(param, "sympify"):
        return param.sympify()
    # Fallback for Qiskit < 2.0
    return _sympify(param._symbol_expr)  # pylint: disable=protected-access


def _param_is_constant(param: ParameterExpression) -> bool:
    """Return ``True`` when *param* contains no free symbols.

    In Qiskit 1.x this was checked via ``param._symbol_expr is None``;
    the public API ``param.parameters`` is stable across all versions.
    """
    return len(param.parameters) == 0


def _param_to_float(param: ParameterExpression) -> float:
    """Extract the numeric value from a *constant* ``ParameterExpression``.

    Replaces the private ``param._coeff`` accessor.
    """
    return float(param)


def _param_free_symbols(param: ParameterExpression):
    """Return the free ``Parameter`` objects inside *param*.

    * Qiskit < 2.0 exposed ``param._parameter_symbols.keys()``.
    * The public ``param.parameters`` property works in all versions.
    """
    return param.parameters
