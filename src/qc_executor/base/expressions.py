"""Compile SymPy angle and coefficient expressions into fast callables.

Backends evaluate every symbolic gate angle for every circuit execution, so the
callable has to be cheap to build and cheap to call.  ``sympy.lambdify`` is
neither when it is handed the full parameter vector of a large circuit: its
preprocessing is linear in the number of arguments (about 25 ms for a
230-parameter circuit, per gate), which dominated VQE runs.  Almost every angle
is ``c * theta[i]`` or a short linear combination, so those are turned into
plain closures over the parameter indices; anything else is lambdified over the
symbols it actually uses and wrapped to pick them out of the full vector.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Sequence

import sympy as sp
from sympy import lambdify


def _as_number(value: sp.Basic) -> float | complex:
    """Convert a numeric SymPy atom to a Python number, real if possible."""
    number = complex(value)
    return number.real if number.imag == 0 else number


def _linear_terms(expr: sp.Basic):
    """Return ``(constant, [(symbol, coefficient), ...])`` for a linear expression.

    Returns:
        The decomposition, or ``None`` if ``expr`` is not linear in its symbols
        with numeric coefficients.
    """
    constant: float | complex = 0.0
    terms = []
    for term, coeff in expr.as_coefficients_dict().items():
        if not coeff.is_number:
            return None
        if term is sp.S.One:
            constant = _as_number(coeff)
        elif isinstance(term, sp.Symbol):
            terms.append((term, _as_number(coeff)))
        else:
            return None
    return constant, terms


def _build(expr: sp.Basic, symbols: tuple, modules: Any, printer: Any) -> Callable[..., Any]:
    index = {symbol: position for position, symbol in enumerate(symbols)}
    free = expr.free_symbols
    if not free.issubset(index):
        # Keep lambdify's behaviour (a NameError on call) for unknown symbols.
        return lambdify(symbols, expr, modules=modules, printer=printer)

    linear = _linear_terms(expr)
    if linear is not None:
        constant, terms = linear
        if len(terms) == 1 and constant == 0:
            ((symbol, coeff),) = terms
            position = index[symbol]
            if coeff == 1:
                return lambda *values: values[position]
            return lambda *values: coeff * values[position]
        positions = [(index[symbol], coeff) for symbol, coeff in terms]

        def linear_function(*values):
            result = constant
            for position, coeff in positions:
                result = result + coeff * values[position]
            return result

        return linear_function

    used = sorted(free, key=lambda symbol: index[symbol])
    positions = [index[symbol] for symbol in used]
    function = lambdify(used, expr, modules=modules, printer=printer)
    return lambda *values: function(*[values[position] for position in positions])


@lru_cache(maxsize=8192)
def _build_cached(expr: sp.Basic, symbols: tuple) -> Callable[..., Any]:
    return _build(expr, symbols, None, None)


def compile_expression(
    expr: sp.Basic, symbols: Sequence[sp.Symbol], modules: Any = None, printer: Any = None
) -> Callable[..., Any]:
    """Return a callable ``f(*values)`` evaluating ``expr``.

    Args:
        expr: The expression to evaluate.
        symbols: The symbols whose values are passed positionally to ``f``, in
            that order; typically all parameters of a circuit or operator.
        modules: Passed to :func:`sympy.lambdify` for non-linear expressions.
        printer: Passed to :func:`sympy.lambdify` for non-linear expressions.

    Returns:
        A callable taking one value per entry of ``symbols``.  Linear
        expressions become closures over the parameter positions; other
        expressions are lambdified over the symbols they use.  Results for the
        default ``modules``/``printer`` are cached, so repeated angles across
        gates and circuits compile once.
    """
    symbols = tuple(symbols)
    if modules is None and printer is None:
        return _build_cached(expr, symbols)
    return _build(expr, symbols, modules, printer)
