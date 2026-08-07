"""Translation between SymPy expressions and Qiskit ``ParameterExpression``.

The framework-independent layer expresses gate angles and observable
coefficients as SymPy, while Qiskit's primitives — in particular the OpTree
derivative machinery — differentiate and bind its own ``ParameterExpression``
objects.  This module is the boundary between the two.

Qiskit offers a one-way ``ParameterExpression.sympify()``; there is no public
constructor taking a SymPy expression.  :func:`to_qiskit_expr` therefore walks
the expression tree and rebuilds it out of Qiskit's own arithmetic, which
supports ``+ - * / **`` plus a fixed set of unary functions.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List

import sympy as sp
from qiskit.circuit import ParameterVector

from ..parameters import Parameter, canonicalize
from ._compat import _param_is_constant, _param_to_float, _param_to_sympy

__all__ = [
    "UnsupportedExpressionError",
    "QiskitParameterFactory",
    "to_qiskit_expr",
    "from_qiskit_expr",
    "to_qiskit_params",
]


class UnsupportedExpressionError(NotImplementedError):
    """Raised when a SymPy expression has no Qiskit equivalent."""


#: SymPy function -> the ``ParameterExpression`` method implementing it.
_UNARY_FUNCTIONS: Dict[Any, str] = {
    sp.sin: "sin",
    sp.cos: "cos",
    sp.tan: "tan",
    sp.asin: "arcsin",
    sp.acos: "arccos",
    sp.atan: "arctan",
    sp.exp: "exp",
    sp.log: "log",
    sp.Abs: "abs",
    sp.sign: "sign",
    sp.conjugate: "conjugate",
}


class QiskitParameterFactory:
    """Creates and reuses ``ParameterVector`` objects by name.

    Qiskit parameters are compared by identity of their underlying symbol, so
    every reference to ``x[0]`` within one circuit must resolve to the same
    object.  The factory holds one vector per name and grows it on demand.
    """

    def __init__(self) -> None:
        self._vectors: Dict[str, ParameterVector] = {}

    def vector(self, name: str, min_length: int) -> ParameterVector:
        """Return the vector called ``name``, at least ``min_length`` long.

        Args:
            name: The parameter-vector name.
            min_length: Minimum number of elements required.

        Returns:
            The cached ``ParameterVector``, resized if necessary.
        """
        existing = self._vectors.get(name)
        if existing is None:
            created = ParameterVector(name, min_length)
            self._vectors[name] = created
            return created
        if len(existing) < min_length:
            # resize() preserves the identity of existing elements.
            existing.resize(min_length)
        return existing

    def element(self, param: Parameter):
        """Return the Qiskit element corresponding to ``param``.

        Args:
            param: A framework-independent parameter.

        Returns:
            The matching ``ParameterVectorElement``.
        """
        index = 0 if param.index is None else param.index
        return self.vector(param.vector_name, index + 1)[index]

    @property
    def vectors(self) -> Dict[str, ParameterVector]:
        """The vectors created so far, keyed by name."""
        return self._vectors


def _rebuild(expr: sp.Basic, factory: QiskitParameterFactory) -> Any:
    """Recursively rebuild a SymPy expression using Qiskit arithmetic."""
    # Constants first: this covers Integer, Float, Rational, pi, E and any
    # symbol-free combination of them.
    if not expr.free_symbols:
        value = complex(expr)
        if value.imag != 0.0:
            raise UnsupportedExpressionError(
                f"Complex constants are not supported by Qiskit parameters: {expr}"
            )
        return value.real

    if isinstance(expr, sp.Symbol):
        if not isinstance(expr, Parameter):
            expr = Parameter(expr.name)
        return factory.element(expr)

    if expr.is_Add:
        terms = [_rebuild(arg, factory) for arg in expr.args]
        result = terms[0]
        for term in terms[1:]:
            result = result + term
        return result

    if expr.is_Mul:
        factors = [_rebuild(arg, factory) for arg in expr.args]
        result = factors[0]
        for factor in factors[1:]:
            result = result * factor
        return result

    # exp() must be handled before Pow: SymPy stores exp(x) as its own function.
    if expr.func in _UNARY_FUNCTIONS:
        if len(expr.args) != 1:  # pragma: no cover - defensive
            # Unreachable with the current table: SymPy normalises the only
            # multi-argument candidate, log(x, b), into log(x)/log(b).
            raise UnsupportedExpressionError(f"Expected a unary function, got {expr}")
        operand = _rebuild(expr.args[0], factory)
        method = _UNARY_FUNCTIONS[expr.func]
        if not hasattr(operand, method):  # pragma: no cover - defensive
            # Unreachable in practice: a symbol-free operand is folded to a float
            # by the constant branch above, which also folds the enclosing call.
            raise UnsupportedExpressionError(
                f"{method}() requires a parameter expression operand, got {expr}"
            )
        return getattr(operand, method)()

    if expr.is_Pow:
        base, exponent = expr.args
        if exponent.free_symbols:
            raise UnsupportedExpressionError(
                f"Qiskit parameters do not support symbolic exponents: {expr}"
            )
        return _rebuild(base, factory) ** float(exponent)

    raise UnsupportedExpressionError(
        f"Cannot translate {type(expr).__name__} to a Qiskit parameter expression: {expr}"
    )


#: Process-wide factory backing every conversion that does not pass its own.
#
# Qiskit compares parameters by UUID, not by name, so two independently created
# ``ParameterVector("x", 2)`` objects have unequal elements.  Binding and
# differentiation both need ``x[0]`` in a circuit and ``x[0]`` in an observable
# to be the *same* Qiskit object, so all conversions share one factory keyed by
# name.  It holds one small vector per parameter name.
_DEFAULT_FACTORY = QiskitParameterFactory()


def default_factory() -> QiskitParameterFactory:
    """Return the shared factory used when no explicit one is supplied."""
    return _DEFAULT_FACTORY


def to_qiskit_expr(expr: Any, factory: QiskitParameterFactory | None = None) -> Any:
    """Convert a number or SymPy expression to a Qiskit parameter expression.

    Args:
        expr: A number, a :class:`~qc_executor.parameters.Parameter`, or any
            SymPy expression built from parameters.
        factory: Reuse this factory instead of the shared one.  Mainly useful in
            tests that need isolated parameter identities.

    Returns:
        A ``float`` for numeric input, otherwise a ``ParameterExpression``.

    Raises:
        UnsupportedExpressionError: If the expression uses a construct with no
            Qiskit equivalent (e.g. ``Piecewise``, ``floor``, symbolic powers).
    """
    if not isinstance(expr, sp.Basic):
        return expr
    return _rebuild(expr, factory if factory is not None else _DEFAULT_FACTORY)


def from_qiskit_expr(expr: Any) -> Any:
    """Convert a Qiskit parameter expression back to SymPy.

    Args:
        expr: A number or ``ParameterExpression``.

    Returns:
        A ``float`` for constant input, otherwise a SymPy expression whose
        symbols are :class:`~qc_executor.parameters.Parameter` instances.
    """
    if isinstance(expr, (int, float)):
        return float(expr)
    if not hasattr(expr, "sympify"):
        return expr
    if _param_is_constant(expr):
        return _param_to_float(expr)
    return canonicalize(_param_to_sympy(expr))


def to_qiskit_params(
    params: Iterable[Parameter], factory: QiskitParameterFactory | None = None
) -> List[Any]:
    """Convert parameters to their Qiskit counterparts.

    Args:
        params: The parameters to convert.
        factory: Reuse this factory instead of the shared one.  Must be the same
            factory used for any expressions these parameters are compared or
            bound against.

    Returns:
        The matching ``ParameterVectorElement`` objects, in input order.
    """
    resolved = factory if factory is not None else _DEFAULT_FACTORY
    return [resolved.element(param) for param in params]


def make_angle_converter() -> Callable[[Any], Any]:
    """Return a converter that keeps parameter identity across repeated calls.

    Every angle in one circuit must be translated with the same factory so that
    two references to ``x[0]`` produce the same Qiskit object.

    Returns:
        A callable converting a single angle.
    """
    factory = QiskitParameterFactory()
    return lambda angle: to_qiskit_expr(angle, factory)
