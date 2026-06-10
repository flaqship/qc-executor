"""Framework-independent parameter types backed by SymPy.

A :class:`Parameter` is a symbolic placeholder for a gate angle or an operator
coefficient. Parameters support arithmetic (``theta[0] * 2 + phi[1]``) because
they subclass :class:`sympy.Symbol`; SymPy evaluates the resulting expressions
symbolically. The expressions can later be converted to a backend's native
parameter type (e.g. a Qiskit ``ParameterVector``) by the circuit/operator
converters.

This layer deliberately does **not** depend on Qiskit, so circuits and
operators can be defined independently of any quantum framework.
"""

from __future__ import annotations

import re
from typing import List

import sympy as sp

# Matches a parameter symbol name like "theta[3]" -> ("theta", 3)
_NAME_PATTERN = re.compile(r"^([A-Za-z_]\w*)\[(\d+)\]$")


class Parameter(sp.Symbol):
    """A single named parameter element, e.g. ``theta[0]``.

    Subclasses :class:`sympy.Symbol` so it can be used directly inside
    arithmetic expressions that SymPy understands::

        x = ParameterVector("x", 2)
        expr = 2 * x[0] + x[1]      # -> sympy expression

    Args:
        vector_name: Name of the owning parameter vector (e.g. ``"theta"``).
        index: Position of this element within the vector.
    """

    # Allow instance attributes in addition to sympy.Symbol's __slots__.
    __slots__ = ("_vector_name", "_index")

    def __new__(cls, vector_name: str, index: int) -> "Parameter":
        name = f"{vector_name}[{index}]"
        # real=True so generated expressions stay in the real domain, matching
        # the "only real parameters" assumption used throughout the project.
        obj = super().__new__(cls, name, real=True)
        obj._vector_name = vector_name
        obj._index = index
        return obj

    @property
    def vector_name(self) -> str:
        """Name of the owning parameter vector."""
        return self._vector_name

    @property
    def index(self) -> int:
        """Index of this element within its vector."""
        return self._index


class ParameterVector:
    """An ordered, resizable collection of :class:`Parameter` elements.

    Mirrors the mental model of Qiskit's ``ParameterVector`` but is backed by
    SymPy and carries no Qiskit dependency. Behaves like a sequence: supports
    indexing, iteration and ``len()``.

    Args:
        name: Name shared by all elements (e.g. ``"theta"``).
        length: Initial number of elements.
    """

    def __init__(self, name: str, length: int = 0):
        self._name = name
        self._params: List[Parameter] = [Parameter(name, i) for i in range(length)]

    @property
    def name(self) -> str:
        """The shared name of the vector."""
        return self._name

    @property
    def params(self) -> List[Parameter]:
        """The contained :class:`Parameter` elements (do not mutate)."""
        return self._params

    def index(self, value: Parameter) -> int:
        """Return the position of ``value`` within the vector."""
        return self._params.index(value)

    def resize(self, length: int) -> None:
        """Grow or shrink the vector to ``length`` elements."""
        if length > len(self._params):
            self._params.extend(
                Parameter(self._name, i) for i in range(len(self._params), length)
            )
        else:
            del self._params[length:]

    def __getitem__(self, key) -> Parameter:
        return self._params[key]

    def __iter__(self):
        return iter(self._params)

    def __len__(self) -> int:
        return len(self._params)

    def __str__(self) -> str:
        return f"{self._name}, {[str(p) for p in self._params]}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self._name!r}, length={len(self)})"


def free_parameters(expr) -> List[Parameter]:
    """Return the :class:`Parameter` symbols contained in a value or expression.

    Args:
        expr: A float, a :class:`Parameter`, or any SymPy expression built from
            parameters.

    Returns:
        Sorted list of the :class:`Parameter` symbols, ordered by
        (vector_name, index). Empty if ``expr`` carries no free parameters.
    """
    if not isinstance(expr, sp.Basic):
        return []
    params = [s for s in expr.free_symbols if isinstance(s, Parameter)]
    return sorted(params, key=lambda p: (p.vector_name, p.index))


def parse_symbol_name(name: str):
    """Split a parameter symbol name like ``"theta[3]"`` into ``("theta", 3)``.

    Returns ``(name, None)`` for plain names without an index.
    """
    match = _NAME_PATTERN.match(name)
    if match:
        return match.group(1), int(match.group(2))
    return name, None
