"""Framework-independent parameter types backed by SymPy.

A :class:`Parameter` is a symbolic placeholder for a gate angle or an operator
coefficient.  Parameters subclass :class:`sympy.Symbol`, so arithmetic such as
``2 * theta[0] + phi[1]`` yields an ordinary SymPy expression with no work on
our side, and no quantum framework is involved at any point.

Element names keep the ``"vector[index]"`` spelling (``"theta[0]"``) that the
executor parameter-passing API and every backend already rely on.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Sequence, Tuple, overload

import sympy as sp

__all__ = [
    "Parameter",
    "Parameters",
    "canonicalize",
    "free_parameters",
    "parse_symbol_name",
    "sort_parameters",
]

#: Matches a parameter symbol name such as ``"theta[3]"`` -> ``("theta", 3)``.
_NAME_PATTERN = re.compile(r"^([A-Za-z_]\w*)\[(\d+)\]$")

#: Sort key for un-indexed parameters, so they order before ``name[0]``.
_NO_INDEX = -1


def parse_symbol_name(name: str) -> Tuple[str, int | None]:
    """Split a parameter symbol name into its vector name and index.

    Args:
        name: A symbol name such as ``"theta[3]"`` or ``"theta"``.

    Returns:
        ``("theta", 3)`` for indexed names, ``("theta", None)`` otherwise.
    """
    match = _NAME_PATTERN.match(name)
    if match:
        return match.group(1), int(match.group(2))
    return name, None


class Parameter(sp.Symbol):  # pylint: disable=too-many-ancestors
    """A single named parameter, e.g. ``theta[0]``.

    Both spellings construct the same interned object, because SymPy caches
    symbols by name and assumptions::

        Parameter("theta", 0) is Parameter("theta[0]")   # True

    Parameters are created with ``real=True``: gate angles and observable
    coefficients are real throughout this project, and the assumption lets
    SymPy simplify expressions such as ``conjugate(theta[0])``.

    Args:
        name: Either the vector name (with ``index`` given) or a full element
            name such as ``"theta[0]"``.
        index: Position within the owning vector.  Omit for scalar parameters
            and when ``name`` already carries the index.
        ``**assumptions``: Extra SymPy assumptions; ``real=True`` by default.
    """

    __slots__ = ("_vector_name", "_index")

    def __new__(cls, name: str, index: int | None = None, **assumptions: Any) -> "Parameter":
        if not isinstance(name, str):
            raise TypeError(
                f"Parameter name must be a string, got {type(name).__name__}. "
                'Use Parameter("theta", 0) or Parameter("theta[0]").'
            )
        if index is None:
            vector_name, index = parse_symbol_name(name)
            full_name = name
        else:
            vector_name = name
            full_name = f"{name}[{index}]"
        assumptions.setdefault("real", True)
        obj = super().__new__(cls, full_name, **assumptions)
        # Re-assigning on a cache hit is idempotent: the values derive from the name.
        obj._vector_name = vector_name
        obj._index = index
        return obj

    def __getnewargs_ex__(self) -> Tuple[Tuple[str], dict]:
        """Support ``pickle`` and ``copy.deepcopy``.

        ``sympy.Symbol`` reconstructs through ``__new__``, so the arguments must
        match this class's signature.  Passing the full element name lets
        :meth:`__new__` recover the vector name and index.  Without this, the
        inherited implementation would call ``Parameter("theta[0]")`` with the
        wrong arity for any subclass taking a separate index argument.
        """
        return (self.name,), dict(self.assumptions0)

    @property
    def vector_name(self) -> str:
        """Name of the owning parameter vector."""
        return self._vector_name

    @property
    def index(self) -> int | None:
        """Position within the owning vector, or ``None`` for scalars."""
        return self._index

    @property
    def sort_key_tuple(self) -> Tuple[str, int]:
        """Ordering key placing ``x[9]`` before ``x[10]`` and ``p`` before ``x``."""
        return (self._vector_name, _NO_INDEX if self._index is None else self._index)


class Parameters(Sequence[Parameter]):
    """An ordered collection of :class:`Parameter` elements sharing a name.

    Behaves like a sequence, so ``len()``, indexing, slicing and iteration all
    work::

        x = Parameters("x", 3)
        angle = x[0] * 2 + x[1]

    Args:
        name: Name shared by all elements, e.g. ``"theta"``.
        length: Initial number of elements.
    """

    __slots__ = ("_name", "_params")

    def __init__(self, name: str, length: int = 0):
        if length < 0:
            raise ValueError(f"length must be non-negative, got {length}")
        self._name = name
        self._params: List[Parameter] = [Parameter(name, i) for i in range(length)]

    @property
    def name(self) -> str:
        """The name shared by every element."""
        return self._name

    @property
    def params(self) -> List[Parameter]:
        """The contained elements.  Treat as read-only."""
        return self._params

    def index(self, value: Any, start: int = 0, stop: int | None = None) -> int:
        """Return the position of ``value`` within the vector."""
        if stop is None:
            stop = len(self._params)
        return self._params.index(value, start, stop)

    def resize(self, length: int) -> None:
        """Grow or shrink the vector to ``length`` elements.

        Growing preserves existing element identities, so expressions already
        built from this vector stay valid.

        Args:
            length: The new number of elements.
        """
        if length < 0:
            raise ValueError(f"length must be non-negative, got {length}")
        current = len(self._params)
        if length > current:
            self._params.extend(Parameter(self._name, i) for i in range(current, length))
        else:
            del self._params[length:]

    @overload
    def __getitem__(self, key: int) -> Parameter: ...

    @overload
    def __getitem__(self, key: slice) -> List[Parameter]: ...

    def __getitem__(self, key):
        return self._params[key]

    def __len__(self) -> int:
        return len(self._params)

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._name, len(self._params)))

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Parameters)
            and self._name == other._name
            and len(self._params) == len(other._params)
        )

    def __str__(self) -> str:
        """The vector name.

        Deliberately just the name: callers pass ``str(vector)`` where a
        parameter-name string is expected, e.g. when requesting a gradient with
        respect to a whole vector.
        """
        return self._name

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self._name!r}, length={len(self._params)})"


def free_parameters(expr: Any) -> List[Parameter]:
    """Return the :class:`Parameter` symbols inside a value or expression.

    Args:
        expr: A number, a :class:`Parameter`, or any SymPy expression.

    Returns:
        The contained parameters sorted by ``(vector_name, index)``.  Empty when
        ``expr`` is numeric or carries no free parameters.
    """
    if not isinstance(expr, sp.Basic):
        return []
    params = [s for s in expr.free_symbols if isinstance(s, Parameter)]
    return sorted(params, key=lambda p: p.sort_key_tuple)


def sort_parameters(params: Iterable[Parameter]) -> List[Parameter]:
    """Sort parameters by ``(vector_name, index)``.

    Index ordering is numeric, so ``x[9]`` precedes ``x[10]``.

    Args:
        params: The parameters to sort.

    Returns:
        A new sorted list.
    """
    return sorted(params, key=lambda p: p.sort_key_tuple)


def canonicalize(expr: Any) -> Any:
    """Replace plain :class:`sympy.Symbol` instances with :class:`Parameter`.

    SymPy compares symbols by exact type and assumptions, so a bare
    ``sympy.Symbol("x[0]")`` is *not* equal to ``Parameter("x", 0)``.  Any
    expression arriving from outside — most notably from
    ``qiskit.circuit.ParameterExpression.sympify()`` — must pass through here
    before being compared against or substituted with our parameters.

    Args:
        expr: A number or SymPy expression.

    Returns:
        ``expr`` with every foreign symbol replaced by the matching
        :class:`Parameter`.  Numbers are returned unchanged.
    """
    if not isinstance(expr, sp.Basic):
        return expr
    # Key on the symbol objects themselves: two symbols with the same name but
    # different assumptions are distinct keys, and both must be replaced.
    replacements = {
        s: Parameter(s.name) for s in expr.free_symbols if not isinstance(s, Parameter)
    }
    return expr.xreplace(replacements) if replacements else expr
