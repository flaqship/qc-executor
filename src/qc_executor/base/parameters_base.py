"""Framework-independent helpers for normalising and binding parameter values.

Executors accept parameter values as keyword arguments in either vector form
(``x=[0.1, 0.2]``) or indexed form (``x[0]=0.1, x[1]=0.2``).  Every backend then
has to turn those into concrete numbers for the symbols appearing in a circuit
or observable.  The functions here are the single implementation of that path,
shared by all backends.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import sympy as sp

from ..parameters import Parameter, canonicalize

__all__ = [
    "normalize_values",
    "flatten_indexed",
    "build_binding",
    "substitute",
    "evaluate",
]

#: Matches an indexed parameter keyword such as ``"x[0]"``.
_INDEXED_KEY = re.compile(r"^([a-zA-Z_]\w*)\[(\d+)\]$")


def normalize_values(**parameters: Any) -> dict:
    """Normalise parameter keyword arguments to vector form.

    Indexed keys are collected into lists, so ``x[0]=0.1, x[1]=0.2`` becomes
    ``{"x": [0.1, 0.2]}``.  Vector-form keys pass through unchanged.

    Args:
        ``**parameters``: Parameter values as passed to a public executor method.

    Returns:
        Dictionary keyed by vector name.

    Raises:
        ValueError: If a name is given in both forms, or if an indexed
            parameter has gaps in its indices.
    """
    normalized: dict = {}
    indexed: dict[str, dict[int, Any]] = {}
    vector_names = set()

    for key, value in parameters.items():
        match = _INDEXED_KEY.match(key)
        if match:
            indexed.setdefault(match.group(1), {})[int(match.group(2))] = value
        else:
            vector_names.add(key)
            normalized[key] = value

    conflicting = sorted(vector_names.intersection(indexed))
    if conflicting:
        raise ValueError(
            f"Cannot mix vector and indexed parameter forms for: {', '.join(conflicting)}"
        )

    for name, index_map in indexed.items():
        vector_form = [index_map.get(i) for i in range(max(index_map) + 1)]
        if any(value is None for value in vector_form):
            raise ValueError(
                f"Incomplete indexed parameters for '{name}': "
                "missing indices would produce None values in the vector form."
            )
        normalized[name] = vector_form

    return normalized


def flatten_indexed(parameters: Mapping[str, Any]) -> dict[str, float]:
    """Expand vector-form values into per-element ``"name[i]"`` keys.

    ``{"x": [0.1, 0.2]}`` becomes ``{"x[0]": 0.1, "x[1]": 0.2}``.  Scalars are
    kept under their plain name as well as ``"name[0]"``, so either spelling
    resolves.

    Args:
        parameters: Values keyed by vector name.

    Returns:
        Dictionary keyed by individual parameter name.
    """
    flat: dict[str, float] = {}
    for name, value in parameters.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            for i, element in enumerate(np.asarray(value).reshape(-1)):
                flat[f"{name}[{i}]"] = float(element)
        else:
            flat[name] = float(value)
            flat[f"{name}[0]"] = float(value)
    return flat


def build_binding(
    free_params: Iterable[Parameter], parameters: Mapping[str, Any]
) -> dict[Parameter, float]:
    """Map executor parameter values onto the symbols they bind.

    Args:
        free_params: The parameters appearing in a circuit or observable.
        parameters: Values in vector form, as returned by :func:`normalize_values`.

    Returns:
        Dictionary mapping each resolved parameter to its numeric value.

    Raises:
        ValueError: If a parameter in ``free_params`` has no supplied value.
    """
    flat = flatten_indexed(parameters)
    binding: dict[Parameter, float] = {}
    missing: list[str] = []

    for param in free_params:
        if param.name in flat:
            binding[param] = flat[param.name]
        else:
            missing.append(param.name)

    if missing:
        raise ValueError(f"Missing parameter values for: {', '.join(sorted(missing))}")

    return binding


def substitute(expr: Any, binding: Mapping[Parameter, float]) -> Any:
    """Substitute values into an expression, keeping unbound symbols symbolic.

    Args:
        expr: A number or SymPy expression.
        binding: Values to substitute, as returned by :func:`build_binding`.

    Returns:
        A ``float`` if the result is fully determined, otherwise the partially
        substituted SymPy expression.
    """
    if not isinstance(expr, sp.Basic):
        return float(expr)
    result = canonicalize(expr).xreplace(dict(binding))
    if result.free_symbols:
        return result
    return float(result)


def evaluate(expr: Any, binding: Mapping[Parameter, float]) -> float:
    """Substitute values into an expression and require a numeric result.

    Args:
        expr: A number or SymPy expression.
        binding: Values to substitute, as returned by :func:`build_binding`.

    Returns:
        The numeric value of ``expr``.

    Raises:
        ValueError: If symbols remain unbound after substitution.
    """
    result = substitute(expr, binding)
    if not isinstance(result, float):
        remaining = sorted(str(s) for s in result.free_symbols)
        raise ValueError(f"Expression is not fully bound; missing: {', '.join(remaining)}")
    return result


def values_to_sequence(value: Any) -> Sequence[float]:
    """Coerce a scalar or array-like parameter value into a flat sequence.

    Args:
        value: A scalar, list, tuple or array.

    Returns:
        A flat sequence of floats.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in np.asarray(value).reshape(-1)]
    return [float(value)]
