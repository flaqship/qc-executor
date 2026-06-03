"""Shared parameter-binding helpers for Qiskit circuit/operator wrappers."""

from __future__ import annotations

import numpy as np


def _param_name(p) -> str:
    """Return the vector name of a parameter, falling back to its plain name."""
    return p.vector.name if hasattr(p, "vector") else p.name


def _param_index(p) -> int:
    """Return the index of a parameter within its vector, or 0 if standalone."""
    return p.index if hasattr(p, "index") else 0


def build_params_dict(free_parameters, parameter_values: dict) -> dict:
    """Map executor parameter values onto concrete Qiskit ``Parameter`` objects.

    Args:
        free_parameters: Iterable of Qiskit parameters present in the object.
        parameter_values: Dictionary mapping parameter-vector names to values.

    Returns:
        Dictionary mapping each matched Qiskit parameter to its bound value.
    """
    params_dict: dict = {}

    for param_name, values in parameter_values.items():
        # Ensure values is a list
        if not isinstance(values, (list, np.ndarray)):
            values = [values]

        # Match parameters with provided values.
        # Guard against standalone Parameter objects (no .vector/.index).
        matching_params = [p for p in free_parameters if _param_name(p) == param_name]
        # Sort by index to ensure correct ordering
        matching_params = sorted(matching_params, key=_param_index)

        for i, param in enumerate(matching_params):
            if i < len(values):
                params_dict[param] = values[i]

    return params_dict
