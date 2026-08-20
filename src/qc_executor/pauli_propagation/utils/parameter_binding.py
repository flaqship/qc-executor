"""Parameter binding for the Pauli-propagation gate list.

Split out of the former Qiskit converter, which this backend no longer needs:
circuits now arrive through the shared framework-independent IR.  Nothing here
touches Qiskit.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .gates import Gate, LayerBarrier

__all__ = ["bind_parameters"]


def bind_parameters(
    gates: List[Gate | LayerBarrier],
    parameter_values: Dict[str, float],
) -> Dict[str, float]:
    """Create a complete parameter binding dict for gates.

    Extracts concrete parameter values from gates and merges them with
    provided parameter_values.

    Args:
        gates: List of internal gates (may include LayerBarrier markers)
        parameter_values: Dict mapping parameter names to values

    Returns:
        Complete parameter dict with all required values

    Raises:
        ValueError: If required parameters are missing
        TypeError: If an unexpected object type is in the gates list
    """
    # First, expand any list values in parameter_values to indexed format
    # This converts {"x": [0.1], "p": [0.5, 0.6]} to individual indexed entries
    expanded_params = {}
    for key, value in parameter_values.items():
        if isinstance(value, (list, tuple)):
            # For each value in the list, create an indexed key
            for idx, val in enumerate(value):
                indexed_key = f"{key}[{idx}]"
                if not isinstance(val, (int, float, np.number)):
                    raise TypeError(
                        f"Parameter '{indexed_key}' has invalid type {type(val)}. "
                        f"Expected float or numeric value."
                    )
                expanded_params[indexed_key] = float(val)
        elif isinstance(value, (int, float, np.number)):
            expanded_params[key] = float(value)
        else:
            raise TypeError(
                f"Parameter '{key}' has invalid value type {type(value)}. "
                f"Expected float or list of floats."
            )

    # Start with the expanded parameter dict
    result = dict(expanded_params)

    # Collect required parameters and extract concrete values
    required_params = set()
    for gate in gates:
        # Explicitly skip LayerBarrier markers
        if isinstance(gate, LayerBarrier):
            continue

        # All other objects must be Gate instances
        if not isinstance(gate, Gate):
            raise TypeError(
                f"Unexpected object in gates list: {type(gate)!r}; "
                "expected a Gate or LayerBarrier."
            )

        if gate.is_parametric():
            if hasattr(gate, "param_expr") and gate.param_expr is not None:
                for symbol in gate.param_expr.free_symbols:
                    if symbol.name not in result:
                        required_params.add(symbol.name)
            elif gate.param_name:
                if gate.param_name not in result:
                    if hasattr(gate, "param_value") and gate.param_value is not None:
                        result[gate.param_name] = gate.param_value
                    else:
                        required_params.add(gate.param_name)
            elif hasattr(gate, "param_value") and gate.param_value is not None:
                pass

    # Check for missing parameters
    missing = required_params - set(result.keys())
    if missing:
        raise ValueError(f"Missing parameter values for: {missing}")

    return result
