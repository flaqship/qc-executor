"""Pauli Propagation Executor.

Implements quantum circuit execution using Heisenberg picture (Pauli propagation).
"""

import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Union, overload

import numpy as np

from ..base import ExecutorBase, QuantumCircuitBase, QuantumOperatorBase
from .pauli_propagation_circuit import PauliPropagationCircuit
from .pauli_propagation_observable import PauliPropagationObservable
from .pauli_types import PauliSum
from .propagation import propagate
from .qiskit_converter import bind_parameters
from .state_overlap import overlap_with_zero
from .symmetry import NoSymmetry
from .truncation import TruncationStats, truncate_by_coeff, truncate_combined

if TYPE_CHECKING:
    from .symmetry import SymmetryStrategy


def _as_list(obj):
    if isinstance(obj, list):
        return obj
    return [obj]


def _normalize_parameters(parameters: Dict) -> Dict[str, float]:
    """Normalize parameters from list format to indexed format.

    Converts parameters from:
        {"x": [0.1], "p": [0.3], "pop": [0.5, 0.6]}
    To:
        {"x[0]": 0.1, "p[0]": 0.3, "pop[0]": 0.5, "pop[1]": 0.6}

    Also accepts already-normalized parameters with indexed keys.

    Args:
        parameters: Parameter dictionary, values can be floats or lists

    Returns:
        Normalized parameter dictionary with indexed string keys

    Raises:
        TypeError: If parameter value is neither float nor list
        ValueError: If parameter name is invalid
    """
    if not parameters:
        return {}

    normalized = {}

    for name, value in parameters.items():
        if isinstance(value, (list, tuple)):
            # Convert list format: x=[0.1, 0.2] -> {"x[0]": 0.1, "x[1]": 0.2}
            for idx, v in enumerate(value):
                if not isinstance(v, (int, float, np.number)):
                    raise TypeError(
                        f"Parameter '{name}[{idx}]' has invalid type {type(v)}. "
                        f"Expected float or numeric value."
                    )
                normalized[f"{name}[{idx}]"] = float(v)
        elif isinstance(value, (int, float, np.number)):
            # Already a scalar, accept as-is
            normalized[name] = float(value)
        elif isinstance(value, str) and "[" in value:
            # Already indexed format like "x[0]", keep as-is
            normalized[name] = value
        else:
            raise TypeError(
                f"Parameter '{name}' has invalid value type {type(value)}. "
                f"Expected float or list of floats."
            )

    return normalized


def _derivative_param_to_name(param):
    """Convert a Parameter/ParameterVector/string to parameter name string(s)."""
    if isinstance(param, str):
        return param
    if hasattr(param, "name"):
        return param.name
    return str(param)


def _create_projector_observable(bitstring: str, nqubits: int) -> PauliSum:
    """Create projector |b><b| as a PauliSum.

    The projector onto computational basis state |b> is:
    |b><b| = tensor_i [(I + (-1)^{b_i} Z_i)/2]

    This expands to a sum of 2^n Pauli terms.

    Args:
        bitstring: Binary string (e.g., "0101")
        nqubits: Number of qubits

    Returns:
        PauliSum representing |b><b|
    """
    if len(bitstring) != nqubits:
        raise ValueError(f"Bitstring length {len(bitstring)} doesn't match nqubits {nqubits}")

    # Start with scalar 1
    result = PauliSum(nqubits)
    result.add_term("I" * nqubits, 1.0)

    # Iteratively build up the tensor product
    for qubit_idx in range(nqubits):
        bit = int(bitstring[nqubits - 1 - qubit_idx])
        sign = -1 if bit == 1 else 1

        # Multiply result by (I + sign*Z)/2 on this qubit
        # This is equivalent to: result = result * (I + sign*Z_i) / 2
        new_result = PauliSum(nqubits)

        for term, coeff in result:
            # Add the I component (term unchanged)
            new_result.add_term(term, coeff / 2.0)

            # Add the Z component (apply Z to this qubit)
            from .pauli_algebra import term_to_string

            term_str = term_to_string(term, nqubits)
            term_list = list(term_str)

            # Apply Z at qubit_idx
            string_index = nqubits - 1 - qubit_idx

            if term_list[string_index] == "I":
                term_list[string_index] = "Z"
                phase = 1
            elif term_list[string_index] == "X":
                term_list[string_index] = "Y"
                phase = 1j
            elif term_list[string_index] == "Y":
                term_list[string_index] = "X"
                phase = -1j
            elif term_list[string_index] == "Z":
                # Z * Z = I
                term_list[string_index] = "I"
                phase = 1
            else:
                raise ValueError(f"Unknown Pauli: {term_list[string_index]}")

            new_term_str = "".join(term_list)
            new_result.add_term(new_term_str, sign * coeff * phase / 2.0)

        result = new_result

    return result


class PauliPropagationExecutor(ExecutorBase):
    """Executor for quantum circuits using Pauli propagation (Heisenberg picture).

    This executor propagates observables backward through quantum circuits,
    enabling efficient computation of expectation values for sparse observables.

    Args:
        shots: Number of measurement shots (not used for exact simulation)
        seed: Random seed for reproducibility (not used for exact simulation)
        log_file: Path to log file (not implemented)
        caching: Whether to use caching (not implemented yet)
        cache_dir: Directory for caching (not implemented yet)
        truncate_threshold: Coefficient threshold for automatic truncation (None = no truncation)
        max_weight: Maximum Pauli weight for truncation (None = no weight limit)
        symmetry_strategy: Strategy for Pauli symmetry merging (None = no merging)
    """

    _native_circuit_class = PauliPropagationCircuit
    _native_observable_class = PauliPropagationObservable

    def __init__(
        self,
        shots: Union[int, None] = None,
        seed: Union[int, None] = None,
        log_file: Union[str, None] = None,
        log_level: str = "WARNING",
        caching: Union[bool, None] = None,
        cache_dir: str = "cache",
        max_cache_size: Union[int, None] = None,
        truncate_threshold: Union[float, None] = None,
        max_weight: Union[int, None] = None,
        symmetry_strategy: Optional["SymmetryStrategy"] = None,
    ):
        super().__init__(
            shots=shots,
            seed=seed,
            log_file=log_file,
            log_level=log_level,
            caching=caching,
            cache_dir=cache_dir,
            max_cache_size=max_cache_size,
        )
        self.truncate_threshold = truncate_threshold
        self.max_weight = max_weight
        self.symmetry_strategy = (
            symmetry_strategy if symmetry_strategy is not None else NoSymmetry()
        )

        # Statistics tracking
        self.last_truncation_stats: Optional[TruncationStats] = None
        self._random = np.random.default_rng(seed)

    @property
    def shots(self) -> Union[int, None]:
        return self._shots

    @shots.setter
    def shots(self, value: Union[int, None]) -> None:
        self._shots = value

    @property
    def remote(self) -> bool:
        """Return False (Pauli propagation is local execution)."""
        return False

    def _expectation_value(
        self,
        circuit,
        operator,
        **parameters,
    ) -> Union[float, np.ndarray]:
        is_single_circuit = not isinstance(circuit, list)
        is_single_operator = not isinstance(operator, list)

        circuits = _as_list(circuit)
        operators = _as_list(operator)

        for circ in circuits:
            if not isinstance(circ, PauliPropagationCircuit):
                raise TypeError(
                    "PauliPropagationExecutor expects PauliPropagationCircuit inputs only."
                )
        for op in operators:
            if not isinstance(op, PauliPropagationObservable):
                raise TypeError(
                    "PauliPropagationExecutor expects PauliPropagationObservable inputs only."
                )

        results = []
        for circ in circuits:
            for op in operators:
                exp_val = self._compute_single_expectation(circ, op, parameters)
                results.append(exp_val)

        if is_single_circuit and is_single_operator:
            return float(results[0].real)
        return np.array([r.real for r in results])

    def _compute_single_expectation(
        self,
        circuit: PauliPropagationCircuit,
        operator: PauliPropagationObservable,
        parameters: Dict,
    ) -> complex:
        """Compute expectation value for a single circuit and operator.

        Args:
            circuit: Quantum circuit (raw Qiskit type)
            operator: Observable operator
            parameters: Parameter binding dictionary (can be in list or indexed format)

        Returns:
            Complex expectation value
        """
        gates = circuit.gates

        # Normalize parameters from list format (x=[0.1]) to indexed format (x[0]=0.1)
        normalized_params = _normalize_parameters(parameters)

        # Bind parameters if needed (bind_parameters expects gates list, not circuit)
        bound_params = bind_parameters(gates, normalized_params)

        # Assign parameters to observable if it has parametric coefficients
        effective_operator = operator
        if operator.is_parametrized:
            effective_operator = operator.assign_parameters(normalized_params)

        observable = effective_operator.pauli_sum

        # Use observable-level symmetry when explicitly configured.
        # Fall back to executor-level symmetry otherwise.
        if not observable.has_active_symmetry:
            observable.symmetry = self.symmetry_strategy

        # Propagate observable through circuit (Heisenberg picture)
        # Pass truncation params so terms are pruned during propagation
        propagated = propagate(
            gates,
            observable,
            bound_params,
            max_weight=self.max_weight,
            truncate_threshold=self.truncate_threshold,
        )

        # Final truncation pass (cheap cleanup)
        if self.truncate_threshold is not None or self.max_weight is not None:
            propagated, stats = truncate_combined(
                propagated,
                min_coeff=self.truncate_threshold if self.truncate_threshold else 1e-15,
                max_weight=self.max_weight,
                inplace=True,
            )
            self.last_truncation_stats = stats

        # Compute overlap with |0> state
        expectation = overlap_with_zero(propagated)

        return expectation

    def _expectation_value_derivatives(
        self,
        circuit,
        operator,
        *derivative_params,
        **parameter_values,
    ) -> Union[float, np.ndarray, dict]:
        """Calculate derivatives of expectation value.

        Handles circuit parameters (via parameter-shift rule) and observable
        coefficient parameters (via analytical derivatives). When a parameter
        appears in both, contributions are computed separately and summed.

        Accepts both the library's QuantumCircuit/QuantumOperator wrappers
        and raw Qiskit types. Derivative parameters can be strings,
        Parameter objects, or ParameterVector objects.

        Args:
            circuit: Quantum circuit(s)
            operator: Observable operator(s)
            *derivative_params: Parameter(s) to differentiate with respect to
            **parameter_values: Parameter values (can be in list or indexed format)

        Returns:
            If single param requested: numpy array or float of gradients
            If multiple params requested: dict mapping param names to gradient arrays
        """
        import sympy as sp

        from .pauli_algebra import term_to_string

        # Convert derivative params to string names (base parameter names without indices)
        param_names = []
        for p in derivative_params:
            if isinstance(p, (list, tuple)):
                param_names.extend([_derivative_param_to_name(x) for x in p])
            elif hasattr(p, "__iter__") and not isinstance(p, str):
                param_names.extend([_derivative_param_to_name(x) for x in p])
            else:
                param_names.append(_derivative_param_to_name(p))

        # Remove duplicates while preserving order
        seen = set()
        unique_param_names = []
        for name in param_names:
            if name not in seen:
                unique_param_names.append(name)
                seen.add(name)
        param_names = unique_param_names

        is_single_derivative = len(param_names) == 1

        # Build a mapping from base parameter names to their values
        param_mapping = {}
        for key, value in parameter_values.items():
            if isinstance(value, (list, tuple)):
                param_mapping[key] = value
            else:
                param_mapping[key] = [value]

        # Get observable and circuit parameters
        if isinstance(operator, PauliPropagationObservable):
            native_operator = operator
        else:
            native_operator = self._transpile_observable(operator)
        observable_params = set(native_operator.parameters)

        if isinstance(circuit, PauliPropagationCircuit):
            native_circuit = circuit
        else:
            native_circuit = PauliPropagationCircuit.from_quantum_circuit(circuit)
        circuit_params = set(native_circuit.parameters)

        # Compute gradients for each derivative parameter
        result_dict = {}

        for param_name in param_names:
            # Normalize the param name (handle indexed format)
            if "[" in param_name:
                base_name = param_name.split("[")[0].strip()
                # Extract the index from indexed format like "pop[0]"
                index_str = param_name.split("[")[1].rstrip("]")
                specific_index = int(index_str) if index_str.isdigit() else None
            else:
                base_name = param_name
                specific_index = None

            # Find the parameter value - could be under base_name or param_name
            if param_name in param_mapping:
                all_values = param_mapping[param_name]
                if specific_index is not None and isinstance(all_values, (list, tuple)):
                    # Indexed parameter - use only the specific index
                    param_values_list = [all_values[specific_index]]
                else:
                    param_values_list = all_values
            elif base_name in param_mapping:
                all_values = param_mapping[base_name]
                if specific_index is not None and isinstance(all_values, (list, tuple)):
                    # Indexed parameter - use only the specific index
                    param_values_list = [all_values[specific_index]]
                else:
                    param_values_list = all_values
            else:
                raise ValueError(f"Parameter '{param_name}' not found in provided values")

            gradients_for_param = []

            # Classify parameter location
            # For base_name, check if any parameter starts with it (handles indexed params)
            in_observable = param_name in observable_params
            if not in_observable and "[" not in param_name:
                # Base name - check if any observable param starts with this base name
                in_observable = any(
                    obs_param.startswith(base_name + "[") for obs_param in observable_params
                )

            in_circuit = param_name in circuit_params
            if not in_circuit and "[" not in param_name:
                # Base name - check if any circuit param starts with this base name
                in_circuit = any(
                    circ_param.startswith(base_name + "[") for circ_param in circuit_params
                )

            for idx, param_value in enumerate(param_values_list):
                gradient = 0.0

                # For indexed parameters, use the full indexed name
                # For base parameters with multiple values, construct the indexed name
                effective_param_name = param_name
                if "[" not in param_name and "[" in base_name:
                    # param_name was already indexed
                    pass
                elif "[" not in param_name and len(param_values_list) > 1:
                    # Base name with multiple values - use indexed notation
                    effective_param_name = f"{base_name}[{idx}]"

                # === OBSERVABLE CONTRIBUTION ===
                if in_observable:
                    # Compute analytical derivative for observable coefficients
                    # Use the actual symbol from the observable's parameter dict
                    param_symbol = None
                    if effective_param_name in native_operator._parameters:
                        param_symbol = native_operator._parameters[effective_param_name]
                    elif param_name in native_operator._parameters:
                        param_symbol = native_operator._parameters[param_name]
                    elif base_name in native_operator._parameters:
                        param_symbol = native_operator._parameters[base_name]
                    else:
                        # Try to find a matching symbol by name
                        for sym_name, sym in native_operator._parameters.items():
                            if sym_name == effective_param_name or sym_name == param_name:
                                param_symbol = sym
                                break

                    if (
                        param_symbol is not None
                        and hasattr(native_operator, "_parametric_coeffs")
                        and native_operator._parametric_coeffs
                    ):
                        # Observable has parametric coefficients - iterate through them
                        for term, coeff_expr in native_operator._parametric_coeffs.items():
                            # Check if parameter appears in this coefficient
                            if param_symbol in coeff_expr.free_symbols:
                                # Compute derivative: dcoeff/dparam
                                coeff_derivative = sp.diff(coeff_expr, param_symbol)
                                coeff_deriv_value = float(
                                    coeff_derivative.subs(param_symbol, param_value)
                                )

                                # Create single-term observable for this Pauli
                                pauli_str = term_to_string(term, native_operator.num_qubits)
                                single_term_obs = PauliPropagationObservable(
                                    paulis=[pauli_str],
                                    coeffs=[1.0],
                                    num_qubits=native_operator.num_qubits,
                                )

                                # Compute expectation of this Pauli term
                                term_exp = self.expectation_value(
                                    native_circuit, single_term_obs, **parameter_values
                                )

                                # Add contribution: (dcoeff/dparam) * <Pauli>
                                gradient += coeff_deriv_value * term_exp

                # === CIRCUIT CONTRIBUTION ===
                if in_circuit:
                    # Use parameter-shift rule for circuit parameters
                    # Handle indexed vs. non-indexed format differently
                    is_indexed = "[" in param_name

                    params_plus = dict(parameter_values)
                    params_minus = dict(parameter_values)

                    if is_indexed:
                        # For indexed format like "theta[0]", keep as scalar
                        params_plus[param_name] = param_value + np.pi / 2
                        params_minus[param_name] = param_value - np.pi / 2
                    else:
                        # For base format like "theta", handle list vs. scalar
                        if param_name in params_plus and isinstance(
                            params_plus[param_name], (list, tuple)
                        ):
                            params_plus[param_name] = list(params_plus[param_name])
                            params_plus[param_name][idx] = param_value + np.pi / 2
                        else:
                            params_plus[param_name] = [param_value + np.pi / 2]

                        if param_name in params_minus and isinstance(
                            params_minus[param_name], (list, tuple)
                        ):
                            params_minus[param_name] = list(params_minus[param_name])
                            params_minus[param_name][idx] = param_value - np.pi / 2
                        else:
                            params_minus[param_name] = [param_value - np.pi / 2]

                    # Compute expectation values with shifted parameters
                    exp_plus = self.expectation_value(
                        native_circuit, native_operator, **params_plus
                    )
                    exp_minus = self.expectation_value(
                        native_circuit, native_operator, **params_minus
                    )

                    # Apply parameter-shift rule
                    circuit_gradient = (exp_plus - exp_minus) / 2.0
                    gradient += circuit_gradient

                gradients_for_param.append(gradient)

            # Store gradient(s) for this parameter
            if len(gradients_for_param) == 1:
                result_dict[param_name] = np.array([gradients_for_param[0]])
            else:
                result_dict[param_name] = np.array(gradients_for_param)

        # Return format based on number of parameters requested
        if is_single_derivative:
            # Return just the value for single parameter
            single_key = list(result_dict.keys())[0]
            single_value = result_dict[single_key]
            # For indexed format or array format
            if isinstance(single_value, np.ndarray):
                if single_value.shape == (1,):
                    return float(single_value[0])
                return single_value
            return float(single_value)
        else:
            # For multiple parameters, group by base name if they're indexed
            final_dict = {}
            for key, value in result_dict.items():
                if "[" in key:
                    # Extract base name from indexed name
                    import re

                    match = re.match(r"(\w+)", key)
                    if match:
                        base_name = match.group(1)
                        if base_name not in final_dict:
                            final_dict[base_name] = {}
                        # Extract index
                        idx_match = re.search(r"\[(\d+)\]", key)
                        idx = int(idx_match.group(1)) if idx_match else 0
                        final_dict[base_name][idx] = value
                else:
                    # Base name format
                    final_dict[key] = value

            # Convert indexed dicts to arrays
            for key in final_dict:
                if isinstance(final_dict[key], dict):
                    indices = sorted(final_dict[key].keys())
                    final_dict[key] = np.array([final_dict[key][i] for i in indices])

            # Return dictionary for multiple parameters
            return final_dict

    @staticmethod
    def _apply_single_qubit_gate(
        state: np.ndarray, matrix: np.ndarray, qubit: int, nqubits: int
    ) -> np.ndarray:
        tensor = state.reshape([2] * nqubits)
        perm = [qubit] + [idx for idx in range(nqubits) if idx != qubit]
        inv_perm = np.argsort(perm)
        transformed = np.transpose(tensor, perm).reshape(2, -1)
        transformed = matrix @ transformed
        transformed = transformed.reshape([2] + [2] * (nqubits - 1))
        transformed = np.transpose(transformed, inv_perm)
        return transformed.reshape(-1)

    @staticmethod
    def _apply_two_qubit_gate(
        state: np.ndarray, matrix: np.ndarray, qubit_a: int, qubit_b: int, nqubits: int
    ) -> np.ndarray:
        if qubit_a == qubit_b:
            raise ValueError("Two-qubit gate requires distinct qubits.")
        tensor = state.reshape([2] * nqubits)
        perm = [qubit_a, qubit_b] + [
            idx for idx in range(nqubits) if idx not in (qubit_a, qubit_b)
        ]
        inv_perm = np.argsort(perm)
        transformed = np.transpose(tensor, perm).reshape(4, -1)
        transformed = matrix @ transformed
        transformed = transformed.reshape([2, 2] + [2] * (nqubits - 2))
        transformed = np.transpose(transformed, inv_perm)
        return transformed.reshape(-1)

    @staticmethod
    def _resolve_angle(gate, parameters: Dict[str, float]) -> float:
        """Resolve the angle value for a rotation gate.

        Args:
            gate: PauliRotation gate
            parameters: Dict mapping parameter names to values

        Returns:
            The resolved angle as a float
        """
        if gate.param_expr is not None:
            # Symbolic expression - substitute parameter values
            import sympy as sp

            subs_dict = {}
            for symbol in gate.param_expr.free_symbols:
                param_name = symbol.name
                if param_name not in parameters:
                    raise ValueError(f"Missing parameter value for '{param_name}'")
                subs_dict[symbol] = parameters[param_name]

            result = gate.param_expr.subs(subs_dict)
            if not result.is_number:
                raise ValueError(f"Expression {gate.param_expr} could not be fully evaluated")
            return float(result)

        if gate.param_value is None:
            raise ValueError("Parametric gate has neither param_expr nor param_value.")
        return float(gate.param_value)

    def _simulate_statevector(
        self, circuit: PauliPropagationCircuit, parameters: Dict[str, float]
    ) -> np.ndarray:
        from .gates import CliffordGate, LayerBarrier, PauliRotation

        # Normalize parameters from list format to indexed format
        normalized_params = _normalize_parameters(parameters)

        nqubits = circuit.num_qubits
        state = np.zeros(2**nqubits, dtype=complex)
        state[0] = 1.0

        x = np.array([[0, 1], [1, 0]], dtype=complex)
        y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        z = np.array([[1, 0], [0, -1]], dtype=complex)
        h = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
        s = np.array([[1, 0], [0, 1j]], dtype=complex)
        t = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

        cnot = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=complex,
        )
        cz = np.diag([1, 1, 1, -1]).astype(complex)
        swap = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=complex,
        )

        single_qubit_map = {"X": x, "Y": y, "Z": z, "H": h, "S": s, "T": t}

        for gate in circuit.gates:
            if isinstance(gate, LayerBarrier):
                continue

            if isinstance(gate, CliffordGate):
                if gate.gate_type in single_qubit_map:
                    state = self._apply_single_qubit_gate(
                        state, single_qubit_map[gate.gate_type], gate.qubits[0], nqubits
                    )
                elif gate.gate_type in ["CNOT", "CX"]:
                    state = self._apply_two_qubit_gate(
                        state, cnot, gate.qubits[0], gate.qubits[1], nqubits
                    )
                elif gate.gate_type == "CZ":
                    state = self._apply_two_qubit_gate(
                        state, cz, gate.qubits[0], gate.qubits[1], nqubits
                    )
                elif gate.gate_type == "SWAP":
                    state = self._apply_two_qubit_gate(
                        state, swap, gate.qubits[0], gate.qubits[1], nqubits
                    )
                else:
                    raise ValueError(f"Unsupported Clifford gate type: {gate.gate_type}")
                continue

            if isinstance(gate, PauliRotation):
                theta = self._resolve_angle(gate, normalized_params)
                if len(gate.symbols) == 1:
                    pauli = single_qubit_map[gate.symbols[0]]
                    rotation = np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta / 2) * pauli
                    state = self._apply_single_qubit_gate(state, rotation, gate.qubits[0], nqubits)
                elif len(gate.symbols) == 2:
                    pauli_a = single_qubit_map[gate.symbols[0]]
                    pauli_b = single_qubit_map[gate.symbols[1]]
                    generator = np.kron(pauli_a, pauli_b)
                    rotation = np.cos(theta / 2) * np.eye(4) - 1j * np.sin(theta / 2) * generator
                    state = self._apply_two_qubit_gate(
                        state, rotation, gate.qubits[0], gate.qubits[1], nqubits
                    )
                else:
                    raise ValueError("Only 1- and 2-qubit Pauli rotations are supported.")
                continue

            raise TypeError(f"Unsupported gate object in circuit: {type(gate)!r}")

        return state

    def _sample(self, circuit, **parameters) -> Union[dict, List[dict]]:
        """Sample measurement outcomes from quantum circuit execution.

        Accepts both the library's QuantumCircuit wrapper and raw Qiskit types.

        Args:
            circuit: Quantum circuit(s) to sample from
            **parameters: Parameter values

        Returns:
            Dictionary mapping bitstrings to counts (or list for batch execution)
        """
        is_single = not isinstance(circuit, list)
        circuits = _as_list(circuit)

        results = []

        for circ in circuits:
            if not isinstance(circ, PauliPropagationCircuit):
                raise TypeError(
                    "PauliPropagationExecutor expects PauliPropagationCircuit inputs only."
                )

            sv = self._simulate_statevector(circ, parameters)

            # Compute probabilities
            probs = np.abs(sv) ** 2
            probs = probs / np.sum(probs)  # Normalize to handle numerical errors

            # Determine number of shots
            shots = self._shots if self._shots is not None else 1024

            # Sample from probability distribution
            nqubits = circ.num_qubits
            indices = self._random.choice(len(sv), size=shots, p=probs)

            # Convert indices to bitstrings and count
            counts = {}
            for idx in indices:
                bitstring = format(idx, f"0{nqubits}b")
                counts[bitstring] = counts.get(bitstring, 0) + 1

            results.append(counts)

        return results[0] if is_single else results

    def _statevector(self, circuit, **parameters) -> Union[np.ndarray, List[np.ndarray]]:
        """Compute statevector from circuit execution.

        Accepts both the library's QuantumCircuit wrapper and raw Qiskit types.

        Note: This implementation uses Qiskit's statevector simulator as a fallback
        since extracting full complex amplitudes from Pauli propagation alone
        requires additional phase information that is non-trivial to obtain.

        WARNING: Exponentially expensive for large qubit counts.

        Args:
            circuit: Quantum circuit(s)
            **parameters: Parameter values

        Returns:
            Statevector as complex numpy array (or list of arrays for batch)
        """
        is_single = not isinstance(circuit, list)
        circuits = _as_list(circuit)

        statevectors = []

        for circ in circuits:
            if not isinstance(circ, PauliPropagationCircuit):
                raise TypeError(
                    "PauliPropagationExecutor expects PauliPropagationCircuit inputs only."
                )

            nqubits = circ.num_qubits

            # Warn for large systems
            if nqubits > 15:
                warnings.warn(
                    f"Computing statevector for {nqubits} qubits requires a "
                    f"{2**nqubits}-dimensional vector. This may be slow and memory-intensive.",
                    RuntimeWarning,
                )

            statevectors.append(self._simulate_statevector(circ, parameters))

        return statevectors[0] if is_single else statevectors

    def _transpile_circuit(self, circuit: QuantumCircuitBase) -> PauliPropagationCircuit:
        """Transpile a circuit to PauliPropagationCircuit format.

        Accepts both native PauliPropagationCircuit and generic QuantumCircuit types.
        If a generic QuantumCircuit is provided, it is converted from its internal
        Qiskit representation.

        Args:
            circuit (QuantumCircuitBase): Circuit to transpile (native or generic)

        Returns:
            PauliPropagationCircuit: Transpiled circuit in native format

        Raises:
            TypeError: If circuit type is not supported
        """
        return PauliPropagationCircuit.from_quantum_circuit(circuit)

    @overload
    def transpile_observable(
        self,
        operator: QuantumOperatorBase,
    ) -> PauliPropagationObservable: ...

    @overload
    def transpile_observable(
        self,
        operator: QuantumOperatorBase,
        symmetry_strategy: "SymmetryStrategy",
    ) -> PauliPropagationObservable: ...

    @overload
    def transpile_observable(
        self,
        operator: List[QuantumOperatorBase],
    ) -> List[PauliPropagationObservable]: ...

    @overload
    def transpile_observable(
        self,
        operator: List[QuantumOperatorBase],
        symmetry_strategy: "SymmetryStrategy",
    ) -> List[PauliPropagationObservable]: ...

    def transpile_observable(
        self,
        operator: Union[QuantumOperatorBase, List[QuantumOperatorBase]],
        symmetry_strategy: Optional["SymmetryStrategy"] = None,
    ) -> Union[PauliPropagationObservable, List[PauliPropagationObservable]]:
        """
        Transpile the operator for execution on Pauli Propagation backend.

        Accepts both native PauliPropagationObservable and generic QuantumOperator types.
        When a list of operators is provided, each operator is transpiled and cached individually.

        Args:
            operator (Union[QuantumOperatorBase, List[QuantumOperatorBase]]): The
                quantum operator or a list of operators to transpile.
            symmetry_strategy (Optional[SymmetryStrategy]): Strategy for symmetry handling.
                If provided, takes precedence over executor-level default.

        Returns:
            Union[PauliPropagationObservable, List[PauliPropagationObservable]]: The
                transpiled operator(s) in native format.
        """
        self._logger.info("Transpiling operator")
        if isinstance(operator, list):
            return [self._transpile_observable_cached(op, symmetry_strategy) for op in operator]
        return self._transpile_observable_cached(operator, symmetry_strategy)

    def _transpile_observable_cached(
        self,
        operator: QuantumOperatorBase,
        symmetry_strategy: Optional["SymmetryStrategy"] = None,
    ) -> PauliPropagationObservable:
        if self._result_cache is not None:
            key = self._make_result_key("transpile_observable", operator, symmetry_strategy)
            if key in self._result_cache:
                self._logger.debug("Result cache hit for transpile_observable")
                return self._result_cache[key]
            result = self._transpile_observable_with_symmetry(operator, symmetry_strategy)
            self._result_cache[key] = result
            return result
        return self._transpile_observable_with_symmetry(operator, symmetry_strategy)

    def _transpile_observable(self, operator: QuantumOperatorBase) -> PauliPropagationObservable:
        return self._transpile_observable_with_symmetry(operator)

    def _transpile_observable_with_symmetry(
        self,
        operator: QuantumOperatorBase,
        symmetry_strategy: Optional["SymmetryStrategy"] = None,
    ) -> PauliPropagationObservable:
        """Transpile an operator to PauliPropagationObservable format.

        Accepts both native PauliPropagationObservable and generic QuantumOperator types.
        If symmetry_strategy is provided, it takes precedence and is assigned to the
        observable. Otherwise, falls back to the executor's default symmetry_strategy.

        Args:
            operator (QuantumOperatorBase): Operator to transpile (native or generic)
            symmetry_strategy (Optional[object]): Symmetry strategy to assign. If None,
                uses executor-level default (self.symmetry_strategy)

        Returns:
            PauliPropagationObservable: Transpiled observable in native format

        Raises:
            TypeError: If operator type is not supported
        """
        effective_symmetry = (
            symmetry_strategy if symmetry_strategy is not None else self.symmetry_strategy
        )
        result = PauliPropagationObservable.from_quantum_operator(operator, effective_symmetry)
        if not result.has_active_symmetry:
            result.symmetry = effective_symmetry
        return result

    def get_truncation_stats(self) -> Optional[TruncationStats]:
        """Get statistics from last truncation operation.

        Returns:
            TruncationStats from most recent execution, or None if no truncation
        """
        return self.last_truncation_stats
