"""Pauli Propagation Executor.

Implements quantum circuit execution using Heisenberg picture (Pauli propagation).
"""

import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

from ..base import ExecutorBase
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
        bit = int(bitstring[qubit_idx])
        sign = -1 if bit == 1 else 1

        # Multiply result by (I + sign*Z)/2 on this qubit
        # This is equivalent to: result = result * (I + sign*Z_i) / 2
        new_result = PauliSum(nqubits)

        for term, coeff in result:
            # Add the I component (term unchanged)
            new_result.add_term(term, coeff / 2.0)

            # Add the Z component (apply Z to this qubit)
            z_string = list("I" * nqubits)
            # Build Pauli string with Z at qubit_idx
            from .pauli_algebra import pauli_multiply, string_to_term, term_to_string

            term_str = term_to_string(term, nqubits)
            term_list = list(term_str)

            # Apply Z at qubit_idx
            if term_list[qubit_idx] == "I":
                term_list[qubit_idx] = "Z"
                phase = 1
            elif term_list[qubit_idx] == "X":
                term_list[qubit_idx] = "Y"
                phase = 1j
            elif term_list[qubit_idx] == "Y":
                term_list[qubit_idx] = "X"
                phase = -1j
            elif term_list[qubit_idx] == "Z":
                # Z * Z = I
                term_list[qubit_idx] = "I"
                phase = 1
            else:
                raise ValueError(f"Unknown Pauli: {term_list[qubit_idx]}")

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
            parameters: Parameter binding dictionary

        Returns:
            Complex expectation value
        """
        gates = circuit.gates

        # Bind parameters if needed (bind_parameters expects gates list, not circuit)
        bound_params = bind_parameters(gates, parameters)

        observable = operator.pauli_sum

        # Attach symmetry strategy to observable for automatic merging
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
    ) -> Union[float, np.ndarray]:
        """Calculate derivatives of expectation value using parameter-shift rule.

        Accepts both the library's QuantumCircuit/QuantumOperator wrappers
        and raw Qiskit types. Derivative parameters can be strings,
        Parameter objects, or ParameterVector objects.

        Args:
            circuit: Quantum circuit(s)
            operator: Observable operator(s)
            *derivative_params: Parameter(s) to differentiate with respect to
            **parameter_values: Parameter values

        Returns:
            Derivative of expectation value (float for single param, array for multiple)
        """
        # Convert derivative params to string names
        param_names = []
        for p in derivative_params:
            if isinstance(p, (list, tuple)):
                param_names.extend([_derivative_param_to_name(x) for x in p])
            elif hasattr(p, "__iter__") and not isinstance(p, str):
                param_names.extend([_derivative_param_to_name(x) for x in p])
            else:
                param_names.append(_derivative_param_to_name(p))

        is_single_derivative = len(param_names) == 1

        # Compute gradients for each derivative parameter
        gradients = []

        for param_name in param_names:
            # Get current parameter value (default to 0 if not provided)
            param_value = parameter_values.get(param_name, 0.0)

            # Create shifted parameter dictionaries
            params_plus = parameter_values.copy()
            params_plus[param_name] = param_value + np.pi / 2

            params_minus = parameter_values.copy()
            params_minus[param_name] = param_value - np.pi / 2

            # Compute expectation values with shifted parameters
            exp_plus = self.expectation_value(circuit, operator, **params_plus)
            exp_minus = self.expectation_value(circuit, operator, **params_minus)

            # Apply parameter-shift rule
            gradient = (exp_plus - exp_minus) / 2.0

            gradients.append(gradient)

        # Return scalar for single derivative, array for multiple
        if is_single_derivative:
            return (
                float(gradients[0])
                if isinstance(gradients[0], (int, float, np.number))
                else gradients[0]
            )
        else:
            return np.array(gradients)

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
        if gate.param_name is not None:
            if gate.param_name not in parameters:
                raise ValueError(f"Missing parameter value for '{gate.param_name}'")
            return float(parameters[gate.param_name])
        if gate.param_value is None:
            raise ValueError("Parametric gate has neither param_name nor param_value.")
        return float(gate.param_value)

    def _simulate_statevector(
        self, circuit: PauliPropagationCircuit, parameters: Dict[str, float]
    ) -> np.ndarray:
        from .gates import CliffordGate, LayerBarrier, PauliRotation

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
                theta = self._resolve_angle(gate, parameters)
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

    def _transpile_circuit(self, circuit: PauliPropagationCircuit) -> PauliPropagationCircuit:
        if not isinstance(circuit, PauliPropagationCircuit):
            raise TypeError(
                "PauliPropagationExecutor expects PauliPropagationCircuit inputs only."
            )
        return circuit

    def get_truncation_stats(self) -> Optional[TruncationStats]:
        """Get statistics from last truncation operation.

        Returns:
            TruncationStats from most recent execution, or None if no truncation
        """
        return self.last_truncation_stats
