"""Pauli Propagation Executor.

Implements quantum circuit execution using Heisenberg picture (Pauli propagation).
"""

import warnings
from typing import Dict, List, Optional, Union

import numpy as np

from ..base import ExecutorBase, QuantumCircuitBase, QuantumOperatorBase
from .operator_converter import convert_operator
from .pauli_types import PauliSum
from .propagation import batch_propagate, propagate
from .qiskit_converter import bind_parameters, convert_circuit
from .state_overlap import overlap_with_zero
from .truncation import TruncationStats, truncate_by_coeff, truncate_combined

# Try to import Qiskit
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Pauli, SparsePauliOp

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def _unwrap_circuit(circuit):
    """Extract raw Qiskit circuit from wrapper if needed."""
    if isinstance(circuit, QuantumCircuitBase):
        return circuit._qiskit_circuit
    return circuit


def _unwrap_operator(operator):
    """Extract raw Qiskit operator from wrapper if needed."""
    if isinstance(operator, QuantumOperatorBase):
        return operator._qiskit_operator
    return operator


def _unwrap_circuits(circuits):
    """Unwrap a single circuit or list of circuits."""
    if isinstance(circuits, list):
        return [_unwrap_circuit(c) for c in circuits]
    return _unwrap_circuit(circuits)


def _unwrap_operators(operators):
    """Unwrap a single operator or list of operators."""
    if isinstance(operators, list):
        return [_unwrap_operator(o) for o in operators]
    return (
        _unwrap_operators(operators)
        if isinstance(operators, list)
        else _unwrap_operator(operators)
    )


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
            from .pauli_algebra import (pauli_multiply, string_to_term,
                                        term_to_string)

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
    """

    def __init__(
        self,
        shots: Union[int, None] = None,
        seed: Union[int, None] = None,
        log_file: Union[str, None] = None,
        caching: Union[bool, None] = None,
        cache_dir: str = "cache",
        truncate_threshold: Union[float, None] = None,
        max_weight: Union[int, None] = None,
    ):
        super().__init__(shots, seed, log_file, caching, cache_dir)
        self.truncate_threshold = truncate_threshold
        self.max_weight = max_weight

        # Statistics tracking
        self.last_truncation_stats: Optional[TruncationStats] = None

    @property
    def remote(self) -> bool:
        """Return False (Pauli propagation is local execution)."""
        return False

    def expectation_value(
        self,
        circuit,
        operator,
        **parameters,
    ) -> Union[float, np.ndarray]:
        """Calculate expectation value using Pauli propagation.

        Accepts both the library's QuantumCircuit/QuantumOperator wrappers
        and raw Qiskit types.

        When multiple operators are provided, all operators for a given circuit
        are propagated in a single gate-loop pass via batch_propagate(), which
        is approximately N times faster than N individual propagate() calls for
        N operators.

        Args:
            circuit: Quantum circuit(s) to execute
            operator: Observable operator(s) to measure
            **parameters: Parameter values for parametric circuits

        Returns:
            Expectation value (float for single circuit/operator, array for batches)
        """
        # Handle single vs batch execution
        is_single_circuit = not isinstance(circuit, list)
        is_single_operator = not isinstance(operator, list)

        circuits = [circuit] if is_single_circuit else circuit
        operators = [operator] if is_single_operator else operator

        results = []

        if is_single_operator:
            # Fast path: single operator — use existing per-circuit logic
            for circ in circuits:
                raw_circ = _unwrap_circuit(circ)
                raw_op = _unwrap_operator(operators[0])
                exp_val = self._compute_single_expectation(raw_circ, raw_op, parameters)
                results.append(exp_val)
        else:
            # Batch path: multiple operators share one gate-loop pass per circuit.
            # Operator conversion only depends on nqubits (same for all circuits),
            # so convert once before the circuit loop.
            nqubits = _unwrap_circuit(circuits[0]).num_qubits
            observables = [convert_operator(_unwrap_operator(op), nqubits) for op in operators]

            for circ in circuits:
                raw_circ = _unwrap_circuit(circ)

                # Convert circuit and bind parameters once for all operators
                gates = convert_circuit(raw_circ)
                bound_params = bind_parameters(gates, parameters)

                # Single gate-loop pass over all observables
                propagated_list = batch_propagate(
                    gates,
                    observables,
                    bound_params,
                    max_weight=self.max_weight,
                    truncate_threshold=self.truncate_threshold,
                )

                # Post-truncation and expectation value extraction
                for propagated in propagated_list:
                    if self.truncate_threshold is not None or self.max_weight is not None:
                        propagated, stats = truncate_combined(
                            propagated,
                            min_coeff=(
                                self.truncate_threshold if self.truncate_threshold else 1e-15
                            ),
                            max_weight=self.max_weight,
                            inplace=True,
                        )
                        self.last_truncation_stats = stats

                    results.append(overlap_with_zero(propagated))

        # Return format based on input
        if is_single_circuit and is_single_operator:
            return float(results[0].real)  # Single value
        else:
            return np.array([r.real for r in results])

    def _compute_single_expectation(
        self,
        circuit: "QuantumCircuit",
        operator: Union[SparsePauliOp, Pauli, str],
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
        # Convert circuit to internal gates
        gates = convert_circuit(circuit)

        # Bind parameters if needed (bind_parameters expects gates list, not circuit)
        bound_params = bind_parameters(gates, parameters)

        # Convert operator to PauliSum
        nqubits = circuit.num_qubits
        observable = convert_operator(operator, nqubits)

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

    def expectation_value_derivatives(
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

    def sample(self, circuit, **parameters) -> Union[dict, List[dict]]:
        """Sample measurement outcomes from quantum circuit execution.

        Accepts both the library's QuantumCircuit wrapper and raw Qiskit types.

        Args:
            circuit: Quantum circuit(s) to sample from
            **parameters: Parameter values

        Returns:
            Dictionary mapping bitstrings to counts (or list for batch execution)
        """
        # Handle batch execution
        is_single = not isinstance(circuit, list)
        circuits = [circuit] if is_single else circuit

        results = []

        for circ in circuits:
            raw_circ = _unwrap_circuit(circ)

            # Get statevector
            sv = self.statevector(raw_circ, **parameters)

            # Compute probabilities
            probs = np.abs(sv) ** 2
            probs = probs / np.sum(probs)  # Normalize to handle numerical errors

            # Determine number of shots
            shots = self._shots if self._shots is not None else 1024

            # Set random seed if provided
            if self._seed is not None:
                np.random.seed(self._seed)

            # Sample from probability distribution
            nqubits = raw_circ.num_qubits
            indices = np.random.choice(len(sv), size=shots, p=probs)

            # Convert indices to bitstrings and count
            counts = {}
            for idx in indices:
                bitstring = format(idx, f"0{nqubits}b")
                counts[bitstring] = counts.get(bitstring, 0) + 1

            results.append(counts)

        return results[0] if is_single else results

    def statevector(self, circuit, **parameters) -> Union[np.ndarray, List[np.ndarray]]:
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
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for statevector computation")

        from qiskit.quantum_info import Statevector

        # Handle batch execution
        is_single = not isinstance(circuit, list)
        circuits = [circuit] if is_single else circuit

        statevectors = []

        for circ in circuits:
            raw_circ = _unwrap_circuit(circ)
            nqubits = raw_circ.num_qubits

            # Warn for large systems
            if nqubits > 15:
                warnings.warn(
                    f"Computing statevector for {nqubits} qubits requires a "
                    f"{2**nqubits}-dimensional vector. This may be slow and memory-intensive.",
                    RuntimeWarning,
                )

            # Bind parameters if needed
            bound_circ = raw_circ
            if raw_circ.parameters:
                bound_circ = raw_circ.assign_parameters(parameters)

            # Use Qiskit's statevector simulator
            sv = Statevector.from_label("0" * nqubits)
            sv = sv.evolve(bound_circ)

            statevectors.append(sv.data)

        return statevectors[0] if is_single else statevectors

    def get_truncation_stats(self) -> Optional[TruncationStats]:
        """Get statistics from last truncation operation.

        Returns:
            TruncationStats from most recent execution, or None if no truncation
        """
        return self.last_truncation_stats
