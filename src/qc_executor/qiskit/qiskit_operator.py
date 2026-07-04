from __future__ import annotations

from collections import OrderedDict
from typing import List

import numpy as np
import sympy as sp
from qiskit.circuit import ParameterVector as QiskitParameterVector
from qiskit.quantum_info import SparsePauliOp

from ..abstraction.abstract_parameter import Parameter, free_parameters


def _build_qiskit_parameter_map(coeffs) -> dict:
    """Map each native :class:`Parameter` used in ``coeffs`` to a Qiskit element.

    Groups parameters by vector name and builds one Qiskit ``ParameterVector``
    per group sized to the highest index seen. Returns ``{native Parameter:
    qiskit element}``. Mirrors the gate-angle handling in ``qiskit_circuit.py``
    (kept separate so the operator bridge does not depend on the circuit module).
    """
    max_index: dict = {}
    for coeff in coeffs:
        for param in free_parameters(coeff):
            name = param.vector_name
            max_index[name] = max(max_index.get(name, -1), param.index)

    symbol_to_qiskit: dict = {}
    for name, top in sorted(max_index.items()):
        qiskit_vec = QiskitParameterVector(name, top + 1)
        for i in range(top + 1):
            symbol_to_qiskit[Parameter(name, i)] = qiskit_vec[i]
    return symbol_to_qiskit


def _coeff_to_qiskit(coeff, symbol_to_qiskit: dict):
    """Convert one coefficient (numeric or symbolic) to a Qiskit-usable value.

    A symbolic SymPy coefficient is replayed onto Qiskit parameter objects via
    ``lambdify`` (Qiskit ``ParameterExpression`` supports +, -, *, /, **), so
    parametrized coefficients survive the conversion.
    """
    if not isinstance(coeff, sp.Basic):
        return coeff  # plain number (float / int / complex)

    symbols = free_parameters(coeff)
    if not symbols:  # constant expression
        value = complex(coeff)
        return value.real if value.imag == 0 else value

    func = sp.lambdify(symbols, coeff, modules="math")
    return func(*[symbol_to_qiskit[s] for s in symbols])


def to_qiskit_operator(operator):
    """Return a Qiskit ``SparsePauliOp`` for any supported operator type.

    Operator counterpart to ``to_qiskit_circuit``:

    * executor wrappers expose ``_qiskit_operator`` and are returned directly;
    * a raw ``SparsePauliOp`` is returned unchanged;
    * an ``AbstractQuantumOperator`` carries plain ``paulis``/``coeffs`` (coeffs
      may be SymPy parameter expressions) and is converted here. Pauli labels
      share Qiskit's little-endian convention, so the strings pass through 1:1.
    """
    inner = getattr(operator, "_qiskit_operator", None)
    if inner is not None:
        return inner
    if isinstance(operator, SparsePauliOp):
        return operator
    if hasattr(operator, "paulis") and hasattr(operator, "coeffs"):
        symbol_to_qiskit = _build_qiskit_parameter_map(operator.coeffs)
        coeffs = [_coeff_to_qiskit(c, symbol_to_qiskit) for c in operator.coeffs]
        return SparsePauliOp(list(operator.paulis), coeffs=coeffs)
    return operator


class QiskitOperator:
    """
    Wrapper for Qiskit operators (SparsePauliOp) used by QiskitExecutor.
    Handles parameter management for parametrized operators.
    """

    def __init__(self, operator):
        """
        Initialize QiskitOperator wrapper.

        Args:
            operator: QuantumOperator object (from qc_executor.quantum_operator)
        """
        # Build/extract the internal qiskit operator. ``to_qiskit_operator``
        # converts an AbstractQuantumOperator's (paulis, coeffs) into a real
        # SparsePauliOp and passes existing Qiskit operators through unchanged.
        self._qiskit_operator = to_qiskit_operator(operator)

        self._num_qubits = self._qiskit_operator.num_qubits

        # Group parameters by ParameterVector name
        self._parameter_dimensions = OrderedDict()
        self._free_parameters = set()

        for p in self._qiskit_operator.parameters:
            self._free_parameters.add(p)
            name = p.vector.name
            self._parameter_dimensions[name] = self._parameter_dimensions.get(name, 0) + 1

    @classmethod
    def from_quantum_operator(cls, operator):
        """Create a native Qiskit operator wrapper from a generic operator."""
        return cls(operator)

    @property
    def num_qubits(self) -> int:
        """Number of qubits the operator acts on."""
        return self._num_qubits

    @property
    def hash(self) -> int:
        """Hash of the operator for caching."""
        return hash(str(self._qiskit_operator))

    @property
    def parameter_names(self) -> List[str]:
        """List of parameter vector names."""
        return list(self._parameter_dimensions.keys())

    @property
    def parameter_dimensions(self) -> dict:
        """Dictionary mapping parameter names to their dimensions."""
        return dict(self._parameter_dimensions)

    @property
    def qiskit_operator(self):
        """Access to the underlying Qiskit SparsePauliOp."""
        return self._qiskit_operator

    @property
    def free_parameters(self) -> set:
        """Set of all free parameters in the operator."""
        return self._free_parameters

    def bind_parameters(self, parameter_values: dict):
        """
        Bind parameter values to the operator.

        Args:
            parameter_values: Dictionary mapping parameter names to values

        Returns:
            Bound Qiskit SparsePauliOp
        """
        # Convert parameter_values dict (with vector names) to Qiskit format
        params_dict = {}

        for param_name, values in parameter_values.items():
            # Ensure values is a list
            if not isinstance(values, (list, np.ndarray)):
                values = [values]

            # Match parameters from operator with provided values
            matching_params = [p for p in self._free_parameters if p.vector.name == param_name]

            # Sort by index to ensure correct ordering
            matching_params = sorted(matching_params, key=lambda x: x.index)

            for i, param in enumerate(matching_params):
                if i < len(values):
                    params_dict[param] = values[i]

        # Bind parameters to operator
        if params_dict:
            return self._qiskit_operator.assign_parameters(params_dict)
        else:
            return self._qiskit_operator

    def copy(self):
        """Return a copy of the operator wrapper."""
        return QiskitOperator(self._qiskit_operator.copy())

    def __str__(self):
        return str(self._qiskit_operator)

    def __repr__(self):
        return f"QiskitOperator({self.num_qubits} qubits, {len(self._free_parameters)} parameters)"
