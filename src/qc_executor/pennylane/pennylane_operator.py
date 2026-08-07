"""PennyLane operator conversion from the shared sparse Pauli representation.

This wrapper used to hold a Qiskit ``SparsePauliOp``, read its labels and turn
its ``ParameterExpression`` coefficients into callables.  It now compiles the
framework-independent representation directly, which removes Qiskit from this
backend.

Labels need no reversal any more.  Qiskit renders qubit 0 rightmost, so the old
code reversed every label before handing it to ``string_to_pauli_word``; both
sides now read qubit 0 leftmost.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple, cast

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import sympy as sp
from pennylane import pauli
from sympy import lambdify

from ..base import QuantumOperatorBase
from ..base.operator_ir import PauliIR
from ..parameters import Parameter, sort_parameters
from ._sympy_interface import _get_sympy_interface


def _resolve_coefficient(coeff, symbol_tuple, printer, modules):
    """Convert one coefficient to a float or a PennyLane-compatible callable.

    Args:
        coeff: The coefficient, a number or a SymPy expression.
        symbol_tuple: Symbols the callable takes, in argument order.
        printer: SymPy printer targeting PennyLane's autograd numpy.
        modules: Module list for :func:`sympy.lambdify`.

    Returns:
        A float for a constant coefficient, else a callable of the symbols.

    Raises:
        ValueError: If a constant coefficient has an imaginary part.
    """
    if isinstance(coeff, sp.Basic) and coeff.free_symbols:
        return lambdify(symbol_tuple, coeff, modules=modules, printer=printer)
    value = complex(coeff)
    if value.imag != 0:
        raise ValueError("Imaginary part of operator coefficient is not supported")
    return float(value.real)


class PennyLaneOperator:
    """Convert generic quantum operators to PennyLane-native operators.

    Args:
        operator: The operator, or a list of operators to evaluate together.

    Raises:
        ValueError: If the argument is not an operator or a list of operators.
    """

    @classmethod
    def from_quantum_operator(cls, operator: QuantumOperatorBase) -> "PennyLaneOperator":
        """Create a PennyLane native operator from a generic operator."""
        return cls(operator)

    def __init__(
        self,
        operator: QuantumOperatorBase | List[QuantumOperatorBase],
    ) -> None:

        if isinstance(operator, QuantumOperatorBase):
            self._islist = False
            self._irs: List[PauliIR] = [operator.ir]
        elif isinstance(operator, list):
            if not all(isinstance(op, QuantumOperatorBase) for op in operator):
                raise ValueError("Unsupported operator type")
            if not operator:
                raise ValueError("Unsupported operator type")
            self._islist = True
            self._irs = [cast(QuantumOperatorBase, op).ir for op in operator]
        else:
            raise ValueError("Unsupported operator type")

        self._num_qubits = self._irs[0].num_qubits

        self._pennylane_operator_param_functions: List[Any] = []
        self._pennylane_operator_parameters: List[str] = []
        self._pennylane_words: List[Any] = []
        self._pennylane_operator_parameter_dimensions: Dict[str, int] = {}

        self.build_operator_instructions(self._irs)

    @property
    def num_qubits(self) -> int:
        """Number of qubits the operator acts on."""
        return self._num_qubits

    @property
    def parameter_names(self) -> list:
        """List of operator parameter names"""
        return self._pennylane_operator_parameters

    @property
    def parameter_dimensions(self) -> dict:
        """Dictionary with the dimension of each operator parameter"""
        return self._pennylane_operator_parameter_dimensions

    @property
    def hash(self) -> bytes:
        """Hashable object of the operator for caching"""
        return b"".join(ir.fingerprint() for ir in self._irs)

    def build_operator_instructions(self, operators: Sequence[PauliIR]) -> None:
        """Compile the representation into PennyLane Pauli words and coefficient callables.

        Args:
            operators: One representation per observable to evaluate.
        """
        self._pennylane_operator_param_functions = []
        self._pennylane_operator_parameters = []
        self._pennylane_words = []
        self._pennylane_operator_parameter_dimensions = {}

        for ir in operators:
            for parameter in sort_parameters(ir.free_parameters):
                name = parameter.vector_name
                if name not in self._pennylane_operator_parameters:
                    self._pennylane_operator_parameters.append(name)
                    self._pennylane_operator_parameter_dimensions[name] = 1
                else:
                    self._pennylane_operator_parameter_dimensions[name] += 1

        printer, modules = _get_sympy_interface()

        # One symbol slot per parameter occurrence, in the same order the
        # dimensions above were counted, so the executor's flattened argument
        # list lines up with what the lambdified callables expect.
        symbol_tuple: Tuple[Parameter, ...] = tuple(
            parameter for ir in operators for parameter in sort_parameters(ir.free_parameters)
        )

        for ir in operators:
            self._pennylane_operator_param_functions.append(
                [
                    _resolve_coefficient(coeff, symbol_tuple, printer, modules)
                    for coeff in ir.coeffs
                ]
            )
            # Qubit 0 is leftmost on both sides, so the label passes through.
            self._pennylane_words.append(
                [pauli.string_to_pauli_word(label) for label in ir.to_labels()]
            )

        if not self._islist:
            self._pennylane_operator_param_functions = self._pennylane_operator_param_functions[0]
            self._pennylane_words = self._pennylane_words[0]

    def _observable(self, coeff_functions, words, obs_param_list):
        """Build one weighted observable from its coefficients and Pauli words.

        Coefficients are weights, not decoration: an unweighted ``qml.sum`` here
        silently returned the value for all-ones coefficients.

        Args:
            coeff_functions: Floats or callables, one per word.
            words: The PennyLane Pauli words.
            obs_param_list: Flattened parameter values for the callables.

        Returns:
            The measurement process, or ``0.0`` for an empty operator.
        """
        if len(words) == 0:
            return 0.0
        coeffs = [
            coeff(*obs_param_list) if callable(coeff) else coeff for coeff in coeff_functions
        ]
        return qml.expval(qml.Hamiltonian(coeffs, words))

    def build_pennylane_observable(self) -> Callable:
        """Return a callable that measures this observable inside a QNode.

        Returns:
            A callable taking one sequence of values per operator parameter.
        """

        def pennylane_observable(*args):
            """PennyLane observable that can be called with parameters"""

            # Collects the args values connected to the observable parameters
            obs_param_list = sum(
                [list(args[i]) for i in range(len(self._pennylane_operator_parameters))],
                [],
            )

            if self._islist:
                return pnp.stack(
                    tuple(
                        self._observable(
                            self._pennylane_operator_param_functions[i], words, obs_param_list
                        )
                        for i, words in enumerate(self._pennylane_words)
                    )
                )
            return self._observable(
                self._pennylane_operator_param_functions, self._pennylane_words, obs_param_list
            )

        return pennylane_observable
