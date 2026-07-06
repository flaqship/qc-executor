from __future__ import annotations

from typing import List

import pennylane as qml
import pennylane.numpy as pnp
import pennylane.pauli as pauli
import sympy as sp
from sympy import lambdify

from ..abstraction.abstract_quantum_operator import AbstractQuantumOperator
from ..base import QuantumOperatorBase


def _get_sympy_interface():
    """
    Returns the sympy interface that is used in the parameter conversion.

    Necessary for the correct conversion of sympy coefficient expressions to
    python functions in PennyLane.

    Returns:
        Tuple of sympy printer and sympy modules
    """
    # SymPy printer for pennylane numpy implementation has to be set manually,
    # otherwise math functions are used in lambdify instead of pennylane.numpy functions
    from sympy.printing.numpy import NumPyPrinter as Printer

    user_functions = {}
    printer = Printer(
        {
            "fully_qualified_modules": False,
            "inline": True,
            "allow_unknown_functions": True,
            "user_functions": user_functions,
        }
    )
    # Use Pennylane numpy for sympy lambdify
    modules = pnp

    # The functions down below can be used to switch between different gradient engines
    # as tensorflow, jax and torch. However, this is not supported and implemented yet.

    #     # SymPy printer for pennylane numpy implementation has to be set manually,
    #     # otherwise math functions are used in lambdify instead of pennylane.numpy functions
    #     from sympy.printing.tensorflow import TensorflowPrinter as Printer  # type: ignore

    #     user_functions = {}
    #     printer = Printer(
    #         {
    #             "fully_qualified_modules": False,
    #             "inline": True,
    #             "allow_unknown_functions": True,
    #             "user_functions": user_functions,
    #         }
    #     )  #
    #     modules = tf

    # elif self._gradient_engine == "jax":
    #     from sympy.printing.numpy import JaxPrinter as Printer  # type: ignore

    #     user_functions = {}
    #     printer = Printer(
    #         {
    #             "fully_qualified_modules": False,
    #             "inline": True,
    #             "allow_unknown_functions": True,
    #             "user_functions": user_functions,
    #         }
    #     )  #
    #     modules = jnp
    # elif self._gradient_engine == "torch" or self._gradient_engine == "pytorch":
    #     from sympy.printing.pycode import PythonCodePrinter as Printer  # type: ignore

    #     user_functions = {}
    #     printer = Printer(
    #         {
    #             "fully_qualified_modules": False,
    #             "inline": True,
    #             "allow_unknown_functions": True,
    #             "user_functions": user_functions,
    #         }
    #     )  #
    #     modules = torch

    # else:
    #     # tbd for jax and tensorflow
    #     printer = None
    #     modules = None

    return printer, modules


class PennyLaneOperator:
    """Convert generic quantum operators to PennyLane-native operators.

    Args:
        operator (QuantumOperatorBase | list[QuantumOperatorBase]):
            Operator definition(s) to convert.
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
            self._operator = AbstractQuantumOperator.from_quantum_operator(operator)
            self._num_qubits = self._operator.num_qubits
        elif isinstance(operator, list):
            if all([isinstance(op, QuantumOperatorBase) for op in operator]):
                self._operator = [
                    AbstractQuantumOperator.from_quantum_operator(op) for op in operator
                ]
            else:
                raise ValueError("Unsupported operator type")
            self._num_qubits = self._operator[0].num_qubits
        else:
            raise ValueError("Unsupported operator type")

        self._pennylane_operator_param_functions = []
        self._pennylane_operator_parameters = []
        self._pennylane_words = []
        self._pennylane_operator_parameter_dimensions = {}

        self.build_operator_instructions(self._operator)

    @property
    def parameter_names(self) -> list:
        """List of operator parameter names"""
        return self._pennylane_operator_parameters

    @property
    def parameter_dimensions(self) -> dict:
        """Dictionary with the dimension of each operator parameter"""
        return self._pennylane_operator_parameter_dimensions

    @property
    def hash(self) -> int:
        """Hashable object of the operator for caching.

        Uses ``str(op)`` per operator (not ``str`` of the list, which would
        fall back to ``__repr__`` and drop the coefficients).
        """
        if isinstance(self._operator, list):
            return hash(tuple(str(op) for op in self._operator))
        return hash(str(self._operator))

    def build_operator_instructions(
        self, operator: List[AbstractQuantumOperator] | AbstractQuantumOperator
    ):
        """
        Function to build the instructions for the PennyLane operator directly
        from the abstract operator data (no Qiskit round-trip).

        Pauli label strings are converted into PennyLane Pauli words and the
        (possibly symbolic SymPy) coefficients into autograd-compatible python
        functions.

        Args:
            operator (List[AbstractQuantumOperator] | AbstractQuantumOperator):
                Abstract operator(s) to convert to PennyLane
        """

        self._pennylane_operator_param_functions = []
        self._pennylane_operator_parameters = []
        self._pennylane_words = []
        self._pennylane_operator_parameter_dimensions = {}

        islist = isinstance(operator, list)
        operators = operator if islist else [operator]

        printer, modules = _get_sympy_interface()

        # Register parameter vectors in first-seen order; the dimension counts
        # the distinct parameters per vector (parameters shared by several
        # operators are counted once).
        seen_params: list = []
        for op in operators:
            for param in op.parameters:  # unique, sorted by (vector_name, index)
                if param in seen_params:
                    continue
                seen_params.append(param)
                name = param.vector_name
                if name not in self._pennylane_operator_parameters:
                    self._pennylane_operator_parameters.append(name)
                    self._pennylane_operator_parameter_dimensions[name] = 1
                else:
                    self._pennylane_operator_parameter_dimensions[name] += 1

        # The lambdify argument order must match how ``pennylane_observable``
        # collects its args: one block per parameter vector in registration
        # order, sorted by index inside each block.
        symbol_tuple = tuple(
            sorted(
                seen_params,
                key=lambda p: (
                    self._pennylane_operator_parameters.index(p.vector_name),
                    p.index,
                ),
            )
        )

        for op in operators:
            pennylane_operator_param_function_ = []
            for coeff in op.coeffs:
                if isinstance(coeff, sp.Basic) and coeff.free_symbols:
                    f = lambdify(symbol_tuple, coeff, modules=modules, printer=printer)
                    pennylane_operator_param_function_.append(f)
                else:
                    # complex() handles plain numbers, numpy scalars and
                    # constant sympy expressions alike.
                    value = complex(coeff)
                    if value.imag != 0:
                        raise ValueError(
                            "Imaginary part of operator coefficient is not supported"
                        )
                    pennylane_operator_param_function_.append(float(value.real))
            self._pennylane_operator_param_functions.append(pennylane_operator_param_function_)

        # Convert Pauli strings into PennyLane Pauli words. The abstract labels
        # are little-endian (rightmost char = qubit 0) whereas PennyLane's
        # ``string_to_pauli_word`` assigns left-to-right (leftmost char = wire 0),
        # so the label is reversed to keep operator wires aligned with the circuit.
        for op in operators:
            self._pennylane_words.append(
                [pauli.string_to_pauli_word(p[::-1]) for p in op.paulis]
            )

        if not islist:
            self._pennylane_operator_param_functions = self._pennylane_operator_param_functions[0]
            self._pennylane_words = self._pennylane_words[0]

    def build_pennylane_observable(self):
        """
        Function to build the PennyLane observable from the abstract operator.

        The function returns a callable PennyLane observable that can be called with
        parameters. It is built from the instructions previously generated by
        :meth:`build_operator_instructions`.

        Returns:
            Callable PennyLane circuit
        """

        def pennylane_observable(*args):
            """PennyLane circuit that can be called with parameters"""

            # Collects the args values connected to the observable parameters
            obs_param_list = sum(
                [list(args[i]) for i in range(len(self._pennylane_operator_parameters))],
                [],
            )

            if isinstance(self._operator, list):
                expval_list = []
                for i, obs in enumerate(self._pennylane_words):
                    if len(obs_param_list) > 0:
                        coeff_list = []
                        for coeff in self._pennylane_operator_param_functions[i]:
                            if callable(coeff):
                                evaluated_param = coeff(*obs_param_list)
                                coeff_list.append(evaluated_param)
                            else:
                                coeff_list.append(coeff)
                        expval_list.append(qml.expval(qml.Hamiltonian(coeff_list, obs)))
                    else:
                        # No symbolic parameters: coefficients are plain numbers.
                        # Weight each Pauli word by its numeric coefficient (a bare
                        # sum would drop them) and measure the combined observable.
                        if len(self._pennylane_words[i]) == 0:
                            expval_list.append(0.0)
                        else:
                            expval_list.append(
                                qml.expval(
                                    sum(
                                        coeff * word
                                        for coeff, word in zip(
                                            self._pennylane_operator_param_functions[i],
                                            self._pennylane_words[i],
                                        )
                                    )
                                )
                            )
                return pnp.stack(tuple(expval_list))
            else:
                if len(obs_param_list) > 0:
                    coeff_list = []
                    for coeff in self._pennylane_operator_param_functions:
                        if callable(coeff):
                            evaluated_param = coeff(*obs_param_list)
                            coeff_list.append(evaluated_param)
                        else:
                            coeff_list.append(coeff)
                    return qml.expval(qml.Hamiltonian(coeff_list, self._pennylane_words))
                else:
                    # No symbolic parameters: coefficients are plain numbers.
                    # Weight each Pauli word by its numeric coefficient (a bare
                    # sum would drop them) and measure the combined observable.
                    if len(self._pennylane_words) == 0:
                        return 0.0
                    else:
                        return qml.expval(
                            sum(
                                coeff * word
                                for coeff, word in zip(
                                    self._pennylane_operator_param_functions,
                                    self._pennylane_words,
                                )
                            )
                        )

        return pennylane_observable
