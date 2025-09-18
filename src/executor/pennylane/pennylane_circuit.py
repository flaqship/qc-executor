import numpy as np
from typing import Union, List
from sympy import lambdify, sympify

from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.circuit import Clbit, ParameterExpression
from qiskit.quantum_info import SparsePauliOp

from qiskit.compiler import transpile

import pennylane as qml
import pennylane.numpy as pnp

from .pennylane_gates import pennylane_target, qiskit_pennylane_gate_dict

from ..utils.decompose_to_std import decompose_to_std
from ..quantum_circuit import QuantumCircuit


def _get_sympy_interface():
    """
    Returns the sympy interface that is used in the parameter conversion.

    Necessary for the correct conversion of sympy expressions in Qiskit to
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


class PennyLaneCircuit:

    def __init__(
        self,
        circuit: QuantumCircuit,
    ) -> None:

        # Transpile circuit to supported basis gates and expand blocks automatically
        self._qiskit_circuit = transpile(
            decompose_to_std(circuit._qiskit_circuit),
            target=pennylane_target,
            optimization_level=0,
        )

        self._num_qubits = self._qiskit_circuit.num_qubits
        self._num_clbits = self._qiskit_circuit.num_clbits

        self._pennylane_gates = []
        self._pennylane_gates_param_function = []
        self._pennylane_gates_wires = []
        self._pennylane_conditions = []
        self._pennylane_gates_parameters = []
        self._pennylane_gates_parameters_dimensions = {}

        # Build circuit instructions for the pennylane circuit from the qiskit circuit
        self._build_circuit_instructions(self._qiskit_circuit)

        # self._pennylane_circuit = self.build_pennylane_circuit()

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the circuit"""
        return self._num_qubits

    @property
    def pennylane_circuit(self) -> callable:
        """PennyLane circuit that can be called with parameters"""
        return self._pennylane_circuit

    @property
    def parameter_names(self) -> list:
        """List of circuit parameter names"""
        return self._pennylane_gates_parameters

    @property
    def parameter_dimensions(self) -> dict:
        """Dictionary with the dimension of each circuit parameter"""
        return self._pennylane_gates_parameters_dimensions

    @property
    def hash(self) -> str:
        """Hashable object of the circuit and observable for caching"""
        return hash(str(self._qiskit_circuit))

    def get_pennylane_circuit(self) -> callable:
        """Builds and returns the PennyLane circuit as callable function"""
        self._pennylane_circuit = self.build_pennylane_circuit()
        return self._pennylane_circuit

    def __call__(self, *args, **kwargs):
        return self._pennylane_circuit(*args, **kwargs)

    def _build_circuit_instructions(self, circuit: QuantumCircuit) -> tuple:
        """
        Function to build the instructions for the PennyLane circuit from the Qiskit circuit.

        This functions converts the Qiskit gates and parameter expressions to PennyLane compatible
        gates and functions.

        Args:
            circuit (QuantumCircuit): Qiskit circuit to convert to PennyLane

        Returns:
            Tuple with lists of PennyLane gates, PennyLane gate parameter functions,
            PennyLane gate wires, PennyLane gate parameters and PennyLane gate parameter dimensions
        """

        self._pennylane_gates = []
        self._pennylane_gates_param_function = []
        self._pennylane_gates_wires = []
        self._pennylane_conditions = []
        self._pennylane_gates_parameters = []
        self._pennylane_gates_parameters_dimensions = {}

        symbol_tuple = tuple([sympify(p._symbol_expr) for p in circuit.parameters])

        for param in circuit.parameters:
            if param.vector.name not in self._pennylane_gates_parameters:
                self._pennylane_gates_parameters.append(param.vector.name)
                self._pennylane_gates_parameters_dimensions[param.vector.name] = 1
            else:
                self._pennylane_gates_parameters_dimensions[param.vector.name] += 1

        printer, modules = _get_sympy_interface()

        for op in circuit.data:

            # catch conditions of the gate
            # only c_if is supported, the other cases have been caught before
            if not hasattr(op.operation, "condition") or op.operation.condition is None:
                # No condition (usually the case)
                self._pennylane_conditions.append(None)
            else:
                classical_bits = op.operation.condition[0]
                val = op.operation.condition[1]
                if isinstance(classical_bits, Clbit):
                    i = circuit.find_bit(classical_bits).index
                else:
                    i = [circuit.find_bit(b).index for b in classical_bits]
                # add indices of classical bits containing measured values
                # and value of the conditions (measurement equal to val)
                self._pennylane_conditions.append((i, val))

            param_tuple = None
            if len(op.operation.params) >= 1:
                param_tuple = ()
                for param in op.operation.params:
                    if isinstance(param, ParameterExpression):
                        if param._symbol_expr == None:
                            param = param._coeff
                        else:
                            symbol_expr = sympify(param._symbol_expr)
                            f = lambdify(
                                symbol_tuple, symbol_expr, modules=modules, printer=printer
                            )

                            param_tuple += (f,)
                    else:
                        param_tuple += (param,)

            self._pennylane_gates_param_function.append(param_tuple)

            if op.operation.name == "measure":
                # Capture special case of measurement, that is stored in classical bits
                # In the pennylane implementation, classical bits are introduced as an array
                wires = [
                    circuit.find_bit(op.qubits[i]).index for i in range(op.operation.num_qubits)
                ]
                clbits = [
                    circuit.find_bit(op.clbits[i]).index for i in range(op.operation.num_clbits)
                ]
                self._pennylane_gates.append(("measure", clbits))
                self._pennylane_gates_wires.append(wires)
            else:
                # All other gates
                if op.operation.name not in qiskit_pennylane_gate_dict:
                    raise NotImplementedError(
                        f"Gate {op.operation.name} is unfortunatly not supported in sQUlearn's PennyLane backend."
                    )

                self._pennylane_gates.append(qiskit_pennylane_gate_dict[op.operation.name])
                wires = [
                    circuit.find_bit(op.qubits[i]).index for i in range(op.operation.num_qubits)
                ]
                self._pennylane_gates_wires.append(wires)

    def build_pennylane_circuit(self):
        """
        Function to build the PennyLane circuit from the Qiskit circuit and observable.

        The functions returns a callable PennyLane circuit that can be called with parameters.
        The PennyLane circuit is built from the instructions previously generated from the Qiskit
        circuit and observable.

        Returns:
            Callable PennyLane circuit
        """

        def pennylane_circuit(*args):
            """PennyLane circuit that can be called with parameters"""

            measurements = [0] * self._num_clbits

            # Collects the args values connected to the circuit parameters
            circ_param_list = sum(
                [list(args[i]) for i in range(len(self._pennylane_gates_parameters))], []
            )

            # Loop through all penny lane gates
            for i, op in enumerate(self._pennylane_gates):

                if isinstance(op, tuple):
                    # Special case for measurement
                    # add measurement to the circuit and store the result in the measurements array
                    if op[0] == "measure":
                        for j, wire in enumerate(self._pennylane_gates_wires[i]):
                            measurements[op[1][j]] = qml.measure(wire)
                else:
                    # Evaluate the (non-linear) parameter expression of the gate
                    if self._pennylane_gates_param_function[i] != None:
                        evaluated_param = tuple(
                            [
                                func(*circ_param_list) if callable(func) else func
                                for func in self._pennylane_gates_param_function[i]
                            ]
                        )
                    else:
                        evaluated_param = None

                    # Treat c_if conditions of the gate (if present)
                    if self._pennylane_conditions[i] != None:
                        # Calculate the value of the classical bit(s) involved in the condition
                        if isinstance(self._pennylane_conditions[i][0], list):
                            # conditions involving multiple classical bits -> convert to integer
                            val = 0
                            for j in range(len(self._pennylane_conditions[i][0])):
                                val += 2**j * measurements[self._pennylane_conditions[i][0][j]]
                        else:
                            val = measurements[self._pennylane_conditions[i][0]]

                        if evaluated_param is not None:
                            # The case that the gate has parameters
                            if isinstance(val, int):
                                # Conditional values are already integers
                                if val == self._pennylane_conditions[i][1]:
                                    op(*evaluated_param, wires=self._pennylane_gates_wires[i])
                            else:
                                # Otherwise, pennylane condition
                                qml.cond(val == self._pennylane_conditions[i][1], op)(
                                    *evaluated_param, wires=self._pennylane_gates_wires[i]
                                )
                        else:
                            # The case that the gate has no parameters
                            if isinstance(val, int):
                                # Conditional values are already integers
                                if val == self._pennylane_conditions[i][1]:
                                    op(wires=self._pennylane_gates_wires[i])
                            else:
                                # Otherwise, pennylane condition
                                qml.cond(val == self._pennylane_conditions[i][1], op)(
                                    wires=self._pennylane_gates_wires[i]
                                )
                    else:
                        if evaluated_param is not None:
                            op(*evaluated_param, wires=self._pennylane_gates_wires[i])
                        else:
                            op(wires=self._pennylane_gates_wires[i])

        return pennylane_circuit
