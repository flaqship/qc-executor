"""Circuit module for converting Qiskit circuits to PennyLane native circuits."""

from __future__ import annotations

from typing import Any, Callable, Optional, cast

import pennylane as qml
from qiskit import transpile
from qiskit.circuit import Clbit, ParameterExpression
from qiskit.circuit import QuantumCircuit as QiskitQuantumCircuit
from sympy import lambdify

from ..quantum_circuit import QuantumCircuit
from ..utils.decompose_to_std import decompose_to_std
from ..utils.qiskit_compat import _param_is_constant, _param_to_float, _param_to_sympy
from ._sympy_interface import _get_sympy_interface
from .pennylane_gates import pennylane_target, qiskit_pennylane_gate_dict


class PennyLaneCircuit:
    """PennyLane circuit representation converted from a generic QuantumCircuit."""

    @classmethod
    def from_quantum_circuit(cls, circuit: QuantumCircuit) -> "PennyLaneCircuit":
        """Create a PennyLane native circuit from a generic circuit."""
        return cls(circuit)

    def __init__(
        self,
        circuit: QuantumCircuit,
    ) -> None:

        # Transpile circuit to supported basis gates and expand blocks automatically
        self._qiskit_circuit = transpile(
            decompose_to_std(circuit.qiskit_circuit),
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
        self._pennylane_circuit = None

        # Build circuit instructions for the pennylane circuit from the qiskit circuit
        self._build_circuit_instructions(self._qiskit_circuit)

        # self._pennylane_circuit = self.build_pennylane_circuit()

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the circuit"""
        return self._num_qubits

    @property
    def pennylane_circuit(self) -> Optional[Callable]:
        """PennyLane circuit that can be called with parameters"""
        if self._pennylane_circuit is None:
            self._pennylane_circuit = self.build_pennylane_circuit()
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
    def hash(self) -> int:
        """Hashable object of the circuit and observable for caching"""
        return hash(str(self._qiskit_circuit))

    def __call__(self, *args, **kwargs):
        return self.pennylane_circuit(*args, **kwargs)

    def _get_gate_condition(self, circuit: QiskitQuantumCircuit, gate_operation: Any):
        """Get the classical condition for a gate, or None if unconditional."""
        if (
            not hasattr(gate_operation.operation, "condition")
            or gate_operation.operation.condition is None
        ):
            return None
        classical_bits = gate_operation.operation.condition[0]
        val = gate_operation.operation.condition[1]
        if isinstance(classical_bits, Clbit):
            bit_indices = circuit.find_bit(classical_bits).index
        else:
            bit_indices = [circuit.find_bit(b).index for b in classical_bits]
        return (bit_indices, val)

    def _get_gate_param_tuple(self, gate_operation, symbol_tuple, printer, modules):
        """Build the parameter function tuple for a gate, or None if no parameters."""
        if len(gate_operation.operation.params) < 1:
            return None
        param_tuple = ()
        for param in gate_operation.operation.params:
            if isinstance(param, ParameterExpression):
                if _param_is_constant(param):
                    param = _param_to_float(param)
                else:
                    symbol_expr = _param_to_sympy(param)
                    f = lambdify(symbol_tuple, symbol_expr, modules=modules, printer=printer)
                    param_tuple += (f,)
            else:
                param_tuple += (param,)
        return param_tuple

    def _build_circuit_instructions(self, circuit: QiskitQuantumCircuit) -> None:
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

        symbol_tuple = tuple(_param_to_sympy(p) for p in circuit.parameters)

        for param in circuit.parameters:
            if param.vector.name not in self._pennylane_gates_parameters:
                self._pennylane_gates_parameters.append(param.vector.name)
                self._pennylane_gates_parameters_dimensions[param.vector.name] = 1
            else:
                self._pennylane_gates_parameters_dimensions[param.vector.name] += 1

        printer, modules = _get_sympy_interface()

        for gate_operation in circuit.data:
            self._pennylane_conditions.append(self._get_gate_condition(circuit, gate_operation))
            self._pennylane_gates_param_function.append(
                self._get_gate_param_tuple(gate_operation, symbol_tuple, printer, modules)
            )

            if gate_operation.operation.name == "measure":
                # Capture special case of measurement, that is stored in classical bits
                # In the pennylane implementation, classical bits are introduced as an array
                wires = [
                    circuit.find_bit(gate_operation.qubits[i]).index
                    for i in range(gate_operation.operation.num_qubits)
                ]
                clbits = [
                    circuit.find_bit(gate_operation.clbits[i]).index
                    for i in range(gate_operation.operation.num_clbits)
                ]
                self._pennylane_gates.append(("measure", clbits))
                self._pennylane_gates_wires.append(wires)
            else:
                # All other gates
                if gate_operation.operation.name not in qiskit_pennylane_gate_dict:
                    raise NotImplementedError(
                        f"Gate {gate_operation.operation.name} is unfortunatly not supported "
                        "in sQUlearn's PennyLane backend."
                    )
                self._pennylane_gates.append(
                    qiskit_pennylane_gate_dict[gate_operation.operation.name]
                )
                wires = [
                    circuit.find_bit(gate_operation.qubits[i]).index
                    for i in range(gate_operation.operation.num_qubits)
                ]
                self._pennylane_gates_wires.append(wires)

    def _apply_conditional_gate(
        self, circuit_gate, evaluated_param, condition, measurements, wires
    ):
        """Apply a gate that has a classical condition."""
        condition_idx, condition_target = condition
        if isinstance(condition_idx, list):
            val = sum(2**j * measurements[condition_idx[j]] for j in range(len(condition_idx)))
        else:
            val = measurements[condition_idx]

        if isinstance(val, int):
            if val == condition_target:
                if evaluated_param is not None:
                    circuit_gate(*evaluated_param, wires=wires)
                else:
                    circuit_gate(wires=wires)
        else:
            # Cast to Callable so the type checker knows it accepts abitrary arguments
            cond_fn = cast(Callable[..., Any], qml.cond(val == condition_target, circuit_gate))
            if evaluated_param is not None:
                cond_fn(*evaluated_param, wires=wires)
            else:
                cond_fn(wires=wires)

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

            measurements: list = [0] * self._num_clbits

            # Collects the args values connected to the circuit parameters
            circ_param_list = sum(
                [list(args[i]) for i in range(len(self._pennylane_gates_parameters))], []
            )

            # Loop through all penny lane gates
            for i, circuit_gate in enumerate(self._pennylane_gates):

                if isinstance(circuit_gate, tuple):
                    # Special case for measurement
                    # add measurement to the circuit and store the result in the measurements array
                    if circuit_gate[0] == "measure":
                        for j, wire in enumerate(self._pennylane_gates_wires[i]):
                            measurements[circuit_gate[1][j]] = qml.measure(wire)
                else:
                    # Evaluate the (non-linear) parameter expression of the gate
                    evaluated_param = None
                    if self._pennylane_gates_param_function[i] is not None:
                        evaluated_param = tuple(
                            func(*circ_param_list) if callable(func) else func
                            for func in self._pennylane_gates_param_function[i]
                        )

                    wires = self._pennylane_gates_wires[i]
                    condition = self._pennylane_conditions[i]

                    # Treat c_if conditions of the gate (if present)
                    if condition is not None:
                        self._apply_conditional_gate(
                            circuit_gate, evaluated_param, condition, measurements, wires
                        )
                    elif evaluated_param is not None:
                        circuit_gate(*evaluated_param, wires=wires)
                    else:
                        circuit_gate(wires=wires)

        return pennylane_circuit
