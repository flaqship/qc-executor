from __future__ import annotations

from typing import Callable, List

import pennylane.numpy as pnp
import sympy as sp
from sympy import lambdify

from ..abstraction.abstract_parameter import free_parameters
from ..abstraction.abstract_quantum_circuit import (
    AbstractQuantumCircuit,
    Barrier,
    RotationGate,
)
from .pennylane_gates import qiskit_pennylane_gate_dict


def _get_sympy_interface():
    """
    Returns the sympy interface that is used in the parameter conversion.

    Necessary for the correct conversion of sympy expressions to python
    functions in PennyLane.

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

    return printer, modules


class PennyLaneCircuit:
    """PennyLane runtime representation built from an :class:`AbstractQuantumCircuit`.

    The translation logic (framework-independent gate list -> PennyLane
    operations) lives in this backend class, so the abstraction layer stays
    free of any PennyLane dependency. A :class:`PennyLaneCircuit` is callable
    once :meth:`build_pennylane_circuit` (or :meth:`get_pennylane_circuit`) has
    been invoked.

    Args:
        circuit (AbstractQuantumCircuit): The framework-independent circuit to
            translate into PennyLane operations.
    """

    @classmethod
    def from_quantum_circuit(cls, circuit: AbstractQuantumCircuit) -> "PennyLaneCircuit":
        """Create a PennyLane native circuit from an abstract circuit."""
        return cls(circuit)

    def __init__(
        self,
        circuit: AbstractQuantumCircuit,
    ) -> None:

        self._circuit = circuit
        self._num_qubits = circuit.num_qubits
        self._num_clbits = 0

        self._pennylane_gates: List = []
        self._pennylane_gates_param_function: List = []
        self._pennylane_gates_wires: List = []
        self._pennylane_gates_parameters: List = []
        self._pennylane_gates_parameters_dimensions: dict = {}
        self._pennylane_circuit: Callable | None = None

        # Build the PennyLane instructions directly from the abstract gate list
        self._build_circuit_instructions(circuit)

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the circuit"""
        return self._num_qubits

    @property
    def pennylane_circuit(self) -> Callable:
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
    def hash(self) -> int:
        """Hashable object of the circuit for caching"""
        return hash(self._circuit)

    def get_pennylane_circuit(self) -> Callable:
        """Builds and returns the PennyLane circuit as callable function"""
        self._pennylane_circuit = self.build_pennylane_circuit()
        return self._pennylane_circuit

    def __call__(self, *args, **kwargs):
        return self._pennylane_circuit(*args, **kwargs)

    @staticmethod
    def _build_angle_function(angle, symbol_tuple, printer, modules):
        """Convert a single gate angle into a constant or a callable.

        Numeric angles are returned unchanged. Symbolic angles are lambdified
        over the full ordered parameter tuple, so they can later be evaluated
        from the flat list of parameter values passed at call time.

        Args:
            angle: Numeric value or symbolic (SymPy) angle expression.
            symbol_tuple: Ordered tuple of all circuit parameters.
            printer: SymPy printer used by lambdify.
            modules: Numeric module backend used by lambdify (pennylane.numpy).

        Returns:
            The numeric angle, or a callable ``f(*flat_params) -> number``.
        """
        if not isinstance(angle, sp.Basic):
            return angle  # plain float / int
        if not free_parameters(angle):
            return float(angle)  # constant symbolic expression
        return lambdify(symbol_tuple, angle, modules=modules, printer=printer)

    def _build_circuit_instructions(self, circuit: AbstractQuantumCircuit) -> None:
        """
        Build the PennyLane instructions directly from the abstract gate list.

        This converts the typed gates and symbolic angle expressions of the
        :class:`AbstractQuantumCircuit` into PennyLane compatible gate callables
        and parameter functions.

        Args:
            circuit (AbstractQuantumCircuit): Abstract circuit to convert to PennyLane
        """

        self._pennylane_gates = []
        self._pennylane_gates_param_function = []
        self._pennylane_gates_wires = []
        self._pennylane_gates_parameters = []
        self._pennylane_gates_parameters_dimensions = {}

        # Ordered tuple of all symbolic parameters (sorted by vector name, index).
        # The order defines the flat parameter layout used in build_pennylane_circuit.
        symbol_tuple = tuple(circuit.parameters)

        for param in symbol_tuple:
            if param.vector_name not in self._pennylane_gates_parameters:
                self._pennylane_gates_parameters.append(param.vector_name)
                self._pennylane_gates_parameters_dimensions[param.vector_name] = 1
            else:
                self._pennylane_gates_parameters_dimensions[param.vector_name] += 1

        printer, modules = _get_sympy_interface()

        for gate in circuit._gates:

            if isinstance(gate, Barrier):
                # Barriers are no-ops for simulation
                continue

            if gate.name not in qiskit_pennylane_gate_dict:
                raise NotImplementedError(
                    f"Gate {gate.name} is unfortunatly not supported in sQUlearn's PennyLane backend."
                )

            param_tuple = None
            if isinstance(gate, RotationGate):
                param_tuple = (
                    self._build_angle_function(gate.angle, symbol_tuple, printer, modules),
                )

            self._pennylane_gates_param_function.append(param_tuple)
            self._pennylane_gates.append(qiskit_pennylane_gate_dict[gate.name])
            self._pennylane_gates_wires.append(list(gate.qubits))

    def build_pennylane_circuit(self):
        """
        Function to build the PennyLane circuit from the abstract circuit.

        The function returns a callable PennyLane circuit that can be called with
        parameters. The PennyLane circuit is built from the instructions previously
        generated from the abstract circuit.

        Returns:
            Callable PennyLane circuit
        """

        def pennylane_circuit(*args):
            """PennyLane circuit that can be called with parameters"""

            # Collects the args values connected to the circuit parameters
            circ_param_list = sum(
                [list(args[i]) for i in range(len(self._pennylane_gates_parameters))], []
            )

            # Loop through all pennylane gates
            for i, circuit_gate in enumerate(self._pennylane_gates):

                # Evaluate the (non-linear) parameter expression of the gate
                if self._pennylane_gates_param_function[i] is not None:
                    evaluated_param = tuple(
                        func(*circ_param_list) if callable(func) else func
                        for func in self._pennylane_gates_param_function[i]
                    )
                    circuit_gate(*evaluated_param, wires=self._pennylane_gates_wires[i])
                else:
                    circuit_gate(wires=self._pennylane_gates_wires[i])

        return pennylane_circuit
