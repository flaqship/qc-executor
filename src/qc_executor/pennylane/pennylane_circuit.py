"""PennyLane native circuit, compiled from the framework-independent circuit IR."""

from __future__ import annotations

from typing import Any, Callable, Optional, cast

import pennylane as qml
import sympy as sp
from sympy import lambdify

from ..base.circuit_base import QuantumCircuitBase
from ..base.circuit_ir import CircuitIR, Instruction
from ..base.gate_set import GATE_DEFS, OPCODE_BY_NAME, OpCode
from ..parameters import sort_parameters
from ._sympy_interface import _get_sympy_interface
from .pennylane_gates import qiskit_pennylane_gate_dict

#: Opcodes PennyLane executes directly, derived from its gate table.  PennyLane
#: covers almost the whole gate set, so the shared lowering pass has little to
#: do; it replaced the ``qiskit.transpile(target=pennylane_target)`` call.
_SUPPORTED = frozenset(
    OPCODE_BY_NAME[name] for name in qiskit_pennylane_gate_dict if name in OPCODE_BY_NAME
) | {OpCode.BARRIER}


class PennyLaneCircuit(QuantumCircuitBase):
    """A quantum circuit that compiles to PennyLane.

    Built like any other circuit -- ``PennyLaneCircuit(2)`` then
    ``circuit.h(0)`` -- or converted from an existing one with
    :meth:`from_quantum_circuit`.  Compilation into PennyLane operations is
    lazy and re-runs whenever the instruction store changes.

    Args:
        num_qubits: Number of qubits in the circuit.
        num_clbits: Number of classical bits, for mid-circuit measurement.
        _ir: Adopt this instruction store instead of starting empty.
    """

    @classmethod
    def supported_opcodes(cls) -> frozenset:
        """Return the opcodes PennyLane executes directly.

        Without this the base default (the whole gate set) would apply and the
        lowering pass would silently stop rewriting anything.
        """
        return _SUPPORTED

    def __init__(
        self,
        num_qubits: int = 0,
        num_clbits: int = 0,
        *,
        _ir: "CircuitIR | None" = None,
    ) -> None:
        super().__init__(num_qubits, num_clbits, _ir=_ir)

        self._pennylane_gates = []
        self._pennylane_gates_param_function = []
        self._pennylane_gates_wires = []
        self._pennylane_conditions = []
        self._pennylane_circuit = None
        self._compiled_revision = -1

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def _ensure_compiled(self) -> None:
        """Compile the instruction store into PennyLane operations, once per revision."""
        if self._compiled_revision != self._ir.revision:
            self._build_circuit_instructions(self._lowered_ir())
            self._compiled_revision = self._ir.revision
            self._pennylane_circuit = None

    def _build_native(self) -> Callable:
        """Compile the instruction store into the callable PennyLane circuit."""
        return self.build_pennylane_circuit()

    @property
    def pennylane_circuit(self) -> Optional[Callable]:
        """PennyLane circuit that can be called with parameters"""
        self._ensure_compiled()
        if self._pennylane_circuit is None:
            self._pennylane_circuit = self.build_pennylane_circuit()
        return self._pennylane_circuit

    @property
    def parameter_names(self) -> list:
        """List of circuit parameter names"""
        return list(self.parameter_dimensions)

    @property
    def parameter_dimensions(self) -> dict:
        """Dictionary with the dimension of each circuit parameter.

        Derived from the instruction store rather than from the compiled
        operations, so it answers without forcing compilation.  Lowering does
        not change which parameters appear, only which gates carry them.
        """
        dimensions: dict = {}
        for parameter in self.parameters:
            dimensions[parameter.vector_name] = dimensions.get(parameter.vector_name, 0) + 1
        return dimensions

    def __call__(self, *args, **kwargs):
        return self.pennylane_circuit(*args, **kwargs)

    @staticmethod
    def _get_gate_condition(instruction: Instruction):
        """Get the classical condition for an instruction, or None if unconditional.

        Reads the condition straight off the IR.  The previous implementation
        looked for ``Instruction.condition``, which Qiskit removed in 2.0, so
        conditional gates had silently stopped being applied.

        Args:
            instruction: The instruction to inspect.

        Returns:
            ``(bit_indices, value)`` where ``bit_indices`` is an int for a
            single classical bit and a list otherwise, or ``None``.
        """
        condition = instruction.condition
        if condition is None:
            return None
        bits = list(condition.clbits)
        return (bits[0] if len(bits) == 1 else bits, condition.value)

    @staticmethod
    def _get_gate_param_tuple(instruction: Instruction, symbol_tuple, printer, modules):
        """Build the parameter function tuple for a gate, or None if it has none.

        Angles arrive as numbers or SymPy expressions; symbolic ones are
        lambdified onto PennyLane's autograd-aware numpy.
        """
        if not instruction.params:
            return None
        param_tuple: tuple = ()
        for param in instruction.params:
            if isinstance(param, sp.Basic) and param.free_symbols:
                param_tuple += (lambdify(symbol_tuple, param, modules=modules, printer=printer),)
            else:
                param_tuple += (float(param),)
        return param_tuple

    def _build_circuit_instructions(self, ir: CircuitIR) -> None:
        """Build the PennyLane instruction lists from the circuit IR.

        Walks the lowered instruction store directly; this used to iterate a
        transpiled Qiskit circuit and look gates up by their Qiskit name.

        Args:
            ir: The lowered instruction store.

        Raises:
            NotImplementedError: For gates PennyLane cannot express.
        """
        self._pennylane_gates = []
        self._pennylane_gates_param_function = []
        self._pennylane_gates_wires = []
        self._pennylane_conditions = []

        # Same order parameter_dimensions counts in, so the executor's
        # flattened argument list lines up with the lambdified callables.
        symbol_tuple = tuple(sort_parameters(ir.free_parameters))

        printer, modules = _get_sympy_interface()

        for instruction in ir:
            # Barriers are compiler directives with no effect on the statevector.
            if instruction.opcode is OpCode.BARRIER:
                continue

            self._pennylane_conditions.append(self._get_gate_condition(instruction))
            self._pennylane_gates_param_function.append(
                self._get_gate_param_tuple(instruction, symbol_tuple, printer, modules)
            )

            if instruction.opcode is OpCode.MEASURE:
                # Measurement results are stored in a classical-bit array.
                self._pennylane_gates.append(("measure", list(instruction.clbits)))
                self._pennylane_gates_wires.append(list(instruction.qubits))
                continue

            name = GATE_DEFS[instruction.opcode].name
            if name not in qiskit_pennylane_gate_dict:
                raise NotImplementedError(
                    f"Gate {name} is unfortunatly not supported "
                    "in sQUlearn's PennyLane backend."
                )
            self._pennylane_gates.append(qiskit_pennylane_gate_dict[name])
            self._pennylane_gates_wires.append(list(instruction.qubits))

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
        """Return a callable PennyLane circuit built from the instruction store.

        Compiles the store first if it has changed since the last call.

        Returns:
            Callable PennyLane circuit
        """
        self._ensure_compiled()

        def pennylane_circuit(*args):
            """PennyLane circuit that can be called with parameters"""

            measurements: list = [0] * self.num_clbits

            # Collects the args values connected to the circuit parameters
            circ_param_list = sum([list(args[i]) for i in range(len(self.parameter_names))], [])

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
