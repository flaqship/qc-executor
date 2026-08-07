"""Qulacs circuit wrapper built on the framework-independent circuit IR."""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional

import numpy as np
import sympy as sp
from qulacs import ParametricQuantumCircuit  # pylint: disable=no-name-in-module
from qulacs import QuantumCircuit as QulacsQuantumCircuit  # pylint: disable=no-name-in-module
from sympy import lambdify

from ..base.circuit_ir import CircuitIR
from ..base.decompose import decompose_ir
from ..base.gate_set import GATE_DEFS, OpCode
from ..parameters import Parameter, sort_parameters
from ..quantum_circuit import QuantumCircuit
from .qulacs_gates import qiskit_qulacs_gate_dict, qiskit_qulacs_param_gate_dict

#: Opcodes Qulacs executes directly, taken from the gate tables below.  Anything
#: else is lowered by the shared decomposition pass, which replaced the
#: ``qiskit.transpile(target=qulacs_target)`` call this class used to make.
_SUPPORTED = frozenset(
    {
        OpCode.I,
        OpCode.H,
        OpCode.X,
        OpCode.Y,
        OpCode.Z,
        OpCode.S,
        OpCode.SDG,
        OpCode.T,
        OpCode.TDG,
        OpCode.SWAP,
        OpCode.CX,
        OpCode.CZ,
        OpCode.CCX,
        OpCode.RX,
        OpCode.RY,
        OpCode.RZ,
        OpCode.BARRIER,
    }
)


class QulacsCircuit:
    """Wrapper class that compiles a circuit IR into a Qulacs-compatible circuit."""

    @classmethod
    def from_quantum_circuit(cls, circuit: QuantumCircuit) -> "QulacsCircuit":
        """Create a Qulacs native circuit from a generic circuit."""
        return cls(circuit)

    @classmethod
    def supported_opcodes(cls) -> frozenset:
        """Return the opcodes Qulacs executes directly."""
        return _SUPPORTED

    def __init__(
        self,
        circuit: QuantumCircuit,
    ) -> None:
        # Lower to the supported basis instead of transpiling through Qiskit.
        self._ir: CircuitIR = decompose_ir(circuit.ir, _SUPPORTED)
        self._num_qubits = self._ir.num_qubits

        self._operation_list = []
        self._qubit_list = []
        self._func_list = []
        self._func_grad_list = []
        self._free_parameters = set()
        self._used_parameters = []
        self._qulacs_gates_parameters = {}
        self._symbol_tuple_circuit = tuple()
        self._rebuild_circuit_func = True
        self._circuit_func = None
        self._num_clbits = self._ir.num_clbits
        self._build_circuit_instructions(self._ir)

        self._circuit_func_cache = {}
        self._outer_jacobi_circuit_cache = {}
        self._qulacs_circuit = None

    @property
    def num_qubits(self) -> int:
        """Number of qubits of the circuit"""
        return self._num_qubits

    @property
    def qulacs_circuit(self) -> Optional[Callable]:
        """Qulacs circuit that can be called with parameters"""
        return self._qulacs_circuit

    @property
    def parameter_names(self) -> list:
        """List of circuit parameter names"""
        return list(self._qulacs_gates_parameters.keys())

    @property
    def parameter_dimensions(self) -> dict:
        """Dictionary with the dimension of each circuit parameter"""
        return self._qulacs_gates_parameters

    @property
    def circuit_arguments(self) -> dict:
        """Dictionary of all circuit and observable parameters names"""
        return self._qulacs_gates_parameters

    @property
    def hash(self) -> int:
        """Hashable object of the circuit and observable for caching"""
        return hash(self._ir)

    @property
    def free_parameters(self) -> set:
        """Return the set of free (non-bound) parameters in the circuit."""
        return self._free_parameters

    def get_qulacs_circuit(self) -> Callable:
        """Builds and returns the Qulacs circuit as callable function"""
        self._qulacs_circuit = self.get_circuit_func()
        return self._qulacs_circuit

    def __call__(self, *args, **kwargs):
        if self._qulacs_circuit is None:
            self._qulacs_circuit = self.get_circuit_func()
        return self._qulacs_circuit(*args, **kwargs)

    def __hash__(self):
        return hash(self._ir.fingerprint())

    def _add_parameter_expression(self, angle: Any) -> Any:
        """
        Adds a parameter expression to the circuit and do the pre-processing.

        Angles arrive as plain numbers or SymPy expressions.  Symbolic angles are
        lambdified once, and their derivatives come from ``sympy.diff``; this
        used to go through Qiskit's ``ParameterExpression.gradient``.

        Args:
            angle: Angle of rotation, numeric or symbolic.

        Returns:
            tuple: the angle callable (or constant), the per-parameter gradient
                callables, the parameters used, and whether it is symbolic.

        Raises:
            TypeError: If the angle is neither numeric nor a SymPy expression.
        """
        func_list_element = None
        func_grad_list_element = None
        parameterized = False
        used_parameters = None

        # Change sign because of the way Qulacs defines the rotation gates
        angle = -angle

        if isinstance(angle, sp.Basic) and angle.free_symbols:
            parameterized = True
            func_list_element = lambdify(self._symbol_tuple_circuit, angle)
            func_grad_list_element = []
            used_parameters = []
            for param_element in sort_parameters(
                s for s in angle.free_symbols if isinstance(s, Parameter)
            ):
                self._free_parameters.add(param_element)
                used_parameters.append(param_element)
                derivative = sp.diff(angle, param_element)
                if derivative.free_symbols:
                    func_grad_list_element.append(lambdify(self._symbol_tuple_circuit, derivative))
                else:
                    # Call-by-value so the closure keeps this gate's constant.
                    value = float(derivative)
                    func_grad_list_element.append(lambda *_args, value=value: value)
        elif isinstance(angle, (float, int, sp.Basic)):
            func_list_element = float(angle)
            func_grad_list_element = None
        else:
            raise TypeError(
                f"Unsupported type for angle: {type(angle)}. "
                "Expected a number or a SymPy expression."
            )

        return func_list_element, func_grad_list_element, used_parameters, parameterized

    def _add_single_qubit_gate(self, gate_name: str, qubits: int | Iterable[int]):
        """
        Adds a single qubit gate to the circuit.

        Args:
            gate_name (str): Name of the gate
            qubits (int or Iterable[int]): qubit indices
        """
        qubits = [qubits] if isinstance(qubits, int) else qubits
        for q in qubits:
            self._operation_list.append(gate_name)
            if q >= self.num_qubits:
                raise ValueError(f"Qubit index {q} is out of range")
            self._qubit_list.append([q])
            self._func_list.append(None)
            self._func_grad_list.append(None)
            self._used_parameters.append([])

        self._rebuild_circuit_func = True

    def _add_two_qubit_gate(
        self, gate_name: str, qubit1: int | Iterable[int], qubit2: int | Iterable[int]
    ) -> None:
        """
        Adds a two qubit gate to the circuit.

        Args:
            gate_name (str): Name of the gate
            qubit1 (int or Iterable[int]): qubit indices of the first qubit (e.g. control)
            qubit2 (int or Iterable[int]): qubit indices of the second qubit (e.g. target)
        """
        qubit1 = [qubit1] if isinstance(qubit1, int) else qubit1
        qubit2 = [qubit2] if isinstance(qubit2, int) else qubit2

        for control, target in zip(qubit1, qubit2):
            if control >= self.num_qubits or target >= self.num_qubits:
                raise ValueError("Qubit index is out of range")
            self._operation_list.append(gate_name)
            self._qubit_list.append([control, target])
            self._func_list.append(None)
            self._func_grad_list.append(None)
            self._used_parameters.append([])

        self._rebuild_circuit_func = True

    def _add_three_qubit_gate(self, gate_name: str, qubit1: int, qubit2: int, qubit3: int) -> None:
        """
        Adds a three qubit gate to the circuit.

        Args:
            gate_name (str): Name of the gate
            qubit1 (int): qubit index of the first qubit (e.g. first control)
            qubit2 (int): qubit index of the second qubit (e.g. second control)
            qubit3 (int): qubit index of the third qubit (e.g. target)
        """
        for q in (qubit1, qubit2, qubit3):
            if q >= self.num_qubits:
                raise ValueError(f"Qubit index {q} is out of range")

        self._operation_list.append(gate_name)
        self._qubit_list.append([qubit1, qubit2, qubit3])
        self._func_list.append(None)
        self._func_grad_list.append(None)
        self._used_parameters.append([])

        self._rebuild_circuit_func = True

    def _add_parameterized_single_qubit_gate(
        self,
        gate_name: str,
        qubits: int | Iterable[int],
        angle: Any,
    ):
        """
        Adds a single qubit parameterized gate to the circuit.

        Args:
            gate_name (str): Name of the gate
            qubits (int or Iterable[int]): qubit indices
            angle: Angle of rotation; a number or a SymPy expression
        """
        func_list_element, func_grad_list_element, used_parameters, parameterized = (
            self._add_parameter_expression(angle)
        )

        qubits = [qubits] if isinstance(qubits, int) else qubits
        for q in qubits:
            if q >= self.num_qubits:
                raise ValueError(f"Qubit index {q} is out of range")
            if parameterized:
                self._operation_list.append(gate_name)
            else:
                self._operation_list.append(gate_name)
            self._qubit_list.append([q])
            self._func_list.append(func_list_element)
            self._func_grad_list.append(func_grad_list_element)
            if used_parameters is None:
                self._used_parameters.append([])
            else:
                self._used_parameters.append(used_parameters)

        self._rebuild_circuit_func = True

    def _add_parameterized_two_qubit_gate(
        self,
        gate_name: str,
        qubit1: int | Iterable[int],
        qubit2: int | Iterable[int],
        angle: Any,
    ):
        """
        Adds a single qubit parameterized gate to the circuit.

        Args:
            gate_name (str): Name of the gate
            qubits (int or Iterable[int]): qubit indices
            angle: Angle of rotation; a number or a SymPy expression
        """
        func_list_element, func_grad_list_element, used_parameters, parameterized = (
            self._add_parameter_expression(angle)
        )

        qubit1 = [qubit1] if isinstance(qubit1, int) else qubit1
        qubit2 = [qubit2] if isinstance(qubit2, int) else qubit2

        for control, target in zip(qubit1, qubit2):
            if control >= self.num_qubits or target >= self.num_qubits:
                raise ValueError("Qubit index is out of range")
            if parameterized:
                self._operation_list.append(gate_name)
            else:
                self._operation_list.append(gate_name)
            self._qubit_list.append([control, target])
            self._func_list.append(func_list_element)
            self._func_grad_list.append(func_grad_list_element)
            if used_parameters is None:
                self._used_parameters.append([])
            else:
                self._used_parameters.append(used_parameters)

        self._rebuild_circuit_func = True

    def _build_circuit_instructions(self, ir: CircuitIR) -> None:
        """Build the Qulacs instruction lists from the circuit IR.

        Walks the lowered instruction store directly; this used to iterate a
        transpiled Qiskit circuit and look gates up by their Qiskit name.

        Args:
            ir: The lowered instruction store.

        Raises:
            NotImplementedError: For gates or features Qulacs cannot express.
        """
        self._operation_list = []
        self._param_list = []
        self._qubit_list = []
        self._func_list = []
        self._func_grad_list = []
        self._free_parameters = set()
        self._qulacs_gates_parameters = {}
        self._symbol_tuple_circuit = tuple()

        for param in sort_parameters(ir.free_parameters):
            name = param.vector_name
            self._qulacs_gates_parameters[name] = self._qulacs_gates_parameters.get(name, 0) + 1

        self._symbol_tuple_circuit = tuple(sort_parameters(ir.free_parameters))

        for instruction in ir:
            # Barriers are compiler directives with no effect on the statevector.
            if instruction.opcode is OpCode.BARRIER:
                continue

            if instruction.condition is not None or instruction.opcode in {
                OpCode.MEASURE,
                OpCode.RESET,
            }:
                raise NotImplementedError(
                    "Mid-circuit measurement, reset and classical conditions are not "
                    "supported by the Qulacs backend."
                )

            name = GATE_DEFS[instruction.opcode].name
            if name not in qiskit_qulacs_gate_dict and name not in qiskit_qulacs_param_gate_dict:
                raise NotImplementedError(
                    f"Gate {name} is unfortunatly not supported in sQUlearn's Qulacs backend."
                )

            wires = list(instruction.qubits)
            parameterized_gate = bool(instruction.params)

            if len(wires) == 1:
                if not parameterized_gate:
                    self._add_single_qubit_gate(name, wires)
                else:
                    self._add_parameterized_single_qubit_gate(name, wires, instruction.params[0])
            elif len(wires) == 2:
                if not parameterized_gate:
                    self._add_two_qubit_gate(name, wires[0], wires[1])
                else:
                    self._add_parameterized_two_qubit_gate(
                        name, wires[0], wires[1], instruction.params[0]
                    )
            elif len(wires) == 3 and not parameterized_gate:
                self._add_three_qubit_gate(name, wires[0], wires[1], wires[2])
            else:
                raise NotImplementedError(
                    "Only up to three qubit (non-parameterized) gates are supported "
                    "in sQUlearn's Qulacs backend."
                )

    def get_circuit_func(self, gradient_param=None):
        """Returns the Qulacs circuit function for the circuit."""

        if isinstance(gradient_param, Parameter):
            gradient_param = [gradient_param]
        gradient_param = list(gradient_param) if gradient_param is not None else []

        is_parameterized = len(gradient_param)
        parameterized_operations = [
            any(param in gradient_param for param in self._used_parameters[i])
            for i, _ in enumerate(self._operation_list)
        ]

        cache_value = "no_gradient"
        if is_parameterized:
            cache_value = tuple(gradient_param)

        if cache_value in self._circuit_func_cache:
            return self._circuit_func_cache[cache_value]

        def qulacs_circuit(*args):

            # Collects the args values connected to the circuit parameters
            circ_param_list = sum([list(args[i]) for i in range(len(self.parameter_names))], [])

            if is_parameterized:
                circuit = ParametricQuantumCircuit(self.num_qubits)
            else:
                circuit = QulacsQuantumCircuit(self.num_qubits)

            # Build the Qulacs circuit and evaluate the parametric terms
            for i, qulacs_operation in enumerate(self._operation_list):
                if self._func_list[i] is None:
                    qiskit_qulacs_gate_dict[qulacs_operation](circuit, *self._qubit_list[i])
                elif isinstance(self._func_list[i], (float, int)):
                    qiskit_qulacs_gate_dict[qulacs_operation](
                        circuit, self._func_list[i], *self._qubit_list[i]
                    )
                else:
                    value = self._func_list[i](*circ_param_list)
                    if parameterized_operations[i]:
                        assert isinstance(circuit, ParametricQuantumCircuit)
                        qiskit_qulacs_param_gate_dict[qulacs_operation](
                            circuit, value, *self._qubit_list[i]
                        )
                    else:
                        qiskit_qulacs_gate_dict[qulacs_operation](
                            circuit, value, *self._qubit_list[i]
                        )

            return circuit

        self._circuit_func_cache[cache_value] = qulacs_circuit

        return qulacs_circuit

    def get_gradient_outer_jacobian(
        self,
        gradient_parameters: Parameter | List[Parameter] | None = None,
    ):
        """Returns the outer jacobian needed for the chain rule in circuit derivatives.

        Qulacs does not support multiple parameters and parameter expressions,
        so we need to calculate a transformation which also includes the gradient of the
        parameter expression.

        Args:
            gradient_parameters (Parameter | List[Parameter] | None):
                Parameters to calculate the gradient for
        """

        if isinstance(gradient_parameters, Parameter):
            gradient_parameters = [gradient_parameters]
        gradient_parameters = list(gradient_parameters) if gradient_parameters is not None else []
        gradient_param_dict = {p: i for i, p in enumerate(gradient_parameters)}

        cache_value = "no_gradient"
        if len(gradient_parameters) > 0:
            cache_value = tuple(gradient_parameters)

        if cache_value in self._outer_jacobi_circuit_cache:
            return self._outer_jacobi_circuit_cache[cache_value]

        def outer_jacobian(*args):

            # Collects the args values connected to the circuit parameters
            circ_param_list = sum([list(args[i]) for i in range(len(self.parameter_names))], [])

            relevant_operations = [
                i
                for i in range(len(self._operation_list))
                if any(param in gradient_parameters for param in self._used_parameters[i])
            ]

            outer_jacobian = np.zeros((len(relevant_operations), len(gradient_parameters)))

            for i, operation in enumerate(relevant_operations):
                for j, param in enumerate(self._used_parameters[operation]):
                    if param in gradient_parameters:
                        outer_jacobian[i, gradient_param_dict[param]] = self._func_grad_list[
                            operation
                        ][j](*circ_param_list)

            return outer_jacobian

        self._outer_jacobi_circuit_cache[cache_value] = outer_jacobian

        return outer_jacobian
