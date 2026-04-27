from __future__ import annotations

from typing import Iterable, List

import numpy as np
from qiskit import transpile
from qiskit.circuit import ParameterExpression
from qiskit.circuit.parametervector import ParameterVectorElement
from qulacs import ParametricQuantumCircuit
from qulacs import QuantumCircuit as QulacsQuantumCircuit
from sympy import lambdify

from ..quantum_circuit import QuantumCircuit
from ..utils.decompose_to_std import decompose_to_std
from ..utils.qiskit_compat import _param_free_symbols, _param_to_sympy
from ..utils.qiskit_hash_functions import _circuit_key
from .qulacs_gates import qiskit_qulacs_gate_dict, qiskit_qulacs_param_gate_dict, qulacs_target


class QulacsCircuit:

    @classmethod
    def from_quantum_circuit(cls, circuit: QuantumCircuit) -> "QulacsCircuit":
        """Create a Qulacs native circuit from a generic circuit."""
        return cls(circuit)

    def __init__(
        self,
        circuit: QuantumCircuit,
    ) -> None:

        # Transpile circuit to supported basis gates and expand blocks automatically
        self._qiskit_circuit = transpile(
            decompose_to_std(circuit._qiskit_circuit),
            target=qulacs_target,
            optimization_level=0,
        )
        self._num_qubits = self._qiskit_circuit.num_qubits

        # Build circuit instructions for the qulacs observable from the qiskit circuit
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
        self._num_clbits = self._qiskit_circuit.num_clbits
        self._build_circuit_instructions(self._qiskit_circuit)

        self._circuit_func_cache = {}
        self._outer_jacobi_circuit_cache = {}

    @property
    def num_qubits(self) -> int:
        """Number of qubits of the circuit"""
        return self._num_qubits

    @property
    def qulacs_circuit(self) -> callable:
        """Qulacs circuit that can be called with parameters"""
        return self._qulacs_circuit

    @property
    def parameter_names(self) -> list:
        """List of circuit parameter names"""
        return self._qulacs_gates_parameters.keys()

    @property
    def parameter_dimensions(self) -> dict:
        """Dictionary with the dimension of each circuit parameter"""
        return self._qulacs_gates_parameters

    @property
    def circuit_arguments(self) -> list:
        """List of all circuit and observable parameters names"""
        return self._qulacs_gates_parameters

    def get_qulacs_circuit(self) -> callable:
        """Builds and returns the Qulacs circuit as callable function"""
        self._qulacs_circuit = self.build_qulacs_circuit()
        return self._qulacs_circuit

    def __call__(self, *args, **kwargs):
        return self._qulacs_circuit(*args, **kwargs)

    def __hash__(self):
        return hash(_circuit_key(self._qiskit_circuit))

    def _add_parameter_expression(
        self, angle: ParameterVectorElement | ParameterExpression | float
    ):
        """
        Adds a parameter expression to the circuit and do the pre-processing.

        Args:
            angle (ParameterVectorElement or ParameterExpression or float): angle of rotation

        Returns:
            int: index of the parameter in the parameter vector
            callable: function to calculate the parameter expression
            callable: function to calculate the gradient of the parameter expression

        """

        # In case angle is a float or something similar, we do not need to add a parameter
        func_list_element = None
        func_grad_list_element = None
        parameterized = False
        used_parameters = None

        # Change sign because of the way Qulacs defines the rotation gates
        angle = -angle

        if isinstance(angle, (float, int)):
            # Single float value
            func_list_element = angle
            func_grad_list_element = None
        elif isinstance(angle, ParameterVectorElement):
            # Single parameter vector element
            parameterized = True
            func_list_element = lambdify(self._symbol_tuple_circuit, _param_to_sympy(angle))
            func_grad_list_element = [lambda x: 1.0]
            self._free_parameters.add(angle)
            used_parameters = [angle]

        elif isinstance(angle, ParameterExpression):
            # Parameter is in a expression (equation)
            parameterized = True
            func_list_element = lambdify(self._symbol_tuple_circuit, _param_to_sympy(angle))
            func_grad_list_element = []
            used_parameters = []
            for param_element in _param_free_symbols(angle):
                self._free_parameters.add(param_element)
                used_parameters.append(param_element)
                # information about the gradient of the parameter expression
                param_grad = angle.gradient(param_element)
                if isinstance(param_grad, float):
                    # create a call by value labmda function
                    func_grad_list_element.append(lambda *arg, param_grad=param_grad: param_grad)
                else:
                    func_grad_list_element.append(
                        lambdify(self._symbol_tuple_circuit, _param_to_sympy(param_grad))
                    )
        else:
            raise TypeError(
                f"Unsupported type for angle: {type(angle)}. "
                "Expected float, int, ParameterVectorElement or ParameterExpression."
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
                raise ValueError(f"Qubit index is out of range")
            self._operation_list.append(gate_name)
            self._qubit_list.append([control, target])
            self._func_list.append(None)
            self._func_grad_list.append(None)
            self._used_parameters.append([])

        self._rebuild_circuit_func = True

    def _add_parameterized_single_qubit_gate(
        self,
        gate_name: str,
        qubits: int | Iterable[int],
        angle: ParameterVectorElement | ParameterExpression | float,
    ):
        """
        Adds a single qubit parameterized gate to the circuit.

        Args:
            gate_name (str): Name of the gate
            qubits (int or Iterable[int]): qubit indices
            angle (ParameterVectorElement or float): angle of rotation, ca be a parameter
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
        angle: ParameterVectorElement | ParameterExpression | float,
    ):
        """
        Adds a single qubit parameterized gate to the circuit.

        Args:
            gate_name (str): Name of the gate
            qubits (int or Iterable[int]): qubit indices
            angle (ParameterVectorElement or float): angle of rotation, ca be a parameter
        """
        func_list_element, func_grad_list_element, used_parameters, parameterized = (
            self._add_parameter_expression(angle)
        )

        qubit1 = [qubit1] if isinstance(qubit1, int) else qubit1
        qubit2 = [qubit2] if isinstance(qubit2, int) else qubit2

        for control, target in zip(qubit1, qubit2):
            if control >= self.num_qubits or target >= self.num_qubits:
                raise ValueError(f"Qubit index is out of range")
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

    def _build_circuit_instructions(self, circuit: QuantumCircuit) -> tuple:
        """
        Function to build the instructions for the Qulacs circuit from the Qiskit circuit.

        This functions converts the Qiskit gates and parameter expressions to Qulacs compatible
        gates and functions.

        Args:
            circuit (QuantumCircuit): Qiskit circuit to convert to Qulacs

        Returns:
            Tuple with lists of Qulacs gates, Qulacs gate parameter functions,
            Qulacs gate wires, Qulacs gate parameters and Qulacs gate parameter dimensions
        """

        self._operation_list = []
        self._param_list = []
        self._qubit_list = []
        self._func_list = []
        self._func_grad_list = []
        self._free_parameters = set()
        self._qulacs_gates_parameters = {}
        self._symbol_tuple_circuit = tuple()

        for param in circuit.parameters:
            name = param.vector.name
            if name not in self._qulacs_gates_parameters:
                self._qulacs_gates_parameters[name] = 1
            else:
                self._qulacs_gates_parameters[name] += 1

        self._symbol_tuple_circuit = tuple([_param_to_sympy(p) for p in circuit.parameters])

        for gate_operation in circuit.data:

            # catch conditions of the gate
            # only c_if is supported, the other cases have been caught before
            if (
                hasattr(gate_operation.operation, "condition")
                and gate_operation.operation.condition is not None
                or gate_operation.operation.name == "measure"
            ):
                raise NotImplementedError(
                    "Conditions are not supported in sQUlearn's Qulacs backend."
                )

            if (
                gate_operation.operation.name not in qiskit_qulacs_gate_dict
                and gate_operation.operation.name not in qiskit_qulacs_param_gate_dict
            ):
                raise NotImplementedError(
                    f"Gate {gate_operation.operation.name} is unfortunatly not supported in sQUlearn's Qulacs backend."
                )

            paramterized_gate = len(gate_operation.operation.params) >= 1
            single_qubit_date = len(gate_operation.qubits) == 1

            wires = [
                circuit.find_bit(gate_operation.qubits[i]).index
                for i in range(gate_operation.operation.num_qubits)
            ]

            if single_qubit_date:
                if not paramterized_gate:
                    self._add_single_qubit_gate(gate_operation.operation.name, wires)
                else:
                    self._add_parameterized_single_qubit_gate(
                        gate_operation.operation.name, wires, gate_operation.operation.params[0]
                    )
            else:
                if len(gate_operation.qubits) > 2:
                    raise NotImplementedError(
                        "Only two qubit gates are supported in sQUlearn's Qulacs backend."
                    )
                if not paramterized_gate:
                    self._add_two_qubit_gate(gate_operation.operation.name, wires[0], wires[1])
                else:
                    self._add_parameterized_two_qubit_gate(
                        gate_operation.operation.name,
                        wires[0],
                        wires[1],
                        gate_operation.operation.params[0],
                    )

    def get_circuit_func(self, gradient_param=None):
        """Returns the Qulacs circuit function for the circuit."""

        if isinstance(gradient_param, ParameterVectorElement):
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
        gradient_parameters: ParameterVectorElement | List[ParameterVectorElement] | None = None,
    ):
        """Returns the outer jacobian needed for the chain rule in circuit derivatives.

        Qulacs does not support multiple parameters and parameter expressions,
        so we need to calculate a transformation which also includes the gradient of the
        parameter expression.

        Args:
            gradient_parameters (ParameterVectorElement | List[ParameterVectorElement] | None): Parameters to calculate the gradient for
        """

        if isinstance(gradient_parameters, ParameterVectorElement):
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
