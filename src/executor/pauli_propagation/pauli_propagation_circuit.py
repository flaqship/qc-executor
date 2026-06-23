"""Pauli propagation native circuit datatype."""

from __future__ import annotations

from typing import Any, Dict, List

import sympy as sp

from executor.base.circuit_base import QuantumCircuitBase
from executor.utils.qiskit_compat import _param_is_constant, _param_to_float, _param_to_sympy

from .utils.gates import CliffordGate, Gate, LayerBarrier, PauliRotation


class PauliPropagationCircuit(QuantumCircuitBase):
    """Backend-native circuit representation for Pauli propagation.

    This datatype stores operations directly in the internal gate representation
    used by the propagation engine and does not depend on Qiskit objects.
    """

    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self._gates: List[Gate | LayerBarrier] = []
        self._parameters: Dict[str, sp.Symbol] = {}  # Maps parameter names to sympy symbols

    @classmethod
    def from_quantum_circuit(cls, circuit: QuantumCircuitBase) -> "PauliPropagationCircuit":
        """Create a PauliPropagationCircuit from a generic circuit."""
        if isinstance(circuit, cls):
            return circuit

        if not hasattr(circuit, "_qiskit_circuit"):
            raise TypeError(
                "PauliPropagationCircuit.from_quantum_circuit expects a generic QuantumCircuit "
                f"or {cls.__name__}, got {type(circuit).__name__}"
            )

        from .utils.qiskit_converter import convert_circuit

        pp_circuit = cls(circuit.num_qubits)
        pp_circuit._gates = convert_circuit(circuit._qiskit_circuit, use_cache=True)

        for gate in pp_circuit._gates:
            if isinstance(gate, PauliRotation):
                pp_circuit._record_parameter(gate.param_expr)

        return pp_circuit

    @property
    def gates(self) -> List[Gate | LayerBarrier]:
        """Return a shallow copy of gate instructions."""
        return list(self._gates)

    @property
    def parameters(self) -> List[str]:
        """Return parameter names used by the circuit."""
        return list(self._parameters.keys())

    @property
    def num_parameters(self) -> int:
        return len(self._parameters)

    @property
    def is_parameterized(self) -> bool:
        return self.num_parameters > 0

    def draw(self) -> str:
        return "\n".join(str(g) for g in self._gates)

    def _record_parameter(self, param_expression: sp.Expr | None) -> None:
        """Record sympy symbols from a parameter expression."""
        if param_expression is None:
            return
        # Extract all free symbols from the expression
        for symbol in param_expression.free_symbols:
            if symbol.name not in self._parameters:
                self._parameters[symbol.name] = symbol

    @staticmethod
    def _extract_parameter(parameter: Any) -> tuple[sp.Expr | None, float | None]:
        """Extract parameter as sympy expression or concrete float value.

        Returns:
            (sympy_expr, None) for parametric gates
            (None, float_value) for concrete gates
        """
        if isinstance(parameter, (int, float)):
            return None, float(parameter)

        # Handle Qiskit ParameterExpression
        if hasattr(parameter, "parameters"):
            if _param_is_constant(parameter):
                return None, _param_to_float(parameter)
            else:
                # Convert to sympy expression
                return _param_to_sympy(parameter), None

        # Handle direct sympy expressions
        if isinstance(parameter, sp.Expr):
            if parameter.is_number:
                return None, float(parameter)
            else:
                return parameter, None

        raise TypeError(f"Unsupported parameter type: {type(parameter)!r}")

    def h(self, qubits: int | List[int]):
        qubit_list = [qubits] if isinstance(qubits, int) else qubits
        for qubit in qubit_list:
            self._gates.append(CliffordGate("H", qubit, self._num_qubits))

    def s(self, qubits: int | List[int]):
        qubit_list = [qubits] if isinstance(qubits, int) else qubits
        for qubit in qubit_list:
            self._gates.append(CliffordGate("S", qubit, self._num_qubits))

    def sdag(self, qubits: int | List[int]):
        self.rz(qubits, -1.5707963267948966)

    def t(self, qubits: int | List[int]):
        qubit_list = [qubits] if isinstance(qubits, int) else qubits
        for qubit in qubit_list:
            self._gates.append(CliffordGate("T", qubit, self._num_qubits))

    def tdag(self, qubits: int | List[int]):
        self.rz(qubits, -0.7853981633974483)

    def p(self, qubits: int | List[int], angle: float):
        self.rz(qubits, angle)

    def x(self, qubits: int | List[int]):
        qubit_list = [qubits] if isinstance(qubits, int) else qubits
        for qubit in qubit_list:
            self._gates.append(CliffordGate("X", qubit, self._num_qubits))

    def y(self, qubits: int | List[int]):
        qubit_list = [qubits] if isinstance(qubits, int) else qubits
        for qubit in qubit_list:
            self._gates.append(CliffordGate("Y", qubit, self._num_qubits))

    def z(self, qubits: int | List[int]):
        qubit_list = [qubits] if isinstance(qubits, int) else qubits
        for qubit in qubit_list:
            self._gates.append(CliffordGate("Z", qubit, self._num_qubits))

    def rx(self, qubits: int | List[int], angle: float):
        qubit_list = [qubits] if isinstance(qubits, int) else qubits
        for qubit in qubit_list:
            param_expr, param_value = self._extract_parameter(angle)
            self._record_parameter(param_expr)
            self._gates.append(
                PauliRotation(
                    ["X"],
                    qubit,
                    self._num_qubits,
                    param_expr=param_expr,
                    param_value=param_value,
                )
            )

    def ry(self, qubits: int | List[int], angle: float):
        qubit_list = [qubits] if isinstance(qubits, int) else qubits
        for qubit in qubit_list:
            param_expr, param_value = self._extract_parameter(angle)
            self._record_parameter(param_expr)
            self._gates.append(
                PauliRotation(
                    ["Y"],
                    qubit,
                    self._num_qubits,
                    param_expr=param_expr,
                    param_value=param_value,
                )
            )

    def rz(self, qubits: int | List[int], angle: float):
        qubit_list = [qubits] if isinstance(qubits, int) else qubits
        for qubit in qubit_list:
            param_expr, param_value = self._extract_parameter(angle)
            self._record_parameter(param_expr)
            self._gates.append(
                PauliRotation(
                    ["Z"],
                    qubit,
                    self._num_qubits,
                    param_expr=param_expr,
                    param_value=param_value,
                )
            )

    def cx(self, control_qubit: int, target_qubit: int):
        self._gates.append(CliffordGate("CNOT", [control_qubit, target_qubit], self._num_qubits))

    def cy(self, control_qubit: int, target_qubit: int):
        self.sdag(target_qubit)
        self.cx(control_qubit, target_qubit)
        self.s(target_qubit)

    def cz(self, control_qubit: int, target_qubit: int):
        self._gates.append(CliffordGate("CZ", [control_qubit, target_qubit], self._num_qubits))

    def crx(self, control_qubit: int, target_qubit: int, angle: float):
        raise NotImplementedError("CRX is not yet supported in PauliPropagationCircuit.")

    def cry(self, control_qubit: int, target_qubit: int, angle: float):
        raise NotImplementedError("CRY is not yet supported in PauliPropagationCircuit.")

    def crz(self, control_qubit: int, target_qubit: int, angle: float):
        raise NotImplementedError("CRZ is not yet supported in PauliPropagationCircuit.")

    def rxx(self, control_qubit: int, target_qubit: int, angle: float):
        param_expr, param_value = self._extract_parameter(angle)
        self._record_parameter(param_expr)
        self._gates.append(
            PauliRotation(
                ["X", "X"],
                [control_qubit, target_qubit],
                self._num_qubits,
                param_expr=param_expr,
                param_value=param_value,
            )
        )

    def ryy(self, control_qubit: int, target_qubit: int, angle: float):
        param_expr, param_value = self._extract_parameter(angle)
        self._record_parameter(param_expr)
        self._gates.append(
            PauliRotation(
                ["Y", "Y"],
                [control_qubit, target_qubit],
                self._num_qubits,
                param_expr=param_expr,
                param_value=param_value,
            )
        )

    def rzz(self, control_qubit: int, target_qubit: int, angle: float):
        param_expr, param_value = self._extract_parameter(angle)
        self._record_parameter(param_expr)
        self._gates.append(
            PauliRotation(
                ["Z", "Z"],
                [control_qubit, target_qubit],
                self._num_qubits,
                param_expr=param_expr,
                param_value=param_value,
            )
        )

    def rzx(self, control_qubit: int, target_qubit: int, angle: float):
        raise NotImplementedError("RZX is not yet supported in PauliPropagationCircuit.")

    def swap(self, qubit1: int, qubit2: int):
        self._gates.append(CliffordGate("SWAP", [qubit1, qubit2], self._num_qubits))

    def barrier(self, qubits: int | List[int]):
        self._gates.append(LayerBarrier())

    def measure(self):
        raise NotImplementedError("Measurement is not represented in PauliPropagationCircuit.")

    def compose(self, qc: "QuantumCircuitBase", qubits: List[int]) -> "PauliPropagationCircuit":
        if not isinstance(qc, PauliPropagationCircuit):
            raise TypeError("compose currently supports PauliPropagationCircuit only.")
        if len(qubits) != qc.num_qubits:
            raise ValueError("Length of qubits mapping must match composed circuit qubit count.")

        mapped = self.copy()
        qubit_map = {source: target for source, target in enumerate(qubits)}

        for gate in qc.gates:
            if isinstance(gate, LayerBarrier):
                mapped._gates.append(LayerBarrier())
                continue

            remapped_qubits = [qubit_map[q] for q in gate.qubits]
            if isinstance(gate, PauliRotation):
                mapped._record_parameter(gate.param_expr)
                mapped._gates.append(
                    PauliRotation(
                        list(gate.symbols),
                        remapped_qubits if len(remapped_qubits) > 1 else remapped_qubits[0],
                        mapped.num_qubits,
                        param_expr=gate.param_expr,
                        param_value=gate.param_value,
                    )
                )
            elif isinstance(gate, CliffordGate):
                mapped._gates.append(
                    CliffordGate(
                        gate.gate_type,
                        remapped_qubits if len(remapped_qubits) > 1 else remapped_qubits[0],
                        mapped.num_qubits,
                    )
                )

        return mapped

    def assign_parameters(self, parameters: Dict[str, float]) -> "PauliPropagationCircuit":
        """Bind symbolic parameters to concrete values.

        Args:
            parameters: Dict mapping parameter names to float values

        Returns:
            New circuit with parameters substituted
        """
        assigned = self.copy()
        new_gates: List[Gate | LayerBarrier] = []

        # Build substitution dict for sympy
        subs_dict = {}
        for param_name, param_value in parameters.items():
            if param_name in assigned._parameters:
                subs_dict[assigned._parameters[param_name]] = param_value

        for gate in assigned._gates:
            if isinstance(gate, PauliRotation) and gate.param_expr is not None:
                # Substitute parameters in the expression
                substituted_expr = gate.param_expr.subs(subs_dict)

                # If fully evaluated, convert to concrete value
                if substituted_expr.is_number:
                    new_gates.append(
                        PauliRotation(
                            list(gate.symbols),
                            gate.qubits if len(gate.qubits) > 1 else gate.qubits[0],
                            assigned.num_qubits,
                            param_expr=None,
                            param_value=float(substituted_expr),
                        )
                    )
                else:
                    # Partially evaluated, keep as expression
                    new_gates.append(
                        PauliRotation(
                            list(gate.symbols),
                            gate.qubits if len(gate.qubits) > 1 else gate.qubits[0],
                            assigned.num_qubits,
                            param_expr=substituted_expr,
                            param_value=None,
                        )
                    )
            else:
                new_gates.append(gate)

        assigned._gates = new_gates

        # Update parameter tracking - remove fully bound parameters
        remaining_params = {}
        for param_name, param_symbol in assigned._parameters.items():
            if param_name not in parameters:
                remaining_params[param_name] = param_symbol
        assigned._parameters = remaining_params

        return assigned

    def invert(self) -> "PauliPropagationCircuit":
        inverse = PauliPropagationCircuit(self.num_qubits)
        inverse._parameters = dict(self._parameters)

        for gate in reversed(self._gates):
            if isinstance(gate, LayerBarrier):
                inverse._gates.append(LayerBarrier())
            elif isinstance(gate, PauliRotation):
                if gate.param_expr is not None:
                    # Negate the symbolic expression
                    inverse._gates.append(
                        PauliRotation(
                            list(gate.symbols),
                            gate.qubits if len(gate.qubits) > 1 else gate.qubits[0],
                            self.num_qubits,
                            param_expr=-gate.param_expr,
                            param_value=None,
                        )
                    )
                else:
                    # Negate the concrete value
                    inverse._gates.append(
                        PauliRotation(
                            list(gate.symbols),
                            gate.qubits if len(gate.qubits) > 1 else gate.qubits[0],
                            self.num_qubits,
                            param_expr=None,
                            param_value=-float(gate.param_value),
                        )
                    )
            elif isinstance(gate, CliffordGate):
                inverse._gates.append(
                    CliffordGate(
                        gate.gate_type,
                        gate.qubits if len(gate.qubits) > 1 else gate.qubits[0],
                        self.num_qubits,
                    )
                )

        return inverse

    def copy(self) -> "PauliPropagationCircuit":
        copied = PauliPropagationCircuit(self.num_qubits)
        copied._parameters = dict(self._parameters)
        copied._gates = []

        for gate in self._gates:
            if isinstance(gate, LayerBarrier):
                copied._gates.append(LayerBarrier())
            elif isinstance(gate, PauliRotation):
                copied._gates.append(
                    PauliRotation(
                        list(gate.symbols),
                        gate.qubits if len(gate.qubits) > 1 else gate.qubits[0],
                        copied.num_qubits,
                        param_expr=gate.param_expr,
                        param_value=gate.param_value,
                    )
                )
            elif isinstance(gate, CliffordGate):
                copied._gates.append(
                    CliffordGate(
                        gate.gate_type,
                        gate.qubits if len(gate.qubits) > 1 else gate.qubits[0],
                        copied.num_qubits,
                    )
                )

        return copied

    def circuit_metrics(self) -> dict:
        gate_count = sum(1 for gate in self._gates if isinstance(gate, Gate))
        depth = gate_count
        return {
            "num_qubits": self.num_qubits,
            "num_gates": gate_count,
            "depth": depth,
            "num_parameters": self.num_parameters,
        }

    def from_qasm(self, qasm: str) -> None:
        raise NotImplementedError("QASM import is intentionally not supported for this datatype.")

    def to_qasm(self) -> str:
        raise NotImplementedError("QASM export is intentionally not supported for this datatype.")

    def __str__(self):
        return f"PauliPropagationCircuit(num_qubits={self.num_qubits}, gates={len(self._gates)})"

    def __repr__(self):
        return self.__str__()
