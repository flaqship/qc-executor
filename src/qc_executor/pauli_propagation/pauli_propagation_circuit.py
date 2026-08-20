"""Pauli propagation native circuit datatype."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import sympy as sp

from qc_executor.base.circuit_base import QuantumCircuitBase
from qc_executor.utils.qiskit_compat import _param_is_constant, _param_to_float, _param_to_sympy

from .utils.gates import CliffordGate, Gate, LayerBarrier, PauliRotation


def _qubit_arg(qubits: List[int]) -> int | List[int]:
    """Return the single qubit index or the full list, matching gate constructors."""
    return qubits if len(qubits) > 1 else qubits[0]


def _clone_gate(gate: Gate | LayerBarrier, num_qubits: int) -> Gate | LayerBarrier | None:
    """Create a fresh copy of a gate instruction (None for unknown gate types)."""
    if isinstance(gate, LayerBarrier):
        return LayerBarrier()
    if isinstance(gate, PauliRotation):
        return PauliRotation(
            list(gate.symbols),
            _qubit_arg(gate.qubits),
            num_qubits,
            param_expr=gate.param_expr,
            param_value=gate.param_value,
        )
    if isinstance(gate, CliffordGate):
        return CliffordGate(gate.gate_type, _qubit_arg(gate.qubits), num_qubits)
    return None


def _record_symbols(param_expression: sp.Expr | None, parameters: Dict[str, sp.Symbol]) -> None:
    """Record all free symbols of an expression in a name -> symbol mapping."""
    if param_expression is None:
        return
    for symbol in param_expression.free_symbols:
        parameters.setdefault(symbol.name, symbol)


class PauliPropagationCircuit(QuantumCircuitBase):
    """Backend-native circuit representation for Pauli propagation.

    This datatype stores operations directly in the internal gate representation
    used by the propagation engine and does not depend on Qiskit objects.
    """

    def __init__(
        self,
        num_qubits: int,
        *,
        gates: Sequence[Gate | LayerBarrier] | None = None,
        parameter_symbols: Dict[str, sp.Symbol] | None = None,
    ):
        super().__init__(num_qubits)
        self._gates: List[Gate | LayerBarrier] = list(gates) if gates is not None else []
        # Maps parameter names to sympy symbols
        self._parameters: Dict[str, sp.Symbol] = (
            dict(parameter_symbols) if parameter_symbols is not None else {}
        )

    @classmethod
    def from_quantum_circuit(cls, circuit: QuantumCircuitBase) -> "PauliPropagationCircuit":
        """Create a PauliPropagationCircuit from a generic circuit."""
        if isinstance(circuit, cls):
            return circuit

        if not hasattr(circuit, "qiskit_circuit"):
            raise TypeError(
                "PauliPropagationCircuit.from_quantum_circuit expects a generic QuantumCircuit "
                f"or {cls.__name__}, got {type(circuit).__name__}"
            )

        # pylint: disable-next=import-outside-toplevel
        from .utils.qiskit_converter import convert_circuit

        # The wrapped Qiskit circuit is exposed via the public qiskit_circuit
        # property (hasattr-checked above); getattr keeps the duck typing
        # opaque to static type checkers.
        gates = convert_circuit(getattr(circuit, "qiskit_circuit"), use_cache=True)

        parameters: Dict[str, sp.Symbol] = {}
        for gate in gates:
            if isinstance(gate, PauliRotation):
                _record_symbols(gate.param_expr, parameters)

        return cls(circuit.num_qubits, gates=gates, parameter_symbols=parameters)

    @property
    def gates(self) -> List[Gate | LayerBarrier]:
        """Return a shallow copy of gate instructions."""
        return list(self._gates)

    @property
    def parameters(self) -> List[str]:
        """Return parameter names used by the circuit."""
        return list(self._parameters.keys())

    @property
    def parameter_symbols(self) -> Dict[str, sp.Symbol]:
        """Return a copy of the parameter-name to sympy-symbol mapping."""
        return dict(self._parameters)

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
        _record_symbols(param_expression, self._parameters)

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
            # Convert to sympy expression
            return _param_to_sympy(parameter), None

        # Handle direct sympy expressions
        if isinstance(parameter, sp.Expr):
            if parameter.is_number:
                return None, float(parameter)
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
        self.rz(qubits, -np.pi / 2)

    def t(self, qubits: int | List[int]):
        # T equals RZ(π/4) up to a global phase, which cancels in Heisenberg
        # conjugation; the rotation is exact (T is not a Clifford gate)
        self.rz(qubits, np.pi / 4)

    def tdag(self, qubits: int | List[int]):
        self.rz(qubits, -np.pi / 4)

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

        merged_gates = [
            cloned
            for cloned in (_clone_gate(gate, self.num_qubits) for gate in self._gates)
            if cloned is not None
        ]
        merged_parameters = dict(self._parameters)
        qubit_map = dict(enumerate(qubits))

        for gate in qc.gates:
            if isinstance(gate, LayerBarrier):
                merged_gates.append(LayerBarrier())
                continue

            remapped_qubits = [qubit_map[q] for q in gate.qubits]
            if isinstance(gate, PauliRotation):
                _record_symbols(gate.param_expr, merged_parameters)
                merged_gates.append(
                    PauliRotation(
                        list(gate.symbols),
                        _qubit_arg(remapped_qubits),
                        self.num_qubits,
                        param_expr=gate.param_expr,
                        param_value=gate.param_value,
                    )
                )
            elif isinstance(gate, CliffordGate):
                merged_gates.append(
                    CliffordGate(gate.gate_type, _qubit_arg(remapped_qubits), self.num_qubits)
                )

        return PauliPropagationCircuit(
            self.num_qubits, gates=merged_gates, parameter_symbols=merged_parameters
        )

    def assign_parameters(self, parameters: Dict[str, float]) -> "PauliPropagationCircuit":
        """Bind symbolic parameters to concrete values.

        Args:
            parameters: Dict mapping parameter names to float values

        Returns:
            New circuit with parameters substituted
        """
        new_gates: List[Gate | LayerBarrier] = []

        # Build substitution dict for sympy
        subs_dict = {}
        for param_name, param_value in parameters.items():
            if param_name in self._parameters:
                subs_dict[self._parameters[param_name]] = param_value

        for gate in self._gates:
            if isinstance(gate, PauliRotation) and gate.param_expr is not None:
                # Substitute parameters in the expression
                substituted_expr = gate.param_expr.subs(subs_dict)

                # If fully evaluated, convert to concrete value
                if substituted_expr.is_number:
                    new_gates.append(
                        PauliRotation(
                            list(gate.symbols),
                            _qubit_arg(gate.qubits),
                            self.num_qubits,
                            param_expr=None,
                            param_value=float(substituted_expr),
                        )
                    )
                else:
                    # Partially evaluated, keep as expression
                    new_gates.append(
                        PauliRotation(
                            list(gate.symbols),
                            _qubit_arg(gate.qubits),
                            self.num_qubits,
                            param_expr=substituted_expr,
                            param_value=None,
                        )
                    )
            else:
                cloned = _clone_gate(gate, self.num_qubits)
                if cloned is not None:
                    new_gates.append(cloned)

        # Update parameter tracking - remove fully bound parameters
        remaining_params = {
            name: symbol for name, symbol in self._parameters.items() if name not in parameters
        }

        return PauliPropagationCircuit(
            self.num_qubits, gates=new_gates, parameter_symbols=remaining_params
        )

    def invert(self) -> "PauliPropagationCircuit":
        inverted_gates: List[Gate | LayerBarrier] = []

        for gate in reversed(self._gates):
            if isinstance(gate, LayerBarrier):
                inverted_gates.append(LayerBarrier())
            elif isinstance(gate, PauliRotation):
                if gate.param_expr is not None:
                    # Negate the symbolic expression
                    inverted_gates.append(
                        PauliRotation(
                            list(gate.symbols),
                            _qubit_arg(gate.qubits),
                            self.num_qubits,
                            param_expr=-gate.param_expr,
                            param_value=None,
                        )
                    )
                else:
                    # Negate the concrete value
                    inverted_gates.append(
                        PauliRotation(
                            list(gate.symbols),
                            _qubit_arg(gate.qubits),
                            self.num_qubits,
                            param_expr=None,
                            param_value=-float(gate.param_value),
                        )
                    )
            elif isinstance(gate, CliffordGate):
                inverted_gates.append(
                    CliffordGate(gate.gate_type, _qubit_arg(gate.qubits), self.num_qubits)
                )

        return PauliPropagationCircuit(
            self.num_qubits, gates=inverted_gates, parameter_symbols=self._parameters
        )

    def copy(self) -> "PauliPropagationCircuit":
        copied_gates = [
            cloned
            for cloned in (_clone_gate(gate, self.num_qubits) for gate in self._gates)
            if cloned is not None
        ]
        return PauliPropagationCircuit(
            self.num_qubits, gates=copied_gates, parameter_symbols=self._parameters
        )

    def replace_gate(self, index: int, gate: Gate | LayerBarrier) -> "PauliPropagationCircuit":
        """Return a new circuit with the gate at ``index`` replaced.

        Unreplaced gate objects are shared with this circuit; gates are
        immutable after construction, so this is safe.
        """
        new_gates = list(self._gates)
        new_gates[index] = gate
        return PauliPropagationCircuit(
            self.num_qubits, gates=new_gates, parameter_symbols=self._parameters
        )

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
