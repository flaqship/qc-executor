"""Pauli propagation native circuit datatype."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union

from executor.base.circuit_base import QuantumCircuitBase

from .gates import CliffordGate, Gate, LayerBarrier, PauliRotation


class PauliPropagationCircuit(QuantumCircuitBase):
    """Backend-native circuit representation for Pauli propagation.

    This datatype stores operations directly in the internal gate representation
    used by the propagation engine and does not depend on Qiskit objects.
    """

    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self._gates: List[Union[Gate, LayerBarrier]] = []
        self._parameter_names: List[str] = []
        self._parameter_name_set: set[str] = set()

    @property
    def gates(self) -> List[Union[Gate, LayerBarrier]]:
        """Return a shallow copy of gate instructions."""
        return list(self._gates)

    @property
    def parameters(self) -> List[str]:
        """Return parameter names used by the circuit."""
        return list(self._parameter_names)

    @property
    def num_parameters(self) -> int:
        return len(self._parameter_names)

    @property
    def is_parameterized(self) -> bool:
        return self.num_parameters > 0

    def draw(self) -> str:
        return "\n".join(str(g) for g in self._gates)

    def _record_parameter(self, parameter_name: str | None) -> None:
        if parameter_name is None:
            return
        if parameter_name not in self._parameter_name_set:
            self._parameter_name_set.add(parameter_name)
            self._parameter_names.append(parameter_name)

    @staticmethod
    def _extract_parameter(parameter: Any) -> tuple[str | None, float | None]:
        if isinstance(parameter, (int, float)):
            return None, float(parameter)

        if hasattr(parameter, "name"):
            return str(parameter.name), None

        expression_params = getattr(parameter, "parameters", None)
        if expression_params:
            expression_params = list(expression_params)
            if expression_params:
                first_parameter = expression_params[0]
                return str(getattr(first_parameter, "name", first_parameter)), None

        raise TypeError(f"Unsupported parameter type: {type(parameter)!r}")

    def h(self, qubits: Union[int, List[int]]):
        qubit_list = [qubits] if isinstance(qubits, int) else qubits
        for qubit in qubit_list:
            self._gates.append(CliffordGate("H", qubit, self._num_qubits))

    def s(self, qubits: Union[int, List[int]]):
        qubit_list = [qubits] if isinstance(qubits, int) else qubits
        for qubit in qubit_list:
            self._gates.append(CliffordGate("S", qubit, self._num_qubits))

    def sdag(self, qubits: Union[int, List[int]]):
        self.rz(qubits, -1.5707963267948966)

    def t(self, qubits: Union[int, List[int]]):
        qubit_list = [qubits] if isinstance(qubits, int) else qubits
        for qubit in qubit_list:
            self._gates.append(CliffordGate("T", qubit, self._num_qubits))

    def tdag(self, qubits: Union[int, List[int]]):
        self.rz(qubits, -0.7853981633974483)

    def p(self, qubits: Union[int, List[int]], angle: float):
        self.rz(qubits, angle)

    def x(self, qubits: Union[int, List[int]]):
        qubit_list = [qubits] if isinstance(qubits, int) else qubits
        for qubit in qubit_list:
            self._gates.append(CliffordGate("X", qubit, self._num_qubits))

    def y(self, qubits: Union[int, List[int]]):
        qubit_list = [qubits] if isinstance(qubits, int) else qubits
        for qubit in qubit_list:
            self._gates.append(CliffordGate("Y", qubit, self._num_qubits))

    def z(self, qubits: Union[int, List[int]]):
        qubit_list = [qubits] if isinstance(qubits, int) else qubits
        for qubit in qubit_list:
            self._gates.append(CliffordGate("Z", qubit, self._num_qubits))

    def rx(self, qubits: Union[int, List[int]], angle: float):
        qubit_list = [qubits] if isinstance(qubits, int) else qubits
        for qubit in qubit_list:
            param_name, param_value = self._extract_parameter(angle)
            self._record_parameter(param_name)
            self._gates.append(
                PauliRotation(
                    ["X"],
                    qubit,
                    self._num_qubits,
                    param_name=param_name,
                    param_value=param_value,
                )
            )

    def ry(self, qubits: Union[int, List[int]], angle: float):
        qubit_list = [qubits] if isinstance(qubits, int) else qubits
        for qubit in qubit_list:
            param_name, param_value = self._extract_parameter(angle)
            self._record_parameter(param_name)
            self._gates.append(
                PauliRotation(
                    ["Y"],
                    qubit,
                    self._num_qubits,
                    param_name=param_name,
                    param_value=param_value,
                )
            )

    def rz(self, qubits: Union[int, List[int]], angle: float):
        qubit_list = [qubits] if isinstance(qubits, int) else qubits
        for qubit in qubit_list:
            param_name, param_value = self._extract_parameter(angle)
            self._record_parameter(param_name)
            self._gates.append(
                PauliRotation(
                    ["Z"],
                    qubit,
                    self._num_qubits,
                    param_name=param_name,
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
        param_name, param_value = self._extract_parameter(angle)
        self._record_parameter(param_name)
        self._gates.append(
            PauliRotation(
                ["X", "X"],
                [control_qubit, target_qubit],
                self._num_qubits,
                param_name=param_name,
                param_value=param_value,
            )
        )

    def ryy(self, control_qubit: int, target_qubit: int, angle: float):
        param_name, param_value = self._extract_parameter(angle)
        self._record_parameter(param_name)
        self._gates.append(
            PauliRotation(
                ["Y", "Y"],
                [control_qubit, target_qubit],
                self._num_qubits,
                param_name=param_name,
                param_value=param_value,
            )
        )

    def rzz(self, control_qubit: int, target_qubit: int, angle: float):
        param_name, param_value = self._extract_parameter(angle)
        self._record_parameter(param_name)
        self._gates.append(
            PauliRotation(
                ["Z", "Z"],
                [control_qubit, target_qubit],
                self._num_qubits,
                param_name=param_name,
                param_value=param_value,
            )
        )

    def rzx(self, control_qubit: int, target_qubit: int, angle: float):
        raise NotImplementedError("RZX is not yet supported in PauliPropagationCircuit.")

    def swap(self, qubit1: int, qubit2: int):
        self._gates.append(CliffordGate("SWAP", [qubit1, qubit2], self._num_qubits))

    def barrier(self, qubits: Union[int, List[int]]):
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
                mapped._record_parameter(gate.param_name)
                mapped._gates.append(
                    PauliRotation(
                        list(gate.symbols),
                        remapped_qubits if len(remapped_qubits) > 1 else remapped_qubits[0],
                        mapped.num_qubits,
                        param_name=gate.param_name,
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

    def assign_parameters(self, parameters: Dict[str, float]):
        assigned = self.copy()
        new_gates: List[Union[Gate, LayerBarrier]] = []

        for gate in assigned._gates:
            if isinstance(gate, PauliRotation) and gate.param_name in parameters:
                new_gates.append(
                    PauliRotation(
                        list(gate.symbols),
                        gate.qubits if len(gate.qubits) > 1 else gate.qubits[0],
                        assigned.num_qubits,
                        param_name=None,
                        param_value=float(parameters[gate.param_name]),
                    )
                )
            else:
                new_gates.append(gate)

        assigned._gates = new_gates
        return assigned

    def invert(self) -> "PauliPropagationCircuit":
        inverse = PauliPropagationCircuit(self.num_qubits)
        inverse._parameter_names = list(self._parameter_names)
        inverse._parameter_name_set = set(self._parameter_name_set)

        for gate in reversed(self._gates):
            if isinstance(gate, LayerBarrier):
                inverse._gates.append(LayerBarrier())
            elif isinstance(gate, PauliRotation):
                angle = (
                    gate.param_name if gate.param_name is not None else -float(gate.param_value)
                )
                if gate.param_name is not None:
                    inverse._gates.append(
                        PauliRotation(
                            list(gate.symbols),
                            gate.qubits if len(gate.qubits) > 1 else gate.qubits[0],
                            self.num_qubits,
                            param_name=gate.param_name,
                            param_value=gate.param_value,
                        )
                    )
                else:
                    inverse._gates.append(
                        PauliRotation(
                            list(gate.symbols),
                            gate.qubits if len(gate.qubits) > 1 else gate.qubits[0],
                            self.num_qubits,
                            param_name=None,
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
        copied._parameter_names = list(self._parameter_names)
        copied._parameter_name_set = set(self._parameter_name_set)
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
                        param_name=gate.param_name,
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

    def __hash__(self):
        signature: Sequence[Any] = []
        items: List[Any] = []
        for gate in self._gates:
            if isinstance(gate, LayerBarrier):
                items.append(("BARRIER",))
            elif isinstance(gate, CliffordGate):
                items.append(("CLIFFORD", gate.gate_type, tuple(gate.qubits)))
            elif isinstance(gate, PauliRotation):
                items.append(
                    (
                        "ROT",
                        tuple(gate.symbols),
                        tuple(gate.qubits),
                        gate.param_name,
                        gate.param_value,
                    )
                )
        signature = tuple(items)
        return hash((self.num_qubits, signature))

    def __str__(self):
        return f"PauliPropagationCircuit(num_qubits={self.num_qubits}, gates={len(self._gates)})"

    def __repr__(self):
        return self.__str__()
