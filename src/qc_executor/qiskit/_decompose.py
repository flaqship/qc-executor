"""Utilities for decomposing Qiskit circuits into standard gate sets."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import standard_gates


def decompose_to_std(circuit: QuantumCircuit, gate_list: list | None = None) -> QuantumCircuit:
    """Decompose a circuit until only gates from the allowed set remain.

    Args:
        circuit (QuantumCircuit): Circuit to decompose.
        gate_list (list | None, optional): List of gate names considered
            standard. Gates not in this list are decomposed recursively. If
            ``None`` or empty, defaults to gates from
            ``qiskit.circuit.library.standard_gates`` plus ``cx``, ``cy``,
            ``cz``, and ``measure``.

    Returns:
        QuantumCircuit: The decomposed circuit.
    """
    if not gate_list:
        gate_list = [*dir(standard_gates), "cx", "cy", "cz", "measure"]
    decompose_names = [
        instruction.operation.name
        for instruction in circuit.data
        if instruction.operation.name not in gate_list
    ]
    circuit_new = circuit.decompose(decompose_names)

    while decompose_names and circuit != circuit_new:
        circuit = circuit_new
        decompose_names = [
            instruction.operation.name
            for instruction in circuit.data
            if instruction.operation.name not in gate_list
        ]
        circuit_new = circuit.decompose(decompose_names)

    return circuit_new
