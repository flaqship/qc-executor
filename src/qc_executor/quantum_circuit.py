"""The framework-independent quantum circuit.

``QuantumCircuit`` is a thin leaf on :class:`~qc_executor.base.circuit_base.
QuantumCircuitBase`: the entire builder API, the parameter handling and the
Pauli-evolution helpers live in the base and are shared verbatim with every
backend's native circuit type.  This class adds no representation of its own —
it is simply the circuit that has not been compiled for any backend yet.
"""

from __future__ import annotations

from typing import Any

from .base import QuantumCircuitBase

__all__ = ["QuantumCircuit"]


class QuantumCircuit(QuantumCircuitBase):
    """A quantum circuit that is not tied to any quantum framework.

    Args:
        num_qubits: Number of qubits in the circuit.
        num_clbits: Number of classical bits for mid-circuit measurement.
    """

    @property
    def qiskit_circuit(self) -> Any:
        """The circuit compiled to Qiskit.

        Provided so backends that still consume Qiskit objects keep working
        while they are migrated onto the IR.  Requires the ``qiskit`` extra.

        Returns:
            The equivalent ``qiskit.QuantumCircuit``.
        """
        # Imported lazily so the core package stays free of Qiskit.
        from .qiskit._ir_bridge import ir_to_qiskit  # pylint: disable=import-outside-toplevel

        return ir_to_qiskit(self._ir)
