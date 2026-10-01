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

    @classmethod
    def from_qiskit(cls, qiskit_circuit: Any) -> "QuantumCircuit":
        """Import a Qiskit circuit into the framework-independent IR.

        The circuit's instructions are translated into the instruction store,
        so the result runs on every backend -- unlike
        :meth:`QiskitCircuit.from_qiskit <qc_executor.qiskit.QiskitCircuit.from_qiskit>`,
        which only carries a pre-built circuit for the Qiskit backend and is the
        right tool for ISA-transpiled circuits.  Requires the ``qiskit`` extra.

        Args:
            qiskit_circuit: The ``qiskit.QuantumCircuit`` to import.

        Returns:
            A new circuit holding the translated instructions.

        Raises:
            UnsupportedGateError: If the circuit is transpiled or contains an
                instruction the IR cannot express; see
                :func:`~qc_executor.qiskit._ir_bridge.qiskit_to_ir`.
        """
        # Imported lazily so the core package stays free of Qiskit.
        from .qiskit._ir_bridge import qiskit_to_ir  # pylint: disable=import-outside-toplevel

        ir = qiskit_to_ir(qiskit_circuit)
        return cls(ir.num_qubits, ir.num_clbits, _ir=ir)
