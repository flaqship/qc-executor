"""The framework-independent quantum operator.

``QuantumOperator`` is a thin leaf on
:class:`~qc_executor.base.operator_base.QuantumOperatorBase`: the sparse Pauli
representation, the algebra and the parameter handling all live in the base and
are shared verbatim with every backend's native operator type.
"""

from __future__ import annotations

from typing import Any

from .base import QuantumOperatorBase

__all__ = ["QuantumOperator"]


class QuantumOperator(QuantumOperatorBase):
    """A weighted sum of Pauli strings, not tied to any quantum framework.

    Qubit ``q`` is character ``q`` of the label, so ``QuantumOperator(["ZI"],
    [1.0])`` measures Z on qubit 0.

    Args:
        paulis: Pauli labels making up the operator.
        coeffs: One coefficient per label; numbers or SymPy expressions.
        num_qubits: Width, required only when no labels are given.
    """

    @property
    def qiskit_operator(self) -> Any:
        """The operator translated to Qiskit.

        Provided so backends that still consume Qiskit objects keep working
        while they are migrated onto the shared representation.  Requires the
        ``qiskit`` extra.

        Returns:
            The equivalent ``qiskit.quantum_info.SparsePauliOp``.
        """
        # Imported lazily so the core package stays free of Qiskit.
        from .qiskit._ir_bridge import (  # pylint: disable=import-outside-toplevel
            pauli_ir_to_sparse_pauli_op,
        )

        return pauli_ir_to_sparse_pauli_op(self._ir)
