"""Qrisp operator wrapper backed by the shared Pauli representation."""

from __future__ import annotations

from typing import Any

from ..base.operator_base import QuantumOperatorBase


class QrispOperator(QuantumOperatorBase):
    """An observable represented by Executor's shared ``PauliIR``."""

    @classmethod
    def from_quantum_operator(cls, operator: Any, **options: Any) -> "QrispOperator":
        if not isinstance(operator, QuantumOperatorBase):
            raise TypeError("QrispOperator expects a QuantumOperatorBase instance")
        return super().from_quantum_operator(operator, **options)  # type: ignore[return-value]

    def _build_native(self):
        return self.ir

    def _rebuild(self, ir):
        return type(self)(_ir=ir)