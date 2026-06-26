"""Parameter and Parameters types for parameterised quantum circuits."""

from __future__ import annotations

from qiskit.circuit.parametervector import ParameterVector as Parameters
from qiskit.circuit.parametervector import ParameterVectorElement as Parameter

__all__ = ["Parameter", "Parameters"]
