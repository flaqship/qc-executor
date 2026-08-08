"""Qiskit native operator, compiled from the shared sparse Pauli representation."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, List

from qc_executor.base.operator_base import QuantumOperatorBase
from qc_executor.base.operator_ir import PauliIR
from qc_executor.qiskit._ir_bridge import pauli_ir_to_sparse_pauli_op, sparse_pauli_op_to_pauli_ir
from qc_executor.qiskit._param_binding import build_params_dict


class QiskitOperator(QuantumOperatorBase):
    """An observable that compiles to Qiskit's ``SparsePauliOp``.

    Built like any other operator -- ``QiskitOperator(["ZI"], [1.0])`` -- or
    converted from an existing one with :meth:`from_quantum_operator`.  The
    ``SparsePauliOp`` is produced on demand, so the whole inherited algebra
    (:meth:`compose`, :meth:`adjoint`, :meth:`apply_layout`, ...) is available
    and stays in this abstraction.

    Args:
        paulis: Pauli labels, qubit 0 leftmost.
        coeffs: One coefficient per label.
        num_qubits: Width, required only when no labels are given.
        _ir: Adopt this representation instead of building one.
    """

    def _build_native(self):
        """Compile the representation into a Qiskit ``SparsePauliOp``."""
        return pauli_ir_to_sparse_pauli_op(self._ir)

    @classmethod
    def from_quantum_operator(cls, operator: Any, **options: Any) -> "QiskitOperator":
        """Convert an operator, or import a ``SparsePauliOp`` passed directly.

        Args:
            operator: A :class:`~qc_executor.base.operator_base.QuantumOperatorBase`
                or a raw ``qiskit.quantum_info.SparsePauliOp``.
            ``**options``: Accepted for interface parity; unused.

        Returns:
            The operator in this native form.
        """
        if not isinstance(operator, QuantumOperatorBase):
            return cls(_ir=sparse_pauli_op_to_pauli_ir(operator))
        return super().from_quantum_operator(operator, **options)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def qiskit_operator(self):
        """The underlying Qiskit ``SparsePauliOp``."""
        return self.native

    @property
    def free_parameters(self) -> set:
        """The Qiskit parameter objects appearing in the compiled operator."""
        return set(self.native.parameters)

    @property
    def parameter_dimensions(self) -> dict:
        """Number of elements used from each Qiskit parameter vector."""
        dimensions: OrderedDict = OrderedDict()
        for parameter in self.native.parameters:
            name = parameter.vector.name if hasattr(parameter, "vector") else parameter.name
            dimensions[name] = dimensions.get(name, 0) + 1
        return dict(dimensions)

    @property
    def parameter_names(self) -> List[str]:
        """Names of the Qiskit parameter vectors used by the operator."""
        return list(self.parameter_dimensions)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def bind_parameters(self, parameter_values: dict):
        """Bind values and return the resulting **Qiskit** operator.

        Distinct from :meth:`assign_parameters`, which stays in this
        abstraction and returns a :class:`QiskitOperator`.

        Args:
            parameter_values: Values keyed by parameter name.

        Returns:
            The bound ``SparsePauliOp``.
        """
        params_dict = build_params_dict(self.free_parameters, parameter_values)
        if params_dict:
            return self.native.assign_parameters(params_dict)
        return self.native

    def _rebuild(self, ir: PauliIR) -> "QiskitOperator":
        """Wrap a new representation in this operator's type."""
        return type(self)(_ir=ir)

    def __repr__(self):
        return f"QiskitOperator({self.num_qubits} qubits, {len(self.free_parameters)} parameters)"
