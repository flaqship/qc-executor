"""Qiskit native circuit, compiled from the framework-independent circuit IR."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any, List

from qc_executor.base.circuit_base import QuantumCircuitBase
from qc_executor.base.circuit_ir import CircuitIR
from qc_executor.qiskit._hash import _circuit_key
from qc_executor.qiskit._ir_bridge import SUPPORTED_OPCODES, ir_to_qiskit
from qc_executor.qiskit._param_binding import build_params_dict


class QiskitCircuit(QuantumCircuitBase):
    """A quantum circuit that compiles to Qiskit.

    Built like any other circuit -- ``QiskitCircuit(2)`` then ``circuit.h(0)``
    -- or converted from an existing one with :meth:`from_quantum_circuit`.
    The Qiskit circuit is produced on demand and rebuilt whenever the
    instruction store changes.

    A pre-built Qiskit circuit can also be adopted through :meth:`from_qiskit`,
    which stores it directly instead of deriving it; see that method for why.

    Args:
        num_qubits: Number of qubits in the circuit.
        num_clbits: Number of classical bits, for mid-circuit measurement.
        _ir: Adopt this instruction store instead of starting empty.
        _native: Adopt this Qiskit circuit instead of compiling one.
    """

    @classmethod
    def supported_opcodes(cls) -> frozenset:
        """Return the opcodes the Qiskit bridge emits directly."""
        return SUPPORTED_OPCODES

    def __init__(
        self,
        num_qubits: int = 0,
        num_clbits: int = 0,
        *,
        _ir: "CircuitIR | None" = None,
        _native: Any = None,
    ):
        super().__init__(num_qubits, num_clbits, _ir=_ir)
        self._native_override = _native

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------

    def _build_native(self):
        """Compile the instruction store into a Qiskit circuit."""
        if self._native_override is not None:
            return self._native_override
        return ir_to_qiskit(self._lowered_ir())

    @classmethod
    def from_quantum_circuit(cls, circuit: Any) -> "QiskitCircuit":
        """Convert a circuit, or adopt a Qiskit circuit passed directly.

        Args:
            circuit: A :class:`~qc_executor.base.circuit_base.QuantumCircuitBase`
                or a raw ``qiskit.QuantumCircuit``.

        Returns:
            The circuit in this native form.
        """
        if not isinstance(circuit, QuantumCircuitBase):
            return cls.from_qiskit(circuit)
        return super().from_quantum_circuit(circuit)  # type: ignore[return-value]

    @classmethod
    def from_qiskit(cls, qiskit_circuit) -> "QiskitCircuit":
        """Adopt an existing Qiskit circuit without deriving it from the IR.

        The circuit is stored as-is rather than imported, because the ones that
        arrive here cannot be expressed in the IR: an ISA-transpiled circuit
        carries a global phase, a ``TranspileLayout``, possibly ``Delay`` and
        ``IfElseOp`` instructions, and is widened to the backend's full qubit
        count.  Round-tripping it would silently discard all of that, so the
        instruction store is left empty and this object is a carrier for the
        compiled circuit.

        Args:
            qiskit_circuit: The Qiskit circuit to adopt.

        Returns:
            A :class:`QiskitCircuit` wrapping it.
        """
        return cls(
            qiskit_circuit.num_qubits,
            qiskit_circuit.num_clbits,
            _native=qiskit_circuit,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def qiskit_circuit(self):
        """The underlying Qiskit circuit."""
        return self.native

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the circuit."""
        if self._native_override is not None:
            return self._native_override.num_qubits
        return super().num_qubits

    @property
    def free_parameters(self) -> set:
        """The Qiskit parameter objects appearing in the compiled circuit."""
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
        """Names of the Qiskit parameter vectors used by the circuit."""
        return list(self.parameter_dimensions)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def bind_parameters(self, parameter_values: dict):
        """Bind values and return the resulting **Qiskit** circuit.

        Distinct from :meth:`assign_parameters`, which stays in this
        abstraction and returns a :class:`QiskitCircuit`.

        Args:
            parameter_values: Values keyed by parameter name.

        Returns:
            The bound ``qiskit.QuantumCircuit``.
        """
        params_dict = build_params_dict(self.free_parameters, parameter_values)
        if params_dict:
            return self.native.assign_parameters(params_dict)
        return self.native

    def copy(self) -> "QiskitCircuit":
        """Return an independent copy of this circuit."""
        if self._native_override is not None:
            return type(self).from_qiskit(self._native_override.copy())
        return type(self)(self.num_qubits, self.num_clbits, _ir=self._ir.copy())

    def fingerprint(self) -> bytes:
        """Return a stable digest of the circuit's content."""
        if self._native_override is not None:
            # An adopted circuit has no instruction store to digest, so the
            # digest comes from the Qiskit content key.  It has to be content
            # based: repr() on a Qiskit circuit includes the object address,
            # which would give equal circuits different hashes.
            return hashlib.blake2b(
                repr(_circuit_key(self._native_override)).encode("utf-8"), digest_size=32
            ).digest()
        return super().fingerprint()

    def __hash__(self) -> int:
        return hash(self.fingerprint())

    def __eq__(self, other: Any) -> bool:
        if type(self) is not type(other):
            return False
        if (self._native_override is None) != (other._native_override is None):
            return False
        if self._native_override is not None:
            return self._native_override == other._native_override
        return self._ir == other.ir

    def __repr__(self):
        return f"QiskitCircuit({self.num_qubits} qubits, {len(self.free_parameters)} parameters)"
