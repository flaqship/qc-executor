"""Base classes for quantum circuits across different quantum frameworks."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.quantum_info import SparsePauliOp

from .operator_base import QuantumOperatorBase


class QuantumCircuitBase(ABC):
    """
    Base class for quantum circuits for different quantum frameworks.

    Args:
        num_qubits (int): Number of qubits in the circuit
    """

    def __init__(self, num_qubits: int):
        self._num_qubits = num_qubits
        self._free_parameters = set()

    @classmethod
    @abstractmethod
    def from_quantum_circuit(cls, circuit: "QuantumCircuitBase") -> "QuantumCircuitBase":
        """Create a backend-native circuit from a generic quantum circuit."""
        raise NotImplementedError

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits in the circuit."""
        return self._num_qubits

    @property
    def parameters(self) -> List[ParameterVectorElement]:
        """Return the free trainable parameters in the circuit."""
        return sorted(list(self._free_parameters), key=lambda x: x.index)

    @property
    def num_parameters(self) -> int:
        """Return the number of free trainable parameters in the circuit."""
        return len(self.parameters)

    @property
    def is_parameterized(self) -> bool:
        """Check if the wavefunction is parameterized."""
        return len(self.parameters) > 0

    @property
    def parameter_vector_names(self) -> List[str]:
        """Return the unique parameter-vector names in parameter order.

        Standalone parameters contribute their plain name; string parameter
        names of the form ``"theta[0]"`` are reduced to their vector name.
        """
        names: List[str] = []
        for parameter in self.parameters:
            if isinstance(parameter, ParameterVectorElement):
                name = parameter.vector.name
            elif isinstance(parameter, str):
                match = re.match(r"^([a-zA-Z_]\w*)\[\d+\]$", parameter)
                name = match.group(1) if match else parameter
            else:
                name = parameter.name
            if name not in names:
                names.append(name)
        return names

    @abstractmethod
    def draw(self) -> str:
        """Returns printable string representation of the circuit."""
        raise NotImplementedError

    @abstractmethod
    def h(self, qubits: int | List[int]):
        """Add hadamard gates"""
        raise NotImplementedError

    @abstractmethod
    def s(self, qubits: int | List[int]):
        """Add S gates"""
        raise NotImplementedError

    @abstractmethod
    def sdag(self, qubits: int | List[int]):
        """Add Sdag gates"""
        raise NotImplementedError

    @abstractmethod
    def t(self, qubits: int | List[int]):
        """Add T gates"""
        raise NotImplementedError

    @abstractmethod
    def tdag(self, qubits: int | List[int]):
        """Add Tdg gates"""
        raise NotImplementedError

    @abstractmethod
    def p(self, qubits: int | List[int], angle: float):
        """Add P gates"""
        raise NotImplementedError

    def cp(self, control_qubit: int, target_qubit: int, angle: float):
        """Add CP gates"""
        raise NotImplementedError

    @abstractmethod
    def x(self, qubits: int | List[int]):
        """Add X gates"""
        raise NotImplementedError

    @abstractmethod
    def y(self, qubits: int | List[int]):
        """Add Y gates"""
        raise NotImplementedError

    @abstractmethod
    def z(self, qubits: int | List[int]):
        """Add Z gates"""
        raise NotImplementedError

    @abstractmethod
    def rx(self, qubits: int | List[int], angle: float):
        """Add RX gates"""
        raise NotImplementedError

    @abstractmethod
    def ry(self, qubits: int | List[int], angle: float):
        """Add RY gates"""
        raise NotImplementedError

    @abstractmethod
    def rz(self, qubits: int | List[int], angle: float):
        """Add RZ gates"""
        raise NotImplementedError

    @abstractmethod
    def cx(self, control_qubit: int, target_qubit: int):
        """Add CNOT gates"""
        raise NotImplementedError

    @abstractmethod
    def cy(self, control_qubit: int, target_qubit: int):
        """Add CY gates"""
        raise NotImplementedError

    @abstractmethod
    def cz(self, control_qubit: int, target_qubit: int):
        """Add CZ gates"""
        raise NotImplementedError

    def cnot(self, control_qubit: int, target_qubit: int):
        """Add CNOT gates"""
        self.cx(control_qubit, target_qubit)

    def ecr(self, control_qubit: int, target_qubit: int):
        """Add ECR gates"""
        raise NotImplementedError
        # self.cx(control_qubit, target_qubit)
        # self.h(target_qubit)
        # self.cz(control_qubit, target_qubit)
        # self.h(target_qubit)

    @abstractmethod
    def crx(self, control_qubit: int, target_qubit: int, angle: float):
        """Add CRX gates"""
        raise NotImplementedError

    @abstractmethod
    def cry(self, control_qubit: int, target_qubit: int, angle: float):
        """Add CRX gates"""
        raise NotImplementedError

    @abstractmethod
    def crz(self, control_qubit: int, target_qubit: int, angle: float):
        """Add CRX gates"""
        raise NotImplementedError

    @abstractmethod
    def rxx(self, control_qubit: int, target_qubit: int, angle: float):
        """Add RXX gates"""
        raise NotImplementedError

    @abstractmethod
    def ryy(self, control_qubit: int, target_qubit: int, angle: float):
        """Add RYY gates"""
        raise NotImplementedError

    @abstractmethod
    def rzz(self, control_qubit: int, target_qubit: int, angle: float):
        """Add RZZ gates"""
        raise NotImplementedError

    @abstractmethod
    def rzx(self, control_qubit: int, target_qubit: int, angle: float):
        """Add RZX gates"""
        raise NotImplementedError

    @abstractmethod
    def swap(self, qubit1: int, qubit2: int):
        """Add SWAP gates"""
        raise NotImplementedError

    def cswap(self, control_qubit: int, qubit1: int, qubit2: int):
        """Add a controlled-SWAP (Fredkin) gate"""
        raise NotImplementedError

    @abstractmethod
    def ch(self, control_qubit: int, target_qubit: int):
        """Add CH (controlled-Hadamard) gates"""
        raise NotImplementedError

    @abstractmethod
    def i(self, qubits: int | List[int]):
        """Add Identity gates"""
        raise NotImplementedError

    @abstractmethod
    def u(self, qubits: int | List[int], theta: float, phi: float, lam: float):
        """Add U gates (general single-qubit unitary, 3 parameters)"""
        raise NotImplementedError

    @abstractmethod
    def cu(
        self,
        control_qubit: int,
        target_qubit: int,
        theta: float,
        phi: float,
        lam: float,
        gamma: float,
    ):
        """Add CU gates (general controlled unitary, 4 parameters)"""
        raise NotImplementedError

    @abstractmethod
    def barrier(self, qubits: int | List[int]):
        """Add barrier gates"""
        raise NotImplementedError

    @abstractmethod
    def measure(self):
        """Add measure gates"""
        raise NotImplementedError

    def pauli_string(self, pauli_string: str) -> None:
        """Apply a Pauli string to the circuit.

        Args:
            pauli_string (str): Pauli string to apply to the circuit
                (big-endian, the leftmost character acts on qubit 0).

        """
        if len(pauli_string) != self.num_qubits:
            raise ValueError("Pauli string length does not match number of qubits")

        for i, pauli in enumerate(pauli_string):
            if pauli == "X":
                self.x(i)
            elif pauli == "Y":
                self.y(i)
            elif pauli == "Z":
                self.z(i)
            elif pauli == "I":
                pass  # Identity gate (I) can be skipped as it does nothing

    def _apply_basis_change(self, paulis: List[str], qubits: List[int]) -> None:
        """Apply basis change for non-trivial Paulis."""
        for p, q in zip(paulis, qubits):
            if p == "X":
                self.h(q)
            elif p == "Y":
                self.sdag(q)
                self.h(q)
            elif p != "Z":
                raise ValueError(f"Unknown Pauli operator: {p}")

    def _undo_basis_change(self, paulis: List[str], qubits: List[int]) -> None:
        """Undo basis change for non-trivial Paulis."""
        for p, q in zip(paulis, qubits):
            if p == "X":
                self.h(q)
            elif p == "Y":
                self.h(q)
                self.s(q)

    def _apply_cnot_ladder(self, qubits: List[int]) -> None:
        """Apply the forward CNOT ladder for Pauli evolution."""
        if not qubits:
            return
        control = qubits[0]
        for target in qubits[1:]:
            self.cx(control, target)
            control = target

    def _undo_cnot_ladder(self, qubits: List[int]) -> None:
        """Undo the CNOT ladder after the phase rotation."""
        if not qubits:
            return
        control = qubits[-1]
        for target in reversed(qubits[:-1]):
            self.cx(target, control)
            control = target

    def pauli_evolution(
        self,
        operator: QuantumOperatorBase | SparsePauliOp,
        parameter: ParameterVectorElement | ParameterExpression | float,
        working_qubits: List[int] | None = None,
    ) -> None:
        """Apply the Pauli evolution ``exp(-i t P)`` for a single Pauli string.

        Args:
            operator (QuantumOperatorBase | SparsePauliOp): The Pauli operator
                to evolve. Must contain a single Pauli string.
            parameter (ParameterVectorElement | ParameterExpression | float):
                The evolution parameter.
            working_qubits (List[int] | None): Optional qubits to use as
                working qubits.
        """
        self.controlled_pauli_evolution(
            operator,
            parameter,
            working_qubits=working_qubits,
            control_qubits=None,
            control_state=None,
        )

    def controlled_pauli_evolution(
        self,
        operator: Union[QuantumOperatorBase, SparsePauliOp, list],
        parameter: Union[ParameterVectorElement, ParameterExpression, float, list],
        working_qubits: Union[List[List[int]], List[int], None] = None,
        control_qubits: Union[List[List[int]], List[int], int, None] = None,
        control_state: Union[List[str], str, None] = None,
    ) -> None:
        """Apply a (controlled) Pauli evolution ``exp(-i t P)``.

        Supports lists of single-Pauli-string operators applied on disjoint
        working qubits, an optional single control qubit per operator, and a
        control state of ``"0"`` or ``"1"``.

        Args:
            operator: The Pauli operator(s) to evolve. Each must contain a
                single Pauli string; ``SparsePauliOp`` is accepted directly.
            parameter: The evolution parameter(s).
            working_qubits: Optional qubits to use as working qubits, per
                operator.
            control_qubits: Optional qubits controlling the evolution, per
                operator.
            control_state (Union[List[str], str, None]): Optional control
                state (``"0"`` or ``"1"``) for the control qubits.

        Raises:
            TypeError: If the operator input types are invalid.
            ValueError: If operators are not single Pauli strings, have
                complex coefficients, or the qubit assignment is inconsistent.
            NotImplementedError: If more than one control qubit is requested.
        """
        # Normalize the operator input to lists
        if isinstance(operator, (SparsePauliOp, QuantumOperatorBase)):
            operator = [operator]
            if isinstance(parameter, (ParameterVectorElement, float, int, ParameterExpression)):
                parameter = [parameter]
            if working_qubits is None:
                working_qubits = [None]
            elif isinstance(working_qubits, int):
                working_qubits = [[working_qubits]]
            elif isinstance(working_qubits, list):
                if working_qubits and isinstance(working_qubits[0], int):
                    working_qubits = [working_qubits]
            if control_qubits is None:
                control_qubits = [None]
            elif isinstance(control_qubits, int):
                control_qubits = [control_qubits]
            if control_state is None:
                control_state = [None]
            elif isinstance(control_state, str):
                control_state = [control_state]
        elif not isinstance(operator, list):
            raise TypeError(
                "Operator must be a SparsePauliOp, a QuantumOperatorBase, or a list thereof"
            )
        else:
            if not isinstance(parameter, list):
                parameter = [parameter] * len(operator)
            if working_qubits is None:
                working_qubits = [None] * len(operator)
            if isinstance(working_qubits, int):
                working_qubits = [[working_qubits]] * len(operator)
            elif isinstance(working_qubits, list):
                if working_qubits and isinstance(working_qubits[0], int):
                    working_qubits = [working_qubits] * len(operator)
            if control_qubits is None:
                control_qubits = [None] * len(operator)
            elif isinstance(control_qubits, int):
                control_qubits = [control_qubits] * len(operator)
            if control_state is None:
                control_state = [None] * len(operator)
            elif isinstance(control_state, str):
                control_state = [control_state] * len(operator)

        # TODO: The list branch does not validate that parameter/working_qubits/
        # control_qubits/control_state list lengths match len(operator); decide
        # whether mismatches should raise or broadcast before hardening this.
        for state in control_state:
            if state not in (None, "0", "1"):
                raise ValueError(f'control_state entries must be "0" or "1", got {state!r}')

        # Pre-processed values for each operator
        pauli_strings_operators = []
        coefficient_operators = []
        # Indices of the Pauli string entries that are not the identity
        qubits_operator = []
        # Working qubits for each operator (without identity strings)
        working_qubits_operator = []
        control_qubits_operator = []

        for i, o in enumerate(operator):
            op = getattr(o, "qiskit_operator", o)
            if isinstance(op, SparsePauliOp):
                # Qiskit labels are little-endian; convert to the public
                # big-endian convention (leftmost character acts on qubit 0).
                labels = [label[::-1] for label in op.paulis.to_labels()]
                coeffs = list(op.coeffs)
            elif isinstance(op, QuantumOperatorBase):
                labels = list(op.paulis)
                coeffs = list(op.coeffs)
            else:
                raise TypeError(f"Expected SparsePauliOp or QuantumOperatorBase, got {type(o)}")
            if len(coeffs) != 1:
                raise ValueError("Only operators with single Pauli strings are supported")
            pauli_str, coeff = labels[0], coeffs[0]

            # Preprocess the coefficient
            if isinstance(coeff, (ParameterVectorElement, ParameterExpression)):
                # the 1j fixes a bug in qiskit
                coeff = -1j * (1j * coeff)
            else:
                coeff = np.real_if_close(coeff)
                if np.iscomplexobj(coeff):
                    raise ValueError("Complex coefficients are not supported")
                coeff = float(np.real(coeff))
            coefficient_operators.append(coeff * parameter[i])

            # Remove the Identity term from the Pauli string
            qubits_operator.append([i for i, p in enumerate(pauli_str) if p != "I"])
            pauli_strings_operators.append([p for p in pauli_str if p != "I"])

        # Assign or validate control and working qubits
        all_possible_qubits = set(list(range(self.num_qubits)))
        controlled_qubits = set()
        workedon_qubits = set()
        for i, cq in enumerate(control_qubits):
            # check control qubits
            if cq is not None:
                if isinstance(cq, int):
                    cq = [cq]
                if any(idx not in all_possible_qubits for idx in cq):
                    raise ValueError("Qubits are not supported!")
                controlled_qubits.update(cq)
            control_qubits_operator.append(cq)

        # remove all controlled qubits from all_possible qubits
        all_possible_qubits = all_possible_qubits - controlled_qubits

        for i, working_qubits_ in enumerate(working_qubits):
            if working_qubits_ is None:
                # Default assignment: label position idx maps to the idx-th
                # free qubit (ascending, excluding control qubits and qubits
                # already assigned to earlier operators) — control qubits may
                # sit at any index, not just the lowest ones.
                # TODO: This requires as many free qubits as the Pauli string
                # is long, including its identity positions. Decide whether a
                # sparse operator such as "XI" should instead be packed onto
                # its non-identity positions only (then it would need just one
                # free qubit); until then it raises "Not enough qubits left".
                candidates = sorted(all_possible_qubits)
                if max(qubits_operator[i], default=-1) >= len(candidates):
                    raise ValueError("Not enough qubits left for implementing pauli evolution!")
                wqubits = [candidates[idx] for idx in qubits_operator[i]]
            else:
                wqubits = [working_qubits_[idx] for idx in qubits_operator[i]]
                if any(q < 0 or q >= self.num_qubits for q in wqubits):
                    raise ValueError("Working qubits are out of range for this circuit!")

            all_possible_qubits = all_possible_qubits - set(wqubits)

            if any(idx in workedon_qubits for idx in wqubits):
                raise ValueError("No distinct support qubits between the operators!")
            if any(idx in controlled_qubits for idx in wqubits):
                raise ValueError("Controlled qubits must be distinct from working qubits!")
            workedon_qubits.update(wqubits)
            working_qubits_operator.append(wqubits)

        # Apply basis change for non-trivial Paulis
        for paulis, wqubits in zip(pauli_strings_operators, working_qubits_operator):
            self._apply_basis_change(paulis, wqubits)

        # Apply chains of CNOTs
        for wqubits in working_qubits_operator:
            self._apply_cnot_ladder(wqubits)

        for i in range(len(operator)):
            if len(qubits_operator[i]) == 0:
                # Identity operator -> Phase rotation
                if control_qubits_operator[i]:
                    control = control_qubits_operator[i]
                    if len(control) == 1:
                        if control_state[i] == "0":
                            self.x(control[0])
                        self.rz(control[0], -coefficient_operators[i])
                        if control_state[i] == "0":
                            self.x(control[0])
                    else:
                        raise NotImplementedError("Multiple control not implemented yet")
                else:
                    # No control qubits, phase rotation is ignored
                    pass
            else:
                # Pauli rotations
                if control_qubits_operator[i]:
                    # Controlled pauli rotation
                    control = control_qubits_operator[i]
                    if len(control) == 1:
                        if control_state[i] == "0":
                            self.x(control[0])
                        self.crz(
                            control[0],
                            working_qubits_operator[i][-1],
                            2 * coefficient_operators[i],
                        )
                        if control_state[i] == "0":
                            self.x(control[0])
                    else:
                        raise NotImplementedError("Multiple control not implemented yet")
                else:
                    # Default pauli rotations
                    self.rz(working_qubits_operator[i][-1], 2 * coefficient_operators[i])

        # Undo the CNOT ladders
        for wqubits in working_qubits_operator:
            self._undo_cnot_ladder(wqubits)

        # Undo basis change for non-trivial Paulis
        for paulis, wqubits in zip(pauli_strings_operators, working_qubits_operator):
            self._undo_basis_change(paulis, wqubits)

    def compose(
        self,
        qc: "QuantumCircuitBase",
        qubits: List[int] | None = None,
        clbits: List[int] | None = None,
        new_parameters: bool = True,
    ) -> "QuantumCircuitBase":
        """Compose another circuit into this one.

        The qubit mapping is validated here; backends implement the actual
        merging in :meth:`_backend_specific_compose`.

        Args:
            qc (QuantumCircuitBase): Circuit to compose with.
            qubits (List[int] | None): Qubit indices of ``self`` that the
                qubits of ``qc`` are mapped onto. Defaults to the identity
                mapping, which requires equal qubit counts.
            clbits (List[int] | None): Classical-bit indices of ``self`` that
                the classical bits of ``qc`` are mapped onto.
            new_parameters (bool): If True (default), the parameters of ``qc``
                are appended after the parameters of ``self``. If False, the
                parameters of both circuits are merged positionally.

        Returns:
            QuantumCircuitBase: This circuit, after in-place composition.

        Raises:
            TypeError: If ``qc`` is not a :class:`QuantumCircuitBase`.
            ValueError: If the qubit mapping is invalid.
        """
        if not isinstance(qc, QuantumCircuitBase):
            raise TypeError(f"compose expects a QuantumCircuitBase, got {type(qc).__name__}")
        if qubits is None:
            if self.num_qubits != qc.num_qubits:
                raise ValueError(
                    "When qubits=None, both circuits must have the same number of qubits "
                    f"(got self.num_qubits={self.num_qubits}, qc.num_qubits={qc.num_qubits})."
                )
            qubits = list(range(qc.num_qubits))
        if len(qubits) != qc.num_qubits:
            raise ValueError(
                "Length of qubits mapping must match the composed circuit qubit count "
                f"(got len(qubits)={len(qubits)}, qc.num_qubits={qc.num_qubits})."
            )
        if any(q < 0 or q >= self.num_qubits for q in qubits):
            raise ValueError("Qubit mapping contains indexes out of range for the target circuit.")
        if len(set(qubits)) != len(qubits):
            raise ValueError("Qubit mapping contains duplicate indices.")
        return self._backend_specific_compose(qc, qubits, clbits, new_parameters)

    def _backend_specific_compose(
        self,
        qc: "QuantumCircuitBase",
        qubits: List[int],
        clbits: List[int] | None,
        new_parameters: bool,
    ) -> "QuantumCircuitBase":
        """Merge ``qc`` into this circuit; ``qubits`` is validated and complete.

        The default implementation merges in place via the shared qiskit
        representation, so it works for every circuit type that exposes a
        ``qiskit_circuit``. Parameters of both circuits are re-indexed into
        a single fresh parameter vector so that repeatedly composing
        circuits that use identically named parameter vectors never
        collides: the parameters of ``self`` keep their positions and the
        parameters of ``qc`` are appended (or merged positionally for
        ``new_parameters=False``).

        Raises:
            NotImplementedError: If either circuit does not expose the
                shared qiskit representation.
        """
        # TODO: This default merge is qiskit-specific because the abstraction
        # layer has no framework-independent way to enumerate the operations
        # of a circuit yet. Circuit types without a qiskit representation
        # (currently PauliPropagationCircuit) therefore do not support
        # compose. Once such a gate-level representation exists, reimplement
        # the merge against it.
        own = getattr(self, "qiskit_circuit", None)
        other = getattr(qc, "qiskit_circuit", None)
        if own is None or other is None:
            raise NotImplementedError(
                f"compose is not supported between {type(self).__name__} and "
                f"{type(qc).__name__}: the circuits share no qiskit representation."
            )

        if own.parameters and other.parameters:
            # TODO: Merging squashes both circuits into a single vector named
            # after self's first parameter, so qc's parameters are renamed
            # (e.g. "y[0]" becomes "x[1]") and keyword access via the old name
            # stops working. Decide whether the original names should be kept.
            own_params = list(own.parameters)
            other_params = list(other.parameters)
            first = own_params[0]
            name = first.vector.name if isinstance(first, ParameterVectorElement) else first.name
            if new_parameters:
                merged = ParameterVector(name, len(own_params) + len(other_params))
                other_target = merged[len(own_params) :]
            else:
                merged = ParameterVector(name, max(len(own_params), len(other_params)))
                other_target = merged[: len(other_params)]
            own.assign_parameters(dict(zip(own_params, merged[: len(own_params)])), inplace=True)
            other_assigned = other.assign_parameters(
                dict(zip(other_params, other_target)), inplace=False
            )
            own.compose(other_assigned, qubits=qubits, clbits=clbits, inplace=True)
        else:
            own.compose(other, qubits=qubits, clbits=clbits, inplace=True)
        return self

    def fixate_parameters(self, parameters: np.ndarray) -> None:
        """Bind all free parameters, removing them from the circuit.

        Args:
            parameters (np.ndarray): Values to assign, in parameter order.
        """
        raise NotImplementedError

    def assign_parameters(self, parameters: dict):
        """Change parameters in the circuit.

        Args:
            parameters (np.array): parameters to assign to the circuit
        """
        raise NotImplementedError

    def invert(self) -> "QuantumCircuitBase":
        """Invert the circuit."""
        raise NotImplementedError

    def copy(self) -> "QuantumCircuitBase":
        """Return a copy of the circuit."""
        raise NotImplementedError

    def circuit_metrics(self) -> dict:
        """count number of gates in the circuit"""
        raise NotImplementedError

    def from_qasm(self, qasm: str) -> None:
        """Load the circuit from a qasm string"""
        raise NotImplementedError

    def to_qasm(self) -> str:
        """Convert the circuit to a qasm string"""
        raise NotImplementedError

    def structural_key(self) -> tuple:
        """Return a hashable key describing the current circuit structure.

        The key changes whenever the circuit is mutated, which makes it a
        sound cache key even for circuits that are modified in place. It must
        be recomputed on every use and never memoized.
        """
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError(
            "Hashing is not implemented for this class. "
            "Please implement the __hash__ method in the derived class."
        )

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError
