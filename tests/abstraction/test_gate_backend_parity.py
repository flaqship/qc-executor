"""Every abstraction gate must mean the same thing in Qiskit and in PennyLane.

``SUPPORTED_GATES`` is the shared vocabulary, but each backend learns about a
gate in its own way:

* Qiskit resolves the gate name as a method name (``getattr(qc, name)``).
* PennyLane looks the name up in ``qiskit_pennylane_gate_dict``.

Nothing in the code forces the three to agree -- a gate could sit in the registry
with no PennyLane mapping, and would then work on Qiskit and raise on PennyLane
at circuit-build time, far away from the cause. These tests close that gap: they
compare the actual unitaries and fail if the registry, the two backends, the
inversion tables and the gate methods drift apart.

Convention: Qiskit's ``Operator`` is little-endian (qubit 0 = least significant
bit), PennyLane's ``qml.matrix`` is big-endian (wire 0 = most significant).
``reverse_qargs()`` rewrites the Qiskit matrix into PennyLane's ordering so the
two are directly comparable.
"""

from __future__ import annotations

import inspect

import numpy as np
import pennylane as qml
import pytest
from qiskit.circuit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.quantum_info import Operator

from qc_executor.abstraction import AbstractQuantumCircuit
from qc_executor.abstraction.abstract_quantum_circuit import (
    _CLIFFORD_INVERSE_MAP,
    _SELF_INVERSE,
)
from qc_executor.abstraction.gates import (
    SUPPORTED_GATES,
    Barrier,
    CliffordGate,
    RotationGate,
)
from qc_executor.pennylane.pennylane_circuit import PennyLaneCircuit
from qc_executor.pennylane.pennylane_gates import qiskit_pennylane_gate_dict
from qc_executor.qiskit.qiskit_circuit import _abstract_to_qiskit

THETA = 0.7231

#: How to call each gate method: ``method name -> (args, qubits needed)``.
#: Two tests keep this honest: every gate method must appear here, and every
#: entry of SUPPORTED_GATES must be produced by at least one of these calls.
GATE_CASES: dict[str, tuple[tuple, int]] = {
    # one qubit, no angle
    "id": ((0,), 1),
    "h": ((0,), 1),
    "s": ((0,), 1),
    "sdag": ((0,), 1),
    "sx": ((0,), 1),
    "sxdag": ((0,), 1),
    "t": ((0,), 1),
    "tdag": ((0,), 1),
    "x": ((0,), 1),
    "y": ((0,), 1),
    "z": ((0,), 1),
    # one qubit, one angle
    "rx": ((0, THETA), 1),
    "ry": ((0, THETA), 1),
    "rz": ((0, THETA), 1),
    "p": ((0, THETA), 1),
    # two qubits, no angle
    "cx": ((0, 1), 2),
    "cnot": ((0, 1), 2),
    "cy": ((0, 1), 2),
    "cz": ((0, 1), 2),
    "ch": ((0, 1), 2),
    "cs": ((0, 1), 2),
    "csdag": ((0, 1), 2),
    "ecr": ((0, 1), 2),
    "swap": ((0, 1), 2),
    # two qubits, one angle
    "cp": ((0, 1, THETA), 2),
    "crx": ((0, 1, THETA), 2),
    "cry": ((0, 1, THETA), 2),
    "crz": ((0, 1, THETA), 2),
    "rxx": ((0, 1, THETA), 2),
    "ryy": ((0, 1, THETA), 2),
    "rzz": ((0, 1, THETA), 2),
    "rzx": ((0, 1, THETA), 2),
    # three qubits, no angle
    "ccx": ((0, 1, 2), 3),
    "toffoli": ((0, 1, 2), 3),
    "cswap": ((0, 1, 2), 3),
}

#: Public methods of AbstractQuantumCircuit that do not add a gate.
_NON_GATE_METHODS = frozenset(
    {
        "from_quantum_circuit",
        "draw",
        "measure",
        "barrier",
        "assign_parameters",
        "compose",
        "invert",
        "copy",
        "circuit_metrics",
    }
)


def _gate_methods() -> set[str]:
    """Public gate-adding methods declared on AbstractQuantumCircuit."""
    return {
        name
        for name, member in vars(AbstractQuantumCircuit).items()
        if inspect.isfunction(member)
        and not name.startswith("_")
        and name not in _NON_GATE_METHODS
    }


def _build(method: str) -> AbstractQuantumCircuit:
    args, num_qubits = GATE_CASES[method]
    circuit = AbstractQuantumCircuit(num_qubits)
    getattr(circuit, method)(*args)
    return circuit


def _qiskit_unitary(circuit: AbstractQuantumCircuit) -> np.ndarray:
    """Unitary of the Qiskit build, rewritten into PennyLane's wire ordering."""
    return Operator(_abstract_to_qiskit(circuit)).reverse_qargs().data


def _pennylane_unitary(circuit: AbstractQuantumCircuit) -> np.ndarray:
    """Unitary of the PennyLane build."""
    callable_circuit = PennyLaneCircuit(circuit).get_pennylane_circuit()
    return qml.matrix(callable_circuit, wire_order=list(range(circuit.num_qubits)))()


class TestVocabularyIsWiredUpEverywhere:
    """A gate in the registry must be reachable and runnable on both backends."""

    @pytest.mark.parametrize("name", sorted(SUPPORTED_GATES))
    def test_gate_is_mapped_in_pennylane(self, name):
        """PennyLane resolves gates through a dict -- an unmapped name crashes at build time."""
        assert name in qiskit_pennylane_gate_dict, (
            f"'{name}' is in SUPPORTED_GATES but not in qiskit_pennylane_gate_dict. "
            "The PennyLane backend would raise NotImplementedError."
        )

    @pytest.mark.parametrize("name", sorted(SUPPORTED_GATES))
    def test_gate_is_a_qiskit_method(self, name):
        """Qiskit resolves gates via getattr -- the name must be a QuantumCircuit method."""
        assert hasattr(QiskitQuantumCircuit, name), (
            f"'{name}' is in SUPPORTED_GATES but Qiskit's QuantumCircuit has no such method."
        )

    @pytest.mark.parametrize("name", sorted(SUPPORTED_GATES))
    def test_gate_is_invertible(self, name):
        """invert() raises for any non-rotation gate missing from the inversion tables."""
        if SUPPORTED_GATES[name].num_angles:
            return  # rotations invert by negating the angle
        assert name in _SELF_INVERSE or name in _CLIFFORD_INVERSE_MAP, (
            f"'{name}' is in neither _SELF_INVERSE nor _CLIFFORD_INVERSE_MAP. "
            "invert() would raise on any circuit containing it."
        )

    def test_every_gate_method_is_covered(self):
        """A new gate method without a GATE_CASES entry would go untested."""
        missing = _gate_methods() - set(GATE_CASES)
        assert not missing, (
            f"Gate method(s) {sorted(missing)} have no GATE_CASES entry. Add one so "
            "the gate is checked against both backends."
        )

    def test_no_stale_gate_cases(self):
        stale = set(GATE_CASES) - _gate_methods()
        assert not stale, f"GATE_CASES lists method(s) that no longer exist: {sorted(stale)}"

    def test_every_supported_gate_is_reachable(self):
        """A registry entry no method can produce is dead weight -- and untested."""
        produced = {gate.name for method in GATE_CASES for gate in _build(method)._gates}
        unreachable = set(SUPPORTED_GATES) - produced
        assert not unreachable, (
            f"Gate(s) {sorted(unreachable)} are in SUPPORTED_GATES but no "
            "AbstractQuantumCircuit method produces them."
        )


class TestQiskitPennyLaneUnitaryParity:
    """The same abstract gate must produce the same unitary on both backends."""

    @pytest.mark.parametrize("method", sorted(GATE_CASES))
    def test_unitaries_match(self, method):
        circuit = _build(method)
        np.testing.assert_allclose(
            _qiskit_unitary(circuit),
            _pennylane_unitary(circuit),
            atol=1e-10,
            err_msg=f"Qiskit and PennyLane disagree on the unitary of '{method}'",
        )

    def test_full_circuit_of_every_gate_matches(self):
        """All gates stacked into one circuit -- catches wire-ordering slips."""
        circuit = AbstractQuantumCircuit(3)
        for method, (args, _) in GATE_CASES.items():
            getattr(circuit, method)(*args)

        np.testing.assert_allclose(
            _qiskit_unitary(circuit), _pennylane_unitary(circuit), atol=1e-10
        )


class TestInverseParity:
    """invert() must produce the actual inverse, on both backends."""

    @pytest.mark.parametrize("method", sorted(GATE_CASES))
    def test_gate_times_its_inverse_is_identity(self, method):
        circuit = _build(method)
        inverse = circuit.invert()
        dim = 2**circuit.num_qubits

        for name, unitary_of in (("qiskit", _qiskit_unitary), ("pennylane", _pennylane_unitary)):
            np.testing.assert_allclose(
                unitary_of(inverse) @ unitary_of(circuit),
                np.eye(dim),
                atol=1e-10,
                err_msg=f"invert() of '{method}' is not the inverse on the {name} backend",
            )

    def test_inverse_of_full_circuit_is_identity(self):
        circuit = AbstractQuantumCircuit(3)
        for method, (args, _) in GATE_CASES.items():
            getattr(circuit, method)(*args)
        inverse = circuit.invert()

        for unitary_of in (_qiskit_unitary, _pennylane_unitary):
            np.testing.assert_allclose(
                unitary_of(inverse) @ unitary_of(circuit), np.eye(8), atol=1e-10
            )


class TestGateValidation:
    """_add() rejects anything the vocabulary does not allow."""

    def test_unknown_gate_name_is_rejected(self):
        # A name no real gate will ever take, so this stays a valid canary even
        # as the vocabulary grows.
        circuit = AbstractQuantumCircuit(2)
        with pytest.raises(ValueError, match="Unknown gate 'not_a_gate'"):
            circuit._add(CliffordGate("not_a_gate", (0, 1)))

    def test_wrong_qubit_count_is_rejected(self):
        circuit = AbstractQuantumCircuit(3)
        with pytest.raises(ValueError, match="acts on 3 qubit"):
            circuit._add(CliffordGate("ccx", (0, 1)))

    def test_repeated_qubit_is_rejected(self):
        circuit = AbstractQuantumCircuit(2)
        with pytest.raises(ValueError, match="repeated qubit"):
            circuit._add(CliffordGate("cx", (0, 0)))

    def test_qubit_outside_circuit_is_rejected(self):
        circuit = AbstractQuantumCircuit(2)
        with pytest.raises(ValueError, match="outside the circuit"):
            circuit.h(5)

    def test_rotation_built_as_clifford_is_rejected(self):
        circuit = AbstractQuantumCircuit(1)
        with pytest.raises(ValueError, match="must be built as a RotationGate"):
            circuit._add(CliffordGate("rx", (0,)))

    def test_clifford_built_as_rotation_is_rejected(self):
        circuit = AbstractQuantumCircuit(1)
        with pytest.raises(ValueError, match="must be built as a CliffordGate"):
            circuit._add(RotationGate("h", (0,), THETA))

    def test_barrier_is_always_accepted(self):
        circuit = AbstractQuantumCircuit(2)
        circuit.barrier([0, 1])
        assert isinstance(circuit._gates[0], Barrier)
