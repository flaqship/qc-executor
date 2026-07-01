"""Regression tests for the framework-independent AbstractQuantumCircuit.

Focus: multi-qubit *broadcast* gate calls (e.g. ``h(range(n))``) must expand
into one gate per qubit, identical to calling the gate once per qubit.

Regression guard for a fixed bug: ``range`` was not recognised as iterable, so
``h(range(4))`` stored a *single* gate whose "qubit" was the range object
itself. Qiskit happened to tolerate it; PennyLane returned a wrong, inflated
statevector (32 amplitudes instead of 16).
"""

from __future__ import annotations

import numpy as np
import pytest

from executor.abstraction import AbstractQuantumCircuit, ParameterVector
from executor.abstraction.gates import CliffordGate, RotationGate


class TestBroadcastExpansion:
    """Broadcasting a gate over several qubits must expand to one gate each."""

    def test_range_expands_like_individual_calls(self):
        broadcast = AbstractQuantumCircuit(4)
        broadcast.h(range(4))

        individual = AbstractQuantumCircuit(4)
        for q in range(4):
            individual.h(q)

        assert broadcast._gates == individual._gates

    @pytest.mark.parametrize("qubits", [range(3), [0, 1, 2], (0, 1, 2)])
    def test_iterable_forms_are_equivalent(self, qubits):
        qc = AbstractQuantumCircuit(3)
        qc.h(qubits)
        assert qc._gates == [CliffordGate("h", (q,)) for q in (0, 1, 2)]

    def test_int_stays_a_single_gate(self):
        qc = AbstractQuantumCircuit(3)
        qc.h(1)
        assert qc._gates == [CliffordGate("h", (1,))]

    def test_rotation_broadcasts_over_range(self):
        x = ParameterVector("x", 1)
        qc = AbstractQuantumCircuit(3)
        qc.ry(range(3), 2 * x[0])
        assert qc._gates == [RotationGate("ry", (q,), 2 * x[0]) for q in (0, 1, 2)]

    def test_qubit_index_is_never_a_range_object(self):
        # The exact failure mode: a range object leaked in as the qubit index.
        qc = AbstractQuantumCircuit(4)
        qc.h(range(4))
        assert len(qc._gates) == 4
        assert all(isinstance(g.qubits[0], int) for g in qc._gates)


class TestBackendParity:
    """The broadcast bug surfaced as a wrong PennyLane statevector."""

    def test_broadcast_statevector_matches_across_backends(self):
        pytest.importorskip("pennylane")
        from executor import Executor

        qc = AbstractQuantumCircuit(4)
        qc.h(range(4))  # the broadcast form that used to break PennyLane
        qc.cx(0, 1)

        sv_qiskit = Executor.create("qiskit").statevector(qc)
        sv_pennylane = Executor.create("pennylane").statevector(qc)

        assert sv_qiskit.shape == (16,)  # not 32 (the old, inflated shape)
        assert sv_pennylane.shape == (16,)
        assert np.allclose(sv_qiskit, sv_pennylane)
