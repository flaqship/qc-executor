"""Tests for the framework-independent AbstractQuantumOperator.

Three layers:
1. Construction / properties / parameters (pure, no backend).
2. Algebra parity against Qiskit ``SparsePauliOp`` (the reference semantics).
3. End-to-end expectation value through the pauli_propagation backend, compared
   against an independent Qiskit statevector reference.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from executor.abstraction import AbstractQuantumOperator, ParameterVector


def _to_sparse(op: AbstractQuantumOperator) -> SparsePauliOp:
    """Convert an abstract operator into a Qiskit SparsePauliOp for comparison."""
    return SparsePauliOp(op.paulis, coeffs=op.coeffs).simplify()


def _equiv(abstract_op: AbstractQuantumOperator, sparse_ref: SparsePauliOp) -> bool:
    """True if the abstract operator equals the Qiskit reference (up to ~0 terms)."""
    diff = (_to_sparse(abstract_op) - sparse_ref).simplify()
    return not np.any(np.round(diff.coeffs, 9))


class TestConstruction:
    def test_basic_properties(self):
        op = AbstractQuantumOperator(["ZZ", "IX"], [0.5, 0.3])
        assert op.paulis == ["ZZ", "IX"]
        assert op.coeffs == [0.5, 0.3]
        assert op.num_qubits == 2
        assert op.num_paulis == 2

    def test_default_coeffs_are_one(self):
        assert AbstractQuantumOperator(["XY"]).coeffs == [1.0]

    def test_mismatched_pauli_lengths_raise(self):
        with pytest.raises(ValueError):
            AbstractQuantumOperator(["ZZ", "I"])

    def test_coeff_count_must_match(self):
        with pytest.raises(ValueError):
            AbstractQuantumOperator(["ZZ"], [1.0, 2.0])

    def test_empty_requires_num_qubits(self):
        with pytest.raises(ValueError):
            AbstractQuantumOperator([])
        assert AbstractQuantumOperator([], num_qubits=3).num_qubits == 3

    def test_equality_is_order_independent(self):
        a = AbstractQuantumOperator(["ZZ", "IX"], [0.5, 0.3])
        b = AbstractQuantumOperator(["IX", "ZZ"], [0.3, 0.5])
        assert a == b
        assert hash(a) == hash(b)


class TestParameters:
    def test_symbolic_coeffs_detected(self):
        x = ParameterVector("x", 1)
        op = AbstractQuantumOperator(["XZ", "ZI"], [2 * x[0], 1.0])
        assert op.is_parametrized
        assert op.parameters == [x[0]]
        assert op.num_parameters == 1

    def test_assign_parameters(self):
        x = ParameterVector("x", 1)
        op = AbstractQuantumOperator(["XZ"], [2 * x[0]])
        op.assign_parameters({x[0]: 0.5})
        assert op.coeffs == [1.0]
        assert not op.is_parametrized


class TestAlgebraParityWithQiskit:
    """Each abstract operation must match Qiskit's SparsePauliOp semantics."""

    def test_adjoint(self):
        op = AbstractQuantumOperator(["XZ", "YI"], [1.0, 2.0 + 1j])
        assert _equiv(op.adjoint(), _to_sparse(op).adjoint())

    def test_transpose(self):
        op = AbstractQuantumOperator(["XZ", "YI"], [1.0, 2.0])
        assert _equiv(op.transpose(), _to_sparse(op).transpose())

    def test_conjugate(self):
        op = AbstractQuantumOperator(["XZ", "YI"], [1.0, 1j])
        assert _equiv(op.conjugate(), _to_sparse(op).conjugate())

    def test_simplify_combines_terms(self):
        op = AbstractQuantumOperator(["XX", "XX", "ZZ"], [1, 2, 5])
        assert _equiv(op.simplify(), SparsePauliOp(["XX", "ZZ"], [3, 5]))

    def test_compose(self):
        a = AbstractQuantumOperator(["XZ", "YI"], [1.0, 2.0])
        b = AbstractQuantumOperator(["ZZ", "XY"], [0.5, 1j])
        ref = _to_sparse(a).compose(_to_sparse(b))
        assert _equiv(a.compose(b), ref)

    def test_apply_layout(self):
        op = AbstractQuantumOperator(["XZ", "YI"], [1.0, 2.0])
        ref = _to_sparse(op).apply_layout([0, 2], 3)
        assert _equiv(op.apply_layout([0, 2], num_qubits=3), ref)


class TestEndToEnd:
    """The abstract operator must work as the observable in expectation_value."""

    def test_pauli_propagation_expectation_matches_qiskit(self):
        from qiskit import QuantumCircuit as QiskitCircuit
        from qiskit.quantum_info import Statevector

        from executor import Executor, QuantumCircuit

        # Qiskit-backed circuit (the abstract-circuit path is wired separately).
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        obs = AbstractQuantumOperator(["ZZ", "IX"], [0.5, 0.3])

        executor = Executor.create("pauli_propagation", seed=0)
        value = executor.expectation_value(
            executor.transpile_circuit(qc),
            executor.transpile_operator(obs),
        )
        value = float(np.real_if_close(value))

        ref_qc = QiskitCircuit(2)
        ref_qc.h(0)
        ref_qc.cx(0, 1)
        reference = float(
            np.real(
                Statevector(ref_qc).expectation_value(SparsePauliOp(["ZZ", "IX"], [0.5, 0.3]))
            )
        )

        assert np.isclose(value, reference)
