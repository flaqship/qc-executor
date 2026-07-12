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

from qc_executor.abstraction import (
    AbstractQuantumCircuit,
    AbstractQuantumOperator,
    ParameterVector,
)


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

    def test_invalid_pauli_characters_raise(self):
        # Without the check "AB" only failed deep inside compose (KeyError) and
        # lowercase "zx" silently skipped the Y sign flip in transpose.
        with pytest.raises(ValueError, match="Invalid Pauli label"):
            AbstractQuantumOperator(["AB"], [1.0])
        with pytest.raises(ValueError, match="Invalid Pauli label"):
            AbstractQuantumOperator(["zx"], [1.0])
        # append builds through the constructor, so it inherits the check
        with pytest.raises(ValueError, match="Invalid Pauli label"):
            AbstractQuantumOperator(["XX"], [1.0]).append("AB")

    def test_num_qubits_must_match_pauli_width(self):
        # An inconsistent operator used to blow up later (IndexError in
        # apply_layout) instead of failing at construction time.
        with pytest.raises(ValueError, match="does not match"):
            AbstractQuantumOperator(["ZZ"], [1.0], num_qubits=3)
        assert AbstractQuantumOperator(["ZZ"], [1.0], num_qubits=2).num_qubits == 2

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


class TestFromQuantumOperator:
    """from_quantum_operator must yield a framework-free abstract operator.

    Foreign coefficient types (Qiskit ``ParameterExpression``) have to be
    normalized to SymPy expressions over native ``Parameter`` symbols —
    otherwise the abstract operator would silently carry Qiskit objects that
    break ``parameters``/``assign_parameters`` later.
    """

    @staticmethod
    def _foreign_operator(sparse):
        """Minimal stand-in for a backend-native operator (paulis + coeffs)."""

        class _Foreign:
            paulis = [str(p) for p in sparse.paulis]
            coeffs = list(sparse.coeffs)

        return _Foreign()

    def test_numeric_foreign_operator(self):
        op = AbstractQuantumOperator.from_quantum_operator(
            self._foreign_operator(SparsePauliOp(["ZZ", "IX"], [0.5, 0.3]))
        )
        assert isinstance(op, AbstractQuantumOperator)
        assert op.paulis == ["ZZ", "IX"]
        assert np.allclose([complex(c) for c in op.coeffs], [0.5, 0.3])
        assert not op.is_parametrized

    def test_parametrized_qiskit_coeffs_become_native_parameters(self):
        from qiskit.circuit import ParameterVector as QiskitParameterVector

        from qc_executor.abstraction.abstract_parameter import Parameter

        theta = QiskitParameterVector("theta", 2)
        sparse = SparsePauliOp(["ZZ", "IX"], coeffs=[2 * theta[0], theta[1]])
        op = AbstractQuantumOperator.from_quantum_operator(self._foreign_operator(sparse))

        assert op.is_parametrized
        assert all(isinstance(p, Parameter) for p in op.parameters)
        assert [(p.vector_name, p.index) for p in op.parameters] == [
            ("theta", 0),
            ("theta", 1),
        ]
        op.assign_parameters(dict(zip(op.parameters, [0.4, 0.9])))
        assert np.allclose([complex(c) for c in op.coeffs], [0.8, 0.9])

    def test_abstract_input_is_copied(self):
        original = AbstractQuantumOperator(["ZZ"], [0.5])
        converted = AbstractQuantumOperator.from_quantum_operator(original)
        assert converted == original
        assert converted is not original


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


class TestPureSemanticsAndValidation:
    """Algebra methods must be pure (like SparsePauliOp) and validate input."""

    def test_compose_does_not_mutate_operands(self):
        a = AbstractQuantumOperator(["XZ"], [1.0])
        b = AbstractQuantumOperator(["ZZ"], [0.5])
        result = a.compose(b)
        assert result is not a
        assert a.paulis == ["XZ"] and a.coeffs == [1.0]
        assert b.paulis == ["ZZ"] and b.coeffs == [0.5]

    def test_append_returns_new_operator(self):
        op = AbstractQuantumOperator(["XX"], [1.0])
        updated = op.append("ZZ", 2.0)
        assert updated is not op
        assert op.paulis == ["XX"]
        assert updated.paulis == ["XX", "ZZ"] and updated.coeffs == [1.0, 2.0]

    def test_properties_return_copies(self):
        op = AbstractQuantumOperator(["XX"], [1.0])
        op.paulis.append("ZZ")  # mutating the returned lists ...
        op.coeffs.append(2.0)
        assert op.paulis == ["XX"]  # ... must not affect the operator
        assert op.coeffs == [1.0]

    def test_assign_parameters_returns_self(self):
        x = ParameterVector("x", 1)
        op = AbstractQuantumOperator(["XZ"], [2 * x[0]])
        assert op.assign_parameters({x[0]: 0.5}) is op

    def test_apply_layout_rejects_wrong_length(self):
        op = AbstractQuantumOperator(["XZ"], [1.0])
        with pytest.raises(ValueError, match="entries"):
            op.apply_layout([0])

    def test_apply_layout_rejects_duplicates(self):
        op = AbstractQuantumOperator(["XZ"], [1.0])
        with pytest.raises(ValueError, match="unique"):
            op.apply_layout([1, 1])

    def test_apply_layout_rejects_out_of_range(self):
        # Without validation layout entry 3 on a 3-qubit target would write to
        # a negative string index and silently corrupt the result.
        op = AbstractQuantumOperator(["XZ"], [1.0])
        with pytest.raises(ValueError, match="lie in"):
            op.apply_layout([0, 3], num_qubits=3)

    def test_group_commuting_is_qubit_wise_like_qiskit(self):
        # XX and YY commute generally but not qubit-wise: the abstract grouping
        # must match Qiskit's qubit-wise grouping and put them in separate groups.
        abstract_groups = AbstractQuantumOperator(["XX", "YY"]).group_commuting()
        qiskit_groups = SparsePauliOp(["XX", "YY"]).group_commuting(qubit_wise=True)
        assert len(abstract_groups) == 2
        assert len(qiskit_groups) == 2


class TestEndToEnd:
    """The abstract operator must work as the observable in expectation_value."""

    def test_pauli_propagation_expectation_matches_qiskit(self):
        from qiskit import QuantumCircuit as QiskitCircuit
        from qiskit.quantum_info import Statevector

        from qc_executor import Executor
        from qc_executor.pauli_propagation.pauli_propagation_circuit import (
            PauliPropagationCircuit,
        )

        # PP-native circuit (PP has no abstract-circuit bridge; out of scope).
        qc = PauliPropagationCircuit(2)
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


class TestBackendExpectationParity:
    """Qiskit and PennyLane must agree on the abstract operator's expectation.

    Regression guard for two PennyLane operator bugs (now fixed) that only
    surface for an *asymmetric* observable on a per-qubit-asymmetric state:
      * numeric coefficients were dropped (a bare ``sum`` of Pauli words), and
      * Pauli labels were mapped to wires in reversed (big-endian) order.
    Both are invisible for symmetric/all-``H`` states, so the circuit below gives
    every qubit a distinct rotation.
    """

    @staticmethod
    def _asymmetric_circuit() -> AbstractQuantumCircuit:
        qc = AbstractQuantumCircuit(4)
        qc.h(range(4))
        for q in range(4):
            qc.ry(q, 0.3 * (q + 1))  # distinct angle per qubit -> wire order matters
        qc.cx(0, 1)
        qc.cx(2, 3)
        return qc

    @staticmethod
    def _reference(qc: AbstractQuantumCircuit, sparse: SparsePauliOp, **binds) -> float:
        from qiskit.quantum_info import Statevector

        from qc_executor.qiskit.qiskit_circuit import to_qiskit_circuit

        state = Statevector(to_qiskit_circuit(qc))
        return float(np.real(state.expectation_value(sparse)))

    def _run(self, backend: str, qc, op, **params) -> float:
        from qc_executor import Executor

        ex = Executor.create(backend)
        value = ex.expectation_value(qc, ex.transpile_operator(op), **params)
        return float(np.real_if_close(value))

    @pytest.mark.parametrize("backend", ["qiskit", "pennylane"])
    def test_numeric_observable_matches_reference(self, backend):
        pytest.importorskip(backend)
        qc = self._asymmetric_circuit()
        obs = AbstractQuantumOperator(["IIIZ", "IZIZ", "XIIX"], [0.5, 1.2, 0.8])
        reference = self._reference(qc, _to_sparse(obs))
        assert np.isclose(self._run(backend, qc, obs), reference)

    @pytest.mark.parametrize("backend", ["qiskit", "pennylane"])
    def test_parametrized_observable_matches_reference(self, backend):
        pytest.importorskip(backend)
        qc = self._asymmetric_circuit()
        w = ParameterVector("w", 2)
        obs = AbstractQuantumOperator(["IIIZ", "XIIX"], [2 * w[0], w[1]])
        reference = self._reference(qc, SparsePauliOp(["IIIZ", "XIIX"], coeffs=[2 * 0.4, 0.9]))
        assert np.isclose(self._run(backend, qc, obs, w=[0.4, 0.9]), reference)

    @pytest.mark.parametrize("backend", ["qiskit", "pennylane"])
    def test_raw_abstract_observable_without_transpile(self, backend):
        # The base class auto-converts raw observables (_ensure_native_operator),
        # so expectation_value must accept an AbstractQuantumOperator directly.
        # Regression guard: the qiskit backend used to crash on this call
        # (AttributeError deep in _observable_key) because only pennylane
        # converted internally.
        pytest.importorskip(backend)
        from qc_executor import Executor

        qc = self._asymmetric_circuit()
        obs = AbstractQuantumOperator(["IIIZ", "XIIX"], [0.5, 0.8])
        reference = self._reference(qc, _to_sparse(obs))
        ex = Executor.create(backend)
        value = float(np.real_if_close(ex.expectation_value(qc, obs)))
        assert np.isclose(value, reference)

    def test_sympy_function_coefficient_matches_reference(self):
        # sin(w0) as a coefficient is only expressible in SymPy — a Qiskit
        # ``ParameterExpression`` round-trip cannot represent it. This guards
        # the direct abstract -> PennyLane path (PennyLane-only: the Qiskit
        # backend converts to SparsePauliOp and cannot support this).
        pytest.importorskip("pennylane")
        import sympy as sp

        qc = self._asymmetric_circuit()
        w = ParameterVector("w", 1)
        obs = AbstractQuantumOperator(["IIIZ", "XIIX"], [sp.sin(w[0]), 0.8])
        reference = self._reference(
            qc, SparsePauliOp(["IIIZ", "XIIX"], coeffs=[np.sin(0.7), 0.8])
        )
        assert np.isclose(self._run("pennylane", qc, obs, w=[0.7]), reference)
