"""Tests for IBM Runtime backend support in QiskitExecutor.

All tests in this module require ``qiskit-ibm-runtime`` to be installed.
They use FakeBackend instances so that no real IBM Quantum credentials are
needed.
"""

from random import seed
import numpy as np
import pytest

# Skip the entire module if qiskit-ibm-runtime is not installed
qiskit_ibm_runtime = pytest.importorskip("qiskit_ibm_runtime")

from executor.qiskit.qiskit_executor import (
    QiskitExecutor,
    _is_backend_instance,
    _detect_backend_flags,
)
from executor.qiskit.qiskit_circuit import QiskitCircuit
from executor import Executor, QuantumCircuit
from executor.quantum_operator import QuantumOperator
from executor.utils.qiskit_compat import (
    QISKIT_RUNTIME_AVAILABLE,
    QISKIT_RUNTIME_SMALLER_0_21,
    QISKIT_RUNTIME_SMALLER_0_23,
    QISKIT_RUNTIME_SMALLER_0_28,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_fake_backend():
    """Return a fake IBM backend for testing.

    Tries several fake providers that ship with different qiskit-ibm-runtime
    versions.
    """
    try:
        from qiskit_ibm_runtime.fake_provider import FakeManilaV2

        return FakeManilaV2()
    except ImportError:
        pass
    try:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

        return FakeSherbrooke()
    except ImportError:
        pass
    try:
        from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2

        return FakeAlmadenV2()
    except ImportError:
        pytest.skip("No fake backend available in this qiskit-ibm-runtime version")


def _build_circuit(num_qubits, operations):
    qc = QuantumCircuit(num_qubits)
    for gate_name, gate_args in operations:
        getattr(qc, gate_name)(*gate_args)
    return qc


def _get_fake_session_or_skip():
    backend = _get_fake_backend()
    try:
        session_cls = qiskit_ibm_runtime.Session
        return session_cls(backend=backend)
    except Exception as exc:
        pytest.skip(f"Could not create runtime Session for fake backend: {exc}")


# ---------------------------------------------------------------------------
# Version flag sanity
# ---------------------------------------------------------------------------


class TestVersionFlags:
    def test_runtime_available(self):
        assert QISKIT_RUNTIME_AVAILABLE is True

    def test_runtime_version_flags_are_bool(self):
        assert isinstance(QISKIT_RUNTIME_SMALLER_0_21, bool)
        assert isinstance(QISKIT_RUNTIME_SMALLER_0_23, bool)
        assert isinstance(QISKIT_RUNTIME_SMALLER_0_28, bool)


# ---------------------------------------------------------------------------
# Backend detection helpers
# ---------------------------------------------------------------------------


class TestBackendDetection:
    def test_is_backend_instance_with_fake_backend(self):
        backend = _get_fake_backend()
        assert _is_backend_instance(backend) is True

    def test_is_backend_instance_with_string(self):
        assert _is_backend_instance("statevector") is False

    def test_detect_flags_fake_backend(self):
        backend = _get_fake_backend()
        remote, ibm_quantum = _detect_backend_flags(backend)
        # Fake backends contain "fake" in their string representation
        # and should NOT be detected as remote/ibm_quantum
        assert remote is False
        assert ibm_quantum is False


# ---------------------------------------------------------------------------
# QiskitExecutor with FakeBackend
# ---------------------------------------------------------------------------


class TestQiskitExecutorFakeBackend:
    def test_init_with_fake_backend(self):
        backend = _get_fake_backend()
        executor = QiskitExecutor(backend=backend, shots=1024)
        assert isinstance(executor, QiskitExecutor)
        assert executor.shots == 1024

    def test_remote_property_fake(self):
        backend = _get_fake_backend()
        executor = QiskitExecutor(backend=backend, shots=1024)
        # Fake backends are local
        assert executor.remote is False

    def test_ibm_quantum_property_fake(self):
        backend = _get_fake_backend()
        executor = QiskitExecutor(backend=backend, shots=1024)
        assert executor.ibm_quantum is False

    def test_session_is_none_for_fake_backend(self):
        """Fake backends in job mode should not create a session."""
        backend = _get_fake_backend()
        executor = QiskitExecutor(backend=backend, shots=1024)
        assert executor.session is None

    def test_estimator_is_runtime_type(self):
        """The estimator should be a runtime primitive (V1 or V2 depending on version).

        For runtime < 0.21, fake backends fall back to local BackendEstimator
        because V1 runtime primitives require a QiskitRuntimeService account.
        """
        backend = _get_fake_backend()
        executor = QiskitExecutor(backend=backend, shots=1024)
        if QISKIT_RUNTIME_SMALLER_0_21:
            # Fallback to local primitive for fake backends
            from qiskit.primitives import BaseEstimatorV1 as _BaseV1

            assert isinstance(executor._estimator, _BaseV1)
        elif QISKIT_RUNTIME_SMALLER_0_28:
            from qiskit_ibm_runtime import EstimatorV2 as ExpectedEstimator

            assert isinstance(executor._estimator, ExpectedEstimator)
        else:
            # >= 0.28: Estimator IS V2
            from qiskit_ibm_runtime import Estimator as ExpectedEstimator

            assert isinstance(executor._estimator, ExpectedEstimator)

    def test_sampler_is_runtime_type(self):
        """The sampler should be a runtime primitive (V1 or V2 depending on version).

        For runtime < 0.21, fake backends fall back to local BackendSampler
        because V1 runtime primitives require a QiskitRuntimeService account.
        """
        backend = _get_fake_backend()
        executor = QiskitExecutor(backend=backend, shots=1024)
        if QISKIT_RUNTIME_SMALLER_0_21:
            from qiskit.primitives import BaseSamplerV1 as _BaseV1

            assert isinstance(executor._sampler, _BaseV1)
        elif QISKIT_RUNTIME_SMALLER_0_28:
            from qiskit_ibm_runtime import SamplerV2 as ExpectedSampler

            assert isinstance(executor._sampler, ExpectedSampler)
        else:
            from qiskit_ibm_runtime import Sampler as ExpectedSampler

            assert isinstance(executor._sampler, ExpectedSampler)

    def test_context_manager(self):
        """Test that the executor can be used as a context manager."""
        backend = _get_fake_backend()
        with QiskitExecutor(backend=backend, shots=1024) as executor:
            assert isinstance(executor, QiskitExecutor)

    def test_invalid_backend_string_raises(self):
        with pytest.raises(ValueError, match="Unknown backend string"):
            QiskitExecutor(backend="non_existent")

    def test_invalid_backend_type_raises(self):
        with pytest.raises(TypeError, match="Backend instance"):
            QiskitExecutor(backend=42)

    def test_init_with_injected_session_backend(self):
        session = _get_fake_session_or_skip()
        executor = QiskitExecutor(backend=session, shots=1024)

        assert executor.session is session

    def test_close_session_closes_injected_session(self):
        session = _get_fake_session_or_skip()
        executor = QiskitExecutor(backend=session, shots=1024)

        executor.close_session()
        assert executor.session is None

    def test_session_with_execution_mode_raises(self):
        session = _get_fake_session_or_skip()
        with pytest.raises(ValueError, match="must remain 'job'"):
            QiskitExecutor(backend=session, shots=1024, execution_mode="session")

    def test_statevector_raises_on_remote(self):
        """Statevector is not available on remote backends.

        We simulate this by manually setting the flag.
        """
        backend = _get_fake_backend()
        executor = QiskitExecutor(backend=backend, shots=1024)
        # Force remote flag for this test
        executor._remote_backend = True
        qc = _build_circuit(1, [("h", [0])])
        with pytest.raises(RuntimeError, match="remote"):
            executor.statevector(qc)


# ---------------------------------------------------------------------------
# Transpilation
# ---------------------------------------------------------------------------


class TestTranspilationFakeBackend:
    def test_transpile_circuit_returns_qiskit_circuit(self):
        backend = _get_fake_backend()
        executor = QiskitExecutor(backend=backend, shots=1024)
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        transpiled = executor.transpile_circuit(qc)
        assert isinstance(transpiled, QiskitCircuit)

    def test_transpiled_circuit_has_qubits(self):
        backend = _get_fake_backend()
        executor = QiskitExecutor(backend=backend, shots=1024)
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        transpiled = executor.transpile_circuit(qc)
        # Should have at least as many qubits as backend
        assert transpiled.num_qubits >= 2


# ---------------------------------------------------------------------------
# QiskitCircuit._from_qiskit
# ---------------------------------------------------------------------------


class TestQiskitCircuitFromQiskit:
    def test_from_qiskit_roundtrip(self):
        from qiskit.circuit import QuantumCircuit as QiskitQC

        qc = QiskitQC(2)
        qc.h(0)
        qc.cx(0, 1)
        wrapper = QiskitCircuit._from_qiskit(qc)
        assert isinstance(wrapper, QiskitCircuit)
        assert wrapper.num_qubits == 2

    def test_from_qiskit_with_parameters(self):
        from qiskit.circuit import QuantumCircuit as QiskitQC, ParameterVector

        p = ParameterVector("p", 2)
        qc = QiskitQC(1)
        qc.rx(p[0], 0)
        qc.ry(p[1], 0)
        wrapper = QiskitCircuit._from_qiskit(qc)
        assert len(wrapper.free_parameters) == 2
        assert "p" in wrapper.parameter_names


# ---------------------------------------------------------------------------
# Execution on FakeBackend (integration-level)
# ---------------------------------------------------------------------------


class TestExecutionFakeBackend:
    """Integration tests that actually run circuits on a fake backend.

    These tests are slower because the fake backend simulates noise.
    """

    def test_expectation_value_bell_zz(self):
        backend = _get_fake_backend()
        executor = QiskitExecutor(backend=backend, shots=4096, seed=42)
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        op = QuantumOperator(["ZZ"], [1.0])
        result = executor.expectation_value(qc, op)
        # Noisy result should still be close to 1.0 for ZZ on Bell state
        assert isinstance(result, (float, np.floating, np.ndarray))

    def test_sample_bell_state(self):
        backend = _get_fake_backend()
        executor = QiskitExecutor(backend=backend, shots=1024, seed=42)
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        result = executor.sample(qc)
        assert isinstance(result, dict)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Factory auto-detection tests
# ---------------------------------------------------------------------------


class TestExecutorFactoryQiskitAutoDetection:
    """Test that Executor.create() auto-detects QiskitExecutor for native
    Qiskit objects (Backend, Session, Batch, primitives).

    All tests use fake / local backends so no IBM Quantum credentials are
    required.
    """

    def test_create_with_fake_backend(self):
        """Executor.create() should return a QiskitExecutor for a BackendV2."""
        from executor.qiskit import QiskitExecutor

        backend = _get_fake_backend()
        executor = Executor.create(backend, shots=1024, seed=42)

        assert isinstance(executor, QiskitExecutor)
        assert executor.shots == 1024
        assert executor._seed == 42
        assert executor.remote is False
        assert executor.ibm_quantum is False

    def test_create_with_session(self):
        """Executor.create() should return a QiskitExecutor for a Session."""
        from executor.qiskit import QiskitExecutor

        session = _get_fake_session_or_skip()
        try:
            executor = Executor.create(session, shots=1024)
            assert isinstance(executor, QiskitExecutor)
            assert executor.session is session
            assert executor.shots == 1024
        finally:
            session.close()

    def test_create_with_statevector_estimator(self):
        """Executor.create() should accept a StatevectorEstimator directly."""
        from qiskit.primitives import StatevectorEstimator
        from executor.qiskit import QiskitExecutor

        estimator = StatevectorEstimator()
        executor = Executor.create(estimator)

        assert isinstance(executor, QiskitExecutor)

    def test_create_with_statevector_sampler(self):
        """Executor.create() should accept a StatevectorSampler directly."""
        from qiskit.primitives import StatevectorSampler
        from executor.qiskit import QiskitExecutor

        sampler = StatevectorSampler()
        executor = Executor.create(sampler)

        assert isinstance(executor, QiskitExecutor)

    def test_create_with_backend_estimator(self):
        """Executor.create() should accept a BackendEstimatorV2."""
        from executor.qiskit import QiskitExecutor
        from executor.utils.qiskit_compat import QISKIT_SMALLER_1_2

        backend = _get_fake_backend()
        if QISKIT_SMALLER_1_2:
            from qiskit.primitives import BackendEstimator as BackendEstimatorCls
        else:
            from qiskit.primitives import BackendEstimatorV2 as BackendEstimatorCls

        estimator = BackendEstimatorCls(backend=backend)
        executor = Executor.create(estimator, shots=256)

        assert isinstance(executor, QiskitExecutor)

    def test_create_with_runtime_estimator(self):
        """Executor.create() should accept a runtime Estimator (V1 or V2).

        V1 runtime primitives (runtime < 0.21) require IBM Quantum credentials
        even for fake backends and cannot be tested without an account.
        """
        if QISKIT_RUNTIME_SMALLER_0_21:
            pytest.skip(
                "Runtime V1 primitives require IBM Quantum credentials; "
                "cannot be instantiated with a fake backend."
            )

        from executor.qiskit import QiskitExecutor

        backend = _get_fake_backend()
        if QISKIT_RUNTIME_SMALLER_0_28:
            from qiskit_ibm_runtime import EstimatorV2

            estimator = EstimatorV2(mode=backend)
        else:
            from qiskit_ibm_runtime import Estimator as EstimatorV2

            estimator = EstimatorV2(mode=backend)

        executor = Executor.create(estimator, shots=256)
        assert isinstance(executor, QiskitExecutor)

    def test_create_with_runtime_sampler(self):
        """Executor.create() should accept a runtime Sampler (V1 or V2).

        V1 runtime primitives (runtime < 0.21) require IBM Quantum credentials
        even for fake backends and cannot be tested without an account.
        """
        if QISKIT_RUNTIME_SMALLER_0_21:
            pytest.skip(
                "Runtime V1 primitives require IBM Quantum credentials; "
                "cannot be instantiated with a fake backend."
            )

        from executor.qiskit import QiskitExecutor

        backend = _get_fake_backend()
        if QISKIT_RUNTIME_SMALLER_0_28:
            from qiskit_ibm_runtime import SamplerV2

            sampler = SamplerV2(mode=backend)
        else:
            from qiskit_ibm_runtime import Sampler as SamplerV2

            sampler = SamplerV2(mode=backend)

        executor = Executor.create(sampler, shots=256)
        assert isinstance(executor, QiskitExecutor)

    def test_get_accepted_types_contains_backend(self):
        """get_accepted_types() must include qiskit.providers.Backend."""
        from qiskit.providers import Backend
        from executor.qiskit import QiskitExecutor

        accepted = QiskitExecutor.get_accepted_backend_types()

        assert any(issubclass(Backend, t) or t is Backend for t in accepted)

    def test_get_accepted_types_contains_statevector_primitives(self):
        """get_accepted_types() must include the built-in statevector primitives."""
        from qiskit.primitives import StatevectorEstimator, StatevectorSampler
        from executor.qiskit import QiskitExecutor

        accepted = QiskitExecutor.get_accepted_backend_types()

        assert StatevectorEstimator in accepted
        assert StatevectorSampler in accepted

    def test_get_accepted_types_contains_session_and_batch(self):
        """get_accepted_types() must include Session and Batch."""
        from qiskit_ibm_runtime import Session, Batch
        from executor.qiskit import QiskitExecutor

        accepted = QiskitExecutor.get_accepted_backend_types()

        assert Session in accepted
        assert Batch in accepted

    def test_get_accepted_types_no_dummy_classes(self):
        """Dummy sentinel classes must not appear in get_accepted_types()."""
        from executor.qiskit import QiskitExecutor

        accepted = QiskitExecutor.get_accepted_backend_types()

        # All returned types must originate from a real package, not from
        # the executor module itself.
        executor_module = "executor.qiskit.qiskit_executor"
        for t in accepted:
            assert (
                getattr(t, "__module__", "") != executor_module
            ), f"Dummy class {t.__name__} must not appear in get_accepted_types()"

    def test_get_accepted_types_returns_list_of_types(self):
        """get_accepted_types() must return a list of actual type objects."""
        from executor.qiskit import QiskitExecutor

        accepted = QiskitExecutor.get_accepted_backend_types()

        assert isinstance(accepted, list)
        assert len(accepted) > 0
        for t in accepted:
            assert isinstance(t, type), f"Expected type, got {t!r}"

    def test_fake_backend_isinstance_check_against_accepted_types(self):
        """A FakeBackend instance must match at least one accepted type."""
        from executor.qiskit import QiskitExecutor

        backend = _get_fake_backend()
        accepted = QiskitExecutor.get_accepted_backend_types()

        assert any(isinstance(backend, t) for t in accepted)
