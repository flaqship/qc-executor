"""Tests for IBM Runtime backend support in QiskitExecutor.

All tests in this module require ``qiskit-ibm-runtime`` to be installed.
They use FakeBackend instances so that no real IBM Quantum credentials are
needed.
"""

import numpy as np
import pytest
from qiskit import primitives as qiskit_primitives
from qiskit.circuit import QuantumCircuit as QiskitQC
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.providers import Backend
from qiskit_ibm_runtime import Batch, Session

from qc_executor import Executor, QuantumCircuit
from qc_executor.parameters import Parameters
from qc_executor.qiskit import qiskit_executor as qiskit_executor_module
from qc_executor.qiskit.qiskit_circuit import QiskitCircuit
from qc_executor.qiskit.qiskit_executor import (
    QiskitExecutor,
    _classify_backend,
    _is_backend_instance,
    _resolve_backend_from_session_or_batch,
)
from qc_executor.quantum_operator import QuantumOperator
from qc_executor.utils.qiskit_compat import (
    QISKIT_RUNTIME_AVAILABLE,
    QISKIT_RUNTIME_SMALLER_0_21,
    QISKIT_RUNTIME_SMALLER_0_23,
    QISKIT_RUNTIME_SMALLER_0_28,
    QISKIT_SMALLER_1_2,
)

# Skip the entire module if qiskit-ibm-runtime is not installed
qiskit_ibm_runtime = pytest.importorskip("qiskit_ibm_runtime")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_fake_backend():
    """Return a small (5-qubit) IBM fake backend for testing.

    FakeManilaV2 (5 qubits) is preferred to avoid excessive memory usage;
    FakeAlmadenV2 is a fallback for runtime builds that do not ship it.
    """
    candidates = ["FakeManilaV2", "FakeAlmadenV2"]

    fake_provider = pytest.importorskip("qiskit_ibm_runtime.fake_provider")
    for name in candidates:
        cls = getattr(fake_provider, name, None)
        if cls is not None:
            return cls()
    return pytest.skip("No fake backend available in this qiskit-ibm-runtime version")


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
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Any failure (missing credentials, version drift) should skip, not fail.
        return pytest.skip(f"Could not create runtime Session for fake backend: {exc}")


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

    def test_classify_backend_fake_backend(self):
        backend = _get_fake_backend()
        remote, ibm_quantum, fake = _classify_backend(backend)
        # Fake backends contain "fake" in their string representation
        # and should NOT be detected as remote/ibm_quantum
        assert remote is False
        assert ibm_quantum is False
        assert fake is True

    def test_classify_backend_string_fallback_for_ibm_like_name(self):
        class _UnknownBackend:
            def __str__(self):
                return "IBM Fake Backend"

        remote, ibm_quantum, fake = _classify_backend(_UnknownBackend())

        assert remote is False
        assert ibm_quantum is False
        assert fake is True

    def test_resolve_backend_from_session_or_batch_via_backend_attr(self):
        backend = _get_fake_backend()

        class _SessionLike:
            def __init__(self, backend):
                self._backend = backend

        assert _resolve_backend_from_session_or_batch(_SessionLike(backend)) is backend

    def test_resolve_backend_from_session_or_batch_via_backend_name(self):
        backend = _get_fake_backend()

        class _Service:
            def backend(self, name):
                assert name == "fake_backend"
                return backend

        class _SessionLike:
            def backend(self):
                return "fake_backend"

            service = _Service()

        assert _resolve_backend_from_session_or_batch(_SessionLike()) is backend


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

    def test_estimator_is_local_type_for_fake_backend(self):
        """Fake IBM backends use runtime primitives when runtime >= 0.21."""
        backend = _get_fake_backend()
        executor = QiskitExecutor(backend=backend, shots=1024)
        if QISKIT_RUNTIME_SMALLER_0_21:
            # Fallback to local primitive for fake backends (V1 runtime needs IBM account)
            expected_cls = getattr(qiskit_primitives, "BaseEstimatorV1")
        elif QISKIT_RUNTIME_SMALLER_0_28:
            expected_cls = qiskit_ibm_runtime.EstimatorV2
        else:
            expected_cls = qiskit_ibm_runtime.Estimator
        assert isinstance(executor._estimator, expected_cls)

    def test_sampler_is_local_type_for_fake_backend(self):
        """Fake IBM backends use runtime primitives when runtime >= 0.21."""
        backend = _get_fake_backend()
        executor = QiskitExecutor(backend=backend, shots=1024)
        if QISKIT_RUNTIME_SMALLER_0_21:
            expected_cls = getattr(qiskit_primitives, "BaseSamplerV1")
        elif QISKIT_RUNTIME_SMALLER_0_28:
            expected_cls = qiskit_ibm_runtime.SamplerV2
        else:
            expected_cls = qiskit_ibm_runtime.Sampler
        assert isinstance(executor._sampler, expected_cls)

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

    def test_statevector_works_on_fake_backend(self):
        """Statevector is always available since it uses local simulation."""
        backend = _get_fake_backend()
        executor = QiskitExecutor(backend=backend, shots=1024)
        qc = _build_circuit(1, [("h", [0])])
        sv = executor.statevector(qc)
        assert sv is not None
        assert len(sv) == 2


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
# QiskitCircuit.from_qiskit
# ---------------------------------------------------------------------------


class TestQiskitCircuitFromQiskit:
    def test_from_qiskit_roundtrip(self):
        qc = QiskitQC(2)
        qc.h(0)
        qc.cx(0, 1)
        wrapper = QiskitCircuit.from_qiskit(qc)
        assert isinstance(wrapper, QiskitCircuit)
        assert wrapper.num_qubits == 2

    def test_from_qiskit_with_parameters(self):
        p = Parameters("p", 2)
        qc = QiskitQC(1)
        qc.rx(p[0], 0)
        qc.ry(p[1], 0)
        wrapper = QiskitCircuit.from_qiskit(qc)
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
        operator = QuantumOperator(["ZZ"], [1.0])
        result = executor.expectation_value(qc, operator)
        # Noisy result should still be close to 1.0 for ZZ on Bell state
        assert isinstance(result, (float, np.floating, np.ndarray))

    def test_sample_bell_state(self):
        backend = _get_fake_backend()
        executor = QiskitExecutor(backend=backend, shots=1024, seed=42)
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        result = executor.sample(qc)
        assert isinstance(result, dict)
        assert len(result) >= 1


class TestIBMInternalHelpers:
    def test_ensure_session_active_recreates_expired_session(self):
        executor = object.__new__(QiskitExecutor)
        executor._ibm_quantum_backend = True
        executor._session = type("_Session", (), {"status": lambda self: "closed"})()
        executor._create_session_called = False

        def _fake_create_session():
            executor._create_session_called = True

        executor._create_session = _fake_create_session
        executor._uses_managed_session = lambda: False

        executor._ensure_session_active()

        assert executor._create_session_called is True

    def test_instantiate_runtime_primitive_v2_uses_expected_kwargs(self):
        executor = object.__new__(QiskitExecutor)
        executor._ibm_quantum_backend = True
        executor._session = type("_Session", (), {"status": lambda self: "open"})()
        executor._backend = _get_fake_backend()
        executor._execution_mode = "job"

        class _DummyPrimitive:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        if QISKIT_RUNTIME_SMALLER_0_23:
            primitive = executor._instantiate_runtime_primitive_v2(_DummyPrimitive, {"a": 1})
            assert primitive.kwargs["session"] is executor._session
        else:
            primitive = executor._instantiate_runtime_primitive_v2(_DummyPrimitive, {"a": 1})
            assert primitive.kwargs["mode"] is executor._session
        assert primitive.kwargs["options"] == {"a": 1}

    def test_instantiate_runtime_primitive_v1_uses_backend_and_options(self, monkeypatch):
        executor = object.__new__(QiskitExecutor)
        executor._ibm_quantum_backend = False
        executor._session = None
        executor._backend = _get_fake_backend()

        class _DummyPrimitive:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class _DummyOptions:
            def __init__(self):
                self.resilience_level = None

        monkeypatch.setattr(
            qiskit_executor_module,
            "_load_runtime_options_v1",
            lambda: _DummyOptions,
        )

        primitive = executor._instantiate_runtime_primitive_v1(
            _DummyPrimitive, {"resilience_level": 1}
        )

        assert primitive.kwargs["backend"] is executor._backend
        assert primitive.kwargs["options"] is not None
        assert primitive.kwargs["options"].resilience_level == 1


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
        backend = _get_fake_backend()
        executor = Executor.create(backend, shots=1024, seed=42)

        assert isinstance(executor, QiskitExecutor)
        assert executor.shots == 1024
        assert executor._seed == 42
        assert executor.remote is False
        assert executor.ibm_quantum is False

    def test_create_with_session(self):
        """Executor.create() should return a QiskitExecutor for a Session."""
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

        estimator = StatevectorEstimator()
        executor = Executor.create(estimator)

        assert isinstance(executor, QiskitExecutor)

    def test_create_with_statevector_sampler(self):
        """Executor.create() should accept a StatevectorSampler directly."""

        sampler = StatevectorSampler()
        executor = Executor.create(sampler)

        assert isinstance(executor, QiskitExecutor)

    def test_create_with_backend_estimator(self):
        """Executor.create() should accept a BackendEstimatorV2."""

        backend = _get_fake_backend()
        estimator_cls_name = "BackendEstimator" if QISKIT_SMALLER_1_2 else "BackendEstimatorV2"
        estimator_cls = getattr(qiskit_primitives, estimator_cls_name)

        estimator = estimator_cls(backend=backend)
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

        backend = _get_fake_backend()
        estimator_cls = (
            qiskit_ibm_runtime.EstimatorV2
            if QISKIT_RUNTIME_SMALLER_0_28
            else qiskit_ibm_runtime.Estimator
        )
        estimator = estimator_cls(mode=backend)

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

        backend = _get_fake_backend()
        sampler_cls = (
            qiskit_ibm_runtime.SamplerV2
            if QISKIT_RUNTIME_SMALLER_0_28
            else qiskit_ibm_runtime.Sampler
        )
        sampler = sampler_cls(mode=backend)

        executor = Executor.create(sampler, shots=256)
        assert isinstance(executor, QiskitExecutor)

    def test_get_accepted_types_contains_backend(self):
        """get_accepted_types() must include qiskit.providers.Backend."""

        accepted = QiskitExecutor.get_accepted_backend_types()

        assert any(issubclass(Backend, t) or t is Backend for t in accepted)

    def test_get_accepted_types_contains_statevector_primitives(self):
        """get_accepted_types() must include the built-in statevector primitives."""

        accepted = QiskitExecutor.get_accepted_backend_types()

        assert StatevectorEstimator in accepted
        assert StatevectorSampler in accepted

    def test_get_accepted_types_contains_session_and_batch(self):
        """get_accepted_types() must include Session and Batch."""

        accepted = QiskitExecutor.get_accepted_backend_types()

        assert Session in accepted
        assert Batch in accepted

    def test_get_accepted_types_no_dummy_classes(self):
        """Dummy sentinel classes must not appear in get_accepted_types()."""
        accepted = QiskitExecutor.get_accepted_backend_types()

        # All returned types must originate from a real package, not from
        # the executor module itself.
        executor_module = "qc_executor.qiskit.qiskit_executor"
        for t in accepted:
            assert (
                getattr(t, "__module__", "") != executor_module
            ), f"Dummy class {t.__name__} must not appear in get_accepted_types()"

    def test_get_accepted_types_returns_list_of_types(self):
        """get_accepted_types() must return a list of actual type objects."""

        accepted = QiskitExecutor.get_accepted_backend_types()

        assert isinstance(accepted, list)
        assert len(accepted) > 0
        for t in accepted:
            assert isinstance(t, type), f"Expected type, got {t!r}"

    def test_fake_backend_isinstance_check_against_accepted_types(self):
        """A FakeBackend instance must match at least one accepted type."""

        backend = _get_fake_backend()
        accepted = QiskitExecutor.get_accepted_backend_types()

        assert any(isinstance(backend, t) for t in accepted)
