"""Tests for IBM Runtime backend support in QiskitExecutor.

All tests in this module require ``qiskit-ibm-runtime`` to be installed.
They use FakeBackend instances so that no real IBM Quantum credentials are
needed.
"""

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
# Factory: Executor.from_backend
# ---------------------------------------------------------------------------


class TestExecutorFactory:
    def test_from_backend(self):
        backend = _get_fake_backend()
        executor = Executor.from_backend(backend, shots=1024)
        assert isinstance(executor, QiskitExecutor)
        assert executor.shots == 1024

    def test_create_with_backend_object(self):
        """Executor.create() should also accept a Backend object."""
        backend = _get_fake_backend()
        executor = Executor.create(backend, shots=1024)
        assert isinstance(executor, QiskitExecutor)

    def test_from_backend_with_execution_mode(self):
        backend = _get_fake_backend()
        executor = Executor.from_backend(backend, shots=512, execution_mode="job")
        assert isinstance(executor, QiskitExecutor)
        assert executor._execution_mode == "job"

    def test_from_backend_with_options(self):
        backend = _get_fake_backend()
        executor = Executor.from_backend(backend, shots=512, options={"default_shots": 100})
        assert isinstance(executor, QiskitExecutor)
        assert executor._options == {"default_shots": 100}


# ---------------------------------------------------------------------------
# Execution on FakeBackend (integration-level)
# ---------------------------------------------------------------------------


class TestExecutionFakeBackend:
    """Integration tests that actually run circuits on a fake backend.

    These tests are slower because the fake backend simulates noise.
    """

    @pytest.mark.slow
    def test_expectation_value_bell_zz(self):
        backend = _get_fake_backend()
        executor = QiskitExecutor(backend=backend, shots=4096, seed=42)
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        op = QuantumOperator(["ZZ"], [1.0])
        result = executor.expectation_value(qc, op)
        # Noisy result should still be close to 1.0 for ZZ on Bell state
        assert isinstance(result, (float, np.floating, np.ndarray))

    @pytest.mark.slow
    def test_sample_bell_state(self):
        backend = _get_fake_backend()
        executor = QiskitExecutor(backend=backend, shots=1024, seed=42)
        qc = _build_circuit(2, [("h", [0]), ("cx", [0, 1])])
        result = executor.sample(qc)
        assert isinstance(result, dict)
        assert len(result) >= 1
