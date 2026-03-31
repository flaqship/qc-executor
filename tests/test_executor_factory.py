"""Tests for the Executor factory class."""

import pytest

from executor.base.executor_base import ExecutorBase
from executor.factory import Executor


class MockExecutor(ExecutorBase):
    """Mock executor for testing."""

    def __init__(self, shots=None, **kwargs):
        """Initialize mock executor."""
        super().__init__(shots=shots, **kwargs)

    def _expectation_value(self, circuit, operator, parameters=None):
        """Mock implementation."""
        return 0.5

    def _expectation_value_derivatives(self, circuit, operator, parameters=None):
        """Mock implementation."""
        import numpy as np

        return np.array([0.1, 0.2])

    def _sample(self, circuit, parameters=None):
        """Mock implementation."""
        return {"00": 500, "11": 500}

    def _statevector(self, circuit, parameters=None):
        """Mock implementation."""
        import numpy as np

        return np.array([1.0, 0.0])

    def _transpile_circuit(self, circuit):
        """Mock implementation."""
        return circuit

    def _transpile_operator(self, operator):
        """Mock implementation."""
        return operator

    @classmethod
    def get_accepted_backend_types(cls) -> list[type]:
        """Mock accepted backend types."""
        return []


class TestExecutorFactory:
    """Test cases for Executor factory."""

    def test_executor_not_instantiable(self):
        """Test that Executor cannot be instantiated directly."""
        with pytest.raises(TypeError, match="cannot be instantiated"):
            Executor()

    def test_register_decorator(self):
        """Test that @Executor.register decorator works."""

        # Register a mock backend
        @Executor.register("mock_test")
        class TestExecutor(ExecutorBase):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def _expectation_value(self, circuit, operator, parameters=None):
                return 0.0

            def _expectation_value_derivatives(self, circuit, operator, parameters=None):
                import numpy as np

                return np.array([0.0])

            def _sample(self, circuit, parameters=None):
                return {}

            def _statevector(self, circuit, parameters=None):
                import numpy as np

                return np.array([1.0])

            def _transpile_circuit(self, circuit):
                return circuit

        # Check if backend is registered
        assert "mock_test" in Executor.available_backends()

        # Clean up
        if "mock_test" in Executor._registry:
            del Executor._registry["mock_test"]

    def test_register_validates_base_class(self):
        """Test that register checks for ExecutorBase inheritance."""
        with pytest.raises(TypeError, match="must inherit from ExecutorBase"):

            @Executor.register("invalid")
            class InvalidExecutor:
                pass

    def test_available_backends_empty(self):
        """Test available_backends when no backends are registered."""
        # Save current registry
        original_registry = Executor._registry.copy()
        original_alias_map = Executor._backend_alias_map.copy()
        original_alias_registry_size = Executor._alias_registry_size
        original_discovered = Executor._plugins_discovered

        try:
            # Clear registry
            Executor._registry.clear()
            Executor._plugins_discovered = False

            # Discover plugins (will populate registry)
            backends = Executor.available_backends()

            # Should have at least one backend after discovery
            assert isinstance(backends, list)

        finally:
            # Restore original registry
            Executor._registry = original_registry
            Executor._backend_alias_map = original_alias_map
            Executor._alias_registry_size = original_alias_registry_size
            Executor._plugins_discovered = original_discovered

    def test_create_unknown_backend(self):
        """Test that create raises helpful error for unknown backend."""
        with pytest.raises(ValueError, match="Backend 'nonexistent' not found"):
            Executor.create("nonexistent")

    def test_create_unknown_backend_message_includes_available(self):
        """Test that error message includes available backends."""
        with pytest.raises(ValueError) as exc_info:
            Executor.create("nonexistent_backend_xyz")

        error_msg = str(exc_info.value)
        assert "Available backends:" in error_msg
        assert "Known backend aliases:" in error_msg
        # No install hint for completely unknown backend names
        assert "pip install" not in error_msg

    def test_create_qiskit_missing_message_uses_qiskit_full_extra(self):
        """Test that qiskit backend install hint points to qiskit-full."""
        original_registry = Executor._registry.copy()
        original_alias_map = Executor._backend_alias_map.copy()
        original_alias_registry_size = Executor._alias_registry_size
        original_discovered = Executor._plugins_discovered

        try:
            Executor._registry = {k: v for k, v in Executor._registry.items() if k != "qiskit"}
            Executor._plugins_discovered = True

            with pytest.raises(ValueError) as exc_info:
                Executor.create("qiskit")

            assert "pip install executor[qiskit-full]" in str(exc_info.value)
        finally:
            Executor._registry = original_registry
            Executor._backend_alias_map = original_alias_map
            Executor._alias_registry_size = original_alias_registry_size
            Executor._plugins_discovered = original_discovered

    def test_create_with_kwargs(self):
        """Test that create passes kwargs to backend constructor."""

        # Register mock backend
        @Executor.register("mock_kwargs")
        class MockKwargsExecutor(MockExecutor):
            pass

        try:
            # Create with kwargs
            executor = Executor.create("mock_kwargs", shots=1024, seed=42)

            # Verify it's the right type
            assert isinstance(executor, MockKwargsExecutor)
            assert executor.shots == 1024

        finally:
            # Clean up
            if "mock_kwargs" in Executor._registry:
                del Executor._registry["mock_kwargs"]


class TestExecutorIntegration:
    """Integration tests with actual backends."""

    def test_qiskit_in_available_backends(self):
        """Test that qiskit backend is available."""
        backends = Executor.available_backends()
        assert "qiskit" in backends

    def test_create_qiskit(self):
        """Test creating QiskitExecutor via factory."""
        executor = Executor.create("qiskit")

        from executor.qiskit import QiskitExecutor

        assert isinstance(executor, QiskitExecutor)
        assert executor.shots is None

    def test_create_qiskit_via_statevector_alias(self):
        """Test creating QiskitExecutor using the statevector alias."""
        from executor.qiskit import QiskitExecutor

        executor = Executor.create("statevector")
        assert isinstance(executor, QiskitExecutor)

    def test_create_qiskit_via_aer_alias(self):
        """Test creating QiskitExecutor using the aer alias."""
        try:
            import qiskit_aer  # noqa: F401
        except ImportError:
            pytest.skip("qiskit-aer not installed")

        from executor.qiskit import QiskitExecutor

        executor = Executor.create("aer")
        assert isinstance(executor, QiskitExecutor)

    def test_autodetect_pennylane_device_instance(self):
        """Test auto-detection when passing a PennyLane device instance."""
        try:
            import pennylane as qml
        except ImportError:
            pytest.skip("PennyLane not installed")

        from executor.pennylane import PennyLaneExecutor

        dev = qml.device("default.qubit", wires=1)
        executor = Executor.create(dev)

        assert isinstance(executor, PennyLaneExecutor)
        assert executor._device is dev

    def test_create_pennylane_via_device_string_alias(self):
        """Test creating PennyLaneExecutor using a device string alias."""
        try:
            import pennylane as qml  # noqa: F401
        except ImportError:
            pytest.skip("PennyLane not installed")

        from executor.pennylane import PennyLaneExecutor

        executor = Executor.create("default.qubit", wires=1)
        assert isinstance(executor, PennyLaneExecutor)

    def test_pennylane_backend_available(self):
        """Test that pennylane backend is available if installed."""
        backends = Executor.available_backends()

        try:
            import pennylane

            assert "pennylane" in backends
        except ImportError:
            # PennyLane not installed - should not be in backends
            assert "pennylane" not in backends

    def test_qulacs_backend_available(self):
        """Test that qulacs backend is available if installed."""
        backends = Executor.available_backends()

        try:
            import qulacs

            assert "qulacs" in backends
        except ImportError:
            # Qulacs not installed - should not be in backends
            assert "qulacs" not in backends


class TestExecutorBackendSwitching:
    """Test backend switching functionality."""

    def test_get_config(self):
        """Test that get_config returns executor configuration."""
        executor = Executor.create("qiskit", shots=1024, seed=42, caching=True)

        config = executor.get_config()

        assert config["shots"] == 1024
        assert config["seed"] == 42
        assert config["caching"] is True
        assert "log_level" in config
        assert "cache_dir" in config

    def test_switch_backend_preserves_config(self):
        """Test that switch_backend preserves configuration."""
        executor = Executor.create("qiskit", shots=1024, seed=42)

        # Switch to pennylane (if available)
        try:
            import pennylane

            new_executor = executor.switch_backend("pennylane")

            assert new_executor.shots == 1024
            assert new_executor._seed == 42

            from executor.pennylane import PennyLaneExecutor

            assert isinstance(new_executor, PennyLaneExecutor)
        except ImportError:
            # PennyLane not installed - just verify method exists
            assert hasattr(executor, "switch_backend")

    def test_switch_backend_with_overrides(self):
        """Test that switch_backend can override configuration."""
        executor = Executor.create("qiskit", shots=1024, seed=42)

        # Switch with overrides
        try:
            import pennylane

            new_executor = executor.switch_backend("pennylane", shots=2048, seed=99)

            assert new_executor.shots == 2048
            assert new_executor._seed == 99
        except ImportError:
            # PennyLane not installed - skip this test
            pass


class TestExecutorAliasRegistration:
    """Tests for backend alias indexing, duplicate detection and rebuild safety."""

    def test_register_duplicate_alias_raises(self):
        original_registry = Executor._registry.copy()
        original_alias_map = Executor._backend_alias_map.copy()
        original_alias_registry_size = Executor._alias_registry_size
        original_discovered = Executor._plugins_discovered

        try:
            Executor._registry = {}
            Executor._backend_alias_map = {}
            Executor._alias_registry_size = 0
            Executor._plugins_discovered = True

            @Executor.register("dup_a")
            class DupA(MockExecutor):
                @classmethod
                def get_accepted_backend_aliases(cls) -> list[str]:
                    return ["dup.alias"]

            with pytest.raises(ValueError, match="Duplicate backend alias 'dup.alias'"):

                @Executor.register("dup_b")
                class DupB(MockExecutor):
                    @classmethod
                    def get_accepted_backend_aliases(cls) -> list[str]:
                        return ["dup.alias"]

        finally:
            Executor._registry = original_registry
            Executor._backend_alias_map = original_alias_map
            Executor._alias_registry_size = original_alias_registry_size
            Executor._plugins_discovered = original_discovered

    def test_create_uses_alias_routing(self):
        original_registry = Executor._registry.copy()
        original_alias_map = Executor._backend_alias_map.copy()
        original_alias_registry_size = Executor._alias_registry_size
        original_discovered = Executor._plugins_discovered

        class AliasExecutor(MockExecutor):
            def __init__(self, backend=None, **kwargs):
                super().__init__(**kwargs)
                self.backend = backend

            @classmethod
            def get_accepted_backend_aliases(cls) -> list[str]:
                return ["alpha.backend"]

        try:
            Executor._registry = {}
            Executor._backend_alias_map = {}
            Executor._alias_registry_size = 0
            Executor._plugins_discovered = True

            Executor.register("alias_backend")(AliasExecutor)
            executor = Executor.create("alpha.backend", shots=5)

            assert isinstance(executor, AliasExecutor)
            assert executor.backend == "alpha.backend"
            assert executor.shots == 5
        finally:
            Executor._registry = original_registry
            Executor._backend_alias_map = original_alias_map
            Executor._alias_registry_size = original_alias_registry_size
            Executor._plugins_discovered = original_discovered

    def test_rebuild_alias_map_after_registry_direct_change(self):
        original_registry = Executor._registry.copy()
        original_alias_map = Executor._backend_alias_map.copy()
        original_alias_registry_size = Executor._alias_registry_size
        original_discovered = Executor._plugins_discovered

        class RebuildExecutor(MockExecutor):
            def __init__(self, backend=None, **kwargs):
                super().__init__(**kwargs)
                self.backend = backend

            @classmethod
            def get_accepted_backend_aliases(cls) -> list[str]:
                return ["rebuild.alias"]

        try:
            Executor._registry = {}
            Executor._backend_alias_map = {}
            Executor._alias_registry_size = 0
            Executor._plugins_discovered = True

            # Direct registry mutation simulates existing tests/integrations.
            Executor._registry["rebuilder"] = RebuildExecutor

            executor = Executor.create("rebuild.alias")
            assert isinstance(executor, RebuildExecutor)
            assert executor.backend == "rebuild.alias"
        finally:
            Executor._registry = original_registry
            Executor._backend_alias_map = original_alias_map
            Executor._alias_registry_size = original_alias_registry_size
            Executor._plugins_discovered = original_discovered
