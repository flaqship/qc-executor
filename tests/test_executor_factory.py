"""Tests for the Executor factory class."""

import importlib.util

import numpy as np
import pytest

from qc_executor.base.executor_base import ExecutorBase
from qc_executor.factory import Executor
from qc_executor.qiskit import QiskitExecutor


class MockExecutor(ExecutorBase):
    """Mock executor for testing."""

    def __init__(self, shots=None, **kwargs):
        """Initialize mock executor."""
        super().__init__(shots=shots, **kwargs)

    def _expectation_value(self, circuit, observable, parameters=None):
        """Mock implementation."""
        return 0.5

    def _expectation_value_derivatives(self, circuit, observable, parameters=None):
        """Mock implementation."""
        return np.array([0.1, 0.2])

    def _sample(self, circuit, parameters=None):
        """Mock implementation."""
        return {"00": 500, "11": 500}

    def _statevector(self, circuit, parameters=None):
        """Mock implementation."""
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
        class RegisteredExecutor(ExecutorBase):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def _expectation_value(self, circuit, observable, parameters=None):
                return 0.0

            def _expectation_value_derivatives(self, circuit, observable, parameters=None):
                return np.array([0.0])

            def _sample(self, circuit, parameters=None):
                return {}

            def _statevector(self, circuit, parameters=None):
                return np.array([1.0])

            def _transpile_circuit(self, circuit):
                return circuit

        # Check if backend is registered and maps to the decorated class
        assert "mock_test" in Executor.available_backends()
        assert Executor._registry["mock_test"] is RegisteredExecutor

        # Clean up
        if "mock_test" in Executor._registry:
            del Executor._registry["mock_test"]

    def test_register_validates_base_class(self):
        """Test that register checks for ExecutorBase inheritance."""

        class NotAnExecutor:
            pass

        with pytest.raises(TypeError, match="must inherit from ExecutorBase"):
            Executor.register("invalid")(NotAnExecutor)

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

            assert "pip install qc-executor[qiskit-full]" in str(exc_info.value)
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

    def test_create_triggers_plugin_discovery_when_needed(self, monkeypatch):
        original_discovered = Executor._plugins_discovered

        called = {"discover": False}

        def fake_discover(cls):
            called["discover"] = True
            cls._plugins_discovered = True

        monkeypatch.setattr(Executor, "_discover_plugins", classmethod(fake_discover))

        try:
            Executor._plugins_discovered = False

            with pytest.raises(ValueError, match="Backend 'still_missing' not found"):
                Executor.create("still_missing")

            assert called["discover"] is True
        finally:
            Executor._plugins_discovered = original_discovered

    def test_create_unknown_alias_uses_resolved_backend_extra_hint(self, monkeypatch):
        original_registry = Executor._registry.copy()
        original_alias_map = Executor._backend_alias_map.copy()
        original_alias_registry_size = Executor._alias_registry_size
        original_discovered = Executor._plugins_discovered

        try:
            Executor._registry = {}
            Executor._backend_alias_map = {"statevector": "qiskit"}
            Executor._alias_registry_size = 0
            Executor._plugins_discovered = True

            monkeypatch.setattr(
                Executor, "_rebuild_backend_alias_map", classmethod(lambda cls: None)
            )

            with pytest.raises(ValueError) as exc_info:
                Executor.create("statevector")

            assert "pip install qc-executor[qiskit-full]" in str(exc_info.value)
        finally:
            Executor._registry = original_registry
            Executor._backend_alias_map = original_alias_map
            Executor._alias_registry_size = original_alias_registry_size
            Executor._plugins_discovered = original_discovered

    def test_create_non_string_skips_backend_with_failing_type_query(self):
        original_registry = Executor._registry.copy()
        original_alias_map = Executor._backend_alias_map.copy()
        original_alias_registry_size = Executor._alias_registry_size
        original_discovered = Executor._plugins_discovered

        class Marker:
            pass

        class FailingTypesExecutor(MockExecutor):
            @classmethod
            def get_accepted_backend_types(cls) -> list[type]:
                raise ImportError("boom")

        class MatchingTypesExecutor(MockExecutor):
            def __init__(self, backend=None, **kwargs):
                super().__init__(**kwargs)
                self.backend = backend

            @classmethod
            def get_accepted_backend_types(cls) -> list[type]:
                return [Marker]

        try:
            Executor._registry = {
                "failing": FailingTypesExecutor,
                "matching": MatchingTypesExecutor,
            }
            Executor._backend_alias_map = {}
            Executor._alias_registry_size = len(Executor._registry)
            Executor._plugins_discovered = True

            marker = Marker()
            executor = Executor.create(marker)

            assert isinstance(executor, MatchingTypesExecutor)
            assert executor.backend is marker
        finally:
            Executor._registry = original_registry
            Executor._backend_alias_map = original_alias_map
            Executor._alias_registry_size = original_alias_registry_size
            Executor._plugins_discovered = original_discovered

    def test_create_non_string_reports_accepted_types_summary(self):
        original_registry = Executor._registry.copy()
        original_alias_map = Executor._backend_alias_map.copy()
        original_alias_registry_size = Executor._alias_registry_size
        original_discovered = Executor._plugins_discovered

        class IntOnlyExecutor(MockExecutor):
            @classmethod
            def get_accepted_backend_types(cls) -> list[type]:
                return [int]

        class EmptyTypesExecutor(MockExecutor):
            @classmethod
            def get_accepted_backend_types(cls) -> list[type]:
                return []

        try:
            Executor._registry = {
                "int_only": IntOnlyExecutor,
                "empty": EmptyTypesExecutor,
            }
            Executor._backend_alias_map = {}
            Executor._alias_registry_size = len(Executor._registry)
            Executor._plugins_discovered = True

            with pytest.raises(ValueError) as exc_info:
                Executor.create(object())

            error_msg = str(exc_info.value)
            assert "No registered executor accepts an object of type" in error_msg
            assert "Accepted types per backend: {'int_only': ['int']}" in error_msg
        finally:
            Executor._registry = original_registry
            Executor._backend_alias_map = original_alias_map
            Executor._alias_registry_size = original_alias_registry_size
            Executor._plugins_discovered = original_discovered

    def test_discover_plugins_fallback_uses_select(self, monkeypatch):
        select_called = {"value": False}

        class FakeEntryPoint:
            name = "ok"

            def load(self):
                return None

        class FakeSelection:
            def select(self, group):
                select_called["value"] = True
                assert group == "qc_executor.backends"
                return [FakeEntryPoint()]

        def fake_entry_points(*_args, **kwargs):
            if "group" in kwargs:
                raise TypeError("old API")
            return FakeSelection()

        monkeypatch.setattr("qc_executor.factory.entry_points", fake_entry_points)

        Executor._plugins_discovered = False
        Executor._discover_plugins()
        assert Executor._plugins_discovered is True
        assert select_called["value"] is True

    def test_discover_plugins_fallback_uses_dict_get(self, monkeypatch):
        get_called = {"value": False}

        class FakeEntryPoint:
            name = "ok-dict"

            def load(self):
                return None

        class FakeEntryPointsDict(dict):
            def get(self, key, default=None):
                get_called["value"] = True
                return super().get(key, default)

        def fake_entry_points(*_args, **kwargs):
            if "group" in kwargs:
                raise TypeError("old API")
            return FakeEntryPointsDict({"qc_executor.backends": [FakeEntryPoint()]})

        monkeypatch.setattr("qc_executor.factory.entry_points", fake_entry_points)

        Executor._plugins_discovered = False
        Executor._discover_plugins()
        assert Executor._plugins_discovered is True
        assert get_called["value"] is True


class TestExecutorIntegration:
    """Integration tests with actual backends."""

    def test_qiskit_in_available_backends(self):
        """Test that qiskit backend is available."""
        backends = Executor.available_backends()
        assert "qiskit" in backends

    def test_create_qiskit(self):
        """Test creating QiskitExecutor via factory."""
        executor = Executor.create("qiskit")

        assert isinstance(executor, QiskitExecutor)
        assert executor.shots is None

    def test_create_qiskit_via_statevector_alias(self):
        """Test creating QiskitExecutor using the statevector alias."""
        executor = Executor.create("statevector")
        assert isinstance(executor, QiskitExecutor)

    def test_create_qiskit_via_aer_alias(self):
        """Test creating QiskitExecutor using the aer alias."""
        if importlib.util.find_spec("qiskit_aer") is None:
            pytest.skip("qiskit-aer not installed")

        executor = Executor.create("aer")
        assert isinstance(executor, QiskitExecutor)

    def test_autodetect_pennylane_device_instance(self):
        """Test auto-detection when passing a PennyLane device instance."""
        # PennyLane is an optional backend, imported lazily so the module can be
        # collected without it installed.
        try:
            import pennylane as qml  # pylint: disable=import-outside-toplevel
            from qc_executor.pennylane import (  # pylint: disable=import-outside-toplevel
                PennyLaneExecutor,
            )
        except ImportError:
            pytest.skip("PennyLane not installed")

        dev = qml.device("default.qubit", wires=1)
        executor = Executor.create(dev)

        assert isinstance(executor, PennyLaneExecutor)
        assert executor._device is dev

    def test_create_pennylane_via_device_string_alias(self):
        """Test creating PennyLaneExecutor using a device string alias."""
        if importlib.util.find_spec("pennylane") is None:
            pytest.skip("PennyLane not installed")

        # pylint: disable-next=import-outside-toplevel
        from qc_executor.pennylane import PennyLaneExecutor

        executor = Executor.create("default.qubit", wires=1)
        assert isinstance(executor, PennyLaneExecutor)

    def test_pennylane_backend_available(self):
        """Test that pennylane backend is available if installed."""
        backends = Executor.available_backends()

        if importlib.util.find_spec("pennylane") is not None:
            assert "pennylane" in backends
        else:
            assert "pennylane" not in backends

    def test_qulacs_backend_available(self):
        """Test that qulacs backend is available if installed."""
        backends = Executor.available_backends()

        if importlib.util.find_spec("qulacs") is not None:
            assert "qulacs" in backends
        else:
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
        if importlib.util.find_spec("pennylane") is None:
            # PennyLane not installed - just verify method exists
            assert hasattr(executor, "switch_backend")
            return

        # pylint: disable-next=import-outside-toplevel
        from qc_executor.pennylane import PennyLaneExecutor

        new_executor = executor.switch_backend("pennylane")

        assert new_executor.shots == 1024
        assert new_executor._seed == 42
        assert isinstance(new_executor, PennyLaneExecutor)

    def test_switch_backend_with_overrides(self):
        """Test that switch_backend can override configuration."""
        executor = Executor.create("qiskit", shots=1024, seed=42)

        # Switch with overrides
        if importlib.util.find_spec("pennylane") is None:
            pytest.skip("PennyLane not installed")

        new_executor = executor.switch_backend("pennylane", shots=2048, seed=99)

        assert new_executor.shots == 2048
        assert new_executor._seed == 99


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

            assert Executor._registry["dup_a"] is DupA

            class DupB(MockExecutor):
                @classmethod
                def get_accepted_backend_aliases(cls) -> list[str]:
                    return ["dup.alias"]

            with pytest.raises(ValueError, match="Duplicate backend alias 'dup.alias'"):
                Executor.register("dup_b")(DupB)

        finally:
            Executor._registry = original_registry
            Executor._backend_alias_map = original_alias_map
            Executor._alias_registry_size = original_alias_registry_size
            Executor._plugins_discovered = original_discovered

    def test_create_alias_rebuilds_when_alias_points_to_missing_backend(self):
        original_registry = Executor._registry.copy()
        original_alias_map = Executor._backend_alias_map.copy()
        original_alias_registry_size = Executor._alias_registry_size
        original_discovered = Executor._plugins_discovered

        class ReindexedAliasExecutor(MockExecutor):
            def __init__(self, backend=None, **kwargs):
                super().__init__(**kwargs)
                self.backend = backend

            @classmethod
            def get_accepted_backend_aliases(cls) -> list[str]:
                return ["stale.alias"]

        try:
            Executor._registry = {"real_backend": ReindexedAliasExecutor}
            Executor._backend_alias_map = {"stale.alias": "missing_backend"}
            Executor._alias_registry_size = 1
            Executor._plugins_discovered = True

            executor = Executor.create("stale.alias")

            assert isinstance(executor, ReindexedAliasExecutor)
            assert executor.backend == "stale.alias"
        finally:
            Executor._registry = original_registry
            Executor._backend_alias_map = original_alias_map
            Executor._alias_registry_size = original_alias_registry_size
            Executor._plugins_discovered = original_discovered

    def test_create_alias_with_explicit_backend_kwarg_raises(self):
        original_registry = Executor._registry.copy()
        original_alias_map = Executor._backend_alias_map.copy()
        original_alias_registry_size = Executor._alias_registry_size
        original_discovered = Executor._plugins_discovered

        class AliasConflictExecutor(MockExecutor):
            @classmethod
            def get_accepted_backend_aliases(cls) -> list[str]:
                return ["conflict.alias"]

        try:
            Executor._registry = {}
            Executor._backend_alias_map = {}
            Executor._alias_registry_size = 0
            Executor._plugins_discovered = True

            Executor.register("alias_conflict")(AliasConflictExecutor)

            with pytest.raises(ValueError, match="Conflicting 'backend' argument"):
                Executor.create("conflict.alias", backend="duplicate")
        finally:
            Executor._registry = original_registry
            Executor._backend_alias_map = original_alias_map
            Executor._alias_registry_size = original_alias_registry_size
            Executor._plugins_discovered = original_discovered

    def test_index_backend_aliases_ignores_empty_aliases(self):
        original_alias_map = Executor._backend_alias_map.copy()

        class EmptyAliasExecutor(MockExecutor):
            @classmethod
            def get_accepted_backend_aliases(cls) -> list[str]:
                return ["", "   ", "valid.alias"]

        try:
            Executor._backend_alias_map = {}
            Executor._index_backend_aliases("empty_alias", EmptyAliasExecutor)

            assert Executor._backend_alias_map == {"valid.alias": "empty_alias"}
        finally:
            Executor._backend_alias_map = original_alias_map

    def test_rebuild_backend_alias_map_ignores_empty_aliases(self):
        original_registry = Executor._registry.copy()
        original_alias_map = Executor._backend_alias_map.copy()
        original_alias_registry_size = Executor._alias_registry_size

        class RebuildEmptyAliasExecutor(MockExecutor):
            @classmethod
            def get_accepted_backend_aliases(cls) -> list[str]:
                return ["", "  ", "rebuilt.alias"]

        try:
            Executor._registry = {"rebuilt": RebuildEmptyAliasExecutor}
            Executor._backend_alias_map = {}
            Executor._alias_registry_size = 0

            Executor._rebuild_backend_alias_map()
            assert Executor._backend_alias_map == {"rebuilt.alias": "rebuilt"}
        finally:
            Executor._registry = original_registry
            Executor._backend_alias_map = original_alias_map
            Executor._alias_registry_size = original_alias_registry_size

    def test_rebuild_backend_alias_map_duplicate_alias_raises(self):
        original_registry = Executor._registry.copy()
        original_alias_map = Executor._backend_alias_map.copy()
        original_alias_registry_size = Executor._alias_registry_size

        class RebuildDupA(MockExecutor):
            @classmethod
            def get_accepted_backend_aliases(cls) -> list[str]:
                return ["dup.rebuild"]

        class RebuildDupB(MockExecutor):
            @classmethod
            def get_accepted_backend_aliases(cls) -> list[str]:
                return ["dup.rebuild"]

        try:
            Executor._registry = {
                "rebuild_a": RebuildDupA,
                "rebuild_b": RebuildDupB,
            }
            Executor._backend_alias_map = {}
            Executor._alias_registry_size = 0

            with pytest.raises(ValueError, match="Duplicate backend alias 'dup.rebuild'"):
                Executor._rebuild_backend_alias_map()
        finally:
            Executor._registry = original_registry
            Executor._backend_alias_map = original_alias_map
            Executor._alias_registry_size = original_alias_registry_size

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
