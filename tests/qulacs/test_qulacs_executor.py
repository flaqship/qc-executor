import logging

import pytest

from executor.qulacs import QulacsExecutor


class TestQulacsExecutor:
    def test_placeholder(self):
        pass

    # ========================================================================
    # Logging Tests
    # ========================================================================

    def test_logging_default_level(self):
        """Test that default logging level is WARNING."""
        executor = QulacsExecutor()
        assert executor._logger.level == logging.WARNING

    def test_logging_info_level(self):
        """Test that INFO logging level is set correctly."""
        executor = QulacsExecutor(log_level="INFO")
        assert executor._logger.level == logging.INFO

    def test_logging_debug_level(self):
        """Test that DEBUG logging level is set correctly."""
        executor = QulacsExecutor(log_level="DEBUG")
        assert executor._logger.level == logging.DEBUG

    def test_logging_invalid_level_raises(self):
        """Test that an invalid log_level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log_level"):
            QulacsExecutor(log_level="TRACE")

    def test_logging_to_file(self, tmp_path):
        """Test that log messages are written to the specified log file."""
        log_file = str(tmp_path / "qulacs_executor.log")
        executor = QulacsExecutor(log_level="INFO", log_file=log_file)
        executor._logger.info("qulacs test log message")

        with open(log_file) as f:
            content = f.read()
        assert "qulacs test log message" in content

        for handler in executor._logger.handlers[:]:
            handler.close()
            executor._logger.removeHandler(handler)

    # ========================================================================
    # Cache Size Tests
    # ========================================================================

    def test_cache_size_restriction_circuits(self):
        """Test that circuit cache respects max_cache_size."""
        executor = QulacsExecutor(max_cache_size=1)
        assert executor._circuit_cache.max_size == 1

    def test_cache_size_restriction_operators(self):
        """Test that operator cache respects max_cache_size."""
        executor = QulacsExecutor(max_cache_size=1)
        assert executor._operator_cache.max_size == 1

    def test_unlimited_cache_size_by_default(self):
        """Test that caches are unlimited when max_cache_size is not specified."""
        executor = QulacsExecutor()
        assert executor._max_cache_size is None
        assert executor._circuit_cache.max_size is None
        assert executor._operator_cache.max_size is None
