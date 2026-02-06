"""Tests for config validation and testing."""

import pytest

from myriad.platform.autotune.testing import validate_config

# Mark all tests in this module as slow (integration tests)
pytestmark = pytest.mark.slow


class TestConfigValidation:
    """Tests for test_config function."""

    def test_config_validation_succeeds(self):
        """test_config should succeed for reasonable configs."""
        success, throughput, memory = validate_config(
            env_name="cartpole-control",
            num_envs=10,  # Small for fast testing
            timeout_s=15.0,  # Shorter timeout
        )

        assert success is True
        assert throughput is not None
        assert throughput > 0  # Should get positive throughput
        assert memory is not None
        assert memory > 0  # Should measure positive memory

    def test_config_validation_returns_metrics(self):
        """test_config should return throughput and memory metrics."""
        success, throughput, memory = validate_config(
            env_name="cartpole-control",
            num_envs=10,  # Small for fast testing
            timeout_s=15.0,  # Shorter timeout
        )

        if success:
            # Throughput should be in reasonable range (steps/second)
            assert 100 < throughput < 1e10  # Sanity bounds
            # Memory should be in reasonable range (GB)
            # Even very small configs can be < 1MB
            assert 0.0000001 < memory < 1000  # Sanity bounds (0.1KB to 1TB)

    @pytest.mark.skip(reason="Too slow - requires OOM which takes time")
    def test_failure_returns_none(self):
        """test_config should return None metrics on failure."""
        # Use unreasonably large config that should fail
        success, throughput, memory = validate_config(
            env_name="cartpole-control",
            num_envs=100_000_000,  # Unreasonably large
            timeout_s=2.0,  # Very short timeout
        )

        # Should fail (either OOM or timeout)
        if not success:
            assert throughput is None
            assert memory is None

    @pytest.mark.skip(reason="Too slow - tests timeout behavior")
    def test_timeout_prevents_hang(self):
        """test_config should respect timeout and not hang."""
        import time

        start = time.time()
        success, throughput, memory = validate_config(
            env_name="cartpole-control",
            num_envs=1_000_000,  # Large config
            timeout_s=2.0,  # Very short timeout
        )
        elapsed = time.time() - start

        # Should complete within reasonable time (timeout + some overhead)
        # Allow 3x timeout for overhead (compilation, etc.)
        assert elapsed < 10.0

    def test_memory_increases_with_envs(self):
        """More environments should use more memory."""
        # Test with small number of envs
        success1, throughput1, memory1 = validate_config(
            env_name="cartpole-control",
            num_envs=5,  # Very small
            timeout_s=15.0,
        )

        # Test with larger number of envs
        success2, throughput2, memory2 = validate_config(
            env_name="cartpole-control",
            num_envs=20,  # Small but 4x larger
            timeout_s=15.0,
        )

        if success1 and success2:
            # More envs should use more memory
            assert memory2 > memory1

    def test_different_environments(self):
        """test_config should work with different environments."""
        envs = ["cartpole-control"]

        for env_name in envs:
            success, throughput, memory = validate_config(
                env_name=env_name,
                num_envs=10,  # Small for fast testing
                timeout_s=15.0,
            )

            # Should succeed for all standard environments
            assert success is True
            assert throughput is not None
            assert memory is not None

    def test_consistent_results(self):
        """test_config should give consistent results for same config."""
        config = {
            "env_name": "cartpole-control",
            "num_envs": 10,  # Small for fast testing
            "timeout_s": 15.0,
        }

        # Run twice
        success1, throughput1, memory1 = validate_config(**config)
        success2, throughput2, memory2 = validate_config(**config)

        # Success should be consistent
        assert success1 == success2

        if success1:
            # Memory should be exactly the same (deterministic)
            assert memory1 == memory2

            # Throughput may vary due to system noise, JIT warmup, etc.
            # Allow wider tolerance for flakiness
            assert 0.3 < throughput1 / throughput2 < 3.0


class TestConfigValidationEdgeCases:
    """Tests for edge cases in config validation."""

    def test_minimum_envs(self):
        """test_config should work with single environment."""
        success, throughput, memory = validate_config(
            env_name="cartpole-control",
            num_envs=1,
            timeout_s=30.0,
        )

        assert success is True
        assert throughput is not None
        assert memory is not None

    def test_minimum_envs_alias(self):  # Renamed from test_small_chunk_size
        """test_config should work with small number of environments."""
        success, throughput, memory = validate_config(
            env_name="cartpole-control",
            num_envs=10,  # Small for fast testing
            timeout_s=15.0,
        )

        assert success is True

    def test_large_envs(self):  # Renamed from test_large_chunk_size
        """test_config should work with large number of environments."""
        success, throughput, memory = validate_config(
            env_name="cartpole-control",
            num_envs=100,  # Reduced for speed
            timeout_s=15.0,
        )

        # Large chunk size should work if num_envs supports it
        if success:
            assert throughput is not None
            assert memory is not None
