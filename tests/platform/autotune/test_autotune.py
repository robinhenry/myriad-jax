"""Tests for autotune logic, cache management, and configuration validation."""

import jax

from myriad.platform.autotune.cache import load_cache, save_cache
from myriad.platform.autotune.testing import validate_config
from myriad.platform.autotune.utils import (
    ceil_to_valid_scale,
    floor_to_valid_scale,
    get_hardware_id,
    make_config_key,
)


class TestHardware:
    """Tests for hardware identification."""

    def test_hardware_id_deterministic(self):
        """Hardware ID should be consistent and end with platform."""
        hw_id1 = get_hardware_id()
        hw_id2 = get_hardware_id()
        platform = jax.devices()[0].platform

        assert hw_id1 == hw_id2
        assert hw_id1.endswith(f"_{platform}")


class TestConfigKey:
    """Tests for configuration key generation."""

    def test_config_key_logic(self):
        """Config keys should be deterministic and different for different inputs."""
        key1 = make_config_key("env1", "agent1", 1000, "hw1")
        key2 = make_config_key("env1", "agent1", 1000, "hw1")
        key3 = make_config_key("env2", "agent1", 1000, "hw1")

        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 16


class TestScaling:
    """Tests for scaling functions."""

    def test_scaling_logic(self):
        """Should correctly ceil and floor to valid scales."""
        assert ceil_to_valid_scale(150) == 1_000
        assert floor_to_valid_scale(150) == 100
        assert ceil_to_valid_scale(10_000) == 10_000
        assert floor_to_valid_scale(10_000) == 10_000


class TestCache:
    """Tests for cache persistence."""

    def test_cache_roundtrip(self, monkeypatch, tmp_path):
        """Save then load should preserve data."""
        cache_path = tmp_path / "autotune_profiles.json"
        monkeypatch.setattr("myriad.platform.autotune.cache.get_cache_path", lambda: cache_path)

        data = {
            "hardware": {"hw1": {"memory_gb": 16.0}},
            "env_profiles": {"env1": {"memory_mb": 1.5}},
            "agent_profiles": {},
            "validated_configs": {},
            "chunk_size_configs": {},
        }

        save_cache(data)
        assert load_cache() == data

    def test_load_empty_cache(self, monkeypatch, tmp_path):
        """Loading a non-existent cache should return the default structure."""
        cache_path = tmp_path / "missing.json"
        monkeypatch.setattr("myriad.platform.autotune.cache.get_cache_path", lambda: cache_path)

        cache = load_cache()
        assert isinstance(cache, dict)
        assert "hardware" in cache
        assert "env_profiles" in cache

    def test_corrupted_cache(self, monkeypatch, tmp_path):
        """Should handle corrupted cache gracefully."""
        cache_path = tmp_path / "autotune_profiles.json"
        monkeypatch.setattr("myriad.platform.autotune.cache.get_cache_path", lambda: cache_path)

        cache_path.write_text("{ invalid }")
        cache = load_cache()
        assert isinstance(cache, dict)
        assert "hardware" in cache


class TestValidation:
    """Tests for configuration validation (running JAX)."""

    def test_config_validation_basic(self):
        """Verify that validate_config runs and returns valid metrics."""
        success, throughput, memory = validate_config(
            env_name="cartpole-control",
            num_envs=10,
            timeout_s=30.0,
        )

        assert success is True
        assert throughput is not None and throughput > 0
        assert memory is not None and memory > 0

    def test_config_validation_scaling(self):
        """Verify that more environments use more memory."""
        res1 = validate_config(env_name="cartpole-control", num_envs=1, timeout_s=30.0)
        res2 = validate_config(env_name="cartpole-control", num_envs=10, timeout_s=30.0)

        if res1[0] and res2[0]:
            assert res2[2] > res1[2]
