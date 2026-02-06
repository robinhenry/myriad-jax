"""Tests for autotune utility functions."""

import hashlib

import jax

from myriad.platform.autotune.utils import (
    VALID_CHUNK_SIZES,
    VALID_NUM_ENVS_SCALE,
    ceil_to_valid_scale,
    floor_to_valid_scale,
    get_hardware_id,
    make_config_key,
    round_to_valid_scale,
)


class TestConstants:
    """Tests for module constants."""

    def test_valid_chunk_sizes(self):
        """VALID_CHUNK_SIZES should be powers of 2."""
        for size in VALID_CHUNK_SIZES:
            # Check if power of 2
            assert size > 0
            assert (size & (size - 1)) == 0

    def test_chunk_sizes_sorted(self):
        """VALID_CHUNK_SIZES should be sorted."""
        assert VALID_CHUNK_SIZES == sorted(VALID_CHUNK_SIZES)

    def test_valid_num_envs_scale(self):
        """VALID_NUM_ENVS_SCALE should have reasonable values."""
        assert len(VALID_NUM_ENVS_SCALE) > 0
        assert all(x > 0 for x in VALID_NUM_ENVS_SCALE)

    def test_num_envs_scale_sorted(self):
        """VALID_NUM_ENVS_SCALE should be sorted."""
        assert VALID_NUM_ENVS_SCALE == sorted(VALID_NUM_ENVS_SCALE)


class TestHardwareId:
    """Tests for get_hardware_id."""

    def test_hardware_id_format(self):
        """Hardware ID should have expected format."""
        hw_id = get_hardware_id()

        assert isinstance(hw_id, str)
        assert len(hw_id) > 0
        assert "_" in hw_id

        # Should end with platform type
        assert hw_id.endswith("_cpu") or hw_id.endswith("_gpu") or hw_id.endswith("_tpu")

    def test_hardware_id_deterministic(self):
        """Hardware ID should be consistent across calls."""
        hw_id1 = get_hardware_id()
        hw_id2 = get_hardware_id()

        assert hw_id1 == hw_id2

    def test_hardware_id_matches_platform(self):
        """Hardware ID platform suffix should match JAX platform."""
        hw_id = get_hardware_id()
        platform = jax.devices()[0].platform

        assert hw_id.endswith(f"_{platform}")


class TestConfigKey:
    """Tests for make_config_key."""

    def test_config_key_format(self):
        """Config key should be an MD5 hash prefix."""
        key = make_config_key("cartpole", "dqn", 10000, "GPU_gpu")

        assert isinstance(key, str)
        assert len(key) == 16  # First 16 chars of MD5
        # Should be valid hex
        assert all(c in "0123456789abcdef" for c in key)

    def test_config_key_deterministic(self):
        """Same inputs should produce same key."""
        key1 = make_config_key("cartpole", "dqn", 10000, "GPU_gpu")
        key2 = make_config_key("cartpole", "dqn", 10000, "GPU_gpu")

        assert key1 == key2

    def test_config_key_different_inputs(self):
        """Different inputs should produce different keys."""
        key1 = make_config_key("cartpole", "dqn", 10000, "GPU_gpu")
        key2 = make_config_key("pendulum", "dqn", 10000, "GPU_gpu")
        key3 = make_config_key("cartpole", "ppo", 10000, "GPU_gpu")
        key4 = make_config_key("cartpole", "dqn", 20000, "GPU_gpu")
        key5 = make_config_key("cartpole", "dqn", 10000, "CPU_cpu")

        # All should be different
        keys = [key1, key2, key3, key4, key5]
        assert len(keys) == len(set(keys))

    def test_config_key_none_buffer_size(self):
        """Config key should handle None buffer_size."""
        key1 = make_config_key("cartpole", "dqn", None, "GPU_gpu")
        key2 = make_config_key("cartpole", "dqn", None, "GPU_gpu")

        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 16

    def test_config_key_none_vs_buffer(self):
        """None buffer_size should differ from any numeric value."""
        key_none = make_config_key("cartpole", "dqn", None, "GPU_gpu")
        key_buffer = make_config_key("cartpole", "dqn", 10000, "GPU_gpu")

        assert key_none != key_buffer

    def test_config_key_matches_hash(self):
        """Config key should be MD5 hash prefix of config string."""
        env = "cartpole"
        agent = "dqn"
        buffer_size = 10000
        hardware_id = "GPU_gpu"

        key = make_config_key(env, agent, buffer_size, hardware_id)

        # Compute expected hash
        config_str = f"{env}:{agent}:buffer{buffer_size}:{hardware_id}"
        expected = hashlib.md5(config_str.encode()).hexdigest()[:16]

        assert key == expected


class TestScalingFunctions:
    """Tests for ceil_to_valid_scale and floor_to_valid_scale."""

    def test_ceil_exact_match(self):
        """Values exactly matching scale should return same value (ceil)."""
        for scale in VALID_NUM_ENVS_SCALE:
            assert ceil_to_valid_scale(scale) == scale

    def test_ceil_round_up(self):
        """Values should round up to next scale."""
        assert ceil_to_valid_scale(150) == 1_000
        assert ceil_to_valid_scale(1_500) == 10_000

    def test_ceil_maximum_value(self):
        """Values larger than max should return max scale (ceil)."""
        max_scale = VALID_NUM_ENVS_SCALE[-1]
        assert ceil_to_valid_scale(max_scale + 1_000_000) == max_scale

    def test_floor_exact_match(self):
        """Values exactly matching scale should return same value (floor)."""
        for scale in VALID_NUM_ENVS_SCALE:
            assert floor_to_valid_scale(scale) == scale

    def test_floor_round_down(self):
        """Values should round down to previous scale."""
        assert floor_to_valid_scale(150) == 100
        assert floor_to_valid_scale(1_500) == 1_000

    def test_floor_minimum_value(self):
        """Values smaller than min should return min scale (floor)."""
        min_scale = VALID_NUM_ENVS_SCALE[0]
        assert floor_to_valid_scale(1) == min_scale
        assert floor_to_valid_scale(50) == min_scale

    def test_backward_compatibility(self):
        """round_to_valid_scale should be an alias for ceil_to_valid_scale."""
        assert round_to_valid_scale is ceil_to_valid_scale
