"""Utility functions for auto-tuning."""

import hashlib

import jax

# Valid values for search (powers of 2 and common scales)
VALID_CHUNK_SIZES = [8, 16, 32, 64, 128, 256, 512]
VALID_NUM_ENVS_SCALE = [100, 1_000, 10_000, 100_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]


def get_hardware_id() -> str:
    """Get unique identifier for current hardware.

    Returns:
        Hardware ID string (e.g., "RTX4090_gpu" or "CPU_cpu")
    """
    device = jax.devices()[0]
    platform = device.platform

    if platform == "gpu":
        # Try to get GPU name
        device_str = str(device.device_kind)
        # Create simple ID from device kind
        gpu_name = device_str.split()[-1] if device_str else "GPU"
        return f"{gpu_name}_{platform}"
    else:
        # CPU
        return f"CPU_{platform}"


def make_config_key(env: str, agent: str, buffer_size: int | None, hardware_id: str) -> str:
    """Create cache key for a specific configuration.

    Args:
        env: Environment name
        agent: Agent name
        buffer_size: Replay buffer size (if applicable)
        hardware_id: Hardware identifier

    Returns:
        MD5 hash of the configuration (16 characters)
    """
    buffer_str = str(buffer_size) if buffer_size else "none"
    key = f"{env}:{agent}:buffer{buffer_str}:{hardware_id}"
    # Hash to keep keys short
    return hashlib.md5(key.encode()).hexdigest()[:16]


def ceil_to_valid_scale(num_envs: int) -> int:
    """Round num_envs up to nearest valid scale.

    Args:
        num_envs: Number of environments

    Returns:
        Smallest value from VALID_NUM_ENVS_SCALE >= num_envs
    """
    for scale in VALID_NUM_ENVS_SCALE:
        if num_envs <= scale:
            return scale
    return VALID_NUM_ENVS_SCALE[-1]


def floor_to_valid_scale(num_envs: int) -> int:
    """Round num_envs down to nearest valid scale.

    Args:
        num_envs: Number of environments

    Returns:
        Largest value from VALID_NUM_ENVS_SCALE <= num_envs
    """
    best_scale = VALID_NUM_ENVS_SCALE[0]
    for scale in VALID_NUM_ENVS_SCALE:
        if scale <= num_envs:
            best_scale = scale
        else:
            break
    return best_scale


# Alias for backward compatibility
round_to_valid_scale = ceil_to_valid_scale
