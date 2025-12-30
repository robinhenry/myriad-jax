"""Hardware, environment, and agent profiling for auto-tuning."""

from datetime import datetime

import jax

from .utils import get_hardware_id


def profile_hardware(cache: dict) -> dict:
    """Profile hardware and add to cache.

    Args:
        cache: Autotune cache dict

    Returns:
        Updated cache with hardware profile
    """
    hardware_id = get_hardware_id()
    device = jax.devices()[0]
    platform = device.platform

    # Try to get memory info
    total_memory_gb = None
    if platform == "gpu":
        stats = device.memory_stats()
        total_memory_gb = stats.get("bytes_limit", 0) / (1024**3)
    else:
        # CPU - use conservative estimate
        total_memory_gb = 8.0

    cache["hardware"][hardware_id] = {
        "platform": platform,
        "device": str(device),
        "total_memory_gb": total_memory_gb,
        "available_memory_gb": total_memory_gb * 0.9,  # 90% available
        "profiled_at": datetime.now().isoformat(),
    }

    return cache


def profile_env(env_name: str, cache: dict) -> dict:
    """Profile an environment and add to cache.

    Args:
        env_name: Environment name
        cache: Autotune cache dict

    Returns:
        Updated cache with environment profile
    """
    from .testing import validate_config

    # Test with small num_envs to measure per-env memory
    test_num_envs = 1000
    success, _, memory_gb = validate_config(env_name, "random", test_num_envs, 64)

    if not success or memory_gb is None:
        raise RuntimeError(f"Failed to profile environment {env_name}")

    # Estimate per-env memory (rough approximation)
    memory_mb = (memory_gb * 1024) / test_num_envs

    cache["env_profiles"][env_name] = {
        "memory_mb_per_env": memory_mb,
        "profiled_at": datetime.now().isoformat(),
        "profiled_with_num_envs": test_num_envs,
    }

    return cache


def profile_agent(agent_name: str, cache: dict) -> dict:
    """Profile an agent and add to cache.

    For now, uses conservative defaults based on agent type.
    Future: Could actually instantiate and measure agent.

    Args:
        agent_name: Agent name
        cache: Autotune cache dict

    Returns:
        Updated cache with agent profile
    """
    # Conservative defaults by agent type
    agent_defaults = {
        "random": {"overhead_mb": 0.1},
        "pid": {"overhead_mb": 0.1},
        "bangbang": {"overhead_mb": 0.1},
        "dqn": {"overhead_mb": 2.0},
        "pqn": {"overhead_mb": 2.0},
        "ppo": {"overhead_mb": 5.0},
        "sac": {"overhead_mb": 5.0},
        "td3": {"overhead_mb": 5.0},
    }

    overhead_mb = agent_defaults.get(agent_name, {"overhead_mb": 10.0})["overhead_mb"]

    cache["agent_profiles"][agent_name] = {
        "overhead_mb": overhead_mb,
        "profiled_at": datetime.now().isoformat(),
        "method": "default",  # Could be "measured" in future
    }

    return cache
