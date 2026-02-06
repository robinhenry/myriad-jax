"""Hardware, environment, and agent profiling for auto-tuning."""

import logging
from datetime import datetime

import jax

from myriad.agents import get_agent_info

from .testing import validate_config
from .utils import get_hardware_id

logger = logging.getLogger(__name__)

# Constants for memory estimation
DEFAULT_CPU_MEMORY_GB = 4.0
AVAILABLE_MEMORY_FRACTION = 0.9

# Agent overhead heuristics
OVERHEAD_HEURISTICS = {
    "off_policy": 5.0,
    "on_policy": 2.0,
    "classical": 0.5,
    "unknown": 10.0,
}


def _get_cpu_memory_gb() -> float:
    """Attempt to get total CPU memory in GB."""
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except (ImportError, Exception):
        # Fallback to conservative default
        return DEFAULT_CPU_MEMORY_GB


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
        # CPU - use psutil or conservative estimate
        total_memory_gb = _get_cpu_memory_gb()

    cache["hardware"][hardware_id] = {
        "platform": platform,
        "device": str(device),
        "total_memory_gb": total_memory_gb,
        "available_memory_gb": total_memory_gb * AVAILABLE_MEMORY_FRACTION,
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

    # Test with small num_envs to measure per-env memory
    test_num_envs = 1000
    success, _, memory_gb = validate_config(env_name, test_num_envs)

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

    Args:
        agent_name: Agent name
        cache: Autotune cache dict

    Returns:
        Updated cache with agent profile
    """
    info = get_agent_info(agent_name)

    if info:
        # Heuristic based on agent properties
        if info.is_off_policy:
            overhead_mb = OVERHEAD_HEURISTICS["off_policy"]
        elif info.is_on_policy:
            overhead_mb = OVERHEAD_HEURISTICS["on_policy"]
        else:
            overhead_mb = OVERHEAD_HEURISTICS["classical"]
    else:
        # Fallback for unregistered agents
        overhead_mb = OVERHEAD_HEURISTICS["unknown"]

    cache["agent_profiles"][agent_name] = {
        "overhead_mb": overhead_mb,
        "profiled_at": datetime.now().isoformat(),
        "method": "registry_heuristic",
    }

    return cache
