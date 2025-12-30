"""Search algorithms for finding optimal configurations."""

import logging
from typing import Optional

from .testing import validate_config
from .utils import get_hardware_id, round_to_valid_scale

logger = logging.getLogger(__name__)


def estimate_max_envs(
    env_name: str,
    agent_name: str,
    buffer_size: Optional[int],
    cache: dict,
) -> tuple[int, int]:
    """Estimate maximum num_envs from cached profiles.

    Args:
        env_name: Environment name
        agent_name: Agent name
        buffer_size: Replay buffer size (if applicable)
        cache: Autotune cache

    Returns:
        Tuple of (estimated_max_envs, suggested_chunk_size)
    """
    hardware_id = get_hardware_id()

    env_profile = cache["env_profiles"][env_name]
    agent_profile = cache["agent_profiles"][agent_name]
    hardware = cache["hardware"][hardware_id]

    # Available memory (MB)
    available_mb = hardware["available_memory_gb"] * 1024

    # Agent overhead
    agent_overhead_mb = agent_profile["overhead_mb"]
    available_mb -= agent_overhead_mb

    # Buffer overhead (rough estimate)
    if buffer_size:
        # Assume ~80 bytes per transition
        buffer_mb = (buffer_size * 80) / (1024**2)
        available_mb -= buffer_mb

    # Apply safety margin (80% utilization for conservative estimate)
    available_mb *= 0.8

    # Per-env memory
    per_env_mb = env_profile["memory_mb_per_env"]

    # Estimate max envs
    estimated_max = int(available_mb / per_env_mb)

    # Round to valid scale
    estimated_max = round_to_valid_scale(estimated_max)

    # Suggest chunk size based on num_envs
    if estimated_max < 10_000:
        chunk_size = 256
    elif estimated_max < 100_000:
        chunk_size = 128
    elif estimated_max < 1_000_000:
        chunk_size = 64
    else:
        chunk_size = 32

    return estimated_max, chunk_size


def probe_upward(
    env_name: str,
    agent_name: str,
    start_envs: int,
    chunk_size: int,
) -> tuple[int, int, float, float]:
    """Probe upward from conservative estimate to find actual maximum.

    Args:
        env_name: Environment name
        agent_name: Agent name
        start_envs: Starting number of environments (conservative)
        chunk_size: Scan chunk size

    Returns:
        Tuple of (max_envs, optimal_chunk, throughput, memory_gb)
    """
    logger.info("[3/3] Finding actual maximum...")

    current = start_envs
    last_success = start_envs
    last_throughput = 0.0
    last_memory = 0.0

    # Phase 1: Exponential growth
    multiplier = 1.5
    for _ in range(5):  # Max 5 exponential steps
        next_envs = int(current * multiplier)
        next_envs = round_to_valid_scale(next_envs)

        if next_envs == current:  # Can't grow further
            break

        logger.info(f"  Probing {next_envs:,} envs...")

        success, throughput, memory = validate_config(env_name, agent_name, next_envs, chunk_size)

        if success and throughput is not None:
            logger.info("  ✓")
            last_success = next_envs
            last_throughput = throughput
            last_memory = memory if memory else 0.0
            current = next_envs
        else:
            logger.info("  ✗ (OOM)")
            break

    # Phase 2: Binary search refinement
    low = last_success
    high = int(current * multiplier) if last_success == current else current

    iterations = 0
    max_iterations = 5

    while iterations < max_iterations and high > low:
        mid = (low + high) // 2
        mid = round_to_valid_scale(mid)

        if mid == low or mid == high:
            break

        logger.info(f"  Refining {mid:,} envs...")

        success, throughput, memory = validate_config(env_name, agent_name, mid, chunk_size)

        if success and throughput is not None:
            logger.info("  ✓")
            last_success = mid
            last_throughput = throughput
            last_memory = memory if memory else 0.0
            low = mid
        else:
            logger.info("  ✗ (OOM)")
            high = mid

        iterations += 1

    return last_success, chunk_size, last_throughput, last_memory
