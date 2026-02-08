"""Search algorithms for finding optimal configurations."""

import logging

from .testing import validate_config
from .utils import ceil_to_valid_scale, floor_to_valid_scale, get_hardware_id

logger = logging.getLogger(__name__)

# Constants for search heuristics
ESTIMATION_SAFETY_MARGIN = 0.8
BUFFER_BYTES_PER_TRANSITION = 80
DEFAULT_START_CHUNK_SIZE = 64
PROBE_MULTIPLIER = 1.5
MAX_PROBE_STEPS = 5
MAX_BINARY_SEARCH_STEPS = 5


def estimate_max_envs(
    env_name: str,
    agent_name: str,
    buffer_size: int | None,
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
        buffer_mb = (buffer_size * BUFFER_BYTES_PER_TRANSITION) / (1024**2)
        available_mb -= buffer_mb

    # Apply safety margin for conservative estimate
    available_mb *= ESTIMATION_SAFETY_MARGIN

    # Per-env memory
    per_env_mb = env_profile["memory_mb_per_env"]

    # Estimate max envs
    estimated_max = int(available_mb / per_env_mb)

    # Round down to valid scale for conservative estimate
    estimated_max = floor_to_valid_scale(estimated_max)

    # Use a safe default chunk size for initial probing
    return estimated_max, DEFAULT_START_CHUNK_SIZE


def probe_upward(
    env_name: str,
    start_envs: int,
    chunk_size: int,
) -> tuple[int, int, float, float]:
    """Probe upward from conservative estimate to find actual maximum.

    Args:
        env_name: Environment name
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
    multiplier = PROBE_MULTIPLIER
    for _ in range(MAX_PROBE_STEPS):
        next_envs = int(current * multiplier)
        next_envs = ceil_to_valid_scale(next_envs)

        if next_envs == current:  # Can't grow further
            break

        logger.debug(f"  Probing {next_envs:,} envs...")

        success, throughput, memory = validate_config(env_name, next_envs)

        if success and throughput is not None:
            logger.debug("  ✓")
            last_success = next_envs
            last_throughput = throughput
            last_memory = memory if memory else 0.0
            current = next_envs
        else:
            logger.debug("  ✗ (OOM)")
            break

    # Phase 2: Binary search refinement
    low = last_success
    high = int(current * multiplier) if last_success == current else current

    iterations = 0
    max_iterations = MAX_BINARY_SEARCH_STEPS

    while iterations < max_iterations and high > low:
        mid = (low + high) // 2
        mid = ceil_to_valid_scale(mid)

        if mid == low or mid == high:
            break

        logger.debug(f"  Refining {mid:,} envs...")

        success, throughput, memory = validate_config(env_name, mid)

        if success and throughput is not None:
            logger.debug("  ✓")
            last_success = mid
            last_throughput = throughput
            last_memory = memory if memory else 0.0
            low = mid
        else:
            logger.debug("  ✗ (OOM)")
            high = mid

        iterations += 1

    return last_success, chunk_size, last_throughput, last_memory
