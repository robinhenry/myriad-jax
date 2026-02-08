"""Main API functions for auto-tuning."""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from .cache import load_cache, save_cache
from .profiling import profile_agent, profile_env, profile_hardware
from .search import estimate_max_envs, probe_upward
from .testing import validate_config
from .utils import (
    VALID_CHUNK_SIZES,
    floor_to_valid_scale,
    get_hardware_id,
    make_config_key,
)

logger = logging.getLogger(__name__)


@dataclass
class AutotuneResult:
    """Result from auto-tuning (for suggest_config advanced API)."""

    max_envs: int
    optimal_chunk_size: int
    throughput_steps_per_s: float
    memory_usage_gb: float
    profiling_time_s: float
    from_cache: bool

    def __repr__(self):
        return (
            f"AutotuneResult(\n"
            f"  max_envs={self.max_envs:,},\n"
            f"  scan_chunk_size={self.optimal_chunk_size},\n"
            f"  throughput={self.throughput_steps_per_s/1e6:.1f}M steps/s,\n"
            f"  memory={self.memory_usage_gb:.1f}GB,\n"
            f"  cached={self.from_cache}\n"
            f")"
        )


def _ensure_profiled(env: str, agent: str, cache: dict, hardware_id: str) -> dict:
    """Ensure hardware, env, and agent are profiled in the cache."""
    changed = False

    if hardware_id not in cache["hardware"]:
        logger.info("  ⚡ Hardware... profiling")
        cache = profile_hardware(cache)
        hw = cache["hardware"][hardware_id]
        logger.info(f"     {hw['device']} ({hw['available_memory_gb']:.1f}GB available)")
        changed = True
    else:
        logger.debug(f"  ⚡ Hardware... cached ({hardware_id})")

    if env not in cache["env_profiles"]:
        logger.info(f"  ⚡ Environment ({env})... profiling (~10s)")
        cache = profile_env(env, cache)
        logger.info("     Saved to cache ✓")
        changed = True
    else:
        logger.debug(f"  ⚡ Environment ({env})... cached")

    if agent not in cache["agent_profiles"]:
        logger.info(f"  ⚡ Agent ({agent})... profiling")
        cache = profile_agent(agent, cache)
        logger.info("     Saved to cache ✓")
        changed = True
    else:
        logger.debug(f"  ⚡ Agent ({agent})... cached")

    if changed:
        save_cache(cache)

    return cache


def suggest_scan_chunk_size(
    num_envs: int,
    env: str,
    agent: str,
    buffer_size: int | None = None,
    force_revalidate: bool = False,
    verbose: bool = True,
) -> int:
    """Suggest optimal scan_chunk_size for a fixed number of environments.

    This is the primary API for users who know how many environments they want
    to run and need to find the optimal scan_chunk_size for their hardware.

    Args:
        num_envs: Fixed number of parallel environments
        env: Environment name (e.g., "cartpole-control")
        agent: Agent name (e.g., "dqn")
        buffer_size: Replay buffer size (for off-policy agents)
        force_revalidate: Force re-profiling even if cached
        verbose: Control log output verbosity

    Returns:
        Optimal scan_chunk_size for the given configuration

    Example:
        >>> chunk_size = suggest_scan_chunk_size(
        ...     num_envs=100_000,
        ...     env="cartpole-control",
        ...     agent="dqn",
        ... )
        >>> config = create_config(
        ...     env="cartpole-control",
        ...     agent="dqn",
        ...     num_envs=100_000,
        ...     scan_chunk_size=chunk_size,
        ... )
    """
    # Set up logging level based on verbose flag
    if not verbose:
        previous_level = logger.level
        logger.setLevel(logging.WARNING)

    start_time = time.time()

    logger.info("Finding optimal scan_chunk_size...")
    logger.debug(f"Environment: {env}, Agent: {agent}, Envs: {num_envs:,}")

    # Load cache
    cache = load_cache()
    hardware_id = get_hardware_id()

    # Create cache key for this specific configuration
    config_key = make_config_key(env, agent, buffer_size, hardware_id) + f":{num_envs}"

    # Check for cached result
    if not force_revalidate and config_key in cache["chunk_size_configs"]:
        cached = cache["chunk_size_configs"][config_key]
        validated_at = datetime.fromisoformat(cached["validated_at"])
        if datetime.now() - validated_at < timedelta(days=30):
            logger.info(f"✓ Using cached scan_chunk_size: {cached['optimal_chunk_size']}")

            if not verbose:
                logger.setLevel(previous_level)

            return cached["optimal_chunk_size"]

    # Profile components if needed (for memory estimation)
    cache = _ensure_profiled(env, agent, cache, hardware_id)

    # Test chunk sizes to find optimal
    logger.info("Testing chunk sizes...")

    best_chunk_size = None
    best_throughput = 0.0

    # Try chunk sizes from large to small (prefer larger for efficiency)
    for chunk_size in reversed(VALID_CHUNK_SIZES):
        logger.debug(f"  Testing chunk_size={chunk_size}... ")

        success, throughput, memory = validate_config(env, num_envs)

        if success and throughput is not None:
            logger.debug(f"    ✓ ({throughput/1e6:.1f}M steps/s, {memory:.1f}GB)")
            if throughput > best_throughput:
                best_chunk_size = chunk_size
                best_throughput = throughput
        else:
            logger.debug("    ✗ (OOM or timeout)")

        # If we found a working chunk size and smaller ones won't improve throughput much,
        # we can stop early
        if best_chunk_size and chunk_size < best_chunk_size // 4:
            logger.debug(f"  Stopping early (found good chunk_size={best_chunk_size})")
            break

    if best_chunk_size is None:
        raise RuntimeError(
            f"No valid chunk_size found for {num_envs:,} environments. "
            "Try reducing num_envs or increasing available memory."
        )

    # Cache the result
    cache["chunk_size_configs"][config_key] = {
        "optimal_chunk_size": best_chunk_size,
        "throughput_steps_per_s": best_throughput,
        "validated_at": datetime.now().isoformat(),
    }
    save_cache(cache)

    profiling_time = time.time() - start_time

    logger.info(f"✅ Found optimal scan_chunk_size: {best_chunk_size} ({best_throughput/1e6:.1f}M steps/s)")
    logger.debug(f"Search took {profiling_time:.1f}s")

    if not verbose:
        logger.setLevel(previous_level)

    return best_chunk_size


def suggest_config(
    env: str,
    agent: str,
    buffer_size: int | None = None,
    force_revalidate: bool = False,
    verbose: bool = True,
) -> AutotuneResult:
    """Find maximum num_envs and optimal scan_chunk_size for hardware.

    Note:
        This is an advanced/benchmarking function. For typical use cases,
        use `suggest_scan_chunk_size()` instead, which optimizes scan_chunk_size
        for a given fixed number of environments.

    This function will:
    1. Check cache for existing validated config
    2. Profile components if needed (lazy)
    3. Estimate conservative configuration
    4. Validate it works
    5. Probe upward to find actual maximum
    6. Cache the result

    Args:
        env: Environment name (e.g., "cartpole-control")
        agent: Agent name (e.g., "dqn")
        buffer_size: Replay buffer size (for off-policy agents)
        force_revalidate: Force re-profiling even if cached
        verbose: Control log output verbosity

    Returns:
        AutotuneResult with optimal configuration
    """
    # Set up logging level based on verbose flag
    if not verbose:
        previous_level = logger.level
        logger.setLevel(logging.WARNING)

    start_time = time.time()

    logger.info("Auto-tuning configuration...")
    logger.debug(f"Environment: {env}, Agent: {agent}")

    # Load cache
    cache = load_cache()
    hardware_id = get_hardware_id()

    # Check for validated config
    config_key = make_config_key(env, agent, buffer_size, hardware_id)

    if not force_revalidate and config_key in cache["validated_configs"]:
        cached = cache["validated_configs"][config_key]

        # Check if recent (< 30 days)
        validated_at = datetime.fromisoformat(cached["validated_at"])
        if datetime.now() - validated_at < timedelta(days=30):
            logger.info("✓ Using cached configuration (validated recently)")

            if not verbose:
                logger.setLevel(previous_level)

            return AutotuneResult(
                max_envs=cached["max_envs"],
                optimal_chunk_size=cached["optimal_chunk_size"],
                throughput_steps_per_s=cached["throughput_steps_per_s"],
                memory_usage_gb=cached["memory_usage_gb"],
                profiling_time_s=time.time() - start_time,
                from_cache=True,
            )

    # Profile components as needed
    logger.info("Profiling components...")
    cache = _ensure_profiled(env, agent, cache, hardware_id)

    # Estimate configuration
    logger.info("Estimating maximum environments...")

    estimated_max, suggested_chunk = estimate_max_envs(env, agent, buffer_size, cache)

    logger.debug(f"  Conservative estimate: {estimated_max:,} envs, chunk_size: {suggested_chunk}")

    # Validate estimate works
    success, throughput, memory = validate_config(env, estimated_max)

    if not success:
        logger.debug("  ✗ (Failed - estimate too high), retrying half...")
        # Estimate was too high, reduce
        estimated_max = estimated_max // 2
        estimated_max = floor_to_valid_scale(estimated_max)
        success, throughput, memory = validate_config(env, estimated_max)
        if not success:
            raise RuntimeError(
                f"Failed to validate configuration even at {estimated_max:,} envs. "
                "This may indicate insufficient memory or other system issues."
            )

    if throughput is not None:
        logger.debug(f"  ✓ (Works at {throughput/1e6:.1f}M steps/s)")

    # Probe upward to find actual max
    max_envs, optimal_chunk, final_throughput, final_memory = probe_upward(env, estimated_max, suggested_chunk)

    # Cache the validated result
    cache["validated_configs"][config_key] = {
        "max_envs": max_envs,
        "optimal_chunk_size": optimal_chunk,
        "throughput_steps_per_s": final_throughput,
        "memory_usage_gb": final_memory,
        "validated_at": datetime.now().isoformat(),
    }
    save_cache(cache)

    profiling_time = time.time() - start_time

    hw = cache["hardware"][hardware_id]
    logger.info(f"✅ Auto-tune complete: {max_envs:,} envs @ {final_throughput/1e6:.1f}M steps/s")
    logger.debug(f"Memory: {final_memory:.1f}GB / {hw['total_memory_gb']:.1f}GB")

    if not verbose:
        logger.setLevel(previous_level)

    return AutotuneResult(
        max_envs=max_envs,
        optimal_chunk_size=optimal_chunk,
        throughput_steps_per_s=final_throughput,
        memory_usage_gb=final_memory,
        profiling_time_s=profiling_time,
        from_cache=False,
    )
