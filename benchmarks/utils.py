"""Utilities for benchmarking JAX code with proper timing and warmup protocols.

Critical JAX Benchmarking Rules
--------------------------------
1. **Always use .block_until_ready()**: JAX is asynchronous. Without blocking,
   you measure dispatch time, not actual computation time.

2. **Warmup runs**: First execution includes JIT compilation. Always discard
   warmup runs from timing statistics.

3. **Multiple runs**: Report mean ± std to account for system noise.

4. **Device synchronization**: Ensure GPU work completes before timing.
"""

import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np


def get_device_info() -> dict[str, Any]:
    """Get information about available JAX devices.

    Returns:
        Dictionary with device type, count, and memory info (if available).
    """
    devices = jax.devices()
    device_type = devices[0].platform
    device_count = len(devices)

    info = {
        "platform": device_type,
        "device_count": device_count,
        "devices": [str(d) for d in devices],
    }

    # Try to get memory info for GPUs
    if device_type == "gpu":
        try:
            # This is device-specific and may not work on all platforms
            from jax.lib import xla_bridge

            backend = xla_bridge.get_backend()
            if hasattr(backend, "get_memory_info"):
                info["memory_info"] = backend.get_memory_info(devices[0])
        except Exception:
            pass  # Not all platforms support memory queries

    return info


def warmup_jitted_fn(fn: Callable, *args, warmup_steps: int = 10, **kwargs) -> None:
    """Warm up a jitted function to trigger compilation and optimization.

    Critical: Always warmup before benchmarking to exclude compilation time.

    Args:
        fn: The jitted function to warm up
        *args: Arguments to pass to fn
        warmup_steps: Number of warmup iterations (default: 10)
        **kwargs: Keyword arguments to pass to fn
    """
    for _ in range(warmup_steps):
        result = fn(*args, **kwargs)
        # CRITICAL: Block to ensure compilation completes
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, result)


def time_jitted_fn(
    fn: Callable,
    *args,
    num_runs: int = 100,
    warmup_steps: int = 10,
    **kwargs,
) -> dict[str, float]:
    """Time a jitted function with proper warmup and blocking.

    This is the gold standard for benchmarking JAX functions. It:
    1. Warms up the function to trigger compilation
    2. Runs multiple iterations with proper blocking
    3. Reports statistical summary

    Args:
        fn: The jitted function to time
        *args: Arguments to pass to fn
        num_runs: Number of timing runs (default: 100)
        warmup_steps: Number of warmup iterations (default: 10)
        **kwargs: Keyword arguments to pass to fn

    Returns:
        Dictionary with timing statistics:
        - mean: Mean execution time (seconds)
        - std: Standard deviation (seconds)
        - min: Minimum time (seconds)
        - max: Maximum time (seconds)
        - median: Median time (seconds)
        - all_times: Array of all timing measurements
    """
    # Warmup phase: trigger compilation and optimization
    warmup_jitted_fn(fn, *args, warmup_steps=warmup_steps, **kwargs)

    # Timing phase: measure steady-state performance
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        # CRITICAL: Block until GPU/CPU work completes
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, result)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times_array = np.array(times)

    return {
        "mean": float(np.mean(times_array)),
        "std": float(np.std(times_array)),
        "min": float(np.min(times_array)),
        "max": float(np.max(times_array)),
        "median": float(np.median(times_array)),
        "all_times": times_array,
    }


def measure_compilation_time(fn: Callable, *args, **kwargs) -> float:
    """Measure compilation time for a jitted function.

    This measures the overhead of the first call, which includes XLA compilation.

    Args:
        fn: The jitted function to measure
        *args: Arguments to pass to fn
        **kwargs: Keyword arguments to pass to fn

    Returns:
        Compilation time in seconds
    """
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, result)
    return time.perf_counter() - start


def get_array_memory_mb(arr: jnp.ndarray) -> float:
    """Calculate memory usage of a JAX array in megabytes.

    Args:
        arr: JAX array

    Returns:
        Memory usage in MB
    """
    return arr.nbytes / (1024 * 1024)


def estimate_pytree_memory_mb(pytree: Any) -> float:
    """Estimate total memory usage of a PyTree in megabytes.

    Args:
        pytree: A JAX PyTree (nested structure of arrays)

    Returns:
        Estimated memory usage in MB
    """
    leaves = jax.tree_util.tree_leaves(pytree)
    total_bytes = sum(leaf.nbytes for leaf in leaves if isinstance(leaf, (jnp.ndarray, np.ndarray)))
    return total_bytes / (1024 * 1024)


def format_number(num: float, precision: int = 2) -> str:
    """Format large numbers with K/M/B suffixes for readability.

    Args:
        num: Number to format
        precision: Decimal places (default: 2)

    Returns:
        Formatted string (e.g., "1.5M", "100K")
    """
    if num >= 1e9:
        return f"{num/1e9:.{precision}f}B"
    elif num >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def calculate_throughput(total_steps: int, time_seconds: float) -> float:
    """Calculate throughput in steps/second.

    Args:
        total_steps: Total number of steps executed
        time_seconds: Time taken in seconds

    Returns:
        Throughput in steps/second
    """
    return total_steps / time_seconds


def calculate_scaling_efficiency(
    throughput_n: float,
    throughput_baseline: float,
    n_envs: int,
    baseline_envs: int = 1,
) -> float:
    """Calculate parallel scaling efficiency.

    Perfect scaling: efficiency = 1.0
    Sublinear scaling: efficiency < 1.0
    Superlinear scaling: efficiency > 1.0 (rare, usually caching effects)

    Args:
        throughput_n: Throughput at N environments
        throughput_baseline: Throughput at baseline (usually 1 env)
        n_envs: Number of environments at N
        baseline_envs: Number of environments at baseline

    Returns:
        Scaling efficiency (0.0 to 1.0 typically)
    """
    # Expected throughput if scaling were perfect
    expected_throughput = throughput_baseline * (n_envs / baseline_envs)
    return throughput_n / expected_throughput
