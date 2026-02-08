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

import sys
import time
from pathlib import Path
from typing import Any, Callable

import jax
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


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
    measure_compile_time: bool = True,
    **kwargs,
) -> dict[str, float]:
    """Time a jitted function with proper warmup and blocking.

    This is the gold standard for benchmarking JAX functions. It:
    1. Measures compilation time (first call)
    2. Warms up the function (additional calls for optimization)
    3. Runs multiple iterations with proper blocking
    4. Reports statistical summary

    Args:
        fn: The jitted function to time
        *args: Arguments to pass to fn
        num_runs: Number of timing runs (default: 100)
        warmup_steps: Number of warmup iterations (default: 10)
        measure_compile_time: Whether to measure compilation time separately (default: True)
        **kwargs: Keyword arguments to pass to fn

    Returns:
        Dictionary with timing statistics:
        - mean: Mean execution time (seconds)
        - std: Standard deviation (seconds)
        - min: Minimum time (seconds)
        - max: Maximum time (seconds)
        - median: Median time (seconds)
        - all_times: Array of all timing measurements
        - compile_time: Time for first call including compilation (seconds, if measure_compile_time=True)
    """
    # Measure compilation time (first call)
    compile_time = None
    if measure_compile_time:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, result)
        compile_time = time.perf_counter() - start

    # Warmup phase: additional calls for optimization (if warmup requested)
    if warmup_steps > 0:
        # If we measured compile time, we've already done 1 warmup
        actual_warmup = warmup_steps - 1 if measure_compile_time else warmup_steps
        if actual_warmup > 0:
            warmup_jitted_fn(fn, *args, warmup_steps=actual_warmup, **kwargs)

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

    stats = {
        "mean": float(np.mean(times_array)),
        "std": float(np.std(times_array)),
        "min": float(np.min(times_array)),
        "max": float(np.max(times_array)),
        "median": float(np.median(times_array)),
        "all_times": times_array,
    }

    if measure_compile_time:
        stats["compile_time"] = compile_time

    return stats


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
