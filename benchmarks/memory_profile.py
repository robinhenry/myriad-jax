"""Memory profiling benchmark: Track memory usage vs num_envs and scan_chunk_size.

This benchmark measures memory consumption to:
1. Understand memory scaling with num_envs
2. Optimize scan_chunk_size for memory efficiency
3. Identify memory bottlenecks (state size vs scan intermediates)
4. Determine maximum feasible num_envs for given hardware

Usage
-----
Profile memory vs num_envs:
    python benchmarks/memory_profile.py --profile envs

Profile memory vs scan_chunk_size:
    python benchmarks/memory_profile.py --profile scan

Profile both:
    python benchmarks/memory_profile.py --profile both
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import jax
import psutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.config import SCAN_CHUNK_SENSITIVITY_CONFIGS, THROUGHPUT_CONFIGS
from benchmarks.utils import (
    estimate_pytree_memory_mb,
    format_number,
    get_device_info,
)
from myriad.agents.classical.random import make_agent as make_random_agent
from myriad.envs import make_env
from myriad.platform.shared import TrainingEnvState
from myriad.platform.step_functions import make_train_step_fn
from myriad.utils import to_array


def get_process_memory_mb() -> float:
    """Get current process memory usage in MB.

    Returns:
        Memory usage in MB
    """
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def get_gpu_memory_mb() -> dict[str, float]:
    """Get GPU memory usage in MB (if available).

    Returns:
        Dictionary with allocated and reserved memory, or empty dict if N/A
    """
    if jax.devices()[0].platform != "gpu":
        return {}

    try:
        # Try to get memory stats from default device
        device = jax.devices()[0]
        stats = device.memory_stats()
        return {
            "bytes_in_use": stats.get("bytes_in_use", 0) / (1024 * 1024),
            "peak_bytes_in_use": stats.get("peak_bytes_in_use", 0) / (1024 * 1024),
        }
    except Exception:
        return {}


def profile_memory_at_scale(
    env_name: str,
    num_envs: int,
    scan_chunk_size: int,
) -> dict:
    """Profile memory usage for a specific configuration.

    Args:
        env_name: Environment to profile
        num_envs: Number of parallel environments
        scan_chunk_size: JAX scan chunk size

    Returns:
        Dictionary with memory measurements
    """
    print(f"\nProfiling: {env_name} @ {format_number(num_envs)} envs, chunk={scan_chunk_size}")

    try:
        # Baseline memory before setup
        baseline_cpu_memory = get_process_memory_mb()
        baseline_gpu_memory = get_gpu_memory_mb()

        # Create environment
        if env_name == "cartpole":
            env = make_env("cartpole-control")
        elif env_name == "gene-circuit":
            env = make_env("ccas-ccar-control")
        else:
            raise ValueError(f"Unknown environment: {env_name}")

        # Create random agent
        action_space = env.get_action_space(env.config)
        agent = make_random_agent(action_space=action_space)

        # Initialize environments
        key = jax.random.PRNGKey(0)
        reset_keys = jax.random.split(key, num_envs)

        vmapped_reset = jax.vmap(env.reset, in_axes=(0, None, None))
        obs, env_states = vmapped_reset(reset_keys, env.params, env.config)

        to_array_batch = jax.vmap(to_array)
        obs_array = to_array_batch(obs)

        training_state = TrainingEnvState(env_state=env_states, obs=obs_array)
        agent_state = agent.init(key, obs_array[0], agent.params)

        # Measure PyTree memory
        state_memory = estimate_pytree_memory_mb(training_state)
        agent_memory = estimate_pytree_memory_mb(agent_state)

        # Create and JIT step function (triggers compilation)
        step_fn = make_train_step_fn(agent, env, None, num_envs)
        step_fn = jax.jit(step_fn, static_argnames=["batch_size"])

        # Run a single step to trigger compilation and measure post-compile memory
        key, new_agent_state, new_training_state, _, _ = step_fn(key, agent_state, training_state, None, batch_size=1)
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            (key, new_agent_state, new_training_state),
        )

        # Memory after compilation
        post_compile_cpu_memory = get_process_memory_mb()
        post_compile_gpu_memory = get_gpu_memory_mb()

        # Calculate differences
        cpu_memory_delta = post_compile_cpu_memory - baseline_cpu_memory
        gpu_memory_delta = (
            post_compile_gpu_memory.get("bytes_in_use", 0) - baseline_gpu_memory.get("bytes_in_use", 0)
            if post_compile_gpu_memory and baseline_gpu_memory
            else 0
        )

        results = {
            "env_name": env_name,
            "num_envs": num_envs,
            "scan_chunk_size": scan_chunk_size,
            "device": str(jax.devices()[0].platform),
            "state_memory_mb": state_memory,
            "agent_memory_mb": agent_memory,
            "pytree_total_mb": state_memory + agent_memory,
            "baseline_cpu_mb": baseline_cpu_memory,
            "post_compile_cpu_mb": post_compile_cpu_memory,
            "cpu_delta_mb": cpu_memory_delta,
        }

        # Add GPU metrics if available
        if post_compile_gpu_memory:
            results["gpu_allocated_mb"] = post_compile_gpu_memory.get("bytes_in_use", 0)
            results["gpu_peak_mb"] = post_compile_gpu_memory.get("peak_bytes_in_use", 0)
            results["gpu_delta_mb"] = gpu_memory_delta

        print(f"  PyTree memory: {state_memory + agent_memory:.2f} MB")
        print(f"  CPU delta: {cpu_memory_delta:.2f} MB")
        if gpu_memory_delta:
            print(f"  GPU delta: {gpu_memory_delta:.2f} MB")

        results["success"] = True
        return results

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return {
            "env_name": env_name,
            "num_envs": num_envs,
            "scan_chunk_size": scan_chunk_size,
            "device": str(jax.devices()[0].platform),
            "error": str(e),
            "success": False,
        }


def save_results(results: list[dict], output_file: Path):
    """Save memory profiling results to CSV."""
    if not results:
        print("No results to save")
        return

    fieldnames = set()
    for result in results:
        fieldnames.update(result.keys())
    fieldnames = sorted(fieldnames)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Memory profiling benchmark")
    parser.add_argument(
        "--profile",
        type=str,
        default="envs",
        choices=["envs", "scan", "both"],
        help="What to profile: envs (num_envs scaling), scan (scan_chunk_size), or both",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="cartpole",
        choices=["cartpole", "gene-circuit"],
        help="Environment to profile (default: cartpole)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file",
    )

    args = parser.parse_args()

    # Print device info
    device_info = get_device_info()
    print("\n" + "=" * 70)
    print("Myriad Memory Profiling Benchmark")
    print("=" * 70)
    print(f"Device: {device_info['platform']}")
    print("=" * 70)

    all_results = []

    # Profile num_envs scaling
    if args.profile in ["envs", "both"]:
        print("\n>>> Profiling memory vs num_envs")
        for config in THROUGHPUT_CONFIGS:
            result = profile_memory_at_scale(
                env_name=args.env,
                num_envs=config.num_envs,
                scan_chunk_size=config.scan_chunk_size,
            )
            all_results.append(result)

    # Profile scan_chunk_size sensitivity
    if args.profile in ["scan", "both"]:
        print("\n>>> Profiling memory vs scan_chunk_size")
        for config in SCAN_CHUNK_SENSITIVITY_CONFIGS:
            result = profile_memory_at_scale(
                env_name=args.env,
                num_envs=config.num_envs,
                scan_chunk_size=config.scan_chunk_size,
            )
            all_results.append(result)

    # Save results
    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        device_name = device_info["platform"]
        profile_type = args.profile
        output_file = Path(f"benchmarks/results/memory_{profile_type}_{device_name}_{timestamp}.csv")

    save_results(all_results, output_file)

    # Print summary
    print("\n" + "=" * 70)
    print("Memory Profiling Summary")
    print("=" * 70)
    successful = [r for r in all_results if r.get("success", False)]
    if successful:
        print(f"Successfully profiled {len(successful)} configurations")
        max_memory = max(r.get("cpu_delta_mb", 0) for r in successful)
        max_config = max(successful, key=lambda r: r.get("cpu_delta_mb", 0))
        print(f"Peak memory: {max_memory:.2f} MB @ {format_number(max_config['num_envs'])} envs")
    print("=" * 70)


if __name__ == "__main__":
    main()
