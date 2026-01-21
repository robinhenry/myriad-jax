"""Throughput scaling benchmark: steps/second vs num_envs.

This benchmark measures how environment throughput scales with the number of
parallel environments. It tests the core value proposition of Myriad:
massively parallel simulation on a single device.

Key Metrics
-----------
1. Total throughput (steps/second across all environments)
2. Per-environment throughput (overhead metric)
3. Scaling efficiency (actual vs ideal speedup)
4. Memory usage at each scale

Usage
-----
Run on default device (CPU or GPU):
    python benchmarks/throughput.py

Run on specific device:
    JAX_PLATFORM_NAME=cpu python benchmarks/throughput.py
    JAX_PLATFORM_NAME=gpu python benchmarks/throughput.py

Test a single configuration:
    python benchmarks/throughput.py --num-envs 100000 --scan-chunk-size 128

Run full benchmark suite:
    python benchmarks/throughput.py --full
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import jax

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.config import get_throughput_configs
from benchmarks.utils import (
    calculate_throughput,
    format_number,
    get_device_info,
    time_jitted_fn,
    warmup_jitted_fn,
)
from myriad.agents.classical.random import make_agent as make_random_agent
from myriad.envs import make_env
from myriad.platform.shared import TrainingEnvState
from myriad.platform.step_functions import make_train_step_fn
from myriad.utils import to_array
from myriad.utils.memory import estimate_pytree_memory_mb


def setup_environment(env_name: str, num_envs: int):
    """Create environment and random agent for benchmarking.

    Args:
        env_name: Either "cartpole" or "gene-circuit"
        num_envs: Number of parallel environments

    Returns:
        Tuple of (env, agent, initial_state)
    """
    # Create environment
    if env_name == "cartpole":
        env = make_env("cartpole-control")
    elif env_name == "gene-circuit":
        env = make_env("ccas-ccar-control")
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    # Create random agent (no learning, pure environment throughput)
    action_space = env.get_action_space(env.config)
    agent = make_random_agent(action_space=action_space)

    # Initialize environment states
    key = jax.random.PRNGKey(0)
    reset_keys = jax.random.split(key, num_envs)

    # Vmap reset to initialize all environments
    vmapped_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    obs, env_states = vmapped_reset(reset_keys, env.params, env.config)

    # Convert to arrays for platform compatibility
    to_array_batch = jax.vmap(to_array)
    obs_array = to_array_batch(obs)

    training_state = TrainingEnvState(env_state=env_states, obs=obs_array)

    # Initialize agent state (provide sample observation)
    agent_state = agent.init(key, obs_array[0], agent.params)

    return env, agent, training_state, agent_state


def create_benchmark_step_fn(env, agent, num_envs: int):
    """Create a jitted step function for benchmarking.

    This creates a simplified training step without replay buffer,
    focused on pure environment throughput.

    Args:
        env: Environment
        agent: Agent (RandomAgent for pure env benchmarking)
        num_envs: Number of parallel environments

    Returns:
        Jitted step function
    """
    # Create training step without replay buffer
    step_fn = make_train_step_fn(
        agent=agent,
        env=env,
        replay_buffer=None,  # No replay buffer for Random agent
        num_envs=num_envs,
    )

    return step_fn


def benchmark_configuration(
    env_name: str,
    num_envs: int,
    scan_chunk_size: int,
    num_steps: int = 1000,
    warmup_steps: int = 10,
    num_timing_runs: int = 5,
) -> dict:
    """Benchmark a single configuration.

    Args:
        env_name: Environment to benchmark
        num_envs: Number of parallel environments
        scan_chunk_size: JAX scan chunk size
        num_steps: Number of steps to run for timing
        warmup_steps: Warmup iterations
        num_timing_runs: Number of timing runs for statistics

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking: {env_name}")
    print(f"  num_envs={format_number(num_envs)}, scan_chunk_size={scan_chunk_size}")
    print(f"  Device: {jax.devices()[0]}")
    print(f"{'='*70}")

    try:
        # Setup
        env, agent, training_state, agent_state = setup_environment(env_name, num_envs)
        step_fn = create_benchmark_step_fn(env, agent, num_envs)

        # Estimate memory usage
        state_memory = estimate_pytree_memory_mb(training_state)
        agent_memory = estimate_pytree_memory_mb(agent_state)
        total_memory = state_memory + agent_memory

        print(f"  Estimated state memory: {state_memory:.2f} MB")
        print(f"  Estimated agent memory: {agent_memory:.2f} MB")
        print(f"  Total memory: {total_memory:.2f} MB")

        # Create scan loop for multiple steps
        def run_n_steps(key, agent_state, training_state, n_steps: int):
            """Run n training steps using lax.scan."""

            def scan_body(carry, _):
                key, agent_state, training_state = carry
                key, new_agent_state, new_training_state, _, metrics = step_fn(
                    key, agent_state, training_state, None, batch_size=1
                )
                return (key, new_agent_state, new_training_state), metrics

            initial_carry = (key, agent_state, training_state)
            final_carry, _ = jax.lax.scan(scan_body, initial_carry, None, length=n_steps)
            return final_carry

        # JIT the scan loop with scan_chunk_size as a static argument
        jitted_run = jax.jit(run_n_steps, static_argnames=["n_steps"])

        # Warmup
        print(f"  Warming up ({warmup_steps} iterations)...")
        key = jax.random.PRNGKey(42)
        warmup_jitted_fn(
            jitted_run,
            key,
            agent_state,
            training_state,
            num_steps,
            warmup_steps=warmup_steps,
        )

        # Timing runs
        print(f"  Running benchmark ({num_timing_runs} runs of {num_steps} steps)...")
        timing_results = time_jitted_fn(
            jitted_run,
            key,
            agent_state,
            training_state,
            num_steps,
            num_runs=num_timing_runs,
            warmup_steps=0,  # Already warmed up
        )

        # Calculate metrics
        time_per_run = timing_results["mean"]
        total_steps_per_run = num_steps * num_envs
        throughput = calculate_throughput(total_steps_per_run, time_per_run)
        throughput_per_env = throughput / num_envs

        # Results
        results = {
            "env_name": env_name,
            "num_envs": num_envs,
            "scan_chunk_size": scan_chunk_size,
            "num_steps": num_steps,
            "device": str(jax.devices()[0].platform),
            "time_mean_s": timing_results["mean"],
            "time_std_s": timing_results["std"],
            "time_min_s": timing_results["min"],
            "time_max_s": timing_results["max"],
            "total_throughput_steps_per_s": throughput,
            "per_env_throughput_steps_per_s": throughput_per_env,
            "state_memory_mb": state_memory,
            "agent_memory_mb": agent_memory,
            "total_memory_mb": total_memory,
            "success": True,
        }

        print("\n  Results:")
        print(f"    Throughput: {format_number(throughput)} steps/s (total)")
        print(f"    Per-env:    {throughput_per_env:.2f} steps/s")
        print(f"    Time:       {timing_results['mean']:.4f} ± {timing_results['std']:.4f} s")
        print(f"    Memory:     {total_memory:.2f} MB")

        return results

    except Exception as e:
        print(f"\n  ERROR: {str(e)}")
        return {
            "env_name": env_name,
            "num_envs": num_envs,
            "scan_chunk_size": scan_chunk_size,
            "num_steps": num_steps,
            "device": str(jax.devices()[0].platform),
            "error": str(e),
            "success": False,
        }


def save_results(results: list[dict], output_file: Path):
    """Save benchmark results to CSV.

    Args:
        results: List of result dictionaries
        output_file: Path to output CSV file
    """
    if not results:
        print("No results to save")
        return

    # Get all unique keys from results
    fieldnames = set()
    for result in results:
        fieldnames.update(result.keys())
    fieldnames = sorted(fieldnames)

    # Write CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_file}")


def main():
    """Run throughput scaling benchmark."""
    parser = argparse.ArgumentParser(description="Throughput scaling benchmark")
    parser.add_argument(
        "--num-envs",
        type=int,
        help="Number of parallel environments (for single test)",
    )
    parser.add_argument(
        "--scan-chunk-size",
        type=int,
        default=256,
        help="JAX scan chunk size (default: 256)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1000,
        help="Number of steps to run (default: 1000)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="cartpole",
        choices=["cartpole", "gene-circuit"],
        help="Environment to benchmark (default: cartpole)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full benchmark suite (all configurations)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file (default: benchmarks/results/throughput_<device>_<timestamp>.csv)",
    )

    args = parser.parse_args()

    # Print device info
    device_info = get_device_info()
    print("\n" + "=" * 70)
    print("Myriad Throughput Scaling Benchmark")
    print("=" * 70)
    print(f"Device: {device_info['platform']}")
    print(f"Devices: {device_info['devices']}")
    print("=" * 70)

    # Determine configurations to run (auto-detected based on device)
    # Config structure: {env_name: [BenchmarkConfig, ...]}
    if args.full:
        # Run full test matrix (device and env-appropriate configs from config.yaml)
        env_configs = get_throughput_configs()  # Returns dict: env -> configs
    elif args.num_envs:
        # Single configuration for specified env
        env_configs = {
            args.env: [
                type(
                    "Config",
                    (),
                    {
                        "num_envs": args.num_envs,
                        "scan_chunk_size": args.scan_chunk_size,
                        "num_steps": args.num_steps,
                        "warmup_steps": 10,
                        "num_timing_runs": 5,
                    },
                )()
            ]
        }
    else:
        # Quick test: first 3 configs for cartpole only
        print("\nRunning quick test. Use --full for complete benchmark or --num-envs for custom test.")
        env_configs = {"cartpole": get_throughput_configs(env="cartpole")[:3]}

    # Run benchmarks
    all_results = []
    for env_name, configs in env_configs.items():
        for config in configs:
            result = benchmark_configuration(
                env_name=env_name,
                num_envs=config.num_envs,
                scan_chunk_size=config.scan_chunk_size,
                num_steps=config.num_steps,
                warmup_steps=config.warmup_steps,
                num_timing_runs=config.num_timing_runs,
            )
            all_results.append(result)

    # Save results
    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        device_name = device_info["platform"]
        output_file = Path(f"benchmarks/results/throughput_{device_name}_{timestamp}.csv")

    save_results(all_results, output_file)

    # Print summary
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    successful = [r for r in all_results if r.get("success", False)]
    failed = [r for r in all_results if not r.get("success", False)]

    print(f"Successful: {len(successful)}/{len(all_results)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for r in failed:
            print(f"  - {r['env_name']} @ {format_number(r['num_envs'])} envs: {r.get('error', 'Unknown error')}")

    if successful:
        print("\nTop configurations by throughput:")
        sorted_results = sorted(successful, key=lambda x: x["total_throughput_steps_per_s"], reverse=True)
        for i, r in enumerate(sorted_results[:5], 1):
            throughput = r["total_throughput_steps_per_s"]
            num_envs = r["num_envs"]
            env_name = r["env_name"]
            print(
                f"  {i}. {env_name:15s} @ {format_number(num_envs):>8s} envs: {format_number(throughput):>8s} steps/s"
            )

    print("=" * 70)


if __name__ == "__main__":
    main()
