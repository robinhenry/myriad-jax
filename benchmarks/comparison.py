"""Library comparison benchmark: Myriad vs Gymnax vs Gymnasium.

This benchmark compares Myriad against other RL environment libraries:
- Gymnax (JAX-based, fair comparison)
- Gymnasium (CPU-based, shows GPU parallelism advantage)

Key Comparisons
---------------
1. Wall-clock time for fixed total steps
2. Throughput (steps/second)
3. Memory usage
4. Compilation overhead

Fair Comparison Notes
---------------------
- Gymnax: Both JAX-based, both use vmap → fair comparison
- Gymnasium: CPU serial vs GPU parallel → unfair but demonstrates value proposition
- All comparisons use CartPole-v1 environment
- Same total environment steps (10,000 steps)

Usage
-----
Compare all libraries:
    python benchmarks/comparison.py

Compare specific library:
    python benchmarks/comparison.py --library gymnax
    python benchmarks/comparison.py --library gymnasium

Myriad only (baseline):
    python benchmarks/comparison.py --library myriad
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import jax

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.config import get_comparison_config
from benchmarks.utils import (
    calculate_throughput,
    format_number,
    get_device_info,
    measure_compilation_time,
    time_jitted_fn,
)
from myriad.agents.classical.random import make_agent as make_random_agent
from myriad.envs import make_env
from myriad.platform.shared import TrainingEnvState
from myriad.platform.step_functions import make_train_step_fn
from myriad.utils import to_array


def benchmark_myriad(num_envs: int, steps_per_env: int) -> dict:
    """Benchmark Myriad environment throughput.

    Args:
        num_envs: Number of parallel environments
        steps_per_env: Steps per environment

    Returns:
        Benchmark results dictionary
    """
    print("\n>>> Benchmarking Myriad")
    print(f"    {num_envs} parallel environments × {steps_per_env} steps = {num_envs * steps_per_env} total steps")

    # Setup
    env = make_env("cartpole-control")
    action_space = env.get_action_space(env.config)
    agent = make_random_agent(action_space=action_space)

    # Initialize
    key = jax.random.PRNGKey(0)
    reset_keys = jax.random.split(key, num_envs)
    vmapped_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    obs, env_states = vmapped_reset(reset_keys, env.params, env.config)

    to_array_batch = jax.vmap(to_array)
    obs_array = to_array_batch(obs)
    training_state = TrainingEnvState(env_state=env_states, obs=obs_array)
    agent_state = agent.init(key, obs_array[0], agent.params)

    # Create step function
    step_fn = make_train_step_fn(agent, env, None, num_envs)

    def run_steps(key, agent_state, training_state, n_steps: int):
        def scan_body(carry, _):
            key, agent_state, training_state = carry
            key, new_agent_state, new_training_state, _, _ = step_fn(
                key, agent_state, training_state, None, batch_size=1
            )
            return (key, new_agent_state, new_training_state), None

        initial_carry = (key, agent_state, training_state)
        final_carry, _ = jax.lax.scan(scan_body, initial_carry, None, length=n_steps)
        return final_carry

    jitted_run = jax.jit(run_steps, static_argnames=["n_steps"])

    # Measure compilation time
    print("    Measuring compilation time...")
    compile_time = measure_compilation_time(jitted_run, key, agent_state, training_state, steps_per_env)

    # Time execution (excluding compilation)
    print("    Timing execution...")
    timing_results = time_jitted_fn(
        jitted_run,
        key,
        agent_state,
        training_state,
        steps_per_env,
        num_runs=get_comparison_config().num_timing_runs,
        warmup_steps=0,  # Already compiled
    )

    total_steps = num_envs * steps_per_env
    throughput = calculate_throughput(total_steps, timing_results["mean"])

    results = {
        "library": "myriad",
        "num_envs": num_envs,
        "steps_per_env": steps_per_env,
        "total_steps": total_steps,
        "device": str(jax.devices()[0].platform),
        "compile_time_s": compile_time,
        "runtime_mean_s": timing_results["mean"],
        "runtime_std_s": timing_results["std"],
        "throughput_steps_per_s": throughput,
        "total_time_s": compile_time + timing_results["mean"],
    }

    print(f"    Compilation: {compile_time:.4f} s")
    print(f"    Runtime:     {timing_results['mean']:.4f} ± {timing_results['std']:.4f} s")
    print(f"    Throughput:  {format_number(throughput)} steps/s")

    return results


def benchmark_gymnax(num_envs: int, steps_per_env: int) -> dict:
    """Benchmark Gymnax environment throughput.

    Args:
        num_envs: Number of parallel environments
        steps_per_env: Steps per environment

    Returns:
        Benchmark results dictionary
    """
    print("\n>>> Benchmarking Gymnax")
    print(f"    {num_envs} parallel environments × {steps_per_env} steps = {num_envs * steps_per_env} total steps")

    try:
        import gymnax
    except ImportError:
        print("    ERROR: gymnax not installed. Install with: pip install gymnax")
        return {
            "library": "gymnax",
            "error": "gymnax not installed",
            "success": False,
        }

    try:
        # Create environment
        env, env_params = gymnax.make("CartPole-v1")

        # Initialize environments
        key = jax.random.PRNGKey(0)
        reset_keys = jax.random.split(key, num_envs)

        vmapped_reset = jax.vmap(env.reset, in_axes=(0, None))
        obs, env_states = vmapped_reset(reset_keys, env_params)

        # Random action function
        def random_action(key):
            return jax.random.randint(key, (), 0, env.num_actions)

        def run_steps(key, env_states, obs, n_steps: int):
            """Run n steps across all parallel environments."""

            def scan_body(carry, _):
                key, env_states, obs = carry
                # Split keys for actions and steps
                key, action_key, step_key = jax.random.split(key, 3)
                action_keys = jax.random.split(action_key, num_envs)
                step_keys = jax.random.split(step_key, num_envs)

                # Random actions
                actions = jax.vmap(random_action)(action_keys)

                # Step environments
                vmapped_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
                next_obs, next_states, rewards, dones, _ = vmapped_step(step_keys, env_states, actions, env_params)

                # Auto-reset (simplified - Gymnax handles this internally in some versions)
                return (key, next_states, next_obs), None

            initial_carry = (key, env_states, obs)
            final_carry, _ = jax.lax.scan(scan_body, initial_carry, None, length=n_steps)
            return final_carry

        jitted_run = jax.jit(run_steps, static_argnames=["n_steps"])

        # Measure compilation time
        print("    Measuring compilation time...")
        compile_time = measure_compilation_time(jitted_run, key, env_states, obs, steps_per_env)

        # Time execution
        print("    Timing execution...")
        timing_results = time_jitted_fn(
            jitted_run,
            key,
            env_states,
            obs,
            steps_per_env,
            num_runs=get_comparison_config().num_timing_runs,
            warmup_steps=0,
        )

        total_steps = num_envs * steps_per_env
        throughput = calculate_throughput(total_steps, timing_results["mean"])

        results = {
            "library": "gymnax",
            "num_envs": num_envs,
            "steps_per_env": steps_per_env,
            "total_steps": total_steps,
            "device": str(jax.devices()[0].platform),
            "compile_time_s": compile_time,
            "runtime_mean_s": timing_results["mean"],
            "runtime_std_s": timing_results["std"],
            "throughput_steps_per_s": throughput,
            "total_time_s": compile_time + timing_results["mean"],
            "success": True,
        }

        print(f"    Compilation: {compile_time:.4f} s")
        print(f"    Runtime:     {timing_results['mean']:.4f} ± {timing_results['std']:.4f} s")
        print(f"    Throughput:  {format_number(throughput)} steps/s")

        return results

    except Exception as e:
        print(f"    ERROR: {str(e)}")
        return {
            "library": "gymnax",
            "error": str(e),
            "success": False,
        }


def benchmark_gymnasium(total_steps: int) -> dict:
    """Benchmark Gymnasium environment throughput (serial, CPU-only).

    Args:
        total_steps: Total number of steps to run

    Returns:
        Benchmark results dictionary
    """
    print("\n>>> Benchmarking Gymnasium (serial CPU baseline)")
    print(f"    {total_steps} steps (serial)")

    try:
        import gymnasium as gym
    except ImportError:
        print("    ERROR: gymnasium not installed. Install with: pip install gymnasium")
        return {
            "library": "gymnasium",
            "error": "gymnasium not installed",
            "success": False,
        }

    try:
        # Create environment
        env = gym.make("CartPole-v1")

        # Warmup
        print("    Warming up...")
        for _ in range(10):
            env.reset()
            for _ in range(100):
                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    env.reset()

        # Timing runs
        print("    Timing execution...")
        times = []
        for run in range(get_comparison_config().num_timing_runs):
            obs, _ = env.reset()
            start = time.perf_counter()
            steps_done = 0

            while steps_done < total_steps:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                steps_done += 1

                if terminated or truncated:
                    obs, _ = env.reset()

            elapsed = time.perf_counter() - start
            times.append(elapsed)

        env.close()

        mean_time = sum(times) / len(times)
        std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
        throughput = calculate_throughput(total_steps, mean_time)

        results = {
            "library": "gymnasium",
            "num_envs": 1,  # Serial
            "steps_per_env": total_steps,
            "total_steps": total_steps,
            "device": "cpu",
            "compile_time_s": 0.0,  # No compilation
            "runtime_mean_s": mean_time,
            "runtime_std_s": std_time,
            "throughput_steps_per_s": throughput,
            "total_time_s": mean_time,
            "success": True,
        }

        print(f"    Runtime:     {mean_time:.4f} ± {std_time:.4f} s")
        print(f"    Throughput:  {format_number(throughput)} steps/s")

        return results

    except Exception as e:
        print(f"    ERROR: {str(e)}")
        return {
            "library": "gymnasium",
            "error": str(e),
            "success": False,
        }


def save_results(results: list[dict], output_file: Path):
    """Save comparison results to CSV."""
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
    parser = argparse.ArgumentParser(description="Library comparison benchmark")
    parser.add_argument(
        "--library",
        type=str,
        default="all",
        choices=["all", "myriad", "gymnax", "gymnasium"],
        help="Which library to benchmark (default: all)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=get_comparison_config().num_envs,
        help=f"Number of parallel environments for Myriad/Gymnax (default: {get_comparison_config().num_envs})",
    )
    parser.add_argument(
        "--steps-per-env",
        type=int,
        default=get_comparison_config().num_steps,
        help=f"Steps per environment (default: {get_comparison_config().num_steps})",
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
    print("Myriad Library Comparison Benchmark")
    print("=" * 70)
    print(f"Device: {device_info['platform']}")
    print(f"Configuration: {args.num_envs} envs × {args.steps_per_env} steps")
    print("=" * 70)

    # Run benchmarks
    results = []

    if args.library in ["all", "myriad"]:
        result = benchmark_myriad(args.num_envs, args.steps_per_env)
        results.append(result)

    if args.library in ["all", "gymnax"]:
        result = benchmark_gymnax(args.num_envs, args.steps_per_env)
        results.append(result)

    if args.library in ["all", "gymnasium"]:
        # Gymnasium is serial, so use total steps
        total_steps = args.num_envs * args.steps_per_env
        result = benchmark_gymnasium(total_steps)
        results.append(result)

    # Save results
    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        device_name = device_info["platform"]
        output_file = Path(f"benchmarks/results/comparison_{device_name}_{timestamp}.csv")

    save_results(results, output_file)

    # Print comparison summary
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)

    successful = [r for r in results if r.get("success", True)]
    if successful:
        # Sort by throughput
        sorted_results = sorted(successful, key=lambda x: x.get("throughput_steps_per_s", 0), reverse=True)

        print("\nThroughput Ranking:")
        for i, r in enumerate(sorted_results, 1):
            lib = r["library"]
            throughput = r.get("throughput_steps_per_s", 0)
            runtime = r.get("runtime_mean_s", 0)
            print(f"  {i}. {lib:12s}: {format_number(throughput):>8s} steps/s ({runtime:.4f} s)")

        # Speedup comparison
        if len(sorted_results) > 1:
            print("\nSpeedup vs Baseline:")
            baseline = sorted_results[-1]  # Slowest
            baseline_throughput = baseline.get("throughput_steps_per_s", 1)

            for r in sorted_results[:-1]:  # All except baseline
                lib = r["library"]
                throughput = r.get("throughput_steps_per_s", 0)
                speedup = throughput / baseline_throughput
                print(f"  {lib:12s}: {speedup:.2f}x faster than {baseline['library']}")

    print("=" * 70)


if __name__ == "__main__":
    main()
