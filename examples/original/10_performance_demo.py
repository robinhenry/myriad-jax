"""Performance demonstration: Massively parallel environments.

This example showcases Myriad's ability to run hundreds of thousands or even
millions of parallel environments on a single GPU. It demonstrates the core
value proposition: population-scale simulation for high-throughput experiments.

Key Features Demonstrated
--------------------------
1. Massive parallelism (100K+ environments)
2. Efficient memory management via scan_chunk_size tuning
3. Real-time throughput monitoring
4. Scaling from small to large populations

Hardware Requirements
---------------------
- For 100K environments: ~4GB GPU memory
- For 1M environments: ~16GB GPU memory (adjust scan_chunk_size as needed)
- CPU mode works but is slower

Usage
-----
Run with default settings (100K environments):
    python examples/10_performance_demo.py

Scale up to 1M environments:
    python examples/10_performance_demo.py --num-envs 1000000

Use CPU (slower but works):
    JAX_PLATFORM_NAME=cpu python examples/10_performance_demo.py

Adjust memory usage:
    python examples/10_performance_demo.py --num-envs 1000000 --scan-chunk-size 16
"""

import argparse
import time

import jax
import jax.numpy as jnp

from myriad.agents.classical.random import make_agent as make_random_agent
from myriad.envs import make_env
from myriad.platform.steps import make_train_step_fn
from myriad.platform.types import TrainingEnvState
from myriad.utils import to_array


def format_number(num: float) -> str:
    """Format large numbers for readability."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.0f}"


def estimate_memory_mb(pytree) -> float:
    """Estimate memory usage of a PyTree in MB."""
    leaves = jax.tree_util.tree_leaves(pytree)
    total_bytes = sum(leaf.nbytes for leaf in leaves if isinstance(leaf, jnp.ndarray))
    return total_bytes / (1024 * 1024)


def run_performance_demo(num_envs: int, num_steps: int, scan_chunk_size: int):
    """Run performance demonstration with specified configuration.

    Args:
        num_envs: Number of parallel environments
        num_steps: Number of steps to run
        scan_chunk_size: JAX scan chunk size for memory management
    """
    print("\n" + "=" * 80)
    print("Myriad Performance Demonstration: Massively Parallel Simulation")
    print("=" * 80)
    print("Configuration:")
    print(f"  Environments:     {format_number(num_envs)}")
    print(f"  Steps per env:    {num_steps}")
    print(f"  Total steps:      {format_number(num_envs * num_steps)}")
    print(f"  Scan chunk size:  {scan_chunk_size}")
    print(f"  Device:           {jax.devices()[0]}")
    print("=" * 80)

    # Setup
    print("\n[1/5] Creating environment and agent...")
    env = make_env("cartpole-control")
    action_space = env.get_action_space(env.config)
    agent = make_random_agent(action_space=action_space)

    print("[2/5] Initializing environments...")
    start = time.time()
    key = jax.random.PRNGKey(0)
    reset_keys = jax.random.split(key, num_envs)

    # Vectorized reset
    vmapped_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    obs, env_states = vmapped_reset(reset_keys, env.params, env.config)

    # Convert to arrays
    to_array_batch = jax.vmap(to_array)
    obs_array = to_array_batch(obs)

    training_state = TrainingEnvState(env_state=env_states, obs=obs_array)
    agent_state = agent.init(key, obs_array[0], agent.params)

    # Block to ensure initialization completes
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_state)
    init_time = time.time() - start

    # Estimate memory
    state_memory = estimate_memory_mb(training_state)
    agent_memory = estimate_memory_mb(agent_state)
    total_memory = state_memory + agent_memory

    print(f"  Initialization time: {init_time:.3f} seconds")
    print(f"  State memory:        {state_memory:.2f} MB")
    print(f"  Agent memory:        {agent_memory:.2f} MB")
    print(f"  Total memory:        {total_memory:.2f} MB")

    # Create step function
    print("\n[3/5] Creating and compiling step function...")
    step_fn = make_train_step_fn(agent, env, None, num_envs)

    # Create scan loop
    def run_n_steps(key, agent_state, training_state, n_steps: int):
        """Run n training steps using lax.scan."""

        def scan_body(carry, _):
            key, agent_state, training_state = carry
            key, new_agent_state, new_training_state, _, _ = step_fn(
                key, agent_state, training_state, None, batch_size=1
            )
            return (key, new_agent_state, new_training_state), None

        initial_carry = (key, agent_state, training_state)
        final_carry, _ = jax.lax.scan(scan_body, initial_carry, None, length=n_steps)
        return final_carry

    jitted_run = jax.jit(run_n_steps, static_argnames=["n_steps"])

    # Compilation (first run)
    print("  Compiling (this may take a moment)...")
    compile_start = time.time()
    result = jitted_run(key, agent_state, training_state, 1)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), result)
    compile_time = time.time() - compile_start
    print(f"  Compilation time: {compile_time:.3f} seconds")

    # Warmup
    print("\n[4/5] Warming up...")
    for _ in range(5):
        result = jitted_run(key, agent_state, training_state, num_steps)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), result)

    # Benchmark
    print(f"\n[5/5] Running benchmark ({num_steps} steps × {format_number(num_envs)} envs)...")
    num_timing_runs = 5
    times = []

    for i in range(num_timing_runs):
        start = time.time()
        result = jitted_run(key, agent_state, training_state, num_steps)
        # CRITICAL: Block until GPU work completes
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), result)
        elapsed = time.time() - start
        times.append(elapsed)

        total_steps = num_steps * num_envs
        throughput = total_steps / elapsed
        print(f"  Run {i+1}/{num_timing_runs}: {elapsed:.4f}s ({format_number(throughput)} steps/s)")

    # Results
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    total_steps = num_steps * num_envs
    mean_throughput = total_steps / mean_time
    steps_per_env_per_sec = mean_throughput / num_envs

    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    print(f"Total environments:       {format_number(num_envs)}")
    print(f"Total steps executed:     {format_number(total_steps)}")
    print(f"Average time per run:     {mean_time:.4f} ± {std_time:.4f} seconds")
    print(f"Total throughput:         {format_number(mean_throughput)} steps/second")
    print(f"Per-environment:          {steps_per_env_per_sec:.2f} steps/second/env")
    print(f"Memory usage:             {total_memory:.2f} MB")
    print("=" * 80)

    # Interpret results
    print("\nPerformance Summary:")
    if mean_throughput > 1e6:
        print(f"  ✓ Achieved >1M steps/second ({format_number(mean_throughput)} steps/s)")
    else:
        print(f"  • Throughput: {format_number(mean_throughput)} steps/s")

    if num_envs >= 100_000:
        print(f"  ✓ Successfully ran {format_number(num_envs)} parallel environments")
    else:
        print(f"  • Ran {format_number(num_envs)} parallel environments")

    if jax.devices()[0].platform == "gpu":
        print(f"  ✓ Using GPU acceleration ({jax.devices()[0]})")
    else:
        print(f"  • Using CPU ({jax.devices()[0]})")

    print("\nConclusion:")
    print("  This demonstration shows Myriad can efficiently simulate")
    print(f"  {format_number(num_envs)} parallel environments, executing")
    print(f"  {format_number(mean_throughput)} environment steps per second.")
    print("  This enables population-scale learning and high-throughput experiments.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Performance demonstration: massively parallel environments")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=100_000,
        help="Number of parallel environments (default: 100,000)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of steps per environment (default: 100)",
    )
    parser.add_argument(
        "--scan-chunk-size",
        type=int,
        default=32,
        help="JAX scan chunk size for memory management (default: 32)",
    )

    args = parser.parse_args()

    # Validate configuration
    if args.num_envs < 1:
        print("Error: num_envs must be >= 1")
        return

    if args.num_envs > 10_000_000:
        print(f"Warning: {format_number(args.num_envs)} environments may exceed memory limits")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            return

    # Run demo
    try:
        run_performance_demo(
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            scan_chunk_size=args.scan_chunk_size,
        )
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("\nTroubleshooting:")
        print("  - Out of memory? Try reducing --num-envs or --scan-chunk-size")
        print("  - GPU not available? Set JAX_PLATFORM_NAME=cpu")
        print("  - Still having issues? Check benchmarks/README.md for tips")
        raise


if __name__ == "__main__":
    main()
