"""Command-line interface for auto-tuning Myriad configurations.

This script provides a user-friendly interface to the auto-tuning system,
allowing users to find optimal num_envs and scan_chunk_size for their hardware.

Usage
-----
Find maximum configuration for an environment:
    python benchmarks/autotune.py --env cartpole-control --agent dqn

Find configuration for specific target:
    python benchmarks/autotune.py --env cartpole-control --agent dqn --target-envs 1000000

Force re-profiling (ignore cache):
    python benchmarks/autotune.py --env cartpole-control --agent dqn --force

View cached profiles:
    python benchmarks/autotune.py --show-cache
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from myriad.platform.autotune import get_cache_path, load_cache, suggest_config


# Set up logging
def setup_logging(quiet: bool = False):
    """Configure logging for the CLI."""
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",  # Simple format, just the message
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def show_cache():
    """Display cached profiles."""
    cache = load_cache()
    cache_path = get_cache_path()

    print("\n" + "=" * 70)
    print("Cached Autotune Profiles")
    print("=" * 70)
    print(f"Cache location: {cache_path}")
    print()

    # Hardware
    if cache["hardware"]:
        print("Hardware Profiles:")
        for hw_id, hw_info in cache["hardware"].items():
            print(f"  • {hw_id}")
            print(f"    Platform: {hw_info.get('platform', 'unknown')}")
            print(f"    Memory: {hw_info.get('available_memory_gb', 0):.1f} GB")
            print(f"    Profiled: {hw_info.get('profiled_at', 'unknown')}")
    else:
        print("Hardware Profiles: None")

    print()

    # Environments
    if cache["env_profiles"]:
        print("Environment Profiles:")
        for env_name, env_info in cache["env_profiles"].items():
            print(f"  • {env_name}")
            print(f"    Memory: {env_info.get('memory_mb_per_env', 0):.3f} MB/env")
            print(f"    Profiled: {env_info.get('profiled_at', 'unknown')}")
    else:
        print("Environment Profiles: None")

    print()

    # Agents
    if cache["agent_profiles"]:
        print("Agent Profiles:")
        for agent_name, agent_info in cache["agent_profiles"].items():
            print(f"  • {agent_name}")
            print(f"    Overhead: {agent_info.get('overhead_mb', 0):.1f} MB")
            print(f"    Method: {agent_info.get('method', 'unknown')}")
    else:
        print("Agent Profiles: None")

    print()

    # Validated configs
    if cache["validated_configs"]:
        print(f"Validated Configurations: {len(cache['validated_configs'])}")
        for config_key, config_info in cache["validated_configs"].items():
            print(f"  • {config_key}")
            print(f"    Max envs: {config_info.get('max_envs', 0):,}")
            print(f"    Chunk size: {config_info.get('optimal_chunk_size', 0)}")
            print(f"    Throughput: {config_info.get('throughput_steps_per_s', 0)/1e6:.0f}M steps/s")
            print(f"    Validated: {config_info.get('validated_at', 'unknown')}")
    else:
        print("Validated Configurations: None")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-tune Myriad configuration for your hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find maximum configuration
  python benchmarks/autotune.py --env cartpole-control --agent dqn

  # Optimize for specific target
  python benchmarks/autotune.py --env cartpole-control --agent dqn --target-envs 1000000

  # Force re-profiling
  python benchmarks/autotune.py --env cartpole-control --agent dqn --force

  # View cache
  python benchmarks/autotune.py --show-cache
        """,
    )

    parser.add_argument(
        "--env",
        type=str,
        help="Environment name (e.g., 'cartpole-control', 'ccas-ccar-control')",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="random",
        help="Agent name (default: random)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        help="Replay buffer size (for off-policy agents like DQN)",
    )
    parser.add_argument(
        "--target-envs",
        type=int,
        help="Target number of environments (if not specified, finds maximum)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-profiling even if cached",
    )
    parser.add_argument(
        "--show-cache",
        action="store_true",
        help="Show cached profiles and exit",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (only final configuration)",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(quiet=args.quiet)

    # Show cache and exit
    if args.show_cache:
        show_cache()
        return

    # Validate required arguments
    if not args.env:
        parser.error("--env is required (unless using --show-cache)")

    # Run auto-tuning
    try:
        result = suggest_config(
            env=args.env,
            agent=args.agent,
            buffer_size=args.buffer_size,
            force_revalidate=args.force,
            verbose=not args.quiet,
        )

        if args.quiet:
            # Just print the configuration
            print(f"{result.max_envs},{result.optimal_chunk_size}")
        else:
            # Full output already printed by suggest_config
            print("\nTo use this configuration:")
            print("  config = create_config(")
            print(f'      env="{args.env}",')
            print(f'      agent="{args.agent}",')
            print(f"      num_envs={result.max_envs},")
            print(f"      scan_chunk_size={result.optimal_chunk_size},")
            if args.buffer_size:
                print(f"      buffer_size={args.buffer_size},")
            print("  )")

    except Exception as e:
        print(f"\nError during auto-tuning: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
