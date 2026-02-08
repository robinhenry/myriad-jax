#!/usr/bin/env python
"""Run all benchmarks and generate plots.

Usage
-----
    python benchmarks/run_all.py           # Run full suite
    python benchmarks/run_all.py --quick   # Quick validation only
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f">>> {description}")
    print(f"    {' '.join(cmd)}")
    print("=" * 70)

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all Myriad benchmarks")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick validation only (small scale)",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args()

    benchmarks_dir = Path(__file__).parent
    python = sys.executable

    results = []

    if args.quick:
        # Quick validation
        results.append(
            run_command(
                [python, str(benchmarks_dir / "throughput.py"), "--num-envs", "100", "--num-steps", "10"],
                "Quick throughput test",
            )
        )
        results.append(
            run_command(
                [python, str(benchmarks_dir / "memory_profile.py"), "--profile", "envs", "--env", "cartpole"],
                "Quick memory profile",
            )
        )
        results.append(
            run_command(
                [
                    python,
                    str(benchmarks_dir / "comparison.py"),
                    "--library",
                    "myriad",
                    "--num-envs",
                    "100",
                    "--steps-per-env",
                    "100",
                ],
                "Quick comparison (Myriad only)",
            )
        )
    else:
        # Full benchmark suite
        results.append(
            run_command(
                [python, str(benchmarks_dir / "throughput.py"), "--full"],
                "Throughput scaling benchmark",
            )
        )
        results.append(
            run_command(
                [python, str(benchmarks_dir / "memory_profile.py"), "--profile", "envs"],
                "Memory profiling",
            )
        )
        results.append(
            run_command(
                [python, str(benchmarks_dir / "comparison.py")],
                "Library comparison (Myriad vs Gymnax vs Gymnasium)",
            )
        )

    # Generate plots
    if not args.skip_plots:
        results.append(
            run_command(
                [python, str(benchmarks_dir / "plot_results.py")],
                "Generate plots",
            )
        )

    # Summary
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if all(results):
        print("\nAll benchmarks completed successfully!")
        print("\nResults: benchmarks/results/")
        print("Plots:   benchmarks/plots/")
    else:
        print("\nSome benchmarks failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
