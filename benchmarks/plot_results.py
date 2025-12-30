"""Generate plots from benchmark results.

This script reads CSV results from benchmarks and generates publication-quality
plots for documentation and README.

Plots Generated
---------------
1. Throughput vs num_envs (log-log scale)
2. Scaling efficiency vs num_envs
3. Memory usage vs num_envs
4. scan_chunk_size sensitivity
5. Library comparison bar charts

Usage
-----
Plot all results in results directory:
    python benchmarks/plot_results.py

Plot specific result file:
    python benchmarks/plot_results.py --input benchmarks/results/throughput_gpu_20240101.csv

Generate specific plot type:
    python benchmarks/plot_results.py --plot-type throughput
"""

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# Use a clean style
plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")


def read_csv(file_path: Path) -> list[dict[str, Any]]:
    """Read benchmark results from CSV file.

    Args:
        file_path: Path to CSV file

    Returns:
        List of result dictionaries
    """
    results = []
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            converted_row = {}
            for key, value in row.items():
                # Try to convert to number
                try:
                    if "." in value:
                        converted_row[key] = float(value)
                    else:
                        converted_row[key] = int(value)
                except (ValueError, AttributeError):
                    converted_row[key] = value
            results.append(converted_row)
    return results


def plot_throughput_scaling(results: list[dict], output_dir: Path):
    """Plot throughput vs num_envs on log-log scale.

    Args:
        results: Benchmark results
        output_dir: Directory to save plot
    """
    # Filter successful results
    successful = [r for r in results if r.get("success", True)]
    if not successful:
        print("No successful results to plot")
        return

    # Group by environment
    by_env = {}
    for r in successful:
        env_name = r.get("env_name", "unknown")
        if env_name not in by_env:
            by_env[env_name] = []
        by_env[env_name].append(r)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"cartpole": "#2E86AB", "gene-circuit": "#A23B72"}
    markers = {"cartpole": "o", "gene-circuit": "s"}

    for env_name, env_results in by_env.items():
        # Sort by num_envs
        env_results = sorted(env_results, key=lambda x: x.get("num_envs", 0))

        num_envs = [r["num_envs"] for r in env_results]
        throughput = [r.get("total_throughput_steps_per_s", 0) for r in env_results]

        # Plot
        color = colors.get(env_name, "blue")
        marker = markers.get(env_name, "o")
        label = env_name.replace("-", " ").title()

        ax.loglog(num_envs, throughput, marker=marker, linewidth=2, markersize=8, label=label, color=color)

    ax.set_xlabel("Number of Parallel Environments", fontsize=12, fontweight="bold")
    ax.set_ylabel("Throughput (steps/second)", fontsize=12, fontweight="bold")
    ax.set_title("Myriad Throughput Scaling", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add reference line for perfect scaling
    if by_env:
        first_env_results = list(by_env.values())[0]
        if first_env_results:
            first_result = sorted(first_env_results, key=lambda x: x.get("num_envs", 0))[0]
            baseline_envs = first_result["num_envs"]
            baseline_throughput = first_result.get("total_throughput_steps_per_s", 0)

            max_envs = max(r["num_envs"] for results in by_env.values() for r in results)
            perfect_scaling_x = [baseline_envs, max_envs]
            perfect_scaling_y = [baseline_throughput, baseline_throughput * (max_envs / baseline_envs)]
            ax.plot(
                perfect_scaling_x,
                perfect_scaling_y,
                "--",
                color="gray",
                alpha=0.5,
                label="Perfect Scaling",
                linewidth=1.5,
            )
            ax.legend(fontsize=11)

    plt.tight_layout()
    output_path = output_dir / "throughput_scaling.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_scaling_efficiency(results: list[dict], output_dir: Path):
    """Plot scaling efficiency vs num_envs.

    Args:
        results: Benchmark results
        output_dir: Directory to save plot
    """
    successful = [r for r in results if r.get("success", True)]
    if not successful:
        return

    # Group by environment
    by_env = {}
    for r in successful:
        env_name = r.get("env_name", "unknown")
        if env_name not in by_env:
            by_env[env_name] = []
        by_env[env_name].append(r)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"cartpole": "#2E86AB", "gene-circuit": "#A23B72"}
    markers = {"cartpole": "o", "gene-circuit": "s"}

    for env_name, env_results in by_env.items():
        env_results = sorted(env_results, key=lambda x: x.get("num_envs", 0))
        if len(env_results) < 2:
            continue

        # Calculate efficiency relative to smallest configuration
        baseline = env_results[0]
        _baseline_envs = baseline["num_envs"]
        baseline_per_env_throughput = baseline.get("per_env_throughput_steps_per_s", 1)

        num_envs = []
        efficiencies = []

        for r in env_results:
            current_per_env = r.get("per_env_throughput_steps_per_s", 0)
            efficiency = current_per_env / baseline_per_env_throughput if baseline_per_env_throughput > 0 else 0
            num_envs.append(r["num_envs"])
            efficiencies.append(efficiency)

        color = colors.get(env_name, "blue")
        marker = markers.get(env_name, "o")
        label = env_name.replace("-", " ").title()

        ax.semilogx(num_envs, efficiencies, marker=marker, linewidth=2, markersize=8, label=label, color=color)

    # Perfect efficiency reference line
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect Efficiency")

    ax.set_xlabel("Number of Parallel Environments", fontsize=12, fontweight="bold")
    ax.set_ylabel("Scaling Efficiency", fontsize=12, fontweight="bold")
    ax.set_title("Parallel Scaling Efficiency", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.2)

    plt.tight_layout()
    output_path = output_dir / "scaling_efficiency.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_memory_usage(results: list[dict], output_dir: Path):
    """Plot memory usage vs num_envs.

    Args:
        results: Memory profiling results
        output_dir: Directory to save plot
    """
    successful = [r for r in results if r.get("success", True)]
    if not successful:
        return

    # Sort by num_envs
    successful = sorted(successful, key=lambda x: x.get("num_envs", 0))

    num_envs = [r["num_envs"] for r in successful]

    # Determine which memory metric to plot (GPU or CPU)
    has_gpu = "gpu_delta_mb" in successful[0]
    if has_gpu:
        memory = [r.get("gpu_delta_mb", 0) for r in successful]
        memory_label = "GPU Memory (MB)"
    else:
        memory = [r.get("cpu_delta_mb", 0) for r in successful]
        memory_label = "CPU Memory (MB)"

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(num_envs, memory, marker="o", linewidth=2, markersize=8, color="#E63946", label="Measured Memory")

    ax.set_xlabel("Number of Parallel Environments", fontsize=12, fontweight="bold")
    ax.set_ylabel(memory_label, fontsize=12, fontweight="bold")
    ax.set_title("Memory Usage Scaling", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "memory_scaling.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_scan_chunk_sensitivity(results: list[dict], output_dir: Path):
    """Plot throughput vs scan_chunk_size.

    Args:
        results: Benchmark results with varying scan_chunk_size
        output_dir: Directory to save plot
    """
    successful = [r for r in results if r.get("success", True)]
    if not successful:
        return

    # Filter to single num_envs (should be same for all in sensitivity test)
    target_num_envs = successful[0].get("num_envs")
    filtered = [r for r in successful if r.get("num_envs") == target_num_envs]

    if not filtered:
        return

    # Sort by scan_chunk_size
    filtered = sorted(filtered, key=lambda x: x.get("scan_chunk_size", 0))

    scan_sizes = [r["scan_chunk_size"] for r in filtered]
    throughput = [r.get("total_throughput_steps_per_s", 0) for r in filtered]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogx(
        scan_sizes,
        throughput,
        marker="o",
        linewidth=2,
        markersize=8,
        color="#06A77D",
        label=f"{target_num_envs:,} environments",
    )

    ax.set_xlabel("Scan Chunk Size", fontsize=12, fontweight="bold")
    ax.set_ylabel("Throughput (steps/second)", fontsize=12, fontweight="bold")
    ax.set_title("scan_chunk_size Sensitivity Analysis", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "scan_chunk_sensitivity.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_library_comparison(results: list[dict], output_dir: Path):
    """Plot library comparison bar chart.

    Args:
        results: Library comparison results
        output_dir: Directory to save plot
    """
    successful = [r for r in results if r.get("success", True)]
    if not successful:
        return

    # Sort by throughput
    successful = sorted(successful, key=lambda x: x.get("throughput_steps_per_s", 0))

    libraries = [r["library"].capitalize() for r in successful]
    throughput = [r.get("throughput_steps_per_s", 0) for r in successful]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"Myriad": "#2E86AB", "Gymnax": "#06A77D", "Gymnasium": "#E63946"}
    bar_colors = [colors.get(lib, "gray") for lib in libraries]

    bars = ax.bar(libraries, throughput, color=bar_colors, alpha=0.8, edgecolor="black")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 1e6:
            label = f"{height/1e6:.2f}M"
        elif height > 1e3:
            label = f"{height/1e3:.2f}K"
        else:
            label = f"{height:.0f}"
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, label, ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Throughput (steps/second)", fontsize=12, fontweight="bold")
    ax.set_title("Library Performance Comparison", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Log scale if range is large
    if max(throughput) / min(throughput) > 100:
        ax.set_yscale("log")

    plt.tight_layout()
    output_path = output_dir / "library_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots from benchmark results")
    parser.add_argument(
        "--input",
        type=str,
        help="Input CSV file (default: auto-detect from benchmarks/results/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results/plots",
        help="Output directory for plots (default: benchmarks/results/plots)",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["all", "throughput", "efficiency", "memory", "scan", "comparison"],
        default="all",
        help="Type of plot to generate (default: all)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find input files
    if args.input:
        input_files = [Path(args.input)]
    else:
        results_dir = Path("benchmarks/results")
        if not results_dir.exists():
            print(f"Results directory not found: {results_dir}")
            print("Run benchmarks first or specify --input")
            return

        input_files = list(results_dir.glob("*.csv"))
        if not input_files:
            print(f"No CSV files found in {results_dir}")
            return

    print(f"\nGenerating plots from {len(input_files)} result file(s)")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    # Process each result file
    for input_file in input_files:
        print(f"\nProcessing: {input_file.name}")
        results = read_csv(input_file)

        if not results:
            print(f"  No results found in {input_file.name}")
            continue

        # Determine plot type based on filename or --plot-type
        filename = input_file.stem

        if args.plot_type in ["all", "throughput"] and "throughput" in filename:
            plot_throughput_scaling(results, output_dir)
            plot_scaling_efficiency(results, output_dir)

        if args.plot_type in ["all", "memory"] and "memory" in filename:
            plot_memory_usage(results, output_dir)

        if args.plot_type in ["all", "scan"] and ("scan" in filename or "memory" in filename):
            plot_scan_chunk_sensitivity(results, output_dir)

        if args.plot_type in ["all", "comparison"] and "comparison" in filename:
            plot_library_comparison(results, output_dir)

    print("\n" + "=" * 70)
    print(f"Plots saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
