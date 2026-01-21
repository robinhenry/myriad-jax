"""Generate plots from benchmark results.

Generates three plots (with CPU/GPU side-by-side if both available):
1. throughput_scaling.png - Throughput vs num_envs
2. memory_scaling.png - Memory usage vs num_envs
3. library_comparison.png - Myriad vs Gymnax vs Gymnasium

Usage
-----
    python benchmarks/plot_results.py
"""

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

COLORS = {
    "cartpole": "#2E86AB",
    "gene-circuit": "#A23B72",
    "myriad": "#2E86AB",
    "gymnax": "#06A77D",
    "gymnasium": "#E63946",
}
MARKERS = {"cartpole": "o", "gene-circuit": "s"}


def read_csv(file_path: Path) -> list[dict[str, Any]]:
    """Read benchmark results from CSV file."""
    results = []
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = {}
            for key, value in row.items():
                if value == "":
                    converted[key] = None
                elif value in ("True", "true"):
                    converted[key] = True
                elif value in ("False", "false"):
                    converted[key] = False
                else:
                    try:
                        converted[key] = float(value) if "." in value else int(value)
                    except (ValueError, TypeError):
                        converted[key] = value
            results.append(converted)
    return results


def is_successful(r: dict) -> bool:
    """Check if result is successful (treats None/missing as success)."""
    return r.get("success") is not False


def fmt(n: float) -> str:
    """Format number with K/M suffix."""
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.0f}K"
    return f"{n:.0f}"


def split_by_device(results: list[dict]) -> dict[str, list[dict]]:
    """Split results by device (cpu/gpu)."""
    by_device: dict[str, list] = {}
    for r in results:
        device = r.get("device", "unknown").lower()
        by_device.setdefault(device, []).append(r)
    return by_device


def plot_throughput_scaling(results: list[dict], output_dir: Path):
    """Plot throughput vs num_envs, always showing CPU and GPU side-by-side."""
    data = [r for r in results if is_successful(r) and r.get("total_throughput_steps_per_s")]
    by_device = split_by_device(data)

    # Always show both CPU and GPU panels
    devices = ["cpu", "gpu"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    axes = axes[0]

    for idx, device in enumerate(devices):
        ax = axes[idx]
        device_data = by_device.get(device, [])

        if not device_data:
            # Empty panel - show "No data" message
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14, color="gray", transform=ax.transAxes)
            ax.set_xlabel("Number of Parallel Environments", fontsize=11)
            ax.set_ylabel("Throughput (steps/second)", fontsize=11)
            ax.set_title(device.upper(), fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)
            continue

        # Group by environment
        by_env: dict[str, list] = {}
        for r in device_data:
            by_env.setdefault(r.get("env_name", "unknown"), []).append(r)

        for env_name, env_data in by_env.items():
            # Deduplicate by num_envs (take best)
            seen = {}
            for r in env_data:
                key = r["num_envs"]
                if key not in seen or r["total_throughput_steps_per_s"] > seen[key]["total_throughput_steps_per_s"]:
                    seen[key] = r
            env_data = sorted(seen.values(), key=lambda x: x["num_envs"])

            num_envs = [r["num_envs"] for r in env_data]
            throughput = [r["total_throughput_steps_per_s"] for r in env_data]

            color = COLORS.get(env_name, "#666666")
            marker = MARKERS.get(env_name, "o")
            label = env_name.replace("-", " ").title()

            ax.loglog(num_envs, throughput, marker=marker, linewidth=2.5, markersize=9, label=label, color=color)

            # Annotate points
            for x, y in zip(num_envs, throughput):
                ax.annotate(fmt(y), (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

        # Perfect scaling reference
        if by_env:
            first = sorted(list(by_env.values())[0], key=lambda x: x["num_envs"])
            if len(first) >= 2:
                base_envs, base_tp = first[0]["num_envs"], first[0]["total_throughput_steps_per_s"]
                max_envs = max(r["num_envs"] for env_data in by_env.values() for r in env_data)
                ax.plot(
                    [base_envs, max_envs],
                    [base_tp, base_tp * max_envs / base_envs],
                    "--",
                    color="gray",
                    alpha=0.5,
                    linewidth=1.5,
                    label="Perfect Scaling",
                )

        ax.set_xlabel("Number of Parallel Environments", fontsize=11)
        ax.set_ylabel("Throughput (steps/second)", fontsize=11)
        ax.set_title(device.upper(), fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Myriad Throughput Scaling", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "throughput_scaling.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def plot_library_comparison(results: list[dict], output_dir: Path):
    """Plot vertical bar chart comparing libraries, always showing CPU and GPU side-by-side."""
    data = [r for r in results if is_successful(r) and r.get("throughput_steps_per_s")]
    by_device = split_by_device(data)

    # Always show both CPU and GPU panels
    devices = ["cpu", "gpu"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), squeeze=False)
    axes = axes[0]

    for idx, device in enumerate(devices):
        ax = axes[idx]
        device_data = by_device.get(device, [])

        if not device_data:
            # Empty panel
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14, color="gray", transform=ax.transAxes)
            ax.set_ylabel("Throughput (steps/second)", fontsize=11)
            ax.set_title(device.upper(), fontsize=13, fontweight="bold")
            ax.grid(True, axis="y", alpha=0.3)
            continue

        num_envs = device_data[0].get("num_envs", "?")

        # Deduplicate by library (take best)
        by_lib = {}
        for r in device_data:
            lib = r["library"]
            if lib not in by_lib or r["throughput_steps_per_s"] > by_lib[lib]["throughput_steps_per_s"]:
                by_lib[lib] = r

        # Sort by throughput ascending (lowest first, highest last)
        sorted_data = sorted(by_lib.values(), key=lambda x: x["throughput_steps_per_s"])

        libraries = [r["library"].capitalize() for r in sorted_data]
        throughput = [r["throughput_steps_per_s"] for r in sorted_data]
        colors = [COLORS.get(r["library"].lower(), "#666666") for r in sorted_data]

        bars = ax.bar(libraries, throughput, color=colors, alpha=0.85, edgecolor="black", width=0.6)

        # Value labels on top
        for bar, tp in zip(bars, throughput):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                fmt(tp),
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        # Speedup labels inside bars
        if len(throughput) > 1:
            baseline = min(throughput)
            for bar, tp in zip(bars, throughput):
                if tp > baseline * 1.5:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 0.5,
                        f"{tp/baseline:.0f}×",
                        ha="center",
                        va="center",
                        color="white",
                        fontweight="bold",
                        fontsize=12,
                    )

        ax.set_ylabel("Throughput (steps/second)", fontsize=11)
        ax.set_title(f"{device.upper()} ({fmt(num_envs)} envs)", fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        if max(throughput) / max(min(throughput), 1) > 50:
            ax.set_yscale("log")

    fig.suptitle("Library Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "library_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def plot_memory_scaling(results: list[dict], output_dir: Path):
    """Plot memory usage vs num_envs, always showing CPU and GPU side-by-side."""
    data = [
        r
        for r in results
        if is_successful(r) and (r.get("cpu_delta_mb") is not None or r.get("gpu_allocated_mb") is not None)
    ]
    by_device = split_by_device(data)

    # Always show both CPU and GPU panels
    devices = ["cpu", "gpu"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    axes = axes[0]

    for idx, device in enumerate(devices):
        ax = axes[idx]
        device_data = by_device.get(device, [])
        is_gpu = device == "gpu"

        ylabel = "GPU Memory (MB)" if is_gpu else "CPU Memory Delta (MB)"

        if not device_data:
            # Empty panel
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14, color="gray", transform=ax.transAxes)
            ax.set_xlabel("Number of Parallel Environments", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(device.upper(), fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)
            continue

        # Group by environment
        by_env: dict[str, list] = {}
        for r in device_data:
            by_env.setdefault(r.get("env_name", "unknown"), []).append(r)

        for env_name, env_data in by_env.items():
            # Deduplicate by num_envs
            seen = {}
            for r in env_data:
                seen[r["num_envs"]] = r
            env_data = sorted(seen.values(), key=lambda x: x["num_envs"])

            num_envs = [r["num_envs"] for r in env_data]

            if is_gpu:
                memory = [r.get("gpu_allocated_mb") or r.get("gpu_delta_mb") or 0 for r in env_data]
            else:
                memory = [r.get("cpu_delta_mb") or 0 for r in env_data]

            # Filter zero values
            valid = [(n, m) for n, m in zip(num_envs, memory) if m > 0]
            if not valid:
                continue
            num_envs, memory = zip(*valid)

            color = COLORS.get(env_name, "#666666")
            marker = MARKERS.get(env_name, "o")
            label = env_name.replace("-", " ").title()

            ax.loglog(num_envs, memory, marker=marker, linewidth=2.5, markersize=9, label=label, color=color)

            # Annotate
            for x, y in zip(num_envs, memory):
                ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

        ax.set_xlabel("Number of Parallel Environments", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(device.upper(), fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Memory Usage", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "memory_scaling.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark plots")
    parser.add_argument("--output-dir", type=str, default="benchmarks/plots", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("benchmarks/results")
    csv_files = sorted(results_dir.glob("*.csv"))

    if not csv_files:
        print("No CSV files found in benchmarks/results/. Run benchmarks first.")
        return

    print(f"\nReading {len(csv_files)} result file(s)...")

    throughput_data: list[dict] = []
    memory_data: list[dict] = []
    comparison_data: list[dict] = []

    for f in csv_files:
        data = read_csv(f)
        name = f.stem.lower()
        if "throughput" in name:
            throughput_data.extend(data)
        elif "memory" in name:
            memory_data.extend(data)
        elif "comparison" in name:
            comparison_data.extend(data)

    print(f"\nGenerating plots to {output_dir}/")
    print("-" * 40)

    if throughput_data:
        plot_throughput_scaling(throughput_data, output_dir)

    if memory_data:
        plot_memory_scaling(memory_data, output_dir)

    if comparison_data:
        plot_library_comparison(comparison_data, output_dir)

    print("-" * 40)
    print("Done!")


if __name__ == "__main__":
    main()
