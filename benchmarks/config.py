"""Configuration for Myriad benchmarks.

This module defines the test matrix for performance benchmarking.
"""

from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    num_envs: int
    scan_chunk_size: int
    num_steps: int = 1000  # Steps to run for timing
    warmup_steps: int = 10  # Warmup iterations
    num_timing_runs: int = 5  # Statistical runs (exclude warmup)


# Test matrix for throughput scaling
# Strategy: Start with standard scan_chunk_size (256), then reduce it
# as num_envs increases to manage memory
THROUGHPUT_CONFIGS = [
    # Small scale: standard configuration
    BenchmarkConfig(num_envs=100, scan_chunk_size=256, num_steps=1000),
    BenchmarkConfig(num_envs=1_000, scan_chunk_size=256, num_steps=1000),
    BenchmarkConfig(num_envs=10_000, scan_chunk_size=256, num_steps=1000),
    # Large scale: start reducing scan_chunk_size to manage memory
    BenchmarkConfig(num_envs=100_000, scan_chunk_size=128, num_steps=1000),
    BenchmarkConfig(num_envs=500_000, scan_chunk_size=64, num_steps=500),
    BenchmarkConfig(num_envs=1_000_000, scan_chunk_size=32, num_steps=100),
    # Extreme scale: minimal scan_chunk_size (if memory allows)
    BenchmarkConfig(num_envs=2_000_000, scan_chunk_size=16, num_steps=100),
    BenchmarkConfig(num_envs=5_000_000, scan_chunk_size=8, num_steps=50),
]

# Test matrix for scan_chunk_size sensitivity
# Fix num_envs at a high value and sweep scan_chunk_size
SCAN_CHUNK_SENSITIVITY_CONFIGS = [
    BenchmarkConfig(num_envs=100_000, scan_chunk_size=8, num_steps=500),
    BenchmarkConfig(num_envs=100_000, scan_chunk_size=16, num_steps=500),
    BenchmarkConfig(num_envs=100_000, scan_chunk_size=32, num_steps=500),
    BenchmarkConfig(num_envs=100_000, scan_chunk_size=64, num_steps=500),
    BenchmarkConfig(num_envs=100_000, scan_chunk_size=128, num_steps=500),
    BenchmarkConfig(num_envs=100_000, scan_chunk_size=256, num_steps=500),
    BenchmarkConfig(num_envs=100_000, scan_chunk_size=512, num_steps=500),
]

# Comparison configurations for library benchmarking
# Use moderate scale to be fair across libraries
COMPARISON_CONFIG = BenchmarkConfig(
    num_envs=1_000,
    scan_chunk_size=256,
    num_steps=10_000,  # Run longer for statistical significance
    warmup_steps=10,
    num_timing_runs=10,
)
