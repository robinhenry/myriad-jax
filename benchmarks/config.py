"""Configuration for Myriad benchmarks.

Loads device and environment-specific configs from config.yaml.
Structure: device (cpu/gpu) -> benchmark type -> environment -> configs
"""

from dataclasses import dataclass
from pathlib import Path

import jax
import yaml


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    num_envs: int
    scan_chunk_size: int
    num_steps: int = 1000
    warmup_steps: int = 10
    num_timing_runs: int = 5


def get_device() -> str:
    """Detect current JAX device platform."""
    return jax.devices()[0].platform


def _load_yaml() -> dict:
    """Load raw YAML config."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_config(device: str | None = None) -> dict:
    """Load config for specified device.

    Args:
        device: "cpu" or "gpu". Auto-detected if None.

    Returns:
        Merged config dict with base settings and device-specific benchmarks.
    """
    cfg = _load_yaml()

    device = device or get_device()
    if device not in ("cpu", "gpu"):
        device = "cpu"

    base = cfg.get("base", {})
    device_cfg = cfg.get(device, {})

    return {**base, **device_cfg}


def get_envs(device: str | None = None) -> list[str]:
    """Get list of available environments for benchmarking."""
    cfg = load_config(device)
    # Get envs from throughput config keys
    throughput = cfg.get("throughput", {})
    return list(throughput.keys())


def get_throughput_configs(
    device: str | None = None, env: str | None = None
) -> list[BenchmarkConfig] | dict[str, list[BenchmarkConfig]]:
    """Get throughput benchmark configs.

    Args:
        device: "cpu" or "gpu". Auto-detected if None.
        env: Environment name. If None, returns dict of all envs.

    Returns:
        List of configs for specified env, or dict mapping env -> configs.
    """
    cfg = load_config(device)
    throughput = cfg.get("throughput", {})

    def make_configs(env_configs: list) -> list[BenchmarkConfig]:
        return [
            BenchmarkConfig(
                num_envs=c["num_envs"],
                scan_chunk_size=c["scan_chunk_size"],
                num_steps=c.get("num_steps", 1000),
                warmup_steps=cfg.get("warmup_steps", 10),
                num_timing_runs=cfg.get("num_timing_runs", 5),
            )
            for c in env_configs
        ]

    if env:
        return make_configs(throughput.get(env, []))

    return {env_name: make_configs(configs) for env_name, configs in throughput.items()}


def get_scan_sensitivity_configs(
    device: str | None = None, env: str | None = None
) -> list[BenchmarkConfig] | dict[str, list[BenchmarkConfig]]:
    """Get scan chunk sensitivity configs.

    Args:
        device: "cpu" or "gpu". Auto-detected if None.
        env: Environment name. If None, returns dict of all envs.

    Returns:
        List of configs for specified env, or dict mapping env -> configs.
    """
    cfg = load_config(device)
    scan_sensitivity = cfg.get("scan_sensitivity", {})

    def make_configs(env_configs: list) -> list[BenchmarkConfig]:
        return [
            BenchmarkConfig(
                num_envs=c["num_envs"],
                scan_chunk_size=c["scan_chunk_size"],
                num_steps=c.get("num_steps", 500),
                warmup_steps=cfg.get("warmup_steps", 10),
                num_timing_runs=cfg.get("num_timing_runs", 5),
            )
            for c in env_configs
        ]

    if env:
        return make_configs(scan_sensitivity.get(env, []))

    return {env_name: make_configs(configs) for env_name, configs in scan_sensitivity.items()}


def get_comparison_config(device: str | None = None, env: str = "cartpole") -> BenchmarkConfig:
    """Get library comparison config.

    Args:
        device: "cpu" or "gpu". Auto-detected if None.
        env: Environment name.

    Returns:
        BenchmarkConfig for the specified environment.
    """
    cfg = load_config(device)
    comparison = cfg.get("comparison", {})
    comp = comparison.get(env, {"num_envs": 1000, "scan_chunk_size": 256, "num_steps": 10000})

    return BenchmarkConfig(
        num_envs=comp.get("num_envs", 1000),
        scan_chunk_size=comp.get("scan_chunk_size", 256),
        num_steps=comp.get("num_steps", 10000),
        warmup_steps=cfg.get("warmup_steps", 10),
        num_timing_runs=cfg.get("num_timing_runs", 10),
    )
