# Myriad Performance Benchmarks

Validation benchmarks for performance claims and regression testing.

> **For users**: See [Auto-Tuning Guide](../docs/02_user-guide/05_auto_tuning.md) and [Benchmarking Guide](../docs/02_user-guide/06_benchmarking.md).

## Quick Start

```bash
# Run all benchmarks and generate plots
python benchmarks/run_all.py

# Or run quick validation only
python benchmarks/run_all.py --quick

# Or validate without generating plots
bash benchmarks/run_quick_test.sh
```

Results → `benchmarks/results/`. Plots → `benchmarks/plots/`.

### Individual Scripts

```bash
python benchmarks/throughput.py --full
python benchmarks/memory_profile.py --profile envs
python benchmarks/comparison.py
python benchmarks/plot_results.py
```

**Generated plots:**
- `throughput_scaling.png` - Throughput vs num_envs
- `memory_scaling.png` - Memory usage vs num_envs (GPU or CPU)
- `library_comparison.png` - Myriad vs Gymnax vs Gymnasium

## Scripts

| Script | Purpose | Key Options |
|--------|---------|-------------|
| `run_all.py` | Run all benchmarks + plots | `--quick`, `--skip-plots` |
| `throughput.py` | Steps/sec vs num_envs | `--full`, `--num-envs N`, `--env NAME` |
| `memory_profile.py` | Memory usage profiling | `--profile envs\|scan\|both`, `--env NAME` |
| `comparison.py` | Myriad vs Gymnax vs Gymnasium | `--library NAME` |
| `plot_results.py` | Generate 3 plots | `--output-dir PATH` |

Force device: `JAX_PLATFORM_NAME=cpu` or `JAX_PLATFORM_NAME=gpu`.

## Configuration

Edit `benchmarks/config.yaml` to adjust test matrix. Structure:

```yaml
cpu:
  throughput:
    cartpole:
      - { num_envs: 100, scan_chunk_size: 256, num_steps: 1000 }
      - { num_envs: 10_000, scan_chunk_size: 256, num_steps: 500 }
    gene-circuit:
      - { num_envs: 100, scan_chunk_size: 256, num_steps: 500 }
      # Fewer/smaller configs for slower environments
gpu:
  throughput:
    cartpole:
      # More configs, higher scales for GPU
```

## `scan_chunk_size` Guide

| num_envs | Recommended |
|----------|-------------|
| < 10K    | 256         |
| 100K     | 64-128      |
| 1M       | 16-32       |

**Higher** = better throughput, more memory. **Lower** = less memory, larger scales.

## Troubleshooting

**OOM**: Reduce `--scan-chunk-size` or `--num-envs`.

**Slow**: First run is JIT compilation. Reduce `--scan-chunk-size` for faster compilation.

**GPU not found**: `python -c "import jax; print(jax.devices())"`.
