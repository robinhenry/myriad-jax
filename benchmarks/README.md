# Myriad Performance Benchmarks

This directory contains benchmarks for validating Myriad's performance claims and tracking regressions over time.

## Overview

The benchmarks measure:
1. **Throughput scaling**: Steps/second vs number of parallel environments
2. **Memory usage**: RAM/VRAM consumption at different scales
3. **Library comparison**: Myriad vs Gymnax vs Gymnasium
4. **Scan chunk sensitivity**: Impact of `scan_chunk_size` on performance and memory

## Quick Start

### Run All Benchmarks

```bash
# Full throughput benchmark suite (CPU or GPU)
python benchmarks/throughput.py --full

# Memory profiling
python benchmarks/memory_profile.py --profile both

# Library comparison
python benchmarks/comparison.py

# Generate plots
python benchmarks/plot_results.py
```

### Quick Test

```bash
# Fast test on small scale
python benchmarks/throughput.py --num-envs 1000

# View performance demo
python examples/10_performance_demo.py
```

## Benchmark Scripts

### `throughput.py` - Throughput Scaling

Measures how environment throughput scales with number of parallel environments.

**Usage:**
```bash
# Full benchmark suite (100 to 1M+ environments)
python benchmarks/throughput.py --full

# Single configuration test
python benchmarks/throughput.py --num-envs 100000 --scan-chunk-size 128

# Specific environment
python benchmarks/throughput.py --env gene-circuit --full

# Force CPU or GPU
JAX_PLATFORM_NAME=cpu python benchmarks/throughput.py
JAX_PLATFORM_NAME=gpu python benchmarks/throughput.py
```

**Test Matrix:**
| num_envs | scan_chunk_size | Expected Throughput |
|----------|----------------|---------------------|
| 100      | 256            | ~100K steps/s       |
| 1K       | 256            | ~1M steps/s         |
| 10K      | 256            | ~10M steps/s        |
| 100K     | 128            | ~50M steps/s        |
| 1M       | 32             | >100M steps/s       |

**Output:** `benchmarks/results/throughput_<device>_<timestamp>.csv`

### `memory_profile.py` - Memory Profiling

Profiles memory usage to optimize `scan_chunk_size` and find maximum feasible `num_envs`.

**Usage:**
```bash
# Profile memory vs num_envs
python benchmarks/memory_profile.py --profile envs

# Profile memory vs scan_chunk_size
python benchmarks/memory_profile.py --profile scan

# Profile both
python benchmarks/memory_profile.py --profile both
```

**Output:** `benchmarks/results/memory_<type>_<device>_<timestamp>.csv`

### `comparison.py` - Library Comparison

Compares Myriad against Gymnax (JAX-based) and Gymnasium (CPU baseline).

**Usage:**
```bash
# Compare all libraries
python benchmarks/comparison.py

# Test single library
python benchmarks/comparison.py --library myriad
python benchmarks/comparison.py --library gymnax

# Custom configuration
python benchmarks/comparison.py --num-envs 10000 --steps-per-env 1000
```

**Requirements:**
- Gymnax comparison: `pip install gymnax`
- Gymnasium comparison: `pip install gymnasium`

**Output:** `benchmarks/results/comparison_<device>_<timestamp>.csv`

### `plot_results.py` - Visualization

Generates publication-quality plots from benchmark results.

**Usage:**
```bash
# Plot all results in results directory
python benchmarks/plot_results.py

# Plot specific file
python benchmarks/plot_results.py --input benchmarks/results/throughput_gpu_20240101.csv

# Generate specific plot type
python benchmarks/plot_results.py --plot-type throughput
python benchmarks/plot_results.py --plot-type comparison
```

**Plots Generated:**
- `throughput_scaling.png`: Throughput vs num_envs (log-log)
- `scaling_efficiency.png`: Parallel efficiency analysis
- `memory_scaling.png`: Memory usage vs num_envs
- `scan_chunk_sensitivity.png`: Optimal scan_chunk_size
- `library_comparison.png`: Bar chart comparison

**Output:** `benchmarks/results/plots/`

## Understanding `scan_chunk_size`

The `scan_chunk_size` parameter controls how many training steps are batched into a single `jax.lax.scan`. This is a **memory vs compilation trade-off**:

### High `scan_chunk_size` (e.g., 256-512)
- ✓ Fewer Python overheads
- ✓ Better throughput for moderate num_envs
- ✗ More GPU memory usage (intermediate states)
- ✗ Longer compilation time

### Low `scan_chunk_size` (e.g., 16-32)
- ✓ Lower GPU memory usage
- ✓ Enables larger num_envs
- ✗ More Python-JAX boundary crossings
- ✗ Slightly lower throughput

### Rule of Thumb
```
num_envs < 10K:     scan_chunk_size = 256 (default)
num_envs = 100K:    scan_chunk_size = 64-128
num_envs = 1M:      scan_chunk_size = 16-32
num_envs > 1M:      scan_chunk_size = 8-16
```

Run `memory_profile.py --profile scan` to find the optimal value for your hardware.

## Hardware Requirements

### CPU (Mac M-series or similar)
- **100 envs**: Works easily
- **1K envs**: Fast, no issues
- **10K envs**: Slower but functional
- **100K+ envs**: Very slow, not recommended

### GPU (NVIDIA)
- **4GB VRAM**: Up to ~50K environments
- **8GB VRAM**: Up to ~100K environments
- **16GB VRAM**: Up to ~500K environments
- **24GB+ VRAM**: 1M+ environments possible

*Actual limits depend on environment complexity and `scan_chunk_size`.*

## Interpreting Results

### Throughput Scaling

**Good scaling:**
- Linear or near-linear on log-log plot
- Efficiency > 0.8 at 100K environments
- Throughput >1M steps/s on modern GPU

**Suboptimal scaling:**
- Plateauing at high num_envs → reduce scan_chunk_size
- Low efficiency → check for compilation overhead
- GPU not faster than CPU → check device selection

### Memory Usage

**Expected pattern:**
- Linear scaling with num_envs (state size)
- Step increase at certain num_envs (scan intermediates)
- GPU memory > CPU memory (device arrays)

**If you hit OOM:**
1. Reduce `scan_chunk_size` (most effective)
2. Reduce `num_envs`
3. Use simpler environment (CartPole vs gene circuit)
4. Switch to CPU (slower but larger capacity)

### Library Comparison

**Expected results:**
- **Myriad > Gymnax**: Should be competitive (both JAX + vmap)
- **Myriad >> Gymnasium**: Large speedup due to parallelism (10-100x)
- **GPU >> CPU**: Significant advantage for large num_envs

**Fair comparison notes:**
- Gymnax: Apples-to-apples (both JAX-based)
- Gymnasium: Demonstrates GPU parallelism value, not direct comparison

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce scan_chunk_size
python benchmarks/throughput.py --num-envs 1000000 --scan-chunk-size 16

# Or reduce num_envs
python benchmarks/throughput.py --num-envs 100000
```

### Slow Compilation

First run is always slow (JIT compilation). If subsequent runs are also slow:

```bash
# Reduce scan_chunk_size to reduce compilation time
python benchmarks/throughput.py --scan-chunk-size 64
```

### GPU Not Found

```bash
# Check available devices
python -c "import jax; print(jax.devices())"

# Force CPU
JAX_PLATFORM_NAME=cpu python benchmarks/throughput.py
```

### Results Don't Match Claims

1. **Check warmup**: First run includes compilation, exclude from stats
2. **Check blocking**: Must call `.block_until_ready()` for accurate timing
3. **Check device**: CPU vs GPU makes huge difference
4. **Check environment**: Gene circuit is slower than CartPole

## Adding New Benchmarks

To add a new benchmark:

1. Create `benchmarks/your_benchmark.py`
2. Use utilities from `benchmarks/utils.py`
3. Follow patterns from existing benchmarks:
   - Proper warmup with `warmup_jitted_fn()`
   - Timing with `time_jitted_fn()`
   - Always use `.block_until_ready()`
4. Save results to CSV in `benchmarks/results/`
5. Add plotting support in `plot_results.py`

## Continuous Integration

For regression testing, run benchmarks in CI:

```yaml
# Example GitHub Actions workflow
- name: Run benchmarks
  run: |
    python benchmarks/throughput.py --num-envs 1000
    python benchmarks/comparison.py --library myriad
```

Compare results against previous runs to catch performance regressions.

## Citation

When reporting benchmark results, include:

1. **Hardware**: GPU/CPU model, memory
2. **Configuration**: num_envs, scan_chunk_size
3. **Environment**: CartPole vs gene circuit
4. **Myriad version**: Git commit or release tag
5. **JAX version**: `python -c "import jax; print(jax.__version__)"`

Example:
```
Throughput: 50M steps/s
Hardware: NVIDIA RTX 4090 (24GB)
Config: 100K envs, scan_chunk_size=128
Environment: CartPole-control
Myriad: v0.1.0 (commit abc123)
JAX: 0.4.23
```

## Performance Goals

From `ROADMAP.md` Phase 1.1:

- **Target**: >1M steps/second at 100K environments
- **Validation**: Published graphs in README
- **Environments**: CartPole + gene circuit
- **Platforms**: CPU (Mac) + GPU

Use these benchmarks to validate we meet these goals.
