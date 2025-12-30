# Auto-Tuning for Performance Optimization

Myriad's auto-tuning system automatically finds the optimal `scan_chunk_size` for your hardware, eliminating manual trial-and-error tuning.

## Why Auto-Tune?

**Problem**: The optimal `scan_chunk_size` depends on:
- Your hardware (GPU/CPU, memory)
- Your environment (state size, dynamics complexity)
- Your agent (network size, buffer size)
- Number of parallel environments

**Manual approach** is tedious:
1. Guess `scan_chunk_size` → OOM or slow performance
2. Try different values → tedious trial-and-error
3. No guarantee you found the optimal value

**Auto-tune** solves this:
- Finds optimal `scan_chunk_size` in 30-60 seconds (first run)
- Uses cached results (<1 second) for subsequent runs
- Guarantees optimal configuration for your hardware

## Quick Start

### Using create_config (Recommended)

```python
from myriad import create_config, train_and_evaluate

# Automatically find optimal scan_chunk_size
config = create_config(
    env="cartpole-control",
    agent="dqn",
    num_envs=100_000,
    auto_tune=True,  # ← Automatically finds optimal scan_chunk_size
)

results = train_and_evaluate(config)
```

**First run** (~30-60s):
- Profiles your hardware
- Tests environment
- Finds optimal scan_chunk_size for 100K environments
- Caches results

**Subsequent runs** (<1s):
- Uses cached optimal scan_chunk_size
- No profiling overhead

### Manual API

```python
from myriad.platform.autotune import suggest_scan_chunk_size

# Find optimal scan_chunk_size for specific configuration
chunk_size = suggest_scan_chunk_size(
    num_envs=100_000,
    env="cartpole-control",
    agent="dqn",
)

# Use it in your config
config = create_config(
    env="cartpole-control",
    agent="dqn",
    num_envs=100_000,
    scan_chunk_size=chunk_size,
)
```

## How It Works

### Two-Stage Process

**Stage 1: Component Profiling (Lazy)**

Profiles only what you use, only once:

```
First time with cartpole + dqn @ 100K envs:
  ⚡ Hardware... profiling (5s)
  ⚡ Environment (cartpole-control)... profiling (10s)

Second time with cartpole + dqn @ 100K envs:
  ✓ Using cached scan_chunk_size (instant)

Third time with cartpole + dqn @ 200K envs:
  ✓ Hardware... cached
  ✓ Environment... cached
  Testing different chunk sizes... (~15s)
```

**Stage 2: Chunk Size Optimization**

Tests different chunk sizes to find the best one:

```
Testing chunk_size=512... ✗ (OOM)
Testing chunk_size=256... ✓ (350M steps/s)
Testing chunk_size=128... ✓ (380M steps/s)
Testing chunk_size=64...  ✓ (420M steps/s) ← Best!
Testing chunk_size=32...  ✓ (410M steps/s)

✅ Optimal scan_chunk_size: 64
```

### Caching Strategy

**Component-Based Caching**:

```json
{
  "hardware": {
    "RTX4090_24GB": {...}
  },
  "env_profiles": {
    "cartpole-control": {...}
  },
  "agent_profiles": {
    "dqn": {...}
  },
  "chunk_size_configs": {
    "cartpole:dqn:100000:RTX4090": {
      "optimal_chunk_size": 64,
      "throughput_steps_per_s": 420000000,
      "validated_at": "2025-12-30T..."
    }
  }
}
```

**Cache location**: `~/.myriad/autotune_profiles.json`

**Benefits**:
- Reuse components across configurations
- Share cache between projects
- Only profile what you use

## API Reference

### suggest_scan_chunk_size()

Find optimal scan_chunk_size for a fixed number of environments.

```python
def suggest_scan_chunk_size(
    num_envs: int,
    env: str,
    agent: str,
    buffer_size: Optional[int] = None,
    force_revalidate: bool = False,
    verbose: bool = True,
) -> int:
    """Suggest optimal scan_chunk_size for a fixed number of environments.

    Args:
        num_envs: Fixed number of parallel environments
        env: Environment name (e.g., "cartpole-control")
        agent: Agent name (e.g., "dqn")
        buffer_size: Replay buffer size (for off-policy agents)
        force_revalidate: Force re-profiling even if cached
        verbose: Control log output verbosity

    Returns:
        Optimal scan_chunk_size for the given configuration
    """
```

### With Buffer Size (Off-Policy Agents)

```python
chunk_size = suggest_scan_chunk_size(
    num_envs=100_000,
    env="cartpole-control",
    agent="dqn",
    buffer_size=10_000,  # For off-policy agents
)
```

### Silent Mode

```python
chunk_size = suggest_scan_chunk_size(
    num_envs=100_000,
    env="cartpole-control",
    agent="dqn",
    verbose=False,  # No output
)
```

### Force Re-Profiling

```python
chunk_size = suggest_scan_chunk_size(
    num_envs=100_000,
    env="cartpole-control",
    agent="dqn",
    force_revalidate=True,  # Ignore cache
)
```

## Performance

| Run | Components | Time |
|-----|-----------|------|
| First (cold cache) | Profile all | ~30-60s |
| Second (warm cache) | Use cache | <1s |
| New num_envs | Test chunk sizes | ~15-20s |
| New environment | Profile env + test | ~25-30s |

**Why it's fast**:
- Lazy profiling (only what you use)
- Component reuse (cache hardware, envs, agents separately)
- Configuration caching (exact match = instant)

## Troubleshooting

### "No valid chunk_size found"

**Cause**: All chunk sizes cause OOM for the given num_envs

**Solution**:
1. Reduce `num_envs`
2. Use simpler agent (e.g., random, PID)
3. Reduce buffer_size for off-policy agents
4. Add more RAM/VRAM

### Cache Stale or Incorrect

**Solution**: Clear and re-profile
```bash
rm ~/.myriad/autotune_profiles.json
```

Then run your code again to rebuild the cache.

### Wrong Hardware Detected

**Cause**: JAX defaulted to CPU when GPU available

**Solution**: Force GPU
```bash
JAX_PLATFORM_NAME=gpu python your_script.py
```

## Advanced Usage

### Sharing Cache Across Machines

Copy cache file:
```bash
# On profiled machine
cp ~/.myriad/autotune_profiles.json /path/to/shared/

# On target machine
cp /path/to/shared/autotune_profiles.json ~/.myriad/
```

**Note**: Only share if hardware is identical!

### Profile-Once Workflow for Clusters

For clusters with identical nodes:
```bash
# On login node (profile once)
python -c "
from myriad.platform.autotune import suggest_scan_chunk_size
chunk = suggest_scan_chunk_size(100_000, 'cartpole-control', 'dqn')
"

# Distribute cache to all compute nodes
rsync ~/.myriad/autotune_profiles.json compute-nodes:~/.myriad/

# On compute nodes (instant, uses cache)
python train.py  # with auto_tune=True
```

## Best Practices

1. **Use auto_tune=True by default** - Let the system optimize for you
2. **Profile once per hardware** - Cache makes subsequent runs instant
3. **Force re-profile after major changes** - New JAX version, OS update, etc.
4. **Share cache in clusters** - Profile once, distribute to identical nodes
5. **Start with small experiments** - Verify autotune works before large runs

## Comparison: Manual vs Auto-Tune

| Manual Tuning | Auto-Tune |
|---------------|-----------|
| Trial-and-error (hours) | Systematic search (minutes) |
| Suboptimal (guesswork) | Optimal (tested) |
| Needs re-tuning per setup | Caches and reuses |
| Error-prone (OOM crashes) | Safe (tests before use) |
| No guarantees | Proven optimal for hardware |

## Related Documentation

- [Performance Benchmarking](../../benchmarks/README.md) - For advanced benchmarking
- [Configuration Guide](../contributing/configuration.md) - Config system details
- [Running Experiments](running_experiments.md) - Training workflows
