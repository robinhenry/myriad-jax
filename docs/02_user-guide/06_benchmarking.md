# Benchmarking Your Experiments

How to measure the performance of your own Myriad experiments using built-in benchmarking utilities.

## Overview

Myriad provides JAX-aware benchmarking utilities that handle the subtleties of accurate performance measurement:

- **Proper warmup**: Exclude JIT compilation from timings
- **Blocking**: Wait for GPU work to complete (`.block_until_ready()`)
- **Statistical analysis**: Mean, std, median, min/max over multiple runs

## Quick Start

```python
from benchmarks.utils import time_jitted_fn, warmup_jitted_fn
import jax
import jax.numpy as jnp

@jax.jit
def my_function(x):
    return jnp.dot(x, x.T)

x = jnp.ones((1000, 1000))

# Time the function (with automatic warmup)
stats = time_jitted_fn(my_function, x, num_runs=100, warmup_steps=10)

print(f"Mean: {stats['mean']*1000:.2f}ms")
print(f"Std:  {stats['std']*1000:.2f}ms")
print(f"Min:  {stats['min']*1000:.2f}ms")
print(f"Max:  {stats['max']*1000:.2f}ms")
```

## Benchmarking Utilities

### `time_jitted_fn()` - The Gold Standard

This is the recommended way to benchmark JAX functions:

```python
def time_jitted_fn(
    fn: Callable,
    *args,
    num_runs: int = 100,
    warmup_steps: int = 10,
    measure_compile_time: bool = True,
    **kwargs,
) -> dict[str, float]:
    """Time a jitted function with proper warmup and blocking.

    Returns:
        Dictionary with timing statistics:
        - mean: Mean execution time (seconds)
        - std: Standard deviation (seconds)
        - min: Minimum time (seconds)
        - max: Maximum time (seconds)
        - median: Median time (seconds)
        - all_times: Array of all timing measurements
        - compile_time: Compilation time (seconds, if measure_compile_time=True)
    """
```

**What it does**:
1. Measures compilation time (first call)
2. Warms up the function (additional calls for optimization)
3. Runs `num_runs` iterations with proper blocking
4. Returns statistical summary including compilation overhead

**Example - Basic Usage**:
```python
from myriad import create_config, make_train_step_fn
from benchmarks.utils import time_jitted_fn

# Setup
config = create_config(env="cartpole-control", agent="dqn", num_envs=1000)
env = make_env(config.env)
agent = make_agent(config.agent, env.get_action_space(env.config))
step_fn = make_train_step_fn(agent, env, None, config.num_envs)

# Benchmark (includes compilation time by default)
stats = time_jitted_fn(
    step_fn,
    key, agent_state, training_state, None,
    batch_size=1,
    num_runs=50,
    warmup_steps=5
)

print(f"Compilation: {stats['compile_time']*1000:.1f}ms (one-time cost)")
print(f"Execution:   {stats['mean']*1000:.1f}ms (per run)")
print(f"Throughput:  {1000 / stats['mean']:.0f} steps/s")
```

**Example - Understanding Compilation Overhead**:
```python
import jax
import jax.numpy as jnp
from benchmarks.utils import time_jitted_fn

@jax.jit
def my_expensive_function(x):
    return jnp.dot(x, x.T)

x = jnp.ones((5000, 5000))
stats = time_jitted_fn(my_expensive_function, x, num_runs=10)

# Calculate when compilation cost amortizes
compilation_overhead = stats['compile_time'] / stats['mean']
print(f"Compilation takes {compilation_overhead:.1f}x a single execution")
print(f"After ~{int(compilation_overhead)} runs, compilation cost amortizes")

# Total time for different scenarios
for n_runs in [1, 10, 100, 1000]:
    total = stats['compile_time'] + (n_runs * stats['mean'])
    print(f"{n_runs:>4} runs: {total:.2f}s total")
```

### `warmup_jitted_fn()` - Manual Warmup

For cases where you want to control warmup separately:

```python
def warmup_jitted_fn(
    fn: Callable,
    *args,
    warmup_steps: int = 10,
    **kwargs
) -> None:
    """Warm up a jitted function to trigger compilation."""
```

**Example**:
```python
# Warm up first
warmup_jitted_fn(my_function, x, warmup_steps=5)

# Then time manually
import time
start = time.perf_counter()
result = my_function(x)
result.block_until_ready()  # CRITICAL!
elapsed = time.perf_counter() - start
```

### `measure_compilation_time()` - JIT Overhead

Measure how long compilation takes:

```python
from benchmarks.utils import measure_compilation_time

compile_time = measure_compilation_time(my_function, x)
print(f"Compilation took: {compile_time:.2f}s")
```

## Critical JAX Benchmarking Rules

### 1. Always Use `.block_until_ready()`

JAX is **asynchronous** - it dispatches work and returns immediately. Without blocking, you measure dispatch time, not actual computation time.

```python
# ❌ WRONG - measures dispatch, not computation
start = time.time()
result = jitted_fn(x)
elapsed = time.time() - start  # Way too fast!

# ✅ CORRECT - waits for GPU to finish
start = time.time()
result = jitted_fn(x)
result.block_until_ready()  # Block until GPU completes
elapsed = time.time() - start
```

### 2. Always Warmup

The first execution includes JIT compilation and is much slower. Always discard warmup runs.

```python
# ❌ WRONG - includes compilation in timing
times = []
for _ in range(100):
    start = time.time()
    result = fn(x)
    result.block_until_ready()
    times.append(time.time() - start)
# First run is 10-100x slower!

# ✅ CORRECT - warmup first
warmup_jitted_fn(fn, x, warmup_steps=10)
times = []
for _ in range(100):
    start = time.time()
    result = fn(x)
    result.block_until_ready()
    times.append(time.time() - start)
```

### 3. Multiple Runs for Statistics

Single measurements are noisy. Always run multiple times and report statistics.

```python
# ❌ WRONG - single measurement
elapsed = time_one_run(fn, x)

# ✅ CORRECT - statistical summary
stats = time_jitted_fn(fn, x, num_runs=100)
print(f"Mean: {stats['mean']:.4f}s ± {stats['std']:.4f}s")
```

## Helper Functions

### `calculate_throughput()`

Convert timing to throughput:

```python
from benchmarks.utils import calculate_throughput

total_steps = num_envs * num_iterations
time_seconds = stats['mean']
throughput = calculate_throughput(total_steps, time_seconds)

print(f"Throughput: {throughput/1e6:.1f}M steps/s")
```

### `format_number()` - Pretty Printing

Format large numbers with K/M/B suffixes:

```python
from benchmarks.utils import format_number

print(f"Throughput: {format_number(50_000_000)} steps/s")
# Output: "50.00M steps/s"
```

### `calculate_scaling_efficiency()`

Measure parallel scaling efficiency:

```python
from benchmarks.utils import calculate_scaling_efficiency

baseline_throughput = 1_000_000   # 1 env
parallel_throughput = 50_000_000  # 100 envs

efficiency = calculate_scaling_efficiency(
    throughput_n=parallel_throughput,
    throughput_baseline=baseline_throughput,
    n_envs=100,
    baseline_envs=1
)

print(f"Scaling efficiency: {efficiency:.2%}")
# Perfect scaling = 1.0 (100%)
```

## Common Benchmarking Patterns

### Measure Environment Throughput

```python
from myriad import create_config
from myriad.envs import make_env
from benchmarks.utils import time_jitted_fn
import jax

config = create_config(env="cartpole-control", num_envs=10000)
env = make_env(config.env)

# Create vectorized reset and step
reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))
step_fn = jax.vmap(env.step, in_axes=(None, 0, 0, None))

# Benchmark reset
keys = jax.random.split(jax.random.PRNGKey(0), 10000)
reset_stats = time_jitted_fn(
    reset_fn, keys, env.params, env.config,
    num_runs=50
)

print(f"Reset: {10000/reset_stats['mean']:.0f} envs/s")

# Benchmark step
obs, state = reset_fn(keys, env.params, env.config)
actions = jnp.zeros((10000,) + env.get_action_space(env.config).shape)

step_stats = time_jitted_fn(
    step_fn, env.params, state, actions, env.config,
    num_runs=100
)

print(f"Step: {10000/step_stats['mean']:.0f} steps/s")
```

### Measure Agent Throughput

```python
from myriad.agents import make_agent
from benchmarks.utils import time_jitted_fn

agent = make_agent("dqn", action_space)

# Benchmark select_action
obs = jnp.ones((10000, 4))  # Batch of observations
select_stats = time_jitted_fn(
    agent.select_action,
    jax.random.PRNGKey(0), agent_state, obs, agent.params,
    deterministic=False,
    num_runs=100
)

print(f"Action selection: {10000/select_stats['mean']:.0f} actions/s")
```

### Compare Configurations

```python
from benchmarks.utils import time_jitted_fn
import jax

configs = [
    {"num_envs": 1000, "scan_chunk": 256},
    {"num_envs": 10000, "scan_chunk": 128},
    {"num_envs": 100000, "scan_chunk": 64},
]

for cfg in configs:
    step_fn = create_step_fn(cfg["num_envs"], cfg["scan_chunk"])
    stats = time_jitted_fn(step_fn, *args, num_runs=50)

    throughput = cfg["num_envs"] / stats['mean']
    print(f"{cfg['num_envs']:>6} envs, chunk {cfg['scan_chunk']:>3}: "
          f"{throughput/1e6:>6.1f}M steps/s")
```

## Memory Profiling

### Estimate PyTree Memory

```python
from myriad.utils.memory import estimate_pytree_memory_mb

# Estimate memory of any JAX PyTree
memory_mb = estimate_pytree_memory_mb(agent_state)
print(f"Agent state: {memory_mb:.1f} MB")

memory_mb = estimate_pytree_memory_mb(training_state)
print(f"Training state: {memory_mb:.1f} MB")
```

### Track Memory Usage

```python
import jax

# Get initial memory
initial = jax.devices()[0].memory_stats()

# Run your code
result = expensive_operation()

# Check memory increase
final = jax.devices()[0].memory_stats()
used_mb = (final['bytes_in_use'] - initial['bytes_in_use']) / 1024**2
print(f"Memory used: {used_mb:.1f} MB")
```

## Device Information

```python
from benchmarks.utils import get_device_info

info = get_device_info()
print(f"Platform: {info['platform']}")
print(f"Devices: {info['device_count']}")
print(f"Device list: {info['devices']}")
```

## Best Practices

1. **Always warmup** - First run includes compilation
2. **Always block** - JAX is async, use `.block_until_ready()`
3. **Run multiple times** - Report mean ± std, not single measurements
4. **Isolate measurements** - One thing at a time (env vs agent vs full loop)
5. **Document setup** - Hardware, JAX version, configuration
6. **Check device** - Verify GPU is actually being used (`jax.devices()`)

## Troubleshooting

### "Results are inconsistent"

- Not calling `.block_until_ready()` - measuring dispatch, not computation
- Not warming up - including compilation in measurements
- System under load - close other programs

### "First run is much slower"

- Normal! That's JIT compilation
- Use warmup to exclude it: `time_jitted_fn(..., warmup_steps=10)`

### "GPU not faster than CPU"

- Check device: `print(jax.devices())`
- Force GPU: `JAX_PLATFORM_NAME=gpu python script.py`
- Problem size might be too small for GPU to shine

### "Out of memory during benchmarking"

- Reduce `num_runs` (e.g., 100 → 10)
- Use smaller test data
- Clear cache between runs: `jax.clear_caches()`

## See Also

- [Auto-Tuning Guide](auto_tuning.md) - Automatic configuration optimization
- [Validation Benchmarks](../../benchmarks/README.md) - Running Myriad's validation suite
- [Performance Demo](../../examples/10_performance_demo.py) - Interactive demonstration
