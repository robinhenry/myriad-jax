# JAX Architecture

```{warning}
**For Contributors**

This document explains Myriad's internal JAX architecture for developers modifying the engine. If you're just using Myriad to build experiments, you don't need most of this detail. See the [User Guide](../02_user-guide/01_concepts.md) instead.
```

## Pure Functional Design

Myriad's internals are built on **pure functional JAX**, where all core functions are stateless and side-effect free. As a contributor, you must maintain this constraint when adding features.

**Why this constraint exists:**

1. **Automatic vectorization** with `jax.vmap` (enables 100k+ parallel envs)
2. **XLA compilation** with `jax.jit` (10-100x speedups)
3. **Reproducibility** (same inputs → same outputs, critical for science)
4. **Debuggability** (no hidden state to track down)

## Core principles

### Stateless functions

**Bad (stateful):**
```python
class Environment:
    def __init__(self):
        self.state = None

    def reset(self):
        self.state = sample_initial_state()
        return self.get_obs()

    def step(self, action):
        self.state = physics_step(self.state, action)
        return self.get_obs(), self.reward(), self.done()
```

**Good (stateless):**
```python
def reset(state, key, config):
    new_state = sample_initial_state(key, config)
    obs = get_obs(new_state, config)
    return new_state, obs

def step(key, state, action, params, config):
    next_state = physics_step(state, action, params, config)
    obs = get_obs(next_state, config)
    reward = compute_reward(state, action, next_state)
    done = check_termination(next_state, config)
    return obs, next_state, reward, done, {}
```

### Explicit state passing

All state is passed explicitly as PyTrees:

```python
# Environment state
env_state = EnvState(physics=PhysicsState(...), t=0)

# Agent state
agent_state = AgentState(params=params, opt_state=opt_state)

# Training loop explicitly threads state
env_state, agent_state = train_step(env_state, agent_state, key)
```

### No Python control flow in jitted functions

**Bad:**
```python
@jax.jit
def step(state, action):
    if state.x > 0:  # TracerArrayConversionError!
        return state.replace(x=0)
    else:
        return state
```

**Good:**
```python
@jax.jit
def step(state, action):
    new_x = jnp.where(state.x > 0, 0.0, state.x)
    return state.replace(x=new_x)
```

Or:

```python
@jax.jit
def step(state, action):
    return jax.lax.cond(
        state.x > 0,
        lambda s: s.replace(x=0),
        lambda s: s,
        state
    )
```

### Static vs dynamic arguments

**Static arguments** (known at compile time):

```python
@partial(jax.jit, static_argnames=["config"])
def step(state, action, config):
    # config is static, changing it requires recompilation
    ...
```

**Dynamic arguments** (change at runtime):

```python
@jax.jit
def step(state, action, params):
    # params is dynamic, can change without recompilation
    ...
```

## Vectorization with vmap

Run thousands of environments in parallel:

```python
# Single environment
state, obs = reset(state, key, config)

# Batched environments (automatic!)
states, obs_batch = jax.vmap(reset, in_axes=(0, 0, None))(
    states,  # (num_envs,)
    keys,    # (num_envs,)
    config   # Broadcast to all
)
```

## Compilation with jit

```python
# Define pure function
def train_step(state, action):
    ...
    return new_state, metrics

# JIT compile
train_step_jit = jax.jit(train_step)

# First call: slow (compilation)
state, metrics = train_step_jit(state, action)

# Subsequent calls: fast (compiled)
state, metrics = train_step_jit(state, action)
```

## Scan for loops

Replace Python loops with `jax.lax.scan`:

**Bad:**
```python
# Python loop (not compiled)
for i in range(1000):
    state = step(state, action)
```

**Good:**
```python
# Compiled scan
def scan_fn(state, _):
    return step(state, action), None

final_state, _ = jax.lax.scan(scan_fn, init_state, None, length=1000)
```

## PyTrees

JAX automatically handles nested structures:

```python
from flax import struct

@struct.dataclass
class State:
    physics: PhysicsState
    t: int
    params: PhysicsParams

# JAX ops work automatically
states = jax.tree_map(lambda x: x * 2, state)
```

## Common pitfalls

### Using numpy instead of jax.numpy

**Bad:**
```python
import numpy as np
x = np.array([1, 2, 3])  # Won't work in JAX
```

**Good:**
```python
import jax.numpy as jnp
x = jnp.array([1, 2, 3])  # JAX-compatible
```

### Side effects in jitted functions

**Bad:**
```python
@jax.jit
def train_step(state):
    print(f"Step {state.t}")  # Side effect!
    return state
```

**Good:**
```python
@jax.jit
def train_step(state):
    return state

# Print outside jitted code
state = train_step(state)
print(f"Step {state.t}")
```

### Dynamic shapes

**Bad:**
```python
@jax.jit
def process(x):
    if len(x) > 10:  # Dynamic shape!
        return x[:10]
    return x
```

**Good:**
```python
@jax.jit
def process(x):
    return x[:10]  # Fixed shape
```

## Shared physics modules

For common physics patterns (stochastic simulation, numerical integration), Myriad provides shared utilities in `src/myriad/physics/`.

### Gillespie algorithm (`myriad.physics.gillespie`)

Exact stochastic simulation for chemical reaction networks.

**When to use:**
- Biological systems (gene circuits, protein networks)
- Chemical kinetics
- Population dynamics with discrete entities

**Pattern:**
```python
from myriad.physics.gillespie import run_gillespie_loop

def step_physics(key, state, action, params, config):
    return run_gillespie_loop(
        key=key,
        initial_state=state,
        action=action,
        config=config,
        target_time=state.time + config.timestep,
        max_steps=config.max_gillespie_steps,
        compute_propensities_fn=compute_propensities,  # You provide
        apply_reaction_fn=apply_reaction,              # You provide
        get_time_fn=lambda s: s.time,
        update_time_fn=lambda s, t: s._replace(time=t),
    )
```

**What you implement:**
- `compute_propensities(state, action, config)`: Reaction rates for current state
- `apply_reaction(state, reaction_idx)`: State update for each reaction (use `jax.lax.switch`)

**Performance:**
- For systems with <20 reactions: `jax.lax.switch` is fastest
- For systems with >50 reactions: Consider tau-leaping approximation

See `src/myriad/envs/ccas_ccar/physics.py` for example.

---

## Performance tips

1. **Batch operations**: Use `vmap` to process multiple items
2. **Chunk execution**: Use `scan` to reduce Python overhead
3. **Static arguments**: Mark config as static for better optimization
4. **Profile first**: Use JAX profiling tools to find bottlenecks
