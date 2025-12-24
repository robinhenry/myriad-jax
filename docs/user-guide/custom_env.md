# Add a custom environment

Add your own physics to Aion by implementing the three-layer pattern. This guide shows how to create a simple damped oscillator environment.

## Step 1: Create the physics module

Create `src/aion/envs/oscillator/physics.py`:

```python
"""Pure stateless physics for a damped harmonic oscillator."""

from typing import NamedTuple
import chex
import jax.numpy as jnp
from flax import struct


class PhysicsState(NamedTuple):
    """Physical state: position and velocity."""
    x: chex.Array      # Position (m)
    v: chex.Array      # Velocity (m/s)


@struct.dataclass
class PhysicsConfig:
    """Static physics constants (compile-time)."""
    dt: float = 0.01        # Integration timestep (s)
    force_scale: float = 1.0  # Action scaling


@struct.dataclass
class PhysicsParams:
    """Dynamic parameters for domain randomization."""
    omega: chex.Array = 2.0    # Natural frequency (rad/s)
    gamma: chex.Array = 0.1    # Damping coefficient


def step_physics(
    state: PhysicsState,
    action: chex.Array,
    params: PhysicsParams,
    config: PhysicsConfig,
) -> PhysicsState:
    """Evolve physics by one timestep.

    Dynamics: ẍ + 2γẋ + ω²x = F(action)

    Args:
        state: Current (x, v)
        action: External force
        params: System parameters (ω, γ)
        config: Static configuration

    Returns:
        Next state after dt
    """
    x, v = state
    force = action * config.force_scale

    # Acceleration: a = -2γv - ω²x + F
    a = -2.0 * params.gamma * v - params.omega**2 * x + force

    # Euler integration
    x_next = x + config.dt * v
    v_next = v + config.dt * a

    return PhysicsState(x_next, v_next)
```

!!! warning "JAX constraints"
    - `step_physics()` must be a pure function (no side effects)
    - All arrays must be JAX arrays (`jnp.ndarray`, not `np.ndarray`)
    - Avoid Python `if/else` statements (use `jnp.where` or `jax.lax.cond`)

## Step 2: Create a task wrapper

Create `src/aion/envs/oscillator/tasks/control.py`:

```python
"""Control task: Drive oscillator to origin."""

from typing import Any, Dict, NamedTuple, Tuple
import chex
import jax.numpy as jnp
from flax import struct

from aion.core.spaces import Box
from aion.envs.environment import Environment
from ..physics import PhysicsConfig, PhysicsParams, PhysicsState, step_physics


class ControlTaskState(NamedTuple):
    """Task state wrapping physics."""
    physics: PhysicsState
    t: chex.Array  # Timestep counter


@struct.dataclass
class ControlTaskConfig:
    """Task configuration."""
    physics: PhysicsConfig = struct.field(default_factory=PhysicsConfig)
    max_steps: int = 500
    max_position: float = 5.0


@struct.dataclass
class ControlTaskParams:
    """Task parameters."""
    physics: PhysicsParams = struct.field(default_factory=PhysicsParams)


def _reset(
    key: chex.PRNGKey,
    state: ControlTaskState,
    params: ControlTaskParams,
    config: ControlTaskConfig,
) -> Tuple[ControlTaskState, chex.Array]:
    """Reset to random initial state."""
    import jax.random as jr

    # Sample random initial position and velocity
    key_x, key_v = jr.split(key)
    x = jr.uniform(key_x, minval=-2.0, maxval=2.0)
    v = jr.uniform(key_v, minval=-1.0, maxval=1.0)

    physics_state = PhysicsState(x, v)
    task_state = ControlTaskState(physics=physics_state, t=jnp.array(0))
    obs = get_obs(task_state, config)

    return task_state, obs


def _step(
    key: chex.PRNGKey,
    state: ControlTaskState,
    action: chex.Array,
    params: ControlTaskParams,
    config: ControlTaskConfig,
) -> Tuple[chex.Array, ControlTaskState, chex.Array, chex.Array, Dict[str, Any]]:
    """Step the control task."""
    # 1. Apply physics (Layer A)
    next_physics = step_physics(
        state.physics, action, params.physics, config.physics
    )

    # 2. Compute reward (negative squared distance to origin)
    reward = -(next_physics.x**2 + next_physics.v**2)

    # 3. Check termination
    t_next = state.t + 1
    out_of_bounds = jnp.abs(next_physics.x) > config.max_position
    time_limit = t_next >= config.max_steps
    done = out_of_bounds | time_limit

    # 4. Create next state
    next_state = ControlTaskState(physics=next_physics, t=t_next)
    obs = get_obs(next_state, config)

    return obs, next_state, reward, done, {}


def get_obs(state: ControlTaskState, config: ControlTaskConfig) -> chex.Array:
    """Extract observation from state."""
    return jnp.array([state.physics.x, state.physics.v])


def make_env(**kwargs) -> Environment:
    """Factory function for registry."""
    config = ControlTaskConfig(**kwargs)
    params = ControlTaskParams()

    # Return Environment NamedTuple
    from aion.envs.environment import Environment
    return Environment(
        reset=_reset,
        step=_step,
        get_obs=get_obs,
        action_space=Box(low=-1.0, high=1.0, shape=(1,)),
        observation_space=Box(low=-10.0, high=10.0, shape=(2,)),
        default_params=params,
        config=config,
    )
```

## Step 3: Register the environment

Edit `src/aion/envs/__init__.py`:

```python
from .oscillator.tasks import control as oscillator_control  # Add this

ENV_REGISTRY = {
    "ccas_ccar_v1": ccas_ccar_v1.make_env,
    "cartpole-control": cartpole_control.make_env,
    "cartpole-sysid": cartpole_sysid.make_env,
    "oscillator-control": oscillator_control.make_env,  # Add this
}
```

## Step 4: Create Hydra config

Create `configs/env/oscillator_control.yaml`:

```yaml
# Physics configuration
physics:
  dt: 0.01
  force_scale: 1.0

# Task configuration
max_steps: 500
max_position: 5.0
```

## Step 5: Test your environment

Create `tests/envs/oscillator/test_physics.py`:

```python
"""Test oscillator physics."""

import jax
import jax.numpy as jnp
import pytest

from aion.envs.oscillator.physics import (
    PhysicsConfig,
    PhysicsParams,
    PhysicsState,
    step_physics,
)


def test_undamped_oscillation():
    """Undamped oscillator should conserve energy."""
    # Setup: No damping, no forcing
    config = PhysicsConfig(dt=0.001)
    params = PhysicsParams(omega=1.0, gamma=0.0)
    state = PhysicsState(x=1.0, v=0.0)  # Released from rest

    # Initial energy: E = 0.5 * ω² * x²
    energy_0 = 0.5 * params.omega**2 * state.x**2

    # Simulate 100 steps
    action = jnp.array(0.0)  # No external force
    for _ in range(100):
        state = step_physics(state, action, params, config)

    # Final energy
    energy_final = 0.5 * params.omega**2 * state.x**2 + 0.5 * state.v**2

    # Energy should be conserved (within numerical error)
    assert jnp.isclose(energy_0, energy_final, atol=1e-3)


def test_critical_damping():
    """Critically damped system should not oscillate."""
    config = PhysicsConfig(dt=0.01)
    params = PhysicsParams(omega=1.0, gamma=1.0)  # Critical damping
    state = PhysicsState(x=1.0, v=0.0)

    # Simulate and track zero-crossings
    action = jnp.array(0.0)
    positions = [state.x]

    for _ in range(200):
        state = step_physics(state, action, params, config)
        positions.append(state.x)

    # Count sign changes (oscillations)
    sign_changes = jnp.sum(jnp.diff(jnp.sign(jnp.array(positions))) != 0)

    # Critically damped: at most 1 zero crossing
    assert sign_changes <= 1
```

Run the tests:

```bash
python -m pytest tests/envs/oscillator/
```

## Step 6: Train an agent

```bash
python scripts/train.py \
  env=oscillator_control \
  agent=pqn \
  run.num_envs=10000 \
  run.total_timesteps=1e6
```

Or programmatically:

```python
from aion.envs import make_env
from aion.agents import make_agent
from aion.configs.default import Config
from aion.platform.runner import train_and_evaluate

# Create environment
env = make_env("oscillator-control")

# Create config
config = Config(
    env={"_target_": "oscillator-control"},
    agent={"_target_": "pqn"},
    run={"num_envs": 10000, "total_timesteps": 1_000_000}
)

# Train
train_and_evaluate(config)
```

## Common issues

### Problem: `TracerArrayConversionError`

Python control flow in jitted functions causes this.

**Bad:**
```python
if x > 0:
    return x
else:
    return -x
```

**Good:**
```python
return jnp.where(x > 0, x, -x)
```

### Problem: Recompilation on every call

Static arguments must be marked explicitly.

**Bad:**
```python
@jax.jit
def step(state, config):  # config changes → recompile
    ...
```

**Good:**
```python
@partial(jax.jit, static_argnames=["config"])
def step(state, config):  # config is static
    ...
```

### Problem: Arrays have wrong type

Use `jnp.array()`, not `np.array()`.

**Bad:**
```python
import numpy as np
state = PhysicsState(x=np.array(0.0), ...)  # Will fail in JAX
```

**Good:**
```python
import jax.numpy as jnp
state = PhysicsState(x=jnp.array(0.0), ...)  # JAX-compatible
```

## Next steps

- [Core Concepts](concepts.md): Understanding the three-layer pattern
- [Custom Agent Guide](custom_agent.md): Implement learning algorithms
- [Running Experiments](running_experiments.md): Train agents on your tasks
