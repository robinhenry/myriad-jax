# Add a custom environment

Add your own physics to Myriad by implementing the three-layer pattern.

This guide covers two types of environments:

1. **Deterministic systems** (ODE-based): Damped oscillator example
2. **Stochastic systems** (Gillespie-based): Chemical reaction networks

## Step 1: Create the physics module

Create `src/myriad/envs/oscillator/physics.py`:

```python
"""Pure stateless physics for a damped harmonic oscillator."""

from typing import NamedTuple
import chex
import jax.numpy as jnp
from flax import struct


class PhysicsState(NamedTuple):
    """Physical state: position and velocity.

    For fully observable systems, this also serves as the observation type.
    Add conversion methods for neural network compatibility.

    The platform automatically converts NamedTuple observations to arrays for
    efficient processing. The .to_array() method is used during conversion.
    """
    x: chex.Array      # Position (m)
    v: chex.Array      # Velocity (m/s)

    def to_array(self) -> chex.Array:
        """Convert to array for neural networks.

        This method is called automatically by the platform to convert observations
        to homogeneous arrays for efficient JAX operations. Agents can also use the
        `myriad.utils.to_array()` utility to handle both NamedTuple and array observations.
        """
        return jnp.stack([self.x, self.v])

    @classmethod
    def from_array(cls, arr: chex.Array) -> "PhysicsState":
        """Create from array."""
        return cls(x=arr[0], v=arr[1])  # type: ignore


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

Create `src/myriad/envs/oscillator/tasks/control.py`:

```python
"""Control task: Drive oscillator to origin."""

from typing import Any, Dict, NamedTuple, Tuple
import chex
import jax.numpy as jnp
from flax import struct

from myriad.core.spaces import Box
from myriad.envs.environment import Environment
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


def get_obs(state: ControlTaskState, config: ControlTaskConfig) -> PhysicsState:
    """Extract observation from state.

    For this fully observable system, observation = physics state.
    Returns PhysicsState directly (no conversion needed).
    """
    return state.physics  # Direct passthrough!


def make_env(**kwargs) -> Environment:
    """Factory function for registry."""
    config = ControlTaskConfig(**kwargs)
    params = ControlTaskParams()

    # Return Environment NamedTuple
    from myriad.envs.environment import Environment
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

## Step 2b: Partially Observable Systems (Optional)

If your system has **limited observability** (agent doesn't observe full state), create a dedicated observation type:

```python
# In tasks/control.py

class OscillatorObs(NamedTuple):
    """Partial observation: only position, not velocity."""
    x: chex.Array            # Observable: position
    x_target: chex.Array     # Observable: target
    # Note: velocity is NOT observable!

    def to_array(self) -> chex.Array:
        """Convert to array for neural networks."""
        return jnp.stack([self.x, self.x_target])

    @classmethod
    def from_array(cls, arr: chex.Array) -> "OscillatorObs":
        """Create from array."""
        return cls(x=arr[0], x_target=arr[1])  # type: ignore


def get_obs(state: ControlTaskState, config: ControlTaskConfig) -> OscillatorObs:
    """Extract partial observation.

    PhysicsState has (x, v) but agent only sees (x, target).
    This creates a partially observable problem.
    """
    return OscillatorObs(
        x=state.physics.x,
        x_target=jnp.array(0.0),  # Target is always origin
    )
```

**When to use each pattern:**

| System Type | Observation = State? | Pattern | Example |
|-------------|---------------------|---------|---------|
| **Fully Observable** | Yes | Reuse `PhysicsState` | CartPole, Oscillator |
| **Partially Observable** | No | Create dedicated `*Obs` type | CCAS-CCAR, POMDP tasks |

**Benefits of this approach:**

- **Type safety**: `PhysicsState ≠ OscillatorObs` makes partial observability explicit
- **No duplication**: Fully observable systems reuse physics types
- **Semantic clarity**: Field names document what agent perceives

## Step 3: Register the environment

Edit `src/myriad/envs/__init__.py`:

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

from myriad.envs.oscillator.physics import (
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
from myriad.envs import make_env
from myriad.agents import make_agent
from myriad.configs.default import Config
from myriad.platform.runner import train_and_evaluate

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

---

## Gillespie-based environments (stochastic systems)

For chemical reaction networks and other stochastic systems, use the shared Gillespie engine.

### Create physics module

Create `src/myriad/envs/my_circuit/physics.py`:

```python
"""Stochastic gene circuit physics using Gillespie algorithm."""

from typing import NamedTuple
import chex
import jax
import jax.numpy as jnp
from flax import struct

from myriad.physics.gillespie import run_gillespie_loop


class PhysicsState(NamedTuple):
    """Physical state of gene circuit."""
    time: chex.Array
    protein_A: chex.Array  # Concentration
    protein_B: chex.Array


@struct.dataclass
class PhysicsConfig:
    """Static physics constants."""
    timestep_minutes: float = 5.0
    max_gillespie_steps: int = 10000

    # Reaction rates
    k_production_A: float = 1.0
    k_degradation: float = 0.01


@struct.dataclass
class PhysicsParams:
    """Dynamic parameters (for domain randomization)."""
    ...


def compute_propensities(
    state: PhysicsState,
    action: chex.Array,
    config: PhysicsConfig,
) -> chex.Array:
    """Compute reaction rates.

    Reactions:
    1. ∅ → A  (rate: k_production_A * action)
    2. A → ∅  (rate: k_degradation * A)
    3. A → B  (rate: k_conversion * A)
    4. B → ∅  (rate: k_degradation * B)
    """
    A, B = state.protein_A, state.protein_B
    U = action  # Control input

    r1 = config.k_production_A * U
    r2 = config.k_degradation * A
    r3 = 0.5 * A  # Conversion rate
    r4 = config.k_degradation * B

    return jnp.array([r1, r2, r3, r4])


def apply_reaction(
    state: PhysicsState,
    reaction_idx: chex.Array,
) -> PhysicsState:
    """Apply selected reaction to state.

    Uses jax.lax.switch for efficiency.
    """
    def r0(s): return s._replace(protein_A=s.protein_A + 1)
    def r1(s): return s._replace(protein_A=jnp.maximum(s.protein_A - 1, 0))
    def r2(s): return s._replace(protein_A=jnp.maximum(s.protein_A - 1, 0),
                                   protein_B=s.protein_B + 1)
    def r3(s): return s._replace(protein_B=jnp.maximum(s.protein_B - 1, 0))

    return jax.lax.switch(reaction_idx, [r0, r1, r2, r3], state)


def step_physics(
    key: chex.PRNGKey,
    state: PhysicsState,
    action: chex.Array,
    params: PhysicsParams,
    config: PhysicsConfig,
) -> PhysicsState:
    """Step physics using shared Gillespie engine."""
    final_state = run_gillespie_loop(
        key=key,
        initial_state=state,
        action=action,
        config=config,
        target_time=state.time + config.timestep_minutes,
        max_steps=config.max_gillespie_steps,
        compute_propensities_fn=compute_propensities,
        apply_reaction_fn=apply_reaction,
        get_time_fn=lambda s: s.time,
        update_time_fn=lambda s, t: s._replace(time=t),
    )
    return final_state
```

!!! note "Shared Gillespie engine"
    The `run_gillespie_loop()` function handles the stochastic simulation loop.
    You only implement system-specific parts:

    - `compute_propensities()`: Reaction rates based on current state
    - `apply_reaction()`: How each reaction changes the state

### Key differences from deterministic systems

| Aspect | Deterministic (ODE) | Stochastic (Gillespie) |
|--------|---------------------|------------------------|
| **Physics step** | Euler/RK integration | Gillespie algorithm |
| **State updates** | Continuous (floats) | Discrete (molecule counts) |
| **Requires RNG key** | No | Yes (for `step_physics`) |
| **Reproducibility** | Deterministic | Stochastic (seed-dependent) |
| **Common use** | Mechanical systems | Biological/chemical systems |

### Performance considerations

The Gillespie algorithm runs many micro-steps per RL timestep. For efficiency:

- Keep `max_gillespie_steps` reasonable (10,000 is typical)
- Use `jax.lax.switch` for `apply_reaction()` (faster than matrix operations for <20 reactions)
- Consider tau-leaping for systems with >50 reactions (approximate but 100x faster)

### Example: CcaS-CcaR circuit

See `src/myriad/envs/ccas_ccar/physics.py` for a real implementation with:

- 5 chemical reactions
- Hill function kinetics
- Light-controlled gene expression

---

## Next steps

- [Core Concepts](concepts.md): Understanding the three-layer pattern
- [Custom Agent Guide](custom_agent.md): Implement learning algorithms
- [Running Experiments](running_experiments.md): Train agents on your tasks
