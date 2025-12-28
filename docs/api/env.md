# Environment API

Myriad environments follow a pure functional design where all state is explicit and all functions are JIT-compatible.

## Factory Function

```{eval-rst}
.. autofunction:: myriad.envs.make_env
```

## Environment Container

The `Environment` container bundles all environment functions and configuration:

```{eval-rst}
.. autoclass:: myriad.envs.environment.Environment
   :members:
   :undoc-members:
   :show-inheritance:
```

## Protocols

### EnvironmentConfig

```{eval-rst}
.. autoclass:: myriad.envs.environment.EnvironmentConfig
   :members:
   :undoc-members:
   :show-inheritance:
```

### EnvironmentParams

```{eval-rst}
.. autoclass:: myriad.envs.environment.EnvironmentParams
   :members:
   :undoc-members:
   :show-inheritance:
```

### EnvironmentState

```{eval-rst}
.. autoclass:: myriad.envs.environment.EnvironmentState
   :members:
   :undoc-members:
   :show-inheritance:
```

## Available Environments

### CartPole Control

**ID:** `cartpole-control`

Standard inverted pendulum balancing task.

- **Action space:** Discrete(2) - force left or right
- **Observation space:** Box(4) - [x, x_dot, theta, theta_dot]
- **Reward:** +1 per timestep balanced
- **Termination:** |theta| > 12°, |x| > 2.4m, or t >= 500 steps

### CartPole SysID

**ID:** `cartpole-sysid`

Active parameter learning variant with belief state tracking.

- **Action space:** Discrete(2) - force left or right
- **Observation space:** Box(8) - [x, theta, belief_mean, belief_cov]
- **Reward:** Fisher information trace (information gain)
- **Termination:** t >= 1000 steps

### Gene Circuit (CcaS-CcaR)

**ID:** `ccas-ccar-control` or `ccas-ccar-sysid`

Bacterial gene circuit with growth, division, and multi-timescale dynamics.

- **Action space:** Box(1) - light intensity [0, 1]
- **Observation space:** Box(n) - protein concentrations, cell length
- **Reward:** Tracking error to target trajectory
- **Termination:** t >= max_steps

## Design Rationale

### Config vs Params

Environments separate **static configuration** (EnvConfig) from **dynamic parameters** (EnvParams):

- **EnvConfig**: Passed as `static_argnames` to `jax.jit`. Changes trigger recompilation but enable better optimization. Use for: physics constants, termination thresholds, max_steps, environment structure.

- **EnvParams**: Runtime parameters that vary without recompilation. Use for: randomized dynamics, curriculum learning, domain randomization.

### Pure Functions

All environment functions are pure:

```python
# Reset: key, params, config → obs, state
obs, state = env.reset(key, env.params, env.config)

# Step: key, state, action, params, config → obs, state, reward, done, info
next_obs, next_state, reward, done, info = env.step(
key, state, action, env.params, env.config
)
```

This design enables:
- Parallel execution via `jax.vmap`
- Efficient compilation
- Reproducible rollouts
- Gradient-based optimization

## Next Steps

- [Custom Environment Guide](../user-guide/custom_env.md): Build your own environment
- [Core Concepts](../user-guide/concepts.md): Understand the three-layer architecture
- [Environment Examples](https://github.com/robinhenry/myriad-jax/tree/main/src/myriad/envs): Browse source code
