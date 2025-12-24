# Environment reference

## Available environments

### CartPole control

**ID:** `cartpole-control`

**Description:** Standard inverted pendulum balancing task.

**Action space:** `Discrete(2)` - force left or right

**Observation space:** `Box(4)` - `[x, x_dot, theta, theta_dot]`

**Reward:** +1 per timestep balanced

**Termination:**

- `|theta| > 0.21` rad (12 degrees)
- `|x| > 2.4` m
- `t >= 500` steps

### CartPole SysID

**ID:** `cartpole-sysid`

**Description:** Active parameter learning variant.

**Action space:** `Discrete(2)` - force left or right

**Observation space:** `Box(8)` - `[x, theta, belief_mean, belief_cov]`

**Reward:** Fisher information trace (information gain)

**Termination:**

- `t >= 1000` steps

### Gene circuit

**ID:** `ccas_ccar_v1`

**Description:** Bacterial gene circuit with growth and division dynamics.

**Action space:** `Box(1)` - light intensity

**Observation space:** `Box(n)` - protein concentrations, cell length

**Reward:** Tracking error to target trajectory

**Termination:**

- `t >= max_steps`

## Environment protocol

```python
class Environment(Protocol):
    def reset(
        self,
        state: EnvState,
        key: PRNGKey,
        config: EnvConfig
    ) -> tuple[EnvState, Observation]:
        ...

    def step(
        self,
        key: PRNGKey,
        state: EnvState,
        action: Array,
        params: EnvParams,
        config: EnvConfig
    ) -> tuple[Observation, EnvState, Reward, Done, Info]:
        ...

    def get_obs(
        self,
        state: EnvState,
        config: EnvConfig
    ) -> Observation:
        ...

    action_space: Space
    observation_space: Space
    default_params: EnvParams
    config: EnvConfig
```

## Next steps

- [Custom Environment Guide](../user-guide/custom_env.md): Implementation guide
- [Core Concepts](../user-guide/concepts.md): Understanding the three-layer pattern
