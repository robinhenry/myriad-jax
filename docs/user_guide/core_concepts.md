# Core concepts

## Three-layer design

Every experiment has three layers:

```
┌─────────────────────────────────────┐
│  Learner  (DQN, PQN, PID, Random)  │  Picks actions from observations
└──────────────┬──────────────────────┘
               │ obs, reward, done
┌──────────────▼──────────────────────┐
│  Task  (Control)                    │  Defines obs, reward, termination
└──────────────┬──────────────────────┘
               │ state, action → next_state
┌──────────────▼──────────────────────┐
│  Physics  (step_physics)            │  Pure dynamics, no RL concepts
└─────────────────────────────────────┘
```

**Physics** is a stateless pure function: `(state, action, params, config) → next_state`. No rewards, no termination. You can call it directly for MPC planning or test it against analytical solutions.

**Task** wraps physics with what the agent observes (`get_obs`), what it optimizes (reward), and when episodes end. Multiple tasks can share the same physics — you write the dynamics once.

**Learner** only sees `(obs, reward, done)`. It doesn't know or care what system it's controlling.

Layers talk to each other through Python Protocols (structural typing), not inheritance.

## JAX-native

Environment and agent functions are pure. They work with `jax.jit`, `jax.vmap`, and `jax.lax.scan`.

- **State is data.** Environment state is a NamedTuple PyTree, not a mutable object.
- **No Python control flow in JIT paths.** Use `jax.lax.cond` / `jax.lax.select`, not `if/else`.
- **Observations are NamedTuples.** Classical controllers access fields by name (`obs.theta`). Neural networks call `obs.to_array()`. Both work with `vmap`.

One `vmap` call gets you 100,000 parallel environments on a single GPU.

## Evaluation vs. training

Two execution paths.

**Evaluation** runs a fixed agent (no learning):

```python
config = create_eval_config("cartpole-control", "pid", kp=1.0)
results = evaluate(config)
```

**Training** trains an RL agent, then evaluates:

```python
config = create_config("cartpole-control", "dqn", num_envs=64, steps_per_env=2000)
results = train_and_evaluate(config)
```

Both return a `Results` object with metrics. Pass `return_episodes=True` to `evaluate()` to get per-step trajectory data.

## What's in the box

### Environments

Environments are registered as `{system}-{task}`:

| Environment | System | Task |
|---|---|---|
| `cartpole-control` | Classic cart-pole | Balance the pole |
| `pendulum-control` | Classic pendulum | Swing up and balance |
| `ccas-ccar-control` | Gene circuit (CcaS/CcaR) | Track target protein expression |

```python
from myriad.envs import make_env, list_envs

list_envs()           # all registered names
env = make_env("cartpole-control")
```

### Agents

| Agent | Type | Actions |
|---|---|---|
| `random` | Classical | Any |
| `bangbang` | Classical | Discrete |
| `pid` | Classical | Continuous |
| `dqn` | RL | Discrete |
| `pqn` | RL | Continuous |

Classical agents don't learn — use `create_eval_config` + `evaluate`. RL agents need `create_config` + `train_and_evaluate`.
