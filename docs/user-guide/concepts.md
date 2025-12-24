# Core Concepts

## The Three-Layer Architecture

Aion separates your experiment into three independent layers. This architecture isn't just an implementation detail—it's designed to match how you actually think about scientific problems.

### Why this matters for your work

Traditional RL platforms force you to bundle physics, task objectives, and learning algorithms into a single monolithic environment. This creates friction:

**Scenario: You want to run two experiments on the same physical system**

- **Experiment A**: Control the system to a target state (classic RL)
- **Experiment B**: Learn the unknown parameters of the system (active learning)

In Gym or Gymnax, you'd duplicate the entire physics implementation twice. In Aion, you write the physics once and wrap it with different task definitions.

**Scenario: You want to use model-based planning**

Your MPC controller needs direct access to the dynamics function to simulate trajectories. Traditional environments hide this inside `step()`, forcing you to either hack around the API or rewrite the physics.

In Aion, the physics layer is directly callable—no environment wrapper needed.

## The layers

```
┌─────────────────────────────────────┐
│   Layer C: Learner                  │
│   (DQN, PQN, Random)                │
│   - Agnostic to physics/tasks       │
└──────────────┬──────────────────────┘
               │ Standard interface
               │ obs, reward, done
┌──────────────▼──────────────────────┐
│   Layer B: Task Wrapper             │
│   (Control, SysID)                  │
│   - Observation function            │
│   - Reward function                 │
│   - Termination logic               │
└──────────────┬──────────────────────┘
               │ Direct function call
               │ state, action → next_state
┌──────────────▼──────────────────────┐
│   Layer A: Pure Physics             │
│   (step_physics)                    │
│   - Stateless dynamics              │
│   - No RL concepts                  │
└─────────────────────────────────────┘
```

## Layer A: Pure physics

**Responsibility:** Implement ground truth dynamics without any RL concepts.

**Requirements:**

- Stateless pure function
- All parameters passed as arguments (no class attributes)
- Works with `jax.vmap` and `jax.jit`
- No rewards, observations, or termination logic

**Example from `cartpole/physics.py`:**

```python
def step_physics(
    state: PhysicsState,
    action: chex.Array,
    params: PhysicsParams,
    config: PhysicsConfig,
) -> PhysicsState:
    """Pure physics step. No task logic."""
    x, x_dot, theta, theta_dot = state

    # Convert action to force
    force = (2 * action - 1) * config.force_magnitude

    # Cart-pole dynamics (equations of motion)
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    temp = (force + config.pole_mass * config.pole_length * theta_dot**2 * sin_theta) / total_mass
    theta_acc = (config.gravity * sin_theta - cos_theta * temp) / denominator
    x_acc = temp - config.pole_mass * config.pole_length * theta_acc * cos_theta / total_mass

    # Euler integration
    return PhysicsState(
        x + config.dt * x_dot,
        x_dot + config.dt * x_acc,
        theta + config.dt * theta_dot,
        theta_dot + config.dt * theta_acc
    )
```

**Benefits:**

- MPC planners can call `step_physics()` directly for trajectory optimization
- Test physics against analytical solutions
- Reuse across multiple tasks

## Layer B: Task wrapper

**Responsibility:** Wrap physics with task-specific logic.

**Defines:**

- What the agent observes (`get_obs()`)
- What the agent optimizes (reward function)
- When episodes end (termination conditions)

**Example: Control task from `cartpole/tasks/control.py`:**

```python
def _step(
    key: chex.PRNGKey,
    state: ControlTaskState,
    action: chex.Array,
    params: ControlTaskParams,
    config: ControlTaskConfig,
) -> Tuple[obs, next_state, reward, done, info]:
    # 1. Apply pure physics (Layer A)
    next_physics = step_physics(
        state.physics, action, params.physics, config.physics
    )

    # 2. Compute reward (task-specific)
    reward = jnp.float32(1.0)  # +1 per step

    # 3. Check termination (task-specific)
    done = check_termination(next_physics, t_next, config.task)

    # 4. Get observation (task-specific)
    obs = get_cartpole_obs(next_physics)

    return obs, next_state, reward, done, {}
```

**Same physics, different tasks:**

The CartPole physics is reused by two task wrappers:

| Task | File | Observation | Reward | Purpose |
|------|------|-------------|--------|---------|
| Control | `tasks/control.py` | `[x, ẋ, θ, θ̇]` | +1 per step | Standard balancing |
| SysID | `tasks/sysid.py` | `[x, θ, belief_μ, belief_σ]` | Fisher information | Parameter learning |

Both wrappers call the same `step_physics()` function from Layer A.

## Layer C: Learner

**Responsibility:** Standard RL algorithms that interact via the environment protocol.

**Key point:** The agent doesn't know whether it's solving control or system ID. It just sees:

```python
obs, state, reward, done, info = env.step(state, action, key)
```

**Implemented agents:**

- `dqn.py`: Deep Q-Network (discrete actions)
- `pqn.py`: Parametric Q-Network (continuous actions)
- `random.py`: Baseline

All agents work with all tasks automatically.

## The benefit: Modularity

```python
from aion.envs.cartpole import physics

# Task 1: Balancing (control)
control_env = make_env("cartpole-control")
dqn_agent = make_agent("dqn", ...)
metrics = train(control_env, dqn_agent)

# Task 2: Parameter learning (SysID)
sysid_env = make_env("cartpole-sysid")
pqn_agent = make_agent("pqn", ...)
metrics = train(sysid_env, pqn_agent)

# Task 3: Model-based planning (direct physics access)
def mpc_planner(state, horizon=10):
    """Use physics directly for trajectory optimization."""
    best_actions = optimize_trajectory(
        physics_fn=physics.step_physics,
        initial_state=state,
        horizon=horizon
    )
    return best_actions[0]
```

## File organization

```
src/aion/envs/cartpole/
├── physics.py          # Layer A: Pure dynamics
└── tasks/
    ├── control.py      # Layer B: Control wrapper
    └── sysid.py        # Layer B: SysID wrapper

src/aion/agents/
├── dqn.py              # Layer C: DQN agent
├── pqn.py              # Layer C: PQN agent
└── random.py           # Layer C: Random agent
```

## Protocol-based design

Layers communicate via Python Protocols (structural typing), not inheritance:

```python
# Layer B must implement this protocol
class Environment(Protocol):
    def reset(self, state, key, config): ...
    def step(self, key, state, action, params, config): ...
    def get_obs(self, state, config): ...

# Layer C must implement this protocol
class Agent(Protocol):
    def select_action(self, params, obs, key): ...
    def update(self, params, transition): ...
```

This allows maximum implementation flexibility while maintaining type safety.

## A concrete example: Gene circuit control

The gene circuit environment (`ccas_ccar_v1`) demonstrates why this separation is valuable for messy real-world systems.

**The physics layer** implements:

- Protein expression dynamics (ODEs)
- Cell growth mechanics
- Stochastic division events

**Multiple task wrappers** reuse this physics:

- **Control task**: Keep protein expression at a target level
- **SysID task**: Infer unknown reaction rates from observations
- **Active learning task**: Design experiments to maximize information gain

**Any agent** works with any task automatically. The gene circuit uses the same DQN and PQN agents as CartPole—no special modifications required.

If you can cleanly separate these concerns for a system with multi-timescale dynamics, stochastic behavior, and 10+ unknown parameters, you can handle your system too.

## Next steps

- [Custom Environment Guide](custom_env.md): Implement your own three-layer environment
- [Running Experiments](running_experiments.md): Train agents on your tasks
- [Custom Agent Guide](custom_agent.md): Implement learning algorithms
