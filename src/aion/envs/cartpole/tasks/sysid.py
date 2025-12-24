"""System Identification task wrapper for CartPole.

SysID task: Learn to identify unknown physics parameters through active exploration.

Key differences from control task:
- Physics parameters (pole_mass, pole_length) are randomized per episode
- Reward encourages information-seeking behavior (e.g., rapid state changes)
- Agent maintains internal belief state and learns parameter estimation
- Environment provides true parameters in info dict for meta-learning
"""

from dataclasses import replace
from typing import Any, Dict, NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from aion.core.spaces import Discrete
from aion.envs.environment import Environment

from ..physics import PhysicsConfig, PhysicsParams, PhysicsState, create_physics_params, step_physics
from .base import (
    TaskConfig,
    check_termination,
    get_cartpole_action_space,
    get_cartpole_obs,
    get_cartpole_obs_shape,
    sample_initial_physics,
)


class SysIDTaskState(NamedTuple):
    """State for the SysID task.

    Simple state: just physics + timestep (NO belief tracking).
    The agent (Layer C) handles belief updates internally.

    Attributes:
        physics: The underlying physics state (x, x_dot, theta, theta_dot)
        t: Current timestep counter
    """

    physics: PhysicsState
    t: chex.Array


@struct.dataclass
class SysIDTaskConfig:
    """Configuration for the CartPole SysID task."""

    physics: PhysicsConfig = struct.field(default_factory=PhysicsConfig)
    task: TaskConfig = struct.field(default_factory=TaskConfig)

    # Reward shaping for information-seeking behavior
    reward_type: str = "state_change"  # "state_change", "action_diversity", "sparse"
    reward_scale: float = 1.0

    # Domain randomization ranges for unknown parameters
    pole_mass_min: float = 0.05
    pole_mass_max: float = 0.15
    pole_length_min: float = 0.3
    pole_length_max: float = 0.7

    @property
    def max_steps(self) -> int:
        """Required by EnvironmentConfig protocol."""
        return self.task.max_steps


@struct.dataclass
class SysIDTaskParams:
    """Parameters for the SysID task.

    These parameters are randomized per episode to create diverse physics.
    """

    physics: PhysicsParams = struct.field(default_factory=PhysicsParams)

    # Randomized physics parameters (the "unknown" parameters to estimate)
    # These vary per episode to force agent to learn active identification
    # Note: Using float | chex.Array for JAX compatibility (runtime values are Arrays)
    pole_mass: float | chex.Array = 0.1  # Will be randomized in [pole_mass_min, pole_mass_max]
    pole_length: float | chex.Array = 0.5  # Will be randomized in [pole_length_min, pole_length_max]


def _step(
    key: chex.PRNGKey,
    state: SysIDTaskState,
    action: chex.Array,
    params: SysIDTaskParams,
    config: SysIDTaskConfig,
) -> Tuple[chex.Array, SysIDTaskState, chex.Array, chex.Array, Dict[str, Any]]:
    """Step the SysID task forward one timestep.

    Args:
        key: RNG key (unused for deterministic physics, but part of protocol)
        state: Current task state
        action: Discrete action {0, 1}
        params: Task parameters (includes randomized pole_mass, pole_length)
        config: Task configuration (static)

    Returns:
        obs_next: Next observation [x, x_dot, theta, theta_dot]
        next_state: Next task state
        reward: Information-seeking reward
        done: Termination flag (1.0 if done, 0.0 otherwise)
        info: Dict with true physics parameters for meta-learning
    """
    # Step the pure physics (using randomized params)
    # Create modified config with randomized parameters
    physics_config_with_params = replace(
        config.physics,
        pole_mass=params.pole_mass,
        pole_length=params.pole_length,
    )

    next_physics = step_physics(state.physics, action, params.physics, physics_config_with_params)

    # Increment timestep
    t_next = state.t + 1

    # Check termination
    done = check_termination(next_physics, t_next, config.task)

    # Compute reward (proxy for information content)
    reward = compute_sysid_reward(state.physics, next_physics, action, config)

    # Create next state
    next_state = SysIDTaskState(physics=next_physics, t=t_next)

    # Extract observation (same as control task - pure physics)
    obs_next = get_obs(next_state, params, config)

    # Include physics params in info (agent can use for meta-learning)
    info = {
        "true_pole_mass": params.pole_mass,
        "true_pole_length": params.pole_length,
    }

    return obs_next, next_state, reward, done, info


def _reset(
    key: chex.PRNGKey,
    params: SysIDTaskParams,
    config: SysIDTaskConfig,
) -> Tuple[chex.Array, SysIDTaskState]:
    """Reset the SysID task to initial state.

    Initializes the pole with small random perturbations and randomizes physics parameters.

    Args:
        key: RNG key for random initialization
        params: Task parameters (will use randomized pole_mass, pole_length)
        config: Task configuration (static)

    Returns:
        obs: Initial observation
        state: Initial task state

    Note:
        Parameter randomization happens in make_env() when creating params.
        This function uses the pre-randomized params.
    """
    # Sample initial physics state with small random perturbations
    physics = sample_initial_physics(key)

    state = SysIDTaskState(physics=physics, t=jnp.array(0))
    obs = get_obs(state, params, config)

    return obs, state


def compute_sysid_reward(
    prev_state: PhysicsState,
    next_state: PhysicsState,
    action: chex.Array,
    config: SysIDTaskConfig,
) -> chex.Array:
    """Compute reward proxy for information content.

    Different reward types encourage different exploration strategies:
    - "state_change": Magnitude of state change (rapid dynamics = informative)
    - "action_diversity": Encourage diverse action sequences (placeholder)
    - "sparse": No intermediate reward (only terminal reward if agent estimates correctly)

    Args:
        prev_state: Previous physics state
        next_state: Next physics state
        action: Action taken (unused in current implementation)
        config: Task configuration

    Returns:
        reward: Scalar reward value
    """
    if config.reward_type == "state_change":
        # Euclidean distance in state space
        # Larger state changes indicate more informative transitions
        state_diff = jnp.array(
            [
                next_state.x - prev_state.x,
                next_state.x_dot - prev_state.x_dot,
                next_state.theta - prev_state.theta,
                next_state.theta_dot - prev_state.theta_dot,
            ]
        )
        reward = jnp.linalg.norm(state_diff) * config.reward_scale

    elif config.reward_type == "action_diversity":
        # Reward based on action entropy (encourage exploration)
        # Note: This is a placeholder - real implementation would track action history
        reward = 1.0 * config.reward_scale

    else:  # sparse
        # No intermediate reward - agent must use internal belief for guidance
        reward = 0.0

    return reward


# JIT the step and reset functions with config as static argument
step = jax.jit(_step, static_argnames=["config"])
reset = jax.jit(_reset, static_argnames=["config"])


def get_obs(
    state: SysIDTaskState,
    params: SysIDTaskParams,
    config: SysIDTaskConfig,
) -> chex.Array:
    """Extract observation from state.

    For SysID task, observation is pure physics (same as control task).
    Agent receives [x, x_dot, theta, theta_dot] and learns to infer parameters from dynamics.

    Args:
        state: Current task state
        params: Task parameters (unused)
        config: Task configuration (unused)

    Returns:
        Observation array of shape (4,)
    """
    return get_cartpole_obs(state.physics)


def get_obs_shape(config: SysIDTaskConfig) -> Tuple[int, ...]:
    """Get the shape of the observation space.

    Args:
        config: Task configuration (unused)

    Returns:
        Observation shape tuple (same as control task)
    """
    return get_cartpole_obs_shape()


def get_action_space(config: SysIDTaskConfig) -> Discrete:
    """Get the discrete action space for the environment.

    Args:
        config: Task configuration (unused)

    Returns:
        Discrete space with 2 actions: 0 (push left) and 1 (push right)
    """
    return get_cartpole_action_space()


def create_randomized_params(key: chex.PRNGKey, config: SysIDTaskConfig) -> SysIDTaskParams:
    """Create randomized task parameters for domain randomization.

    Samples physics parameters from uniform distributions defined in config.

    Args:
        key: RNG key for randomization
        config: Task configuration with randomization ranges

    Returns:
        SysIDTaskParams with randomized pole_mass and pole_length
    """
    key_mass, key_length = jax.random.split(key)

    pole_mass = jax.random.uniform(
        key_mass,
        minval=config.pole_mass_min,
        maxval=config.pole_mass_max,
    )

    pole_length = jax.random.uniform(
        key_length,
        minval=config.pole_length_min,
        maxval=config.pole_length_max,
    )

    return SysIDTaskParams(
        physics=create_physics_params(),
        pole_mass=pole_mass,
        pole_length=pole_length,
    )


def make_env(
    config: SysIDTaskConfig | None = None,
    params: SysIDTaskParams | None = None,
    **kwargs,
) -> Environment[SysIDTaskState, SysIDTaskParams, SysIDTaskConfig]:
    """Create a CartPole SysID task environment.

    Args:
        config: Custom SysIDTaskConfig. If None, uses defaults.
        params: Custom SysIDTaskParams. If None, creates with nominal values.
                For randomization, use create_randomized_params() instead.
        **kwargs: Keyword arguments for creating config if not provided.

    Returns:
        Environment instance for the SysID task

    Note:
        This creates an environment with nominal (non-randomized) parameters.
        For domain randomization, call create_randomized_params() with a random key
        and pass the result as params to env.reset().
    """
    if config is None:
        # Parse kwargs into nested config structure
        physics_fields = {"gravity", "cart_mass", "pole_mass", "pole_length", "force_magnitude", "dt"}
        task_fields = {"max_steps", "theta_threshold", "x_threshold"}
        sysid_fields = {
            "reward_type",
            "reward_scale",
            "pole_mass_min",
            "pole_mass_max",
            "pole_length_min",
            "pole_length_max",
        }

        physics_kwargs = {k: v for k, v in kwargs.items() if k in physics_fields}
        task_kwargs = {k: v for k, v in kwargs.items() if k in task_fields}
        sysid_kwargs = {k: v for k, v in kwargs.items() if k in sysid_fields}

        config = SysIDTaskConfig(
            physics=PhysicsConfig(**physics_kwargs) if physics_kwargs else PhysicsConfig(),
            task=TaskConfig(**task_kwargs) if task_kwargs else TaskConfig(),
            **sysid_kwargs,
        )

    if params is None:
        # Use nominal parameters (not randomized)
        params = SysIDTaskParams(physics=create_physics_params())

    return Environment(
        step=step,
        reset=reset,
        get_action_space=get_action_space,
        get_obs_shape=get_obs_shape,
        params=params,
        config=config,
    )
