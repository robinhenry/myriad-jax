"""Control task wrapper for CcaS-CcaR + GFP gene circuit.

Standard tracking task: Control GFP expression (F) to match a target trajectory.
Reward: Negative absolute error between F and F_target.

Task variants:
- Constant target: Fixed GFP level (default: F_target = 25)
- Sinewave target: Time-varying sinusoidal trajectory
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

from myriad.core.spaces import Discrete
from myriad.core.types import PRNGKey
from myriad.envs.environment import Environment
from myriad.utils import filter_kwargs

from ..physics import PhysicsConfig, PhysicsParams, PhysicsParamsPrior, PhysicsState
from .base import (
    BaseCcasrGfpTaskConfig,
    CcasrGfpControlObs,
    TaskConfig,
    check_termination,
    generate_constant_target,
    generate_sinewave_target,
    get_action_space as _get_action_space,
    sample_initial_physics,
    step_physics_interval,
)


class ControlTaskState(NamedTuple):
    """State for the control task.

    Attributes:
        physics: The underlying physics state (time, H, F)
        t: Current timestep counter (RL timesteps, not Gillespie time)
        U: Previous action (light input from last timestep, for action-toggle detection)
        F_target: Target trajectory for GFP expression [current, t+1, ..., t+n_horizon]
    """

    physics: PhysicsState
    t: Array
    U: Array
    F_target: Array


@struct.dataclass
class ControlTaskConfig(BaseCcasrGfpTaskConfig):
    """Configuration for the CcaS-CcaR control task."""

    # Target generation
    target_type: str = "constant"  # "constant" or "sinewave"
    n_horizon: int = 1  # Number of future timesteps to include in observation

    # Constant target parameters
    F_target_constant: float = 25.0

    # Sinewave target parameters
    sinewave_period_minutes: float = 600.0  # 10 hours
    sinewave_amplitude: float = 20.0
    sinewave_vshift: float = 30.0

    @property
    def dt(self) -> float:
        """Timestep duration in minutes."""
        return self.physics.timestep_minutes


@struct.dataclass
class ControlTaskParams:
    """Parameters for the control task."""

    physics: PhysicsParams = struct.field(default_factory=PhysicsParams)


@struct.dataclass
class ControlTaskParamsPrior:
    """Prior distribution over control task parameters."""

    physics: PhysicsParamsPrior = struct.field(default_factory=PhysicsParamsPrior)

    def sample(self, key: "PRNGKey") -> ControlTaskParams:
        return ControlTaskParams(physics=self.physics.sample(key))


def _step(
    key: PRNGKey,
    state: ControlTaskState,
    action: Array,
    params: ControlTaskParams,
    config: ControlTaskConfig,
) -> tuple[CcasrGfpControlObs, ControlTaskState, Array, Array, dict[str, Any]]:
    """Step the control task forward one timestep.

    Args:
        key: RNG key for stochastic physics simulation
        state: Current task state
        action: Discrete action {0, 1} for light input
        params: Task parameters
        config: Task configuration (static)

    Returns:
        obs_next: Next observation
        next_state: Next task state
        reward: Reward (negative absolute error)
        done: Termination flag (1.0 if done, 0.0 otherwise)
        info: Dict with current protein levels for logging
    """
    key_physics, key_target = jax.random.split(key)
    next_physics, t_next = step_physics_interval(
        key_physics, state.physics, state.t, state.U, action, params.physics, config.physics
    )

    # config.target_type is static — resolve at trace time with plain if/else
    if config.target_type == "sinewave":
        F_target_next = generate_sinewave_target(
            key_target,
            t_next,
            config.n_horizon,
            config.physics.timestep_minutes,
            config.sinewave_period_minutes,
            config.sinewave_amplitude,
            config.sinewave_vshift,
        )
    else:
        F_target_next = generate_constant_target(config.n_horizon, config.F_target_constant)

    # Check termination
    done = check_termination(t_next, config.task)

    # Compute reward (negative absolute error)
    # Use the current target (first element of F_target array)
    reward = -jnp.abs(next_physics.F - state.F_target[0])

    # Create next state (store current action as U for next step's toggle detection)
    next_state = ControlTaskState(physics=next_physics, t=t_next, U=action, F_target=F_target_next)

    # Extract observation
    obs_next = get_obs(next_state, params, config)

    # Info dict for logging
    info = {
        "F": next_physics.F,
        "H": next_physics.H,
        "F_target": state.F_target[0],
    }

    return obs_next, next_state, reward, done, info


def _reset(
    key: PRNGKey,
    params: ControlTaskParams,
    config: ControlTaskConfig,
) -> tuple[CcasrGfpControlObs, ControlTaskState]:
    """Reset the control task to initial state.

    Initializes the system at zero protein concentrations and generates initial target.

    Args:
        key: RNG key for random initialization
        params: Task parameters
        config: Task configuration (static)

    Returns:
        obs: Initial observation (CcasCcarControlObs with named fields)
        state: Initial task state
    """
    key_physics, key_target = jax.random.split(key)

    # Sample initial physics state (zero concentrations)
    physics = sample_initial_physics(key_physics)

    # config.target_type is static — resolve at trace time with plain if/else
    if config.target_type == "sinewave":
        F_target = generate_sinewave_target(
            key_target,
            jnp.array(0),
            config.n_horizon,
            config.physics.timestep_minutes,
            config.sinewave_period_minutes,
            config.sinewave_amplitude,
            config.sinewave_vshift,
        )
    else:
        F_target = generate_constant_target(config.n_horizon, config.F_target_constant)

    # Initialize U=0 (no light). First action toggle (if any) will reset time to 0.
    state = ControlTaskState(physics=physics, t=jnp.array(0), U=jnp.array(0), F_target=F_target)
    obs = get_obs(state, params, config)

    return obs, state


# JIT the step and reset functions with config as static argument
step = jax.jit(_step, static_argnames=["config"])
reset = jax.jit(_reset, static_argnames=["config"])


def get_obs(
    state: ControlTaskState,
    params: ControlTaskParams,
    config: ControlTaskConfig,
) -> CcasrGfpControlObs:
    """Extract observation from state.

    Returns a structured observation with named fields for semantic access
    by classical controllers. Neural network agents can call `.to_array()`.

    Args:
        state: Current task state
        params: Task parameters (unused)
        config: Task configuration

    Returns:
        CcasCcarControlObs with named fields (F_normalized, U_obs, F_target)
    """
    # Normalize F by observation normalizer
    F_normalized = state.physics.F / config.task.F_obs_normalizer

    # Normalize F_target
    F_target_normalized = state.F_target / config.task.F_obs_normalizer

    return CcasrGfpControlObs(
        F_normalized=F_normalized,
        F_target=F_target_normalized,
    )


def get_obs_shape(config: ControlTaskConfig) -> tuple[int, ...]:
    """Get the shape of the observation space.

    Observation: [F, F_target[0:n_horizon+1]]
    Shape: (1 + n_horizon + 1,)

    Args:
        config: Task configuration

    Returns:
        Observation shape tuple
    """
    return (1 + config.n_horizon + 1,)


def get_action_space(config: ControlTaskConfig) -> Discrete:
    """Get the discrete action space for the environment.

    Args:
        config: Task configuration (unused)

    Returns:
        Discrete space with 2 actions: 0 (light off) and 1 (light on)
    """
    return _get_action_space()


def make_env(
    config: ControlTaskConfig | None = None,
    params: ControlTaskParams | None = None,
    params_prior: ControlTaskParamsPrior | None = None,
    **kwargs,
) -> Environment[ControlTaskState, ControlTaskConfig, ControlTaskParams, CcasrGfpControlObs]:
    """Create a Ccasr-gfp control task environment.

    Args:
        config: Custom ControlTaskConfig. If None, uses defaults.
        params: Custom ControlTaskParams. If None, creates from kwargs.
        params_prior: Optional prior for domain randomization. If set,
            ``env.sample_params_fn`` will sample distinct θ* per parallel env.
            Can also be triggered via flat kwargs (e.g. ``nu_scale=0.3``).
        **kwargs: Keyword arguments for creating config/params if not provided.

    Returns:
        Environment instance for the control task

    Example:
        >>> # Constant target at F=25
        >>> env = make_env(F_target_constant=25.0)
        >>> # Domain randomization
        >>> env = make_env(nu_scale=0.3, Kh_scale=0.2)
    """
    if config is None:
        # Distribute flat kwargs to the appropriate nested config dataclass.
        # filter_kwargs introspects dataclass fields, so routing stays in sync
        # automatically when fields are added or removed.
        control_kwargs = {
            k: v
            for k, v in filter_kwargs(kwargs, ControlTaskConfig).items()
            if k not in {"physics", "task"}  # nested — handled separately below
        }
        config = ControlTaskConfig(
            physics=PhysicsConfig(**filter_kwargs(kwargs, PhysicsConfig)),
            task=TaskConfig(**filter_kwargs(kwargs, TaskConfig)),
            **control_kwargs,
        )

    if params is None:
        params = ControlTaskParams(physics=PhysicsParams(**filter_kwargs(kwargs, PhysicsParams)))

    if params_prior is None:
        prior_kwargs = filter_kwargs(kwargs, PhysicsParamsPrior)
        if prior_kwargs:
            params_prior = ControlTaskParamsPrior(physics=PhysicsParamsPrior(**prior_kwargs))

    sample_fn = params_prior.sample if params_prior is not None else None

    return Environment(
        step=step,
        reset=reset,
        get_action_space=get_action_space,
        get_obs_shape=get_obs_shape,
        params=params,
        config=config,
        sample_params_fn=sample_fn,
    )
