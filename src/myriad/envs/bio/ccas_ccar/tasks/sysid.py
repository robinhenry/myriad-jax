"""System Identification task wrapper for CcaS-CcaR gene circuit.

SysID task: Learn to identify unknown gene circuit parameters through active exploration.

Key differences from control task:
- Physics parameters (Kh, Kf, eta, etc.) are randomized per episode
- Reward encourages information-seeking behavior (e.g., state changes)
- Agent maintains internal belief state and learns parameter estimation
- Environment provides true parameters in info dict for meta-learning
"""

from dataclasses import replace
from typing import Any, Dict, NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from myriad.core.spaces import Discrete
from myriad.envs.environment import Environment

from ..physics import PhysicsConfig, PhysicsParams, PhysicsState, create_physics_params, step_physics
from .base import (
    CcasCcarSysIDObs,
    TaskConfig,
    check_termination,
    get_ccas_ccar_action_space,
    sample_initial_physics,
)


class SysIDTaskState(NamedTuple):
    """State for the SysID task.

    Simple state: just physics + timestep (NO belief tracking).
    The agent (Layer C) handles belief updates internally.

    Attributes:
        physics: The underlying physics state (time, H, F)
        t: Current timestep counter
    """

    physics: PhysicsState
    t: chex.Array


@struct.dataclass
class SysIDTaskConfig:
    """Configuration for the CcaS-CcaR SysID task."""

    physics: PhysicsConfig = struct.field(default_factory=PhysicsConfig)
    task: TaskConfig = struct.field(default_factory=TaskConfig)

    # Reward shaping for information-seeking behavior
    reward_type: str = "state_change"  # "state_change", "sparse"
    reward_scale: float = 1.0

    # Domain randomization ranges for unknown parameters
    # Hill coefficient for H (CcaSR cooperativity)
    Kh_min: float = 60.0
    Kh_max: float = 120.0

    # Hill coefficient for F (self-activation cooperativity)
    Kf_min: float = 20.0
    Kf_max: float = 40.0

    # Production rate
    eta_min: float = 0.5
    eta_max: float = 1.5

    # Promoter activity
    a_min: float = 0.5
    a_max: float = 1.5

    @property
    def max_steps(self) -> int:
        """Required by EnvironmentConfig protocol."""
        return self.task.max_steps


@struct.dataclass
class SysIDTaskParams:
    """Parameters for the SysID task.

    These parameters are randomized per episode to create diverse gene circuit dynamics.
    """

    physics: PhysicsParams = struct.field(default_factory=PhysicsParams)

    # Randomized physics parameters (the "unknown" parameters to estimate)
    # These vary per episode to force agent to learn active identification
    Kh: float | chex.Array = 90.0  # Will be randomized in [Kh_min, Kh_max]
    Kf: float | chex.Array = 30.0  # Will be randomized in [Kf_min, Kf_max]
    eta: float | chex.Array = 1.0  # Will be randomized in [eta_min, eta_max]
    a: float | chex.Array = 1.0  # Will be randomized in [a_min, a_max]


def _step(
    key: chex.PRNGKey,
    state: SysIDTaskState,
    action: chex.Array,
    params: SysIDTaskParams,
    config: SysIDTaskConfig,
) -> Tuple[CcasCcarSysIDObs, SysIDTaskState, chex.Array, chex.Array, Dict[str, Any]]:
    """Step the SysID task forward one timestep.

    Args:
        key: RNG key for stochastic physics simulation
        state: Current task state
        action: Discrete action {0, 1}
        params: Task parameters (includes randomized Kh, Kf, eta, a)
        config: Task configuration (static)

    Returns:
        obs_next: Next observation (CcasCcarSysIDObs with named fields)
        next_state: Next task state
        reward: Information-seeking reward
        done: Termination flag (1.0 if done, 0.0 otherwise)
        info: Dict with true physics parameters for meta-learning
    """
    # Step the pure physics (using randomized params)
    # Create modified config with randomized parameters
    physics_config_with_params = replace(
        config.physics,
        Kh=params.Kh,
        Kf=params.Kf,
        eta=params.eta,
        a=params.a,
    )

    next_physics = step_physics(key, state.physics, action, params.physics, physics_config_with_params)

    # Increment timestep
    t_next = state.t + 1

    # Check termination
    done = check_termination(t_next, config.task)

    # Compute reward (proxy for information content)
    reward = compute_sysid_reward(state.physics, next_physics, action, config)

    # Create next state
    next_state = SysIDTaskState(physics=next_physics, t=t_next)

    # Extract observation (pure physics, no target in SysID)
    obs_next = get_obs(next_state, params, config)

    # Include physics params in info (agent can use for meta-learning)
    info = {
        "true_Kh": params.Kh,
        "true_Kf": params.Kf,
        "true_eta": params.eta,
        "true_a": params.a,
        "F": next_physics.F,
        "H": next_physics.H,
    }

    return obs_next, next_state, reward, done, info


def _reset(
    key: chex.PRNGKey,
    params: SysIDTaskParams,
    config: SysIDTaskConfig,
) -> Tuple[CcasCcarSysIDObs, SysIDTaskState]:
    """Reset the SysID task to initial state.

    Initializes the system at zero protein concentrations with randomized physics parameters.

    Args:
        key: RNG key for random initialization
        params: Task parameters (will use randomized Kh, Kf, eta, a)
        config: Task configuration (static)

    Returns:
        obs: Initial observation (CcasCcarSysIDObs with named fields)
        state: Initial task state

    Note:
        Parameter randomization happens in make_env() when creating params.
        This function uses the pre-randomized params.
    """
    # Sample initial physics state (zero concentrations)
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
        # Euclidean distance in state space (H and F)
        # Larger state changes indicate more informative transitions
        state_diff = jnp.array(
            [
                next_state.H - prev_state.H,
                next_state.F - prev_state.F,
            ]
        )
        reward = jnp.linalg.norm(state_diff) * config.reward_scale

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
) -> CcasCcarSysIDObs:
    """Extract observation from state.

    Returns a structured observation with named fields for semantic access.
    Agent receives F measurements and learns to infer parameters from dynamics.

    No target is provided in SysID (agent must explore without task goal).

    Args:
        state: Current task state
        params: Task parameters (unused)
        config: Task configuration

    Returns:
        CcasCcarSysIDObs with named fields (F_normalized, U_obs, padding)
    """
    # Normalize F by observation normalizer
    F_normalized = state.physics.F / config.task.F_obs_normalizer

    # U is set to 0 in observation (agent doesn't directly observe light input)
    U_obs = jnp.array(0.0)

    # No target in SysID
    padding = jnp.array(0.0)

    return CcasCcarSysIDObs(
        F_normalized=F_normalized,
        U_obs=U_obs,
        padding=padding,
    )


def get_obs_shape(config: SysIDTaskConfig) -> Tuple[int, ...]:
    """Get the shape of the observation space.

    For SysID: [F, U, 0] = 3 elements (simpler than control task)

    Args:
        config: Task configuration (unused)

    Returns:
        Observation shape tuple (3,)
    """
    return (3,)


def get_action_space(config: SysIDTaskConfig) -> Discrete:
    """Get the discrete action space for the environment.

    Args:
        config: Task configuration (unused)

    Returns:
        Discrete space with 2 actions: 0 (light off) and 1 (light on)
    """
    return get_ccas_ccar_action_space()


def sample_randomized_params(key: chex.PRNGKey, config: SysIDTaskConfig) -> SysIDTaskParams:
    """Sample randomized physics parameters for a new episode.

    Args:
        key: RNG key for randomization
        config: Task configuration with randomization ranges

    Returns:
        SysIDTaskParams with randomized Kh, Kf, eta, a
    """
    keys = jax.random.split(key, 4)

    Kh = jax.random.uniform(keys[0], minval=config.Kh_min, maxval=config.Kh_max)
    Kf = jax.random.uniform(keys[1], minval=config.Kf_min, maxval=config.Kf_max)
    eta = jax.random.uniform(keys[2], minval=config.eta_min, maxval=config.eta_max)
    a = jax.random.uniform(keys[3], minval=config.a_min, maxval=config.a_max)

    return SysIDTaskParams(
        physics=create_physics_params(),
        Kh=Kh,
        Kf=Kf,
        eta=eta,
        a=a,
    )


def make_env(
    config: SysIDTaskConfig | None = None,
    params: SysIDTaskParams | None = None,
    **kwargs,
) -> Environment[SysIDTaskState, SysIDTaskParams, SysIDTaskConfig]:
    """Create a CcaS-CcaR SysID task environment.

    Args:
        config: Custom SysIDTaskConfig. If None, uses defaults.
        params: Custom SysIDTaskParams. If None, creates from kwargs.
        **kwargs: Keyword arguments for creating config/params if not provided.

    Returns:
        Environment instance for the SysID task

    Example:
        # Default SysID task
        env = make_env()

        # Custom randomization ranges
        env = make_env(
            Kh_min=60.0,
            Kh_max=120.0,
            reward_type="state_change",
            reward_scale=0.1,
        )
    """
    if config is None:
        # Parse kwargs into nested config structure
        physics_fields = {"eta", "nu", "a", "Kh", "nh", "Kf", "nf", "timestep_minutes", "max_gillespie_steps"}
        task_fields = {"max_steps", "F_obs_normalizer"}
        sysid_fields = {
            "reward_type",
            "reward_scale",
            "Kh_min",
            "Kh_max",
            "Kf_min",
            "Kf_max",
            "eta_min",
            "eta_max",
            "a_min",
            "a_max",
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
        # For SysID, we don't randomize params at creation time
        # They will be randomized during reset in the runner
        params = SysIDTaskParams(physics=create_physics_params())

    return Environment(
        step=step,
        reset=reset,
        get_action_space=get_action_space,
        get_obs_shape=get_obs_shape,
        params=params,
        config=config,
    )
