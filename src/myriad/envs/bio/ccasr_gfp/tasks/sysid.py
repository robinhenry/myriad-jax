"""System identification task for the CcaS-CcaR + GFP gene circuit.

The circuit is run with unknown kinetic parameters θ* stored in SysIdTaskParams.
The agent observes GFP fluorescence F(t) and selects light inputs U ∈ {0, 1}.
One episode = one experimental day (288 steps × 5 min = 24 h).

Between episodes the cell state resets (H=0, F=0), but θ* is fixed — it is a
property of the circuit, not the cell. This maps to the myriad convention where
params persist across episodes while state resets.

Reward is zero; the inference algorithm is the agent, and its objective
(information gain or posterior entropy) is computed outside the environment.
"""

from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

from myriad.core.spaces import Discrete
from myriad.core.types import PRNGKey
from myriad.envs.environment import Environment
from myriad.utils import filter_kwargs

from ..physics import PhysicsConfig, PhysicsParams, PhysicsState
from .base import (
    BaseCcasrGfpTaskConfig,
    TaskConfig,
    check_termination,
    get_action_space as _get_action_space,
    sample_initial_physics,
    step_physics_interval,
)


class SysIdTaskState(NamedTuple):
    """State for the system identification task.

    Attributes:
        physics: Underlying stochastic physics state (time, H, F, next_reaction_time)
        t: Timestep counter within the current episode (0 to max_steps)
        U: Previous light input — needed by step_physics for pending-reaction invalidation
    """

    physics: PhysicsState
    t: Array
    U: Array


@struct.dataclass
class SysIdTaskConfig(BaseCcasrGfpTaskConfig):
    """Static configuration for the SysID task."""


@struct.dataclass
class SysIdTaskParams:
    """Dynamic parameters for the SysID task — the unknown circuit parameters θ*.

    These are vmappable: pass different SysIdTaskParams per-env to simulate
    a population of cells each with its own kinetic parameters (Stage 2+).

    Attributes:
        physics: Kinetic parameters (nu, Kh, nh, Kf, nf) — the targets of inference
    """

    physics: PhysicsParams = struct.field(default_factory=PhysicsParams)


class SysIdObs(NamedTuple):
    """Observation for the SysID task.

    Only GFP fluorescence is observable; CcaSR (H) is a hidden species.

    Attributes:
        F_normalized: GFP count divided by F_obs_normalizer
    """

    F_normalized: Array

    def to_array(self) -> Array:
        return jnp.array([self.F_normalized])

    @classmethod
    def from_array(cls, arr: Array) -> "SysIdObs":
        chex.assert_shape(arr, (1,))
        return cls(F_normalized=arr[0])  # type: ignore


def get_obs(state: SysIdTaskState, config: SysIdTaskConfig) -> SysIdObs:
    return SysIdObs(F_normalized=state.physics.F / config.task.F_obs_normalizer)


def _reset(
    key: PRNGKey,
    params: SysIdTaskParams,
    config: SysIdTaskConfig,
) -> tuple[SysIdObs, SysIdTaskState]:
    """Reset to initial cell state (H=0, F=0).

    θ* is unchanged — it lives in params and persists across resets.
    """
    physics = sample_initial_physics(key)
    state = SysIdTaskState(physics=physics, t=jnp.array(0), U=jnp.array(0))
    return get_obs(state, config), state


def _step(
    key: PRNGKey,
    state: SysIdTaskState,
    action: Array,
    params: SysIdTaskParams,
    config: SysIdTaskConfig,
) -> tuple[SysIdObs, SysIdTaskState, Array, Array, dict[str, Any]]:
    """Step the circuit forward one 5-minute interval using θ* from params.

    Args:
        key: RNG key for Gillespie simulation
        state: Current task state
        action: Light input U ∈ {0, 1}
        params: Task parameters carrying θ* (the unknown circuit parameters)
        config: Static task configuration

    Returns:
        obs: Normalized F observation
        next_state: Updated task state
        reward: Always 0.0 (inference objective is external)
        done: 1.0 when t reaches max_steps
        info: Dict with raw F and H values for logging
    """
    next_physics, t_next = step_physics_interval(
        key, state.physics, state.t, state.U, action, params.physics, config.physics
    )
    done = check_termination(t_next, config.task)
    next_state = SysIdTaskState(physics=next_physics, t=t_next, U=action)
    obs = get_obs(next_state, config)
    info = {"F": next_physics.F, "H": next_physics.H}
    return obs, next_state, jnp.array(0.0), done, info


step = jax.jit(_step, static_argnames=["config"])
reset = jax.jit(_reset, static_argnames=["config"])


def get_obs_shape(config: SysIdTaskConfig) -> tuple[int, ...]:
    return (1,)


def get_action_space(config: SysIdTaskConfig) -> Discrete:
    return _get_action_space()


def make_env(
    config: SysIdTaskConfig | None = None,
    params: SysIdTaskParams | None = None,
    **kwargs,
) -> Environment[SysIdTaskState, SysIdTaskConfig, SysIdTaskParams, SysIdObs]:
    """Create a CcaS-CcaR system identification environment.

    Args:
        config: Static task config. If None, built from kwargs.
        params: Task params carrying θ*. If None, uses PhysicsParams defaults.
        **kwargs: Forwarded to SysIdTaskConfig/PhysicsConfig/TaskConfig construction.

    Returns:
        Environment instance for the SysID task.

    Example:
        >>> # Default ground-truth parameters
        >>> env = make_env()
        >>> # Custom θ* for a specific cell
        >>> from myriad.envs.bio.ccasr_gfp.physics import PhysicsParams
        >>> env = make_env(params=SysIdTaskParams(physics=PhysicsParams(nu=0.02, Kh=80.0)))
    """
    if config is None:
        config = SysIdTaskConfig(
            physics=PhysicsConfig(**filter_kwargs(kwargs, PhysicsConfig)),
            task=TaskConfig(**filter_kwargs(kwargs, TaskConfig)),
        )

    if params is None:
        params = SysIdTaskParams(physics=PhysicsParams(**filter_kwargs(kwargs, PhysicsParams)))

    return Environment(
        step=step,
        reset=reset,
        get_action_space=get_action_space,
        get_obs_shape=get_obs_shape,
        params=params,
        config=config,
    )
