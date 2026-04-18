"""System identification task for the 1D optogenetic circuit (opto_hill_1d).

The circuit runs with unknown kinetic parameters θ* = (k_prod, K, n, k_deg) stored
in SysIdTaskParams. The agent observes the fluorescent protein copy number X(t)
and selects continuous light intensities U(t) ∈ [0, 1]. Between episodes the cell
state resets (X=0), but θ* is fixed — it is a property of the circuit, not the cell.

Reward is zero; the inference algorithm is the agent, and its objective
(information gain or posterior entropy) is computed outside the environment.
"""

from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

from myriad.core.spaces import Box
from myriad.core.types import PRNGKey
from myriad.envs.environment import Environment
from myriad.utils import filter_kwargs

from ..physics import (
    PhysicsConfig,
    PhysicsParams,
    PhysicsParamsPrior,
    PhysicsState,
    step_physics,
)


class SysIdTaskState(NamedTuple):
    """State for the opto_hill_1d system identification task.

    Attributes:
        physics: Underlying stochastic physics state (time, X, next_reaction_time)
        t: Timestep counter within the current episode (0 to max_steps)
        U: Previous continuous light input — used by step_physics for
            pending-reaction invalidation when the action changes.
    """

    physics: PhysicsState
    t: Array
    U: Array


@struct.dataclass
class TaskConfig:
    """Task-level configuration (episode length, observation normalization)."""

    max_steps: int = 288  # 288 × 5 min = 24 h
    X_obs_normalizer: float = 100.0


@struct.dataclass
class SysIdTaskConfig:
    """Static configuration for the opto_hill_1d SysID task."""

    physics: PhysicsConfig = struct.field(default_factory=PhysicsConfig)
    task: TaskConfig = struct.field(default_factory=TaskConfig)

    @property
    def max_steps(self) -> int:
        """Required by EnvironmentConfig protocol."""
        return self.task.max_steps


@struct.dataclass
class SysIdTaskParams:
    """Dynamic parameters for the SysID task — the unknown circuit parameters θ*.

    These are vmappable: pass different SysIdTaskParams per-env to simulate
    a population of cells each with its own kinetic parameters.

    Attributes:
        physics: Kinetic parameters (k_prod, K, n, k_deg) — targets of inference
    """

    physics: PhysicsParams = struct.field(default_factory=PhysicsParams)


@struct.dataclass
class SysIdTaskParamsPrior:
    """Prior distribution over SysID task parameters."""

    physics: PhysicsParamsPrior = struct.field(default_factory=PhysicsParamsPrior)

    def sample(self, key: PRNGKey) -> SysIdTaskParams:
        return SysIdTaskParams(physics=self.physics.sample(key))


class SysIdObs(NamedTuple):
    """Observation for the opto_hill_1d SysID task.

    Only the fluorescent protein copy number is observable.

    Attributes:
        X_normalized: X count divided by X_obs_normalizer
    """

    X_normalized: Array

    def to_array(self) -> Array:
        return jnp.array([self.X_normalized])

    @classmethod
    def from_array(cls, arr: Array) -> "SysIdObs":
        chex.assert_shape(arr, (1,))
        return cls(X_normalized=arr[0])  # type: ignore


def get_obs(state: SysIdTaskState, config: SysIdTaskConfig) -> SysIdObs:
    return SysIdObs(X_normalized=state.physics.X / config.task.X_obs_normalizer)


def _sample_initial_physics(key: PRNGKey) -> PhysicsState:
    """Start from zero protein at time 0 (unused key kept for symmetry)."""
    del key
    return PhysicsState.create(time=jnp.array(0.0), X=jnp.array(0.0))


def _reset(
    key: PRNGKey,
    params: SysIdTaskParams,
    config: SysIdTaskConfig,
) -> tuple[SysIdObs, SysIdTaskState]:
    """Reset to initial cell state (X=0). θ* is unchanged — it lives in params."""
    del params  # unused on reset
    physics = _sample_initial_physics(key)
    state = SysIdTaskState(
        physics=physics,
        t=jnp.array(0),
        U=jnp.array(0.0, dtype=jnp.float32),
    )
    return get_obs(state, config), state


def _step(
    key: PRNGKey,
    state: SysIdTaskState,
    action: Array,
    params: SysIdTaskParams,
    config: SysIdTaskConfig,
) -> tuple[SysIdObs, SysIdTaskState, Array, Array, dict[str, Any]]:
    """Step the circuit forward one interval using θ* from params.

    Args:
        key: RNG key for Gillespie simulation
        state: Current task state
        action: Continuous light intensity U ∈ [0, 1] (scalar)
        params: Task parameters carrying θ*
        config: Static task configuration

    Returns:
        obs, next_state, reward (always 0.0), done, info
    """
    interval_start = state.t * config.physics.timestep_minutes
    next_physics = step_physics(
        key,
        state.physics,
        action,
        params.physics,
        config.physics,
        previous_action=state.U,
        interval_start=interval_start,
    )
    t_next = state.t + 1
    done = (t_next >= config.task.max_steps).astype(jnp.float32)
    next_state = SysIdTaskState(physics=next_physics, t=t_next, U=action)
    obs = get_obs(next_state, config)
    info = {"X": next_physics.X, "U": action}
    return obs, next_state, jnp.array(0.0), done, info


step = jax.jit(_step, static_argnames=["config"])
reset = jax.jit(_reset, static_argnames=["config"])


def get_obs_shape(config: SysIdTaskConfig) -> tuple[int, ...]:
    del config
    return (1,)


def get_action_space(config: SysIdTaskConfig) -> Box:
    """Continuous light intensity U ∈ [0, 1]."""
    del config
    return Box(low=0.0, high=1.0, shape=(), dtype=jnp.float32)


def make_env(
    config: SysIdTaskConfig | None = None,
    params: SysIdTaskParams | None = None,
    params_prior: SysIdTaskParamsPrior | None = None,
    **kwargs,
) -> Environment[SysIdTaskState, SysIdTaskConfig, SysIdTaskParams, SysIdObs]:
    """Create an opto_hill_1d system identification environment.

    Args:
        config: Static task config. If None, built from kwargs.
        params: Task params carrying θ*. If None, uses PhysicsParams defaults.
        params_prior: Optional prior for domain randomization. If set,
            ``env.sample_params_fn`` will sample distinct θ* per parallel env.
            Can also be triggered via flat kwargs (e.g. ``k_prod_scale=0.3``).
        **kwargs: Forwarded to SysIdTaskConfig / PhysicsConfig / TaskConfig /
            PhysicsParams / PhysicsParamsPrior via filter_kwargs.

    Returns:
        Environment instance for the SysID task.

    Example:
        >>> env = make_env()
        >>> env = make_env(k_prod_scale=0.3, K_scale=0.2)
    """
    if config is None:
        config = SysIdTaskConfig(
            physics=PhysicsConfig(**filter_kwargs(kwargs, PhysicsConfig)),
            task=TaskConfig(**filter_kwargs(kwargs, TaskConfig)),
        )

    if params is None:
        params = SysIdTaskParams(physics=PhysicsParams(**filter_kwargs(kwargs, PhysicsParams)))

    if params_prior is None:
        prior_kwargs = filter_kwargs(kwargs, PhysicsParamsPrior)
        if prior_kwargs:
            params_prior = SysIdTaskParamsPrior(physics=PhysicsParamsPrior(**prior_kwargs))

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
