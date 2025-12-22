import logging
from typing import Any, Dict, NamedTuple

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from flax import struct

from aion.core import spaces

from .environment import Environment

logger = logging.getLogger(__name__)


@struct.dataclass
class EnvConfig:
    n_horizon: int = 1
    max_steps: int = 12 * 24


@struct.dataclass
class EnvParams:
    timestep_minutes: float = 5.0
    F_obs_normalizer: float = 1.0


class EnvState(NamedTuple):
    time: Array
    U: Array
    H: Array
    F: Array
    F_target: Array
    step: Array


def _step(
    key: PRNGKey, state: EnvState, action: Array, params: EnvParams, config: EnvConfig
) -> tuple[Array, EnvState, Array, Array, Dict[str, Any]]:
    """Step the environment forward one time step."""

    # TODO: implement state update
    next_state = state._replace(step=state.step + 1)

    next_obs = get_obs(next_state, params, config)
    reward = get_reward(next_state, params, config)
    done = next_state.step >= config.max_steps
    info = {}
    return next_obs, next_state, reward, done, info


def _reset(key: PRNGKey, params: EnvParams, config: EnvConfig) -> tuple[Array, EnvState]:
    """Reset the environment to initial state."""
    state = EnvState(
        time=jnp.array(0.0),
        U=jnp.array(0),
        H=jnp.array(0),
        F=jnp.array(0),
        F_target=jnp.zeros(config.n_horizon + 1),
        step=jnp.array(0),
    )
    obs = get_obs(state, params, config)
    return obs, state


step = jax.jit(_step, static_argnames=["config"])
reset = jax.jit(_reset, static_argnames=["config"])


def get_obs(state: EnvState, params: EnvParams, config: EnvConfig) -> Array:
    base_obs = jnp.array([state.F / params.F_obs_normalizer, state.U])
    F_target = state.F_target / params.F_obs_normalizer
    return jnp.concatenate([base_obs, F_target], axis=0)


def get_reward(state: EnvState, params: EnvParams, config: EnvConfig) -> Array:
    return -jnp.abs(state.F - state.F_target)  # type: ignore


def get_obs_shape(config: EnvConfig) -> tuple:
    """Get the size of the observation space."""
    return (2 + config.n_horizon + 1,)


def get_action_space(config: EnvConfig) -> spaces.Discrete:
    """Get the continuous action space for the environment."""
    return spaces.Discrete(2)


def make_env(
    config: EnvConfig | None = None,
    params: EnvParams | None = None,
    **kwargs,
) -> Environment[EnvState, EnvParams, EnvConfig]:

    # If no config is provided, create a default one
    if config is None:
        config = EnvConfig(**kwargs)

    # If no params object is provided, create one from kwargs
    if params is None:
        params = EnvParams(**kwargs)
    elif kwargs:
        logger.warning("`params` object was provided, so keyword arguments %s will be ignored.", list(kwargs.keys()))

    return Environment(
        step=step,
        reset=reset,
        get_action_space=get_action_space,
        get_obs_shape=get_obs_shape,
        params=params,
        config=config,
    )
