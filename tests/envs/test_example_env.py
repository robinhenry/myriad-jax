from typing import cast

import chex
import jax.numpy as jnp
import pytest

from aion.core import spaces
from aion.envs.environment import Environment
from aion.envs.example import EnvConfig, EnvParams, EnvState, _reset, _step, create_env_params, make_env


@pytest.fixture
def env_config():
    return EnvConfig()


def test_default_env_config(env_config: EnvConfig):
    assert env_config.min_x < env_config.max_x
    assert env_config.max_steps > 0


def test_default_env_params(env_config: EnvConfig):
    params = create_env_params()
    assert params.a > 0
    assert env_config.min_x <= params.x_target <= env_config.max_x


def test_create_env_params(env_config: EnvConfig):
    a = 5.3
    x_target = 6.1
    params = create_env_params(a=a, x_target=x_target)
    assert params.a == a
    assert params.x_target == x_target


def test_make_default_env(env_config: EnvConfig):
    env = make_env()

    assert env.config == env_config
    assert env.params == create_env_params()


def test_make_env():
    config = EnvConfig(min_x=1.3, max_x=20.5)
    params = EnvParams(a=2.1, x_target=19.3)
    env = make_env(config=config, params=params)

    assert env.config == config
    assert env.params == params


@pytest.fixture
def env() -> Environment:
    return make_env()


def test_get_action_space(env: Environment):
    space = env.get_action_space(env.config)
    space = cast(spaces.Box, space)

    assert space.shape == ()


def test_get_obs_shape(env: Environment):
    assert env.get_obs_shape(env.config) == (2,)


def test_reset(key: chex.PRNGKey, env: Environment):
    obs, state = _reset(key, env.params, env.config)

    assert isinstance(state, EnvState)
    assert env.config.min_x <= state.x <= env.config.max_x
    assert state.t == 0

    assert obs.shape == env.get_obs_shape(env.config)
    assert obs[0] == state.x  # type: ignore
    assert obs[1] == env.params.x_target  # type: ignore


@pytest.mark.parametrize("action, next_x, expected_reward", [(0.5, 5.6, -2.6), (-0.5, 4.4, -1.4)])
def test_step(key: chex.PRNGKey, env: Environment, action: int, next_x: float, expected_reward: float):
    state = EnvState(x=jnp.array(5.0), t=jnp.array(4))
    params = EnvParams(a=1.2, x_target=3.0)

    obs, next_state, reward, done, info = _step(key, state, jnp.array(action), params, env.config)

    chex.assert_trees_all_close(next_state, EnvState(x=jnp.array(next_x), t=jnp.array(5)))
    chex.assert_trees_all_close(obs, jnp.array([next_x, params.x_target]))
    chex.assert_trees_all_close(reward, expected_reward)
    assert done == 0.0
    assert info == {}


def test_step_done_max_steps(key: chex.PRNGKey, env: Environment):
    state = EnvState(x=jnp.array(5.0), t=jnp.array(env.config.max_steps - 1))
    action = 1  # Any action

    obs, next_state, reward, done, info = _step(key, state, jnp.array(action), env.params, env.config)

    assert next_state.t == env.config.max_steps
    assert done == 1.0


def test_step_done_target_reached(key: chex.PRNGKey):
    config = EnvConfig(target_tol=0.1)
    params = EnvParams(a=1.0, x_target=5.5)
    state = EnvState(x=jnp.array(5.0), t=jnp.array(0))
    action = 0.45  # Any action

    obs, next_state, reward, done, info = _step(key, state, jnp.array(action), params, config)

    assert next_state.t == 1
    assert done == 1.0
