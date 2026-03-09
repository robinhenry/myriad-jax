"""Tests for the CcaS-CcaR system identification task."""

import jax
import jax.numpy as jnp
import pytest

from myriad.envs import make_env as registry_make_env
from myriad.envs.bio.ccasr_gfp.physics import PhysicsParams
from myriad.envs.bio.ccasr_gfp.tasks.sysid import (
    SysIdTaskConfig,
    SysIdTaskParams,
    _reset,
    _step,
    get_obs_shape,
    make_env,
)
from myriad.envs.environment import Environment


class TestConfig:
    def test_defaults(self):
        config = SysIdTaskConfig()
        assert config.max_steps == 288
        assert config.task.F_obs_normalizer > 0
        assert config.physics.timestep_minutes == 5.0

    def test_max_steps_property(self):
        config = SysIdTaskConfig()
        assert config.max_steps == config.task.max_steps


class TestReset:
    def test_initial_state(self):
        key = jax.random.PRNGKey(0)
        config = SysIdTaskConfig()
        params = SysIdTaskParams()
        obs, state = _reset(key, params, config)

        assert state.physics.H == 0.0
        assert state.physics.F == 0.0
        assert state.physics.time == 0.0
        assert state.t == 0
        assert state.U == 0

    def test_obs_shape(self):
        key = jax.random.PRNGKey(0)
        obs, _ = _reset(key, SysIdTaskParams(), SysIdTaskConfig())
        assert obs.to_array().shape == (1,)
        assert get_obs_shape(SysIdTaskConfig()) == (1,)

    def test_deterministic(self):
        key = jax.random.PRNGKey(42)
        config, params = SysIdTaskConfig(), SysIdTaskParams()
        _, state1 = _reset(key, params, config)
        _, state2 = _reset(key, params, config)
        assert state1.physics.H == state2.physics.H
        assert state1.physics.F == state2.physics.F


class TestStep:
    @pytest.fixture
    def env_fixtures(self):
        config = SysIdTaskConfig()
        params = SysIdTaskParams()
        key = jax.random.PRNGKey(0)
        _, state = _reset(key, params, config)
        return config, params, state

    def test_reward_is_zero(self, env_fixtures):
        config, params, state = env_fixtures
        _, _, reward, _, _ = _step(jax.random.PRNGKey(1), state, jnp.array(1), params, config)
        assert reward == 0.0

    def test_timestep_increments(self, env_fixtures):
        config, params, state = env_fixtures
        _, next_state, _, _, _ = _step(jax.random.PRNGKey(1), state, jnp.array(1), params, config)
        assert next_state.t == state.t + 1

    def test_not_done_before_max_steps(self, env_fixtures):
        config, params, state = env_fixtures
        _, _, _, done, _ = _step(jax.random.PRNGKey(1), state, jnp.array(1), params, config)
        assert not done

    def test_done_at_max_steps(self, env_fixtures):
        config, params, state = env_fixtures
        state_at_limit = state._replace(t=jnp.array(config.max_steps - 1))
        _, _, _, done, _ = _step(jax.random.PRNGKey(1), state_at_limit, jnp.array(1), params, config)
        assert done

    @pytest.mark.parametrize("action", [0, 1])
    def test_finite_values(self, env_fixtures, action):
        config, params, state = env_fixtures
        _, next_state, _, _, info = _step(jax.random.PRNGKey(1), state, jnp.array(action), params, config)
        assert jnp.isfinite(next_state.physics.F) and next_state.physics.F >= 0
        assert jnp.isfinite(next_state.physics.H) and next_state.physics.H >= 0
        assert info["F"] == next_state.physics.F

    def test_info_keys(self, env_fixtures):
        config, params, state = env_fixtures
        _, _, _, _, info = _step(jax.random.PRNGKey(1), state, jnp.array(1), params, config)
        assert "F" in info and "H" in info


class TestJitAndVmap:
    def test_jit_reset(self):
        config = SysIdTaskConfig()
        params = SysIdTaskParams()
        reset_jit = jax.jit(_reset, static_argnames=["config"])
        obs, state = reset_jit(jax.random.PRNGKey(0), params, config)
        assert jnp.isfinite(obs.F_normalized)

    def test_jit_step(self):
        config = SysIdTaskConfig()
        params = SysIdTaskParams()
        step_jit = jax.jit(_step, static_argnames=["config"])
        _, state = _reset(jax.random.PRNGKey(0), params, config)
        obs, next_state, reward, done, _ = step_jit(jax.random.PRNGKey(1), state, jnp.array(1), params, config)
        assert jnp.isfinite(reward)

    def test_vmap_over_params(self):
        """Key SysID use case: different θ* per cell, same circuit structure."""
        n_cells = 8
        config = SysIdTaskConfig()

        # Each cell has its own θ* (varying nu across cells)
        nus = jnp.linspace(0.005, 0.05, n_cells)
        params_batch = jax.vmap(lambda nu: SysIdTaskParams(physics=PhysicsParams(nu=nu)))(nus)

        keys = jax.random.split(jax.random.PRNGKey(0), n_cells)
        reset_batch = jax.vmap(_reset, in_axes=(0, 0, None))
        obs_batch, state_batch = reset_batch(keys, params_batch, config)
        assert obs_batch.F_normalized.shape == (n_cells,)

        step_batch = jax.vmap(_step, in_axes=(0, 0, 0, 0, None))
        actions = jnp.ones(n_cells, dtype=jnp.int32)
        keys2 = jax.random.split(jax.random.PRNGKey(1), n_cells)
        obs_batch, state_batch, rewards, dones, _ = step_batch(keys2, state_batch, actions, params_batch, config)
        assert obs_batch.F_normalized.shape == (n_cells,)
        assert jnp.all(rewards == 0.0)


class TestMakeEnv:
    def test_default(self):
        env = make_env()
        assert isinstance(env, Environment)
        assert isinstance(env.config, SysIdTaskConfig)
        assert isinstance(env.params, SysIdTaskParams)

    def test_registry(self):
        env = registry_make_env("ccasr-gfp-sysid")
        assert isinstance(env, Environment)
        assert isinstance(env.config, SysIdTaskConfig)

    def test_custom_params(self):
        params = SysIdTaskParams(physics=PhysicsParams(nu=0.02, Kh=80.0))
        env = make_env(params=params)
        assert env.params.physics.nu == 0.02
        assert env.params.physics.Kh == 80.0
