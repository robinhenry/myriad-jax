"""Tests for CcaS-CcaR control task.

This module tests the control task wrapper including:
- Environment creation with different target types
- Reset and step functionality
- Reward computation (negative absolute error)
- Termination conditions
- JIT and vmap compatibility
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from myriad.core import spaces
from myriad.envs import make_env as make_env_from_registry
from myriad.envs.bio.ccasr_gfp.physics import PhysicsState
from myriad.envs.bio.ccasr_gfp.tasks.base import CcasrGfpControlObs, TaskConfig
from myriad.envs.bio.ccasr_gfp.tasks.control import (
    ControlTaskConfig,
    ControlTaskParams,
    ControlTaskState,
    _reset,
    _step,
    get_action_space,
    get_obs,
    get_obs_shape,
    make_env,
)
from myriad.envs.environment import Environment


class TestControlTaskConfig:
    """Test control task configuration."""

    def test_default_config(self):
        """Test that default config has sensible values."""
        config = ControlTaskConfig()

        assert config.physics.eta > 0
        assert config.physics.nu > 0
        assert config.task.max_steps > 0
        assert config.task.F_obs_normalizer > 0
        assert config.target_type in ["constant", "sinewave"]
        assert config.n_horizon >= 0
        assert config.F_target_constant > 0

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = ControlTaskConfig(target_type="sinewave", F_target_constant=30.0, n_horizon=2)

        assert config.target_type == "sinewave"
        assert config.F_target_constant == 30.0
        assert config.n_horizon == 2


class TestObservations:
    """Test observation generation."""

    def test_obs_shape(self):
        """Test observation shape calculation."""
        # Obs: [F, F_target[0:n_horizon+1]]
        config = ControlTaskConfig(n_horizon=1)
        assert get_obs_shape(config) == (3,)  # [F, F_target[0], F_target[1]]

        config = ControlTaskConfig(n_horizon=0)
        assert get_obs_shape(config) == (2,)  # [F, F_target[0]]

        config = ControlTaskConfig(n_horizon=3)
        assert get_obs_shape(config) == (5,)  # [F, F_target[0:4]]

    def test_get_obs(self):
        """Test observation extraction."""
        config = ControlTaskConfig(n_horizon=1)
        physics = PhysicsState.create(time=jnp.array(10.0), H=jnp.array(50.0), F=jnp.array(25.0))
        F_target = jnp.array([30.0, 30.0])
        state = ControlTaskState(physics=physics, t=jnp.array(2), U=jnp.array(0), F_target=F_target)

        obs = get_obs(state, ControlTaskParams(), config)

        assert isinstance(obs, CcasrGfpControlObs)
        assert jnp.isclose(obs.F_normalized, 25.0 / config.task.F_obs_normalizer)
        # Obs: [F, F_target[0], F_target[1]]
        assert obs.to_array().shape == (3,)

    def test_obs_normalization(self):
        """Test that observations are properly normalized."""
        task_config = TaskConfig(max_steps=288, F_obs_normalizer=100.0)
        config = ControlTaskConfig(n_horizon=0, task=task_config)

        physics = PhysicsState.create(time=jnp.array(0.0), H=jnp.array(0.0), F=jnp.array(50.0))
        state = ControlTaskState(physics=physics, t=jnp.array(0), U=jnp.array(0), F_target=jnp.array([75.0]))

        obs = get_obs(state, ControlTaskParams(), config)

        # Obs: [F_normalized, F_target_normalized]
        assert jnp.isclose(obs.F_normalized, 0.5)  # 50/100
        assert jnp.isclose(obs.F_target[0], 0.75)  # 75/100

    def test_obs_to_array_from_array_roundtrip(self):
        """Test that observation round-trips through array conversion."""
        config = ControlTaskConfig(n_horizon=2)
        physics = PhysicsState.create(time=jnp.array(5.0), H=jnp.array(30.0), F=jnp.array(40.0))
        F_target = jnp.array([25.0, 26.0, 27.0])
        state = ControlTaskState(physics=physics, t=jnp.array(1), U=jnp.array(1), F_target=F_target)

        obs = get_obs(state, ControlTaskParams(), config)
        arr = obs.to_array()
        obs_restored = CcasrGfpControlObs.from_array(arr)

        assert jnp.isclose(obs.F_normalized, obs_restored.F_normalized)
        assert jnp.allclose(obs.F_target, obs_restored.F_target)


class TestActionSpace:
    """Test action space definition."""

    def test_action_space(self):
        """Test that action space is discrete with 2 actions."""
        config = ControlTaskConfig()
        action_space = get_action_space(config)

        assert isinstance(action_space, spaces.Discrete)
        assert action_space.n == 2  # Light off (0) and light on (1)


class TestResetFunction:
    """Test reset functionality."""

    @pytest.fixture
    def config(self):
        return ControlTaskConfig(target_type="constant", F_target_constant=25.0)

    @pytest.fixture
    def params(self):
        return ControlTaskParams()

    def test_reset_state(self, config, params):
        """Test that reset creates zero initial state."""
        key = jax.random.PRNGKey(0)
        obs, state = _reset(key, params, config)

        assert state.physics.time == 0.0
        assert state.physics.H == 0.0
        assert state.physics.F == 0.0
        assert state.t == 0
        assert state.F_target.shape == (config.n_horizon + 1,)

    def test_reset_constant_target(self, config, params):
        """Test reset with constant target."""
        key = jax.random.PRNGKey(0)
        obs, state = _reset(key, params, config)

        assert jnp.all(state.F_target == config.F_target_constant)

    def test_reset_sinewave_target(self, params):
        """Test reset with sinewave target."""
        config = ControlTaskConfig(
            target_type="sinewave",
            n_horizon=2,
            sinewave_period_minutes=600.0,
            sinewave_amplitude=20.0,
            sinewave_vshift=30.0,
        )

        key = jax.random.PRNGKey(0)
        obs, state = _reset(key, params, config)

        assert state.F_target.shape == (3,)
        assert jnp.isclose(state.F_target[0], 30.0, atol=1.0)

    def test_reset_determinism(self, config, params):
        """Test that reset is deterministic with same key."""
        key = jax.random.PRNGKey(42)

        obs1, state1 = _reset(key, params, config)
        obs2, state2 = _reset(key, params, config)

        chex.assert_trees_all_equal(state1, state2)
        chex.assert_trees_all_close(obs1, obs2)


class TestStepFunction:
    """Test step functionality."""

    @pytest.fixture
    def config(self):
        return ControlTaskConfig(target_type="constant", F_target_constant=25.0)

    @pytest.fixture
    def params(self):
        return ControlTaskParams()

    @pytest.fixture
    def state(self):
        physics = PhysicsState.create(time=jnp.array(10.0), H=jnp.array(50.0), F=jnp.array(20.0))
        F_target = jnp.array([25.0, 25.0])
        return ControlTaskState(physics=physics, t=jnp.array(2), U=jnp.array(0), F_target=F_target)

    def test_step_advances_time(self, state, params, config):
        """Test that step advances physics time within the timestep interval.

        Time stays at the last reaction time, preserving Gillespie semantics.
        """
        key = jax.random.PRNGKey(0)
        obs, next_state, reward, done, info = _step(key, state, jnp.array(1), params, config)

        target_time = state.physics.time + config.physics.timestep_minutes
        # Time should be within the interval
        assert next_state.physics.time >= state.physics.time
        assert next_state.physics.time <= target_time
        # Timestep counter should always increment
        assert next_state.t == state.t + 1

    def test_step_reward_computation(self, state, params, config):
        """Test reward is negative absolute error."""
        key = jax.random.PRNGKey(0)
        obs, next_state, reward, done, info = _step(key, state, jnp.array(1), params, config)

        # Reward = -|F - F_target|, should be negative or zero
        assert reward <= 0.0

    def test_step_termination(self, state, params, config):
        """Test termination at max_steps."""
        key = jax.random.PRNGKey(0)

        # Not done at t=2
        obs, next_state, reward, done, info = _step(key, state, jnp.array(1), params, config)
        assert not done

        # Done at t=max_steps-1 (next step reaches max)
        state_at_limit = state._replace(t=jnp.array(config.task.max_steps - 1))
        obs, next_state, reward, done, info = _step(key, state_at_limit, jnp.array(1), params, config)
        assert done

    def test_step_info_dict(self, state, params, config):
        """Test that step returns info dict with F, H, F_target."""
        key = jax.random.PRNGKey(0)
        obs, next_state, reward, done, info = _step(key, state, jnp.array(1), params, config)

        assert "F" in info
        assert "H" in info
        assert "F_target" in info
        assert info["F"] == next_state.physics.F
        assert info["H"] == next_state.physics.H

    @pytest.mark.parametrize("action", [0, 1])
    def test_step_both_actions(self, state, params, config, action):
        """Test that both actions produce valid steps."""
        key = jax.random.PRNGKey(0)
        obs, next_state, reward, done, info = _step(key, state, jnp.array(action), params, config)

        assert jnp.isfinite(next_state.physics.time)
        assert jnp.isfinite(next_state.physics.H) and next_state.physics.H >= 0
        assert jnp.isfinite(next_state.physics.F) and next_state.physics.F >= 0
        assert isinstance(obs, CcasrGfpControlObs)


class TestEnvironmentCreation:
    """Test environment factory functions."""

    def test_make_env_default(self):
        """Test creating environment with defaults."""
        env = make_env()

        assert isinstance(env, Environment)
        assert isinstance(env.config, ControlTaskConfig)
        assert isinstance(env.params, ControlTaskParams)

    def test_make_env_custom_config(self):
        """Test creating environment with custom config."""
        env = make_env(target_type="sinewave", F_target_constant=30.0, max_steps=200)

        assert env.config.target_type == "sinewave"
        assert env.config.F_target_constant == 30.0
        assert env.config.task.max_steps == 200

    def test_make_env_registry(self):
        """Test creating environment from registry."""
        env = make_env_from_registry("ccasr-gfp-control")

        assert isinstance(env, Environment)
        assert isinstance(env.config, ControlTaskConfig)

    def test_config_dt_property(self):
        """ControlTaskConfig.dt delegates to physics.timestep_minutes."""
        env = make_env()
        assert env.config.dt == env.config.physics.timestep_minutes
        assert env.config.dt > 0


class TestFullEpisode:
    """Test running full episodes."""

    @pytest.fixture
    def env(self):
        return make_env(target_type="constant", F_target_constant=25.0)

    def test_full_episode(self, env):
        """Test running a complete episode."""
        key = jax.random.PRNGKey(42)
        key, reset_key = jax.random.split(key)

        obs, state = env.reset(reset_key, env.params, env.config)

        step_count = 0
        done = False
        while not done and step_count < 50:
            key, step_key, action_key = jax.random.split(key, 3)
            action = jax.random.choice(action_key, jnp.array([0, 1]))

            obs, state, reward, done, info = env.step(step_key, state, action, env.params, env.config)
            step_count += 1

            assert jnp.isfinite(state.physics.H) and state.physics.H >= 0
            assert jnp.isfinite(state.physics.F) and state.physics.F >= 0
            assert isinstance(obs, CcasrGfpControlObs)

        assert step_count > 0

    def test_jit_compatibility(self, env):
        """Test that environment functions can be JIT compiled."""
        key = jax.random.PRNGKey(0)

        reset_jit = jax.jit(env.reset, static_argnames=["config"])
        step_jit = jax.jit(env.step, static_argnames=["config"])

        obs, state = reset_jit(key, env.params, env.config)

        key, step_key = jax.random.split(key)
        obs, state, reward, done, info = step_jit(step_key, state, jnp.array(1), env.params, env.config)

        assert jnp.isfinite(reward)
        assert isinstance(obs, CcasrGfpControlObs)

    def test_vmap_compatibility(self, env):
        """Test that environment can be vmapped for parallel execution."""
        batch_size = 10
        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

        reset_batch = jax.vmap(env.reset, in_axes=(0, None, None))
        obs_batch, state_batch = reset_batch(keys, env.params, env.config)

        assert isinstance(obs_batch, CcasrGfpControlObs)
        assert obs_batch.F_normalized.shape == (batch_size,)
        assert state_batch.t.shape == (batch_size,)

        action_batch = jnp.zeros(batch_size, dtype=jnp.int32)
        step_batch = jax.vmap(env.step, in_axes=(0, 0, 0, None, None))

        keys = jax.random.split(jax.random.PRNGKey(1), batch_size)
        obs_batch, state_batch, reward_batch, done_batch, info_batch = step_batch(
            keys, state_batch, action_batch, env.params, env.config
        )

        assert isinstance(obs_batch, CcasrGfpControlObs)
        assert reward_batch.shape == (batch_size,)
        assert done_batch.shape == (batch_size,)
