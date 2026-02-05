"""Tests for CcaS-CcaR control task.

This module tests the control task wrapper including:
- Environment creation with different target types
- Reset functionality
- Step functionality
- Reward computation (negative absolute error)
- Termination conditions
- Observation generation
- Target trajectory generation (constant and sinewave)
- Integration with Gillespie physics
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from myriad.core import spaces
from myriad.envs import make_env as make_env_from_registry
from myriad.envs.bio.ccas_ccar.physics import PhysicsState
from myriad.envs.bio.ccas_ccar.tasks.base import CcasCcarControlObs, TaskConfig
from myriad.envs.bio.ccas_ccar.tasks.control import (
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

        # Physics config
        assert config.physics.eta > 0
        assert config.physics.nu > 0
        assert config.physics.a > 0
        assert config.physics.Kh > 0
        assert config.physics.nh > 0
        assert config.physics.Kf > 0
        assert config.physics.nf > 0
        assert config.physics.timestep_minutes > 0
        assert config.physics.max_gillespie_steps > 0

        # Task config
        assert config.task.max_steps > 0
        assert config.task.F_obs_normalizer > 0

        # Control specific
        assert config.target_type in ["constant", "sinewave"]
        assert config.n_horizon >= 0
        assert config.F_target_constant > 0

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = ControlTaskConfig(
            target_type="sinewave",
            F_target_constant=30.0,
            n_horizon=2,
        )

        assert config.target_type == "sinewave"
        assert config.F_target_constant == 30.0
        assert config.n_horizon == 2

    def test_max_steps_property(self):
        """Test that max_steps property delegates to task config."""
        config = ControlTaskConfig()
        assert config.max_steps == config.task.max_steps


class TestControlTaskState:
    """Test control task state structure."""

    @pytest.fixture
    def state(self):
        """Create a sample state for testing."""
        physics = PhysicsState(time=jnp.array(10.0), H=jnp.array(50.0), F=jnp.array(25.0))
        F_target = jnp.array([25.0, 25.0])
        return ControlTaskState(physics=physics, t=jnp.array(2), F_target=F_target)

    def test_state_structure(self, state):
        """Test that state has correct structure."""
        assert hasattr(state, "physics")
        assert hasattr(state, "t")
        assert hasattr(state, "F_target")

    def test_state_physics(self, state):
        """Test physics state within task state."""
        assert state.physics.time == 10.0
        assert state.physics.H == 50.0
        assert state.physics.F == 25.0

    def test_state_timestep(self, state):
        """Test timestep counter."""
        assert state.t == 2

    def test_state_target(self, state):
        """Test target trajectory."""
        assert state.F_target.shape == (2,)
        assert jnp.all(state.F_target == 25.0)


class TestObservations:
    """Test observation generation."""

    @pytest.fixture
    def config(self):
        """Default config with n_horizon=1."""
        return ControlTaskConfig(n_horizon=1)

    @pytest.fixture
    def state(self):
        """Sample state."""
        physics = PhysicsState(time=jnp.array(10.0), H=jnp.array(50.0), F=jnp.array(25.0))
        F_target = jnp.array([30.0, 30.0])
        return ControlTaskState(physics=physics, t=jnp.array(2), F_target=F_target)

    def test_obs_shape(self, config):
        """Test observation shape calculation."""
        # Obs: [F, U, F_target[0:n_horizon+1]]
        # With n_horizon=1: [F, U, F_target[0], F_target[1]] = 4 elements
        shape = get_obs_shape(config)
        assert shape == (4,)

    def test_obs_shape_different_horizons(self):
        """Test observation shape with different horizons."""
        config = ControlTaskConfig(n_horizon=0)
        assert get_obs_shape(config) == (3,)  # [F, U, F_target[0]]

        config = ControlTaskConfig(n_horizon=3)
        assert get_obs_shape(config) == (6,)  # [F, U, F_target[0], ..., F_target[3]]

    def test_get_obs(self, state, config):
        """Test observation extraction."""

        params = ControlTaskParams()
        obs = get_obs(state, params, config)

        # Observation is a CcasCcarControlObs NamedTuple
        assert isinstance(obs, CcasCcarControlObs)

        # F normalized (use named field)
        assert jnp.isclose(obs.F_normalized, 25.0 / config.task.F_obs_normalizer)

        # U (should be 0 in observation)
        assert obs.U_obs == 0.0

        # F_target normalized
        expected_targets = jnp.array([30.0, 30.0]) / config.task.F_obs_normalizer
        assert jnp.allclose(obs.F_target, expected_targets)

        # Verify array conversion
        obs_array = obs.to_array()
        assert obs_array.shape == (4,)

    def test_obs_normalization(self):
        """Test that observations are properly normalized."""
        # Create config with custom normalizer

        task_config = TaskConfig(max_steps=288, F_obs_normalizer=100.0)
        config = ControlTaskConfig(n_horizon=0, task=task_config)

        physics = PhysicsState(time=jnp.array(0.0), H=jnp.array(0.0), F=jnp.array(50.0))
        state = ControlTaskState(physics=physics, t=jnp.array(0), F_target=jnp.array([75.0]))

        obs = get_obs(state, ControlTaskParams(), config)

        assert jnp.isclose(obs[0], 0.5)  # 50/100
        assert jnp.isclose(obs[2], 0.75)  # 75/100


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
        """Default config."""
        return ControlTaskConfig(target_type="constant", F_target_constant=25.0)

    @pytest.fixture
    def params(self):
        """Default params."""
        return ControlTaskParams()

    def test_reset_state(self, config, params):
        """Test that reset creates zero initial state."""
        key = jax.random.PRNGKey(0)
        obs, state = _reset(key, params, config)

        # Physics should start at zero
        assert state.physics.time == 0.0
        assert state.physics.H == 0.0
        assert state.physics.F == 0.0

        # Timestep counter should be zero
        assert state.t == 0

        # Target should be generated
        assert state.F_target.shape == (config.n_horizon + 1,)

    def test_reset_constant_target(self, config, params):
        """Test reset with constant target."""
        key = jax.random.PRNGKey(0)
        obs, state = _reset(key, params, config)

        # All targets should equal F_target_constant
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

        # Targets should vary (sinewave)
        assert state.F_target.shape == (3,)  # n_horizon + 1
        # At t=0, first target should be close to vshift
        assert jnp.isclose(state.F_target[0], 30.0, atol=1.0)

    def test_reset_observation_shape(self, config, params):
        """Test that reset returns correct observation shape."""

        key = jax.random.PRNGKey(0)
        obs, state = _reset(key, params, config)

        # Observation is a CcasCcarControlObs NamedTuple
        assert isinstance(obs, CcasCcarControlObs)

        # Verify array conversion gives expected shape
        expected_shape = get_obs_shape(config)
        assert obs.to_array().shape == expected_shape

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
        """Default config."""
        return ControlTaskConfig(target_type="constant", F_target_constant=25.0)

    @pytest.fixture
    def params(self):
        """Default params."""
        return ControlTaskParams()

    @pytest.fixture
    def state(self):
        """Sample state."""
        physics = PhysicsState(time=jnp.array(10.0), H=jnp.array(50.0), F=jnp.array(20.0))
        F_target = jnp.array([25.0, 25.0])
        return ControlTaskState(physics=physics, t=jnp.array(2), F_target=F_target)

    def test_step_advances_time(self, state, params, config):
        """Test that step advances physics time."""
        key = jax.random.PRNGKey(0)
        action = jnp.array(1)

        obs, next_state, reward, done, info = _step(key, state, action, params, config)

        # Physics time should advance by timestep_minutes
        expected_time = state.physics.time + config.physics.timestep_minutes
        assert jnp.isclose(next_state.physics.time, expected_time)

    def test_step_increments_counter(self, state, params, config):
        """Test that step increments timestep counter."""
        key = jax.random.PRNGKey(0)
        action = jnp.array(1)

        obs, next_state, reward, done, info = _step(key, state, action, params, config)

        assert next_state.t == state.t + 1

    def test_step_reward_computation(self, state, params, config):
        """Test reward is negative absolute error."""
        key = jax.random.PRNGKey(0)
        action = jnp.array(1)

        obs, next_state, reward, done, info = _step(key, state, action, params, config)

        # Reward = -|F - F_target|
        # We can't predict exact F after Gillespie step, but reward should be negative
        assert reward <= 0.0  # Error is always non-negative

    def test_step_termination(self, state, params, config):
        """Test termination at max_steps."""
        key = jax.random.PRNGKey(0)
        action = jnp.array(1)

        # Not done at t=2
        obs, next_state, reward, done, info = _step(key, state, action, params, config)
        assert not done

        # Done at t=max_steps
        state_at_limit = state._replace(t=jnp.array(config.task.max_steps - 1))
        obs, next_state, reward, done, info = _step(key, state_at_limit, action, params, config)
        assert done

    def test_step_updates_target(self, params, config):
        """Test that step generates new target trajectory."""
        key = jax.random.PRNGKey(0)
        action = jnp.array(1)

        physics = PhysicsState(time=jnp.array(0.0), H=jnp.array(0.0), F=jnp.array(0.0))
        state = ControlTaskState(physics=physics, t=jnp.array(0), F_target=jnp.array([25.0, 25.0]))

        obs, next_state, reward, done, info = _step(key, state, action, params, config)

        # Target should still be constant for constant target type
        assert jnp.all(next_state.F_target == 25.0)

    def test_step_observation_shape(self, state, params, config):
        """Test that step returns correct observation shape."""

        key = jax.random.PRNGKey(0)
        action = jnp.array(1)

        obs, next_state, reward, done, info = _step(key, state, action, params, config)

        # Observation is a CcasCcarControlObs NamedTuple
        assert isinstance(obs, CcasCcarControlObs)

        # Verify array conversion gives expected shape
        expected_shape = get_obs_shape(config)
        assert obs.to_array().shape == expected_shape

    def test_step_info_dict(self, state, params, config):
        """Test that step returns info dict with F, H, F_target."""
        key = jax.random.PRNGKey(0)
        action = jnp.array(1)

        obs, next_state, reward, done, info = _step(key, state, action, params, config)

        assert "F" in info
        assert "H" in info
        assert "F_target" in info

        assert info["F"] == next_state.physics.F
        assert info["H"] == next_state.physics.H
        assert info["F_target"] == state.F_target[0]  # Current target

    @pytest.mark.parametrize("action", [0, 1])
    def test_step_both_actions(self, state, params, config, action):
        """Test that both actions produce valid steps."""

        key = jax.random.PRNGKey(0)

        obs, next_state, reward, done, info = _step(key, state, jnp.array(action), params, config)

        # All state variables should be finite and non-negative
        assert jnp.isfinite(next_state.physics.time)
        assert jnp.isfinite(next_state.physics.H)
        assert jnp.isfinite(next_state.physics.F)
        assert next_state.physics.H >= 0
        assert next_state.physics.F >= 0

        # Observation should be valid
        assert isinstance(obs, CcasCcarControlObs)
        obs_array = obs.to_array()
        assert obs_array.shape == get_obs_shape(config)
        assert jnp.all(jnp.isfinite(obs_array))

    def test_step_stochasticity(self, state, params, config):
        """Test that steps with different keys produce different results."""
        action = jnp.array(1)

        obs1, state1, _, _, _ = _step(jax.random.PRNGKey(0), state, action, params, config)
        obs2, state2, _, _, _ = _step(jax.random.PRNGKey(1), state, action, params, config)

        # Results should differ (stochastic Gillespie)
        # Time advances deterministically, but H and F are stochastic
        assert state1.physics.time == state2.physics.time
        # H or F should be different (with high probability)
        # Run multiple steps to ensure stochastic difference
        for _ in range(5):
            obs1, state1, _, _, _ = _step(jax.random.PRNGKey(0), state1, action, params, config)
            obs2, state2, _, _, _ = _step(jax.random.PRNGKey(1), state2, action, params, config)

        assert not jnp.allclose(state1.physics.H, state2.physics.H) or not jnp.allclose(
            state1.physics.F, state2.physics.F
        )


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
        env = make_env(
            target_type="sinewave",
            F_target_constant=30.0,
            max_steps=200,
        )

        assert env.config.target_type == "sinewave"
        assert env.config.F_target_constant == 30.0
        assert env.config.task.max_steps == 200

    def test_make_env_registry(self):
        """Test creating environment from registry."""
        env = make_env_from_registry("ccas-ccar-control")

        assert isinstance(env, Environment)
        assert isinstance(env.config, ControlTaskConfig)

    def test_make_env_registry_with_kwargs(self):
        """Test creating environment from registry with kwargs."""
        env = make_env_from_registry(
            "ccas-ccar-control",
            F_target_constant=35.0,
        )

        assert env.config.F_target_constant == 35.0


class TestFullEpisode:
    """Test running full episodes."""

    @pytest.fixture
    def env(self):
        """Create environment."""
        return make_env(target_type="constant", F_target_constant=25.0)

    def test_full_episode(self, env):
        """Test running a complete episode."""

        key = jax.random.PRNGKey(42)
        key, reset_key = jax.random.split(key)

        # Reset
        obs, state = env.reset(reset_key, env.params, env.config)

        # Run episode
        done = False
        step_count = 0
        total_reward = 0.0

        while not done and step_count < 50:  # Limit for test speed
            key, step_key, action_key = jax.random.split(key, 3)
            action = jax.random.choice(action_key, jnp.array([0, 1]))

            obs, state, reward, done, info = env.step(step_key, state, action, env.params, env.config)

            total_reward += reward
            step_count += 1

            # Validate state
            assert jnp.isfinite(state.physics.H)
            assert jnp.isfinite(state.physics.F)
            assert state.physics.H >= 0
            assert state.physics.F >= 0

            # Validate observation
            assert isinstance(obs, CcasCcarControlObs)
            obs_array = obs.to_array()
            assert obs_array.shape == env.get_obs_shape(env.config)
            assert jnp.all(jnp.isfinite(obs_array))

        assert step_count > 0

    def test_jit_compatibility(self, env):
        """Test that environment functions can be JIT compiled."""

        key = jax.random.PRNGKey(0)

        # JIT compile reset and step
        reset_jit = jax.jit(env.reset, static_argnames=["config"])
        step_jit = jax.jit(env.step, static_argnames=["config"])

        # Reset
        obs, state = reset_jit(key, env.params, env.config)

        # Step
        key, step_key = jax.random.split(key)
        obs, state, reward, done, info = step_jit(step_key, state, jnp.array(1), env.params, env.config)

        # Should produce valid outputs
        assert jnp.isfinite(reward)
        assert isinstance(obs, CcasCcarControlObs)
        assert obs.to_array().shape == env.get_obs_shape(env.config)

    def test_vmap_compatibility(self, env):
        """Test that environment can be vmapped for parallel execution."""

        batch_size = 10
        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

        # Vmap reset
        reset_batch = jax.vmap(env.reset, in_axes=(0, None, None))
        obs_batch, state_batch = reset_batch(keys, env.params, env.config)

        # obs_batch is a batched CcasCcarControlObs
        assert isinstance(obs_batch, CcasCcarControlObs)
        assert obs_batch.F_normalized.shape == (batch_size,)
        assert state_batch.t.shape == (batch_size,)

        # Vmap step
        action_batch = jnp.zeros(batch_size, dtype=jnp.int32)
        step_batch = jax.vmap(env.step, in_axes=(0, 0, 0, None, None))

        keys = jax.random.split(jax.random.PRNGKey(1), batch_size)
        obs_batch, state_batch, reward_batch, done_batch, info_batch = step_batch(
            keys, state_batch, action_batch, env.params, env.config
        )

        # Verify batched observations
        assert isinstance(obs_batch, CcasCcarControlObs)
        assert obs_batch.F_normalized.shape == (batch_size,)
        assert reward_batch.shape == (batch_size,)
        assert done_batch.shape == (batch_size,)

        # Verify array conversion works for batch
        vmap_to_array = jax.vmap(lambda obs: obs.to_array())
        obs_arrays = vmap_to_array(obs_batch)
        assert obs_arrays.shape[0] == batch_size
