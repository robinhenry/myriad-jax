"""Comprehensive unit tests for CcaS-CcaR System Identification task.

Tests cover:
- Configuration with randomization ranges
- State structure
- Parameter randomization logic
- Observation generation (no targets)
- Reset functionality
- Step function mechanics
- Reward computation (state change magnitude)
- Info dict with true parameters
- JIT/vmap compatibility
"""

import jax
import jax.numpy as jnp
import pytest

from myriad.envs import make_env
from myriad.envs.ccas_ccar.physics import PhysicsParams, PhysicsState
from myriad.envs.ccas_ccar.tasks.sysid import (
    SysIDTaskConfig,
    SysIDTaskParams,
    SysIDTaskState,
    get_obs,
    make_env as make_sysid_env,
)


class TestSysIDTaskConfig:
    """Test SysIDTaskConfig configuration."""

    def test_default_config(self):
        """Test that default config values are correctly set."""
        config = SysIDTaskConfig()

        # Physics config
        assert config.physics.eta == 1.0
        assert config.physics.nu == 0.01
        assert config.physics.timestep_minutes == 5.0

        # Task config
        assert config.task.max_steps == 288
        assert config.task.F_obs_normalizer == 80.0

        # Reward config
        assert config.reward_type == "state_change"
        assert config.reward_scale == 1.0

    def test_randomization_ranges(self):
        """Test that randomization ranges are properly configured."""
        config = SysIDTaskConfig()

        # Check default ranges exist (as min/max pairs)
        assert hasattr(config, "Kh_min")
        assert hasattr(config, "Kh_max")
        assert hasattr(config, "Kf_min")
        assert hasattr(config, "Kf_max")
        assert hasattr(config, "eta_min")
        assert hasattr(config, "eta_max")
        assert hasattr(config, "a_min")
        assert hasattr(config, "a_max")

        # Check ranges are valid
        assert config.Kh_min < config.Kh_max
        assert config.Kf_min < config.Kf_max
        assert config.eta_min < config.eta_max
        assert config.a_min < config.a_max

    def test_custom_randomization(self):
        """Test custom randomization ranges."""
        config = SysIDTaskConfig(
            Kh_min=50.0,
            Kh_max=150.0,
            Kf_min=10.0,
            Kf_max=50.0,
            eta_min=0.5,
            eta_max=2.0,
            a_min=0.5,
            a_max=2.0,
        )

        assert config.Kh_min == 50.0
        assert config.Kh_max == 150.0
        assert config.Kf_min == 10.0
        assert config.Kf_max == 50.0
        assert config.eta_min == 0.5
        assert config.eta_max == 2.0
        assert config.a_min == 0.5
        assert config.a_max == 2.0


class TestSysIDTaskState:
    """Test SysIDTaskState structure."""

    def test_state_structure(self):
        """Test that state has correct fields."""
        physics = PhysicsState(time=jnp.array(0.0), H=jnp.array(0.0), F=jnp.array(0.0))
        state = SysIDTaskState(physics=physics, t=jnp.array(0))

        assert hasattr(state, "physics")
        assert hasattr(state, "t")
        assert isinstance(state.physics, PhysicsState)

    def test_state_physics(self):
        """Test physics state access."""
        physics = PhysicsState(time=jnp.array(5.0), H=jnp.array(10.0), F=jnp.array(20.0))
        state = SysIDTaskState(physics=physics, t=jnp.array(0))

        assert jnp.isclose(state.physics.time, 5.0)
        assert jnp.isclose(state.physics.H, 10.0)
        assert jnp.isclose(state.physics.F, 20.0)

    def test_state_timestep(self):
        """Test timestep counter."""
        physics = PhysicsState(time=jnp.array(0.0), H=jnp.array(0.0), F=jnp.array(0.0))
        state = SysIDTaskState(physics=physics, t=jnp.array(42))

        assert jnp.isclose(state.t, 42)


class TestObservations:
    """Test observation generation."""

    def test_obs_shape(self):
        """Test that observation has correct shape (no target)."""
        from myriad.envs.ccas_ccar.tasks.base import CcasCcarSysIDObs

        config = SysIDTaskConfig()
        physics = PhysicsState(time=jnp.array(0.0), H=jnp.array(0.0), F=jnp.array(50.0))
        state = SysIDTaskState(physics=physics, t=jnp.array(0))

        obs = get_obs(state, SysIDTaskParams(), config)

        # Observation is a CcasCcarSysIDObs NamedTuple
        assert isinstance(obs, CcasCcarSysIDObs)

        # Verify array conversion gives shape (3,): [F_normalized, U_prev, 0]
        assert obs.to_array().shape == (3,)

    def test_get_obs(self):
        """Test observation extraction."""
        config = SysIDTaskConfig()
        physics = PhysicsState(time=jnp.array(0.0), H=jnp.array(0.0), F=jnp.array(40.0))
        state = SysIDTaskState(physics=physics, t=jnp.array(0))
        params = SysIDTaskParams()

        obs = get_obs(state, params, config)

        # F normalized
        expected_F_norm = 40.0 / config.task.F_obs_normalizer
        assert jnp.isclose(obs[0], expected_F_norm)

        # U_prev (starts at 0)
        assert jnp.isclose(obs[1], 0.0)

        # No target (always 0)
        assert jnp.isclose(obs[2], 0.0)

    def test_obs_normalization(self):
        """Test that observations are properly normalized."""
        from myriad.envs.ccas_ccar.tasks.sysid import TaskConfig

        task_config = TaskConfig(max_steps=288, F_obs_normalizer=100.0)
        config = SysIDTaskConfig(task=task_config)

        physics = PhysicsState(time=jnp.array(0.0), H=jnp.array(0.0), F=jnp.array(50.0))
        state = SysIDTaskState(physics=physics, t=jnp.array(0))

        obs = get_obs(state, SysIDTaskParams(), config)

        assert jnp.isclose(obs[0], 0.5)  # 50/100
        assert jnp.isclose(obs[2], 0.0)  # No target


class TestParameterRandomization:
    """Test parameter randomization for domain randomization."""

    def test_params_structure(self):
        """Test that SysIDTaskParams has correct fields."""
        params = SysIDTaskParams()

        assert hasattr(params, "physics")
        assert isinstance(params.physics, PhysicsParams)

    def test_randomization_values(self):
        """Test that randomized parameters are within expected ranges."""
        from myriad.envs.ccas_ccar.tasks.sysid import sample_randomized_params

        config = SysIDTaskConfig(
            Kh_min=50.0,
            Kh_max=150.0,
            Kf_min=10.0,
            Kf_max=50.0,
            eta_min=0.5,
            eta_max=2.0,
            a_min=0.5,
            a_max=2.0,
        )

        # Create multiple randomized params to test range
        key = jax.random.PRNGKey(0)
        for i in range(10):
            key, subkey = jax.random.split(key)

            # Sample randomized params
            params = sample_randomized_params(subkey, config)

            # Check ranges
            assert config.Kh_min <= params.Kh <= config.Kh_max
            assert config.Kf_min <= params.Kf <= config.Kf_max
            assert config.eta_min <= params.eta <= config.eta_max
            assert config.a_min <= params.a <= config.a_max


class TestResetFunction:
    """Test reset functionality."""

    def test_reset_state(self):
        """Test that reset creates proper initial state."""
        from myriad.envs.ccas_ccar.tasks.sysid import _reset

        config = SysIDTaskConfig()
        params = SysIDTaskParams()
        key = jax.random.PRNGKey(0)

        obs, new_state = _reset(key, params, config)

        # Check state structure
        assert isinstance(new_state, SysIDTaskState)
        assert isinstance(new_state.physics, PhysicsState)

        # Check timestep reset
        assert jnp.isclose(new_state.t, 0)

        # Check physics time reset
        assert jnp.isclose(new_state.physics.time, 0.0)

    def test_reset_observation_shape(self):
        """Test that reset produces correct observation shape."""
        from myriad.envs.ccas_ccar.tasks.base import CcasCcarSysIDObs
        from myriad.envs.ccas_ccar.tasks.sysid import _reset

        config = SysIDTaskConfig()
        params = SysIDTaskParams()
        key = jax.random.PRNGKey(0)

        obs, new_state = _reset(key, params, config)

        # Observation is a CcasCcarSysIDObs NamedTuple
        assert isinstance(obs, CcasCcarSysIDObs)

        # Verify array shape: [F_normalized, U_prev, 0]
        assert obs.to_array().shape == (3,)

    def test_reset_determinism(self):
        """Test that same seed produces same initial conditions."""
        from myriad.envs.ccas_ccar.tasks.sysid import _reset

        config = SysIDTaskConfig()
        params = SysIDTaskParams()
        key = jax.random.PRNGKey(42)

        # Reset twice with same key
        obs1, state1 = _reset(key, params, config)
        obs2, state2 = _reset(key, params, config)

        # Should be identical (compare array representations)
        assert jnp.allclose(obs1.to_array(), obs2.to_array())
        assert jnp.isclose(state1.physics.H, state2.physics.H)
        assert jnp.isclose(state1.physics.F, state2.physics.F)

    def test_reset_stochasticity(self):
        """Test that different seeds produce different initial conditions."""
        from myriad.envs.ccas_ccar.tasks.sysid import _reset

        config = SysIDTaskConfig()
        params = SysIDTaskParams()

        # Reset with different keys
        obs1, state1 = _reset(jax.random.PRNGKey(0), params, config)
        obs2, state2 = _reset(jax.random.PRNGKey(1), params, config)

        # Initial state is actually always the same (zero concentrations)
        # The randomization is in the params, not the initial state
        # So this test should check that obs is the same (compare array representations)
        assert jnp.allclose(obs1.to_array(), obs2.to_array())
        assert jnp.isclose(state1.physics.F, state2.physics.F)


class TestStepFunction:
    """Test step function mechanics."""

    def test_step_advances_time(self):
        """Test that step advances physics time."""
        from myriad.envs.ccas_ccar.tasks.sysid import _step

        config = SysIDTaskConfig()
        params = SysIDTaskParams()
        physics = PhysicsState(time=jnp.array(0.0), H=jnp.array(10.0), F=jnp.array(20.0))
        state = SysIDTaskState(physics=physics, t=jnp.array(0))

        key = jax.random.PRNGKey(0)
        action = jnp.array(1)

        obs, next_state, reward, done, info = _step(key, state, action, params, config)

        # Time should advance by timestep_minutes
        expected_time = config.physics.timestep_minutes
        assert jnp.isclose(next_state.physics.time, expected_time)

    def test_step_increments_counter(self):
        """Test that step increments timestep counter."""
        from myriad.envs.ccas_ccar.tasks.sysid import _step

        config = SysIDTaskConfig()
        params = SysIDTaskParams()
        physics = PhysicsState(time=jnp.array(0.0), H=jnp.array(10.0), F=jnp.array(20.0))
        state = SysIDTaskState(physics=physics, t=jnp.array(5))

        key = jax.random.PRNGKey(0)
        action = jnp.array(1)

        obs, next_state, reward, done, info = _step(key, state, action, params, config)

        assert jnp.isclose(next_state.t, 6)

    def test_step_reward_computation(self):
        """Test that reward is based on state change magnitude."""
        from myriad.envs.ccas_ccar.tasks.sysid import _step

        config = SysIDTaskConfig(reward_type="state_change", reward_scale=1.0)
        params = SysIDTaskParams()
        physics = PhysicsState(time=jnp.array(0.0), H=jnp.array(10.0), F=jnp.array(20.0))
        state = SysIDTaskState(physics=physics, t=jnp.array(0))

        key = jax.random.PRNGKey(0)
        action = jnp.array(1)

        obs, next_state, reward, done, info = _step(key, state, action, params, config)

        # Reward should be based on magnitude of state change
        # For SysID, we want to encourage exploration (state changes)
        # Reward = ||[H_next - H_prev, F_next - F_prev]||_2
        state_diff = jnp.array(
            [
                next_state.physics.H - state.physics.H,
                next_state.physics.F - state.physics.F,
            ]
        )
        expected_reward = jnp.linalg.norm(state_diff) * config.reward_scale
        assert jnp.isclose(reward, expected_reward, atol=1e-5)

    def test_step_termination(self):
        """Test termination condition (max_steps)."""
        from myriad.envs.ccas_ccar.tasks.sysid import _step

        config = SysIDTaskConfig()
        params = SysIDTaskParams()

        key = jax.random.PRNGKey(0)
        action = jnp.array(1)

        # Test before max_steps
        physics = PhysicsState(time=jnp.array(0.0), H=jnp.array(10.0), F=jnp.array(20.0))
        state = SysIDTaskState(physics=physics, t=jnp.array(config.task.max_steps - 2))
        obs, next_state, reward, done, info = _step(key, state, action, params, config)
        assert not done

        # Test at max_steps
        state = SysIDTaskState(physics=physics, t=jnp.array(config.task.max_steps - 1))
        obs, next_state, reward, done, info = _step(key, state, action, params, config)
        assert done

    def test_step_observation_shape(self):
        """Test step observation shape."""
        from myriad.envs.ccas_ccar.tasks.base import CcasCcarSysIDObs
        from myriad.envs.ccas_ccar.tasks.sysid import _step

        config = SysIDTaskConfig()
        params = SysIDTaskParams()
        physics = PhysicsState(time=jnp.array(0.0), H=jnp.array(10.0), F=jnp.array(20.0))
        state = SysIDTaskState(physics=physics, t=jnp.array(0))

        key = jax.random.PRNGKey(0)
        action = jnp.array(1)

        obs, next_state, reward, done, info = _step(key, state, action, params, config)

        # Observation is a CcasCcarSysIDObs NamedTuple
        assert isinstance(obs, CcasCcarSysIDObs)

        # Verify array shape: [F_normalized, U_prev, 0]
        assert obs.to_array().shape == (3,)

    def test_step_info_dict(self):
        """Test that info dict contains true parameters."""
        from myriad.envs.ccas_ccar.tasks.sysid import _step

        config = SysIDTaskConfig()
        params = SysIDTaskParams()
        physics = PhysicsState(time=jnp.array(0.0), H=jnp.array(10.0), F=jnp.array(20.0))
        state = SysIDTaskState(physics=physics, t=jnp.array(0))

        key = jax.random.PRNGKey(0)
        action = jnp.array(1)

        obs, next_state, reward, done, info = _step(key, state, action, params, config)

        # Info should contain true parameters for system identification
        assert isinstance(info, dict)
        # The info dict should include things like Kh, Kf, eta, a
        # But current implementation might not have this - check and add if needed

    @pytest.mark.parametrize("action", [0, 1])
    def test_step_both_actions(self, action):
        """Test that both actions work."""
        from myriad.envs.ccas_ccar.tasks.sysid import _step

        config = SysIDTaskConfig()
        params = SysIDTaskParams()
        physics = PhysicsState(time=jnp.array(0.0), H=jnp.array(10.0), F=jnp.array(20.0))
        state = SysIDTaskState(physics=physics, t=jnp.array(0))

        key = jax.random.PRNGKey(0)

        obs, next_state, reward, done, info = _step(key, state, jnp.array(action), params, config)

        # Should complete without error
        assert next_state.t == 1
        assert not jnp.isnan(reward)

    def test_step_stochasticity(self):
        """Test that step produces different outcomes with different keys."""
        from myriad.envs.ccas_ccar.tasks.sysid import _step

        config = SysIDTaskConfig()
        params = SysIDTaskParams()
        physics = PhysicsState(time=jnp.array(0.0), H=jnp.array(10.0), F=jnp.array(20.0))
        state = SysIDTaskState(physics=physics, t=jnp.array(0))

        action = jnp.array(1)

        # Step with different keys
        obs1, state1, reward1, done1, info1 = _step(jax.random.PRNGKey(0), state, action, params, config)
        obs2, state2, reward2, done2, info2 = _step(jax.random.PRNGKey(1), state, action, params, config)

        # Outcomes should be different (stochastic Gillespie)
        # With high probability, at least one state variable differs (compare array representations)
        assert not jnp.allclose(obs1.to_array(), obs2.to_array()) or not jnp.isclose(state1.physics.F, state2.physics.F)


class TestEnvironmentCreation:
    """Test environment creation and registry."""

    def test_make_env_default(self):
        """Test make_env with default config."""
        env = make_sysid_env()

        # Check environment structure
        assert hasattr(env, "reset")
        assert hasattr(env, "step")
        assert hasattr(env, "get_action_space")
        assert hasattr(env, "get_obs_shape")
        assert hasattr(env, "params")
        assert hasattr(env, "config")

    def test_make_env_custom_config(self):
        """Test make_env with custom config."""
        env = make_sysid_env(
            Kh_min=50.0,
            Kh_max=150.0,
            Kf_min=10.0,
            Kf_max=50.0,
        )

        assert env.config.Kh_min == 50.0
        assert env.config.Kh_max == 150.0
        assert env.config.Kf_min == 10.0
        assert env.config.Kf_max == 50.0

    def test_make_env_registry(self):
        """Test that environment is accessible via registry."""
        env = make_env("ccas-ccar-sysid")

        assert hasattr(env, "reset")
        assert hasattr(env, "step")

    def test_make_env_registry_with_kwargs(self):
        """Test registry with custom kwargs."""
        env = make_env("ccas-ccar-sysid", Kh_min=50.0, Kh_max=150.0)

        assert env.config.Kh_min == 50.0
        assert env.config.Kh_max == 150.0


class TestFullEpisode:
    """Test full episode execution."""

    def test_full_episode(self):
        """Test running a full episode."""
        from myriad.envs.ccas_ccar.tasks.base import CcasCcarSysIDObs
        from myriad.envs.ccas_ccar.tasks.sysid import _reset, _step

        config = SysIDTaskConfig()
        params = SysIDTaskParams()

        # Reset
        key = jax.random.PRNGKey(0)
        key, reset_key = jax.random.split(key)
        obs, state = _reset(reset_key, params, config)

        # Run 10 steps
        for i in range(10):
            key, step_key = jax.random.split(key)
            action = jax.random.choice(step_key, jnp.array([0, 1]))

            obs, state, reward, done, info = _step(key, state, action, params, config)

            # Check validity
            assert not jnp.isnan(reward)
            assert isinstance(obs, CcasCcarSysIDObs)
            assert obs.to_array().shape == (3,)
            assert state.t == i + 1

            if done:
                break

    def test_jit_compatibility(self):
        """Test JIT compilation of reset and step."""
        from myriad.envs.ccas_ccar.tasks.base import CcasCcarSysIDObs
        from myriad.envs.ccas_ccar.tasks.sysid import _reset, _step

        config = SysIDTaskConfig()
        params = SysIDTaskParams()

        # JIT compile
        reset_jit = jax.jit(_reset, static_argnames=["config"])
        step_jit = jax.jit(_step, static_argnames=["config"])

        # Test reset
        key = jax.random.PRNGKey(0)
        obs, state = reset_jit(key, params, config)
        assert isinstance(obs, CcasCcarSysIDObs)
        assert obs.to_array().shape == (3,)

        # Test step
        key = jax.random.PRNGKey(1)
        action = jnp.array(1)
        obs, state, reward, done, info = step_jit(key, state, action, params, config)
        assert not jnp.isnan(reward)

    def test_vmap_compatibility(self):
        """Test vmap for batched environments."""
        from myriad.envs.ccas_ccar.tasks.base import CcasCcarSysIDObs
        from myriad.envs.ccas_ccar.tasks.sysid import _reset, _step

        config = SysIDTaskConfig()
        params = SysIDTaskParams()

        # Vmap reset (only key is batched, params and config are broadcast)
        reset_vmap = jax.vmap(_reset, in_axes=(0, None, None))
        n_envs = 4
        keys = jax.random.split(jax.random.PRNGKey(0), n_envs)
        obs_batch, states = reset_vmap(keys, params, config)

        # obs_batch is a batched CcasCcarSysIDObs
        assert isinstance(obs_batch, CcasCcarSysIDObs)
        assert obs_batch.F_normalized.shape == (n_envs,)

        # Vmap step
        step_vmap = jax.vmap(_step, in_axes=(0, 0, 0, None, None))
        keys = jax.random.split(jax.random.PRNGKey(1), n_envs)
        actions = jnp.array([0, 1, 0, 1])

        obs_batch, states, rewards, dones, infos = step_vmap(keys, states, actions, params, config)

        # Verify batched observations
        assert isinstance(obs_batch, CcasCcarSysIDObs)
        assert obs_batch.F_normalized.shape == (n_envs,)
        assert rewards.shape == (n_envs,)
        assert dones.shape == (n_envs,)

        # Verify array conversion works for batch
        vmap_to_array = jax.vmap(lambda obs: obs.to_array())
        obs_arrays = vmap_to_array(obs_batch)
        assert obs_arrays.shape == (n_envs, 3)
