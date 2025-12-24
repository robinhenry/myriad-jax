"""Integration tests for CartPole environments.

This file contains high-level integration tests for the CartPole environments.
Detailed unit tests are organized in the cartpole/ subdirectory:
- cartpole/test_physics.py - Pure physics dynamics tests
- cartpole/test_base.py - Shared task utilities tests
- cartpole/tasks/test_control.py - Control task tests
- cartpole/tasks/test_sysid.py - System identification task tests
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from myriad.envs import make_env as make_env_from_registry
from myriad.envs.cartpole.tasks.control import ControlTaskState
from myriad.envs.cartpole.tasks.sysid import SysIDTaskState
from myriad.envs.environment import Environment


@pytest.mark.parametrize("env_name", ["cartpole-control", "cartpole-sysid"])
def test_env_registry_integration(env_name: str):
    """Test that CartPole environments can be created via ENV_REGISTRY."""
    env = make_env_from_registry(env_name)

    # Verify it's a valid Environment
    assert isinstance(env, Environment)

    # Verify it can be used
    key = jax.random.key(0)
    obs, state = env.reset(key, env.params, env.config)
    assert obs.shape == (4,)

    # Verify step works
    action = jnp.array(1)
    obs, state, reward, done, info = env.step(key, state, action, env.params, env.config)
    assert obs.shape == (4,)
    assert reward.shape == ()
    assert done.shape == ()


def test_control_task_full_episode():
    """Test a complete episode rollout for control task."""
    env = make_env_from_registry("cartpole-control")
    key = jax.random.key(42)

    # Reset
    obs, state = env.reset(key, env.params, env.config)
    assert isinstance(state, ControlTaskState)

    total_reward = 0.0
    steps = 0

    # Run episode with random actions
    for i in range(200):
        key, action_key, step_key = jax.random.split(key, 3)
        action = env.get_action_space(env.config).sample(action_key)
        obs, state, reward, done, info = env.step(step_key, state, action, env.params, env.config)

        total_reward += reward
        steps += 1

        if done == 1.0:
            break

    # Should have completed or failed
    assert steps > 0
    assert total_reward == steps  # Control task gives +1 per step


def test_sysid_task_full_episode():
    """Test a complete episode rollout for SysID task."""
    env = make_env_from_registry("cartpole-sysid")
    key = jax.random.key(42)

    # Reset
    obs, state = env.reset(key, env.params, env.config)
    assert isinstance(state, SysIDTaskState)

    steps = 0

    # Run episode with random actions
    for i in range(200):
        key, action_key, step_key = jax.random.split(key, 3)
        action = env.get_action_space(env.config).sample(action_key)
        obs, state, reward, done, info = env.step(step_key, state, action, env.params, env.config)

        # Info should contain true parameters
        assert "true_pole_mass" in info
        assert "true_pole_length" in info

        steps += 1

        if done == 1.0:
            break

    # Should have completed or failed
    assert steps > 0


def test_control_and_sysid_observation_space_matches():
    """Test that control and sysid tasks have the same observation space."""
    control_env = make_env_from_registry("cartpole-control")
    sysid_env = make_env_from_registry("cartpole-sysid")

    assert control_env.get_obs_shape(control_env.config) == sysid_env.get_obs_shape(sysid_env.config)
    assert control_env.get_action_space(control_env.config).n == sysid_env.get_action_space(sysid_env.config).n


def test_control_and_sysid_action_space_matches():
    """Test that control and sysid tasks have the same action space."""
    control_env = make_env_from_registry("cartpole-control")
    sysid_env = make_env_from_registry("cartpole-sysid")

    control_space = control_env.get_action_space(control_env.config)
    sysid_space = sysid_env.get_action_space(sysid_env.config)

    assert control_space.n == sysid_space.n
    assert control_space.shape == sysid_space.shape


def test_vectorized_environments():
    """Test that environments work with vectorization."""
    env = make_env_from_registry("cartpole-control")
    batch_size = 10
    key = jax.random.key(0)
    keys = jax.random.split(key, batch_size)

    # Vectorize reset
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    obs_batch, states = vmap_reset(keys, env.params, env.config)

    assert obs_batch.shape == (batch_size, 4)
    assert states.t.shape == (batch_size,)

    # Vectorize step
    actions = jnp.ones(batch_size, dtype=jnp.int32)
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None, None))
    obs_batch, states, rewards, dones, infos = vmap_step(keys, states, actions, env.params, env.config)

    assert obs_batch.shape == (batch_size, 4)
    assert rewards.shape == (batch_size,)
    assert dones.shape == (batch_size,)


def test_jit_compilation():
    """Test that environment functions can be JIT compiled."""
    env = make_env_from_registry("cartpole-control")
    key = jax.random.key(0)

    # JIT compile reset and step
    jitted_reset = jax.jit(env.reset, static_argnames=["config"])
    jitted_step = jax.jit(env.step, static_argnames=["config"])

    # Reset
    obs, state = jitted_reset(key, env.params, env.config)
    assert obs.shape == (4,)

    # Step
    action = jnp.array(1)
    obs, state, reward, done, info = jitted_step(key, state, action, env.params, env.config)
    assert obs.shape == (4,)
    assert reward == 1.0


def test_environment_determinism():
    """Test that environments are deterministic with same random key."""
    env = make_env_from_registry("cartpole-control")
    key = jax.random.key(123)

    # Run two episodes with same key
    obs1, state1 = env.reset(key, env.params, env.config)
    obs2, state2 = env.reset(key, env.params, env.config)

    # Should be identical
    chex.assert_trees_all_equal(obs1, obs2)
    chex.assert_trees_all_equal(state1, state2)

    # Step with same key
    action = jnp.array(1)
    obs1, state1, reward1, done1, info1 = env.step(key, state1, action, env.params, env.config)
    obs2, state2, reward2, done2, info2 = env.step(key, state2, action, env.params, env.config)

    chex.assert_trees_all_equal(obs1, obs2)
    chex.assert_trees_all_equal(state1, state2)
    assert reward1 == reward2
    assert done1 == done2
