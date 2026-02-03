"""Integration tests for CartPole environments.

This file contains high-level integration tests for the CartPole environments.
Detailed unit tests are organized in:
- test_physics.py - Pure physics dynamics tests
- tasks/test_base.py - Shared task utilities tests
- tasks/test_control.py - Control task tests
"""

import chex
import jax
import jax.numpy as jnp

from myriad.envs import make_env as make_env_from_registry
from myriad.envs.classic.cartpole.tasks.control import ControlTaskState
from myriad.envs.environment import Environment


def test_env_registry_integration():
    """Test that CartPole control environment can be created via ENV_REGISTRY."""
    env = make_env_from_registry("cartpole-control")

    # Verify it's a valid Environment
    assert isinstance(env, Environment)

    # Verify it can be used
    key = jax.random.key(0)
    obs, state = env.reset(key, env.params, env.config)
    # Observation is a PhysicsState NamedTuple
    assert obs.to_array().shape == (4,)

    # Verify step works
    action = jnp.array(1)
    obs, state, reward, done, info = env.step(key, state, action, env.params, env.config)
    assert obs.to_array().shape == (4,)
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


def test_vectorized_environments():
    """Test that environments work with vectorization."""
    env = make_env_from_registry("cartpole-control")
    batch_size = 10
    key = jax.random.key(0)
    keys = jax.random.split(key, batch_size)

    # Vectorize reset
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    obs_batch, states = vmap_reset(keys, env.params, env.config)

    # obs_batch is a batched PhysicsState
    assert obs_batch.x.shape == (batch_size,)
    assert states.t.shape == (batch_size,)

    # Vectorize step
    actions = jnp.ones(batch_size, dtype=jnp.int32)
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None, None))
    obs_batch, states, rewards, dones, infos = vmap_step(keys, states, actions, env.params, env.config)

    # Verify batched observations
    assert obs_batch.x.shape == (batch_size,)
    assert rewards.shape == (batch_size,)
    assert dones.shape == (batch_size,)

    # Verify array conversion works for batch
    vmap_to_array = jax.vmap(lambda obs: obs.to_array())
    obs_arrays = vmap_to_array(obs_batch)
    assert obs_arrays.shape == (batch_size, 4)


def test_jit_compilation():
    """Test that environment functions can be JIT compiled."""
    env = make_env_from_registry("cartpole-control")
    key = jax.random.key(0)

    # JIT compile reset and step
    jitted_reset = jax.jit(env.reset, static_argnames=["config"])
    jitted_step = jax.jit(env.step, static_argnames=["config"])

    # Reset
    obs, state = jitted_reset(key, env.params, env.config)
    # Observation is a PhysicsState NamedTuple
    assert obs.to_array().shape == (4,)

    # Step
    action = jnp.array(1)
    obs, state, reward, done, info = jitted_step(key, state, action, env.params, env.config)
    assert obs.to_array().shape == (4,)
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
