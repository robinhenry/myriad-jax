from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from myriad.core.types import Transition
from myriad.platform import runners, steps


def test_tree_select_masking():
    mask_scalar = jnp.array(True)
    new_tree = {"value": jnp.array(2.0)}
    old_tree = {"value": jnp.array(-1.0)}
    result = steps.tree_select(mask_scalar, new_tree, old_tree)
    assert np.array(result["value"]) == pytest.approx(2.0)

    mask = jnp.array([True, False], dtype=jnp.bool_)
    new_value = jnp.array([5.0, 10.0])
    old_value = jnp.array([1.0, 1.0])
    expanded = steps._expand_mask(mask, target_ndim=2)
    assert expanded.shape == (2, 1)
    where_result = steps.where_mask(mask, new_value, old_value)
    np.testing.assert_allclose(where_result, np.array([5.0, 1.0]))
    mask_tree_result = steps.mask_tree(mask, {"value": new_value}, {"value": old_value})
    np.testing.assert_allclose(mask_tree_result["value"], np.array([5.0, 1.0]))


def _dummy_train_step_fn(batch_size: int):
    def _step(
        *,
        key: chex.PRNGKey,
        agent_state: jnp.ndarray,
        training_env_states: jnp.ndarray,
        buffer_state: jnp.ndarray,
        batch_size: int,
    ):
        key, _ = jax.random.split(key)
        new_agent = agent_state + 1
        new_env = training_env_states + 1
        new_buffer = buffer_state + 1
        metrics = {"metric": jnp.full((1,), batch_size, dtype=jnp.float32)}
        return key, new_agent, new_env, new_buffer, metrics

    return partial(_step, batch_size=batch_size)


def test_make_chunk_runner_masks_inactive_iterations():
    batch_size = 4
    run_chunk = runners.make_chunk_runner(_dummy_train_step_fn(batch_size), batch_size)

    key = jax.random.PRNGKey(0)
    agent_state = jnp.array(0.0, dtype=jnp.float32)
    env_state = jnp.array(0.0, dtype=jnp.float32)
    buffer_state = jnp.array(0.0, dtype=jnp.float32)
    active_mask = jnp.array([True, False, True])

    (key_out, agent_out, env_out, buffer_out), metrics_history = run_chunk(
        (key, agent_state, env_state, buffer_state),
        active_mask,
    )

    assert agent_out == pytest.approx(2.0)
    assert env_out == pytest.approx(2.0)
    assert buffer_out == pytest.approx(2.0)
    assert metrics_history["metric"].shape == (3, 1)
    assert metrics_history["metric"][1] == pytest.approx(0.0)
    assert key_out.shape == (2,)


def test_where_mask_supports_higher_rank():
    mask = jnp.array([True, False, True])
    new = jnp.array([[1, 1], [2, 2], [3, 3]])
    old = jnp.zeros_like(new)
    result = steps.where_mask(mask, new, old)
    expected = np.array([[1, 1], [0, 0], [3, 3]])
    np.testing.assert_array_equal(result, expected)


def test_make_chunk_runner_with_partial_active_steps():
    """Test that chunk runner correctly handles fewer active steps than chunk size."""
    batch_size = 2
    _chunk_size = 10
    run_chunk = runners.make_chunk_runner(_dummy_train_step_fn(batch_size), batch_size)

    key = jax.random.PRNGKey(42)
    agent_state = jnp.array(0.0, dtype=jnp.float32)
    env_state = jnp.array(0.0, dtype=jnp.float32)
    buffer_state = jnp.array(0.0, dtype=jnp.float32)

    # Only 3 active steps out of 10 total chunk size
    active_mask = jnp.array([True, True, True, False, False, False, False, False, False, False])

    (key_out, agent_out, env_out, buffer_out), metrics_history = run_chunk(
        (key, agent_state, env_state, buffer_state),
        active_mask,
    )

    # State should only advance by 3 (number of active steps)
    assert agent_out == pytest.approx(3.0)
    assert env_out == pytest.approx(3.0)
    assert buffer_out == pytest.approx(3.0)

    # Metrics should have full chunk_size length but only first 3 are non-zero
    assert metrics_history["metric"].shape == (10, 1)
    assert metrics_history["metric"][0] == pytest.approx(batch_size)
    assert metrics_history["metric"][1] == pytest.approx(batch_size)
    assert metrics_history["metric"][2] == pytest.approx(batch_size)
    # Inactive steps have zeroed metrics
    assert metrics_history["metric"][3] == pytest.approx(0.0)
    assert metrics_history["metric"][9] == pytest.approx(0.0)


def test_make_chunk_runner_with_single_step_chunks():
    """Test edge case where chunk_size effectively equals 1."""
    batch_size = 1
    run_chunk = runners.make_chunk_runner(_dummy_train_step_fn(batch_size), batch_size)

    key = jax.random.PRNGKey(123)
    agent_state = jnp.array(5.0, dtype=jnp.float32)
    env_state = jnp.array(10.0, dtype=jnp.float32)
    buffer_state = jnp.array(15.0, dtype=jnp.float32)

    # Single active step
    active_mask = jnp.array([True])

    (key_out, agent_out, env_out, buffer_out), metrics_history = run_chunk(
        (key, agent_state, env_state, buffer_state),
        active_mask,
    )

    assert agent_out == pytest.approx(6.0)
    assert env_out == pytest.approx(11.0)
    assert buffer_out == pytest.approx(16.0)
    assert metrics_history["metric"].shape == (1, 1)
    assert metrics_history["metric"][0] == pytest.approx(batch_size)


def test_make_chunk_runner_all_inactive_steps():
    """Test that chunk runner preserves state when all steps are inactive."""
    batch_size = 2
    run_chunk = runners.make_chunk_runner(_dummy_train_step_fn(batch_size), batch_size)

    key = jax.random.PRNGKey(999)
    agent_state = jnp.array(100.0, dtype=jnp.float32)
    env_state = jnp.array(200.0, dtype=jnp.float32)
    buffer_state = jnp.array(300.0, dtype=jnp.float32)

    # All steps inactive
    active_mask = jnp.array([False, False, False, False, False])

    (key_out, agent_out, env_out, buffer_out), metrics_history = run_chunk(
        (key, agent_state, env_state, buffer_state),
        active_mask,
    )

    # State should be unchanged
    assert agent_out == pytest.approx(100.0)
    assert env_out == pytest.approx(200.0)
    assert buffer_out == pytest.approx(300.0)

    # All metrics should be zero
    assert metrics_history["metric"].shape == (5, 1)
    np.testing.assert_allclose(metrics_history["metric"], np.zeros((5, 1)))


@struct.dataclass
class _DummyEnvState:
    counter: chex.Array


@struct.dataclass
class _DummyAgentState:
    step_count: chex.Array


def _make_dummy_collection_step(num_envs: int):
    """Create a simple collection step function for testing chunked collector."""

    def collection_step(key, agent_state, env_states):
        # Increment counters
        new_env_counter = env_states.counter + 1
        new_agent_step = agent_state.step_count + 1

        # Create dummy transition with meaningful values
        obs = jnp.full((num_envs,), env_states.counter[0], dtype=jnp.float32)
        actions = jnp.ones((num_envs,), dtype=jnp.int32)
        rewards = jnp.full((num_envs,), env_states.counter[0] * 0.1, dtype=jnp.float32)
        next_obs = jnp.full((num_envs,), new_env_counter[0], dtype=jnp.float32)
        dones = jnp.zeros((num_envs,), dtype=jnp.bool_)

        transition = Transition(obs, actions, rewards, next_obs, dones)

        new_env_states = _DummyEnvState(counter=new_env_counter)
        new_agent_state = _DummyAgentState(step_count=new_agent_step)

        key, _ = jax.random.split(key)
        return (key, new_agent_state, new_env_states), transition

    return collection_step


def test_make_chunked_collector_basic_functionality():
    """Test that chunked collector correctly collects rollouts across chunks."""
    num_envs = 3
    chunk_size = 4
    total_steps = 10  # Will require 3 chunks (4 + 4 + 2)

    collection_step = _make_dummy_collection_step(num_envs)
    collect_rollout = runners.make_chunked_collector(collection_step, num_envs, chunk_size, total_steps)

    key = jax.random.PRNGKey(42)
    agent_state = _DummyAgentState(step_count=jnp.array(0, dtype=jnp.int32))
    env_states = _DummyEnvState(counter=jnp.zeros((num_envs,), dtype=jnp.int32))

    key_out, agent_out, env_out, transitions = collect_rollout(key, agent_state, env_states)

    # Agent should have taken total_steps
    assert int(agent_out.step_count) == total_steps

    # Env counters should have advanced by total_steps
    np.testing.assert_array_equal(env_out.counter, np.full((num_envs,), total_steps))

    # Transitions should be shaped (total_steps * num_envs, ...)
    expected_batch_size = total_steps * num_envs
    assert transitions.obs.shape[0] == expected_batch_size
    assert transitions.action.shape[0] == expected_batch_size
    assert transitions.reward.shape[0] == expected_batch_size
    assert transitions.next_obs.shape[0] == expected_batch_size
    assert transitions.done.shape[0] == expected_batch_size

    # Verify key changed
    assert not np.array_equal(key, key_out)


def test_make_chunked_collector_with_perfect_chunk_alignment():
    """Test chunked collector when total_steps divides evenly into chunk_size."""
    num_envs = 2
    chunk_size = 5
    total_steps = 15  # Exactly 3 chunks

    collection_step = _make_dummy_collection_step(num_envs)
    collect_rollout = runners.make_chunked_collector(collection_step, num_envs, chunk_size, total_steps)

    key = jax.random.PRNGKey(123)
    agent_state = _DummyAgentState(step_count=jnp.array(0, dtype=jnp.int32))
    env_states = _DummyEnvState(counter=jnp.zeros((num_envs,), dtype=jnp.int32))

    key_out, agent_out, env_out, transitions = collect_rollout(key, agent_state, env_states)

    assert int(agent_out.step_count) == total_steps
    assert transitions.obs.shape[0] == total_steps * num_envs


def test_make_chunked_collector_single_chunk():
    """Test chunked collector when total_steps < chunk_size (single chunk)."""
    num_envs = 4
    chunk_size = 20
    total_steps = 7  # Less than one full chunk

    collection_step = _make_dummy_collection_step(num_envs)
    collect_rollout = runners.make_chunked_collector(collection_step, num_envs, chunk_size, total_steps)

    key = jax.random.PRNGKey(456)
    agent_state = _DummyAgentState(step_count=jnp.array(0, dtype=jnp.int32))
    env_states = _DummyEnvState(counter=jnp.zeros((num_envs,), dtype=jnp.int32))

    key_out, agent_out, env_out, transitions = collect_rollout(key, agent_state, env_states)

    assert int(agent_out.step_count) == total_steps
    np.testing.assert_array_equal(env_out.counter, np.full((num_envs,), total_steps))
    assert transitions.obs.shape[0] == total_steps * num_envs
