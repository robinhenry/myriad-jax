from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aion.platform import scan_utils


def test_tree_select_masking():
    mask_scalar = jnp.array(True)
    new_tree = {"value": jnp.array(2.0)}
    old_tree = {"value": jnp.array(-1.0)}
    result = scan_utils.tree_select(mask_scalar, new_tree, old_tree)
    assert np.array(result["value"]) == pytest.approx(2.0)

    mask = jnp.array([True, False], dtype=jnp.bool_)
    new_value = jnp.array([5.0, 10.0])
    old_value = jnp.array([1.0, 1.0])
    expanded = scan_utils._expand_mask(mask, target_ndim=2)
    assert expanded.shape == (2, 1)
    where_result = scan_utils.where_mask(mask, new_value, old_value)
    np.testing.assert_allclose(where_result, np.array([5.0, 1.0]))
    mask_tree_result = scan_utils.mask_tree(mask, {"value": new_value}, {"value": old_value})
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
    run_chunk = scan_utils.make_chunk_runner(_dummy_train_step_fn(batch_size), batch_size)

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
    result = scan_utils.where_mask(mask, new, old)
    expected = np.array([[1, 1], [0, 0], [3, 3]])
    np.testing.assert_array_equal(result, expected)
