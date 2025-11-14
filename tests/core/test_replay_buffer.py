import chex
import jax
import jax.numpy as jnp
import pytest

from aion.core.replay_buffer import ReplayBuffer


def make_sample_transition():
    # simple pytree with an observation vector and scalar reward
    return {"obs": jnp.array([0.0], dtype=jnp.float32), "reward": jnp.array(0.0, dtype=jnp.float32)}


def make_transitions(batch_size: int):
    # create deterministic transitions for testing
    obs = jnp.arange(batch_size, dtype=jnp.float32).reshape(batch_size, 1)
    reward = jnp.arange(batch_size, dtype=jnp.float32)
    return {"obs": obs, "reward": reward}


def test_init_buffer_state_and_shapes():
    buf = ReplayBuffer(buffer_size=4)
    sample = make_sample_transition()
    state = buf.init(sample)

    # initial position and size
    assert int(state.position) == 0
    assert int(state.size) == 0

    # data shapes should have leading buffer dimension
    assert state.data["obs"].shape == (4, 1)
    assert state.data["reward"].shape == (4,)


def test_add_updates_position_size_and_wrap():
    buf = ReplayBuffer(buffer_size=4)
    sample = make_sample_transition()
    state = buf.init(sample)

    # add two transitions (batch size 2)
    t1 = make_transitions(2)
    state = buf.add(state, t1)
    assert int(state.size) == 2
    assert int(state.position) == 2

    # add three more transitions; buffer should wrap and size==buffer_size
    t2 = make_transitions(3)
    state = buf.add(state, t2)
    assert int(state.size) == 4
    assert 0 <= int(state.position) < 4


def test_sample_is_deterministic_given_same_key():
    buf = ReplayBuffer(buffer_size=8)
    sample = make_sample_transition()
    state = buf.init(sample)

    # fill buffer with known values
    transitions = make_transitions(5)
    state = buf.add(state, transitions)

    key = jax.random.PRNGKey(0)
    _, batch1 = buf.sample(state, batch_size=3, key=key)
    _, batch2 = buf.sample(state, batch_size=3, key=key)

    # sampling with the same key should be identical
    chex.assert_trees_all_close(batch1, batch2)


def test_add_rejects_empty_pytree():
    buf = ReplayBuffer(buffer_size=4)
    sample = make_sample_transition()
    state = buf.init(sample)

    with pytest.raises(ValueError):
        buf.add(state, {})
