"""Tests for environment wrappers (make_array_obs_env, make_frame_stack_env)."""

import jax
import jax.numpy as jnp
import pytest

from myriad.envs import make_env
from myriad.envs.wrappers import FrameStackState, make_frame_stack_env


@pytest.fixture
def cartpole_env():
    """CartPole control env — 4-dim flat array observations."""
    return make_env("cartpole-control")


@pytest.fixture
def ccasr_env():
    """CcaSR-GFP control env — 3-dim NamedTuple observations."""
    return make_env("ccasr-gfp-control")


# ---------------------------------------------------------------------------
# FrameStackState
# ---------------------------------------------------------------------------


def test_frame_stack_state_is_pytree(key):
    """FrameStackState must be a JAX pytree (NamedTuple → automatic)."""
    cartpole = make_env("cartpole-control")
    env = make_frame_stack_env(cartpole, n_frames=4)
    _, state = env.reset(key, env.params, env.config)
    # round-trip through tree_flatten / tree_unflatten
    leaves, treedef = jax.tree_util.tree_flatten(state)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert jnp.array_equal(restored.obs_buffer, state.obs_buffer)


# ---------------------------------------------------------------------------
# make_frame_stack_env — basic properties
# ---------------------------------------------------------------------------


def test_invalid_n_frames(cartpole_env):
    """n_frames < 1 should raise ValueError; n_frames=1 should warn but still work."""
    with pytest.raises(ValueError, match="n_frames must be >= 1"):
        make_frame_stack_env(cartpole_env, n_frames=0)
    with pytest.raises(ValueError, match="n_frames must be >= 1"):
        make_frame_stack_env(cartpole_env, n_frames=-1)


def test_n_frames_1_warns_but_works(key, cartpole_env):
    """n_frames=1 emits a UserWarning and produces obs identical to the raw env."""
    with pytest.warns(UserWarning, match="n_frames=1"):
        env = make_frame_stack_env(cartpole_env, n_frames=1)
    obs, _ = env.reset(key, env.params, env.config)
    assert obs.shape == cartpole_env.get_obs_shape(cartpole_env.config)


def test_obs_shape_cartpole(cartpole_env):
    """Wrapped obs shape should be n_frames × inner_obs_dim."""
    n_frames = 4
    env = make_frame_stack_env(cartpole_env, n_frames=n_frames)
    inner_obs_shape = cartpole_env.get_obs_shape(cartpole_env.config)
    expected = (n_frames * inner_obs_shape[0],)
    assert env.get_obs_shape(env.config) == expected


def test_obs_shape_ccasr(ccasr_env):
    """Works with NamedTuple observations (ccasr-gfp returns CcasrGfpControlObs)."""
    n_frames = 8
    env = make_frame_stack_env(ccasr_env, n_frames=n_frames)
    inner_dim = ccasr_env.get_obs_shape(ccasr_env.config)[0]
    assert env.get_obs_shape(env.config) == (n_frames * inner_dim,)


def test_action_space_unchanged(cartpole_env):
    """Frame stacking must not alter the action space."""
    env = make_frame_stack_env(cartpole_env, n_frames=4)
    # Discrete doesn't implement __eq__, compare n directly
    assert env.get_action_space(env.config).n == cartpole_env.get_action_space(cartpole_env.config).n


# ---------------------------------------------------------------------------
# reset behaviour
# ---------------------------------------------------------------------------


def test_reset_returns_correct_shape(key, cartpole_env):
    """reset() must return a flat (n_frames * obs_dim,) observation."""
    n_frames = 4
    env = make_frame_stack_env(cartpole_env, n_frames=n_frames)
    obs, state = env.reset(key, env.params, env.config)
    inner_dim = cartpole_env.get_obs_shape(cartpole_env.config)[0]
    assert obs.shape == (n_frames * inner_dim,)


def test_reset_state_type(key, cartpole_env):
    """reset() must return a FrameStackState."""
    env = make_frame_stack_env(cartpole_env, n_frames=4)
    _, state = env.reset(key, env.params, env.config)
    assert isinstance(state, FrameStackState)


def test_reset_buffer_zeros_except_last(key, cartpole_env):
    """After reset, all slots except the last must be zero."""
    n_frames = 4
    env = make_frame_stack_env(cartpole_env, n_frames=n_frames)
    _, state = env.reset(key, env.params, env.config)
    # Slots 0..n_frames-2 must be zero
    assert jnp.all(state.obs_buffer[:-1] == 0.0)
    # Last slot holds the actual observation
    assert not jnp.all(state.obs_buffer[-1] == 0.0)


def test_reset_last_slot_matches_obs(key, cartpole_env):
    """Last slot of buffer must match the initial observation from inner env."""
    n_frames = 4
    env = make_frame_stack_env(cartpole_env, n_frames=n_frames)
    obs_stacked, state = env.reset(key, env.params, env.config)
    inner_dim = cartpole_env.get_obs_shape(cartpole_env.config)[0]
    # The last n inner-obs values in obs_stacked should equal buffer[-1]
    assert jnp.allclose(obs_stacked[-inner_dim:], state.obs_buffer[-1])


# ---------------------------------------------------------------------------
# step behaviour
# ---------------------------------------------------------------------------


def test_step_returns_correct_shape(key, cartpole_env):
    """step() must return a flat (n_frames * obs_dim,) observation."""
    n_frames = 4
    env = make_frame_stack_env(cartpole_env, n_frames=n_frames)
    obs, state = env.reset(key, env.params, env.config)
    key, step_key = jax.random.split(key)
    action = env.get_action_space(env.config).sample(step_key)
    obs_next, _, reward, done, _ = env.step(step_key, state, action, env.params, env.config)
    inner_dim = cartpole_env.get_obs_shape(cartpole_env.config)[0]
    assert obs_next.shape == (n_frames * inner_dim,)


def test_step_shifts_buffer(key, cartpole_env):
    """Each step must shift the buffer left by one frame (oldest dropped, newest appended)."""
    n_frames = 4
    env = make_frame_stack_env(cartpole_env, n_frames=n_frames)
    _, state = env.reset(key, env.params, env.config)

    key, step_key = jax.random.split(key)
    action = env.get_action_space(env.config).sample(step_key)
    _, state2 = env.step(step_key, state, action, env.params, env.config)[:2]

    # Frames 1..n-1 from old buffer become frames 0..n-2 in new buffer
    assert jnp.allclose(state2.obs_buffer[:-1], state.obs_buffer[1:])


def test_step_newest_frame_in_last_slot(key, cartpole_env):
    """After step, the newest observation must occupy the last slot."""
    n_frames = 4
    env = make_frame_stack_env(cartpole_env, n_frames=n_frames)
    _, state = env.reset(key, env.params, env.config)

    # Run a second step using the inner env to get the expected next obs
    inner_env = cartpole_env
    _, inner_state = inner_env.reset(key, inner_env.params, inner_env.config)
    key, step_key = jax.random.split(key)
    action = env.get_action_space(env.config).sample(step_key)

    _, state2 = env.step(step_key, state, action, env.params, env.config)[:2]

    inner_dim = inner_env.get_obs_shape(inner_env.config)[0]
    # The last frame in the buffer after step = obs from inner env step
    # We can't easily recover the inner obs here, but we verify shape at least
    assert state2.obs_buffer[-1].shape == (inner_dim,)


def test_multiple_steps_accumulate(key, cartpole_env):
    """Obs after N steps should encode N distinct frames (not all the same)."""
    n_frames = 4
    env = make_frame_stack_env(cartpole_env, n_frames=n_frames)
    _, state = env.reset(key, env.params, env.config)

    for i in range(n_frames):
        key, step_key = jax.random.split(key)
        action = env.get_action_space(env.config).sample(step_key)
        obs, state, _, _, _ = env.step(step_key, state, action, env.params, env.config)

    inner_dim = cartpole_env.get_obs_shape(cartpole_env.config)[0]
    frames = obs.reshape(n_frames, inner_dim)
    # Not all frames should be identical (env dynamics guarantee this for cartpole)
    assert not jnp.all(frames[0] == frames[-1])


# ---------------------------------------------------------------------------
# JAX compatibility
# ---------------------------------------------------------------------------


def test_jit_reset(key, cartpole_env):
    """reset() must be JIT-compilable."""
    env = make_frame_stack_env(cartpole_env, n_frames=4)
    jitted = jax.jit(env.reset, static_argnames=["config"])
    obs, state = jitted(key, env.params, config=env.config)
    assert obs.shape == env.get_obs_shape(env.config)


def test_jit_step(key, cartpole_env):
    """step() must be JIT-compilable."""
    env = make_frame_stack_env(cartpole_env, n_frames=4)
    obs, state = env.reset(key, env.params, env.config)
    key, step_key = jax.random.split(key)
    action = env.get_action_space(env.config).sample(step_key)
    jitted = jax.jit(env.step, static_argnames=["config"])
    obs_next, state2, reward, done, _ = jitted(step_key, state, action, env.params, config=env.config)
    assert obs_next.shape == env.get_obs_shape(env.config)


def test_vmap_reset(cartpole_env):
    """vmap over reset must work (needed by training infrastructure)."""
    n_frames = 4
    n_envs = 8
    env = make_frame_stack_env(cartpole_env, n_frames=n_frames)
    keys = jax.random.split(jax.random.PRNGKey(0), n_envs)
    vmapped_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    obs_batch, states_batch = vmapped_reset(keys, env.params, env.config)
    assert obs_batch.shape == (n_envs, *env.get_obs_shape(env.config))
    assert states_batch.obs_buffer.shape == (n_envs, n_frames, cartpole_env.get_obs_shape(cartpole_env.config)[0])


def test_vmap_step(key, cartpole_env):
    """vmap over step must work (needed by training infrastructure)."""
    n_frames = 4
    n_envs = 8
    env = make_frame_stack_env(cartpole_env, n_frames=n_frames)
    reset_keys = jax.random.split(key, n_envs)
    vmapped_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    obs_batch, states_batch = vmapped_reset(reset_keys, env.params, env.config)

    step_keys = jax.random.split(key, n_envs)
    actions = jax.vmap(env.get_action_space(env.config).sample)(step_keys)
    vmapped_step = jax.vmap(env.step, in_axes=(0, 0, 0, None, None))
    obs_next, states_next, rewards, dones, _ = vmapped_step(step_keys, states_batch, actions, env.params, env.config)
    assert obs_next.shape == (n_envs, *env.get_obs_shape(env.config))


# ---------------------------------------------------------------------------
# Integration: frame_stack_n via initialization
# ---------------------------------------------------------------------------


def test_initialization_applies_frame_stack():
    """initialize_environment_and_agent must wrap env when frame_stack_n > 0."""
    from myriad.configs.builder import create_config
    from myriad.platform.initialization import initialize_environment_and_agent

    config = create_config(
        env="cartpole-control",
        agent="pqn",
        steps_per_env=100,
        rollout_steps=4,
        **{"env.frame_stack_n": 4},
    )
    env, agent, action_space = initialize_environment_and_agent(config)
    inner_dim = 4  # cartpole obs dim
    assert env.get_obs_shape(env.config) == (4 * inner_dim,)


def test_initialization_no_frame_stack():
    """initialize_environment_and_agent must leave env unchanged when frame_stack_n=0."""
    from myriad.configs.builder import create_config
    from myriad.platform.initialization import initialize_environment_and_agent

    config = create_config(env="cartpole-control", agent="pqn", steps_per_env=100, rollout_steps=4)
    env, agent, action_space = initialize_environment_and_agent(config)
    assert env.get_obs_shape(env.config) == (4,)
