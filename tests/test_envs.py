"""Tests for the toy_problem environment."""
import jax
import jax.numpy as jnp
import pytest
from aion.envs.toy_problem import (
    create_env,
    reset,
    step,
    jit_reset,
    jit_step,
    get_obs,
    EnvState,
    EnvConfig,
)


def test_create_env_params_scalar_target():
    """Test that a scalar target is correctly broadcast to the full episode length."""
    config = EnvConfig(max_steps=50)
    params, _ = create_env(config, x_target=10.0)
    assert params.x_target.shape == (50,)
    assert jnp.all(params.x_target == 10.0)


def test_create_env_params_array_target():
    """Test that a valid array target is accepted."""
    config = EnvConfig(max_steps=100)
    target = jnp.arange(100, dtype=jnp.float32)
    params, _ = create_env(config, x_target=target)
    assert jnp.array_equal(params.x_target, target)


def test_create_env_params_invalid_target_length():
    """Test that an array target with incorrect length raises a ValueError."""
    config = EnvConfig(max_steps=100)
    with pytest.raises(ValueError, match="must match max_steps"):
        create_env(config, x_target=jnp.zeros(50))


def test_reset_function():
    """Test the reset function for correct shapes and initial values."""
    key = jax.random.PRNGKey(0)
    config = EnvConfig(max_steps=10, prediction_horizon=5)
    params, _ = create_env(config)
    obs, state = reset(key, params, config)

    assert obs.shape == (1 + config.prediction_horizon,)
    assert isinstance(state, EnvState)
    assert state.t == 0
    assert config.min_x <= state.x <= config.max_x


def test_step_function_logic():
    """Test a single step for correct state transition, reward, and done signal."""
    key = jax.random.PRNGKey(0)
    config = EnvConfig(max_steps=10)
    params, _ = create_env(config, a=1.0, b=1.0, x_target=0.0)
    
    # Start with a known state
    initial_state = EnvState(x=jnp.array(5.0), t=jnp.array(0))
    action = 0  # Corresponds to u = -1

    # Expected next state: x_next = a*u + b*x = 1*(-1) + 1*5 = 4
    expected_x_next = 4.0
    
    # Expected reward: -|x_next - x_target| = -|4.0 - 0.0| = -4.0
    expected_reward = -4.0

    obs_next, next_state, reward, done, _ = step(key, initial_state, action, params, config)

    assert jnp.isclose(next_state.x, expected_x_next)
    assert next_state.t == 1
    assert jnp.isclose(reward, expected_reward)
    assert done == 0.0


def test_episode_termination():
    """Test that the 'done' signal is correctly triggered at the end of an episode."""
    key = jax.random.PRNGKey(0)
    config = EnvConfig(max_steps=5)
    params, _ = create_env(config)
    state = EnvState(x=jnp.array(5.0), t=jnp.array(4)) # One step before the end
    action = 1

    _, _, _, done, _ = step(key, state, action, params, config)
    assert done == 1.0


def test_state_clipping():
    """Test that the state 'x' is correctly clipped within its bounds."""
    key = jax.random.PRNGKey(0)
    config = EnvConfig(min_x=0.0, max_x=20.0)
    # Set b=0 to ignore previous state, making it easy to force an out-of-bounds state
    params, _ = create_env(config, a=100.0, b=0.0)
    state = EnvState(x=jnp.array(10.0), t=jnp.array(0))
    
    # Action 1 (u=1) would push x to 100, should be clipped to 20
    _, next_state_pos, _, _, _ = step(key, state, 1, params, config)
    assert jnp.isclose(next_state_pos.x, config.max_x)

    # Action 0 (u=-1) would push x to -100, should be clipped to 0
    _, next_state_neg, _, _, _ = step(key, state, 0, params, config)
    assert jnp.isclose(next_state_neg.x, config.min_x)


def test_get_obs_lookahead():
    """Test that the observation correctly includes a window of future targets."""
    config = EnvConfig(prediction_horizon=5, max_steps=20)
    params, _ = create_env(config, x_target=jnp.arange(20, dtype=jnp.float32))
    
    # Test from the middle of the trajectory
    state = EnvState(x=jnp.array(1.0), t=jnp.array(10))
    obs = get_obs(state, params, config)
    expected_targets = jnp.array([10.0, 11.0, 12.0, 13.0, 14.0])
    
    assert obs.shape == (1 + config.prediction_horizon,)
    assert jnp.isclose(obs[0], state.x)
    assert jnp.array_equal(obs[1:], expected_targets)

    # Test at the end of the trajectory, where indices are clipped
    state_end = EnvState(x=jnp.array(1.0), t=jnp.array(18))
    obs_end = get_obs(state_end, params, config)
    expected_targets_end = jnp.array([18.0, 19.0, 19.0, 19.0, 19.0]) # Last value is repeated due to clipping
    
    assert jnp.array_equal(obs_end[1:], expected_targets_end)


# --- JAX Best Practice Tests ---

def test_jit_compilability():
    """Test that the core functions can be J-compiled without error."""
    # This test now uses the explicitly jitted functions from the module
    config = EnvConfig()
    
    try:
        key = jax.random.PRNGKey(0)
        params, _ = create_env(config)
        
        # Test reset
        obs, state = jit_reset(key, params, config)
        
        # Test step
        jit_step(key, state, 0, params, config)

    except Exception as e:
        pytest.fail(f"JIT compilation failed: {e}")


def test_vmap_compatibility():
    """Test that the step function can be vectorized with vmap."""
    num_envs = 4
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, num_envs)
    
    config = EnvConfig()
    params, _ = create_env(config)

    # Create a batch of initial states
    _, single_state = reset(key, params, config)
    batch_state = jax.tree_util.tree_map(lambda x: jnp.array([x] * num_envs), single_state)
    
    # Create a batch of actions
    batch_action = jnp.array([0, 1, 0, 1])

    # vmap the non-jitted step function
    vmapped_step = jax.vmap(step, in_axes=(0, 0, 0, None, None))

    try:
        vmapped_step(keys, batch_state, batch_action, params, config)
    except Exception as e:
        pytest.fail(f"vmap compatibility test failed: {e}")