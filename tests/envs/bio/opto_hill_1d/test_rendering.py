"""Tests for opto_hill_1d rendering utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from myriad.envs.bio.opto_hill_1d.physics import PhysicsState
from myriad.envs.bio.opto_hill_1d.rendering import (
    render_frame,
    render_opto_hill_1d_frame_from_obs,
)
from myriad.envs.bio.opto_hill_1d.tasks.sysid import (
    SysIdTaskConfig,
    SysIdTaskState,
    make_env,
)


@pytest.fixture
def config() -> SysIdTaskConfig:
    return SysIdTaskConfig()


@pytest.fixture
def basic_state() -> SysIdTaskState:
    """Mid-trajectory state with a moderate protein count and a non-trivial U."""
    return SysIdTaskState(
        physics=PhysicsState.create(time=jnp.array(25.0), X=jnp.array(40.0)),
        t=jnp.array(5),
        U=jnp.array(0.7, dtype=jnp.float32),
    )


class TestRenderFrame:
    def test_shape_and_dtype(self, basic_state: SysIdTaskState, config: SysIdTaskConfig):
        frame = render_frame(basic_state, config)
        assert frame.ndim == 3
        assert frame.shape[-1] == 3
        assert frame.dtype == np.uint8

    def test_pixel_values_in_range(self, basic_state: SysIdTaskState, config: SysIdTaskConfig):
        frame = render_frame(basic_state, config)
        assert frame.min() >= 0
        assert frame.max() <= 255

    def test_different_X_produces_different_frames(self, config: SysIdTaskConfig):
        low = SysIdTaskState(
            physics=PhysicsState.create(time=jnp.array(0.0), X=jnp.array(5.0)),
            t=jnp.array(0),
            U=jnp.array(0.0, dtype=jnp.float32),
        )
        high = SysIdTaskState(
            physics=PhysicsState.create(time=jnp.array(0.0), X=jnp.array(90.0)),
            t=jnp.array(0),
            U=jnp.array(0.0, dtype=jnp.float32),
        )
        frame_low = render_frame(low, config)
        frame_high = render_frame(high, config)
        assert not np.array_equal(frame_low, frame_high)

    def test_show_action_strip_true_is_taller_than_false(self, basic_state: SysIdTaskState, config: SysIdTaskConfig):
        """Suppressing the action strip removes a subplot — the rendered frames
        should differ pixel-wise even though the figure size is the same."""
        with_strip = render_frame(basic_state, config, show_action_strip=True)
        without_strip = render_frame(basic_state, config, show_action_strip=False)
        assert with_strip.shape == without_strip.shape  # same figsize
        assert not np.array_equal(with_strip, without_strip)

    def test_with_history(self, config: SysIdTaskConfig):
        history = [
            SysIdTaskState(
                physics=PhysicsState.create(time=jnp.array(float(i)), X=jnp.array(float(i * 2))),
                t=jnp.array(i),
                U=jnp.array(float(i) / 10.0, dtype=jnp.float32),
            )
            for i in range(5)
        ]
        actions = [float(i) / 10.0 for i in range(5)]
        frame = render_frame(
            history[-1],
            config,
            trajectory_history=history,
            action_history=actions,
        )
        assert frame.shape[-1] == 3
        assert frame.dtype == np.uint8


class TestRenderFromObs:
    def test_shape_and_dtype(self):
        obs = np.array([0.5], dtype=np.float32)
        frame = render_opto_hill_1d_frame_from_obs(obs)
        assert frame.ndim == 3
        assert frame.shape[-1] == 3
        assert frame.dtype == np.uint8

    def test_different_obs_produces_different_frames(self):
        frame_low = render_opto_hill_1d_frame_from_obs(np.array([0.1], dtype=np.float32))
        frame_high = render_opto_hill_1d_frame_from_obs(np.array([0.9], dtype=np.float32))
        assert not np.array_equal(frame_low, frame_high)

    def test_end_to_end_from_eval_rollout(self):
        """Render a real observation produced by env.step — integration check."""
        env = make_env()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, env.params, env.config)
        for t in range(5):
            obs, state, _, _, _ = env.step(
                jax.random.PRNGKey(t + 1),
                state,
                jnp.array(0.8, dtype=jnp.float32),
                env.params,
                env.config,
            )
        frame = render_opto_hill_1d_frame_from_obs(np.asarray(obs.to_array()))
        assert frame.dtype == np.uint8
        assert frame.shape[-1] == 3
