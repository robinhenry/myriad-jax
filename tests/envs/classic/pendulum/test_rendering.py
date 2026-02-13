"""Tests for Pendulum rendering utilities."""

import jax.numpy as jnp
import numpy as np
import pytest

from myriad.envs.classic.pendulum.physics import PhysicsState
from myriad.envs.classic.pendulum.rendering import render_frame
from myriad.envs.classic.pendulum.tasks.control import ControlTaskConfig, ControlTaskState


class TestRenderPendulumFrame:
    """Tests for render_pendulum_frame function."""

    @pytest.fixture
    def config(self) -> ControlTaskConfig:
        """Fixture for task configuration."""
        return ControlTaskConfig()

    @pytest.fixture
    def hanging_state(self) -> ControlTaskState:
        """Fixture for hanging Pendulum state (theta=0)."""
        physics = PhysicsState(theta=jnp.array(0.0), theta_dot=jnp.array(0.0))
        return ControlTaskState(physics=physics, t=jnp.array(0))

    @pytest.fixture
    def upright_state(self) -> ControlTaskState:
        """Fixture for upright Pendulum state (theta=pi)."""
        physics = PhysicsState(theta=jnp.array(jnp.pi), theta_dot=jnp.array(0.0))
        return ControlTaskState(physics=physics, t=jnp.array(0))

    @pytest.fixture
    def tilted_state(self) -> ControlTaskState:
        """Fixture for tilted Pendulum state."""
        physics = PhysicsState(theta=jnp.array(0.5), theta_dot=jnp.array(0.1))
        return ControlTaskState(physics=physics, t=jnp.array(0))

    def test_frame_shape_default(self, hanging_state: ControlTaskState, config: ControlTaskConfig):
        """Test that rendered frame has correct shape with default parameters."""
        frame = render_frame(hanging_state, config)
        assert frame.ndim == 3, "Frame should be 3D array (height, width, channels)"
        assert frame.shape[-1] == 3, "Frame should have 3 color channels (RGB)"

    def test_frame_dtype(self, hanging_state: ControlTaskState, config: ControlTaskConfig):
        """Test that rendered frame has correct dtype."""
        frame = render_frame(hanging_state, config)
        assert frame.dtype == np.uint8, "Frame should be uint8 for image data"

    def test_frame_values_in_range(self, hanging_state: ControlTaskState, config: ControlTaskConfig):
        """Test that pixel values are in valid range [0, 255]."""
        frame = render_frame(hanging_state, config)
        assert np.all(frame >= 0), "Pixel values should be >= 0"
        assert np.all(frame <= 255), "Pixel values should be <= 255"

    def test_different_states_produce_different_frames(
        self, hanging_state: ControlTaskState, upright_state: ControlTaskState, config: ControlTaskConfig
    ):
        """Test that different states produce different visual frames."""
        frame1 = render_frame(hanging_state, config)
        frame2 = render_frame(upright_state, config)
        assert not np.array_equal(frame1, frame2), "Different states should produce different frames"

    def test_custom_length(self, tilted_state: ControlTaskState, config: ControlTaskConfig):
        """Test rendering with custom pendulum length works without error."""
        frame_short = render_frame(tilted_state, config, length=0.5)
        frame_long = render_frame(tilted_state, config, length=2.0)
        # Both should render successfully
        assert frame_short.shape[-1] == 3
        assert frame_long.shape[-1] == 3
        # Both should have visual content
        assert frame_short.std() > 0
        assert frame_long.std() > 0

    def test_custom_figsize_and_dpi(self, hanging_state: ControlTaskState, config: ControlTaskConfig):
        """Test rendering with custom figure size and DPI."""
        frame_small = render_frame(hanging_state, config, figsize=(3, 3), dpi=50)
        frame_large = render_frame(hanging_state, config, figsize=(6, 6), dpi=150)
        # Both should render successfully
        assert frame_small.shape[-1] == 3
        assert frame_large.shape[-1] == 3
        # Larger DPI and figsize should produce larger image
        assert frame_large.size > frame_small.size

    def test_tilted_state_renders(self, tilted_state: ControlTaskState, config: ControlTaskConfig):
        """Test that tilted states can be rendered without errors."""
        frame = render_frame(tilted_state, config)
        assert frame.shape[-1] == 3, "Tilted state should render successfully"
        assert frame.dtype == np.uint8

    def test_negative_angle_renders(self, config: ControlTaskConfig):
        """Test rendering with negative angle."""
        physics = PhysicsState(theta=jnp.array(-0.5), theta_dot=jnp.array(0.0))
        state = ControlTaskState(physics=physics, t=jnp.array(0))
        frame = render_frame(state, config)
        assert frame.shape[-1] == 3

    def test_multiple_sequential_renders(
        self, hanging_state: ControlTaskState, tilted_state: ControlTaskState, config: ControlTaskConfig
    ):
        """Test that multiple sequential renders work correctly (no state leakage)."""
        frame1 = render_frame(hanging_state, config)
        frame2 = render_frame(tilted_state, config)
        frame3 = render_frame(hanging_state, config)
        # First and third should be identical (same state)
        assert np.array_equal(frame1, frame3), "Same state should produce identical frames"
        # Second should differ
        assert not np.array_equal(frame1, frame2)

    def test_frame_is_not_empty(self, hanging_state: ControlTaskState, config: ControlTaskConfig):
        """Test that rendered frame contains actual content."""
        frame = render_frame(hanging_state, config)
        assert frame.std() > 0, "Frame should contain visual content with variation"
