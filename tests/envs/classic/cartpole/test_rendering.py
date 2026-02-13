"""Tests for CartPole rendering utilities."""

import jax.numpy as jnp
import numpy as np
import pytest

from myriad.envs.classic.cartpole.physics import PhysicsState
from myriad.envs.classic.cartpole.rendering import render_frame
from myriad.envs.classic.cartpole.tasks.control import ControlTaskConfig, ControlTaskState


class TestRenderCartPoleFrame:
    """Tests for render_cartpole_frame function."""

    @pytest.fixture
    def config(self) -> ControlTaskConfig:
        """Fixture for task configuration."""
        return ControlTaskConfig()

    @pytest.fixture
    def upright_state(self) -> ControlTaskState:
        """Fixture for upright CartPole state."""
        # Cart centered, pole upright
        physics = PhysicsState(
            x=jnp.array(0.0),
            x_dot=jnp.array(0.0),
            theta=jnp.array(0.0),
            theta_dot=jnp.array(0.0),
        )
        return ControlTaskState(physics=physics, t=jnp.array(0))

    @pytest.fixture
    def tilted_state(self) -> ControlTaskState:
        """Fixture for tilted CartPole state."""
        # Cart slightly right, pole tilted 0.2 radians (~11 degrees)
        physics = PhysicsState(
            x=jnp.array(0.5),
            x_dot=jnp.array(0.1),
            theta=jnp.array(0.2),
            theta_dot=jnp.array(0.05),
        )
        return ControlTaskState(physics=physics, t=jnp.array(10))

    @pytest.fixture
    def extreme_state(self) -> ControlTaskState:
        """Fixture for extreme CartPole state near failure."""
        # Cart near edge, pole nearly horizontal
        physics = PhysicsState(
            x=jnp.array(2.0),
            x_dot=jnp.array(0.0),
            theta=jnp.array(0.75),
            theta_dot=jnp.array(0.0),
        )
        return ControlTaskState(physics=physics, t=jnp.array(50))

    def test_frame_shape_default(self, upright_state: ControlTaskState, config: ControlTaskConfig):
        """Test that rendered frame has correct shape with default parameters."""
        frame = render_frame(upright_state, config)
        assert frame.ndim == 3, "Frame should be 3D array (height, width, channels)"
        assert frame.shape[-1] == 3, "Frame should have 3 color channels (RGB)"

    def test_frame_dtype(self, upright_state: ControlTaskState, config: ControlTaskConfig):
        """Test that rendered frame has correct dtype."""
        frame = render_frame(upright_state, config)
        assert frame.dtype == np.uint8, "Frame should be uint8 for image data"

    def test_frame_values_in_range(self, upright_state: ControlTaskState, config: ControlTaskConfig):
        """Test that pixel values are in valid range [0, 255]."""
        frame = render_frame(upright_state, config)
        assert np.all(frame >= 0), "Pixel values should be >= 0"
        assert np.all(frame <= 255), "Pixel values should be <= 255"

    def test_different_states_produce_different_frames(
        self, upright_state: ControlTaskState, tilted_state: ControlTaskState, config: ControlTaskConfig
    ):
        """Test that different states produce different visual frames."""
        frame1 = render_frame(upright_state, config)
        frame2 = render_frame(tilted_state, config)
        # Frames should not be identical for different states
        assert not np.array_equal(frame1, frame2), "Different states should produce different frames"

    def test_custom_pole_length(self, upright_state: ControlTaskState, config: ControlTaskConfig):
        """Test rendering with custom pole length."""
        frame_short = render_frame(upright_state, config, pole_length=0.5)
        frame_long = render_frame(upright_state, config, pole_length=2.0)
        # Both should render successfully
        assert frame_short.shape[-1] == 3
        assert frame_long.shape[-1] == 3
        # Frames should differ due to different pole lengths
        assert not np.array_equal(frame_short, frame_long)

    def test_custom_cart_dimensions(self, upright_state: ControlTaskState, config: ControlTaskConfig):
        """Test rendering with custom cart dimensions."""
        frame = render_frame(upright_state, config, cart_width=1.0, cart_height=0.5)
        assert frame.shape[-1] == 3, "Should render successfully with custom cart dimensions"

    def test_custom_x_limit(self, upright_state: ControlTaskState):
        """Test rendering with custom track limits."""
        config = ControlTaskConfig()
        config = config.replace(task=config.task.replace(x_threshold=5.0))
        frame = render_frame(upright_state, config)
        assert frame.shape[-1] == 3, "Should render successfully with custom x_limit"

    def test_custom_figsize_and_dpi(self, upright_state: ControlTaskState, config: ControlTaskConfig):
        """Test rendering with custom figure size and DPI."""
        frame_small = render_frame(upright_state, config, figsize=(4, 3), dpi=50)
        frame_large = render_frame(upright_state, config, figsize=(8, 6), dpi=150)
        # Both should render successfully
        assert frame_small.shape[-1] == 3
        assert frame_large.shape[-1] == 3
        # Larger DPI and figsize should produce larger image
        assert frame_large.size > frame_small.size

    def test_extreme_state_renders(self, extreme_state: ControlTaskState, config: ControlTaskConfig):
        """Test that extreme states can be rendered without errors."""
        frame = render_frame(extreme_state, config)
        assert frame.shape[-1] == 3, "Extreme state should render successfully"
        assert frame.dtype == np.uint8

    def test_negative_position_renders(self, config: ControlTaskConfig):
        """Test rendering with cart at negative x position."""
        physics = PhysicsState(
            x=jnp.array(-1.5),
            x_dot=jnp.array(0.0),
            theta=jnp.array(0.0),
            theta_dot=jnp.array(0.0),
        )
        state = ControlTaskState(physics=physics, t=jnp.array(0))
        frame = render_frame(state, config)
        assert frame.shape[-1] == 3

    def test_negative_angle_renders(self, config: ControlTaskConfig):
        """Test rendering with negative angle (left tilt)."""
        physics = PhysicsState(
            x=jnp.array(0.0),
            x_dot=jnp.array(0.0),
            theta=jnp.array(-0.3),
            theta_dot=jnp.array(0.0),
        )
        state = ControlTaskState(physics=physics, t=jnp.array(0))
        frame = render_frame(state, config)
        assert frame.shape[-1] == 3

    def test_state_shape_validation(self, config: ControlTaskConfig):
        """Test that function handles state correctly."""
        physics = PhysicsState(
            x=jnp.array(0.0),
            x_dot=jnp.array(0.0),
            theta=jnp.array(0.0),
            theta_dot=jnp.array(0.0),
        )
        state = ControlTaskState(physics=physics, t=jnp.array(0))
        frame = render_frame(state, config)
        assert frame.shape[-1] == 3

    def test_multiple_sequential_renders(
        self, upright_state: ControlTaskState, tilted_state: ControlTaskState, config: ControlTaskConfig
    ):
        """Test that multiple sequential renders work correctly (no state leakage)."""
        # Render multiple frames in sequence
        frame1 = render_frame(upright_state, config)
        frame2 = render_frame(tilted_state, config)
        frame3 = render_frame(upright_state, config)
        # First and third should be identical (same state)
        assert np.array_equal(frame1, frame3), "Same state should produce identical frames"
        # Second should differ
        assert not np.array_equal(frame1, frame2)

    def test_frame_is_not_empty(self, upright_state: ControlTaskState, config: ControlTaskConfig):
        """Test that rendered frame contains actual content (not all zeros or all one color)."""
        frame = render_frame(upright_state, config)
        # Frame should have some variation (not all pixels the same)
        assert frame.std() > 0, "Frame should contain visual content with variation"
