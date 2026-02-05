"""Tests for Pendulum rendering utilities."""

import numpy as np
import pytest

from myriad.envs.classic.pendulum.rendering import render_pendulum_frame


class TestRenderPendulumFrame:
    """Tests for render_pendulum_frame function."""

    @pytest.fixture
    def hanging_state(self) -> np.ndarray:
        """Fixture for hanging Pendulum state (theta=0)."""
        return np.array([0.0, 0.0])  # [theta, theta_dot]

    @pytest.fixture
    def upright_state(self) -> np.ndarray:
        """Fixture for upright Pendulum state (theta=pi)."""
        return np.array([np.pi, 0.0])

    @pytest.fixture
    def tilted_state(self) -> np.ndarray:
        """Fixture for tilted Pendulum state."""
        return np.array([0.5, 0.1])

    @pytest.fixture
    def obs_format_state(self) -> np.ndarray:
        """Fixture for observation format state (cos, sin, theta_dot)."""
        theta = 0.5
        return np.array([np.cos(theta), np.sin(theta), 0.0])

    def test_frame_shape_default(self, hanging_state: np.ndarray):
        """Test that rendered frame has correct shape with default parameters."""
        frame = render_pendulum_frame(hanging_state)
        assert frame.ndim == 3, "Frame should be 3D array (height, width, channels)"
        assert frame.shape[-1] == 3, "Frame should have 3 color channels (RGB)"

    def test_frame_dtype(self, hanging_state: np.ndarray):
        """Test that rendered frame has correct dtype."""
        frame = render_pendulum_frame(hanging_state)
        assert frame.dtype == np.uint8, "Frame should be uint8 for image data"

    def test_frame_values_in_range(self, hanging_state: np.ndarray):
        """Test that pixel values are in valid range [0, 255]."""
        frame = render_pendulum_frame(hanging_state)
        assert np.all(frame >= 0), "Pixel values should be >= 0"
        assert np.all(frame <= 255), "Pixel values should be <= 255"

    def test_different_states_produce_different_frames(self, hanging_state: np.ndarray, upright_state: np.ndarray):
        """Test that different states produce different visual frames."""
        frame1 = render_pendulum_frame(hanging_state)
        frame2 = render_pendulum_frame(upright_state)
        assert not np.array_equal(frame1, frame2), "Different states should produce different frames"

    def test_custom_length(self, tilted_state: np.ndarray):
        """Test rendering with custom pendulum length works without error."""
        frame_short = render_pendulum_frame(tilted_state, length=0.5)
        frame_long = render_pendulum_frame(tilted_state, length=2.0)
        # Both should render successfully
        assert frame_short.shape[-1] == 3
        assert frame_long.shape[-1] == 3
        # Both should have visual content
        assert frame_short.std() > 0
        assert frame_long.std() > 0

    def test_custom_figsize_and_dpi(self, hanging_state: np.ndarray):
        """Test rendering with custom figure size and DPI."""
        frame_small = render_pendulum_frame(hanging_state, figsize=(3, 3), dpi=50)
        frame_large = render_pendulum_frame(hanging_state, figsize=(6, 6), dpi=150)
        # Both should render successfully
        assert frame_small.shape[-1] == 3
        assert frame_large.shape[-1] == 3
        # Larger DPI and figsize should produce larger image
        assert frame_large.size > frame_small.size

    def test_tilted_state_renders(self, tilted_state: np.ndarray):
        """Test that tilted states can be rendered without errors."""
        frame = render_pendulum_frame(tilted_state)
        assert frame.shape[-1] == 3, "Tilted state should render successfully"
        assert frame.dtype == np.uint8

    def test_negative_angle_renders(self, hanging_state: np.ndarray):
        """Test rendering with negative angle."""
        negative_state = hanging_state.copy()
        negative_state[0] = -0.5
        frame = render_pendulum_frame(negative_state)
        assert frame.shape[-1] == 3

    def test_observation_format_input(self, obs_format_state: np.ndarray):
        """Test rendering with observation format (cos, sin, theta_dot)."""
        frame = render_pendulum_frame(obs_format_state)
        assert frame.shape[-1] == 3, "Observation format should render successfully"

    def test_multiple_sequential_renders(self, hanging_state: np.ndarray, tilted_state: np.ndarray):
        """Test that multiple sequential renders work correctly (no state leakage)."""
        frame1 = render_pendulum_frame(hanging_state)
        frame2 = render_pendulum_frame(tilted_state)
        frame3 = render_pendulum_frame(hanging_state)
        # First and third should be identical (same state)
        assert np.array_equal(frame1, frame3), "Same state should produce identical frames"
        # Second should differ
        assert not np.array_equal(frame1, frame2)

    def test_frame_is_not_empty(self, hanging_state: np.ndarray):
        """Test that rendered frame contains actual content."""
        frame = render_pendulum_frame(hanging_state)
        assert frame.std() > 0, "Frame should contain visual content with variation"
