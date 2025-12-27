"""Tests for CartPole rendering utilities."""

import numpy as np
import pytest

from myriad.utils.rendering.cartpole import render_cartpole_frame


class TestRenderCartPoleFrame:
    """Tests for render_cartpole_frame function."""

    @pytest.fixture
    def upright_state(self) -> np.ndarray:
        """Fixture for upright CartPole state."""
        # [x, x_dot, theta, theta_dot] - cart centered, pole upright
        return np.array([0.0, 0.0, 0.0, 0.0])

    @pytest.fixture
    def tilted_state(self) -> np.ndarray:
        """Fixture for tilted CartPole state."""
        # Cart slightly right, pole tilted 0.2 radians (~11 degrees)
        return np.array([0.5, 0.1, 0.2, 0.05])

    @pytest.fixture
    def extreme_state(self) -> np.ndarray:
        """Fixture for extreme CartPole state near failure."""
        # Cart near edge, pole nearly horizontal
        return np.array([2.0, 0.0, 0.75, 0.0])

    def test_frame_shape_default(self, upright_state: np.ndarray):
        """Test that rendered frame has correct shape with default parameters."""
        frame = render_cartpole_frame(upright_state)
        assert frame.ndim == 3, "Frame should be 3D array (height, width, channels)"
        assert frame.shape[-1] == 3, "Frame should have 3 color channels (RGB)"

    def test_frame_dtype(self, upright_state: np.ndarray):
        """Test that rendered frame has correct dtype."""
        frame = render_cartpole_frame(upright_state)
        assert frame.dtype == np.uint8, "Frame should be uint8 for image data"

    def test_frame_values_in_range(self, upright_state: np.ndarray):
        """Test that pixel values are in valid range [0, 255]."""
        frame = render_cartpole_frame(upright_state)
        assert np.all(frame >= 0), "Pixel values should be >= 0"
        assert np.all(frame <= 255), "Pixel values should be <= 255"

    def test_different_states_produce_different_frames(self, upright_state: np.ndarray, tilted_state: np.ndarray):
        """Test that different states produce different visual frames."""
        frame1 = render_cartpole_frame(upright_state)
        frame2 = render_cartpole_frame(tilted_state)
        # Frames should not be identical for different states
        assert not np.array_equal(frame1, frame2), "Different states should produce different frames"

    def test_custom_pole_length(self, upright_state: np.ndarray):
        """Test rendering with custom pole length."""
        frame_short = render_cartpole_frame(upright_state, pole_length=0.5)
        frame_long = render_cartpole_frame(upright_state, pole_length=2.0)
        # Both should render successfully
        assert frame_short.shape[-1] == 3
        assert frame_long.shape[-1] == 3
        # Frames should differ due to different pole lengths
        assert not np.array_equal(frame_short, frame_long)

    def test_custom_cart_dimensions(self, upright_state: np.ndarray):
        """Test rendering with custom cart dimensions."""
        frame = render_cartpole_frame(upright_state, cart_width=1.0, cart_height=0.5)
        assert frame.shape[-1] == 3, "Should render successfully with custom cart dimensions"

    def test_custom_x_limit(self, upright_state: np.ndarray):
        """Test rendering with custom track limits."""
        frame = render_cartpole_frame(upright_state, x_limit=5.0)
        assert frame.shape[-1] == 3, "Should render successfully with custom x_limit"

    def test_custom_figsize_and_dpi(self, upright_state: np.ndarray):
        """Test rendering with custom figure size and DPI."""
        frame_small = render_cartpole_frame(upright_state, figsize=(4, 3), dpi=50)
        frame_large = render_cartpole_frame(upright_state, figsize=(8, 6), dpi=150)
        # Both should render successfully
        assert frame_small.shape[-1] == 3
        assert frame_large.shape[-1] == 3
        # Larger DPI and figsize should produce larger image
        assert frame_large.size > frame_small.size

    def test_extreme_state_renders(self, extreme_state: np.ndarray):
        """Test that extreme states can be rendered without errors."""
        frame = render_cartpole_frame(extreme_state)
        assert frame.shape[-1] == 3, "Extreme state should render successfully"
        assert frame.dtype == np.uint8

    def test_negative_position_renders(self, upright_state: np.ndarray):
        """Test rendering with cart at negative x position."""
        negative_state = upright_state.copy()
        negative_state[0] = -1.5  # Cart at x=-1.5
        frame = render_cartpole_frame(negative_state)
        assert frame.shape[-1] == 3

    def test_negative_angle_renders(self, upright_state: np.ndarray):
        """Test rendering with negative angle (left tilt)."""
        left_tilt_state = upright_state.copy()
        left_tilt_state[2] = -0.3  # Tilt left
        frame = render_cartpole_frame(left_tilt_state)
        assert frame.shape[-1] == 3

    def test_state_shape_validation(self):
        """Test that function handles state array shape correctly."""
        # Should work with 1D array of length 4
        state = np.array([0.0, 0.0, 0.0, 0.0])
        frame = render_cartpole_frame(state)
        assert frame.shape[-1] == 3

    def test_multiple_sequential_renders(self, upright_state: np.ndarray, tilted_state: np.ndarray):
        """Test that multiple sequential renders work correctly (no state leakage)."""
        # Render multiple frames in sequence
        frame1 = render_cartpole_frame(upright_state)
        frame2 = render_cartpole_frame(tilted_state)
        frame3 = render_cartpole_frame(upright_state)
        # First and third should be identical (same state)
        assert np.array_equal(frame1, frame3), "Same state should produce identical frames"
        # Second should differ
        assert not np.array_equal(frame1, frame2)

    def test_frame_is_not_empty(self, upright_state: np.ndarray):
        """Test that rendered frame contains actual content (not all zeros or all one color)."""
        frame = render_cartpole_frame(upright_state)
        # Frame should have some variation (not all pixels the same)
        assert frame.std() > 0, "Frame should contain visual content with variation"
