"""Tests for CcaS-CcaR rendering utilities."""

import numpy as np
import pytest

from myriad.utils.rendering.ccas_ccar import render_ccas_ccar_frame, render_population_heatmap


class TestRenderCcasCcarFrame:
    """Tests for render_ccas_ccar_frame function."""

    @pytest.fixture
    def basic_state(self) -> np.ndarray:
        """Fixture for basic CcaS-CcaR state with constant target.

        Observation format: [F_normalized, U_obs, F_target[0], F_target[1]]
        """
        # F=0.3 (normalized), U=0.0, target=[0.31, 0.32]
        return np.array([0.3, 0.0, 0.31, 0.32])

    @pytest.fixture
    def high_fluorescence_state(self) -> np.ndarray:
        """Fixture for high fluorescence state."""
        # High F value near normalization limit
        return np.array([0.9, 0.0, 0.5, 0.52])

    @pytest.fixture
    def low_fluorescence_state(self) -> np.ndarray:
        """Fixture for low fluorescence state."""
        # Low F value close to zero
        return np.array([0.05, 0.0, 0.3, 0.32])

    @pytest.fixture
    def long_horizon_state(self) -> np.ndarray:
        """Fixture for state with longer prediction horizon."""
        # F=0.3, target trajectory with 5 timesteps
        return np.array([0.3, 0.0, 0.31, 0.32, 0.33, 0.34, 0.35])

    def test_frame_shape_default(self, basic_state: np.ndarray):
        """Test that rendered frame has correct shape with default parameters."""
        frame = render_ccas_ccar_frame(basic_state)
        assert frame.ndim == 3, "Frame should be 3D array (height, width, channels)"
        assert frame.shape[-1] == 3, "Frame should have 3 color channels (RGB)"

    def test_frame_dtype(self, basic_state: np.ndarray):
        """Test that rendered frame has correct dtype."""
        frame = render_ccas_ccar_frame(basic_state)
        assert frame.dtype == np.uint8, "Frame should be uint8 for image data"

    def test_frame_values_in_range(self, basic_state: np.ndarray):
        """Test that pixel values are in valid range [0, 255]."""
        frame = render_ccas_ccar_frame(basic_state)
        assert np.all(frame >= 0), "Pixel values should be >= 0"
        assert np.all(frame <= 255), "Pixel values should be <= 255"

    def test_different_states_produce_different_frames(
        self, basic_state: np.ndarray, high_fluorescence_state: np.ndarray
    ):
        """Test that different states produce different visual frames."""
        frame1 = render_ccas_ccar_frame(basic_state)
        frame2 = render_ccas_ccar_frame(high_fluorescence_state)
        # Frames should not be identical for different states
        assert not np.array_equal(frame1, frame2), "Different states should produce different frames"

    def test_low_fluorescence_renders(self, low_fluorescence_state: np.ndarray):
        """Test rendering with low fluorescence values."""
        frame = render_ccas_ccar_frame(low_fluorescence_state)
        assert frame.shape[-1] == 3, "Low fluorescence state should render successfully"
        assert frame.dtype == np.uint8

    def test_high_fluorescence_renders(self, high_fluorescence_state: np.ndarray):
        """Test rendering with high fluorescence values."""
        frame = render_ccas_ccar_frame(high_fluorescence_state)
        assert frame.shape[-1] == 3, "High fluorescence state should render successfully"
        assert frame.dtype == np.uint8

    def test_long_horizon_renders(self, long_horizon_state: np.ndarray):
        """Test rendering with longer prediction horizon."""
        frame = render_ccas_ccar_frame(long_horizon_state)
        assert frame.shape[-1] == 3, "Long horizon state should render successfully"
        assert frame.dtype == np.uint8

    def test_custom_normalizer(self, basic_state: np.ndarray):
        """Test rendering with custom normalization constant."""
        frame1 = render_ccas_ccar_frame(basic_state, F_obs_normalizer=80.0)
        frame2 = render_ccas_ccar_frame(basic_state, F_obs_normalizer=100.0)
        # Both should render successfully
        assert frame1.shape[-1] == 3
        assert frame2.shape[-1] == 3
        # Frames should differ due to different normalization (affects scale)
        assert not np.array_equal(frame1, frame2)

    def test_custom_figsize_and_dpi(self, basic_state: np.ndarray):
        """Test rendering with custom figure size and DPI."""
        frame_small = render_ccas_ccar_frame(basic_state, figsize=(4, 3), dpi=50)
        frame_large = render_ccas_ccar_frame(basic_state, figsize=(10, 8), dpi=150)
        # Both should render successfully
        assert frame_small.shape[-1] == 3
        assert frame_large.shape[-1] == 3
        # Larger DPI and figsize should produce larger image
        assert frame_large.size > frame_small.size

    def test_zero_fluorescence_renders(self):
        """Test rendering with zero fluorescence."""
        state = np.array([0.0, 0.0, 0.3, 0.32])
        frame = render_ccas_ccar_frame(state)
        assert frame.shape[-1] == 3
        assert frame.dtype == np.uint8

    def test_at_target_renders(self):
        """Test rendering when current F equals target."""
        # F exactly at target (error = 0)
        state = np.array([0.5, 0.0, 0.5, 0.5])
        frame = render_ccas_ccar_frame(state)
        assert frame.shape[-1] == 3
        assert frame.dtype == np.uint8

    def test_multiple_sequential_renders(self, basic_state: np.ndarray, high_fluorescence_state: np.ndarray):
        """Test that multiple sequential renders work correctly (no state leakage)."""
        # Render multiple frames in sequence
        frame1 = render_ccas_ccar_frame(basic_state)
        frame2 = render_ccas_ccar_frame(high_fluorescence_state)
        frame3 = render_ccas_ccar_frame(basic_state)
        # First and third should be identical (same state)
        assert np.array_equal(frame1, frame3), "Same state should produce identical frames"
        # Second should differ
        assert not np.array_equal(frame1, frame2)

    def test_frame_is_not_empty(self, basic_state: np.ndarray):
        """Test that rendered frame contains actual content (not all zeros or all one color)."""
        frame = render_ccas_ccar_frame(basic_state)
        # Frame should have some variation (not all pixels the same)
        assert frame.std() > 0, "Frame should contain visual content with variation"

    def test_minimal_observation(self):
        """Test rendering with minimal observation (n_horizon=0)."""
        # Just F and U, with single target value
        state = np.array([0.3, 0.0, 0.31])
        frame = render_ccas_ccar_frame(state)
        assert frame.shape[-1] == 3
        assert frame.dtype == np.uint8

    def test_different_target_trajectories(self):
        """Test that different target trajectories produce different frames."""
        state1 = np.array([0.3, 0.0, 0.31, 0.32])  # Increasing target
        state2 = np.array([0.3, 0.0, 0.31, 0.29])  # Decreasing target
        frame1 = render_ccas_ccar_frame(state1)
        frame2 = render_ccas_ccar_frame(state2)
        # Frames should differ due to different target trajectories
        assert not np.array_equal(frame1, frame2)


class TestRenderPopulationHeatmap:
    """Tests for render_population_heatmap function."""

    @pytest.fixture
    def small_population_obs(self) -> np.ndarray:
        """Fixture for small population observations (100 cells)."""
        # 100 cells with random fluorescence values
        # Observation format: [F_normalized, U_obs, F_target[0], F_target[1]]
        rng = np.random.RandomState(42)
        return rng.rand(100, 4) * 0.5  # Fluorescence in [0, 0.5]

    @pytest.fixture
    def large_population_obs(self) -> np.ndarray:
        """Fixture for large population observations (10K cells)."""
        rng = np.random.RandomState(42)
        return rng.rand(10_000, 4) * 0.8  # Fluorescence in [0, 0.8]

    @pytest.fixture
    def population_actions(self) -> np.ndarray:
        """Fixture for population actions (100 cells)."""
        rng = np.random.RandomState(42)
        return rng.randint(0, 2, size=100)  # Binary actions (0=OFF, 1=ON)

    def test_frame_shape_default(self, small_population_obs: np.ndarray):
        """Test that rendered frame has correct shape with default parameters."""
        frame = render_population_heatmap(small_population_obs)
        assert frame.ndim == 3, "Frame should be 3D array (height, width, channels)"
        assert frame.shape[-1] == 3, "Frame should have 3 color channels (RGB)"

    def test_frame_dtype(self, small_population_obs: np.ndarray):
        """Test that rendered frame has correct dtype."""
        frame = render_population_heatmap(small_population_obs)
        assert frame.dtype == np.uint8, "Frame should be uint8 for image data"

    def test_frame_values_in_range(self, small_population_obs: np.ndarray):
        """Test that pixel values are in valid range [0, 255]."""
        frame = render_population_heatmap(small_population_obs)
        assert np.all(frame >= 0) and np.all(frame <= 255), "Pixel values should be in [0, 255]"

    def test_with_actions(self, small_population_obs: np.ndarray, population_actions: np.ndarray):
        """Test rendering with action overlay."""
        frame = render_population_heatmap(small_population_obs, actions=population_actions)
        assert frame.ndim == 3
        assert frame.dtype == np.uint8

    def test_without_actions(self, small_population_obs: np.ndarray):
        """Test rendering without actions (should work)."""
        frame = render_population_heatmap(small_population_obs, actions=None)
        assert frame.ndim == 3
        assert frame.dtype == np.uint8

    def test_custom_grid_shape(self, small_population_obs: np.ndarray):
        """Test with custom grid shape."""
        # 100 cells in a 10x10 grid
        frame = render_population_heatmap(small_population_obs, grid_shape=(10, 10))
        assert frame.ndim == 3
        assert frame.dtype == np.uint8

    def test_auto_grid_shape(self, small_population_obs: np.ndarray):
        """Test automatic grid shape computation (should make square grid)."""
        # 100 cells should auto-compute to 10x10 grid
        frame = render_population_heatmap(small_population_obs, grid_shape=None)
        assert frame.ndim == 3

    def test_large_population(self, large_population_obs: np.ndarray):
        """Test with large population (10K cells)."""
        # Should automatically create ~100x100 grid
        frame = render_population_heatmap(large_population_obs)
        assert frame.ndim == 3
        assert frame.dtype == np.uint8

    def test_custom_normalizer(self, small_population_obs: np.ndarray):
        """Test with custom fluorescence normalizer."""
        frame = render_population_heatmap(small_population_obs, F_obs_normalizer=100.0)
        assert frame.ndim == 3

    def test_custom_figsize(self, small_population_obs: np.ndarray):
        """Test with custom figure size."""
        frame = render_population_heatmap(small_population_obs, figsize=(8, 6))
        assert frame.ndim == 3

    def test_custom_dpi(self, small_population_obs: np.ndarray):
        """Test with custom DPI."""
        frame = render_population_heatmap(small_population_obs, dpi=150)
        assert frame.ndim == 3

    def test_different_states_produce_different_frames(self, small_population_obs: np.ndarray):
        """Test that different population states produce different frames."""
        # Create two different states
        rng = np.random.RandomState(123)
        state2 = rng.rand(100, 4) * 0.9  # Different fluorescence distribution

        frame1 = render_population_heatmap(small_population_obs)
        frame2 = render_population_heatmap(state2)

        # Frames should be different (not identical)
        assert not np.array_equal(frame1, frame2)
