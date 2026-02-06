"""Tests for CcaS-CcaR rendering utilities."""

import numpy as np
import pytest

from myriad.envs.bio.ccas_ccar.rendering import render_ccas_ccar_frame, render_population_heatmap


class TestRenderCcasCcarFrame:
    """Tests for render_ccas_ccar_frame function."""

    @pytest.fixture
    def basic_state(self) -> np.ndarray:
        """Fixture for basic CcaS-CcaR state."""
        # [F_normalized, U_obs, F_target[0], F_target[1]]
        return np.array([0.3, 0.0, 0.31, 0.32])

    @pytest.fixture
    def high_fluorescence_state(self) -> np.ndarray:
        """Fixture for high fluorescence state."""
        return np.array([0.9, 0.0, 0.5, 0.52])

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
        assert not np.array_equal(frame1, frame2), "Different states should produce different frames"

    def test_custom_normalizer(self, basic_state: np.ndarray):
        """Test rendering with custom normalization constant."""
        frame1 = render_ccas_ccar_frame(basic_state, F_obs_normalizer=80.0)
        frame2 = render_ccas_ccar_frame(basic_state, F_obs_normalizer=100.0)
        assert frame1.shape[-1] == 3
        assert frame2.shape[-1] == 3
        assert not np.array_equal(frame1, frame2)

    def test_custom_figsize_and_dpi(self, basic_state: np.ndarray):
        """Test rendering with custom figure size and DPI."""
        frame_small = render_ccas_ccar_frame(basic_state, figsize=(4, 3), dpi=50)
        frame_large = render_ccas_ccar_frame(basic_state, figsize=(10, 8), dpi=150)
        assert frame_small.shape[-1] == 3
        assert frame_large.shape[-1] == 3
        assert frame_large.size > frame_small.size

    def test_multiple_sequential_renders(self, basic_state: np.ndarray, high_fluorescence_state: np.ndarray):
        """Test that multiple sequential renders work correctly (no state leakage)."""
        frame1 = render_ccas_ccar_frame(basic_state)
        frame2 = render_ccas_ccar_frame(high_fluorescence_state)
        frame3 = render_ccas_ccar_frame(basic_state)
        assert np.array_equal(frame1, frame3), "Same state should produce identical frames"
        assert not np.array_equal(frame1, frame2)

    def test_frame_is_not_empty(self, basic_state: np.ndarray):
        """Test that rendered frame contains actual content."""
        frame = render_ccas_ccar_frame(basic_state)
        assert frame.std() > 0, "Frame should contain visual content with variation"


class TestRenderPopulationHeatmap:
    """Tests for render_population_heatmap function."""

    @pytest.fixture
    def small_population_obs(self) -> np.ndarray:
        """Fixture for small population observations (100 cells)."""
        rng = np.random.RandomState(42)
        return rng.rand(100, 4) * 0.5

    @pytest.fixture
    def large_population_obs(self) -> np.ndarray:
        """Fixture for large population observations (10K cells)."""
        rng = np.random.RandomState(42)
        return rng.rand(10_000, 4) * 0.8

    @pytest.fixture
    def population_actions(self) -> np.ndarray:
        """Fixture for population actions (100 cells)."""
        rng = np.random.RandomState(42)
        return rng.randint(0, 2, size=100)

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
        assert np.all(frame >= 0) and np.all(frame <= 255)

    def test_with_actions(self, small_population_obs: np.ndarray, population_actions: np.ndarray):
        """Test rendering with action overlay."""
        frame = render_population_heatmap(small_population_obs, actions=population_actions)
        assert frame.ndim == 3
        assert frame.dtype == np.uint8

    def test_large_population(self, large_population_obs: np.ndarray):
        """Test with large population (10K cells)."""
        frame = render_population_heatmap(large_population_obs)
        assert frame.ndim == 3
        assert frame.dtype == np.uint8

    def test_different_states_produce_different_frames(self, small_population_obs: np.ndarray):
        """Test that different population states produce different frames."""
        rng = np.random.RandomState(123)
        state2 = rng.rand(100, 4) * 0.9

        frame1 = render_population_heatmap(small_population_obs)
        frame2 = render_population_heatmap(state2)

        assert not np.array_equal(frame1, frame2)
