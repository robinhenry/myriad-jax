"""Tests for CcaS-CcaR rendering utilities."""

import jax.numpy as jnp
import numpy as np
import pytest

from myriad.envs.bio.ccas_ccar.physics import PhysicsState
from myriad.envs.bio.ccas_ccar.rendering import render_ccas_ccar_frame, render_population_heatmap
from myriad.envs.bio.ccas_ccar.tasks.control import ControlTaskConfig, ControlTaskState


class TestRenderCcasCcarFrame:
    """Tests for render_ccas_ccar_frame function."""

    @pytest.fixture
    def config(self) -> ControlTaskConfig:
        """Fixture for task configuration."""
        return ControlTaskConfig()

    @pytest.fixture
    def basic_state(self) -> ControlTaskState:
        """Fixture for basic CcaS-CcaR state.

        Creates a ControlTaskState with moderate fluorescence.
        """
        physics = PhysicsState(
            time=jnp.array(0.0),
            H=jnp.array(10.0),
            F=jnp.array(24.0),  # F = 24 molecules
            next_reaction_time=jnp.array(jnp.inf),
        )
        return ControlTaskState(
            physics=physics,
            t=jnp.array(0),
            U=jnp.array(0),
            F_target=jnp.array([25.0, 25.6]),  # Target trajectory in molecules
        )

    @pytest.fixture
    def high_fluorescence_state(self) -> ControlTaskState:
        """Fixture for high fluorescence state."""
        physics = PhysicsState(
            time=jnp.array(50.0),
            H=jnp.array(30.0),
            F=jnp.array(72.0),  # F = 72 molecules (high)
            next_reaction_time=jnp.array(jnp.inf),
        )
        return ControlTaskState(
            physics=physics,
            t=jnp.array(10),
            U=jnp.array(1),
            F_target=jnp.array([40.0, 41.6]),  # Target trajectory in molecules
        )

    def test_frame_shape_default(self, basic_state: ControlTaskState, config: ControlTaskConfig):
        """Test that rendered frame has correct shape with default parameters."""
        frame = render_ccas_ccar_frame(basic_state, config)
        assert frame.ndim == 3, "Frame should be 3D array (height, width, channels)"
        assert frame.shape[-1] == 3, "Frame should have 3 color channels (RGB)"

    def test_frame_dtype(self, basic_state: ControlTaskState, config: ControlTaskConfig):
        """Test that rendered frame has correct dtype."""
        frame = render_ccas_ccar_frame(basic_state, config)
        assert frame.dtype == np.uint8, "Frame should be uint8 for image data"

    def test_frame_values_in_range(self, basic_state: ControlTaskState, config: ControlTaskConfig):
        """Test that pixel values are in valid range [0, 255]."""
        frame = render_ccas_ccar_frame(basic_state, config)
        assert np.all(frame >= 0), "Pixel values should be >= 0"
        assert np.all(frame <= 255), "Pixel values should be <= 255"

    def test_different_states_produce_different_frames(
        self,
        basic_state: ControlTaskState,
        high_fluorescence_state: ControlTaskState,
        config: ControlTaskConfig,
    ):
        """Test that different states produce different visual frames."""
        frame1 = render_ccas_ccar_frame(basic_state, config)
        frame2 = render_ccas_ccar_frame(high_fluorescence_state, config)
        assert not np.array_equal(frame1, frame2), "Different states should produce different frames"

    def test_custom_normalizer(self, basic_state: ControlTaskState):
        """Test rendering with custom normalization constant."""
        config1 = ControlTaskConfig()
        config1 = config1.replace(task=config1.task.replace(F_obs_normalizer=80.0))
        config2 = ControlTaskConfig()
        config2 = config2.replace(task=config2.task.replace(F_obs_normalizer=100.0))
        frame1 = render_ccas_ccar_frame(basic_state, config1)
        frame2 = render_ccas_ccar_frame(basic_state, config2)
        assert frame1.shape[-1] == 3
        assert frame2.shape[-1] == 3
        assert not np.array_equal(frame1, frame2)

    def test_custom_figsize_and_dpi(self, basic_state: ControlTaskState, config: ControlTaskConfig):
        """Test rendering with custom figure size and DPI."""
        frame_small = render_ccas_ccar_frame(basic_state, config, figsize=(4, 3), dpi=50)
        frame_large = render_ccas_ccar_frame(basic_state, config, figsize=(10, 8), dpi=150)
        assert frame_small.shape[-1] == 3
        assert frame_large.shape[-1] == 3
        assert frame_large.size > frame_small.size

    def test_multiple_sequential_renders(
        self,
        basic_state: ControlTaskState,
        high_fluorescence_state: ControlTaskState,
        config: ControlTaskConfig,
    ):
        """Test that multiple sequential renders work correctly (no state leakage)."""
        frame1 = render_ccas_ccar_frame(basic_state, config)
        frame2 = render_ccas_ccar_frame(high_fluorescence_state, config)
        frame3 = render_ccas_ccar_frame(basic_state, config)
        assert np.array_equal(frame1, frame3), "Same state should produce identical frames"
        assert not np.array_equal(frame1, frame2)

    def test_frame_is_not_empty(self, basic_state: ControlTaskState, config: ControlTaskConfig):
        """Test that rendered frame contains actual content."""
        frame = render_ccas_ccar_frame(basic_state, config)
        assert frame.std() > 0, "Frame should contain visual content with variation"


class TestRenderPopulationHeatmap:
    """Tests for render_population_heatmap function."""

    @pytest.fixture
    def small_population_obs(self) -> np.ndarray:
        """Fixture for small population observations (100 cells).

        Format matches CcasCcarControlObs.to_array(): [F_normalized, F_target...]
        Using 3 columns for n_horizon=1: [F, F_target[0], F_target[1]]
        """
        rng = np.random.RandomState(42)
        return rng.rand(100, 3) * 0.5

    @pytest.fixture
    def large_population_obs(self) -> np.ndarray:
        """Fixture for large population observations (10K cells).

        Format matches CcasCcarControlObs.to_array(): [F_normalized, F_target...]
        """
        rng = np.random.RandomState(42)
        return rng.rand(10_000, 3) * 0.8

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
        state2 = rng.rand(100, 3) * 0.9

        frame1 = render_population_heatmap(small_population_obs)
        frame2 = render_population_heatmap(state2)

        assert not np.array_equal(frame1, frame2)
