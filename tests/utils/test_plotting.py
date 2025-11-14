"""Tests for the plotting utilities."""

from pathlib import Path

import jax.numpy as jnp
import pytest

from aion.utils.plotting.episodes import plot_episodes


class TestPlotting:
    """Tests for the plotting utilities."""

    @pytest.fixture
    def sample_data(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Fixture for sample observation and reward data."""
        num_episodes = 5
        num_steps = 10
        obs_dim = 2
        observations = jnp.zeros((num_episodes, num_steps, obs_dim))
        rewards = jnp.zeros((num_episodes, num_steps))
        return observations, rewards

    def test_plot_episodes(self, sample_data: tuple[jnp.ndarray, jnp.ndarray], tmp_path: Path):
        """Tests the plot_episodes function."""
        observations, rewards = sample_data
        filename = tmp_path / "test_plot.html"
        plot_episodes(observations, rewards, filename=filename)
        assert filename.exists()
