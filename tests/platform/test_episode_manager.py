"""Tests for episode persistence and rendering management."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from myriad.configs.default import AgentConfig, Config, EnvConfig, RunConfig, WandbConfig
from myriad.platform.episode_manager import render_episodes_to_videos, save_episodes_to_disk


@pytest.fixture
def test_config():
    """Create a minimal test configuration."""
    return Config(
        run=RunConfig(
            seed=42,
            steps_per_env=100,
            num_envs=4,
            batch_size=32,
            buffer_size=1000,
            scan_chunk_size=10,
            eval_frequency=50,
            eval_rollouts=5,
            eval_max_steps=100,
            log_frequency=10,
        ),
        agent=AgentConfig(name="test_agent"),
        env=EnvConfig(name="test_env"),
        wandb=WandbConfig(enabled=False),
    )


@pytest.fixture
def sample_episode_data():
    """Create sample episode data for testing."""
    num_episodes = 3
    max_steps = 10

    # Create episode trajectories with padding
    observations = np.random.randn(num_episodes, max_steps, 4).astype(np.float32)
    actions = np.random.randn(num_episodes, max_steps, 1).astype(np.float32)
    rewards = np.random.randn(num_episodes, max_steps).astype(np.float32)
    dones = np.zeros((num_episodes, max_steps), dtype=bool)

    # Set episode lengths (different for each episode to test trimming)
    episode_lengths = np.array([5, 8, 10], dtype=np.int32)
    episode_returns = np.array([10.5, 15.2, 20.0], dtype=np.float32)

    return {
        "episodes": {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        },
        "episode_length": episode_lengths,
        "episode_return": episode_returns,
    }


class TestSaveEpisodesToDisk:
    """Tests for save_episodes_to_disk function."""

    def test_saves_episodes_with_correct_structure(self, tmp_path, test_config, sample_episode_data, monkeypatch):
        """Episodes should be saved with correct directory structure and metadata."""
        monkeypatch.chdir(tmp_path)
        global_step = 1000
        save_count = 2

        result_dir = save_episodes_to_disk(sample_episode_data, global_step, save_count, test_config)

        # Should return the directory path
        assert result_dir is not None
        episodes_dir = Path(result_dir)
        assert episodes_dir.exists()
        assert episodes_dir.name == "step_00001000"

        # Should save only the requested number of episodes
        episode_files = list(episodes_dir.glob("*.npz"))
        assert len(episode_files) == save_count

        # Check first episode
        ep0_data = np.load(episodes_dir / "episode_0.npz")
        expected_len = sample_episode_data["episode_length"][0]

        # Observations should be trimmed to actual episode length
        assert ep0_data["observations"].shape[0] == expected_len
        assert ep0_data["actions"].shape[0] == expected_len
        assert ep0_data["rewards"].shape[0] == expected_len
        assert ep0_data["dones"].shape[0] == expected_len

        # Metadata should be present
        assert ep0_data["episode_length"] == expected_len
        assert ep0_data["episode_return"] == pytest.approx(sample_episode_data["episode_return"][0])
        assert ep0_data["global_step"] == global_step
        assert ep0_data["seed"] == test_config.run.seed

    def test_saves_all_episodes_when_save_count_exceeds_available(
        self, tmp_path, test_config, sample_episode_data, monkeypatch
    ):
        """Should save all available episodes when save_count is larger."""
        monkeypatch.chdir(tmp_path)
        save_count = 100  # Much larger than available episodes

        result_dir = save_episodes_to_disk(sample_episode_data, 0, save_count, test_config)

        assert result_dir is not None
        episodes_dir = Path(result_dir)
        episode_files = list(episodes_dir.glob("*.npz"))

        # Should save all 3 available episodes
        assert len(episode_files) == 3

    def test_returns_none_when_no_episode_data(self, tmp_path, test_config, monkeypatch):
        """Should return None when episode data is missing."""
        monkeypatch.chdir(tmp_path)
        incomplete_data = {
            "episode_length": np.array([5]),
            "episode_return": np.array([10.0]),
            # Missing "episodes" key
        }

        result = save_episodes_to_disk(incomplete_data, 0, 1, test_config)
        assert result is None

    def test_creates_nested_directory_structure(self, tmp_path, test_config, sample_episode_data, monkeypatch):
        """Should create parent directories if they don't exist."""
        monkeypatch.chdir(tmp_path)

        result_dir = save_episodes_to_disk(sample_episode_data, 5000, 1, test_config)

        assert result_dir is not None
        # Check that base "episodes" dir was created
        assert (tmp_path / "episodes").exists()
        # Check that step directory was created
        assert (tmp_path / "episodes" / "step_00005000").exists()

    def test_handles_zero_save_count(self, tmp_path, test_config, sample_episode_data, monkeypatch):
        """Should handle save_count=0 gracefully."""
        monkeypatch.chdir(tmp_path)

        result = save_episodes_to_disk(sample_episode_data, 0, 0, test_config)

        # Should return None since no episodes were saved
        assert result is None


class TestRenderEpisodesToVideos:
    """Tests for render_episodes_to_videos function."""

    @pytest.fixture
    def mock_render_frame_fn(self):
        """Create a mock rendering function."""

        def render_frame(obs: np.ndarray) -> np.ndarray:
            # Return a dummy RGB frame (64x64x3)
            return np.zeros((64, 64, 3), dtype=np.uint8)

        return render_frame

    def test_renders_saved_episodes_to_videos(
        self, tmp_path, test_config, sample_episode_data, mock_render_frame_fn, monkeypatch
    ):
        """Should render all saved episodes to video files."""
        monkeypatch.chdir(tmp_path)

        # First, save some episodes
        episodes_dir = save_episodes_to_disk(sample_episode_data, 0, 2, test_config)
        assert episodes_dir is not None

        # Mock the actual video rendering to avoid opencv dependency in tests
        with patch("myriad.utils.rendering.render_episode_to_video") as mock_render:
            output_dir = tmp_path / "videos"
            rendered_count = render_episodes_to_videos(
                episodes_dir, mock_render_frame_fn, output_dir=output_dir, fps=30
            )

            # Should have attempted to render 2 episodes
            assert rendered_count == 2
            assert mock_render.call_count == 2

            # Check that render was called with correct arguments
            first_call = mock_render.call_args_list[0]
            _episode_data, render_fn, video_path, fps = (
                first_call[0][0],
                first_call[0][1],
                first_call[0][2],
                first_call[1]["fps"],
            )

            assert render_fn == mock_render_frame_fn
            assert fps == 30
            assert "episode_0.mp4" in str(video_path)

    def test_returns_zero_when_directory_not_found(self, tmp_path, mock_render_frame_fn):
        """Should return 0 when episodes directory doesn't exist."""
        nonexistent_dir = tmp_path / "nonexistent"

        rendered_count = render_episodes_to_videos(nonexistent_dir, mock_render_frame_fn)

        assert rendered_count == 0

    def test_returns_zero_when_no_episode_files(self, tmp_path, mock_render_frame_fn):
        """Should return 0 when directory exists but has no .npz files."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        rendered_count = render_episodes_to_videos(empty_dir, mock_render_frame_fn)

        assert rendered_count == 0

    def test_creates_output_directory_if_missing(
        self, tmp_path, test_config, sample_episode_data, mock_render_frame_fn, monkeypatch
    ):
        """Should create output directory if it doesn't exist."""
        monkeypatch.chdir(tmp_path)

        # Save episodes
        episodes_dir = save_episodes_to_disk(sample_episode_data, 0, 1, test_config)

        # Render to a non-existent output directory
        with patch("myriad.utils.rendering.render_episode_to_video"):
            output_dir = tmp_path / "new_videos_dir"
            assert not output_dir.exists()

            render_episodes_to_videos(episodes_dir, mock_render_frame_fn, output_dir=output_dir)

            # Directory should now exist
            assert output_dir.exists()

    def test_handles_rendering_errors_gracefully(
        self, tmp_path, test_config, sample_episode_data, mock_render_frame_fn, monkeypatch
    ):
        """Should continue rendering other episodes if one fails."""
        monkeypatch.chdir(tmp_path)

        # Save multiple episodes
        episodes_dir = save_episodes_to_disk(sample_episode_data, 0, 3, test_config)

        # Mock render to fail on the second episode
        with patch("myriad.utils.rendering.render_episode_to_video") as mock_render:
            call_count = 0

            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise RuntimeError("Rendering failed!")

            mock_render.side_effect = side_effect

            rendered_count = render_episodes_to_videos(episodes_dir, mock_render_frame_fn)

            # Should have attempted all 3, but only 2 succeeded
            assert rendered_count == 2
            assert mock_render.call_count == 3

    def test_preserves_directory_structure_in_output(
        self, tmp_path, test_config, sample_episode_data, mock_render_frame_fn, monkeypatch
    ):
        """Should preserve relative directory structure when rendering."""
        monkeypatch.chdir(tmp_path)

        # Save episodes
        episodes_dir = save_episodes_to_disk(sample_episode_data, 1000, 1, test_config)

        with patch("myriad.utils.rendering.render_episode_to_video") as mock_render:
            output_dir = tmp_path / "videos"
            render_episodes_to_videos(episodes_dir, mock_render_frame_fn, output_dir=output_dir)

            # Check that video path preserves structure
            video_path = mock_render.call_args[0][2]
            assert "episode_0.mp4" in str(video_path)
            assert video_path.suffix == ".mp4"


class TestIntegration:
    """Integration tests for episode save and render workflow."""

    def test_full_save_and_render_workflow(self, tmp_path, test_config, sample_episode_data, monkeypatch):
        """Test complete workflow: save episodes, then render them."""
        monkeypatch.chdir(tmp_path)

        # Step 1: Save episodes
        episodes_dir = save_episodes_to_disk(sample_episode_data, 500, 2, test_config)
        assert episodes_dir is not None

        # Step 2: Verify saved files can be loaded
        ep_path = Path(episodes_dir) / "episode_0.npz"
        assert ep_path.exists()

        loaded_data = np.load(ep_path)
        assert "observations" in loaded_data
        assert "actions" in loaded_data
        assert "rewards" in loaded_data
        assert "dones" in loaded_data
        assert loaded_data["global_step"] == 500

        # Step 3: Mock render and verify it can process saved data
        def mock_render_frame(obs: np.ndarray) -> np.ndarray:
            # Verify observation shape matches what we saved
            assert obs.shape == (4,)  # Sample data has 4D observations
            return np.zeros((32, 32, 3), dtype=np.uint8)

        with patch("myriad.utils.rendering.render_episode_to_video") as mock_render:
            rendered_count = render_episodes_to_videos(episodes_dir, mock_render_frame, fps=60)

            assert rendered_count == 2
            assert mock_render.call_count == 2

            # Verify render was called with the loaded episode data
            call_args = mock_render.call_args_list[0]
            episode_data_arg = call_args[0][0]
            assert "observations" in episode_data_arg
            assert episode_data_arg["observations"].shape[0] == sample_episode_data["episode_length"][0]
