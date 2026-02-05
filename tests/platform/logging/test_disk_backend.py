"""Tests for disk backend (episode persistence and rendering)."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from myriad.platform.logging.backends.disk import DiskBackend, render_episodes_to_videos


@pytest.fixture
def sample_episode_data():
    """Create sample episode data for testing."""
    num_episodes = 3
    max_steps = 10

    observations = np.random.randn(num_episodes, max_steps, 4).astype(np.float32)
    actions = np.random.randn(num_episodes, max_steps, 1).astype(np.float32)
    rewards = np.random.randn(num_episodes, max_steps).astype(np.float32)
    dones = np.zeros((num_episodes, max_steps), dtype=bool)

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


class TestDiskBackend:
    """Tests for DiskBackend class."""

    def test_saves_episodes_with_correct_structure(self, tmp_path, sample_episode_data):
        """Episodes should be saved with correct directory structure and metadata."""
        backend = DiskBackend(base_dir=tmp_path / "episodes", seed=42)
        global_step = 1000
        save_count = 2

        result_dir = backend.save_episodes(sample_episode_data, global_step, save_count)

        assert result_dir is not None
        episodes_dir = Path(result_dir)
        assert episodes_dir.exists()
        assert episodes_dir.name == "step_00001000"

        episode_files = list(episodes_dir.glob("*.npz"))
        assert len(episode_files) == save_count

        ep0_data = np.load(episodes_dir / "episode_0.npz")
        expected_len = sample_episode_data["episode_length"][0]

        assert ep0_data["observations"].shape[0] == expected_len
        assert ep0_data["actions"].shape[0] == expected_len
        assert ep0_data["rewards"].shape[0] == expected_len
        assert ep0_data["dones"].shape[0] == expected_len

        assert ep0_data["episode_length"] == expected_len
        assert ep0_data["episode_return"] == pytest.approx(sample_episode_data["episode_return"][0])
        assert ep0_data["global_step"] == global_step
        assert ep0_data["seed"] == 42

    def test_saves_all_episodes_when_save_count_exceeds_available(self, tmp_path, sample_episode_data):
        """Should save all available episodes when save_count is larger."""
        backend = DiskBackend(base_dir=tmp_path / "episodes", seed=0)
        save_count = 100

        result_dir = backend.save_episodes(sample_episode_data, 0, save_count)

        assert result_dir is not None
        episodes_dir = Path(result_dir)
        episode_files = list(episodes_dir.glob("*.npz"))
        assert len(episode_files) == 3

    def test_returns_none_when_no_episode_data(self, tmp_path):
        """Should return None when episode data is missing."""
        backend = DiskBackend(base_dir=tmp_path / "episodes", seed=0)
        incomplete_data = {
            "episode_length": np.array([5]),
            "episode_return": np.array([10.0]),
        }

        result = backend.save_episodes(incomplete_data, 0, 1)
        assert result is None

    def test_creates_nested_directory_structure(self, tmp_path, sample_episode_data):
        """Should create parent directories if they don't exist."""
        backend = DiskBackend(base_dir=tmp_path / "deep" / "nested" / "episodes", seed=0)

        result_dir = backend.save_episodes(sample_episode_data, 5000, 1)

        assert result_dir is not None
        assert (tmp_path / "deep" / "nested" / "episodes").exists()
        assert (tmp_path / "deep" / "nested" / "episodes" / "step_00005000").exists()

    def test_handles_zero_save_count(self, tmp_path, sample_episode_data):
        """Should handle save_count=0 gracefully."""
        backend = DiskBackend(base_dir=tmp_path / "episodes", seed=0)

        result = backend.save_episodes(sample_episode_data, 0, 0)
        assert result is None


class TestRenderEpisodesToVideos:
    """Tests for render_episodes_to_videos function."""

    @pytest.fixture
    def mock_render_frame_fn(self):
        """Create a mock rendering function."""

        def render_frame(obs: np.ndarray) -> np.ndarray:
            return np.zeros((64, 64, 3), dtype=np.uint8)

        return render_frame

    def test_renders_saved_episodes_to_videos(self, tmp_path, sample_episode_data, mock_render_frame_fn):
        """Should render all saved episodes to video files."""
        backend = DiskBackend(base_dir=tmp_path / "episodes", seed=0)
        episodes_dir = backend.save_episodes(sample_episode_data, 0, 2)
        assert episodes_dir is not None

        with patch("myriad.utils.rendering.render_episode_to_video") as mock_render:
            output_dir = tmp_path / "videos"
            rendered_count = render_episodes_to_videos(
                episodes_dir, mock_render_frame_fn, output_dir=output_dir, fps=30
            )

            assert rendered_count == 2
            assert mock_render.call_count == 2

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

    def test_creates_output_directory_if_missing(self, tmp_path, sample_episode_data, mock_render_frame_fn):
        """Should create output directory if it doesn't exist."""
        backend = DiskBackend(base_dir=tmp_path / "episodes", seed=0)
        episodes_dir = backend.save_episodes(sample_episode_data, 0, 1)

        with patch("myriad.utils.rendering.render_episode_to_video"):
            output_dir = tmp_path / "new_videos_dir"
            assert not output_dir.exists()

            render_episodes_to_videos(episodes_dir, mock_render_frame_fn, output_dir=output_dir)

            assert output_dir.exists()

    def test_handles_rendering_errors_gracefully(self, tmp_path, sample_episode_data, mock_render_frame_fn):
        """Should continue rendering other episodes if one fails."""
        backend = DiskBackend(base_dir=tmp_path / "episodes", seed=0)
        episodes_dir = backend.save_episodes(sample_episode_data, 0, 3)

        with patch("myriad.utils.rendering.render_episode_to_video") as mock_render:
            call_count = 0

            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise RuntimeError("Rendering failed!")

            mock_render.side_effect = side_effect

            rendered_count = render_episodes_to_videos(episodes_dir, mock_render_frame_fn)

            assert rendered_count == 2
            assert mock_render.call_count == 3


class TestIntegration:
    """Integration tests for episode save and render workflow."""

    def test_full_save_and_render_workflow(self, tmp_path, sample_episode_data):
        """Test complete workflow: save episodes, then render them."""
        backend = DiskBackend(base_dir=tmp_path / "episodes", seed=0)

        # Step 1: Save episodes
        episodes_dir = backend.save_episodes(sample_episode_data, 500, 2)
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
            assert obs.shape == (4,)
            return np.zeros((32, 32, 3), dtype=np.uint8)

        with patch("myriad.utils.rendering.render_episode_to_video") as mock_render:
            rendered_count = render_episodes_to_videos(episodes_dir, mock_render_frame, fps=60)

            assert rendered_count == 2
            assert mock_render.call_count == 2

            call_args = mock_render.call_args_list[0]
            episode_data_arg = call_args[0][0]
            assert "observations" in episode_data_arg
            assert episode_data_arg["observations"].shape[0] == sample_episode_data["episode_length"][0]
