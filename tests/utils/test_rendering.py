"""Tests for video generation utilities."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from myriad.platform.logging.backends.disk import EPISODE_FILE_FORMAT, STEP_DIR_FORMAT
from myriad.utils.rendering import frames_to_video, render_episode_to_video, render_episodes


class TestFramesToVideo:
    """Tests for frames_to_video function."""

    @pytest.fixture
    def sample_frames(self) -> np.ndarray:
        """Fixture for sample RGB frames."""
        num_frames = 10
        height, width = 100, 150
        frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
        for i in range(num_frames):
            frames[i] = i * 20  # Varying brightness
        return frames

    def test_creates_video_file(self, sample_frames: np.ndarray, tmp_path: Path):
        """Test that video file is created successfully."""
        output_path = tmp_path / "test.mp4"
        result_path = frames_to_video(sample_frames, output_path)
        assert result_path.exists()
        assert result_path.suffix == ".mp4"
        assert result_path.stat().st_size > 0

    def test_accepts_list_or_array(self, sample_frames: np.ndarray, tmp_path: Path):
        """Test that function handles both list and array inputs."""
        # Test with array
        array_path = tmp_path / "array.mp4"
        frames_to_video(sample_frames, array_path)
        assert array_path.exists()

        # Test with list
        list_path = tmp_path / "list.mp4"
        frames_to_video(list(sample_frames), list_path)
        assert list_path.exists()

    def test_creates_parent_directories(self, sample_frames: np.ndarray, tmp_path: Path):
        """Test that parent directories are created if they don't exist."""
        output_path = tmp_path / "nested" / "dir" / "test.mp4"
        result_path = frames_to_video(sample_frames, output_path)
        assert result_path.exists()
        assert result_path.parent.exists()

    def test_custom_parameters(self, sample_frames: np.ndarray, tmp_path: Path):
        """Test that custom FPS and quality parameters work."""
        output_path = tmp_path / "custom.mp4"
        result_path = frames_to_video(sample_frames, output_path, fps=30, quality=5)
        assert result_path.exists()


class TestRenderEpisodeToVideo:
    """Tests for render_episode_to_video function."""

    @pytest.fixture
    def sample_episode(self) -> dict[str, np.ndarray]:
        """Fixture for sample episode data."""
        num_steps = 20
        return {
            "observations": np.random.randn(num_steps, 4).astype(np.float32),
            "actions": np.zeros(num_steps, dtype=np.int32),
            "rewards": np.ones(num_steps, dtype=np.float32),
            "dones": np.array([False] * (num_steps - 1) + [True]),
        }

    @pytest.fixture
    def simple_render_fn(self):
        """Fixture for a simple rendering function."""

        def render(obs: np.ndarray) -> np.ndarray:
            intensity = int((np.tanh(obs[0]) + 1) * 127)
            return np.full((100, 100, 3), intensity, dtype=np.uint8)

        return render

    def test_creates_video(self, sample_episode: dict, simple_render_fn, tmp_path: Path):
        """Test that episode video is created successfully."""
        output_path = tmp_path / "episode.mp4"
        result_path = render_episode_to_video(sample_episode, simple_render_fn, output_path)
        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_respects_done_flag(self, simple_render_fn, tmp_path: Path):
        """Test that rendering stops at first done=True."""
        episode = {
            "observations": np.random.randn(50, 4).astype(np.float32),
            "actions": np.zeros(50, dtype=np.int32),
            "rewards": np.ones(50, dtype=np.float32),
            "dones": np.array([False] * 15 + [True] + [False] * 34),  # Done at step 15
        }

        call_count = {"count": 0}

        def counting_render_fn(obs: np.ndarray) -> np.ndarray:
            call_count["count"] += 1
            return simple_render_fn(obs)

        render_episode_to_video(episode, counting_render_fn, tmp_path / "test.mp4")
        assert call_count["count"] == 16  # Renders steps 0-15 inclusive

    def test_max_frames_parameter(self, sample_episode: dict, simple_render_fn, tmp_path: Path):
        """Test that max_frames parameter limits rendering."""
        call_count = {"count": 0}

        def counting_render_fn(obs: np.ndarray) -> np.ndarray:
            call_count["count"] += 1
            return simple_render_fn(obs)

        render_episode_to_video(sample_episode, counting_render_fn, tmp_path / "test.mp4", max_frames=5)
        assert call_count["count"] == 5


class TestRenderEpisodes:
    """Tests for unified render_episodes function."""

    @pytest.fixture
    def mock_env_info(self):
        """Fixture for mocked environment info."""
        from myriad.envs.registration import EnvInfo

        def simple_render(obs: np.ndarray) -> np.ndarray:
            intensity = int((np.tanh(obs[0] if len(obs.shape) == 1 else obs[0]) + 1) * 127)
            return np.full((100, 100, 3), intensity, dtype=np.uint8)

        return EnvInfo(
            name="test-env",
            make_fn=lambda: None,
            config_cls=type(None),
            render_frame_fn=simple_render,
        )

    @pytest.fixture
    def mock_results(self):
        """Fixture for mocked EvaluationResults."""
        mock = MagicMock()
        num_episodes = 3
        max_steps = 20

        mock.episodes = {
            "observations": np.random.randn(num_episodes, max_steps, 4).astype(np.float32),
            "actions": np.zeros((num_episodes, max_steps), dtype=np.int32),
            "rewards": np.ones((num_episodes, max_steps), dtype=np.float32),
            "dones": np.zeros((num_episodes, max_steps), dtype=bool),
        }
        mock.episodes["dones"][:, -1] = True
        mock.seed = 42
        mock.config.env.name = "test-env"
        return mock

    @pytest.fixture
    def sample_episode_dict(self) -> dict[str, np.ndarray]:
        """Fixture for a single episode dict."""
        return {
            "observations": np.random.randn(20, 4).astype(np.float32),
            "actions": np.zeros(20, dtype=np.int32),
            "rewards": np.ones(20, dtype=np.float32) * 0.5,
            "dones": np.array([False] * 19 + [True]),
            "episode_return": 10.0,
            "episode_length": 20,
        }

    # === Mode 1: From EvaluationResults ===

    @patch("myriad.envs.get_env_info")
    def test_from_results_single(self, mock_get_env_info, mock_results, mock_env_info, tmp_path: Path):
        """Test rendering single episode from EvaluationResults."""
        mock_get_env_info.return_value = mock_env_info

        path, meta = render_episodes(results=mock_results, output_path=tmp_path / "test.mp4")

        assert path.exists()
        assert meta["env_name"] == "test-env"
        assert meta["seed"] == 42
        assert "episode_return" in meta

    @patch("myriad.envs.get_env_info")
    def test_from_results_batch(self, mock_get_env_info, mock_results, mock_env_info, tmp_path: Path):
        """Test batch rendering from EvaluationResults."""
        mock_get_env_info.return_value = mock_env_info

        paths, metas = render_episodes(
            results=mock_results,
            episode_index=[0, 1, 2],
            output_path=str(tmp_path / "videos/") + "/",
        )

        assert len(paths) == 3
        assert len(metas) == 3
        assert all(p.exists() for p in paths)
        assert paths[0].name == "test_env_episode_0000.mp4"

    # === Mode 2: From episode dict(s) ===

    @patch("myriad.envs.get_env_info")
    def test_from_episode_dict(self, mock_get_env_info, sample_episode_dict, mock_env_info, tmp_path: Path):
        """Test rendering from episode dict."""
        mock_get_env_info.return_value = mock_env_info

        path, meta = render_episodes(
            episode=sample_episode_dict,
            env_name="test-env",
            output_path=tmp_path / "test.mp4",
        )

        assert path.exists()
        assert meta["episode_return"] == 10.0
        assert meta["episode_length"] == 20

    # === Mode 3: From disk ===

    @patch("myriad.platform.artifact_loader.load_run_config")
    @patch("myriad.envs.get_env_info")
    def test_from_disk_single(
        self, mock_get_env_info, mock_load_config, mock_env_info, sample_episode_dict, tmp_path: Path
    ):
        """Test rendering single episode from disk."""
        mock_get_env_info.return_value = mock_env_info
        mock_config = MagicMock()
        mock_config.env.name = "test-env"
        mock_load_config.return_value = mock_config

        # Create run directory structure
        run_dir = tmp_path / "run"
        episodes_dir = run_dir / "episodes" / "step_00005000"
        episodes_dir.mkdir(parents=True)
        episode_file = episodes_dir / EPISODE_FILE_FORMAT.format(0)
        np.savez(episode_file, **{**sample_episode_dict, "global_step": 5000})

        path, meta = render_episodes(run_dir=run_dir, step=5000, output_path=tmp_path / "test.mp4")

        assert path.exists()
        assert meta["global_step"] == 5000

    @patch("myriad.platform.artifact_loader.load_run_config")
    @patch("myriad.envs.get_env_info")
    def test_from_disk_batch(
        self, mock_get_env_info, mock_load_config, mock_env_info, sample_episode_dict, tmp_path: Path
    ):
        """Test batch rendering from disk."""
        mock_get_env_info.return_value = mock_env_info
        mock_config = MagicMock()
        mock_config.env.name = "test-env"
        mock_load_config.return_value = mock_config

        # Create multiple checkpoints
        run_dir = tmp_path / "run"
        steps = [500, 1000, 1500]
        for s in steps:
            episodes_dir = run_dir / "episodes" / STEP_DIR_FORMAT.format(s)
            episodes_dir.mkdir(parents=True, exist_ok=True)
            np.savez(episodes_dir / EPISODE_FILE_FORMAT.format(0), **{**sample_episode_dict, "global_step": s})

        paths, metas = render_episodes(run_dir=run_dir, step=steps, output_path=str(tmp_path / "videos/") + "/")

        assert len(paths) == 3
        assert [m["global_step"] for m in metas] == steps
        assert paths[0].name == "test_env_step_00000500.mp4"

    # === Batch rendering output path variations ===

    @patch("myriad.envs.get_env_info")
    def test_batch_with_explicit_paths(self, mock_get_env_info, mock_results, mock_env_info, tmp_path: Path):
        """Test batch rendering with explicit list of output paths."""
        mock_get_env_info.return_value = mock_env_info

        output_paths = [tmp_path / f"vid{i}.mp4" for i in range(3)]
        paths, _ = render_episodes(results=mock_results, episode_index=[0, 1, 2], output_path=output_paths)

        assert paths == output_paths
        assert all(p.exists() for p in paths)

    # === Error handling ===

    def test_no_mode_provided(self, tmp_path: Path):
        """Test error when no input mode is provided."""
        with pytest.raises(ValueError, match="Must provide one of"):
            render_episodes(output_path=tmp_path / "test.mp4")

    def test_multiple_modes(self, mock_results, tmp_path: Path):
        """Test error when multiple modes are provided."""
        with pytest.raises(ValueError, match="Cannot mix input modes"):
            render_episodes(results=mock_results, run_dir=tmp_path, step=5000, output_path=tmp_path / "test.mp4")

    def test_missing_env_name_in_episode_mode(self, sample_episode_dict, tmp_path: Path):
        """Test error when using episode mode without env_name."""
        with pytest.raises(ValueError, match="must provide 'env_name'"):
            render_episodes(episode=sample_episode_dict, output_path=tmp_path / "test.mp4")

    def test_missing_step_in_disk_mode(self, tmp_path: Path):
        """Test error when using run_dir mode without step."""
        with pytest.raises(ValueError, match="must provide 'step'"):
            render_episodes(run_dir=tmp_path, output_path=tmp_path / "test.mp4")

    def test_results_without_episodes(self, tmp_path: Path):
        """Test error when EvaluationResults.episodes is None."""
        mock = MagicMock()
        mock.episodes = None
        with pytest.raises(ValueError, match="episodes is None"):
            render_episodes(results=mock, output_path=tmp_path / "test.mp4")

    @patch("myriad.envs.get_env_info")
    def test_batch_mismatched_output_paths(self, mock_get_env_info, mock_results, mock_env_info, tmp_path: Path):
        """Test error when output_path list length doesn't match episodes."""
        mock_get_env_info.return_value = mock_env_info
        with pytest.raises(ValueError, match="output_path list length"):
            render_episodes(results=mock_results, episode_index=[0, 1, 2], output_path=[tmp_path / "v1.mp4"])

    @patch("myriad.envs.get_env_info")
    def test_batch_with_single_file_path(self, mock_get_env_info, mock_results, mock_env_info, tmp_path: Path):
        """Test error when using single file path for batch rendering."""
        mock_get_env_info.return_value = mock_env_info
        with pytest.raises(ValueError, match="must be a directory"):
            render_episodes(results=mock_results, episode_index=[0, 1, 2], output_path=tmp_path / "single.mp4")

    @patch("myriad.envs.get_env_info")
    def test_unknown_environment(self, mock_get_env_info, sample_episode_dict, tmp_path: Path):
        """Test error when environment is not registered."""
        mock_get_env_info.return_value = None
        with pytest.raises(ValueError, match="Unknown environment"):
            render_episodes(episode=sample_episode_dict, env_name="unknown", output_path=tmp_path / "test.mp4")

    @patch("myriad.envs.get_env_info")
    def test_environment_without_render_function(self, mock_get_env_info, sample_episode_dict, tmp_path: Path):
        """Test error when environment doesn't support rendering."""
        from myriad.envs.registration import EnvInfo

        mock_env_info = EnvInfo(name="test-env", make_fn=lambda: None, config_cls=type(None), render_frame_fn=None)
        mock_get_env_info.return_value = mock_env_info

        with pytest.raises(ValueError, match="does not support rendering"):
            render_episodes(episode=sample_episode_dict, env_name="test-env", output_path=tmp_path / "test.mp4")
