"""Tests for video generation utilities."""

from pathlib import Path

import numpy as np
import pytest

from myriad.utils.rendering.video import frames_to_video, render_episode_to_video


class TestFramesToVideo:
    """Tests for frames_to_video function."""

    @pytest.fixture
    def sample_frames_list(self) -> list[np.ndarray]:
        """Fixture for sample RGB frames as list."""
        num_frames = 10
        height, width = 100, 150
        frames = []
        for i in range(num_frames):
            # Create frames with varying colors to ensure they're distinct
            frame = np.full((height, width, 3), fill_value=i * 20, dtype=np.uint8)
            frames.append(frame)
        return frames

    @pytest.fixture
    def sample_frames_array(self) -> np.ndarray:
        """Fixture for sample RGB frames as numpy array."""
        num_frames = 10
        height, width = 100, 150
        frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
        for i in range(num_frames):
            frames[i] = i * 20  # Varying brightness
        return frames

    def test_frames_to_video_creates_file(self, sample_frames_list: list[np.ndarray], tmp_path: Path):
        """Test that video file is created successfully."""
        output_path = tmp_path / "test.mp4"
        result_path = frames_to_video(sample_frames_list, output_path)
        assert result_path.exists(), "Video file should exist"
        assert result_path.suffix == ".mp4", "Should create .mp4 file"

    def test_frames_to_video_with_list_input(self, sample_frames_list: list[np.ndarray], tmp_path: Path):
        """Test that function handles list of frames correctly."""
        output_path = tmp_path / "test_list.mp4"
        result_path = frames_to_video(sample_frames_list, output_path)
        assert result_path.exists()
        # File should have non-zero size
        assert result_path.stat().st_size > 0, "Video file should not be empty"

    def test_frames_to_video_with_array_input(self, sample_frames_array: np.ndarray, tmp_path: Path):
        """Test that function handles numpy array of frames correctly."""
        output_path = tmp_path / "test_array.mp4"
        result_path = frames_to_video(sample_frames_array, output_path)
        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_frames_to_video_creates_parent_directory(self, sample_frames_list: list[np.ndarray], tmp_path: Path):
        """Test that parent directories are created if they don't exist."""
        output_path = tmp_path / "nested" / "dir" / "test.mp4"
        result_path = frames_to_video(sample_frames_list, output_path)
        assert result_path.exists()
        assert result_path.parent.exists()

    def test_frames_to_video_custom_fps(self, sample_frames_list: list[np.ndarray], tmp_path: Path):
        """Test that custom FPS parameter is accepted."""
        output_path = tmp_path / "test_fps.mp4"
        result_path = frames_to_video(sample_frames_list, output_path, fps=30)
        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_frames_to_video_custom_quality(self, sample_frames_list: list[np.ndarray], tmp_path: Path):
        """Test that custom quality parameter is accepted."""
        output_path = tmp_path / "test_quality.mp4"
        result_path = frames_to_video(sample_frames_list, output_path, quality=5)
        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_frames_to_video_returns_path_object(self, sample_frames_list: list[np.ndarray], tmp_path: Path):
        """Test that function returns a Path object."""
        output_path = tmp_path / "test.mp4"
        result_path = frames_to_video(sample_frames_list, output_path)
        assert isinstance(result_path, Path), "Should return Path object"

    def test_frames_to_video_with_string_path(self, sample_frames_list: list[np.ndarray], tmp_path: Path):
        """Test that function accepts string path as well as Path object."""
        output_path = str(tmp_path / "test_string.mp4")
        result_path = frames_to_video(sample_frames_list, output_path)
        assert result_path.exists()
        assert isinstance(result_path, Path)

    def test_frames_to_video_different_sizes(self, tmp_path: Path):
        """Test that videos with different frame counts can be created."""
        # Short video
        short_frames = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(5)]
        short_path = tmp_path / "short.mp4"
        result_short = frames_to_video(short_frames, short_path)
        assert result_short.exists()

        # Longer video
        long_frames = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(50)]
        long_path = tmp_path / "long.mp4"
        result_long = frames_to_video(long_frames, long_path)
        assert result_long.exists()

        # Longer video should be larger
        assert result_long.stat().st_size > result_short.stat().st_size


class TestRenderEpisodeToVideo:
    """Tests for render_episode_to_video function."""

    @pytest.fixture
    def sample_episode_data(self) -> dict[str, np.ndarray]:
        """Fixture for sample episode data."""
        num_steps = 20
        obs_dim = 4
        observations = np.random.randn(num_steps, obs_dim).astype(np.float32)
        actions = np.zeros(num_steps, dtype=np.int32)
        rewards = np.ones(num_steps, dtype=np.float32)
        dones = np.zeros(num_steps, dtype=bool)
        dones[-1] = True  # Mark last step as done
        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        }

    @pytest.fixture
    def early_termination_episode_data(self) -> dict[str, np.ndarray]:
        """Fixture for episode that terminates early."""
        num_steps = 50
        obs_dim = 4
        observations = np.random.randn(num_steps, obs_dim).astype(np.float32)
        actions = np.zeros(num_steps, dtype=np.int32)
        rewards = np.ones(num_steps, dtype=np.float32)
        dones = np.zeros(num_steps, dtype=bool)
        dones[15] = True  # Terminates at step 15
        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        }

    @pytest.fixture
    def simple_render_fn(self):
        """Fixture for a simple rendering function."""

        def render(obs: np.ndarray) -> np.ndarray:
            """Simple render function that creates a solid color frame based on observation."""
            # Create a 100x100 RGB frame
            # Use first observation value to determine color intensity
            intensity = int((np.tanh(obs[0]) + 1) * 127)  # Map to [0, 254]
            frame = np.full((100, 100, 3), intensity, dtype=np.uint8)
            return frame

        return render

    def test_render_episode_to_video_creates_file(
        self,
        sample_episode_data: dict[str, np.ndarray],
        simple_render_fn,
        tmp_path: Path,
    ):
        """Test that episode video is created successfully."""
        output_path = tmp_path / "episode.mp4"
        result_path = render_episode_to_video(
            sample_episode_data,
            simple_render_fn,
            output_path,
        )
        assert result_path.exists(), "Video file should exist"
        assert result_path.stat().st_size > 0, "Video file should not be empty"

    def test_render_episode_respects_done_flag(
        self,
        early_termination_episode_data: dict[str, np.ndarray],
        simple_render_fn,
        tmp_path: Path,
    ):
        """Test that rendering stops at first done=True."""
        output_path = tmp_path / "episode_early.mp4"
        # Track how many times render function is called
        call_count = {"count": 0}
        original_fn = simple_render_fn

        def counting_render_fn(obs: np.ndarray) -> np.ndarray:
            call_count["count"] += 1
            return original_fn(obs)

        render_episode_to_video(
            early_termination_episode_data,
            counting_render_fn,
            output_path,
        )
        # Should render 16 frames (0 to 15 inclusive, where 15 is the done step)
        assert call_count["count"] == 16, "Should only render up to done=True"

    def test_render_episode_with_max_frames(
        self,
        sample_episode_data: dict[str, np.ndarray],
        simple_render_fn,
        tmp_path: Path,
    ):
        """Test that max_frames parameter limits rendering."""
        output_path = tmp_path / "episode_limited.mp4"
        max_frames = 10
        # Track render calls
        call_count = {"count": 0}
        original_fn = simple_render_fn

        def counting_render_fn(obs: np.ndarray) -> np.ndarray:
            call_count["count"] += 1
            return original_fn(obs)

        render_episode_to_video(
            sample_episode_data,
            counting_render_fn,
            output_path,
            max_frames=max_frames,
        )
        assert call_count["count"] == max_frames, f"Should render exactly {max_frames} frames"

    def test_render_episode_custom_fps(
        self,
        sample_episode_data: dict[str, np.ndarray],
        simple_render_fn,
        tmp_path: Path,
    ):
        """Test that custom FPS parameter is accepted."""
        output_path = tmp_path / "episode_fps.mp4"
        result_path = render_episode_to_video(
            sample_episode_data,
            simple_render_fn,
            output_path,
            fps=30,
        )
        assert result_path.exists()

    def test_render_episode_returns_path(
        self,
        sample_episode_data: dict[str, np.ndarray],
        simple_render_fn,
        tmp_path: Path,
    ):
        """Test that function returns a Path object."""
        output_path = tmp_path / "episode.mp4"
        result_path = render_episode_to_video(
            sample_episode_data,
            simple_render_fn,
            output_path,
        )
        assert isinstance(result_path, Path)

    def test_render_episode_with_string_path(
        self,
        sample_episode_data: dict[str, np.ndarray],
        simple_render_fn,
        tmp_path: Path,
    ):
        """Test that function accepts string path."""
        output_path = str(tmp_path / "episode_string.mp4")
        result_path = render_episode_to_video(
            sample_episode_data,
            simple_render_fn,
            output_path,
        )
        assert result_path.exists()
        assert isinstance(result_path, Path)

    def test_render_episode_no_done_flag(
        self,
        simple_render_fn,
        tmp_path: Path,
    ):
        """Test episode without any done=True flags (uses full length)."""
        # Create episode that never terminates
        num_steps = 25
        episode_data = {
            "observations": np.random.randn(num_steps, 4).astype(np.float32),
            "actions": np.zeros(num_steps, dtype=np.int32),
            "rewards": np.ones(num_steps, dtype=np.float32),
            "dones": np.zeros(num_steps, dtype=bool),  # No done flags
        }
        output_path = tmp_path / "episode_no_done.mp4"
        call_count = {"count": 0}
        original_fn = simple_render_fn

        def counting_render_fn(obs: np.ndarray) -> np.ndarray:
            call_count["count"] += 1
            return original_fn(obs)

        render_episode_to_video(
            episode_data,
            counting_render_fn,
            output_path,
        )
        # Should render all frames when no done flag is present
        assert call_count["count"] == num_steps

    def test_render_episode_creates_parent_dirs(
        self,
        sample_episode_data: dict[str, np.ndarray],
        simple_render_fn,
        tmp_path: Path,
    ):
        """Test that parent directories are created if they don't exist."""
        output_path = tmp_path / "nested" / "path" / "episode.mp4"
        result_path = render_episode_to_video(
            sample_episode_data,
            simple_render_fn,
            output_path,
        )
        assert result_path.exists()
        assert result_path.parent.exists()
