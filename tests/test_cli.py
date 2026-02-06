"""Unit tests for the Myriad CLI.

This module tests the Click-based CLI entry points, ensuring:
1. Help messages are displayed correctly.
2. Commands are registered properly.
3. The render command handles input/output and environment lookup correctly.
"""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from myriad.cli import main


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_help(runner):
    """Test the main myriad help command."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Usage: main [OPTIONS] COMMAND [ARGS]..." in result.output
    assert "train" in result.output
    assert "evaluate" in result.output
    assert "sweep" in result.output
    assert "render" in result.output


@pytest.mark.parametrize("command", ["train", "evaluate", "sweep", "render"])
def test_subcommand_help(runner, command):
    """Test help for each subcommand."""
    result = runner.invoke(main, [command, "--help"])
    assert result.exit_code == 0
    assert f"Usage: main {command}" in result.output


def test_render_missing_args(runner):
    """Test render command fails when missing required arguments."""
    # Missing INPUT_PATH and --env
    result = runner.invoke(main, ["render"])
    assert result.exit_code != 0
    # Click uses single quotes for argument names in newer versions
    assert "Error: Missing argument 'INPUT_PATH'" in result.output


def test_render_invalid_env(runner, tmp_path, caplog):
    """Test render command with an invalid environment name."""
    dummy_file = tmp_path / "episode.npz"
    np.savez(dummy_file, observations=np.zeros((10, 4)), dones=np.zeros(10, dtype=bool))

    with caplog.at_level("ERROR"):
        runner.invoke(main, ["render", str(dummy_file), "--env", "non-existent-env"])
        assert "No renderer available for environment 'non-existent-env'" in caplog.text


def test_render_single_file(runner, tmp_path, caplog):
    """Test render command with a single .npz file.

    Note: We mock the actual rendering because it requires imageio/ffmpeg.
    """
    dummy_file = tmp_path / "episode.npz"
    # Create a dummy .npz file that looks like a myriad episode
    np.savez(
        dummy_file,
        observations=np.zeros((5, 4)),
        dones=np.array([False, False, False, False, True]),
        actions=np.zeros((5, 1)),
        rewards=np.zeros(5),
    )

    # We use a real environment but mock the rendering function to avoid dependencies
    from unittest.mock import patch

    with patch("myriad.utils.rendering.render_episode_to_video") as mock_render:
        mock_render.return_value = Path("videos/episode.mp4")

        with caplog.at_level("INFO"):
            result = runner.invoke(main, ["render", str(dummy_file), "--env", "cartpole-control"])

            assert result.exit_code == 0
            assert "Found 1 episode(s) to render" in caplog.text
            assert "Rendering episode.npz..." in caplog.text
            mock_render.assert_called_once()


def test_render_directory(runner, tmp_path, caplog):
    """Test render command with a directory of .npz files."""
    episodes_dir = tmp_path / "episodes"
    episodes_dir.mkdir()

    for i in range(3):
        f = episodes_dir / f"ep_{i}.npz"
        np.savez(f, observations=np.zeros((2, 4)), dones=np.array([False, True]))

    from unittest.mock import patch

    with patch("myriad.utils.rendering.render_episode_to_video") as mock_render:
        with caplog.at_level("INFO"):
            result = runner.invoke(main, ["render", str(episodes_dir), "--env", "cartpole-control"])

            assert result.exit_code == 0
            assert "Found 3 episode(s) to render" in caplog.text
            assert mock_render.call_count == 3
