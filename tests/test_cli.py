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


@pytest.mark.parametrize("command", ["train", "sweep", "evaluate"])
def test_auto_tune_flag_not_in_help(runner, command):
    """--auto-tune should not appear in any command help (it was removed)."""
    result = runner.invoke(main, [command, "--help"])
    assert result.exit_code == 0
    assert "--auto-tune" not in result.output


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


# ---------------------------------------------------------------------------
# sweep-create CLI tests
# ---------------------------------------------------------------------------


def test_sweep_create_help(runner):
    """sweep-create --help should show usage info."""
    result = runner.invoke(main, ["sweep-create", "--help"])
    assert result.exit_code == 0
    assert "YAML_PATH" in result.output


def test_sweep_create_single_sweep(runner, tmp_path):
    """sweep-create with a single sweep should print the sweep ID to stdout."""
    from unittest.mock import patch

    import yaml

    sweep_yaml = tmp_path / "sweep.yaml"
    sweep_yaml.write_text(yaml.dump({"method": "random", "parameters": {}}))

    with patch("myriad.platform.sweep.create_wandb_sweeps", return_value=["ent/proj/abc123"]):
        result = runner.invoke(main, ["sweep-create", str(sweep_yaml), "--project", "my-project"])

    assert result.exit_code == 0
    assert "ent/proj/abc123" in result.output


def test_sweep_create_with_levels(runner, tmp_path):
    """sweep-create --levels should print one sweep ID per level to stdout."""
    from unittest.mock import patch

    import yaml

    sweep_yaml = tmp_path / "sweep.yaml"
    sweep_yaml.write_text(yaml.dump({"method": "random", "parameters": {}}))

    with patch(
        "myriad.platform.sweep.create_wandb_sweeps",
        return_value=["ent/proj/s1", "ent/proj/s2"],
    ):
        result = runner.invoke(
            main,
            [
                "sweep-create",
                str(sweep_yaml),
                "--project",
                "my-project",
                "--levels",
                "512",
                "--levels",
                "1024",
            ],
        )

    assert result.exit_code == 0
    assert "ent/proj/s1" in result.output
    assert "ent/proj/s2" in result.output


def test_sweep_create_error_exits_nonzero(runner, tmp_path):
    """sweep-create should exit with code 1 on RuntimeError from create_wandb_sweeps."""
    from unittest.mock import patch

    import yaml

    sweep_yaml = tmp_path / "sweep.yaml"
    sweep_yaml.write_text(yaml.dump({"method": "random", "parameters": {}}))

    with patch(
        "myriad.platform.sweep.create_wandb_sweeps",
        side_effect=RuntimeError("wandb sweep failed"),
    ):
        result = runner.invoke(main, ["sweep-create", str(sweep_yaml), "--project", "p"])

    assert result.exit_code == 1


def test_sweep_create_project_from_yaml(runner, tmp_path):
    """sweep-create without --project should still work when YAML has 'project' field."""
    from unittest.mock import patch

    import yaml

    sweep_yaml = tmp_path / "sweep.yaml"
    sweep_yaml.write_text(yaml.dump({"method": "random", "project": "yaml-proj", "parameters": {}}))

    with patch(
        "myriad.platform.sweep.create_wandb_sweeps",
        return_value=["ent/yaml-proj/s1"],
    ) as mock_create:
        result = runner.invoke(main, ["sweep-create", str(sweep_yaml)])

    assert result.exit_code == 0
    # project arg should be None (falling back to YAML)
    assert mock_create.call_args[0][1] is None


def test_sweep_create_parses_float_and_str_levels(runner, tmp_path):
    """sweep-create should parse float and non-numeric string level values correctly."""
    from unittest.mock import patch

    import yaml

    sweep_yaml = tmp_path / "sweep.yaml"
    sweep_yaml.write_text(yaml.dump({"method": "random", "parameters": {}}))

    captured_levels: list = []

    def fake_create(yaml_path, project, *, levels=None, **kwargs):
        if levels:
            captured_levels.extend(levels)
        return [f"ent/proj/s{i}" for i in range(len(levels or [None]))]

    with patch("myriad.platform.sweep.create_wandb_sweeps", side_effect=fake_create):
        result = runner.invoke(
            main,
            [
                "sweep-create",
                str(sweep_yaml),
                "--project",
                "p",
                "--levels",
                "1.5",  # float
                "--levels",
                "small",  # string
            ],
        )

    assert result.exit_code == 0
    assert 1.5 in captured_levels
    assert "small" in captured_levels


# ---------------------------------------------------------------------------
# seed-eval CLI tests
# ---------------------------------------------------------------------------


def test_seed_eval_help(runner):
    """seed-eval --help should show usage info."""
    result = runner.invoke(main, ["seed-eval", "--help"])
    assert result.exit_code == 0
    assert "SWEEP_ID" in result.output


def test_seed_eval_invokes_run_seed_eval(runner):
    """seed-eval should call run_seed_eval with correct parsed arguments."""
    from unittest.mock import patch

    with patch("myriad.platform.seed_eval.run_seed_eval") as mock_run:
        result = runner.invoke(
            main,
            [
                "seed-eval",
                "ent/proj/abc123",
                "--top-k",
                "2",
                "--seeds",
                "0,1",
                "--metric",
                "eval/return",
                "--group",
                "test_group",
                "--mode",
                "disabled",
            ],
        )

    assert result.exit_code == 0
    mock_run.assert_called_once()
    call_args, call_kwargs = mock_run.call_args
    assert call_args[0] == "ent/proj/abc123"
    assert call_args[1] == 2  # top_k
    assert call_args[2] == [0, 1]  # seeds


def test_seed_eval_minimize_flag(runner):
    """--minimize flag should pass maximize=False to run_seed_eval."""
    from unittest.mock import patch

    with patch("myriad.platform.seed_eval.run_seed_eval") as mock_run:
        result = runner.invoke(
            main,
            [
                "seed-eval",
                "ent/proj/abc123",
                "--group",
                "g",
                "--mode",
                "disabled",
                "--minimize",
            ],
        )

    assert result.exit_code == 0
    _, call_kwargs = mock_run.call_args
    assert call_kwargs["maximize"] is False


def test_seed_eval_default_maximize(runner):
    """Without --minimize, run_seed_eval should receive maximize=True."""
    from unittest.mock import patch

    with patch("myriad.platform.seed_eval.run_seed_eval") as mock_run:
        result = runner.invoke(
            main,
            ["seed-eval", "ent/proj/abc123", "--group", "g", "--mode", "disabled"],
        )

    assert result.exit_code == 0
    _, call_kwargs = mock_run.call_args
    assert call_kwargs["maximize"] is True
