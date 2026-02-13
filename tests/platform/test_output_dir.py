"""Tests for output_dir utilities."""

from pathlib import Path

from myriad.platform.output_dir import format_artifacts_tree, get_or_create_output_dir


class TestFormatArtifactsTree:
    def test_empty_dir(self, tmp_path):
        result = format_artifacts_tree(tmp_path)
        assert str(tmp_path) in result

    def test_with_results_file(self, tmp_path):
        (tmp_path / "results.pkl").touch()
        result = format_artifacts_tree(tmp_path)
        assert "results.pkl" in result
        assert "metrics & config" in result

    def test_with_log_file(self, tmp_path):
        (tmp_path / "run.log").touch()
        result = format_artifacts_tree(tmp_path)
        assert "run.log" in result
        assert "run log" in result

    def test_with_unknown_file(self, tmp_path):
        (tmp_path / "other.txt").touch()
        result = format_artifacts_tree(tmp_path)
        assert "other.txt" in result

    def test_with_checkpoints_dir(self, tmp_path):
        (tmp_path / "checkpoints").mkdir()
        result = format_artifacts_tree(tmp_path)
        assert "checkpoints/" in result

    def test_with_hydra_dir(self, tmp_path):
        (tmp_path / ".hydra").mkdir()
        result = format_artifacts_tree(tmp_path)
        assert ".hydra/" in result

    def test_with_episodes_dir_with_steps(self, tmp_path):
        episodes = tmp_path / "episodes"
        episodes.mkdir()
        (episodes / "step_00000100").mkdir()
        (episodes / "step_00000200").mkdir()
        result = format_artifacts_tree(tmp_path)
        assert "episodes/" in result
        assert "2 step checkpoints" in result

    def test_with_episodes_dir_one_step(self, tmp_path):
        episodes = tmp_path / "episodes"
        episodes.mkdir()
        (episodes / "step_00000100").mkdir()
        result = format_artifacts_tree(tmp_path)
        assert "1 step checkpoint" in result
        assert "checkpoints" not in result.replace("step checkpoint", "")

    def test_skips_dunder_dirs(self, tmp_path):
        (tmp_path / "__pycache__").mkdir()
        result = format_artifacts_tree(tmp_path)
        assert "__pycache__" not in result

    def test_tree_formatting(self, tmp_path):
        (tmp_path / "results.pkl").touch()
        (tmp_path / "run.log").touch()
        result = format_artifacts_tree(tmp_path)
        assert "├──" in result or "└──" in result


class TestGetOrCreateOutputDir:
    def test_explicit_dir_is_used(self, tmp_path):
        explicit = tmp_path / "my_output"
        result = get_or_create_output_dir(explicit)
        assert result == explicit
        assert explicit.exists()

    def test_explicit_dir_as_string(self, tmp_path):
        explicit = str(tmp_path / "my_output")
        result = get_or_create_output_dir(explicit)
        assert result == Path(explicit)

    def test_hydra_run_returns_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".hydra").mkdir()
        result = get_or_create_output_dir()
        assert result == tmp_path

    def test_non_hydra_creates_timestamped_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = get_or_create_output_dir()
        assert result.exists()
        assert result.is_dir()
