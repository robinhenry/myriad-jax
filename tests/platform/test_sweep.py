"""Tests for myriad.platform.sweep - W&B sweep creation utilities."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


def _make_sweep_yaml(tmp_path: Path, *, project: str | None = "my-project") -> Path:
    """Create a minimal sweep YAML file for testing."""
    cfg: dict = {
        "method": "random",
        "parameters": {
            "agent.learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-4,
                "max": 1e-2,
            },
        },
    }
    if project is not None:
        cfg["project"] = project
    path = tmp_path / "sweep.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


def _fake_run_success(sweep_id: str = "abc123", entity: str = "myentity", project: str = "my-project"):
    """Return a mock subprocess.CompletedProcess with W&B-style output."""
    result = MagicMock()
    result.returncode = 0
    result.stderr = f"wandb: Creating sweep with ID: {sweep_id}\nwandb agent {entity}/{project}/{sweep_id}\n"
    return result


class TestCreateWandbSweeps:
    def test_single_sweep_no_levels(self, tmp_path):
        """create_wandb_sweeps with no levels should create exactly one sweep."""
        from myriad.platform.sweep import create_wandb_sweeps

        yaml_path = _make_sweep_yaml(tmp_path)
        with patch("subprocess.run", return_value=_fake_run_success("s1")) as mock_run:
            sweep_ids = create_wandb_sweeps(yaml_path, project="my-project")

        assert len(sweep_ids) == 1
        assert sweep_ids[0] == "myentity/my-project/s1"
        mock_run.assert_called_once()

    def test_multiple_levels_creates_one_sweep_per_level(self, tmp_path):
        """create_wandb_sweeps with levels should create one sweep per level value."""
        from myriad.platform.sweep import create_wandb_sweeps

        yaml_path = _make_sweep_yaml(tmp_path)
        counter = {"n": 0}

        def fake_run(cmd, **kwargs):
            counter["n"] += 1
            result = MagicMock()
            result.returncode = 0
            result.stderr = f"wandb agent ent/proj/sweep{counter['n']}\n"
            return result

        with patch("subprocess.run", side_effect=fake_run):
            sweep_ids = create_wandb_sweeps(yaml_path, project="proj", levels=[512, 1024, 2048])

        assert len(sweep_ids) == 3
        assert sweep_ids[0] == "ent/proj/sweep1"
        assert sweep_ids[1] == "ent/proj/sweep2"
        assert sweep_ids[2] == "ent/proj/sweep3"

    def test_level_param_patched_in_yaml(self, tmp_path):
        """Each level value should be written into the temp YAML under level_param."""
        from myriad.platform.sweep import create_wandb_sweeps

        yaml_path = _make_sweep_yaml(tmp_path)
        written: list[dict] = []

        def fake_run(cmd, **kwargs):
            with open(cmd[-1]) as f:
                written.append(yaml.safe_load(f))
            result = MagicMock()
            result.returncode = 0
            result.stderr = "wandb agent ent/proj/s1\n"
            return result

        with patch("subprocess.run", side_effect=fake_run):
            create_wandb_sweeps(yaml_path, project="proj", levels=[512], level_param="run.num_envs")

        assert len(written) == 1
        assert written[0]["parameters"]["run.num_envs"] == {"value": 512}

    def test_base_group_suffixed_with_level(self, tmp_path):
        """When base_group is given, wandb.group should be '{base_group}_{level}'."""
        from myriad.platform.sweep import create_wandb_sweeps

        yaml_path = _make_sweep_yaml(tmp_path)
        written: list[dict] = []

        def fake_run(cmd, **kwargs):
            with open(cmd[-1]) as f:
                written.append(yaml.safe_load(f))
            result = MagicMock()
            result.returncode = 0
            result.stderr = "wandb agent ent/proj/s1\n"
            return result

        with patch("subprocess.run", side_effect=fake_run):
            create_wandb_sweeps(yaml_path, project="proj", levels=[1024], base_group="myexp")

        assert written[0]["parameters"]["wandb.group"] == {"value": "myexp_1024"}

    def test_group_falls_back_to_yaml_wandb_group(self, tmp_path):
        """Without base_group, uses existing 'wandb.group' from YAML parameters."""
        from myriad.platform.sweep import create_wandb_sweeps

        cfg = {
            "method": "random",
            "parameters": {"wandb.group": {"value": "existing-group"}},
        }
        yaml_path = tmp_path / "sweep.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f)

        written: list[dict] = []

        def fake_run(cmd, **kwargs):
            with open(cmd[-1]) as f:
                written.append(yaml.safe_load(f))
            result = MagicMock()
            result.returncode = 0
            result.stderr = "wandb agent ent/proj/s1\n"
            return result

        with patch("subprocess.run", side_effect=fake_run):
            create_wandb_sweeps(yaml_path, project="proj", levels=[512])

        assert written[0]["parameters"]["wandb.group"] == {"value": "existing-group_512"}

    def test_group_falls_back_to_project_name(self, tmp_path):
        """Without base_group and no wandb.group in YAML, group is '{project}_{level}'."""
        from myriad.platform.sweep import create_wandb_sweeps

        yaml_path = _make_sweep_yaml(tmp_path)
        written: list[dict] = []

        def fake_run(cmd, **kwargs):
            with open(cmd[-1]) as f:
                written.append(yaml.safe_load(f))
            result = MagicMock()
            result.returncode = 0
            result.stderr = "wandb agent ent/proj/s1\n"
            return result

        with patch("subprocess.run", side_effect=fake_run):
            create_wandb_sweeps(yaml_path, project="myproject", levels=[256])

        assert written[0]["parameters"]["wandb.group"] == {"value": "myproject_256"}

    def test_project_read_from_yaml_when_not_passed(self, tmp_path):
        """Project name should be read from YAML 'project' field if not passed."""
        from myriad.platform.sweep import create_wandb_sweeps

        yaml_path = _make_sweep_yaml(tmp_path, project="yaml-project")

        subprocess_args: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            subprocess_args.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            result.stderr = "wandb agent ent/yaml-project/s1\n"
            return result

        with patch("subprocess.run", side_effect=fake_run):
            sweep_ids = create_wandb_sweeps(yaml_path)  # no project arg

        assert len(sweep_ids) == 1
        assert "yaml-project" in subprocess_args[0]

    def test_raises_value_error_when_project_missing(self, tmp_path):
        """Should raise ValueError when project is absent from both args and YAML."""
        from myriad.platform.sweep import create_wandb_sweeps

        yaml_path = _make_sweep_yaml(tmp_path, project=None)
        with pytest.raises(ValueError, match="project must be specified"):
            create_wandb_sweeps(yaml_path)

    def test_temp_file_deleted_after_success(self, tmp_path):
        """Temporary YAML file written for wandb CLI should be deleted on success."""
        from myriad.platform.sweep import create_wandb_sweeps

        yaml_path = _make_sweep_yaml(tmp_path)
        tmp_files: list[str] = []

        def fake_run(cmd, **kwargs):
            tmp_files.append(cmd[-1])
            result = MagicMock()
            result.returncode = 0
            result.stderr = "wandb agent ent/proj/s1\n"
            return result

        with patch("subprocess.run", side_effect=fake_run):
            create_wandb_sweeps(yaml_path, project="proj")

        assert len(tmp_files) == 1
        assert not Path(tmp_files[0]).exists(), "Temp file was not cleaned up"

    def test_temp_file_deleted_after_failure(self, tmp_path):
        """Temporary YAML file should be deleted even when wandb CLI fails."""
        from myriad.platform.sweep import create_wandb_sweeps

        yaml_path = _make_sweep_yaml(tmp_path)
        tmp_files: list[str] = []

        def fake_run(cmd, **kwargs):
            tmp_files.append(cmd[-1])
            result = MagicMock()
            result.returncode = 1
            result.stderr = "Error: something went wrong"
            return result

        with patch("subprocess.run", side_effect=fake_run):
            with pytest.raises(RuntimeError):
                create_wandb_sweeps(yaml_path, project="proj")

        assert len(tmp_files) == 1
        assert not Path(tmp_files[0]).exists(), "Temp file was not cleaned up after failure"


class TestRegisterSweep:
    def test_raises_runtime_error_on_nonzero_exit(self):
        """_register_sweep should raise RuntimeError when wandb CLI exits non-zero."""
        from myriad.platform.sweep import _register_sweep

        result = MagicMock()
        result.returncode = 1
        result.stderr = "Error: wandb sweep failed"

        with patch("subprocess.run", return_value=result):
            with pytest.raises(RuntimeError, match="wandb sweep failed"):
                _register_sweep({}, "my-project")

    def test_raises_runtime_error_when_no_agent_line(self):
        """_register_sweep should raise RuntimeError when output has no 'wandb agent' line."""
        from myriad.platform.sweep import _register_sweep

        result = MagicMock()
        result.returncode = 0
        result.stderr = "wandb: Sweep created successfully.\n"  # No "wandb agent ..." line

        with patch("subprocess.run", return_value=result):
            with pytest.raises(RuntimeError, match="Could not parse sweep ID"):
                _register_sweep({}, "my-project")

    def test_returns_sweep_id_from_agent_line(self):
        """_register_sweep should extract the sweep ID from 'wandb agent ...' output."""
        from myriad.platform.sweep import _register_sweep

        result = MagicMock()
        result.returncode = 0
        result.stderr = "wandb: Creating sweep\nwandb agent entity/project/abc123\n"

        with patch("subprocess.run", return_value=result):
            sweep_id = _register_sweep({}, "project")

        assert sweep_id == "entity/project/abc123"

    def test_returns_last_agent_line_when_multiple(self):
        """When multiple 'wandb agent' lines exist, _register_sweep uses the last one."""
        from myriad.platform.sweep import _register_sweep

        result = MagicMock()
        result.returncode = 0
        result.stderr = (
            "wandb agent entity/project/first\n" "wandb: Some other output\n" "wandb agent entity/project/last\n"
        )

        with patch("subprocess.run", return_value=result):
            sweep_id = _register_sweep({}, "project")

        assert sweep_id == "entity/project/last"
