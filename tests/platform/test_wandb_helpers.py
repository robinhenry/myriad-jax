"""Tests for myriad.platform.wandb_helpers."""

import polars as pl
import pytest

import myriad.platform.wandb_helpers as wh
from myriad.platform.wandb_helpers import (
    _flatten_dict,
    _resolve_sweep_id,
    _unflatten_dotted_keys,
    _unwrap_wandb_value,
    config_from_wandb_run,
    fetch_run,
    fetch_sweep_runs,
    fetch_top_k_runs,
    runs_to_dataframe,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _MockRun:
    def __init__(self, run_id, name, state, config, summary, *, project="test-project", entity="test-entity"):
        self.id = run_id
        self.name = name
        self.state = state
        self.config = config
        self.summary = summary
        self.project = project
        self.entity = entity


class _WandbStub:
    def __init__(self, *, run=None, sweep_runs=None, projects=None, sweep_map=None):
        self._run = run
        self._sweep_runs = sweep_runs or []
        # projects: list of project name strings (for _resolve_sweep_id)
        self._projects = [type("Project", (), {"name": p})() for p in (projects or [])]
        # sweep_map: {fqid: sweep_obj} — raise if absent, return stub if present
        self._sweep_map = sweep_map or {}

    def Api(self):
        return self

    @property
    def default_entity(self):
        return "test-entity"

    def projects(self, _entity):
        return self._projects

    def run(self, _run_id):
        return self._run

    def sweep(self, sweep_id):
        if self._sweep_map:
            if sweep_id not in self._sweep_map:
                raise ValueError(f"sweep not found: {sweep_id}")
            return self._sweep_map[sweep_id]
        return type("Sweep", (), {"runs": self._sweep_runs})()


def _minimal_config_dict() -> dict:
    return {
        "run": {
            "seed": 0,
            "steps_per_env": 4,
            "num_envs": 1,
            "batch_size": 1,
            "buffer_size": 4,
            "scan_chunk_size": 2,
            "eval_frequency": 1,
            "eval_rollouts": 1,
            "eval_max_steps": 1,
        },
        "agent": {"name": "dummy"},
        "env": {"name": "dummy"},
    }


# ---------------------------------------------------------------------------
# _unwrap_wandb_value
# ---------------------------------------------------------------------------


def test_unwrap_passthrough():
    assert _unwrap_wandb_value(42) == 42
    assert _unwrap_wandb_value("hello") == "hello"


def test_unwrap_value_wrapper():
    assert _unwrap_wandb_value({"value": 3.14}) == 3.14


def test_unwrap_drops_underscore_keys_and_unwraps():
    raw = {"lr": {"value": 0.001}, "_internal": "skip", "batch": 32}
    assert _unwrap_wandb_value(raw) == {"lr": 0.001, "batch": 32}


def test_unwrap_nested():
    raw = {"agent": {"lr": {"value": 0.01}}}
    assert _unwrap_wandb_value(raw) == {"agent": {"lr": 0.01}}


# ---------------------------------------------------------------------------
# _unflatten_dotted_keys
# ---------------------------------------------------------------------------


def test_unflatten_no_dots():
    assert _unflatten_dotted_keys({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_unflatten_simple_dots():
    flat = {"env.name": "cartpole", "agent.lr": 0.001, "run.seed": 42}
    assert _unflatten_dotted_keys(flat) == {
        "env": {"name": "cartpole"},
        "agent": {"lr": 0.001},
        "run": {"seed": 42},
    }


def test_unflatten_deep_dots():
    flat = {"a.b.c": 1}
    assert _unflatten_dotted_keys(flat) == {"a": {"b": {"c": 1}}}


def test_unflatten_mixed_flat_and_nested():
    """Keys with and without dots should merge correctly under the same prefix."""
    flat = {"agent.lr": 0.01, "agent.gamma": 0.99}
    assert _unflatten_dotted_keys(flat) == {"agent": {"lr": 0.01, "gamma": 0.99}}


# ---------------------------------------------------------------------------
# _resolve_sweep_id
# ---------------------------------------------------------------------------


def test_resolve_sweep_id_already_qualified():
    assert _resolve_sweep_id("entity/project/abc123") == "entity/project/abc123"


def test_resolve_sweep_id_ambiguous_raises():
    with pytest.raises(ValueError, match="Ambiguous"):
        _resolve_sweep_id("project/abc123")


def test_resolve_sweep_id_bare_found(monkeypatch):
    fq = "test-entity/my-project/abc123"
    stub = _WandbStub(projects=["other-project", "my-project"], sweep_map={fq: object()})
    monkeypatch.setattr(wh, "wandb", stub)
    assert _resolve_sweep_id("abc123") == fq


def test_resolve_sweep_id_bare_not_found(monkeypatch):
    # sweep_map has an entry for a different ID so the stub raises for the target
    stub = _WandbStub(projects=["p1", "p2"], sweep_map={"test-entity/p1/other": object()})
    monkeypatch.setattr(wh, "wandb", stub)
    with pytest.raises(ValueError, match="Could not find sweep"):
        _resolve_sweep_id("missing123")


# ---------------------------------------------------------------------------
# _flatten_dict
# ---------------------------------------------------------------------------


def test_flatten_dict_flat():
    assert _flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_flatten_dict_nested():
    d = {"agent": {"lr": 0.01, "gamma": 0.99}, "run": {"seed": 0}}
    assert _flatten_dict(d) == {"agent.lr": 0.01, "agent.gamma": 0.99, "run.seed": 0}


# ---------------------------------------------------------------------------
# fetch_run
# ---------------------------------------------------------------------------


def test_fetch_run(monkeypatch):
    mock_run = _MockRun("r1", "run-1", "finished", {}, {})
    monkeypatch.setattr(wh, "wandb", _WandbStub(run=mock_run))
    assert fetch_run("entity/project/r1") is mock_run


# ---------------------------------------------------------------------------
# config_from_wandb_run
# ---------------------------------------------------------------------------


def test_config_from_wandb_run_plain():
    run = _MockRun("r1", "run-1", "finished", _minimal_config_dict(), {})
    config = config_from_wandb_run(run)
    assert config.run.seed == 0


def test_config_from_wandb_run_restores_project_from_run_metadata():
    """wandb section is stripped before logging; project/entity must be recovered from run."""
    run = _MockRun("r1", "run-1", "finished", _minimal_config_dict(), {}, project="my-project", entity="my-entity")
    config = config_from_wandb_run(run)
    assert config.wandb is not None
    assert config.wandb.project == "my-project"
    assert config.wandb.entity == "my-entity"


def test_config_from_wandb_run_does_not_override_existing_project():
    """If the config somehow already has a project, it should be kept."""
    raw = _minimal_config_dict()
    raw["wandb"] = {"project": "stored-project", "entity": "stored-entity"}
    run = _MockRun("r1", "run-1", "finished", raw, {}, project="run-project", entity="run-entity")
    config = config_from_wandb_run(run)
    assert config.wandb is not None
    assert config.wandb.project == "stored-project"
    assert config.wandb.entity == "stored-entity"


def test_config_from_wandb_run_unwraps_sweep_wrappers():
    raw = _minimal_config_dict()
    raw["run"]["seed"] = {"value": 7}
    run = _MockRun("r1", "run-1", "finished", raw, {})
    config = config_from_wandb_run(run)
    assert config.run.seed == 7


def test_config_from_wandb_run_flat_dotted_keys():
    """W&B sweep agents store config as flat dotted keys; these must be unflattened."""
    base = _minimal_config_dict()
    flat: dict = {}
    for section, fields in base.items():
        for k, v in fields.items():
            flat[f"{section}.{k}"] = v
    run = _MockRun("r1", "run-1", "finished", flat, {}, project="sweep-project", entity="sweep-entity")
    config = config_from_wandb_run(run)
    assert config.run.seed == 0
    assert config.agent.name == "dummy"
    assert config.env.name == "dummy"
    assert config.wandb is not None
    assert config.wandb.project == "sweep-project"


# ---------------------------------------------------------------------------
# fetch_sweep_runs
# ---------------------------------------------------------------------------


def test_fetch_sweep_runs_no_filter(monkeypatch):
    runs = [
        _MockRun("r1", "a", "finished", {}, {}),
        _MockRun("r2", "b", "running", {}, {}),
    ]
    monkeypatch.setattr(wh, "wandb", _WandbStub(sweep_runs=runs))
    assert len(fetch_sweep_runs("e/p/s")) == 2


def test_fetch_sweep_runs_state_filter(monkeypatch):
    runs = [
        _MockRun("r1", "a", "finished", {}, {}),
        _MockRun("r2", "b", "crashed", {}, {}),
    ]
    monkeypatch.setattr(wh, "wandb", _WandbStub(sweep_runs=runs))
    result = fetch_sweep_runs("e/p/s", state="finished")
    assert len(result) == 1 and result[0].id == "r1"


# ---------------------------------------------------------------------------
# fetch_top_k_runs
# ---------------------------------------------------------------------------


def test_fetch_top_k_runs_maximize(monkeypatch):
    runs = [
        _MockRun("r1", "a", "finished", {}, {"reward": 10.0}),
        _MockRun("r2", "b", "finished", {}, {"reward": 30.0}),
        _MockRun("r3", "c", "finished", {}, {"reward": 20.0}),
    ]
    monkeypatch.setattr(wh, "wandb", _WandbStub(sweep_runs=runs))
    result = fetch_top_k_runs("e/p/s", "reward", 2, maximize=True)
    assert [r.id for r in result] == ["r2", "r3"]


def test_fetch_top_k_runs_minimize(monkeypatch):
    runs = [
        _MockRun("r1", "a", "finished", {}, {"loss": 0.5}),
        _MockRun("r2", "b", "finished", {}, {"loss": 0.1}),
        _MockRun("r3", "c", "finished", {}, {"loss": 0.3}),
    ]
    monkeypatch.setattr(wh, "wandb", _WandbStub(sweep_runs=runs))
    result = fetch_top_k_runs("e/p/s", "loss", 2, maximize=False)
    assert [r.id for r in result] == ["r2", "r3"]


def test_fetch_top_k_runs_warns_when_fewer_than_k(monkeypatch):
    runs = [_MockRun("r1", "a", "finished", {}, {"reward": 1.0})]
    monkeypatch.setattr(wh, "wandb", _WandbStub(sweep_runs=runs))
    with pytest.warns(UserWarning, match="only 1 finished"):
        result = fetch_top_k_runs("e/p/s", "reward", 3, maximize=True)
    assert len(result) == 1


def test_fetch_top_k_runs_missing_metric_sorts_last(monkeypatch):
    runs = [
        _MockRun("r1", "a", "finished", {}, {}),  # metric absent
        _MockRun("r2", "b", "finished", {}, {"reward": 5.0}),
    ]
    monkeypatch.setattr(wh, "wandb", _WandbStub(sweep_runs=runs))
    result = fetch_top_k_runs("e/p/s", "reward", 2, maximize=True)
    assert result[0].id == "r2"


# ---------------------------------------------------------------------------
# runs_to_dataframe
# ---------------------------------------------------------------------------


def test_runs_to_dataframe_returns_polars():
    runs = [
        _MockRun("r1", "run-1", "finished", {"agent": {"lr": 0.01}}, {"reward": 5.0, "_step": 100}),
        _MockRun("r2", "run-2", "finished", {"agent": {"lr": 0.001}}, {"reward": 8.0}),
    ]
    df = runs_to_dataframe(runs)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 2
    assert "agent.lr" in df.columns
    assert "reward" in df.columns
    assert "_step" not in df.columns  # underscore-prefixed keys excluded


def test_runs_to_dataframe_metrics_filter():
    runs = [_MockRun("r1", "run-1", "finished", {}, {"reward": 5.0, "loss": 0.3})]
    df = runs_to_dataframe(runs, metrics=["reward"])
    assert "reward" in df.columns
    assert "loss" not in df.columns
