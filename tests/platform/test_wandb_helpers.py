"""Tests for myriad.platform.wandb_helpers."""

import polars as pl
import pytest

import myriad.platform.wandb_helpers as wh
from myriad.platform.wandb_helpers import (
    _flatten_dict,
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
    def __init__(self, run_id, name, state, config, summary):
        self.id = run_id
        self.name = name
        self.state = state
        self.config = config
        self.summary = summary


class _WandbStub:
    def __init__(self, *, run=None, sweep_runs=None):
        self._run = run
        self._sweep_runs = sweep_runs or []

    def Api(self):
        return self

    def run(self, _run_id):
        return self._run

    def sweep(self, _sweep_id):
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


def test_config_from_wandb_run_unwraps_sweep_wrappers():
    raw = _minimal_config_dict()
    raw["run"]["seed"] = {"value": 7}
    run = _MockRun("r1", "run-1", "finished", raw, {})
    config = config_from_wandb_run(run)
    assert config.run.seed == 7


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
