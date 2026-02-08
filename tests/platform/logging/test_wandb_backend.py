"""Tests for Wandb backend initialization."""

import pytest

from myriad.configs.default import AgentConfig, Config, EnvConfig, RunConfig, WandbConfig
from myriad.platform.logging.backends import wandb as wandb_backend


class _WandbStub:
    def __init__(self):
        self.init_kwargs: dict | None = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return "stub-run"


def _make_config(*, enabled: bool) -> Config:
    run_cfg = RunConfig(
        seed=0,
        steps_per_env=4,
        num_envs=1,
        batch_size=1,
        buffer_size=4,
        scan_chunk_size=2,
        eval_frequency=1,
        eval_rollouts=1,
        eval_max_steps=1,
        log_frequency=1,
    )
    wandb_cfg = WandbConfig(
        enabled=enabled,
        project="demo" if enabled else "myriad",
        run_name="unit" if enabled else None,
        tags=("unit",) if enabled else (),
    )
    return Config(
        run=run_cfg,
        agent=AgentConfig(name="dummy"),
        env=EnvConfig(name="dummy"),
        wandb=wandb_cfg,
    )


def test_init_wandb_disabled(monkeypatch):
    config = _make_config(enabled=False)
    assert wandb_backend.init_wandb(config) is None


def test_init_wandb_initializes_when_enabled(monkeypatch):
    config = _make_config(enabled=True)
    stub = _WandbStub()
    monkeypatch.setattr(wandb_backend, "wandb", stub)
    monkeypatch.setattr(wandb_backend, "_wandb_import_error", None, raising=False)
    run = wandb_backend.init_wandb(config)
    assert run == "stub-run"
    assert stub.init_kwargs is not None
    assert stub.init_kwargs["config"]["run"]["seed"] == 0


def test_init_wandb_raises_without_package(monkeypatch):
    config = _make_config(enabled=True)
    monkeypatch.setattr(wandb_backend, "wandb", None)
    monkeypatch.setattr(wandb_backend, "_wandb_import_error", ImportError("missing"), raising=False)
    with pytest.raises(RuntimeError):
        wandb_backend.init_wandb(config)
