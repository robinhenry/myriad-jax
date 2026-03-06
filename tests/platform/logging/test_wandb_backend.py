"""Tests for Wandb backend initialization."""

from unittest.mock import MagicMock

import pytest

from myriad.configs.default import AgentConfig, Config, EnvConfig, RunConfig, WandbConfig
from myriad.platform.logging.backends import wandb as wandb_backend


class _WandbStub:
    def __init__(self):
        self.init_kwargs: dict | None = None
        self.run = None  # init_wandb checks wandb.run to detect an active run

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
    assert stub.init_kwargs["config"]["run.seed"] == 0


def test_init_wandb_raises_without_package(monkeypatch):
    config = _make_config(enabled=True)
    monkeypatch.setattr(wandb_backend, "wandb", None)
    monkeypatch.setattr(wandb_backend, "_wandb_import_error", ImportError("missing"), raising=False)
    with pytest.raises(RuntimeError):
        wandb_backend.init_wandb(config)


def test_init_wandb_reuses_active_run(monkeypatch):
    """init_wandb should reuse an existing wandb.run instead of calling init() again."""
    config = _make_config(enabled=True)
    stub = _WandbStub()

    # Simulate a W&B run already active (e.g. sweep_main called wandb.init() first)
    mock_run = MagicMock()
    mock_run.config = {}  # no keys set yet by sweep agent
    stub.run = mock_run

    monkeypatch.setattr(wandb_backend, "wandb", stub)
    monkeypatch.setattr(wandb_backend, "_wandb_import_error", None, raising=False)

    returned = wandb_backend.init_wandb(config)

    # Should return the pre-existing run, not call init
    assert returned is mock_run
    assert stub.init_kwargs is None  # init() must NOT have been called


def test_wandb_backend_local_dir_when_disabled():
    """WandbBackend.local_dir should return None when use_wandb is False."""
    from myriad.platform.logging.backends.wandb import WandbBackend

    backend = WandbBackend(wandb_run=None)
    assert backend.local_dir is None


def test_wandb_backend_local_dir_when_enabled(monkeypatch):
    """WandbBackend.local_dir should return the run's dir as a Path."""
    from myriad.platform.logging.backends.wandb import WandbBackend

    mock_run = MagicMock()
    mock_run.dir = "/tmp/wandb/run-abc"

    stub_wandb = MagicMock()
    monkeypatch.setattr(wandb_backend, "wandb", stub_wandb)

    backend = WandbBackend(wandb_run=mock_run)
    assert backend.local_dir is not None
    assert str(backend.local_dir) == "/tmp/wandb/run-abc"


def test_wandb_backend_log_run_summary(monkeypatch):
    """WandbBackend.log_run_summary should write cfg/* keys to wandb_run.summary."""
    from myriad.platform.logging.backends.wandb import WandbBackend

    mock_run = MagicMock()
    mock_run.summary = MagicMock()

    stub_wandb = MagicMock()
    monkeypatch.setattr(wandb_backend, "wandb", stub_wandb)

    config = Config(
        run=RunConfig(
            steps_per_env=1000,
            num_envs=4,
            eval_rollouts=2,
            eval_max_steps=5,
            batch_size=32,
            buffer_size=1000,
            scan_chunk_size=100,
            eval_frequency=100,
        ),
        agent=AgentConfig(name="dqn"),
        env=EnvConfig(name="cartpole-control"),
        wandb=WandbConfig(enabled=True),
    )

    backend = WandbBackend(wandb_run=mock_run)
    backend.log_run_summary(config)

    mock_run.summary.update.assert_called_once()
    summary_dict = mock_run.summary.update.call_args[0][0]
    assert summary_dict["cfg/num_envs"] == 4
    assert summary_dict["cfg/steps_per_env"] == 1000
    assert summary_dict["cfg/agent"] == "dqn"
    assert summary_dict["cfg/env"] == "cartpole-control"
