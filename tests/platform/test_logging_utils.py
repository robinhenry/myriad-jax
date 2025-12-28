import jax.numpy as jnp
import numpy as np
import pytest

from myriad.configs.default import AgentConfig, Config, EnvConfig, RunConfig, WandbConfig
from myriad.platform import logging_utils


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


def test_drop_none_filters_empty_entries():
    data = {"project": "demo", "entity": None, "mode": "offline"}
    assert logging_utils._drop_none(data) == {"project": "demo", "mode": "offline"}


def test_prepare_metrics_host_returns_host_arrays():
    metrics = {"loss": jnp.arange(5, dtype=jnp.float32)}
    result = logging_utils.prepare_metrics_host(metrics, steps_this_chunk=3)
    assert np.allclose(result["loss"], np.array([0.0, 1.0, 2.0], dtype=np.float32))
    assert isinstance(result["loss"], np.ndarray)


def test_prepare_metrics_host_handles_invalid_inputs():
    assert logging_utils.prepare_metrics_host({}, 1) == {}
    assert logging_utils.prepare_metrics_host({"loss": jnp.ones(3)}, 0) == {}
    assert logging_utils.prepare_metrics_host("not-a-dict", 2) == {}


def test_build_train_payload_uses_last_value():
    metrics_host = {"loss": np.array([0.2, 0.4], dtype=np.float32)}
    payload = logging_utils.build_train_payload(metrics_host)
    assert payload == {"train/loss": pytest.approx(0.4)}


def test_build_train_payload_expands_vector_metric():
    metrics_host = {"advantage": np.array([[1.0, 3.0], [2.0, 6.0]], dtype=np.float32)}
    payload = logging_utils.build_train_payload(metrics_host)
    assert payload["train/advantage/mean"] == pytest.approx(4.0)
    assert payload["train/advantage/std"] == pytest.approx(2.0)
    assert payload["train/advantage/max"] == pytest.approx(6.0)
    assert payload["train/advantage/min"] == pytest.approx(2.0)


def test_summarize_metric_supports_scalar_and_bool():
    scalar_result = logging_utils.summarize_metric("test/", "value", np.array(3.5))
    assert scalar_result == {"test/value": pytest.approx(3.5)}

    bool_result = logging_utils.summarize_metric("test/", "done", np.array([True, False]))
    assert bool_result["test/done/mean"] == pytest.approx(0.5)


def test_summarize_metric_rejects_non_numeric():
    assert logging_utils.summarize_metric("test/", "value", np.array(["a", "b"])) == {}


def test_maybe_init_wandb_disabled(monkeypatch):
    config = _make_config(enabled=False)
    assert logging_utils.maybe_init_wandb(config) is None


def test_maybe_init_wandb_initializes_when_enabled(monkeypatch):
    config = _make_config(enabled=True)
    stub = _WandbStub()
    monkeypatch.setattr(logging_utils, "wandb", stub)
    monkeypatch.setattr(logging_utils, "_wandb_import_error", None, raising=False)
    run = logging_utils.maybe_init_wandb(config)
    assert run == "stub-run"
    assert stub.init_kwargs is not None
    assert stub.init_kwargs["config"]["run"]["seed"] == 0


def test_maybe_init_wandb_raises_without_package(monkeypatch):
    config = _make_config(enabled=True)
    monkeypatch.setattr(logging_utils, "wandb", None)
    monkeypatch.setattr(logging_utils, "_wandb_import_error", ImportError("missing"), raising=False)
    with pytest.raises(RuntimeError):
        logging_utils.maybe_init_wandb(config)
