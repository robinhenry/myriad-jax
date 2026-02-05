"""Tests for logging utilities."""

import jax.numpy as jnp
import numpy as np
import pytest

from myriad.platform.logging import utils


def test_prepare_metrics_host_returns_host_arrays():
    metrics = {"loss": jnp.arange(5, dtype=jnp.float32)}
    result = utils.prepare_metrics_host(metrics, steps_this_chunk=3)
    assert np.allclose(result["loss"], np.array([0.0, 1.0, 2.0], dtype=np.float32))
    assert isinstance(result["loss"], np.ndarray)


def test_prepare_metrics_host_handles_invalid_inputs():
    assert utils.prepare_metrics_host({}, 1) == {}
    assert utils.prepare_metrics_host({"loss": jnp.ones(3)}, 0) == {}


def test_build_train_payload_uses_last_value():
    metrics_host = {"loss": np.array([0.2, 0.4], dtype=np.float32)}
    payload = utils.build_train_payload(metrics_host)
    assert payload == {"train/loss": pytest.approx(0.4)}


def test_build_train_payload_expands_vector_metric():
    metrics_host = {"advantage": np.array([[1.0, 3.0], [2.0, 6.0]], dtype=np.float32)}
    payload = utils.build_train_payload(metrics_host)
    assert payload["train/advantage/mean"] == pytest.approx(4.0)
    assert payload["train/advantage/std"] == pytest.approx(2.0)
    assert payload["train/advantage/max"] == pytest.approx(6.0)
    assert payload["train/advantage/min"] == pytest.approx(2.0)


def test_summarize_metric_supports_scalar_and_bool():
    scalar_result = utils.summarize_metric("test/", "value", np.array(3.5))
    assert scalar_result == {"test/value": pytest.approx(3.5)}

    bool_result = utils.summarize_metric("test/", "done", np.array([True, False]))
    assert bool_result["test/done/mean"] == pytest.approx(0.5)


def test_summarize_metric_rejects_non_numeric():
    assert utils.summarize_metric("test/", "value", np.array(["a", "b"])) == {}
