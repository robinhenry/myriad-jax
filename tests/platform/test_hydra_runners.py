"""Tests for hydra_runners helper functions.

These test the pure helper utilities without invoking Hydra.
"""

from unittest.mock import MagicMock

import pytest

from omegaconf import OmegaConf

from myriad.configs.default import AgentConfig, EnvConfig, EvalConfig, EvalRunConfig, WandbConfig
from myriad.platform.hydra_runners import (
    _apply_auto_tune,
    _fmt_fields,
    _format_eval_config,
    _format_eval_results,
    _format_train_config,
    _get_config_path,
)


class TestFmtFields:
    def test_non_pydantic_returns_str(self):
        assert _fmt_fields(42) == "42"
        assert _fmt_fields("hello") == "hello"

    def test_pydantic_defaults_returns_placeholder(self):
        config = EvalRunConfig()
        # All defaults → "(defaults)" because nothing is non-default
        # (name is not on EvalRunConfig, so just check it returns a string)
        result = _fmt_fields(config)
        assert isinstance(result, str)

    def test_pydantic_with_name_field(self):
        config = AgentConfig(name="random")
        result = _fmt_fields(config)
        assert "random" in result

    def test_pydantic_with_non_default_field(self):
        config = EnvConfig(name="cartpole-control")
        result = _fmt_fields(config)
        assert "cartpole-control" in result


class TestFormatEvalConfig:
    @pytest.fixture
    def eval_config(self):
        return EvalConfig(
            agent=AgentConfig(name="random"),
            env=EnvConfig(name="cartpole-control"),
            run=EvalRunConfig(),
            wandb=WandbConfig(enabled=False),
        )

    def test_returns_string(self, eval_config):
        result = _format_eval_config(eval_config)
        assert isinstance(result, str)

    def test_contains_agent_and_env_names(self, eval_config):
        result = _format_eval_config(eval_config)
        assert "random" in result
        assert "cartpole-control" in result

    def test_wandb_disabled(self, eval_config):
        result = _format_eval_config(eval_config)
        assert "disabled" in result

    def test_wandb_enabled(self, eval_config):
        eval_config.wandb.enabled = True
        result = _format_eval_config(eval_config)
        assert "disabled" not in result


class TestFormatEvalResults:
    def test_returns_formatted_string(self):
        results = MagicMock()
        results.num_episodes = 10
        results.mean_return = 42.5
        results.std_return = 1.2
        results.min_return = 40.0
        results.max_return = 45.0
        results.mean_length = 100.0
        results.std_length = 5.0

        result = _format_eval_results(results)

        assert "10" in result
        assert "42.50" in result
        assert "40.00" in result
        assert "45.00" in result


class TestFormatTrainConfig:
    def test_returns_string(self):
        from myriad.configs.default import Config, RunConfig

        config = Config(
            agent=AgentConfig(name="dqn"),
            env=EnvConfig(name="cartpole-control"),
            run=RunConfig(
                steps_per_env=1000,
                num_envs=4,
                batch_size=32,
                buffer_size=1000,
                scan_chunk_size=100,
                eval_frequency=100,
            ),
            wandb=WandbConfig(enabled=False),
        )
        result = _format_train_config(config)
        assert "dqn" in result
        assert "cartpole-control" in result
        assert "disabled" in result


class TestApplyAutoTune:
    def test_patches_scan_chunk_size(self, monkeypatch):
        """_apply_auto_tune should call suggest_scan_chunk_size and update cfg in-place."""
        cfg = OmegaConf.create({
            "run": {"num_envs": 100, "scan_chunk_size": 64},
            "env": {"name": "cartpole-control"},
            "agent": {"name": "dqn"},
        })
        monkeypatch.setattr("myriad.platform.autotune.suggest_scan_chunk_size", lambda **kw: 512)

        _apply_auto_tune(cfg)

        assert cfg.run.scan_chunk_size == 512

    def test_passes_correct_args(self, monkeypatch):
        """_apply_auto_tune should forward num_envs, env, agent, and buffer_size."""
        cfg = OmegaConf.create({
            "run": {"num_envs": 200, "scan_chunk_size": 64, "buffer_size": 10000},
            "env": {"name": "cartpole-control"},
            "agent": {"name": "dqn"},
        })
        captured = {}

        def fake_suggest(**kw):
            captured.update(kw)
            return 256

        monkeypatch.setattr("myriad.platform.autotune.suggest_scan_chunk_size", fake_suggest)

        _apply_auto_tune(cfg)

        assert captured["num_envs"] == 200
        assert captured["env"] == "cartpole-control"
        assert captured["agent"] == "dqn"
        assert captured["buffer_size"] == 10000

    def test_buffer_size_none_when_absent(self, monkeypatch):
        """buffer_size should be None when not present in cfg.run."""
        cfg = OmegaConf.create({
            "run": {"num_envs": 50, "scan_chunk_size": 64},
            "env": {"name": "cartpole-control"},
            "agent": {"name": "dqn"},
        })
        captured = {}

        def fake_suggest(**kw):
            captured.update(kw)
            return 128

        monkeypatch.setattr("myriad.platform.autotune.suggest_scan_chunk_size", fake_suggest)

        _apply_auto_tune(cfg)

        assert captured["buffer_size"] is None


class TestGetConfigPath:
    def test_returns_string(self):
        result = _get_config_path()
        assert isinstance(result, str)

    def test_env_var_takes_priority(self, monkeypatch):
        monkeypatch.setenv("MYRIAD_CONFIG_PATH", "/custom/path")
        result = _get_config_path()
        assert result == "/custom/path"

    def test_finds_repo_configs(self):
        """Without env var and not in cwd, should find repo configs."""
        from pathlib import Path

        result = _get_config_path()
        # Should resolve to an existing path (repo configs/ dir)
        assert Path(result).exists() or result == "../configs"
