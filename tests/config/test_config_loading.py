"""Test that all configuration files can be loaded and validated.

This test ensures:
1. All experiment configs can be composed and loaded
2. Hydra composition works correctly
3. Pydantic validation passes
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pydantic import ValidationError

from myriad.configs.default import Config


@pytest.fixture(scope="module")
def config_dir():
    """Absolute path to the self-contained test configs directory."""
    return str(Path(__file__).parent.parent / "configs")


@pytest.fixture(scope="module")
def hydra_context(config_dir):
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        yield


class TestBaseConfigs:
    def test_blank_config_requires_explicit_experiment(self, hydra_context):
        """Composing the blank entry-point config alone must fail Pydantic validation."""
        cfg = compose(config_name="config")
        config_dict = OmegaConf.to_object(cfg)
        with pytest.raises((ValidationError, Exception)):
            Config(**config_dict)

    def test_pqn_cartpole(self, hydra_context):
        cfg = compose(config_name="pqn_cartpole")
        config = Config(**OmegaConf.to_object(cfg))
        assert config.agent.name == "pqn"
        assert config.env.name == "cartpole-control"

    def test_ccasr_pqn(self, hydra_context):
        cfg = compose(config_name="ccasr_pqn")
        config = Config(**OmegaConf.to_object(cfg))
        assert config.agent.name == "pqn"
        assert config.env.name == "ccasr-gfp-control"


class TestAgentConfigs:
    @pytest.mark.parametrize("agent_name", ["dqn", "pqn", "random", "bangbang"])
    def test_agent_name_override(self, hydra_context, agent_name):
        """Different agent names can be set via override and pass Pydantic validation."""
        cfg = compose(
            config_name="pqn_cartpole",
            overrides=[f"agent.name={agent_name}"],
        )
        config = Config(**OmegaConf.to_object(cfg))
        assert config.agent.name == agent_name


class TestEnvConfigs:
    @pytest.mark.parametrize(
        "env_name",
        ["cartpole-control", "ccasr-gfp-control"],
    )
    def test_env_name_override(self, hydra_context, env_name):
        """Different env names can be set via override and pass Pydantic validation."""
        cfg = compose(
            config_name="pqn_cartpole",
            overrides=[f"env.name={env_name}"],
        )
        config = Config(**OmegaConf.to_object(cfg))
        assert config.env.name == env_name


class TestCLIOverrides:
    def test_agent_parameter_override(self, hydra_context):
        cfg = compose(
            config_name="pqn_cartpole",
            overrides=["+agent.learning_rate=3e-4"],
        )
        config = Config(**OmegaConf.to_object(cfg))
        assert config.agent.model_dump().get("learning_rate") == pytest.approx(3e-4)

    def test_env_parameter_override(self, hydra_context):
        cfg = compose(
            config_name="pqn_cartpole",
            overrides=["+env.physics.pole_mass=0.2"],
        )
        config = Config(**OmegaConf.to_object(cfg))
        env_dict = config.env.model_dump()
        assert env_dict.get("physics", {}).get("pole_mass") == 0.2

    def test_run_parameter_override(self, hydra_context):
        cfg = compose(
            config_name="pqn_cartpole",
            overrides=["run.num_envs=10000"],
        )
        config = Config(**OmegaConf.to_object(cfg))
        assert config.run.num_envs == 10000


class TestConfigValidation:
    def test_missing_required_field_fails(self, hydra_context):
        cfg = compose(config_name="pqn_cartpole")
        config_dict = OmegaConf.to_object(cfg)
        Config(**config_dict)  # baseline: should not raise

        config_dict["run"].pop("steps_per_env")
        with pytest.raises(Exception):
            Config(**config_dict)

    def test_invalid_type_fails(self, hydra_context):
        cfg = compose(
            config_name="pqn_cartpole",
            overrides=["run.num_envs=not_a_number"],
        )
        with pytest.raises(Exception):
            Config(**OmegaConf.to_object(cfg))
