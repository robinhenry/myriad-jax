"""Test that all configuration files can be loaded and validated.

This test ensures:
1. All base configs (agent/, env/, run/) are valid
2. All experiment configs can be composed and loaded
3. Hydra composition works correctly
4. Pydantic validation passes
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from myriad.configs.default import Config


@pytest.fixture(scope="module")
def config_dir():
    """Get absolute path to configs directory."""
    repo_root = Path(__file__).parent.parent.parent
    return str(repo_root / "configs")


@pytest.fixture(scope="module")
def hydra_context(config_dir):
    """Initialize Hydra context once for all tests."""
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        yield


class TestBaseConfigs:
    """Test that base configs can be loaded."""

    def test_default_config(self, hydra_context):
        """Test default config.yaml loads and validates."""
        cfg = compose(config_name="config")

        # Convert to Pydantic for validation
        config_dict = OmegaConf.to_object(cfg)
        config = Config(**config_dict)

        # Basic sanity checks (now uses pqn_cartpole_default)
        assert config.agent.name == "pqn"
        assert config.env.name == "cartpole-control"
        assert config.wandb.enabled is True

    def test_pqn_config(self, hydra_context):
        """Test pqn_cartpole_default.yaml loads and validates."""
        cfg = compose(config_name="experiments/pqn_cartpole_default")

        config_dict = OmegaConf.to_object(cfg)
        config = Config(**config_dict)

        assert config.agent.name == "pqn"
        assert config.env.name == "cartpole-control"


class TestAgentConfigs:
    """Test that all agent configs can be loaded."""

    @pytest.mark.parametrize("agent_name", ["dqn", "pqn", "random", "bangbang"])
    def test_agent_config_loads(self, hydra_context, agent_name):
        """Test that agent config can be loaded."""
        cfg = compose(config_name="config", overrides=[f"agent={agent_name}"])

        config_dict = OmegaConf.to_object(cfg)
        config = Config(**config_dict)

        assert config.agent.name == agent_name


class TestEnvConfigs:
    """Test that all environment configs can be loaded."""

    @pytest.mark.parametrize(
        "env_config,expected_name",
        [
            ("cartpole_control", "cartpole-control"),
            ("ccasr_gfp_control", "ccasr-gfp-control"),
        ],
    )
    def test_env_config_loads(self, hydra_context, env_config, expected_name):
        """Test that environment config can be loaded."""
        cfg = compose(config_name="config", overrides=[f"env={env_config}"])

        config_dict = OmegaConf.to_object(cfg)
        config = Config(**config_dict)

        assert config.env.name == expected_name


class TestExperimentConfigs:
    """Test that experiment configs can be loaded."""

    @pytest.mark.parametrize(
        "experiment_name",
        [
            "experiments/ccasr_gfp_sinewave_tracking",
        ],
    )
    def test_experiment_config_loads(self, hydra_context, experiment_name):
        """Test that experiment config can be loaded and has overrides."""
        cfg = compose(config_name=experiment_name)

        config_dict = OmegaConf.to_object(cfg)
        config = Config(**config_dict)

        # Just verify it loads and validates
        assert config.agent.name is not None
        assert config.env.name is not None


class TestCLIOverrides:
    """Test that CLI overrides work correctly."""

    def test_agent_parameter_override(self, hydra_context):
        """Test overriding agent parameters via CLI."""
        cfg = compose(config_name="config", overrides=["+agent.learning_rate=3e-4"])  # Use + prefix to add field

        config_dict = OmegaConf.to_object(cfg)
        config = Config(**config_dict)

        # Check override was applied
        # Note: Extra fields are allowed in AgentConfig
        assert config.agent.model_dump().get("learning_rate") == pytest.approx(3e-4)

    def test_env_parameter_override(self, hydra_context):
        """Test overriding nested environment parameters via CLI."""
        cfg = compose(
            config_name="config", overrides=["+env.physics.pole_mass=0.2"]  # Use + prefix to add nested field
        )

        config_dict = OmegaConf.to_object(cfg)
        config = Config(**config_dict)

        # Check override was applied
        # Note: Extra fields are allowed in EnvConfig
        env_dict = config.env.model_dump()
        assert "physics" in env_dict
        assert env_dict["physics"]["pole_mass"] == 0.2

    def test_run_parameter_override(self, hydra_context):
        """Test overriding run parameters via CLI."""
        cfg = compose(config_name="config", overrides=["run.num_envs=10000"])

        config_dict = OmegaConf.to_object(cfg)
        config = Config(**config_dict)

        assert config.run.num_envs == 10000


class TestConfigValidation:
    """Test Pydantic validation catches errors."""

    def test_missing_required_field_fails(self, hydra_context):
        """Test that missing required fields cause validation errors."""
        # This should work (has all required fields)
        cfg = compose(config_name="config")
        config_dict = OmegaConf.to_object(cfg)
        Config(**config_dict)  # Should not raise

        # Manually break the config by removing required field
        config_dict["run"].pop("steps_per_env")

        # Should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            Config(**config_dict)

    def test_invalid_type_fails(self, hydra_context):
        """Test that invalid types cause validation errors."""
        cfg = compose(config_name="config", overrides=["run.num_envs=not_a_number"])
        config_dict = OmegaConf.to_object(cfg)

        # Should raise validation error for non-integer num_envs
        with pytest.raises(Exception):
            Config(**config_dict)
