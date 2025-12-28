"""Training script for W&B sweeps.

This script is designed to work with W&B sweeps for hyperparameter optimization.
It loads the base Hydra configuration and then overrides parameters from W&B.
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from myriad.configs.default import Config
from myriad.platform.hydra_setup import setup_hydra
from myriad.platform.runner import train_and_evaluate

# Suppress excessive JAX logging when running on CPU
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for sweep training.

    This function:
    1. Initializes a W&B run (which pulls sweep parameters)
    2. Overrides Hydra config with sweep parameters
    3. Runs training with the combined configuration
    """
    # Initialize W&B run - this will pull parameters from the sweep
    wandb.init()

    # Update Hydra config with W&B sweep parameters
    # W&B config keys use dots (e.g., "agent.learning_rate")
    # We need to set them in the nested DictConfig structure
    for key, value in wandb.config.items():
        if "." in key:
            # Handle nested keys like "agent.learning_rate"
            parts = key.split(".")
            config_part = cfg
            for part in parts[:-1]:
                if part not in config_part:
                    config_part[part] = {}
                config_part = config_part[part]
            config_part[parts[-1]] = value
        else:
            cfg[key] = value

    # Also update the wandb section of config to ensure W&B integration works
    cfg.wandb.enabled = True
    cfg.wandb.mode = wandb.config.get("wandb.mode", "online")

    # Convert the Hydra configuration into a Pydantic configuration
    config_dict = OmegaConf.to_object(cfg)
    config: Config = Config(**config_dict)  # type: ignore

    print("--- Running sweep with the following configuration ---")
    print(config)
    print("------------------------------------------------------")

    # Call the runner with the configuration
    train_and_evaluate(config)

    # Finish the W&B run
    wandb.finish()


if __name__ == "__main__":
    setup_hydra()
    main()
