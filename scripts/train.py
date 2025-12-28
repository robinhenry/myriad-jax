import logging

import hydra
from omegaconf import DictConfig, OmegaConf

# Import your schema and the runner
from myriad.configs.default import Config
from myriad.platform.hydra_setup import setup_hydra
from myriad.platform.runner import train_and_evaluate

# Suppress excessive JAX logging when running on CPU
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training, decorated by Hydra.

    Hydra will automatically:
    1. Find the `configs/config.yaml` file.
    2. Compose the configuration based on the `defaults` list.
    3. Allow overrides from the command line.
    4. Pass the final configuration as the `cfg` argument.
    """
    # Convert the Hydra configuration into a Pydantic configuration
    config_dict = OmegaConf.to_object(cfg)
    config: Config = Config(**config_dict)  # type: ignore

    logger.info("=" * 60)
    logger.info("Running with the following configuration:")
    logger.info(str(config))
    logger.info("=" * 60)

    # Call your existing runner with the fully-typed and populated config object
    train_and_evaluate(config)


if __name__ == "__main__":
    setup_hydra()
    main()
