import hydra
from omegaconf import DictConfig

# Import your schema and the runner
from aion.configs.default import Config
from aion.platform.runner import train_and_evaluate


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
    # Use hydra.utils.instantiate to directly convert the composed
    # config (cfg) into our typed, structured Config object.
    # Hydra will recursively build AgentConfig and EnvConfig as well.
    final_config: Config = hydra.utils.instantiate(cfg)

    print("--- Running with the following configuration ---")
    print(final_config)
    print("------------------------------------------------")

    # Call your existing runner with the fully-typed and populated config object
    train_and_evaluate(final_config)


if __name__ == "__main__":
    main()
