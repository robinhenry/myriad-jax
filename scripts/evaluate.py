import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from myriad.configs.default import EvalConfig
from myriad.platform.runner import evaluate

# Suppress excessive JAX logging when running on CPU
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for evaluation-only runs (no training).

    Use this for:
    - Classical controllers (random, bang-bang, PID)
    - Pre-trained models
    - Baseline comparisons
    - Debugging and visualization

    Hydra will automatically:
    1. Find the configuration file (default: configs/config.yaml)
    2. Compose the configuration based on the `defaults` list
    3. Allow overrides from the command line
    4. Pass the final configuration as the `cfg` argument

    Examples:
        # Run an evaluation config
        python scripts/evaluate.py --config-name=experiments/eval_bangbang_cartpole

        # Override parameters
        python scripts/evaluate.py --config-name=experiments/eval_bangbang_cartpole eval_rollouts=100
    """
    # Convert the Hydra configuration into a Pydantic configuration
    config_dict = OmegaConf.to_object(cfg)
    config: EvalConfig = EvalConfig(**config_dict)  # type: ignore

    logger.info("=" * 60)
    logger.info("Running evaluation with the following configuration:")
    logger.info(str(config))
    logger.info("=" * 60)

    # Run evaluation and get results
    results = evaluate(config=config, return_episodes=False)

    # Log summary statistics
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Episodes: {results.num_episodes}")
    logger.info(f"Mean return: {results.mean_return:.2f} ± {results.std_return:.2f}")
    logger.info(f"Min return: {results.min_return:.2f}")
    logger.info(f"Max return: {results.max_return:.2f}")
    logger.info(f"Mean episode length: {results.mean_episode_length:.2f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
