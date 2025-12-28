"""Evaluation-only functionality for trained and non-learning agents.

This module provides evaluation capabilities for:
- Non-learning controllers (random, bang-bang, PID)
- Pre-trained models
- Baseline comparisons
- Benchmarking and validation
"""

from __future__ import annotations

import jax
import numpy as np

from myriad.agents.agent import AgentState
from myriad.configs.default import Config, EvalConfig

from .episode_manager import save_episodes_to_disk
from .logging_utils import maybe_init_wandb, wandb
from .metrics_logger import MetricsLogger
from .types import EvaluationResults


def _get_eval_settings(config: Config | EvalConfig) -> tuple[int, int, int]:
    """Extract evaluation settings from either Config or EvalConfig.

    Args:
        config: Either a training Config or an EvalConfig

    Returns:
        Tuple of (seed, eval_rollouts, eval_max_steps)
    """
    if isinstance(config, EvalConfig):
        return config.run.seed, config.run.eval_rollouts, config.run.eval_max_steps
    else:
        return config.run.seed, config.run.eval_rollouts, config.run.eval_max_steps


def evaluate(
    config: Config | EvalConfig,
    agent_state: AgentState | None = None,
    return_episodes: bool = False,
) -> EvaluationResults:
    """
    Evaluation-only entry point (no training).

    Useful for:
    - Non-learning controllers (random, bang-bang, PID)
    - Pre-trained models
    - Baseline comparisons
    - Benchmarking and validation

    Args:
        config: Configuration specifying environment, agent, and evaluation parameters.
            Can be either a full Config (for training) or EvalConfig (evaluation-only).
        agent_state: Optional pre-initialized agent state. If None, agent will be initialized
            with random weights using config.seed (or config.run.seed for Config).
        return_episodes: If True, return full episode trajectories in addition to metrics.
            This includes observations, actions, rewards, and dones for each step.

    Returns:
        EvaluationResults containing:
        - Summary statistics (mean_return, std_return, min, max)
        - Raw episode data (episode_returns, episode_lengths)
        - Optional trajectory data (if return_episodes=True)
        - Metadata (num_episodes, seed)
    """
    from .initialization import initialize_environment_and_agent
    from .step_functions import make_eval_rollout_fn

    # Handle both EvalConfig and Config for wandb
    wandb_config = config.wandb
    wandb_run = (
        maybe_init_wandb(config)
        if isinstance(config, Config)
        else (maybe_init_wandb(config) if wandb_config and wandb_config.enabled else None)
    )

    try:
        # Extract evaluation settings (works for both Config and EvalConfig)
        seed, eval_rollouts, eval_max_steps = _get_eval_settings(config)

        # Initialize RNG
        key = jax.random.PRNGKey(seed)
        key, env_key, agent_key = jax.random.split(key, 3)

        # Create environment and agent using shared initialization
        env, agent, _ = initialize_environment_and_agent(config)

        # Initialize agent state if not provided
        if agent_state is None:
            # Get a sample observation to initialize the agent
            # Use original NamedTuple observation (not converted array) to allow field introspection
            obs, _ = env.reset(env_key, env.params, env.config)
            agent_state = agent.init(agent_key, obs, agent.params)

        # Create and run evaluation rollout
        eval_rollout_fn = make_eval_rollout_fn(agent, env, eval_rollouts, eval_max_steps)
        key, eval_key = jax.random.split(key)
        eval_key, eval_results_jax = eval_rollout_fn(eval_key, agent_state, return_episodes=return_episodes)

        # Convert results from device to host
        episode_returns = jax.device_get(eval_results_jax["episode_return"])
        episode_lengths = jax.device_get(eval_results_jax["episode_length"])

        # Convert episodes if present
        episodes_data = None
        if return_episodes and "episodes" in eval_results_jax:
            episodes_data = {k: jax.device_get(v) for k, v in eval_results_jax["episodes"].items()}

        # Compute summary statistics
        mean_return = float(np.mean(episode_returns))
        std_return = float(np.std(episode_returns))
        min_return = float(np.min(episode_returns))
        max_return = float(np.max(episode_returns))

        mean_length = float(np.mean(episode_lengths))
        std_length = float(np.std(episode_lengths))
        min_length = int(np.min(episode_lengths))
        max_length = int(np.max(episode_lengths))

        # Save episodes to disk if configured (for eval-only runs)
        if isinstance(config, EvalConfig) and config.run.eval_episode_save_frequency > 0:
            # Need to get episodes if not already retrieved
            if episodes_data is None:
                # Re-run with return_episodes=True
                eval_key, eval_results_jax = eval_rollout_fn(eval_key, agent_state, return_episodes=True)
                episodes_data = {k: jax.device_get(v) for k, v in eval_results_jax["episodes"].items()}

            # Prepare episode data for saving
            eval_results_with_episodes = {
                "episodes": episodes_data,
                "episode_length": episode_lengths,
                "episode_return": episode_returns,
            }
            save_count = config.run.eval_episode_save_count or eval_rollouts
            episode_dir = save_episodes_to_disk(
                eval_results_with_episodes, global_step=0, save_count=save_count, config=config
            )

            # Log episodes to wandb if enabled
            if episode_dir is not None and wandb_run is not None:
                metrics_logger = MetricsLogger(wandb_run=wandb_run)
                metrics_logger.log_episodes(episode_dir, global_step=0)

        # Create results object
        results = EvaluationResults(
            mean_return=mean_return,
            std_return=std_return,
            min_return=min_return,
            max_return=max_return,
            mean_length=mean_length,
            std_length=std_length,
            min_length=min_length,
            max_length=max_length,
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
            num_episodes=eval_rollouts,
            seed=seed,
            episodes=episodes_data,
        )

        # Log to wandb if enabled
        if wandb_run is not None:
            metrics_logger = MetricsLogger(wandb_run=wandb_run)
            # Convert to dict format expected by logger
            eval_results_dict = {
                "episode_return": episode_returns,
                "episode_length": episode_lengths,
                "dones": jax.device_get(eval_results_jax.get("dones", np.ones(eval_rollouts, dtype=bool))),
            }
            metrics_logger.log_evaluation(global_step=0, steps_per_env=0, eval_results=eval_results_dict)
            metrics_logger.log_final(0)

        return results

    finally:
        if wandb_run is not None and wandb is not None:
            wandb.finish()
