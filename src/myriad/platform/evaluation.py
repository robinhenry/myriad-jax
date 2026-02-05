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
from .initialization import initialize_environment_and_agent
from .logging_utils import maybe_close_wandb, maybe_init_wandb
from .metrics_logger import MetricsLogger
from .step_functions import make_eval_rollout_fn
from .types import EvaluationResults


# TODO: we might want to consolidate `Config | EvalConfig` into a single type. Does it really make
# sense to have 2 different types? Perhaps there's a way to compose config types such that there's
# a sub-type that could be used as type annotation here. Or perhaps that's a bad idea if it introduces
# unnecessary complexity!
# Actually: why do we even need to support `Config`? Surely, if this is an evaluate-only function, the
# argument should be `EvalConfig`. If needed to support `Config` (eg, when a trained agent is then evaluated),
# the `Config` should be transformed into `EvalConfig` upstream in the calling function.
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

    # Initialize wandb for logging
    wandb_run = maybe_init_wandb(config)

    try:
        # Extract evaluation settings (works for both Config and EvalConfig)
        seed, eval_rollouts, eval_max_steps = config.run.seed, config.run.eval_rollouts, config.run.eval_max_steps

        # Initialize RNG
        key = jax.random.PRNGKey(seed)
        key, env_key, agent_key = jax.random.split(key, 3)

        # Create environment and agent using shared initialization
        env, agent, _ = initialize_environment_and_agent(config)

        # Initialize agent state if not provided
        if agent_state is None:
            # Get a sample observation to initialize the agent
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
        if "episodes" in eval_results_jax:
            episodes_data = {k: jax.device_get(v) for k, v in eval_results_jax["episodes"].items()}

        # Compute summary statistics
        results = EvaluationResults(
            mean_return=float(np.mean(episode_returns)),
            std_return=float(np.std(episode_returns)),
            min_return=float(np.min(episode_returns)),
            max_return=float(np.max(episode_returns)),
            mean_length=float(np.mean(episode_lengths)),
            std_length=float(np.std(episode_lengths)),
            min_length=int(np.min(episode_lengths)),
            max_length=int(np.max(episode_lengths)),
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
            num_episodes=eval_rollouts,
            seed=seed,
            episodes=episodes_data,
        )

        # TODO: why do we check the type of `Config`? HOpefully if we address the previous todo, this should
        # no longer be required.
        # TODO: if `return_episodes == False`, then we have no episodes data and `episodes_data is None`. This
        # means, with the current implementation, we won't save any episodes data to disk. I'm not sure this is
        # the behaviour we want: it might make more sense to still allow to save the episodes to disk, but not
        # keep episodes_data in memory if `return_episodes == False`? I'm not sure.
        # Save episodes to disk if configured (for eval-only runs)
        if isinstance(config, EvalConfig) and episodes_data is not None:
            # Prepare episode data for saving
            eval_results_with_episodes = {
                "episodes": episodes_data,
                "episode_length": episode_lengths,
                "episode_return": episode_returns,
            }
            ## Save all episodes if `eval_episode_save_count = None`
            save_count = config.run.eval_episode_save_count or eval_rollouts
            episode_dir = save_episodes_to_disk(
                eval_results_with_episodes, global_step=0, save_count=save_count, config=config
            )

            # Log episodes to wandb if enabled
            if episode_dir is not None and wandb_run is not None:
                metrics_logger = MetricsLogger(wandb_run=wandb_run)
                metrics_logger.log_episodes(episode_dir, steps_per_env=0)

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
        maybe_close_wandb(wandb_run)
