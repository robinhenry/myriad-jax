"""Evaluation-only functionality for trained and non-learning agents.

This module provides evaluation capabilities for:
- Non-learning controllers (random, bang-bang, PID)
- Pre-trained models
- Baseline comparisons
- Benchmarking and validation
"""

from __future__ import annotations

import logging

import jax
import numpy as np

from myriad.agents.agent import AgentState
from myriad.configs.default import EvalConfig

from .initialization import initialize_environment_and_agent
from .logging import SessionLogger
from .metadata import RunMetadata
from .output_dir import get_or_create_output_dir
from .steps import make_eval_rollout_fn
from .types import EvaluationResults

logger = logging.getLogger(__name__)


def evaluate(
    config: EvalConfig,
    agent_state: AgentState | None = None,
    return_episodes: bool = False,
    save_episodes_to_disk_flag: bool | None = None,
) -> EvaluationResults:
    """
    Evaluation-only entry point (no training).

    Useful for:
    - Non-learning controllers (random, bang-bang, PID)
    - Pre-trained models
    - Baseline comparisons
    - Benchmarking and validation

    Output directory is automatically managed:
    - Under Hydra: uses current directory (Hydra-managed)
    - Otherwise: creates timestamped directory in outputs/

    Args:
        config: EvalConfig specifying environment, agent, and evaluation parameters.
            Use config_to_eval_config() to convert a training Config if needed.
        agent_state: Optional pre-initialized agent state. If None, agent will be initialized
            with random weights using config.run.seed.
        return_episodes: If True, return full episode trajectories in EvaluationResults.episodes.
            This includes observations, actions, rewards, and dones for each step.
        save_episodes_to_disk_flag: If True, save episodes to disk (respects config settings).
            If None, infers from config.run.eval_episode_save_frequency.
            Episodes can be saved to disk without keeping them in memory (return_episodes=False).

    Returns:
        EvaluationResults containing:
        - Summary statistics (mean_return, std_return, min, max)
        - Raw episode data (episode_returns, episode_lengths)
        - Optional trajectory data (if return_episodes=True)
        - Metadata (num_episodes, seed)
    """
    # Get or create output directory
    run_dir = get_or_create_output_dir(None)

    # Config will be saved by results.save() to avoid duplicate I/O
    # Create unified logger (handles W&B init/close automatically)
    session_logger = SessionLogger.for_evaluation(config, run_dir=run_dir)

    try:
        with RunMetadata(run_dir, run_type="evaluation"):
            # Extract evaluation settings
            seed, eval_rollouts, eval_max_steps = config.run.seed, config.run.eval_rollouts, config.run.eval_max_steps

            # Determine if we should save episodes to disk
            if save_episodes_to_disk_flag is None:
                save_episodes_to_disk_flag = config.run.eval_episode_save_frequency > 0

            # Collect episodes if we need them for memory return OR for disk saving
            collect_episodes = return_episodes or save_episodes_to_disk_flag

            # Initialize RNG
            key = jax.random.PRNGKey(seed)
            key, env_key, agent_key = jax.random.split(key, 3)

            # Create environment and agent using shared initialization
            env, agent, _ = initialize_environment_and_agent(config)

            # Initialize agent state if not provided
            if agent_state is None:
                obs, _ = env.reset(env_key, env.params, env.config)
                agent_state = agent.init(agent_key, obs, agent.params)

            # Create and run evaluation rollout
            eval_rollout_fn = make_eval_rollout_fn(agent, env, eval_rollouts, eval_max_steps)
            key, eval_key = jax.random.split(key)
            eval_key, eval_results_jax = eval_rollout_fn(eval_key, agent_state, return_episodes=collect_episodes)

            # Convert results from device to host
            episode_returns = jax.device_get(eval_results_jax["episode_return"])
            episode_lengths = jax.device_get(eval_results_jax["episode_length"])

            # Convert episodes if collected
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
                config=config,  # Store config for reproducibility
                episodes=episodes_data if return_episodes else None,
                agent_state=agent_state,  # Store agent state for potential checkpoint saving
                run_dir=run_dir,  # Store output directory for tests and inspection
            )

            # Log evaluation with single unified call
            # This handles: metrics capture, disk saving, W&B logging, artifact upload
            eval_results_dict = {
                "episode_return": episode_returns,
                "episode_length": episode_lengths,
                "dones": jax.device_get(eval_results_jax.get("dones", np.ones(eval_rollouts, dtype=bool))),
            }
            if episodes_data is not None:
                eval_results_dict["episodes"] = episodes_data

            save_count = config.run.eval_episode_save_count or eval_rollouts
            session_logger.log_evaluation(
                global_step=0,
                steps_per_env=0,
                eval_results=eval_results_dict,
                save_episodes=save_episodes_to_disk_flag,
                episode_save_count=save_count,
            )
            session_logger.log_final(0)

            # Save artifacts directly
            results.save(run_dir, save_checkpoint=config.run.save_agent_checkpoint)

        # Log output directory for user convenience
        logger.info(f"Artifacts saved to: {run_dir}")

        return results

    finally:
        session_logger.finalize()
