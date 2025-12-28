"""Config builder utilities for programmatic use.

This module provides high-level functions to create training and evaluation
configs without requiring detailed knowledge of Pydantic models.
"""

from typing import Any

from myriad.configs.default import AgentConfig, Config, EnvConfig, EvalConfig, EvalRunConfig, RunConfig, WandbConfig

# Default eval_max_steps for each environment
# These match the typical episode lengths for each environment
_ENV_DEFAULTS = {
    "cartpole-control": {"eval_max_steps": 500},
    "cartpole-sysid": {"eval_max_steps": 500},
    "ccas-ccar-control": {"eval_max_steps": 1000},
    "ccas-ccar-sysid": {"eval_max_steps": 1000},
}

# Off-policy agents that require a replay buffer
_OFF_POLICY_AGENTS = {"dqn"}

# On-policy agents that use rollout_steps
_ON_POLICY_AGENTS = {"pqn", "ppo", "a2c"}


def create_config(
    env: str,
    agent: str,
    num_envs: int = 1,
    steps_per_env: int = 1000,
    eval_max_steps: int | None = None,
    eval_frequency: int = 100,
    eval_rollouts: int = 10,
    log_frequency: int = 100,
    seed: int = 42,
    wandb_enabled: bool = False,
    **kwargs: Any,
) -> Config:
    """Create a training config with sensible defaults.

    This is the recommended way to create configs programmatically.
    Provides a simpler interface than constructing nested Pydantic models.

    Args:
        env: Environment name (e.g., "cartpole-control", "ccas-ccar-control")
        agent: Agent name (e.g., "dqn", "pqn", "random")
        num_envs: Number of parallel environments to run
        steps_per_env: Number of steps to run per environment
        eval_max_steps: Maximum steps per evaluation episode.
            If None, uses environment-specific default.
        eval_frequency: Evaluate every N steps (0 to disable)
        eval_rollouts: Number of episodes to run during evaluation
        log_frequency: Log training metrics every N steps
        seed: Random seed for reproducibility
        wandb_enabled: Enable Weights & Biases logging
        **kwargs: Additional config overrides. Can specify nested parameters using
            dot notation (e.g., agent.learning_rate=1e-3) or pass dicts for
            nested configs (e.g., wandb={"project": "my-project"}).

    Returns:
        Fully configured Config object ready for train_and_evaluate()

    Example:
        >>> from myriad import create_config, train_and_evaluate
        >>> config = create_config(
        ...     env="cartpole-control",
        ...     agent="dqn",
        ...     num_envs=1000,
        ...     steps_per_env=100,
        ... )
        >>> results = train_and_evaluate(config)
    """
    # Auto-detect eval_max_steps if not provided
    if eval_max_steps is None:
        env_defaults = _ENV_DEFAULTS.get(env)
        if env_defaults and "eval_max_steps" in env_defaults:
            eval_max_steps = env_defaults["eval_max_steps"]
        else:
            # Fallback default
            eval_max_steps = 1000

    # Extract nested overrides
    env_kwargs = {}
    agent_kwargs = {}
    run_kwargs = {}
    wandb_kwargs = {}

    for key, value in kwargs.items():
        if key.startswith("env."):
            env_kwargs[key.replace("env.", "")] = value
        elif key.startswith("agent."):
            agent_kwargs[key.replace("agent.", "")] = value
        elif key.startswith("run."):
            run_kwargs[key.replace("run.", "")] = value
        elif key.startswith("wandb."):
            wandb_kwargs[key.replace("wandb.", "")] = value
        elif isinstance(value, dict):
            # Handle dict-based nested configs
            if key == "env":
                env_kwargs.update(value)
            elif key == "agent":
                agent_kwargs.update(value)
            elif key == "run":
                run_kwargs.update(value)
            elif key == "wandb":
                wandb_kwargs.update(value)
        else:
            # Try to infer which section it belongs to
            # Common agent params
            if key in ["learning_rate", "batch_size", "gamma", "epsilon"]:
                agent_kwargs[key] = value
            # Common run params (evaluation and training)
            elif key in [
                "buffer_size",  # Off-policy training parameter
                "rollout_steps",
                "scan_chunk_size",
                "eval_episode_save_frequency",
                "eval_episode_save_count",
                "eval_render_videos",
                "eval_video_fps",
            ]:
                run_kwargs[key] = value
            # Common wandb params
            elif key in ["project", "entity", "group", "tags", "run_name"]:
                wandb_kwargs[key] = value
            else:
                # Default to agent config (most flexible with extra="allow")
                agent_kwargs[key] = value

    # Auto-configure training mode based on agent type
    # Only set defaults if user hasn't explicitly provided them
    if agent in _OFF_POLICY_AGENTS:
        # Off-policy agents need buffer_size (run param) and batch_size (agent param)
        if "buffer_size" not in run_kwargs:
            run_kwargs["buffer_size"] = 10000  # Sensible default
        if "batch_size" not in agent_kwargs:
            agent_kwargs["batch_size"] = 32  # Sensible default
    elif agent in _ON_POLICY_AGENTS:
        # On-policy agents need rollout_steps
        if "rollout_steps" not in run_kwargs:
            run_kwargs["rollout_steps"] = 2  # Sensible default

    # Build configs
    env_config = EnvConfig(name=env, **env_kwargs)
    agent_config = AgentConfig(name=agent, **agent_kwargs)
    run_config = RunConfig(
        seed=seed,
        num_envs=num_envs,
        steps_per_env=steps_per_env,
        eval_max_steps=eval_max_steps,
        eval_frequency=eval_frequency,
        eval_rollouts=eval_rollouts,
        log_frequency=log_frequency,
        **run_kwargs,
    )
    wandb_config = WandbConfig(enabled=wandb_enabled, **wandb_kwargs)

    return Config(
        env=env_config,
        agent=agent_config,
        run=run_config,
        wandb=wandb_config,
    )


def create_eval_config(
    env: str,
    agent: str,
    eval_rollouts: int = 10,
    eval_max_steps: int | None = None,
    seed: int = 42,
    wandb_enabled: bool = False,
    **kwargs: Any,
) -> EvalConfig:
    """Create an evaluation-only config with sensible defaults.

    Use this for evaluating non-learning controllers (random, PID, bang-bang)
    or pre-trained models without any training.

    Args:
        env: Environment name (e.g., "cartpole-control")
        agent: Agent name (e.g., "random", "dqn")
        eval_rollouts: Number of episodes to evaluate
        eval_max_steps: Maximum steps per episode.
            If None, uses environment-specific default.
        seed: Random seed for reproducibility
        wandb_enabled: Enable Weights & Biases logging
        **kwargs: Additional config overrides (same as create_config)

    Returns:
        Fully configured EvalConfig object ready for evaluate()

    Example:
        >>> from myriad import create_eval_config, evaluate
        >>> config = create_eval_config(
        ...     env="cartpole-control",
        ...     agent="random",
        ...     eval_rollouts=100,
        ... )
        >>> results = evaluate(config)
    """
    # Auto-detect eval_max_steps if not provided
    if eval_max_steps is None:
        env_defaults = _ENV_DEFAULTS.get(env)
        if env_defaults and "eval_max_steps" in env_defaults:
            eval_max_steps = env_defaults["eval_max_steps"]
        else:
            eval_max_steps = 1000

    # Extract nested overrides (same logic as create_config)
    env_kwargs = {}
    agent_kwargs = {}
    run_kwargs = {}
    wandb_kwargs = {}

    for key, value in kwargs.items():
        if key.startswith("env."):
            env_kwargs[key.replace("env.", "")] = value
        elif key.startswith("agent."):
            agent_kwargs[key.replace("agent.", "")] = value
        elif key.startswith("run."):
            run_kwargs[key.replace("run.", "")] = value
        elif key.startswith("wandb."):
            wandb_kwargs[key.replace("wandb.", "")] = value
        elif isinstance(value, dict):
            if key == "env":
                env_kwargs.update(value)
            elif key == "agent":
                agent_kwargs.update(value)
            elif key == "run":
                run_kwargs.update(value)
            elif key == "wandb":
                wandb_kwargs.update(value)
        else:
            # Same inference logic
            if key in ["learning_rate", "batch_size", "buffer_size", "gamma", "epsilon"]:
                agent_kwargs[key] = value
            elif key in [
                "eval_episode_save_frequency",
                "eval_episode_save_count",
                "eval_render_videos",
                "eval_video_fps",
            ]:
                run_kwargs[key] = value
            elif key in ["project", "entity", "group", "tags", "run_name"]:
                wandb_kwargs[key] = value
            else:
                agent_kwargs[key] = value

    # Build configs
    env_config = EnvConfig(name=env, **env_kwargs)
    agent_config = AgentConfig(name=agent, **agent_kwargs)
    eval_run_config = EvalRunConfig(
        seed=seed,
        eval_rollouts=eval_rollouts,
        eval_max_steps=eval_max_steps,
        **run_kwargs,
    )
    wandb_config = WandbConfig(enabled=wandb_enabled, **wandb_kwargs)

    return EvalConfig(
        env=env_config,
        agent=agent_config,
        run=eval_run_config,
        wandb=wandb_config,
    )
