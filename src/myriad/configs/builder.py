"""Config builder utilities for programmatic use.

This module provides high-level functions to create training and evaluation
configs without requiring detailed knowledge of Pydantic models.
"""

from typing import Any

from myriad.agents import get_agent_info
from myriad.core.types import BaseModel
from myriad.envs import EnvInfo, get_env_info

from .default import AgentConfig, Config, EnvConfig, EvalConfig, EvalRunConfig, RunConfig, WandbConfig

# Default rollout steps for on-policy agents if not specified
DEFAULT_ON_POLICY_ROLLOUT_STEPS = 2


def _distribute_kwargs(
    kwargs: dict[str, Any],
    run_cls: type[BaseModel],
) -> tuple[dict, dict, dict, dict]:
    """Distributes flattened kwargs into nested sections based on Pydantic models.

    Args:
        kwargs: The dictionary of keyword arguments to distribute.
        run_cls: The specific RunConfig class to use for inference (:class:`RunConfig` or :class:`EvalRunConfig`).

    Returns:
        A tuple of (env_kwargs, agent_kwargs, run_kwargs, wandb_kwargs).
    """
    sections: dict[str, dict[str, Any]] = {
        "env": {},
        "agent": {},
        "run": {},
        "wandb": {},
    }

    # Model classes for automated parameter inference
    inference_models: dict[str, type[BaseModel]] = {
        "run": run_cls,
        "wandb": WandbConfig,
        "agent": AgentConfig,
    }

    for key, value in kwargs.items():
        # 1. Explicit dot notation (e.g., "agent.learning_rate")
        if "." in key:
            prefix, attr = key.split(".", 1)
            if prefix in sections:
                sections[prefix][attr] = value
                continue

        # 2. Nested dicts (e.g., wandb={"project": "myriad"})
        if isinstance(value, dict) and key in sections:
            sections[key].update(value)
            continue

        # 3. Inference based on model fields
        # We check Run and Wandb first since they have fixed schemas.
        # AgentConfig has extra="allow", so we only check its explicit fields.
        found = False
        for name, cls in inference_models.items():
            if key in cls.model_fields:
                sections[name][key] = value
                found = True
                break

        if not found:
            # Default to agent config
            sections["agent"][key] = value

    return sections["env"], sections["agent"], sections["run"], sections["wandb"]


def _resolve_eval_max_steps(eval_max_steps: int | None, env_info: EnvInfo | None) -> int | None:
    """Resolves eval_max_steps from explicit > registry config_cls > model defaults."""
    if eval_max_steps is not None:
        return eval_max_steps

    if env_info:
        # Instantiate environment config with defaults to get its max_steps property
        try:
            default_env_config = env_info.config_cls()
            return getattr(default_env_config, "max_steps", None)
        except (TypeError, AttributeError):
            pass
    return None


def create_config(
    env: str,
    agent: str,
    num_envs: int = 1,
    steps_per_env: int = 1000,
    rollout_steps: int | None = None,
    eval_max_steps: int | None = None,
    eval_frequency: int = 100,
    eval_rollouts: int = 10,
    log_frequency: int = 100,
    seed: int = 42,
    wandb_enabled: bool = False,
    auto_tune: bool = False,
    **kwargs: Any,
) -> Config:
    """Create a training config with sensible defaults.

    This is the recommended way to create configs programmatically.
    It provides a simpler interface than constructing nested Pydantic models.

    Args:
        env: Environment name (e.g., "cartpole-control", "ccas-ccar-control")
        agent: Agent name (e.g., "dqn", "pqn", "random")
        num_envs: Number of parallel environments to run (ignored if ``auto_tune=True``)
        steps_per_env: Number of steps to run per environment
        rollout_steps: Number of steps to collect per environment before updating
            (for on-policy agents like PQN). If None, defaults to 2 for on-policy agents.
        eval_max_steps: Maximum steps per evaluation episode.
            If None, uses environment-specific default from registry or Config models.
        eval_frequency: Evaluate every N steps (0 to disable)
        eval_rollouts: Number of episodes to run during evaluation
        log_frequency: Log training metrics every N steps
        seed: Random seed for reproducibility
        wandb_enabled: Enable Weights & Biases logging
        auto_tune: If True, automatically find optimal ``scan_chunk_size`` for the given
            ``num_envs`` on your hardware. First run profiles your system (~30-60s),
            subsequent runs use cached values (<1s). Overrides ``scan_chunk_size`` parameter.
        **kwargs: Additional config overrides. Can specify nested parameters using
            dot notation (e.g., ``agent.learning_rate=1e-3``) or pass dicts for
            nested configs (e.g., ``wandb={"project": "my-project"}``).

    Returns:
        Fully configured Config object ready for :func:`~myriad.platform.train_and_evaluate`
    """
    # Look up agent and environment info
    agent_info = get_agent_info(agent)
    env_info = get_env_info(env)

    # Distribute nested overrides
    env_kwargs, agent_kwargs, run_kwargs, wandb_kwargs = _distribute_kwargs(kwargs, RunConfig)

    # Auto-tune if requested
    if auto_tune:
        from myriad.platform.autotune import suggest_scan_chunk_size

        # Determine buffer_size for off-policy agents
        buffer_size = kwargs.get("buffer_size") or run_kwargs.get("buffer_size")
        if buffer_size is None and agent_info and agent_info.is_off_policy:
            buffer_size = RunConfig.model_fields["buffer_size"].default

        # Run auto-tuning to find optimal scan_chunk_size for the given num_envs
        optimal_chunk_size = suggest_scan_chunk_size(
            num_envs=num_envs,
            env=env,
            agent=agent,
            buffer_size=buffer_size,
            force_revalidate=False,
            verbose=True,
        )

        # Override scan_chunk_size with auto-tuned value
        run_kwargs["scan_chunk_size"] = optimal_chunk_size

    # Auto-configure training mode based on agent type
    if rollout_steps is None:
        # Check if specified in run_kwargs via dot notation or dict
        rollout_steps = run_kwargs.get("rollout_steps")

    if rollout_steps is None and agent_info and agent_info.is_on_policy:
        rollout_steps = DEFAULT_ON_POLICY_ROLLOUT_STEPS

    # Build run config with merged params: explicit > model defaults
    run_params: dict[str, Any] = {
        "seed": seed,
        "num_envs": num_envs,
        "steps_per_env": steps_per_env,
        "rollout_steps": rollout_steps,
        "eval_frequency": eval_frequency,
        "eval_rollouts": eval_rollouts,
        "log_frequency": log_frequency,
        "eval_max_steps": _resolve_eval_max_steps(eval_max_steps, env_info),
        **run_kwargs,
    }
    # Clean up Nones so Pydantic uses its own field defaults where applicable
    run_params = {k: v for k, v in run_params.items() if v is not None}
    run_config = RunConfig(**run_params)

    # Build other configs
    wandb_params: dict[str, Any] = {"enabled": wandb_enabled, **wandb_kwargs}
    return Config(
        env=EnvConfig(name=env, **env_kwargs),
        agent=AgentConfig(name=agent, **agent_kwargs),
        run=run_config,
        wandb=WandbConfig(**wandb_params),
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
            If None, uses environment-specific default from registry or Config models.
        seed: Random seed for reproducibility
        wandb_enabled: Enable Weights & Biases logging
        **kwargs: Additional config overrides (same as create_config)

    Returns:
        Fully configured EvalConfig object ready for :func:`~myriad.platform.evaluate`
    """
    # Look up environment info
    env_info = get_env_info(env)

    # Distribute nested overrides
    env_kwargs, agent_kwargs, run_kwargs, wandb_kwargs = _distribute_kwargs(kwargs, EvalRunConfig)

    # Build run config with merged params: explicit > model defaults
    run_params: dict[str, Any] = {
        "seed": seed,
        "eval_rollouts": eval_rollouts,
        "eval_max_steps": _resolve_eval_max_steps(eval_max_steps, env_info),
        **run_kwargs,
    }
    # Clean up Nones so Pydantic uses its own field defaults where applicable
    run_params = {k: v for k, v in run_params.items() if v is not None}
    eval_run_config = EvalRunConfig(**run_params)

    # Build other configs
    wandb_params: dict[str, Any] = {"enabled": wandb_enabled, **wandb_kwargs}
    return EvalConfig(
        env=EnvConfig(name=env, **env_kwargs),
        agent=AgentConfig(name=agent, **agent_kwargs),
        run=eval_run_config,
        wandb=WandbConfig(**wandb_params),
    )
