"""Shared initialization utilities for environment and agent setup.

This module provides common initialization functions used by both training and evaluation.
"""

import jax
import jax.numpy as jnp

from myriad.agents import make_agent
from myriad.agents.agent import Agent
from myriad.configs.default import Config, EvalConfig
from myriad.core.spaces import Space
from myriad.core.types import BaseModel, PRNGKey
from myriad.envs import make_env
from myriad.envs.environment import Environment


def initialize_environment_and_agent(
    config: Config | EvalConfig,
) -> tuple[Environment, Agent, Space]:
    """Initialize environment and agent from configuration.

    This is shared by both training and evaluation to ensure consistent setup.

    Args:
        config: Configuration specifying environment and agent parameters.
            Can be either a full Config (training) or EvalConfig (evaluation-only).

    Returns:
        Tuple of (environment, agent, action_space)
    """
    # Create the environment
    env_kwargs = get_factory_kwargs(config.env)
    frame_stack_n = env_kwargs.pop("frame_stack_n", 0)
    env = make_env(config.env.name, **env_kwargs)
    if frame_stack_n > 0:
        from myriad.envs.wrappers import make_frame_stack_env

        env = make_frame_stack_env(env, n_frames=frame_stack_n)

    # Create the agent
    agent_kwargs = get_factory_kwargs(config.agent)
    action_space = env.get_action_space(env.config)

    # Resolve epsilon_decay_fraction → epsilon_decay_steps (mirrors create_config logic)
    if "epsilon_decay_fraction" in agent_kwargs:
        fraction = agent_kwargs.pop("epsilon_decay_fraction")
        steps_per_env = getattr(config.run, "steps_per_env", None)
        if steps_per_env is not None:
            agent_kwargs["epsilon_decay_steps"] = max(1, int(fraction * steps_per_env))

    # Resolve lr_decay_fraction → lr_decay_steps
    if "lr_decay_fraction" in agent_kwargs:
        fraction = agent_kwargs.pop("lr_decay_fraction")
        steps_per_env = getattr(config.run, "steps_per_env", None)
        rollout_steps = getattr(config.run, "rollout_steps", None)
        num_minibatches = agent_kwargs.get("num_minibatches")
        num_epochs = agent_kwargs.get("num_epochs")
        if (
            steps_per_env is not None
            and rollout_steps is not None
            and num_minibatches is not None
            and num_epochs is not None
        ):
            num_updates = steps_per_env / rollout_steps
            total_grad_steps = num_updates * num_minibatches * num_epochs
            agent_kwargs["lr_decay_steps"] = max(1, int(fraction * total_grad_steps))

    # Auto-inject dt from environment config if agent doesn't have it
    if "dt" not in agent_kwargs:
        env_dt = getattr(env.config, "dt", None)
        if env_dt is not None:
            agent_kwargs["dt"] = env_dt

    agent = make_agent(config.agent.name, action_space=action_space, **agent_kwargs)

    return env, agent, action_space


def make_params_batch(env: Environment, num_envs: int, key: PRNGKey):
    """Build a (num_envs, ...) params pytree for parallel environments.

    If ``env.sample_params_fn`` is set, samples ``num_envs`` independent parameter
    sets (domain randomization). Otherwise replicates ``env.params`` identically
    across all envs (backward compatible, zero-copy via broadcast_to).

    Args:
        env: The environment (checked for sample_params_fn).
        num_envs: Number of parallel environments.
        key: RNG key used when sampling from the prior.

    Returns:
        A params pytree with a leading (num_envs,) batch dimension.
    """
    if env.sample_params_fn is not None:
        keys = jax.random.split(key, num_envs)
        return jax.vmap(env.sample_params_fn)(keys)
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(jnp.asarray(x), (num_envs, *jnp.asarray(x).shape)),
        env.params,
    )


def get_factory_kwargs(config: BaseModel) -> dict:
    """Convert a Pydantic config object to kwargs for factory functions.

    Extracts the config fields as a dictionary and removes platform-specific
    fields that shouldn't be passed to factory functions.

    Args:
        config: Pydantic config object (e.g., EnvConfig, AgentConfig)

    Returns:
        Dictionary of kwargs suitable for passing to :func:`make_env` or :func:`make_agent`
    """
    kwargs = config.model_dump()
    assert isinstance(kwargs, dict)
    kwargs.pop("name")  # The name is used for lookup, not as a parameter
    return kwargs
