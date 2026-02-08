"""Shared initialization utilities for environment and agent setup.

This module provides common initialization functions used by both training and evaluation.
"""

from __future__ import annotations

from myriad.agents import make_agent
from myriad.agents.agent import Agent
from myriad.configs.default import Config, EvalConfig
from myriad.core.spaces import Space
from myriad.core.types import BaseModel
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
    env = make_env(config.env.name, **env_kwargs)

    # Create the agent
    agent_kwargs = get_factory_kwargs(config.agent)
    action_space = env.get_action_space(env.config)
    agent = make_agent(config.agent.name, action_space=action_space, **agent_kwargs)

    return env, agent, action_space


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
    # batch_size is a platform parameter used for replay buffer sampling,
    # not an agent construction parameter
    kwargs.pop("batch_size", None)
    return kwargs
