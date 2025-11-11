"""
Defines the schema for training run configurations using hierarchical dataclasses.
The default values are managed by Hydra in the .yaml files.
"""
from flax.struct import dataclass


@dataclass
class AgentConfig:
    """Schema for the Agent's configuration."""

    name: str


@dataclass
class EnvConfig:
    """Schema for the Environment's configuration."""

    name: str


@dataclass
class Config:
    """Schema for the top-level configuration of a training run."""

    # --- Run Settings ---
    seed: int
    total_timesteps: int
    num_envs: int

    # --- Training Settings ---
    batch_size: int
    buffer_size: int

    # --- Evaluation ---
    eval_frequency: int
    eval_rollouts: int
    eval_max_steps: int

    # --- Logging ---
    log_frequency: int

    # --- Component Configs ---
    agent: AgentConfig
    env: EnvConfig
