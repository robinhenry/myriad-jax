"""
Defines the schema for training run configurations using hierarchical dataclasses.
The default values are managed by Hydra in the .yaml files.
"""
from pydantic import PositiveInt

from aion.core.types import BaseModel


class WandbConfig(BaseModel):
    """Schema for optional Weights & Biases tracking configuration."""

    enabled: bool
    project: str | None
    entity: str | None
    group: str | None
    job_type: str | None
    run_name: str | None
    mode: str | None
    dir: str | None
    tags: tuple[str, ...] | None


class AgentConfig(BaseModel):
    """Schema for the Agent's configuration."""

    name: str


class EnvConfig(BaseModel):
    """Schema for the Environment's configuration."""

    name: str


class RunConfig(BaseModel):

    # --- Run Settings ---
    seed: int
    total_timesteps: PositiveInt
    num_envs: PositiveInt

    # --- Training Settings ---
    batch_size: PositiveInt
    buffer_size: PositiveInt
    scan_chunk_size: PositiveInt

    # --- Evaluation ---
    eval_frequency: PositiveInt
    eval_rollouts: PositiveInt
    eval_max_steps: PositiveInt

    # --- Logging ---
    log_frequency: PositiveInt


class Config(BaseModel):
    """Schema for the top-level configuration of a training run."""

    # --- Component Configs ---
    run: RunConfig
    agent: AgentConfig
    env: EnvConfig
    wandb: WandbConfig
