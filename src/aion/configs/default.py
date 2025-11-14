"""
Defines the schema for training run configurations using hierarchical dataclasses.
The default values are managed by Hydra in the .yaml files.
"""
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
    total_timesteps: int
    num_envs: int

    # --- Training Settings ---
    batch_size: int
    buffer_size: int
    scan_chunk_size: int

    # --- Evaluation ---
    eval_frequency: int
    eval_rollouts: int
    eval_max_steps: int

    # --- Logging ---
    log_frequency: int


class Config(BaseModel):
    """Schema for the top-level configuration of a training run."""

    # --- Component Configs ---
    run: RunConfig
    agent: AgentConfig
    env: EnvConfig
    wandb: WandbConfig
