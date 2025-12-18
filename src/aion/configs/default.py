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
    batch_size: PositiveInt | None = None  # for off-policy agents: size of batches sampled from replay buffer


class EnvConfig(BaseModel):
    """Schema for the Environment's configuration."""

    name: str


class RunConfig(BaseModel):

    # --- Run Settings ---
    seed: int
    total_timesteps: PositiveInt  # total number of timesteps to train for across all envs
    num_envs: PositiveInt  # number of envs to run in parallel, each doing `total_timesteps // num_envs` steps

    # --- Training Settings ---
    scan_chunk_size: PositiveInt  # number of steps batched together and executed using `jax.lax.scan`
    buffer_size: PositiveInt | None = None  # for off-policy algorithms: replay buffer capacity
    rollout_steps: PositiveInt | None = None  # for on-policy algorithms: steps per env before training

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
