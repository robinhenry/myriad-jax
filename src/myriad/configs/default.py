"""
Defines the schema for training run configurations using hierarchical dataclasses.
The default values are managed by Hydra in the .yaml files.
"""

import warnings

from pydantic import PositiveInt, model_validator

from myriad.core.types import BaseModel


class WandbConfig(BaseModel):
    """Schema for optional Weights & Biases tracking configuration."""

    enabled: bool
    project: str | None = None
    entity: str | None = None
    group: str | None = None
    job_type: str | None = None
    run_name: str | None = None
    mode: str | None = None
    dir: str | None = None
    tags: tuple[str, ...] | None = None


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
    eval_frequency: PositiveInt  # frequency (# of steps per env) between subsequent evaluation events
    eval_rollouts: PositiveInt  # number of episodes to run during each performance evaluation (# of parallel envs)
    eval_max_steps: PositiveInt  # maximum number of steps per evaluation rollout
    eval_episode_save_frequency: int = 0  # frequency to save full episodes (0 = disabled, >0 = save every N steps)
    eval_episode_save_count: int | None = None  # number of episodes to save (None = save all eval_rollouts)
    eval_episode_save_dir: str = "episodes"  # directory for saving episode trajectories

    # --- Logging ---
    log_frequency: PositiveInt  # frequency (# of steps per env) between training metrics logging events

    @model_validator(mode="after")
    def validate_scan_chunk_size_efficiency(self) -> "RunConfig":
        """Warn if scan_chunk_size is configured inefficiently relative to logging/eval frequencies.

        When scan_chunk_size is much larger than the logging or eval frequencies, the training loop
        will frequently create partial chunks with many masked (inactive) iterations. These masked
        iterations still execute but discard their results, wasting computation.
        """
        min_boundary_frequency = min(self.log_frequency, self.eval_frequency)

        if self.scan_chunk_size > 2 * min_boundary_frequency:
            warnings.warn(
                f"Performance warning: scan_chunk_size ({self.scan_chunk_size}) is more than 2x "
                f"the minimum boundary frequency ({min_boundary_frequency}). "
                f"This may lead to wasted computation from masked iterations at logging/eval boundaries. "
                f"Consider reducing scan_chunk_size or increasing log_frequency/eval_frequency. "
                f"See src/myriad/platform/scan_utils.py for details on the chunking strategy.",
                UserWarning,
                stacklevel=2,
            )

        return self

    @model_validator(mode="after")
    def validate_on_policy_frequencies(self) -> "RunConfig":
        """Ensure rollout_steps aligns with logging/eval frequencies for on-policy algorithms.

        On-policy algorithms collect full rollouts before updating. For efficient boundary alignment,
        rollout_steps should divide evenly into the logging and evaluation frequencies.
        """
        if self.rollout_steps is not None:
            # On-policy training mode: check if frequencies are divisible by rollout_steps for clean alignment
            if self.log_frequency % self.rollout_steps != 0 or self.eval_frequency % self.rollout_steps != 0:
                warnings.warn(
                    f"On-policy training: For optimal boundary alignment, rollout_steps ({self.rollout_steps}) "
                    f"should divide evenly into log_frequency ({self.log_frequency}) and "
                    f"eval_frequency ({self.eval_frequency}). Current configuration may cause "
                    f"logging/evaluation to occur at irregular intervals.",
                    UserWarning,
                    stacklevel=2,
                )

            # Additionally warn if scan_chunk_size doesn't align with rollout_steps
            if self.rollout_steps % self.scan_chunk_size != 0 and self.scan_chunk_size % self.rollout_steps != 0:
                warnings.warn(
                    f"On-policy training: For efficient chunked execution, scan_chunk_size ({self.scan_chunk_size}) "
                    f"should divide evenly into rollout_steps ({self.rollout_steps}) or vice versa. "
                    f"This ensures minimal wasted computation from masked iterations.",
                    UserWarning,
                    stacklevel=2,
                )

        return self


class Config(BaseModel):
    """Schema for the top-level configuration of a training run."""

    # --- Component Configs ---
    run: RunConfig
    agent: AgentConfig
    env: EnvConfig
    wandb: WandbConfig
