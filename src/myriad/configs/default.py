"""
Defines the schema for training run configurations using hierarchical Pydantic models.

Default values follow the project philosophy:
- Agent/Env defaults: In factory functions (complex object construction)
- Run/Wandb defaults: In Pydantic models (pure config validation)
- YAML files: Specify experiment-specific overrides only
"""

import warnings

from pydantic import PositiveInt, model_validator

from myriad.core.types import BaseModel


class WandbConfig(BaseModel):
    """Schema for Weights & Biases tracking configuration.

    Default values:
    - Common settings have sensible defaults (enabled, project, mode, etc.)
    - User/experiment-specific settings default to None (entity, group, run_name)
    - Override in experiment configs as needed
    """

    enabled: bool = True  # W&B enabled by default
    project: str = "myriad"  # default project name
    entity: str | None = None  # user or team name (user-specific)
    group: str | None = None  # experiment group (experiment-specific)
    job_type: str = "train"  # default job type
    run_name: str | None = None  # auto-generated if None
    mode: str = "offline"  # offline by default (sync manually)
    dir: str | None = None  # use W&B default directory
    tags: tuple[str, ...] = ()  # default to no tags


class AgentConfig(BaseModel):
    """Schema for the Agent's configuration.

    This config allows extra fields to support agent-specific parameters.
    For example: PID controller needs kp/ki/kd, DQN needs epsilon, etc.
    """

    model_config = {"extra": "allow"}

    name: str
    batch_size: PositiveInt | None = None  # for off-policy agents: size of batches sampled from replay buffer


class EnvConfig(BaseModel):
    """Schema for the Environment's configuration.

    This config allows extra fields to support environment-specific parameters.
    For example: different environments may need custom physics parameters, rendering options, etc.
    """

    model_config = {"extra": "allow"}

    name: str


class EvalConfigBase(BaseModel):
    """Base configuration for evaluation parameters.

    Shared by both RunConfig (training with eval) and EvalConfig (eval-only).
    This eliminates duplication while maintaining clear separation of concerns.

    Design:
    - RunConfig extends this for training runs (adds training-specific params)
    - EvalConfig extends this for eval-only runs (adds env/agent/wandb)
    """

    # --- Core Settings ---
    seed: int = 42  # standard random seed

    # --- Evaluation Parameters ---
    eval_rollouts: PositiveInt = 10  # number of episodes to average over
    eval_max_steps: PositiveInt  # REQUIRED: environment-specific, max steps per episode

    # --- Episode Saving (optional) ---
    # Episodes saved to: outputs/DATE/TIME/episodes/ (hard-coded, relative to Hydra run dir)
    eval_episode_save_frequency: int = 0  # 0 = disabled, >0 = save every N steps
    eval_episode_save_count: int | None = None  # None = save all rollouts

    # --- Video Rendering (optional) ---
    # Videos rendered from saved episodes to: outputs/DATE/TIME/videos/ (relative to Hydra run dir)
    eval_render_videos: bool = False  # Enable automatic video rendering after evaluation
    eval_video_fps: int = 50  # Frames per second for rendered videos


class RunConfig(EvalConfigBase):
    """Configuration for training runs (extends EvalConfigBase).

    Inherits evaluation parameters from EvalConfigBase and adds training-specific settings.

    Note on terminology (diverges from standard RL convention):
    - `steps_per_env`: Number of steps each environment will take (primary parameter).
      This aligns with the "digital twin" / parallel experiments mental model where
      you think "run each experiment for N steps" rather than "total budget of N steps".
    - `total_timesteps`: Total environment interactions across all envs (computed property).
      This is the standard RL metric for sample efficiency comparisons in papers.

    Default values:
    - Inherited: seed, eval_rollouts, eval_episode_save_* (from EvalConfigBase)
    - Sensible defaults for common training params (num_envs, frequencies, etc.)
    - Experiment-specific parameters remain required (steps_per_env, eval_max_steps)
    - Agent-specific parameters remain optional (buffer_size, rollout_steps)
    """

    # --- Training Settings ---
    steps_per_env: PositiveInt  # REQUIRED: experiment-specific, defines run length
    num_envs: PositiveInt = 1  # default to single environment
    scan_chunk_size: PositiveInt = 256  # reasonable default for most cases
    buffer_size: PositiveInt | None = None  # for off-policy algorithms: replay buffer capacity
    rollout_steps: PositiveInt | None = None  # for on-policy algorithms: steps per env before training

    # --- Evaluation (training-specific) ---
    eval_frequency: PositiveInt = 1000  # evaluate every 1k steps per env

    # --- Logging ---
    log_frequency: PositiveInt = 1000  # log every 1k steps per env

    @property
    def total_timesteps(self) -> int:
        """Total environment interactions across all environments (for RL sample efficiency comparisons).

        This is a derived quantity computed as steps_per_env * num_envs.
        It represents the total number of environment interactions, which is the standard
        metric reported in RL papers for sample efficiency comparisons.
        """
        return self.steps_per_env * self.num_envs

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
                f"See src/myriad/platform/runners.py for details on the chunking strategy.",
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
    wandb: WandbConfig | None = None  # None = disabled (provide WandbConfig to enable)


class EvalRunConfig(EvalConfigBase):
    """Evaluation run settings (extends EvalConfigBase).

    This is the eval-only equivalent of RunConfig. It contains only evaluation
    parameters (inherited from EvalConfigBase), with no training-specific settings.

    Used within EvalConfig to maintain consistent structure with Config.
    """

    pass  # Inherits all eval params from EvalConfigBase


class EvalConfig(BaseModel):
    """Schema for evaluation-only runs.

    Matches the structure of Config (training) for consistency:
    - Config has: run, agent, env, wandb
    - EvalConfig has: run, agent, env, wandb (same structure!)

    The key difference: EvalConfig.run is EvalRunConfig (eval params only),
    while Config.run is RunConfig (training + eval params).

    Use this with the `evaluate()` function or `scripts/evaluate.py` for:
    - Classical controllers (random, bang-bang, PID)
    - Pre-trained models
    - Baseline comparisons
    - Debugging and visualization
    """

    # --- Component Configs ---
    run: EvalRunConfig  # Nested run config (matches Config structure)
    agent: AgentConfig
    env: EnvConfig

    # --- Logging (optional) ---
    wandb: WandbConfig | None = None  # None = disabled (provide WandbConfig to enable)


def config_to_eval_config(config: Config) -> EvalConfig:
    """Extract evaluation configuration from a training Config.

    Args:
        config: A training Config

    Returns:
        EvalConfig with evaluation-relevant fields extracted
    """
    return EvalConfig(
        run=EvalRunConfig(
            seed=config.run.seed,
            eval_rollouts=config.run.eval_rollouts,
            eval_max_steps=config.run.eval_max_steps,
            eval_episode_save_frequency=config.run.eval_episode_save_frequency,
            eval_episode_save_count=config.run.eval_episode_save_count,
            eval_render_videos=config.run.eval_render_videos,
            eval_video_fps=config.run.eval_video_fps,
        ),
        agent=config.agent,
        env=config.env,
        wandb=config.wandb,
    )
