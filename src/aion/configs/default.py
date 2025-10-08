"""
Default configuration for training runs, using hierarchical dataclasses.
"""
from flax.struct import dataclass


@dataclass
class AgentConfig:
    """Configuration for the Agent."""

    name: str = "random_agent"
    # --- PQN Agent Params ---
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20000
    target_network_frequency: int = 500
    tau: float = 1.0


@dataclass
class EnvConfig:
    """Configuration for the Environment."""

    name: str = "toy_env_v1"


@dataclass
class Config:
    """Top-level configuration for a training run."""

    # --- Run Settings ---
    seed: int = 42
    total_timesteps: int = 50000
    num_envs: int = 16  # Number of parallel environments

    # --- Training Settings ---
    batch_size: int = 256
    buffer_size: int = 10000

    # --- Logging & Evaluation ---
    log_frequency: int = 100  # Log every 100 training steps

    # --- Component Configs ---
    agent: AgentConfig = AgentConfig()
    env: EnvConfig = EnvConfig()
