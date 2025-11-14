import chex
from flax import struct

from aion.envs.environment import EnvironmentState


@struct.dataclass
class TrainingEnvState:
    """Container for the state of a training environment, including observations."""

    env_state: EnvironmentState
    obs: chex.Array
