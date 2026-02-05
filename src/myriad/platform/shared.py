import chex
from flax import struct

from myriad.envs.environment import EnvironmentState


# TODO: can we not move this to types.py? Is there really going to be a circular import if we do so?
# Seems odd to have a separate file for this, but fine if we need it to avoid circular dependencies.
@struct.dataclass
class TrainingEnvState:
    """Container for the state of a training environment, including observations."""

    env_state: EnvironmentState
    obs: chex.Array
