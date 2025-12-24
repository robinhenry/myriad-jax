from typing import NamedTuple

import chex
from pydantic import BaseModel as _BaseModel, ConfigDict


class Transition(NamedTuple):
    obs: chex.ArrayTree
    action: chex.ArrayTree
    reward: chex.ArrayTree
    next_obs: chex.ArrayTree
    done: chex.ArrayTree


class BaseModel(_BaseModel):
    """Pydantic BaseModel subclass to be used throughout the codebase."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ...
