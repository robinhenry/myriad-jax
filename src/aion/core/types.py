from typing import NamedTuple

import chex
from pydantic import BaseModel as _BaseModel


class Transition(NamedTuple):
    obs: chex.ArrayTree
    action: chex.ArrayTree
    reward: chex.ArrayTree
    next_obs: chex.ArrayTree
    done: chex.ArrayTree


class BaseModel(_BaseModel):
    """Pydantic BaseModel subclass to be used throughout the codebase."""

    class Config:
        arbitrary_types_allowed = True

    ...
