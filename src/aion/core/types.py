from typing import Any, NamedTuple, Protocol

import chex


class Transition(NamedTuple):
    obs: chex.ArrayTree
    action: chex.ArrayTree
    reward: chex.ArrayTree
    next_obs: chex.ArrayTree
    done: chex.ArrayTree


class TrainingEnvState(Protocol):
    """
    A minimal Protocol to describe the environment state used by the runner.

    It documents that env_states has an `obs` attribute and a `replace` method
    (the latter is provided by common JAX-friendly state containers such as
    flax.struct.dataclass). This is a typing-only helper and doesn't affect
    runtime behavior.
    """

    obs: chex.Array

    def replace(self, **kwargs: Any) -> "TrainingEnvState":
        ...
