from typing import NamedTuple

import chex


class Transition(NamedTuple):
    obs: chex.ArrayTree
    action: chex.ArrayTree
    reward: chex.ArrayTree
    next_obs: chex.ArrayTree
    done: chex.ArrayTree
