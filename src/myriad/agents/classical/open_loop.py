"""Open-loop signal agent: replays a fixed action schedule.

The agent steps through a pre-defined action array, wrapping modularly when
the episode is longer than the schedule. This makes it suitable for:

- **Periodic stimuli** — pass one full period; it repeats automatically.
- **Step responses** — OFF for T steps, then ON.
- **PRBS / multi-sine** — any deterministic input signal used in SysID.

Because the action depends only on the internal step counter (not on observations),
this is a pure open-loop controller. The step counter lives in AgentState so that
parallel environments each maintain their own independent position in the schedule.

Programmatic use (not wired to the config system since ``schedule`` is a JAX array):

.. code-block:: python

    from myriad.agents.classical.open_loop import make_agent
    import jax.numpy as jnp

    # Periodic ON/OFF — pass one period; wraps automatically
    schedule = jnp.array([1]*24 + [0]*24, dtype=jnp.int32)
    agent = make_agent(env.get_action_space(env.config), schedule)

    # Step response — OFF for 1 h, then ON for the rest of the episode
    schedule = jnp.array([0]*12 + [1]*276, dtype=jnp.int32)
    agent = make_agent(env.get_action_space(env.config), schedule)
"""

from typing import Any, Tuple

import jax.numpy as jnp
from flax import struct
from jax import Array

from myriad.core.spaces import Space
from myriad.core.types import Observation, PRNGKey

from ..agent import Agent


@struct.dataclass
class AgentParams:
    """Static parameters for the open-loop agent."""

    action_space: Space  # Required by AgentParams protocol
    schedule: Array  # Shape (T,); wraps modularly if episode length > T


@struct.dataclass
class AgentState:
    """Mutable state: step counter tracking position in the schedule."""

    step: Array


def _init(key: PRNGKey, sample_obs: Observation, params: AgentParams) -> AgentState:
    return AgentState(step=jnp.array(0, dtype=jnp.int32))


def _select_action(
    key: PRNGKey,
    obs: Observation,
    state: AgentState,
    params: AgentParams,
    deterministic: bool,
) -> Tuple[Array, AgentState]:
    action = params.schedule[state.step % params.schedule.shape[0]]
    return action, AgentState(step=state.step + jnp.array(1, dtype=state.step.dtype))


def _update(key: PRNGKey, state: AgentState, batch: Any, params: AgentParams) -> Tuple[AgentState, dict]:
    return state, {}


def make_agent(action_space: Space, schedule: Array) -> Agent[AgentState, AgentParams, Observation]:
    """Create an open-loop agent that replays a fixed action schedule.

    Args:
        action_space: Action space (stored for protocol compliance; not used at runtime).
        schedule: 1D integer array of actions. Wraps modularly if the episode
            is longer than the schedule — passing one period of a periodic
            signal is sufficient.

    Returns:
        Agent that returns ``schedule[step % len(schedule)]`` at each call.
    """
    params = AgentParams(action_space=action_space, schedule=schedule)
    return Agent(params=params, init=_init, select_action=_select_action, update=_update)
