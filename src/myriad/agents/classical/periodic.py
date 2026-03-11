"""Periodic ON/OFF agent — convenience wrapper around the open-loop signal agent.

Alternates between action 1 (ON) and action 0 (OFF) with a fixed period.
The schedule is ``[1]*pulse_width + [0]*pulse_width`` and wraps indefinitely.

For more general input signals (step responses, PRBS, multi-sine) use
:mod:`myriad.agents.classical.open_loop` directly.

Registered as ``"periodic"`` in the agent registry.  Accepts ``pulse_width``
as a plain integer kwarg, making it usable from YAML configs and
:func:`~myriad.configs.builder.create_eval_config`.
"""

import jax.numpy as jnp

from myriad.core.spaces import Space
from myriad.core.types import Observation

from ..agent import Agent
from .open_loop import AgentParams, AgentState, make_agent as _make_open_loop_agent


def make_agent(action_space: Space, pulse_width: int = 24, **_kwargs) -> Agent[AgentState, AgentParams, Observation]:
    """Create a periodic (ON/OFF) open-loop agent.

    Args:
        action_space: Action space (stored for protocol compliance; not used at runtime).
        pulse_width: Number of steps per ON phase (equals the OFF phase duration).
            At 5 min/step: ``pulse_width=24`` → 2-hour ON / 2-hour OFF.

    Returns:
        Agent that emits 1 for ``pulse_width`` steps then 0 for ``pulse_width``
        steps, repeating indefinitely.

    Raises:
        ValueError: If ``pulse_width`` is less than 1.
    """
    if pulse_width < 1:
        raise ValueError(f"pulse_width must be >= 1, got {pulse_width}")
    schedule = jnp.array([1] * pulse_width + [0] * pulse_width, dtype=jnp.int32)
    return _make_open_loop_agent(action_space, schedule)
