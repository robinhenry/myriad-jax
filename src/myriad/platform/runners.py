"""JAX chunk runners for batched training and collection.

Uses fixed-size scans with mask-aware execution to avoid recompilation when
chunk sizes vary at logging/eval boundaries.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

from myriad.agents.agent import Agent, AgentState
from myriad.core.replay_buffer import ReplayBufferState
from myriad.core.types import PRNGKey, Transition

from .steps import tree_select
from .types import TrainingEnvState

# Type alias for carry tuple (off-policy)
_CarryT = tuple[PRNGKey, AgentState, TrainingEnvState, ReplayBufferState]

# Type alias for carry tuple (on-policy)
_CarryOnPolicyT = tuple[PRNGKey, AgentState, TrainingEnvState]


def make_chunk_runner(train_step_fn: Callable, batch_size: int) -> Callable:
    """Wrap train_step_fn in a fixed-size, mask-aware scan for off-policy training.

    Uses boolean mask to selectively update state, avoiding recompilation when
    fewer steps are needed at chunk boundaries.

    Returns: jitted (carry, active_mask) -> (carry, metrics_history)
    """

    def run_chunk(carry: _CarryT, active_mask: Array) -> tuple[_CarryT, Any]:
        active_mask = active_mask.astype(jnp.bool_)

        def body(carry: _CarryT, active: Array) -> tuple[_CarryT, dict]:
            key, agent_state, training_env_states, buffer_state = carry
            key_new, agent_state_new, training_env_states_new, buffer_state_new, metrics = train_step_fn(
                key=key,
                agent_state=agent_state,
                training_env_states=training_env_states,
                buffer_state=buffer_state,
                batch_size=batch_size,
            )

            active_mask_scalar = jnp.asarray(active, dtype=jnp.bool_)
            key = jax.lax.select(active_mask_scalar, key_new, key)
            agent_state = tree_select(active_mask_scalar, agent_state_new, agent_state)
            training_env_states = tree_select(active_mask_scalar, training_env_states_new, training_env_states)
            buffer_state = tree_select(active_mask_scalar, buffer_state_new, buffer_state)
            metrics = jax.tree_util.tree_map(
                lambda x: jax.lax.select(active_mask_scalar, x, jnp.zeros_like(x)),
                metrics,
            )

            return (key, agent_state, training_env_states, buffer_state), metrics

        return jax.lax.scan(body, carry, active_mask)

    return jax.jit(run_chunk, donate_argnums=(0,))


def make_on_policy_chunk_runner(rollout_fn: Callable, agent: Agent) -> Callable:
    """Batch multiple rollout-update cycles into a single jitted scan for on-policy training.

    Reduces Python overhead by staying in JAX across multiple cycles.

    Returns: jitted (carry, active_mask) -> (carry, metrics_history)
    """

    def run_on_policy_chunk(carry: _CarryOnPolicyT, active_mask: Array) -> tuple[_CarryOnPolicyT, dict]:
        """Run multiple rollout-update cycles in a single jitted scan."""
        active_mask = active_mask.astype(jnp.bool_)

        def body(carry: _CarryOnPolicyT, active: Array) -> tuple[_CarryOnPolicyT, dict]:
            key, agent_state, training_env_states = carry

            # Collect rollout (jitted)
            key_new, agent_state_new, training_env_states_new, rollout_batch = rollout_fn(
                key, agent_state, training_env_states
            )

            # Update agent with collected rollout (jitted)
            key_new, update_key = jax.random.split(key_new)
            agent_state_new, metrics = agent.update(update_key, agent_state_new, rollout_batch, agent.params)

            # Only update state if this iteration is active
            active_mask_scalar = jnp.asarray(active, dtype=jnp.bool_)
            key = jax.lax.select(active_mask_scalar, key_new, key)
            agent_state = tree_select(active_mask_scalar, agent_state_new, agent_state)
            training_env_states = tree_select(active_mask_scalar, training_env_states_new, training_env_states)

            # Zero out metrics for inactive iterations
            metrics = jax.tree_util.tree_map(
                lambda x: jax.lax.select(active_mask_scalar, x, jnp.zeros_like(x)),
                metrics,
            )

            return (key, agent_state, training_env_states), metrics

        return jax.lax.scan(body, carry, active_mask)

    return jax.jit(run_on_policy_chunk, donate_argnums=(0,))


def make_chunked_collector(collection_step_fn: Callable, total_steps: int) -> Callable:
    """Collect rollouts using a single flat scan for on-policy algorithms.

    Uses a single scan over total_steps (simpler than nested chunking) and returns
    the complete rollout batch (flattened to total_steps * num_envs) for GAE/advantage computation.

    Args:
        collection_step_fn: Function that executes one collection step
        num_envs: Unused, kept for API compatibility
        chunk_size: Unused, kept for API compatibility
        total_steps: Total number of steps to collect

    Returns: jitted (key, agent_state, env_states) -> (key, agent_state, env_states, transitions)
    """

    def collect_rollout(
        key: PRNGKey,
        agent_state: AgentState,
        env_states: TrainingEnvState,
    ) -> tuple[PRNGKey, AgentState, TrainingEnvState, Transition]:
        """Collect a full rollout using a single flat scan."""

        def step_body(carry: tuple, _step_idx: Array):
            """Execute one collection step."""
            key, agent_state, env_states = carry

            # Execute the collection step
            (key_new, agent_state_new, env_states_new), transition = collection_step_fn(key, agent_state, env_states)

            return (key_new, agent_state_new, env_states_new), transition

        # Single flat scan over all steps
        (key, agent_state, env_states), all_transitions = jax.lax.scan(
            step_body, (key, agent_state, env_states), jnp.arange(total_steps)
        )

        # Reshape: (total_steps, num_envs, ...) -> (total_steps * num_envs, ...)
        full_rollout = jax.tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]),
            all_transitions,
        )

        return key, agent_state, env_states, full_rollout

    return jax.jit(collect_rollout, donate_argnums=(0, 1, 2))
