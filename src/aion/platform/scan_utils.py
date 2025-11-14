from __future__ import annotations

from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp

from aion.agents.agent import AgentState
from aion.core.replay_buffer import ReplayBufferState

from .shared import TrainingEnvState


def tree_select(mask: chex.Array, new_tree: Any, old_tree: Any) -> Any:
    """Selects between two pytrees using a scalar boolean mask."""

    return jax.tree_util.tree_map(lambda new, old: jax.lax.select(mask, new, old), new_tree, old_tree)


def _expand_mask(mask: chex.Array, target_ndim: int) -> chex.Array:
    """Reshapes a mask so it can broadcast to a target rank."""

    expand_dims = target_ndim - mask.ndim
    if expand_dims <= 0:
        return mask
    return mask.reshape(mask.shape + (1,) * expand_dims)


def where_mask(mask: chex.Array, new_value: chex.Array, old_value: chex.Array) -> chex.Array:
    """Selects array values using a boolean mask, supporting broadcasting."""

    mask_bool = mask.astype(jnp.bool_)
    return jnp.where(_expand_mask(mask_bool, new_value.ndim), new_value, old_value)


def mask_tree(mask: chex.Array, new_tree: Any, old_tree: Any) -> Any:
    """Selects between two pytrees using a (potentially vector) mask."""

    return jax.tree_util.tree_map(lambda new, old: where_mask(mask, new, old), new_tree, old_tree)


def make_chunk_runner(train_step_fn: Callable, batch_size: int) -> Callable:
    """Wrap ``train_step_fn`` in a fixed-size, mask-aware scan."""

    def run_chunk(
        carry: tuple[chex.PRNGKey, AgentState, TrainingEnvState, ReplayBufferState],
        active_mask: chex.Array,
    ) -> tuple[tuple[chex.PRNGKey, AgentState, TrainingEnvState, ReplayBufferState], Any]:
        active_mask = active_mask.astype(jnp.bool_)

        def body(
            carry: tuple[chex.PRNGKey, AgentState, TrainingEnvState, ReplayBufferState],
            active: chex.Array,
        ) -> tuple[tuple[chex.PRNGKey, AgentState, TrainingEnvState, ReplayBufferState], dict]:
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

    return jax.jit(run_chunk)
