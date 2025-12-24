from __future__ import annotations

from typing import Any, Callable, TypeVar

import chex
import jax
import jax.numpy as jnp

from myriad.agents.agent import AgentState
from myriad.core.replay_buffer import ReplayBufferState
from myriad.core.types import Transition

from .shared import TrainingEnvState

# Generic type for carry state
CarryT = TypeVar("CarryT")


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
    """Wrap ``train_step_fn`` in a fixed-size, mask-aware scan.

    This function creates a chunk runner that executes multiple training steps in a batched
    scan loop. The key design decision is to use a FIXED scan length (determined by the length
    of active_mask) with mask-based state selection to avoid JAX recompilation overhead.

    Design rationale:
    -----------------
    JAX's jit compilation creates optimized code for specific input shapes. If we used a
    dynamic scan length (e.g., jax.lax.scan with length=steps_this_chunk where steps_this_chunk
    varies), we would trigger recompilation every time the chunk size changes. This happens
    frequently at logging/eval boundaries or at the end of training.

    Instead, we:
    1. Always scan for a FIXED number of iterations (len(active_mask))
    2. Use boolean masks to control which iterations actually update state
    3. Inactive iterations execute train_step_fn but discard results via jax.lax.select

    Trade-offs:
    -----------
    - PRO: Single compilation for the chunk runner, avoiding expensive recompilation
    - PRO: Predictable performance characteristics
    - CON: Masked iterations still execute (select actions, step envs) but discard results
    - CON: Wasted computation when steps_this_chunk << chunk_size

    The compilation savings typically outweigh the masked iteration overhead, especially when
    chunk_size is well-tuned relative to logging/eval frequencies.

    Performance tips:
    -----------------
    - Set chunk_size to balance compilation overhead vs step-to-step overhead
    - Avoid chunk_size >> min(log_frequency, eval_frequency) to minimize wasted masked iterations
    - For very small chunk_size (1-5), consider if the overhead is acceptable for your use case

    Args:
        train_step_fn: The training step function to wrap. Must match the signature expected by
            the training loop.
        batch_size: Batch size for sampling from replay buffer (passed to train_step_fn)

    Returns:
        A jitted function that runs a chunk of training steps with mask-aware state updates.
    """

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


def make_chunked_collector(
    collection_step_fn: Callable,
    num_envs: int,
    chunk_size: int,
    total_steps: int,
) -> Callable:
    """Create a chunked rollout collector that gathers transitions efficiently using fixed-size scans.

    This function enables on-policy algorithms to collect rollouts in chunks while maintaining the
    ability to return a full batch of transitions for agent updates. The chunking strategy provides:

    1. **Compilation efficiency**: Fixed chunk sizes avoid recompilation overhead
    2. **Boundary alignment**: Can interleave logging/eval between chunks
    3. **Full-rollout semantics**: Returns complete rollout for advantage computation

    Design:
    -------
    Instead of collecting all rollout_steps at once in a single large scan (which bypasses chunking
    and can cause compilation issues with large rollouts), this collector:

    - Divides the rollout into fixed-size chunks
    - Executes each chunk with mask-aware scanning
    - Accumulates transitions across chunks
    - Returns the full rollout batch for agent update

    This enables on-policy algorithms (PPO, A2C) to use large rollouts (e.g., 2048 steps) while:
    - Benefiting from chunked compilation
    - Respecting logging/eval boundaries
    - Maintaining full-rollout semantics for GAE and advantage computation

    Args:
        collection_step_fn: Function that performs one step of rollout collection.
            Signature: (key, agent_state, env_states) -> ((key, agent_state, env_states), transition)
        num_envs: Number of parallel environments
        chunk_size: Maximum number of steps per chunk (for compilation)
        total_steps: Total number of steps to collect in the rollout

    Returns:
        A jitted function that collects a full rollout in chunks and returns all transitions.
    """
    num_chunks = (total_steps + chunk_size - 1) // chunk_size  # Ceiling division

    def collect_chunked_rollout(
        key: chex.PRNGKey,
        agent_state: AgentState,
        env_states: TrainingEnvState,
    ) -> tuple[chex.PRNGKey, AgentState, TrainingEnvState, Transition]:
        """Collect a full rollout by executing multiple fixed-size chunks."""

        def chunk_body(carry: tuple, chunk_idx: chex.Array):
            """Execute one chunk of the rollout collection."""
            key, agent_state, env_states = carry

            # Determine how many steps are active in this chunk
            steps_collected = chunk_idx * chunk_size
            steps_remaining = total_steps - steps_collected
            steps_this_chunk = jnp.minimum(chunk_size, steps_remaining)

            # Create active mask: True for active iterations, False for inactive (padded) iterations
            active_mask = (jnp.arange(chunk_size) < steps_this_chunk).astype(jnp.bool_)

            def step_body(step_carry: tuple, active: chex.Array):
                """Single step within a chunk."""
                key, agent_state, env_states = step_carry

                # Execute the collection step
                (key_new, agent_state_new, env_states_new), transition = collection_step_fn(
                    key, agent_state, env_states
                )

                # Only update state if this iteration is active
                active_mask_scalar = jnp.asarray(active, dtype=jnp.bool_)
                key = jax.lax.select(active_mask_scalar, key_new, key)
                agent_state = tree_select(active_mask_scalar, agent_state_new, agent_state)
                env_states = tree_select(active_mask_scalar, env_states_new, env_states)

                # Zero out transition data for inactive iterations
                transition = jax.tree_util.tree_map(
                    lambda x: jax.lax.select(active_mask_scalar, x, jnp.zeros_like(x)),
                    transition,
                )

                return (key, agent_state, env_states), transition

            # Run one chunk using scan
            (key, agent_state, env_states), chunk_transitions = jax.lax.scan(
                step_body, (key, agent_state, env_states), active_mask
            )

            return (key, agent_state, env_states), chunk_transitions

        # Collect all chunks
        (key, agent_state, env_states), all_chunk_transitions = jax.lax.scan(
            chunk_body, (key, agent_state, env_states), jnp.arange(num_chunks)
        )

        # Reshape transitions from (num_chunks, chunk_size, num_envs, ...) to (total_steps, num_envs, ...)
        # Then flatten to (total_steps * num_envs, ...)
        def reshape_transitions(x: chex.Array) -> chex.Array:
            # First reshape: (num_chunks, chunk_size, num_envs, ...) -> (num_chunks * chunk_size, num_envs, ...)
            flat_chunks = x.reshape(-1, num_envs, *x.shape[3:])
            # Slice to remove padding: keep only first total_steps
            valid_steps = flat_chunks[:total_steps]
            # Final reshape: (total_steps, num_envs, ...) -> (total_steps * num_envs, ...)
            return valid_steps.reshape(-1, *valid_steps.shape[2:])

        full_rollout = jax.tree_util.tree_map(reshape_transitions, all_chunk_transitions)

        return key, agent_state, env_states, full_rollout

    return jax.jit(collect_chunked_rollout)
