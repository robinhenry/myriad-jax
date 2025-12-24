"""
A JAX-native, functional implementation of a replay buffer.
"""

from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass


class ReplayBufferState(NamedTuple):
    """
    State of the replay buffer. Contains the stored data and the current position.

    Attributes:
        data: A PyTree of JAX arrays where each leaf has a shape of (buffer_size, ...).
        position: The current index in the buffer to write the next transition.
        size: The current number of valid transitions stored in the buffer.
    """

    data: chex.ArrayTree
    position: chex.Array
    size: chex.Array


@dataclass
class ReplayBuffer:
    """
    A class that holds the pure functions for a replay buffer.

    Attributes:
        buffer_size: The maximum number of transitions to store.
    """

    buffer_size: int

    def init(self, sample_transition: chex.ArrayTree) -> ReplayBufferState:
        """
        Initializes the replay buffer state.

        Args:
            sample_transition: A sample transition PyTree to infer shapes and dtypes.

        Returns:
            The initial ReplayBufferState.
        """

        # Convert python scalars to arrays and infer shapes/dtypes robustly
        def _to_array(x):
            if isinstance(x, jnp.ndarray):
                return x
            return jnp.asarray(x)

        sample_transition = jax.tree_util.tree_map(_to_array, sample_transition)

        # Create zero-filled arrays for the buffer data based on the sample transition
        data = jax.tree_util.tree_map(
            lambda x: jnp.zeros((self.buffer_size,) + x.shape, dtype=x.dtype),
            sample_transition,
        )

        return ReplayBufferState(
            data=data,
            position=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32),
        )

    def add(self, state: ReplayBufferState, transitions: chex.ArrayTree) -> ReplayBufferState:
        """
        Adds a batch of transitions to the buffer.

        Args:
            state: The current state of the replay buffer.
            transitions: A PyTree of transitions to add. Each leaf must have a
                         leading dimension matching the number of parallel environments.

        Returns:
            The new ReplayBufferState after adding the transitions.
        """

        # Validate transitions tree is non-empty
        leaves = jax.tree_util.tree_leaves(transitions)
        if len(leaves) == 0:
            raise ValueError("transitions must be a non-empty PyTree")

        # Get the number of transitions to add from the leading dimension
        num_transitions_to_add = leaves[0].shape[0]

        # Calculate the indices to write to, wrapping around the buffer
        idxs = jnp.arange(num_transitions_to_add, dtype=state.position.dtype)
        indices = (state.position + idxs) % self.buffer_size

        # Update the buffer data at the calculated indices
        new_data = jax.tree_util.tree_map(
            lambda buffer_leaf, transition_leaf: buffer_leaf.at[indices].set(transition_leaf),
            state.data,
            transitions,
        )

        new_position = (state.position + num_transitions_to_add) % self.buffer_size
        new_size = jnp.minimum(state.size + num_transitions_to_add, self.buffer_size)

        return ReplayBufferState(data=new_data, position=new_position, size=new_size)

    def sample(
        self, state: ReplayBufferState, batch_size: int, key: chex.PRNGKey
    ) -> tuple[ReplayBufferState, chex.ArrayTree]:
        """
        Samples a random batch of transitions from the buffer.

        Args:
            state: The current state of the replay buffer.
            batch_size: The number of transitions to sample.
            key: A JAX PRNG key for sampling.

        Returns:
            A tuple containing the unchanged buffer state and the sampled batch.
        """

        max_index = jnp.maximum(state.size, 1)  # avoid maxval==0
        sample_indices = jax.random.randint(key, (int(batch_size),), 0, max_index, dtype=jnp.int32)
        sampled = jax.tree_util.tree_map(lambda buf: buf[sample_indices], state.data)

        return state, sampled

    def add_and_sample(
        self,
        state: ReplayBufferState,
        transitions: chex.ArrayTree,
        batch_size: int,
        key: chex.PRNGKey,
    ) -> Tuple[ReplayBufferState, chex.ArrayTree]:
        """
        Adds a batch of transitions to the buffer and samples a random batch.
        This is a pure function.

        Args:
            state: The current state of the replay buffer.
            transitions: A PyTree of transitions to add. Each leaf must have a
                         leading dimension matching the number of parallel environments.
            batch_size: The number of transitions to sample.
            key: A JAX PRNG key for sampling.

        Returns:
            A tuple containing the new buffer state and the sampled batch.
        """
        new_state = self.add(state, transitions)
        return self.sample(new_state, batch_size, key)
