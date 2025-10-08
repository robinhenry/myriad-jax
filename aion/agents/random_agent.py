"""A simple random agent."""

import jax
from gymnax.environments.environment import EnvState
from typing import Tuple


class RandomAgent:
    """
    A simple agent that selects actions randomly.
    """

    def __init__(self, action_space):
        """
        Args:
            action_space: The action space of the environment.
        """
        self.action_space = action_space

    def init(self, key: jax.Array) -> None:
        """
        Initializes the agent's state. For a random agent, there is no state.
        """
        return None

    def select_action(
        self, key: jax.Array, observation: jax.Array, agent_state: EnvState | None
    ) -> Tuple[jax.Array, EnvState | None]:
        """
        Selects a random action from the action space.
        """
        action = self.action_space.sample(key)
        return action, agent_state
