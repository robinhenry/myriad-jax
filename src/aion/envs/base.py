from typing import Any, Callable, NamedTuple

import chex

# Define generic types for state and params
EnvironmentState = Any
EnvironmentParams = Any
EnvironmentConfig = Any


class Environment(NamedTuple):
    """
    A container for the pure functions that define a JAX-based environment.

    This structure acts as a "functional protocol" ensuring that any environment
    implemented this way can be used interchangeably by a generic training loop.
    """

    # The pure, jitted step function
    step: Callable[
        [chex.PRNGKey, EnvironmentState, chex.Array, EnvironmentParams, EnvironmentConfig],
        tuple[chex.Array, EnvironmentState, chex.Array, chex.Array, dict],
    ]

    # The pure, jitted reset function
    reset: Callable[[chex.PRNGKey, EnvironmentParams, EnvironmentConfig], tuple[chex.Array, EnvironmentState]]

    # The pure function to get the action space size
    get_action_space_size: Callable[[], int]

    # Default parameters for the environment
    default_params: EnvironmentParams

    # Static configuration for the environment
    config: EnvironmentConfig
