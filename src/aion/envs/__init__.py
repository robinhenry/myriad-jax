from . import toy_env
from .environment import Environment

# The registry mapping environment IDs to their factory functions
ENV_REGISTRY = {
    "toy_env_v1": toy_env.make_env,
}


def make_env(env_id: str, **kwargs) -> Environment:
    """
    A general factory function to create any registered environment.

    Args:
        env_id: The string identifier of the environment to create.
        **kwargs: Keyword arguments that will be passed to the specific
                  environment's make_env function.

    Returns:
        An instance of the requested Environment.

    Raises:
        ValueError: If the env_id is not found in the registry.
    """
    if env_id not in ENV_REGISTRY:
        raise ValueError(
            f"Environment '{env_id}' not found in the registry. Available environments: {list(ENV_REGISTRY.keys())}"
        )

    # Look up the factory function and call it with the provided arguments
    make_fn = ENV_REGISTRY[env_id]
    return make_fn(**kwargs)
