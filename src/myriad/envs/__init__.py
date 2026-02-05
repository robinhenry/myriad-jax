from .bio.ccas_ccar.tasks import control as ccas_ccar_control
from .classic.cartpole.tasks import control as cartpole_control
from .classic.pendulum.tasks import control as pendulum_control
from .environment import (
    Environment,
    EnvironmentConfig,
    EnvironmentParams,
    EnvironmentState,
)

# The registry mapping environment IDs to their factory functions
ENV_REGISTRY = {
    # Modular CartPole tasks
    "cartpole-control": cartpole_control.make_env,
    # Modular Pendulum tasks
    "pendulum-control": pendulum_control.make_env,
    # Modular CcaS-CcaR tasks
    "ccas-ccar-control": ccas_ccar_control.make_env,
}

__all__ = [
    "make_env",
    "Environment",
    "EnvironmentConfig",
    "EnvironmentParams",
    "EnvironmentState",
    "ENV_REGISTRY",
]


def make_env(env_id: str, **kwargs) -> Environment:
    """
    A general factory function to create any registered environment.

    Args:
        env_id: The string identifier of the environment to create.
        **kwargs: Keyword arguments that will be passed to the specific
                  environment's ``make_env()`` function.

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
    return make_fn(**kwargs)  # type: ignore[operator]
