from .cartpole.tasks import control as cartpole_control, sysid as cartpole_sysid
from .ccas_ccar.tasks import control as ccas_ccar_control, sysid as ccas_ccar_sysid
from .environment import Environment

# The registry mapping environment IDs to their factory functions
ENV_REGISTRY = {
    # Modular CartPole tasks
    "cartpole-control": cartpole_control.make_env,
    "cartpole-sysid": cartpole_sysid.make_env,
    # Modular CcaS-CcaR tasks
    "ccas-ccar-control": ccas_ccar_control.make_env,
    "ccas-ccar-sysid": ccas_ccar_sysid.make_env,
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
    return make_fn(**kwargs)  # type: ignore[operator]
