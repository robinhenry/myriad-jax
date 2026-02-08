"""Environment registration and metadata.

This module provides a structured registry for environments, allowing the platform
to query environment properties (like default max steps) without necessarily
instantiating them.
"""

from typing import Any, Callable, NamedTuple


class EnvInfo(NamedTuple):
    """Metadata for a registered environment.

    Attributes:
        name: The unique identifier for the environment.
        make_fn: The factory function to create the environment.
        config_cls: The configuration class for the environment.
        render_frame_fn: Optional function to render a single frame of the environment.
    """

    name: str
    make_fn: Callable
    config_cls: type
    render_frame_fn: Callable | None = None


# The global registry of environments
_ENV_REGISTRY: dict[str, EnvInfo] = {}


def register_env(
    name: str,
    make_fn: Callable,
    config_cls: type,
    render_frame_fn: Callable | None = None,
) -> None:
    """Register an environment with metadata.

    Args:
        name: Unique identifier for the environment.
        make_fn: Factory function to create the environment.
        config_cls: Configuration class for the environment.
        render_frame_fn: Optional function to render a single frame.
    """
    _ENV_REGISTRY[name] = EnvInfo(
        name=name,
        make_fn=make_fn,
        config_cls=config_cls,
        render_frame_fn=render_frame_fn,
    )


def get_env_info(name: str) -> EnvInfo | None:
    """Get metadata for a registered environment.

    Args:
        name: Unique identifier for the environment.

    Returns:
        EnvInfo if registered, None otherwise.
    """
    return _ENV_REGISTRY.get(name)


def list_envs() -> list[str]:
    """List all registered environment identifiers.

    Returns:
        List of environment names.
    """
    return list(_ENV_REGISTRY.keys())


def make_env(name: str, **kwargs: Any) -> Any:
    """Create an environment instance by name.

    Args:
        name: Unique identifier for the environment.
        **kwargs: Keyword arguments passed to the environment's factory function.

    Returns:
        An instance of the requested Environment.

    Raises:
        ValueError: If the environment name is not found in the registry.
    """
    info = get_env_info(name)
    if info is None:
        available = ", ".join(list_envs())
        raise ValueError(f"Environment '{name}' not found in the registry. Available environments: {available}")

    return info.make_fn(**kwargs)
