"""Inferrer registration and metadata.

Mirrors :mod:`myriad.agents.registration` — provides a global registry
so inferrers can be looked up by name in configuration-driven workflows.
"""

import inspect
from typing import Any, Callable, NamedTuple


class InferrerInfo(NamedTuple):
    """Metadata for a registered inferrer.

    Attributes:
        name: Unique identifier for the inferrer.
        make_fn: Factory function that returns an ``Inferrer`` instance.
    """

    name: str
    make_fn: Callable


_INFERRER_REGISTRY: dict[str, InferrerInfo] = {}


def register_inferrer(name: str, make_fn: Callable) -> None:
    """Register an inferrer by name.

    Args:
        name: Unique identifier (e.g., ``"ode-hmc"``, ``"smc-abc"``).
        make_fn: Factory function that returns an :class:`Inferrer`.
    """
    _INFERRER_REGISTRY[name] = InferrerInfo(name=name, make_fn=make_fn)


def get_inferrer_info(name: str) -> InferrerInfo | None:
    """Look up metadata for a registered inferrer.

    Returns:
        ``InferrerInfo`` if found, ``None`` otherwise.
    """
    return _INFERRER_REGISTRY.get(name)


def list_inferrers() -> list[str]:
    """List all registered inferrer names."""
    return list(_INFERRER_REGISTRY.keys())


def make_inferrer(name: str, **kwargs: Any) -> Any:
    """Create an inferrer instance by name.

    Unknown kwargs are silently filtered if the factory function does not
    accept ``**kwargs``, matching the behaviour of :func:`myriad.agents.make_agent`.

    Raises:
        ValueError: If *name* is not in the registry.
    """
    info = get_inferrer_info(name)
    if info is None:
        available = ", ".join(list_inferrers())
        raise ValueError(f"Inferrer '{name}' not found in the registry. Available inferrers: {available}")

    sig = inspect.signature(info.make_fn)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if not has_var_keyword:
        valid = set(sig.parameters.keys())
        kwargs = {k: v for k, v in kwargs.items() if k in valid}
    return info.make_fn(**kwargs)
