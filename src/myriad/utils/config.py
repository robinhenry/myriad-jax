from pathlib import Path
from typing import Type, TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_config(path: str | Path, config_cls: Type[T]) -> T:
    """
    Load a YAML config file and convert to a Pydantic config object.

    Args:
        path: Path to YAML config file
        config_cls: Pydantic config class to instantiate (e.g., EvalConfig)

    Returns:
        Instantiated and validated config object

    Example:
        >>> from myriad.configs.default import EvalConfig
        >>> config = load_config("config.yaml", EvalConfig)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)

    return config_cls(**config_dict)
