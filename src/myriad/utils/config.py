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


def save_config(config: BaseModel, path: str | Path) -> Path:
    """
    Save a Pydantic config to a YAML file.

    Args:
        path: Path to write the YAML config file
        config: Pydantic config object (e.g., Config, EvalConfig)

    Returns:
        The path written to (for chaining)

    Example:
        >>> from myriad import save_config
        >>> save_config(config, "my_run/config.yaml")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)

    return path
