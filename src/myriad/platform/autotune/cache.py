"""Cache management for auto-tuning profiles."""

import json
from pathlib import Path


def get_cache_path() -> Path:
    """Get path to autotune cache file.

    Returns:
        Path to ~/.myriad/autotune_profiles.json
    """
    cache_dir = Path.home() / ".myriad"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / "autotune_profiles.json"


def load_cache() -> dict:
    """Load autotune cache from disk.

    Returns:
        Dictionary with cached profiles (hardware, env, agent, configs)
    """

    empty_cache = {
        "hardware": {},
        "env_profiles": {},
        "agent_profiles": {},
        "validated_configs": {},
        "chunk_size_configs": {},
    }

    cache_path = get_cache_path()
    if not cache_path.exists():
        return empty_cache

    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        # Corrupted cache, start fresh
        return empty_cache


def save_cache(cache: dict) -> None:
    """Save autotune cache to disk.

    Args:
        cache: Cache dictionary to save
    """
    cache_path = get_cache_path()
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
