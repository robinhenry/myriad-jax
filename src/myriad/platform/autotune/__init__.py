"""Auto-tuning system for optimal scan_chunk_size and hardware profiling.

This module provides automatic optimization of JAX scan chunk sizes and
environment parallelism based on hardware capabilities.

Main API:
    suggest_scan_chunk_size: Find optimal scan_chunk_size for fixed num_envs (primary)
    suggest_config: Find maximum num_envs for hardware (advanced/benchmarking)

Cache management:
    load_cache: Load cached profiles
    save_cache: Save profiles to cache
    get_cache_path: Get cache file location
"""

from .api import AutotuneResult, suggest_config, suggest_scan_chunk_size
from .cache import get_cache_path, load_cache, save_cache

__all__ = [
    # Main API
    "suggest_scan_chunk_size",
    "suggest_config",
    "AutotuneResult",
    # Cache management
    "load_cache",
    "save_cache",
    "get_cache_path",
]
