"""Tests for autotune cache management."""

import json
from pathlib import Path

from myriad.platform.autotune.cache import get_cache_path, load_cache, save_cache


class TestCachePath:
    """Tests for get_cache_path."""

    def test_cache_path_structure(self):
        """Cache path should be in user home directory."""
        cache_path = get_cache_path()

        assert cache_path.name == "autotune_profiles.json"
        assert cache_path.parent.name == ".myriad"
        assert cache_path.parent.parent == Path.home()

    def test_cache_directory_created(self):
        """get_cache_path should create .myriad directory."""
        cache_path = get_cache_path()

        # Directory should exist after calling get_cache_path
        assert cache_path.parent.exists()
        assert cache_path.parent.is_dir()


class TestLoadCache:
    """Tests for load_cache."""

    def test_load_empty_cache(self, monkeypatch, tmp_path):
        """Load cache when file doesn't exist should return empty structure."""
        cache_path = tmp_path / "autotune_profiles.json"
        monkeypatch.setattr("myriad.platform.autotune.cache.get_cache_path", lambda: cache_path)

        cache = load_cache()

        assert isinstance(cache, dict)
        assert "hardware" in cache
        assert "env_profiles" in cache
        assert "agent_profiles" in cache
        assert "validated_configs" in cache
        assert "chunk_size_configs" in cache
        assert cache["hardware"] == {}
        assert cache["env_profiles"] == {}

    def test_load_existing_cache(self, monkeypatch, tmp_path):
        """Load cache from existing file."""
        cache_path = tmp_path / "autotune_profiles.json"
        monkeypatch.setattr("myriad.platform.autotune.cache.get_cache_path", lambda: cache_path)

        # Create a cache file
        test_cache = {
            "hardware": {"hw1": {"memory_gb": 16.0}},
            "env_profiles": {"cartpole": {"memory_mb": 1.5}},
            "agent_profiles": {},
            "validated_configs": {},
            "chunk_size_configs": {},
        }

        with open(cache_path, "w") as f:
            json.dump(test_cache, f)

        # Load it
        cache = load_cache()

        assert cache["hardware"]["hw1"]["memory_gb"] == 16.0
        assert cache["env_profiles"]["cartpole"]["memory_mb"] == 1.5

    def test_load_corrupted_cache(self, monkeypatch, tmp_path):
        """Load corrupted cache should return empty structure."""
        cache_path = tmp_path / "autotune_profiles.json"
        monkeypatch.setattr("myriad.platform.autotune.cache.get_cache_path", lambda: cache_path)

        # Write invalid JSON
        with open(cache_path, "w") as f:
            f.write("{ invalid json }")

        # Should return empty cache, not crash
        cache = load_cache()

        assert isinstance(cache, dict)
        assert cache["hardware"] == {}


class TestSaveCache:
    """Tests for save_cache."""

    def test_save_cache_creates_file(self, monkeypatch, tmp_path):
        """save_cache should create the cache file."""
        cache_path = tmp_path / "autotune_profiles.json"
        monkeypatch.setattr("myriad.platform.autotune.cache.get_cache_path", lambda: cache_path)

        cache = {
            "hardware": {"hw1": {"memory_gb": 32.0}},
            "env_profiles": {},
            "agent_profiles": {},
            "validated_configs": {},
            "chunk_size_configs": {},
        }

        save_cache(cache)

        assert cache_path.exists()
        assert cache_path.is_file()

    def test_save_cache_valid_json(self, monkeypatch, tmp_path):
        """save_cache should write valid JSON."""
        cache_path = tmp_path / "autotune_profiles.json"
        monkeypatch.setattr("myriad.platform.autotune.cache.get_cache_path", lambda: cache_path)

        cache = {
            "hardware": {"hw1": {"memory_gb": 32.0}},
            "env_profiles": {"env1": {"memory_mb": 2.5}},
            "agent_profiles": {},
            "validated_configs": {},
            "chunk_size_configs": {},
        }

        save_cache(cache)

        # Read and verify it's valid JSON
        with open(cache_path, "r") as f:
            loaded = json.load(f)

        assert loaded == cache


class TestCacheRoundtrip:
    """Tests for save/load round-trip behavior."""

    def test_roundtrip_preserves_data(self, monkeypatch, tmp_path):
        """Save then load should preserve all data."""
        cache_path = tmp_path / "autotune_profiles.json"
        monkeypatch.setattr("myriad.platform.autotune.cache.get_cache_path", lambda: cache_path)

        original = {
            "hardware": {
                "hw1": {
                    "memory_gb": 16.0,
                    "device": "gpu",
                    "platform": "NVIDIA",
                }
            },
            "env_profiles": {
                "cartpole": {"memory_mb": 1.5, "profiled_at": "2024-01-01"},
                "pendulum": {"memory_mb": 2.0, "profiled_at": "2024-01-02"},
            },
            "agent_profiles": {"dqn": {"overhead_mb": 10.0, "method": "off-policy"}},
            "validated_configs": {"config1": {"max_envs": 100000, "chunk_size": 1000}},
            "chunk_size_configs": {"config2": {"optimal_chunk_size": 500}},
        }

        save_cache(original)
        loaded = load_cache()

        assert loaded == original

    def test_roundtrip_empty_sections(self, monkeypatch, tmp_path):
        """Round-trip with empty sections should work."""
        cache_path = tmp_path / "autotune_profiles.json"
        monkeypatch.setattr("myriad.platform.autotune.cache.get_cache_path", lambda: cache_path)

        original = {
            "hardware": {},
            "env_profiles": {},
            "agent_profiles": {},
            "validated_configs": {},
            "chunk_size_configs": {},
        }

        save_cache(original)
        loaded = load_cache()

        assert loaded == original

    def test_multiple_save_overwrites(self, monkeypatch, tmp_path):
        """Multiple saves should overwrite previous data."""
        cache_path = tmp_path / "autotune_profiles.json"
        monkeypatch.setattr("myriad.platform.autotune.cache.get_cache_path", lambda: cache_path)

        cache1 = {
            "hardware": {"hw1": {"memory_gb": 16.0}},
            "env_profiles": {},
            "agent_profiles": {},
            "validated_configs": {},
            "chunk_size_configs": {},
        }
        save_cache(cache1)

        cache2 = {
            "hardware": {"hw2": {"memory_gb": 32.0}},
            "env_profiles": {},
            "agent_profiles": {},
            "validated_configs": {},
            "chunk_size_configs": {},
        }
        save_cache(cache2)

        loaded = load_cache()

        # Should have cache2 data, not cache1
        assert "hw2" in loaded["hardware"]
        assert "hw1" not in loaded["hardware"]
        assert loaded["hardware"]["hw2"]["memory_gb"] == 32.0
