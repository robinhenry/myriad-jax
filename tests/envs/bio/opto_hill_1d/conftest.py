"""Shared fixtures for opto_hill_1d tests."""

import pytest

from myriad.envs.bio.opto_hill_1d.physics import PhysicsConfig, PhysicsParams
from myriad.envs.bio.opto_hill_1d.tasks.sysid import TaskConfig


@pytest.fixture
def physics_config():
    return PhysicsConfig()


@pytest.fixture
def physics_params():
    return PhysicsParams()


@pytest.fixture
def task_config():
    return TaskConfig()
