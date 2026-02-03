"""Shared fixtures for CcaS-CcaR gene circuit tests."""

import pytest

from myriad.envs.bio.ccas_ccar.physics import PhysicsConfig, PhysicsParams
from myriad.envs.bio.ccas_ccar.tasks.base import TaskConfig


@pytest.fixture
def physics_config():
    """Default physics configuration."""
    return PhysicsConfig()


@pytest.fixture
def physics_params():
    """Default physics parameters."""
    return PhysicsParams()


@pytest.fixture
def task_config():
    """Default task configuration."""
    return TaskConfig()
