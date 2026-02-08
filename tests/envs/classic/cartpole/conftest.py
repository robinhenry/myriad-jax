import pytest

from myriad.envs.classic.cartpole.physics import PhysicsConfig, PhysicsParams
from myriad.envs.classic.cartpole.tasks.base import TaskConfig


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
