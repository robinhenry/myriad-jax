"""Tests for interaction functions."""

import jax
import jax.numpy as jnp
import pytest

from aion.agents.pqn_agent import PQNAgent, QNetwork
from aion.envs.toy_env_v1 import EnvParams, create_constant_target
from aion.platform.interaction import run_episodes_parallel


class TestInteraction:
    """Tests for interaction functions."""

    @pytest.fixture
    def env_params(self) -> EnvParams:
        """Fixture for environment parameters."""
        return EnvParams(max_steps=20, x_target=create_constant_target(5.0, 20))

    @pytest.fixture
    def agent(self) -> PQNAgent:
        """Fixture for agent."""
        return PQNAgent(
            q_network=QNetwork(action_dim=2),
            learning_rate=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay_steps=1000,
            target_network_frequency=100,
            tau=0.005,
            action_dim=2,
        )

    @pytest.fixture
    def key(self) -> jax.Array:
        """Fixture for JAX random key."""
        return jax.random.PRNGKey(0)

    def test_parallel_episodes(self, env_params: EnvParams, agent: PQNAgent, key: jax.Array):
        """Test parallel episode execution."""
        sample_obs = jnp.array([5.0, 5.0])
        train_state = agent.init(key, sample_obs)

        num_envs = 4
        max_steps = 10

        trajectories = run_episodes_parallel(
            key,
            agent.select_action,
            train_state,
            env_params,
            num_envs,
            max_steps,
        )

        obs, actions, rewards, next_obs, dones = trajectories

        # Check shapes
        assert obs.shape == (max_steps, num_envs, 2)
        assert actions.shape == (max_steps, num_envs)
        assert rewards.shape == (max_steps, num_envs)
        assert next_obs.shape == (max_steps, num_envs, 2)
        assert dones.shape == (max_steps, num_envs)

        # Check action bounds
        assert jnp.all(actions >= 0)
        assert jnp.all(actions < 2)

        # Check that rewards are negative (distance-based)
        assert jnp.all(rewards <= 0)
