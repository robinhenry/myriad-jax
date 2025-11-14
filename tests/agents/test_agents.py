# """Tests for agents."""

# import jax
# import jax.numpy as jnp
# import pytest

# from aion.agents.pqn_agent import PQNAgent, QNetwork


# class TestPQNAgent:
#     """Tests for the PQN agent."""

#     @pytest.fixture
#     def agent(self) -> PQNAgent:
#         """Fixture for PQN agent."""
#         return PQNAgent(
#             q_network=QNetwork(action_dim=2),
#             learning_rate=1e-3,
#             gamma=0.99,
#             epsilon_start=1.0,
#             epsilon_end=0.1,
#             epsilon_decay_steps=1000,
#             target_network_frequency=100,
#             tau=0.005,
#             action_dim=2,
#         )

#     @pytest.fixture
#     def key(self) -> jax.Array:
#         """Fixture for JAX random key."""
#         return jax.random.PRNGKey(42)

#     def test_agent_initialization(self, agent: PQNAgent, key: jax.Array):
#         """Test agent initialization."""
#         sample_obs = jnp.array([5.0, 5.0])
#         train_state = agent.init(key, sample_obs)

#         assert train_state.params is not None
#         assert train_state.target_params is not None
#         assert train_state.step == 0

#     def test_action_selection(self, agent: PQNAgent, key: jax.Array):
#         """Test action selection."""
#         sample_obs = jnp.array([5.0, 5.0])
#         train_state = agent.init(key, sample_obs)

#         key, action_key = jax.random.split(key)
#         action, _ = agent.select_action(action_key, sample_obs, train_state, 0)

#         assert action.shape == ()
#         assert 0 <= action < agent.action_dim

#     def test_agent_update(self, agent: PQNAgent, key: jax.Array):
#         """Test agent update."""
#         sample_obs = jnp.array([5.0, 5.0])
#         train_state = agent.init(key, sample_obs)

#         # Create fake batch
#         batch_size = 32
#         obs = jnp.tile(sample_obs, (batch_size, 1))
#         actions = jax.random.randint(key, (batch_size,), 0, 2)
#         rewards = jax.random.normal(key, (batch_size,))
#         next_obs = obs + 0.1
#         dones = jnp.zeros((batch_size,))

#         batch = (obs, actions, rewards, next_obs, dones)

#         new_train_state, loss = agent.update(train_state, batch)

#         assert new_train_state.step == train_state.step + 1
#         assert isinstance(loss, (float, jnp.ndarray))
