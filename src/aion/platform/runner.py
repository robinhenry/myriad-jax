from functools import partial
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
from flax import struct
from omegaconf import OmegaConf

from aion.agents import make_agent
from aion.agents.agent import Agent, AgentState
from aion.configs.default import Config
from aion.core.replay_buffer import ReplayBuffer, ReplayBufferState
from aion.core.spaces import Space
from aion.core.types import Transition
from aion.envs import make_env
from aion.envs.environment import Environment, EnvironmentState


@struct.dataclass
class TrainingEnvState:
    """Container for the state of a training environment, including observations."""

    env_state: EnvironmentState
    obs: chex.Array


def _make_train_step_fn(
    agent: Agent,
    env: Environment,
    replay_buffer: ReplayBuffer,
    num_envs: int,
) -> Callable:
    """Factory to create a jitted, vmapped training step function"""

    # Vmap the environment step and reset functions
    vmapped_env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None, None))
    vmapped_env_reset = jax.vmap(env.reset, in_axes=(0, None, None))

    @partial(jax.jit, static_argnames=["batch_size"])
    def train_step(
        key: chex.PRNGKey,
        agent_state: AgentState,
        training_env_states: TrainingEnvState,
        buffer_state: ReplayBufferState,
        batch_size: int,
    ) -> tuple[chex.PRNGKey, AgentState, TrainingEnvState, ReplayBufferState, dict]:
        """Executes one step of training across all parallel environments. This function is pure and jitted."""

        last_obs = training_env_states.obs

        # 1. SELECT ACTION
        key, action_key = jax.random.split(key)
        action_keys = jax.random.split(action_key, num_envs)
        actions, agent_state = jax.vmap(agent.select_action, in_axes=(0, 0, 0, None))(
            action_keys, last_obs, agent_state, agent.params
        )

        # 2. STEP THE ENVIRONMENTS
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, next_env_states, rewards, dones, infos = vmapped_env_step(
            step_keys, training_env_states.env_state, actions, env.params, env.config
        )

        # 3. STORE TRANSITIONS AND SAMPLE FROM BUFFER
        key, buffer_key = jax.random.split(key)
        transitions = Transition(last_obs, actions, rewards, next_obs, dones)
        buffer_state, batch = replay_buffer.add_and_sample(buffer_state, transitions, batch_size, buffer_key)

        # 4. UPDATE THE AGENT
        key, update_key = jax.random.split(key)
        agent_state, metrics = agent.update(update_key, agent_state, batch, agent.params)

        # Handle auto-resetting environments that are 'done'
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, num_envs)

        # Reset only the environments that are done
        new_obs, new_env_states = vmapped_env_reset(reset_keys, env.params, env.config)

        # If done, use the new state, otherwise keep the existing one
        final_obs = jnp.where(dones[:, None], new_obs, next_obs)  # type: ignore
        final_env_states = jax.tree_util.tree_map(
            lambda old, new: jnp.where(dones, new, old), next_env_states, new_env_states
        )

        # Store the final states and observations as the new training environment state
        new_training_env_state = TrainingEnvState(env_state=final_env_states, obs=final_obs)

        # Add episode return to metrics for logging
        # metrics["episode_return"] = infos["episode_return"]

        return (
            key,
            agent_state,
            new_training_env_state,
            buffer_state,
            metrics,
        )

    return train_step


def train_and_evaluate(config: Config):
    """
    Main entry point for a training run.
    Initializes everything and runs the outer training loop.
    """
    # 1. INITIALIZE EVERYTHING
    key = jax.random.PRNGKey(config.seed)
    key, env_key, agent_key, buffer_key = jax.random.split(key, 4)

    # Create the env
    env_kwargs = _get_factory_kwargs(config.env)
    env = make_env(config.env.name, **env_kwargs)

    # Create the agent
    agent_kwargs = _get_factory_kwargs(config.agent)
    action_space = env.get_action_space(env.config)
    agent = make_agent(config.agent.name, action_space=action_space, **agent_kwargs)

    # Initialize parallel environments
    env_keys = jax.random.split(env_key, config.num_envs)
    obs, env_states = jax.vmap(env.reset, in_axes=(0, None, None))(env_keys, env.params, env.config)
    training_env_states = TrainingEnvState(env_state=env_states, obs=obs)

    # Initialize agent using the initial observation from one environment
    sample_obs = obs[0]  # type: ignore
    agent_state = agent.init(agent_key, sample_obs, agent.params)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(buffer_size=config.buffer_size)
    sample_transition = _make_sample_transition(buffer_key, sample_obs, action_space)
    buffer_state = replay_buffer.init(sample_transition)

    # Create the jitted training function
    train_step_fn = _make_train_step_fn(agent, env, replay_buffer, config.num_envs)

    # 2. OUTER TRAINING LOOP
    for step in range(config.total_timesteps // config.num_envs):
        key, agent_state, training_env_states, buffer_state, metrics = train_step_fn(
            key=key,
            agent_state=agent_state,
            training_env_states=training_env_states,
            buffer_state=buffer_state,
            batch_size=config.batch_size,
        )

        # Log metrics (this happens outside the JIT loop)
        # if step % config.log_frequency == 0:
        #     # Aggregate returns from all envs that finished in this step
        #     finished_returns = metrics["episode_return"][metrics["episode_return"] != 0]
        #     if len(finished_returns) > 0:
        #         avg_return = finished_returns.mean()
        #         print(f"Step: {step * config.num_envs}, Avg Return: {avg_return:.2f}, Loss: {metrics['loss']:.4f}")

    print("Training finished.")


def _get_factory_kwargs(config: Any) -> dict:
    """Converts a dataclass config object to a dict for factory functions."""
    kwargs = OmegaConf.to_container(config, resolve=True)
    assert isinstance(kwargs, dict)
    kwargs.pop("name")  # The name is used for lookup, not as a parameter
    return kwargs


def _make_sample_transition(key: chex.PRNGKey, sample_obs: chex.Array, action_space: Space) -> Transition:
    """Creates a sample transition PyTree for replay buffer initialization."""
    sample_action = action_space.sample(key)
    sample_reward = jnp.array(0.0, dtype=jnp.float32)
    sample_done = jnp.array(0.0, dtype=jnp.float32)

    return Transition(
        sample_obs,
        sample_action,
        sample_reward,
        sample_obs,  # next_obs has the same shape as obs
        sample_done,
    )
