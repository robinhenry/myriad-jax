"""Core functions for running, training, and evaluating agents in environments."""

from dataclasses import asdict
from functools import partial
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp

import aion.agents
import aion.envs
from aion.agents.base import Agent
from aion.configs.default import Config
from aion.envs.base import Environment
from aion.platform.replay_buffer import ReplayBuffer, ReplayBufferState


def _make_train_step_fn(
    agent: Agent,
    env: Environment,
    replay_buffer: ReplayBuffer,
    num_envs: int,
) -> Callable:
    """
    Factory to create a jitted, vmapped training step function.
    This is the core of the JAX optimization.
    """
    # Vmap the environment step and reset functions
    vmapped_env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None, None))
    vmapped_env_reset = jax.vmap(env.reset, in_axes=(0, None, None))

    @partial(jax.jit, static_argnames=["batch_size"])
    def train_step(
        key: chex.PRNGKey,
        agent_state: chex.ArrayTree,
        env_states: chex.ArrayTree,
        buffer_state: ReplayBufferState,
        batch_size: int,
    ) -> tuple:
        """
        Executes one step of training across all parallel environments.
        This function is pure and jitted.
        """
        # 1. SELECT ACTION
        key, action_key = jax.random.split(key)
        action_keys = jax.random.split(action_key, num_envs)
        # Assuming env_states has an `obs` field/attribute
        actions, _ = jax.vmap(agent.select_action, in_axes=(0, 0, 0, None))(
            action_keys, env_states.obs, agent_state, agent.default_params
        )

        # 2. STEP THE ENVIRONMENT
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, next_env_states, rewards, dones, infos = vmapped_env_step(
            step_keys, env_states, actions, env.default_params, env.config
        )

        # 3. STORE TRANSITIONS AND SAMPLE FROM BUFFER
        key, buffer_key = jax.random.split(key)
        # Assuming env_states.obs is the observation before the step
        transitions = (env_states.obs, actions, rewards, next_obs, dones)
        buffer_state, batch = replay_buffer.add_and_sample(buffer_state, transitions, batch_size, buffer_key)

        # 4. UPDATE THE AGENT
        key, update_key = jax.random.split(key)
        agent_state, metrics = agent.update(update_key, agent_state, batch, agent.default_params)

        # Handle auto-resetting environments that are 'done'
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, num_envs)

        # Reset only the environments that are done
        new_obs, new_env_states = vmapped_env_reset(reset_keys, env.default_params, env.config)

        # If done, use the new state, otherwise keep the existing one
        final_obs = jnp.where(dones[:, None], new_obs, next_obs)
        final_env_states = jax.tree_util.tree_map(
            lambda old, new: jnp.where(dones, new, old), next_env_states, new_env_states
        )
        # The final observation needs to be part of the final state
        final_env_states = final_env_states.replace(obs=final_obs)

        # Add episode return to metrics for logging
        metrics["episode_return"] = infos["episode_return"]

        return (
            key,
            agent_state,
            final_env_states,
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

    # Create env and agent from the config using the registries
    env_kwargs = _get_factory_kwargs(config.env)
    env = aion.envs.make_env(config.env.name, **env_kwargs)

    agent_kwargs = _get_factory_kwargs(config.agent)
    # The agent needs to know the action dimension from the environment
    agent_kwargs["action_dim"] = env.get_action_space_size()
    agent = aion.agents.make_agent(config.agent.name, **agent_kwargs)

    # Initialize states
    key, env_key, agent_key, buffer_key = jax.random.split(key, 4)

    # Initialize parallel environments
    env_keys = jax.random.split(env_key, config.num_envs)
    obs, env_states = jax.vmap(env.reset, in_axes=(0, None, None))(env_keys, env.default_params, env.config)
    env_states = env_states.replace(obs=obs)  # Ensure obs is in state

    agent_state = agent.init(agent_key, obs[0], agent.default_params)  # Init with one sample obs

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(buffer_size=config.buffer_size)
    # A sample transition is (obs, action, reward, next_obs, done)
    # We need to get the correct shapes and dtypes for initialization.
    sample_obs = obs[0]
    # For a discrete action space, the action is a single integer.
    sample_action = jnp.array(0, dtype=jnp.int32)
    sample_reward = jnp.array(0.0, dtype=jnp.float32)
    sample_done = jnp.array(0.0, dtype=jnp.float32)

    sample_transition = (
        sample_obs,
        sample_action,
        sample_reward,
        sample_obs,  # next_obs has the same shape as obs
        sample_done,
    )
    buffer_state = replay_buffer.init(sample_transition)

    # Create the jitted training function
    train_step_fn = _make_train_step_fn(agent, env, replay_buffer, config.num_envs)

    # 2. OUTER TRAINING LOOP (in Python)
    print("Starting training...")
    for step in range(config.total_timesteps // config.num_envs):
        key, agent_state, env_states, buffer_state, metrics = train_step_fn(
            key=key,
            agent_state=agent_state,
            env_states=env_states,
            buffer_state=buffer_state,
            batch_size=config.batch_size,
        )

        # Log metrics (this happens outside the JIT loop)
        if step % config.log_frequency == 0:
            # Aggregate returns from all envs that finished in this step
            finished_returns = metrics["episode_return"][metrics["episode_return"] != 0]
            if len(finished_returns) > 0:
                avg_return = finished_returns.mean()
                print(f"Step: {step * config.num_envs}, Avg Return: {avg_return:.2f}, Loss: {metrics['loss']:.4f}")

    print("Training finished.")


def _get_factory_kwargs(config_obj: Any) -> dict:
    """Converts a dataclass config object to a dict for factory functions."""
    kwargs = asdict(config_obj)
    kwargs.pop("name")  # The name is used for lookup, not as a parameter
    return kwargs
