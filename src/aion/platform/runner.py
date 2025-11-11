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


def _make_eval_rollout_fn(agent: Agent, env: Environment, config: Config) -> Callable:
    """Factory to create a jitted evaluation rollout."""

    vmapped_env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None, None))
    vmapped_env_reset = jax.vmap(env.reset, in_axes=(0, None, None))

    def _broadcast_mask(mask: chex.Array, target: chex.Array) -> chex.Array:
        return jnp.broadcast_to(jnp.expand_dims(mask, axis=-1), target.shape)

    def _mask_tree(mask: chex.Array, new_tree: Any, old_tree: Any) -> Any:
        return jax.tree_util.tree_map(lambda new, old: jnp.where(mask, new, old), new_tree, old_tree)

    @jax.jit
    def eval_rollout(key: chex.PRNGKey, agent_state: AgentState) -> tuple[chex.PRNGKey, chex.Array, chex.Array]:
        n_rollouts = config.eval_rollouts

        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, n_rollouts)
        obs, env_states = vmapped_env_reset(reset_keys, env.params, env.config)

        episode_returns = jnp.zeros((n_rollouts,), dtype=jnp.float32)
        dones = jnp.zeros((n_rollouts,), dtype=bool)

        def body(carry, _):
            key, obs, env_states, episode_returns, dones = carry

            key, action_key = jax.random.split(key)
            action_keys = jax.random.split(action_key, n_rollouts)
            actions, _ = jax.vmap(agent.select_action, in_axes=(0, 0, 0, None))(
                action_keys, obs, agent_state, agent.params
            )

            key, step_key = jax.random.split(key)
            step_keys = jax.random.split(step_key, n_rollouts)
            next_obs, next_env_states, rewards, step_dones, _ = vmapped_env_step(
                step_keys, env_states, actions, env.params, env.config
            )

            active = jnp.logical_not(dones)
            episode_returns = episode_returns + rewards * active.astype(rewards.dtype)

            obs = jnp.where(_broadcast_mask(active, next_obs), next_obs, obs)
            env_states = _mask_tree(active, next_env_states, env_states)

            dones = jnp.logical_or(dones, step_dones.astype(bool))

            return (key, obs, env_states, episode_returns, dones), None

        initial_carry = (key, obs, env_states, episode_returns, dones)
        (key, _, _, episode_returns, dones), _ = jax.lax.scan(body, initial_carry, jnp.arange(config.eval_max_steps))

        return key, episode_returns, dones

    return eval_rollout


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

    # Create the jitted training and evaluation functions
    train_step_fn = _make_train_step_fn(agent, env, replay_buffer, config.num_envs)
    eval_rollout_fn = _make_eval_rollout_fn(agent, env, config)

    # 2. OUTER TRAINING LOOP
    for step in range(config.total_timesteps // config.num_envs):
        key, agent_state, training_env_states, buffer_state, metrics = train_step_fn(
            key=key,
            agent_state=agent_state,
            training_env_states=training_env_states,
            buffer_state=buffer_state,
            batch_size=config.batch_size,
        )

        # Periodically evaluate the policy and log returns
        if (step + 1) % config.eval_frequency == 0:
            key, eval_key = jax.random.split(key)
            eval_key, eval_returns, eval_dones = eval_rollout_fn(eval_key, agent_state)
            eval_returns = jax.device_get(eval_returns)
            eval_dones = jax.device_get(eval_dones)
            mean_return = float(eval_returns.mean())
            global_step = (step + 1) * config.num_envs
            print(f"[eval] step={global_step} mean_episode_return={mean_return:.3f}")

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
