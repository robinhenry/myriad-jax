from dataclasses import asdict
from functools import partial
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np

try:
    import wandb  # type: ignore[import]
except ImportError:  # pragma: no cover - handled at runtime
    wandb = None  # type: ignore[assignment]
else:
    wandb_import_error = None
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


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    """Removes items with None values from a dictionary."""

    return {key: value for key, value in values.items() if value is not None}


def _summarize_metric(prefix: str, name: str, value: Any) -> dict[str, float]:
    """Expands an array-like metric into scalar statistics for logging."""

    try:
        array = np.asarray(value)
    except Exception:  # pragma: no cover - defensive guard
        return {}

    if array.size == 0:
        return {}

    if array.dtype == np.bool_:
        array = array.astype(np.float32)

    if not np.issubdtype(array.dtype, np.number):
        return {}

    array = np.asarray(array, dtype=np.float64)

    if array.ndim == 0:
        try:
            scalar = float(array.item())
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            return {}
        return {f"{prefix}{name}": scalar}

    return {
        f"{prefix}{name}/mean": float(np.nanmean(array)),
        f"{prefix}{name}/std": float(np.nanstd(array)),
        f"{prefix}{name}/max": float(np.nanmax(array)),
        f"{prefix}{name}/min": float(np.nanmin(array)),
    }


def _maybe_init_wandb(config: Config):
    """Initializes a Weights & Biases run when enabled in the config."""

    wandb_config = getattr(config, "wandb", None)
    if wandb_config is None or not wandb_config.enabled:
        return None

    if wandb is None:
        message = (
            "Weights & Biases tracking is enabled but the `wandb` package is not installed. "
            "Install it with `pip install wandb` to proceed."
        )
        raise RuntimeError(message) from wandb_import_error

    init_kwargs: dict[str, Any] = _drop_none(
        {
            "project": wandb_config.project,
            "entity": wandb_config.entity,
            "group": wandb_config.group,
            "job_type": wandb_config.job_type,
            "mode": wandb_config.mode,
            "dir": wandb_config.dir,
        }
    )

    if wandb_config.run_name:
        init_kwargs["name"] = wandb_config.run_name

    if wandb_config.tags:
        init_kwargs["tags"] = list(wandb_config.tags)

    init_kwargs["config"] = asdict(config)

    return wandb.init(**init_kwargs)


def _tree_select(mask: chex.Array, new_tree: Any, old_tree: Any) -> Any:
    """Selects between two pytrees using a scalar boolean mask."""

    return jax.tree_util.tree_map(lambda new, old: jax.lax.select(mask, new, old), new_tree, old_tree)


def _expand_mask(mask: chex.Array, target_ndim: int) -> chex.Array:
    """Reshapes a mask so it can broadcast to a target rank."""

    expand_dims = target_ndim - mask.ndim
    if expand_dims <= 0:
        return mask
    return mask.reshape(mask.shape + (1,) * expand_dims)


def _where_mask(mask: chex.Array, new_value: chex.Array, old_value: chex.Array) -> chex.Array:
    """Selects array values using a boolean mask, supporting broadcasting."""

    mask_bool = mask.astype(jnp.bool_)
    return jnp.where(_expand_mask(mask_bool, new_value.ndim), new_value, old_value)


def _mask_tree(mask: chex.Array, new_tree: Any, old_tree: Any) -> Any:
    """Selects between two pytrees using a (potentially vector) mask."""

    return jax.tree_util.tree_map(lambda new, old: _where_mask(mask, new, old), new_tree, old_tree)


def _make_train_step_fn(
    agent: Agent,
    env: Environment,
    replay_buffer: ReplayBuffer,
    num_envs: int,
) -> Callable:
    """Factory to create a jitted, vmapped training step function"""

    # Vmap the environment step and reset functions so we drive all envs in lockstep
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

        # Select actions using per-env keys while sharing agent state and params
        key, action_key = jax.random.split(key)
        action_keys = jax.random.split(action_key, num_envs)
        actions, agent_state = jax.vmap(agent.select_action, in_axes=(0, 0, None, None))(
            action_keys, last_obs, agent_state, agent.params
        )

        # Step environments in parallel and capture the resulting transitions
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, next_env_states, rewards, dones, _ = vmapped_env_step(
            step_keys, training_env_states.env_state, actions, env.params, env.config
        )
        dones_bool = dones.astype(jnp.bool_)

        # Store recent transitions and sample a learning batch from the buffer
        key, buffer_key = jax.random.split(key)
        transitions = Transition(last_obs, actions, rewards, next_obs, dones_bool)
        buffer_state, batch = replay_buffer.add_and_sample(buffer_state, transitions, batch_size, buffer_key)

        # Update the agent with the sampled batch and report metrics
        key, update_key = jax.random.split(key)
        agent_state, metrics = agent.update(update_key, agent_state, batch, agent.params)

        # Handle auto-resetting environments that are done by splitting keys once more
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, num_envs)

        # Reset only the environments that are done
        new_obs, new_env_states = vmapped_env_reset(reset_keys, env.params, env.config)

        # If done, use the new state, otherwise keep the existing one
        final_obs = _where_mask(dones_bool, new_obs, next_obs)
        final_env_states = _mask_tree(dones_bool, new_env_states, next_env_states)

        # Store the final states and observations as the new training environment state
        new_training_env_state = TrainingEnvState(env_state=final_env_states, obs=final_obs)

        return (
            key,
            agent_state,
            new_training_env_state,
            buffer_state,
            metrics,
        )

    return train_step


def _make_chunk_runner(train_step_fn: Callable, batch_size: int) -> Callable:
    """Creates a JIT-compiled function that executes a chunk of training steps via lax.scan."""

    def run_chunk(
        carry: tuple[chex.PRNGKey, AgentState, TrainingEnvState, ReplayBufferState],
        active_mask: chex.Array,
    ) -> tuple[tuple[chex.PRNGKey, AgentState, TrainingEnvState, ReplayBufferState], Any]:
        # Mask determines how many scan iterations should actually apply the update
        # Remaining iterations keep the carry unchanged to allow fixed-size scans
        active_mask = active_mask.astype(jnp.bool_)

        def body(
            carry: tuple[chex.PRNGKey, AgentState, TrainingEnvState, ReplayBufferState],
            active: chex.Array,
        ) -> tuple[tuple[chex.PRNGKey, AgentState, TrainingEnvState, ReplayBufferState], dict]:
            key, agent_state, training_env_states, buffer_state = carry
            key_new, agent_state_new, training_env_states_new, buffer_state_new, metrics = train_step_fn(
                key=key,
                agent_state=agent_state,
                training_env_states=training_env_states,
                buffer_state=buffer_state,
                batch_size=batch_size,
            )

            # Selectively commit the new state only when the mask signals an active step
            active_mask_scalar = jnp.asarray(active, dtype=jnp.bool_)
            key = jax.lax.select(active_mask_scalar, key_new, key)
            agent_state = _tree_select(active_mask_scalar, agent_state_new, agent_state)
            training_env_states = _tree_select(active_mask_scalar, training_env_states_new, training_env_states)
            buffer_state = _tree_select(active_mask_scalar, buffer_state_new, buffer_state)
            metrics = jax.tree_util.tree_map(
                lambda x: jax.lax.select(active_mask_scalar, x, jnp.zeros_like(x)),
                metrics,
            )

            return (key, agent_state, training_env_states, buffer_state), metrics

        return jax.lax.scan(body, carry, active_mask)

    return jax.jit(run_chunk)


def _make_eval_rollout_fn(agent: Agent, env: Environment, config: Config) -> Callable:
    """Factory to create a jitted evaluation rollout aligned with the training loop style."""

    vmapped_env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None, None))
    vmapped_env_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    num_eval_envs = config.eval_rollouts
    max_eval_steps = config.eval_max_steps

    @jax.jit
    def eval_rollout(key: chex.PRNGKey, agent_state: AgentState) -> tuple[chex.PRNGKey, dict[str, chex.Array]]:
        # Reset evaluation environments
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, num_eval_envs)
        obs, env_states = vmapped_env_reset(reset_keys, env.params, env.config)
        eval_env_state = TrainingEnvState(env_state=env_states, obs=obs)

        # Initialize metric accumulators
        episode_returns = jnp.zeros((num_eval_envs,), dtype=jnp.float32)
        episode_lengths = jnp.zeros((num_eval_envs,), dtype=jnp.int32)
        dones = jnp.zeros((num_eval_envs,), dtype=bool)
        max_steps = jnp.asarray(max_eval_steps, dtype=jnp.int32)

        def cond_fun(
            carry: tuple[chex.PRNGKey, TrainingEnvState, chex.Array, chex.Array, chex.Array, chex.Array]
        ) -> chex.Array:
            _, _, _, _, dones, step = carry
            continue_steps = step < max_steps
            incomplete = jnp.logical_not(jnp.all(dones))
            # Exit early once every evaluation episode terminates
            return jnp.logical_and(continue_steps, incomplete)

        def body_fun(
            carry: tuple[chex.PRNGKey, TrainingEnvState, chex.Array, chex.Array, chex.Array, chex.Array]
        ) -> tuple[chex.PRNGKey, TrainingEnvState, chex.Array, chex.Array, chex.Array, chex.Array]:
            key, env_state, returns, lengths, dones, step = carry

            # Drive each evaluation environment independently but under one loop
            key, action_key, step_key = jax.random.split(key, 3)
            action_keys = jax.random.split(action_key, num_eval_envs)
            actions, _ = jax.vmap(agent.select_action, in_axes=(0, 0, None, None))(
                action_keys, env_state.obs, agent_state, agent.params
            )

            step_keys = jax.random.split(step_key, num_eval_envs)
            next_obs, next_env_states, rewards, step_dones, _ = vmapped_env_step(
                step_keys, env_state.env_state, actions, env.params, env.config
            )

            step_dones = step_dones.astype(jnp.bool_)
            active = jnp.logical_not(dones)
            active_f32 = active.astype(rewards.dtype)

            returns = returns + rewards * active_f32
            lengths = lengths + active.astype(lengths.dtype)

            env_state = TrainingEnvState(
                env_state=_mask_tree(active, next_env_states, env_state.env_state),
                obs=_where_mask(active, next_obs, env_state.obs),
            )

            dones = jnp.logical_or(dones, step_dones)
            step = step + jnp.array(1, dtype=step.dtype)

            return key, env_state, returns, lengths, dones, step

        # Run loop with early termination
        initial_step = jnp.array(0, dtype=jnp.int32)
        initial_carry = (key, eval_env_state, episode_returns, episode_lengths, dones, initial_step)
        final_carry = jax.lax.while_loop(cond_fun, body_fun, initial_carry)
        key, _, final_returns, final_lengths, final_dones, _ = final_carry

        # Package metrics for the caller
        metrics = {
            "episode_return": final_returns,
            "episode_length": final_lengths,
            "dones": final_dones,
        }

        return key, metrics

    return eval_rollout


def _run_training_loop(config: Config, wandb_run: Any) -> None:
    """Executes the training loop and emits host-side and W&B logs."""

    # Initialize everything
    key = jax.random.PRNGKey(config.seed)
    key, env_key, agent_key, buffer_key = jax.random.split(key, 4)

    # Create the environment
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

    # Build jitted execution primitives
    train_step_fn = _make_train_step_fn(agent, env, replay_buffer, config.num_envs)
    eval_rollout_fn = _make_eval_rollout_fn(agent, env, config)

    chunk_size = max(1, config.scan_chunk_size)
    run_chunk_fn = _make_chunk_runner(train_step_fn, config.batch_size)

    total_steps = config.total_timesteps // config.num_envs
    log_frequency = config.log_frequency
    eval_frequency = config.eval_frequency

    steps_completed = 0
    while steps_completed < total_steps:
        remaining_steps = total_steps - steps_completed

        def _steps_until_boundary(current_step: int, frequency: int) -> int:
            if frequency <= 0:
                return chunk_size
            remainder = current_step % frequency
            return frequency if remainder == 0 else frequency - remainder

        # Determine how many scan iterations to run before the next logging or eval boundary
        steps_to_log = _steps_until_boundary(steps_completed, log_frequency)
        steps_to_eval = _steps_until_boundary(steps_completed, eval_frequency)
        steps_this_chunk = min(chunk_size, remaining_steps, steps_to_log, steps_to_eval)
        active_mask = (jnp.arange(chunk_size) < steps_this_chunk).astype(jnp.bool_)
        (key, agent_state, training_env_states, buffer_state), metrics_history = run_chunk_fn(
            (key, agent_state, training_env_states, buffer_state),
            active_mask,
        )

        steps_completed += steps_this_chunk

        # Host-side logging pulls the latest metrics emitted during the chunk
        metrics_host: dict[str, Any] = {}
        if isinstance(metrics_history, dict) and metrics_history and steps_this_chunk > 0:
            sliced_history = {name: values[:steps_this_chunk] for name, values in metrics_history.items()}
            metrics_host = {name: jax.device_get(values) for name, values in sliced_history.items()}

        should_log = log_frequency > 0 and steps_completed % log_frequency == 0 and metrics_host
        if should_log:
            global_step = steps_completed * config.num_envs
            metric_parts: list[str] = []
            train_payload: dict[str, float] = {}
            has_train_metrics = False

            for name, history in metrics_host.items():
                value = history[-1]
                try:
                    metric_parts.append(f"{name}={float(value):.4f}")
                except (TypeError, ValueError):
                    pass

                summary = _summarize_metric("train/", name, value)
                if summary:
                    has_train_metrics = True
                    train_payload.update(summary)

            if metric_parts:
                print(f"[train] step={global_step} {' '.join(metric_parts)}")

            if wandb_run is not None and wandb is not None and has_train_metrics:
                train_payload["train/global_env_steps"] = float(global_step)
                wandb.log(train_payload, step=global_step)

        # Periodically run evaluation rollouts without touching the training buffer
        if eval_frequency > 0 and steps_completed > 0 and steps_completed % eval_frequency == 0:
            key, eval_key = jax.random.split(key)
            eval_key, eval_metrics = eval_rollout_fn(eval_key, agent_state)
            key = eval_key
            eval_metrics_host = {name: jax.device_get(value) for name, value in eval_metrics.items()}
            eval_returns = eval_metrics_host.get("episode_return")
            eval_lengths = eval_metrics_host.get("episode_length")
            eval_dones = eval_metrics_host.get("dones")

            mean_return = float(eval_returns.mean()) if eval_returns is not None else float("nan")
            std_return = float(eval_returns.std()) if eval_returns is not None else float("nan")
            mean_length = float(eval_lengths.mean()) if eval_lengths is not None else float("nan")
            global_step = steps_completed * config.num_envs
            print(
                f"[eval] step={global_step} mean_episode_return={mean_return:.3f} "
                f"std_episode_return={std_return:.3f} mean_episode_length={mean_length:.1f}"
            )

            if wandb_run is not None and wandb is not None:
                eval_payload: dict[str, float] = {}
                if eval_returns is not None:
                    eval_payload.update(_summarize_metric("eval/", "episode_return", eval_returns))
                if eval_lengths is not None:
                    eval_payload.update(_summarize_metric("eval/", "episode_length", eval_lengths))
                if eval_dones is not None:
                    termination_rate = float(np.asarray(eval_dones, dtype=np.float32).mean())
                    eval_payload["eval/termination_rate"] = termination_rate

                if eval_payload:
                    eval_payload["eval/global_env_steps"] = float(global_step)
                    wandb.log(eval_payload, step=global_step)

            if eval_dones is not None and not bool(eval_dones.all()):
                print("[eval] warning: some evaluation environments reached eval_max_steps without terminating.")

    total_env_steps = steps_completed * config.num_envs
    if wandb_run is not None and wandb is not None:
        wandb.log({"train/final_env_steps": float(total_env_steps)}, step=total_env_steps)

    print("Training finished.")


def train_and_evaluate(config: Config):
    """
    Main entry point for a training run.
    Initializes everything and runs the outer training loop.
    """
    wandb_run = _maybe_init_wandb(config)

    try:
        _run_training_loop(config, wandb_run)
    finally:
        if wandb_run is not None and wandb is not None:
            wandb.finish()


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
    sample_done = jnp.array(False, dtype=jnp.bool_)

    return Transition(
        sample_obs,
        sample_action,
        sample_reward,
        sample_obs,  # next_obs has the same shape as obs
        sample_done,
    )
