"""Configuration testing and validation for auto-tuning."""

import time

import jax
import jax.numpy as jnp

from myriad.agents.classical.random import make_agent as make_random_agent
from myriad.envs import make_env
from myriad.platform.runners import make_chunk_runner
from myriad.platform.steps import make_train_step_fn
from myriad.platform.types import TrainingEnvState
from myriad.utils import to_array
from myriad.utils.memory import estimate_pytree_memory_mb


def validate_config(
    env_name: str,
    num_envs: int,
    timeout_s: float = 15.0,
) -> tuple[bool, float | None, float | None]:
    """Validate if a configuration works and measure performance.

    Args:
        env_name: Environment name
        num_envs: Number of parallel environments
        timeout_s: Maximum time to wait for test

    Returns:
        Tuple of (success, throughput_steps_per_s, memory_gb)
        If failed, throughput and memory are None
    """
    try:
        # Create environment
        env = make_env(env_name)
        action_space = env.get_action_space(env.config)

        # Use random agent for profiling (minimal overhead)
        agent = make_random_agent(action_space=action_space)

        # Initialize environments
        key = jax.random.PRNGKey(42)
        reset_keys = jax.random.split(key, num_envs)
        vmapped_reset = jax.vmap(env.reset, in_axes=(0, None, None))
        obs, env_states = vmapped_reset(reset_keys, env.params, env.config)

        to_array_batch = jax.vmap(to_array)
        obs_array = to_array_batch(obs)

        training_state = TrainingEnvState(env_state=env_states, obs=obs_array)
        agent_state = agent.init(key, obs_array[0], agent.params)  # type: ignore[arg-type]

        # Measure memory
        memory_mb = estimate_pytree_memory_mb(training_state) + estimate_pytree_memory_mb(agent_state)

        # Create chunk runner using platform primitives
        step_fn = make_train_step_fn(agent, env, None, num_envs)
        run_chunk = make_chunk_runner(step_fn, batch_size=1)

        # Test 10 steps
        num_steps = 10
        initial_carry = (key, agent_state, training_state, None)
        active_mask = jnp.ones((num_steps,), dtype=jnp.bool_)

        # Warmup (compilation)
        start = time.time()
        final_carry, _ = run_chunk(initial_carry, active_mask)
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, final_carry)

        if time.time() - start > timeout_s:
            return False, None, None

        # Quick timing run
        start = time.time()
        final_carry, _ = run_chunk(final_carry, active_mask)
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, final_carry)
        elapsed = time.time() - start

        # Calculate throughput
        total_steps = num_steps * num_envs
        throughput = total_steps / elapsed

        return True, throughput, memory_mb / 1024  # Convert to GB

    except Exception:
        # Catch OOM or any other errors
        return False, None, None
