"""Configuration testing and validation for auto-tuning."""

import time
from typing import Optional

import jax

from myriad.agents.classical.random import make_agent as make_random_agent
from myriad.envs import make_env
from myriad.platform.steps import make_train_step_fn
from myriad.platform.types import TrainingEnvState
from myriad.utils import to_array
from myriad.utils.memory import estimate_pytree_memory_mb


def validate_config(
    env_name: str,
    agent_name: str,
    num_envs: int,
    chunk_size: int,
    timeout_s: float = 15.0,
) -> tuple[bool, Optional[float], Optional[float]]:
    """Validate if a configuration works and measure performance.

    Args:
        env_name: Environment name
        agent_name: Agent name
        num_envs: Number of parallel environments
        chunk_size: Scan chunk size
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
        agent_state = agent.init(key, obs_array[0], agent.params)

        # Measure memory
        memory_mb = estimate_pytree_memory_mb(training_state) + estimate_pytree_memory_mb(agent_state)

        # Create step function
        step_fn = make_train_step_fn(agent, env, None, num_envs)

        # Run a few steps to test (with JIT compilation)
        def run_steps(key, agent_state, training_state):
            def body(carry, _):
                key, agent_state, training_state = carry
                key, new_agent_state, new_training_state, _, _ = step_fn(
                    key, agent_state, training_state, None, batch_size=1
                )
                return (key, new_agent_state, new_training_state), None

            carry = (key, agent_state, training_state)
            final_carry, _ = jax.lax.scan(body, carry, None, length=10)
            return final_carry

        jitted_run = jax.jit(run_steps)

        # Warmup (compilation)
        start = time.time()
        result = jitted_run(key, agent_state, training_state)
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, result)

        if time.time() - start > timeout_s:
            return False, None, None

        # Quick timing run
        start = time.time()
        result = jitted_run(key, agent_state, training_state)
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, result)
        elapsed = time.time() - start

        # Calculate throughput
        total_steps = 10 * num_envs
        throughput = total_steps / elapsed

        return True, throughput, memory_mb / 1024  # Convert to GB

    except Exception:
        # Catch OOM or any other errors
        return False, None, None
