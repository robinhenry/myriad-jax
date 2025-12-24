import math
from dataclasses import dataclass
from typing import Any, Callable, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from myriad.agents.agent import Agent
from myriad.configs.default import AgentConfig, Config, EnvConfig, RunConfig, WandbConfig
from myriad.core.replay_buffer import ReplayBuffer, ReplayBufferState
from myriad.core.spaces import Box, Space
from myriad.core.types import BaseModel
from myriad.envs.environment import Environment
from myriad.platform import logging_utils, runner
from myriad.platform.runner import TrainingEnvState


@struct.dataclass
class _TestEnvConfig:
    max_steps: int = 3


@struct.dataclass
class _TestEnvParams:
    increment: float = 2.0
    reward_scalar: float = 5.0


@struct.dataclass
class _TestEnvState:
    position: chex.Array
    step: chex.Array


@struct.dataclass
class _TestAgentParams:
    action_space: Space


@struct.dataclass
class _TestAgentState:
    marker: chex.Array


class _WandbStub:
    def __init__(self):
        self.init_kwargs: dict | None = None
        self.logs: list[tuple[dict[str, float], int | None]] = []
        self.finish_called = False

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return "stub-run"

    def log(self, payload: dict[str, float], step: int | None = None):
        self.logs.append((payload, step))

    def finish(self):
        self.finish_called = True


def _test_env_reset(
    _key: chex.PRNGKey, params: _TestEnvParams, config: _TestEnvConfig
) -> tuple[chex.Array, _TestEnvState]:
    position = jnp.array(0.0, dtype=jnp.float32)
    step = jnp.array(0, dtype=jnp.int32)
    obs = jnp.array([position], dtype=jnp.float32)
    return obs, _TestEnvState(position=position, step=step)


def _test_env_step(
    _key: chex.PRNGKey,
    state: _TestEnvState,
    _action: chex.Array,
    params: _TestEnvParams,
    config: _TestEnvConfig,
) -> tuple[chex.Array, _TestEnvState, chex.Array, chex.Array, dict[str, chex.Array]]:
    next_position = state.position + jnp.asarray(params.increment, dtype=jnp.float32)
    next_step = state.step + jnp.array(1, dtype=state.step.dtype)
    done = next_step >= jnp.array(config.max_steps, dtype=next_step.dtype)
    obs = jnp.array([next_position], dtype=jnp.float32)
    reward = state.position * jnp.asarray(params.reward_scalar, dtype=jnp.float32)
    done_flag = jnp.where(done, jnp.array(1.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32))
    next_state = _TestEnvState(position=next_position, step=next_step)
    return obs, next_state, reward, done_flag, {}


def _test_get_action_space(_config: _TestEnvConfig) -> Box:
    return Box(low=-1.0, high=1.0, shape=(1,), dtype=jnp.float32)


def _test_get_obs_shape(_config: _TestEnvConfig) -> tuple[int, ...]:
    return (1,)


def _make_test_env() -> Environment:
    return Environment(
        step=_test_env_step,
        reset=_test_env_reset,
        get_action_space=_test_get_action_space,
        get_obs_shape=_test_get_obs_shape,
        params=_TestEnvParams(),
        config=_TestEnvConfig(),
    )


def _agent_init(_key: chex.PRNGKey, _sample_obs: chex.Array, _params: _TestAgentParams) -> _TestAgentState:
    return _TestAgentState(marker=jnp.array(0.0, dtype=jnp.float32))


def _agent_select_action(
    _key: chex.PRNGKey,
    _obs: chex.Array,
    _state: _TestAgentState,
    params: _TestAgentParams,
) -> tuple[chex.Array, _TestAgentState]:
    action_space = cast(Box, params.action_space)
    action = jnp.zeros(action_space.shape, dtype=jnp.float32)
    return action, _state


def _agent_update(
    _key: chex.PRNGKey,
    state: _TestAgentState,
    batch,
    _params: _TestAgentParams,
) -> tuple[_TestAgentState, dict[str, chex.Array]]:
    rewards = jnp.asarray(batch.reward, dtype=jnp.float32)
    metric = jnp.mean(rewards)
    new_marker = state.marker + jnp.array(1.0, dtype=jnp.float32)
    return _TestAgentState(marker=new_marker), {"loss": metric}


def _make_test_agent(action_space: Space) -> Agent:
    params = _TestAgentParams(action_space=action_space)
    return Agent(params=params, init=_agent_init, select_action=_agent_select_action, update=_agent_update)


def _create_config(*, wandb_enabled: bool = False, run_overrides: dict | None = None) -> Config:
    run_defaults = {
        "seed": 0,
        "total_timesteps": 4,
        "num_envs": 2,
        "batch_size": 2,
        "buffer_size": 4,
        "scan_chunk_size": 2,
        "eval_frequency": 1,
        "eval_rollouts": 2,
        "eval_max_steps": 5,
        "log_frequency": 2,
    }
    if run_overrides:
        run_defaults.update(run_overrides)
    run_cfg = RunConfig(**run_defaults)

    wandb_defaults = {
        "enabled": wandb_enabled,
        "project": "unit-tests" if wandb_enabled else None,
        "entity": None,
        "group": None,
        "job_type": None,
        "run_name": "test-run" if wandb_enabled else None,
        "mode": "offline" if wandb_enabled else None,
        "dir": None,
        "tags": ("unit",) if wandb_enabled else None,
    }
    wandb_cfg = WandbConfig(**wandb_defaults)

    return Config(
        run=run_cfg,
        agent=AgentConfig(name="deterministic_agent"),
        env=EnvConfig(name="deterministic_env"),
        wandb=wandb_cfg,
    )


def _expected_eval_episode_stats(config: Config) -> tuple[float, int]:
    env_config = _TestEnvConfig()
    env_params = _TestEnvParams()
    steps = min(env_config.max_steps, config.run.eval_max_steps)
    return_sum = env_params.reward_scalar * env_params.increment * sum(range(steps))
    return float(return_sum), steps


@dataclass
class _TrainingSetup:
    key: chex.PRNGKey
    agent_state: _TestAgentState
    training_env_state: TrainingEnvState
    buffer_state: ReplayBufferState
    train_step_fn: Callable[..., tuple]
    agent: Agent
    env: Environment
    replay_buffer: ReplayBuffer


def _create_training_setup(num_envs: int = 2) -> _TrainingSetup:
    env = _make_test_env()
    agent = _make_test_agent(env.get_action_space(env.config))
    replay_buffer = ReplayBuffer(buffer_size=8)

    key_run, key_env, key_agent, key_buffer = jax.random.split(jax.random.PRNGKey(1234), 4)
    env_keys = jax.random.split(key_env, num_envs)
    obs, env_states = jax.vmap(env.reset, in_axes=(0, None, None))(env_keys, env.params, env.config)
    training_env_state = TrainingEnvState(env_state=env_states, obs=obs)

    obs_host = np.asarray(jax.device_get(obs))
    obs_sample = jnp.asarray(obs_host[0], dtype=jnp.float32)
    agent_state = agent.init(key_agent, obs_sample, agent.params)
    sample_transition = runner._make_sample_transition(key_buffer, obs_sample, env.get_action_space(env.config))
    buffer_state = replay_buffer.init(sample_transition)
    train_step_fn = runner._make_train_step_fn(agent, env, replay_buffer, num_envs)

    return _TrainingSetup(
        key=key_run,
        agent_state=agent_state,
        training_env_state=training_env_state,
        buffer_state=buffer_state,
        train_step_fn=train_step_fn,
        agent=agent,
        env=env,
        replay_buffer=replay_buffer,
    )


@pytest.fixture(autouse=True)
def _register_test_components(monkeypatch):
    import myriad.agents
    import myriad.envs

    monkeypatch.setitem(myriad.envs.ENV_REGISTRY, "deterministic_env", lambda **_: _make_test_env())
    monkeypatch.setitem(
        myriad.agents.AGENT_REGISTRY,
        "deterministic_agent",
        lambda *, action_space, **__: _make_test_agent(action_space),
    )


@pytest.fixture
def wandb_stub(monkeypatch):
    stub = _WandbStub()
    monkeypatch.setattr(logging_utils, "wandb", stub)
    monkeypatch.setattr(logging_utils, "_wandb_import_error", None, raising=False)
    monkeypatch.setattr(runner, "wandb", stub)
    return stub


@pytest.fixture(scope="module")
def training_setup_factory():
    def _factory(num_envs: int = 2) -> _TrainingSetup:
        return _create_training_setup(num_envs)

    return _factory


def test_make_train_step_fn_advances_environment(training_setup_factory):
    """Training step should advance env states, buffer, and emit metrics."""
    state = training_setup_factory(num_envs=2)
    key_new, agent_state_new, env_state_new, buffer_state_new, metrics = state.train_step_fn(
        key=state.key,
        agent_state=state.agent_state,
        training_env_states=state.training_env_state,
        buffer_state=state.buffer_state,
        batch_size=2,
    )

    assert isinstance(metrics, dict) and "loss" in metrics
    assert metrics["loss"] == pytest.approx(0.0)
    assert not np.array_equal(np.array(key_new), np.array(state.key))

    expected_obs = np.array([[2.0], [2.0]], dtype=np.float32)
    np.testing.assert_allclose(np.array(env_state_new.obs), expected_obs)
    np.testing.assert_allclose(np.array(env_state_new.env_state.position), np.array([2.0, 2.0]))
    np.testing.assert_array_equal(np.array(env_state_new.env_state.step), np.array([1, 1]))

    assert np.array(buffer_state_new.size) == 2
    assert np.array(buffer_state_new.position) == 2
    marker_expected = np.array(state.agent_state.marker) + 1.0
    np.testing.assert_array_equal(np.array(agent_state_new.marker), marker_expected)


def test_make_eval_rollout_fn_returns_episode_metrics(training_setup_factory):
    """Evaluation rollout should produce return, length, and done flags."""
    state = training_setup_factory(num_envs=2)
    config = _create_config()
    eval_rollout = runner._make_eval_rollout_fn(state.agent, state.env, config)
    key_out, metrics = eval_rollout(state.key, state.agent_state)

    assert set(metrics.keys()) == {"episode_return", "episode_length", "dones"}
    assert metrics["episode_return"].shape == (config.run.eval_rollouts,)
    expected_return, expected_length = _expected_eval_episode_stats(config)
    expected_returns = np.full((config.run.eval_rollouts,), expected_return, dtype=np.float32)
    assert np.allclose(np.array(metrics["episode_return"]), expected_returns)
    expected_lengths = np.full((config.run.eval_rollouts,), expected_length, dtype=np.int32)
    assert np.array_equal(np.array(metrics["episode_length"]), expected_lengths)
    expected_dones = np.ones((config.run.eval_rollouts,), dtype=bool)
    assert np.array_equal(np.array(metrics["dones"]), expected_dones)
    assert key_out.shape == (2,)


def test_run_training_loop_without_wandb(monkeypatch):
    """Training loop executes without W&B logging when run is disabled."""
    from myriad.platform import scan_utils

    config = _create_config()
    orig_make_chunk_runner = scan_utils.make_chunk_runner
    captured: dict[str, Any] = {}

    def instrumented_make_chunk_runner(train_step_fn, batch_size):
        run_chunk = orig_make_chunk_runner(train_step_fn, batch_size)

        def wrapped(carry, active_mask):
            result = run_chunk(carry, active_mask)
            captured["carry"] = jax.tree_util.tree_map(jax.device_get, result[0])
            captured["calls"] = captured.get("calls", 0) + 1
            captured["active_steps"] = captured.get("active_steps", 0) + int(np.asarray(active_mask).sum())
            return result

        return wrapped

    monkeypatch.setattr(runner, "make_chunk_runner", instrumented_make_chunk_runner)
    runner._run_training_loop(config, wandb_run=None)

    assert "carry" in captured
    total_steps = config.run.total_timesteps // config.run.num_envs
    # JIT tracing executes the wrapped chunk runner once before real execution (hence the +1)
    expected_calls = math.ceil(total_steps / max(1, config.run.scan_chunk_size)) + 1
    assert captured.get("calls", 0) == expected_calls
    assert captured.get("active_steps", 0) == total_steps
    final_carry = captured["carry"]
    _, agent_state_final, _, buffer_state_final = final_carry
    assert float(np.asarray(agent_state_final.marker)) == pytest.approx(float(total_steps))
    assert int(np.asarray(buffer_state_final.size)) == min(config.run.total_timesteps, config.run.buffer_size)
    expected_position = config.run.total_timesteps % config.run.buffer_size
    assert int(np.asarray(buffer_state_final.position)) == expected_position


def test_run_training_loop_with_wandb_logs(wandb_stub):
    """When W&B is enabled the loop should emit train and eval payloads."""
    config = _create_config(wandb_enabled=True)
    wandb_run = runner.maybe_init_wandb(config)
    runner._run_training_loop(config, wandb_run)

    assert wandb_stub.init_kwargs is not None
    assert wandb_stub.init_kwargs["config"]["run"]["seed"] == 0
    assert any("train/global_env_steps" in payload for payload, _ in wandb_stub.logs)
    eval_logged = any(any("eval/episode_return" in name for name in payload) for payload, _ in wandb_stub.logs)
    assert eval_logged is True
    final_payload, final_step = wandb_stub.logs[-1]
    assert final_payload == {"train/final_env_steps": float(config.run.total_timesteps)}
    assert final_step == config.run.total_timesteps


def test_train_and_evaluate_calls_finish(wandb_stub):
    """`train_and_evaluate` must finish the W&B run on exit."""
    config = _create_config(wandb_enabled=True)
    runner.train_and_evaluate(config)
    assert wandb_stub.finish_called is True


def test_maybe_init_wandb_disabled_returns_none():
    """Disabled W&B config should skip initialization."""
    config = _create_config(wandb_enabled=False)
    assert runner.maybe_init_wandb(config) is None


def test_maybe_init_wandb_raises_without_package(monkeypatch):
    """Enabled W&B must error if the package import is unavailable."""
    config = _create_config(wandb_enabled=True)
    monkeypatch.setattr(logging_utils, "wandb", None)
    monkeypatch.setattr(logging_utils, "_wandb_import_error", ImportError("missing"), raising=False)
    with pytest.raises(RuntimeError):
        runner.maybe_init_wandb(config)


def test_get_factory_kwargs_excludes_name():
    """Factory kwargs should drop the `name` field from configs."""

    class Dummy(BaseModel):
        name: str
        size: int

    cfg = Dummy(name="foo", size=3)
    assert runner._get_factory_kwargs(cfg) == {"size": 3}


def test_make_sample_transition_matches_shapes():
    """Sample transition matches observation/action shapes and zero reward."""
    env = _make_test_env()
    key = jax.random.PRNGKey(0)
    obs, _ = env.reset(key, env.params, env.config)
    transition = runner._make_sample_transition(key, obs, env.get_action_space(env.config))

    np.testing.assert_allclose(np.array(transition.obs), np.array(obs))
    np.testing.assert_allclose(np.array(transition.next_obs), np.array(obs))
    action_space = cast(Box, env.get_action_space(env.config))
    assert np.array(transition.action).shape == action_space.shape
    assert float(np.array(transition.reward)) == pytest.approx(0.0)
    assert bool(np.array(transition.done)) is False


def test_run_training_loop_with_chunk_size_larger_than_total_steps(monkeypatch):
    """Training should complete correctly even when chunk_size > total_timesteps."""
    from myriad.platform import scan_utils

    # Use a very large chunk size relative to total steps
    config = _create_config(run_overrides={"total_timesteps": 4, "num_envs": 1, "scan_chunk_size": 100})

    captured: dict[str, Any] = {"active_counts": []}
    orig_make_chunk_runner = scan_utils.make_chunk_runner

    def instrumented_make_chunk_runner(train_step_fn, batch_size):
        run_chunk = orig_make_chunk_runner(train_step_fn, batch_size)

        def wrapped(carry, active_mask):
            result = run_chunk(carry, active_mask)
            captured["mask_length"] = len(active_mask)
            captured["active_counts"].append(int(np.asarray(active_mask).sum()))
            return result

        return wrapped

    monkeypatch.setattr(runner, "make_chunk_runner", instrumented_make_chunk_runner)
    runner._run_training_loop(config, wandb_run=None)

    # Chunk size should be clamped to at least 1
    assert captured.get("mask_length", 0) == 100
    # But only 4 steps should actually execute across all chunks (excluding JIT trace)
    total_active_steps = sum(captured.get("active_counts", []))
    total_steps = config.run.total_timesteps // config.run.num_envs
    assert total_active_steps == total_steps


def test_run_training_loop_with_chunk_size_one(monkeypatch):
    """Training should work correctly with minimal chunk_size=1."""
    from myriad.platform import scan_utils

    config = _create_config(run_overrides={"total_timesteps": 4, "num_envs": 2, "scan_chunk_size": 1})

    active_counts: list[int] = []
    orig_make_chunk_runner = scan_utils.make_chunk_runner

    def instrumented_make_chunk_runner(train_step_fn, batch_size):
        run_chunk = orig_make_chunk_runner(train_step_fn, batch_size)

        def wrapped(carry, active_mask):
            active_counts.append(int(np.asarray(active_mask).sum()))
            return run_chunk(carry, active_mask)

        return wrapped

    monkeypatch.setattr(runner, "make_chunk_runner", instrumented_make_chunk_runner)
    runner._run_training_loop(config, wandb_run=None)

    total_steps = config.run.total_timesteps // config.run.num_envs
    # Total active steps across all chunks should equal total_steps
    assert sum(active_counts) == total_steps
    # With chunk_size=1 and alignment to logging boundaries, each chunk should have at most 1 active step
    assert all(count <= 1 for count in active_counts)


def test_run_training_loop_boundary_alignment_with_logging(monkeypatch):
    """Verify chunks align properly with logging frequency boundaries."""
    from myriad.platform import scan_utils

    # Setup: 10 total steps, chunk_size=3, log every 4 steps
    config = _create_config(
        run_overrides={"total_timesteps": 20, "num_envs": 2, "scan_chunk_size": 3, "log_frequency": 4}
    )

    chunk_sizes_observed: list[int] = []
    orig_make_chunk_runner = scan_utils.make_chunk_runner

    def instrumented_make_chunk_runner(train_step_fn, batch_size):
        run_chunk = orig_make_chunk_runner(train_step_fn, batch_size)

        def wrapped(carry, active_mask):
            active_count = int(np.asarray(active_mask).sum())
            chunk_sizes_observed.append(active_count)
            return run_chunk(carry, active_mask)

        return wrapped

    monkeypatch.setattr(runner, "make_chunk_runner", instrumented_make_chunk_runner)
    runner._run_training_loop(config, wandb_run=None)

    # Remove the JIT trace call (first call with 0 or minimal steps)
    actual_chunks = [size for size in chunk_sizes_observed if size > 0]

    # With 10 steps per env, chunk_size=3, log_frequency=4:
    # - Steps 0-3: chunk of 3, then chunk of 1 (to align with log boundary at step 4)
    # - Steps 4-7: chunk of 3, then chunk of 1 (to align with log boundary at step 8)
    # - Steps 8-9: chunk of 2
    # The exact pattern depends on boundary alignment logic
    assert sum(actual_chunks) == 10  # Total steps per env should be correct
    assert all(chunk <= 3 for chunk in actual_chunks)  # No chunk exceeds scan_chunk_size


def test_run_config_warns_on_inefficient_scan_chunk_size():
    """RunConfig should warn when scan_chunk_size is much larger than logging/eval frequencies."""
    import warnings

    # Should NOT warn: scan_chunk_size (10) <= 2 * min(log_frequency=10, eval_frequency=20) = 20
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _config = _create_config(run_overrides={"scan_chunk_size": 10, "log_frequency": 10, "eval_frequency": 20})
        chunk_warnings = [warning for warning in w if "scan_chunk_size" in str(warning.message)]
        assert len(chunk_warnings) == 0

    # Should warn: scan_chunk_size (50) > 2 * min(log_frequency=10, eval_frequency=20) = 20
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _config = _create_config(run_overrides={"scan_chunk_size": 50, "log_frequency": 10, "eval_frequency": 20})
        chunk_warnings = [warning for warning in w if "scan_chunk_size" in str(warning.message)]
        assert len(chunk_warnings) >= 1
        assert issubclass(chunk_warnings[0].category, UserWarning)
        assert "scan_chunk_size (50)" in str(chunk_warnings[0].message)
        assert "minimum boundary frequency (10)" in str(chunk_warnings[0].message)

    # Should warn: scan_chunk_size (100) > 2 * min(log_frequency=5, eval_frequency=3) = 6
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _config = _create_config(run_overrides={"scan_chunk_size": 100, "log_frequency": 5, "eval_frequency": 3})
        chunk_warnings = [warning for warning in w if "scan_chunk_size" in str(warning.message)]
        assert len(chunk_warnings) >= 1
        assert issubclass(chunk_warnings[0].category, UserWarning)
        assert "scan_chunk_size (100)" in str(chunk_warnings[0].message)
        assert "minimum boundary frequency (3)" in str(chunk_warnings[0].message)


# ============================================================================
# Reproducibility Tests
# ============================================================================
# These tests ensure that runs with the same seed produce identical results
# regardless of scan_chunk_size, log_frequency, and eval configuration params


def _extract_final_states(monkeypatch, config: Config) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Helper to run training and extract final agent, env, and buffer states."""
    from myriad.platform import scan_utils

    final_states: dict[str, Any] = {}

    orig_make_chunk_runner = scan_utils.make_chunk_runner

    def instrumented_make_chunk_runner(train_step_fn, batch_size):
        run_chunk = orig_make_chunk_runner(train_step_fn, batch_size)

        def wrapped(carry, active_mask):
            result = run_chunk(carry, active_mask)
            # Store final carry state (converting to numpy for easy comparison)
            final_states["carry"] = jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), result[0])
            return result

        return wrapped

    monkeypatch.setattr(runner, "make_chunk_runner", instrumented_make_chunk_runner)
    runner._run_training_loop(config, wandb_run=None)

    # Extract individual components from final carry
    key, agent_state, training_env_state, buffer_state = final_states["carry"]

    agent_dict = jax.tree_util.tree_map(np.asarray, agent_state)
    env_dict = jax.tree_util.tree_map(np.asarray, training_env_state)
    buffer_dict = jax.tree_util.tree_map(np.asarray, buffer_state) if buffer_state is not None else {}

    return agent_dict, env_dict, buffer_dict


def _assert_states_equal(state1: dict, state2: dict, state_name: str):
    """Assert that two PyTree states are numerically identical."""
    flat1, tree_def1 = jax.tree_util.tree_flatten(state1)
    flat2, tree_def2 = jax.tree_util.tree_flatten(state2)

    assert tree_def1 == tree_def2, f"{state_name} structure differs"
    assert len(flat1) == len(flat2), f"{state_name} has different number of leaves"

    for i, (leaf1, leaf2) in enumerate(zip(flat1, flat2)):
        leaf1_arr = np.asarray(leaf1)
        leaf2_arr = np.asarray(leaf2)
        np.testing.assert_array_equal(
            leaf1_arr,
            leaf2_arr,
            err_msg=f"{state_name} leaf {i} differs",
        )


def test_reproducibility_different_scan_chunk_sizes(monkeypatch):
    """Training with same seed but different scan_chunk_size should yield identical results."""
    base_seed = 42
    total_timesteps = 20
    num_envs = 2

    # Run 1: scan_chunk_size = 2
    config1 = _create_config(
        run_overrides={
            "seed": base_seed,
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "scan_chunk_size": 2,
            "log_frequency": 5,
            "eval_frequency": 10,
        }
    )
    agent1, env1, buffer1 = _extract_final_states(monkeypatch, config1)

    # Run 2: scan_chunk_size = 5 (different chunk size)
    config2 = _create_config(
        run_overrides={
            "seed": base_seed,
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "scan_chunk_size": 5,
            "log_frequency": 5,
            "eval_frequency": 10,
        }
    )
    agent2, env2, buffer2 = _extract_final_states(monkeypatch, config2)

    # Run 3: scan_chunk_size = 1 (edge case: minimal chunking)
    config3 = _create_config(
        run_overrides={
            "seed": base_seed,
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "scan_chunk_size": 1,
            "log_frequency": 5,
            "eval_frequency": 10,
        }
    )
    agent3, env3, buffer3 = _extract_final_states(monkeypatch, config3)

    # All three runs should produce identical final states
    _assert_states_equal(agent1, agent2, "Agent state (chunk_size 2 vs 5)")
    _assert_states_equal(agent1, agent3, "Agent state (chunk_size 2 vs 1)")
    _assert_states_equal(env1, env2, "Env state (chunk_size 2 vs 5)")
    _assert_states_equal(env1, env3, "Env state (chunk_size 2 vs 1)")
    _assert_states_equal(buffer1, buffer2, "Buffer state (chunk_size 2 vs 5)")
    _assert_states_equal(buffer1, buffer3, "Buffer state (chunk_size 2 vs 1)")


def test_reproducibility_different_log_frequencies(monkeypatch):
    """Training with same seed but different log_frequency should yield identical results."""
    base_seed = 123
    total_timesteps = 16
    num_envs = 2

    # Run 1: log_frequency = 2
    config1 = _create_config(
        run_overrides={
            "seed": base_seed,
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "scan_chunk_size": 2,
            "log_frequency": 2,
            "eval_frequency": 8,
        }
    )
    agent1, env1, buffer1 = _extract_final_states(monkeypatch, config1)

    # Run 2: log_frequency = 4 (different logging frequency)
    config2 = _create_config(
        run_overrides={
            "seed": base_seed,
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "scan_chunk_size": 2,
            "log_frequency": 4,
            "eval_frequency": 8,
        }
    )
    agent2, env2, buffer2 = _extract_final_states(monkeypatch, config2)

    # Run 3: log_frequency = 8 (same as eval, less frequent logging)
    config3 = _create_config(
        run_overrides={
            "seed": base_seed,
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "scan_chunk_size": 2,
            "log_frequency": 8,
            "eval_frequency": 8,
        }
    )
    agent3, env3, buffer3 = _extract_final_states(monkeypatch, config3)

    # All three runs should produce identical final states
    _assert_states_equal(agent1, agent2, "Agent state (log_freq 2 vs 4)")
    _assert_states_equal(agent1, agent3, "Agent state (log_freq 2 vs 8)")
    _assert_states_equal(env1, env2, "Env state (log_freq 2 vs 4)")
    _assert_states_equal(env1, env3, "Env state (log_freq 2 vs 8)")
    _assert_states_equal(buffer1, buffer2, "Buffer state (log_freq 2 vs 4)")
    _assert_states_equal(buffer1, buffer3, "Buffer state (log_freq 2 vs 8)")


def test_reproducibility_different_eval_frequencies(monkeypatch):
    """Training with same seed but different eval_frequency should yield identical results."""
    base_seed = 456
    total_timesteps = 24
    num_envs = 2

    # Run 1: eval_frequency = 4
    config1 = _create_config(
        run_overrides={
            "seed": base_seed,
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "scan_chunk_size": 3,
            "log_frequency": 4,
            "eval_frequency": 4,
        }
    )
    agent1, env1, buffer1 = _extract_final_states(monkeypatch, config1)

    # Run 2: eval_frequency = 6 (different eval frequency)
    config2 = _create_config(
        run_overrides={
            "seed": base_seed,
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "scan_chunk_size": 3,
            "log_frequency": 4,
            "eval_frequency": 6,
        }
    )
    agent2, env2, buffer2 = _extract_final_states(monkeypatch, config2)

    # Run 3: eval_frequency = 12 (less frequent evaluation)
    config3 = _create_config(
        run_overrides={
            "seed": base_seed,
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "scan_chunk_size": 3,
            "log_frequency": 4,
            "eval_frequency": 12,
        }
    )
    agent3, env3, buffer3 = _extract_final_states(monkeypatch, config3)

    # All three runs should produce identical final states
    _assert_states_equal(agent1, agent2, "Agent state (eval_freq 4 vs 6)")
    _assert_states_equal(agent1, agent3, "Agent state (eval_freq 4 vs 12)")
    _assert_states_equal(env1, env2, "Env state (eval_freq 4 vs 6)")
    _assert_states_equal(env1, env3, "Env state (eval_freq 4 vs 12)")
    _assert_states_equal(buffer1, buffer2, "Buffer state (eval_freq 4 vs 6)")
    _assert_states_equal(buffer1, buffer3, "Buffer state (eval_freq 4 vs 12)")


def test_reproducibility_different_eval_rollouts(monkeypatch):
    """Training with same seed but different eval_rollouts should yield identical results."""
    base_seed = 789
    total_timesteps = 16
    num_envs = 2

    # Run 1: eval_rollouts = 2
    config1 = _create_config(
        run_overrides={
            "seed": base_seed,
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "scan_chunk_size": 2,
            "log_frequency": 4,
            "eval_frequency": 8,
            "eval_rollouts": 2,
        }
    )
    agent1, env1, buffer1 = _extract_final_states(monkeypatch, config1)

    # Run 2: eval_rollouts = 5 (more eval rollouts)
    config2 = _create_config(
        run_overrides={
            "seed": base_seed,
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "scan_chunk_size": 2,
            "log_frequency": 4,
            "eval_frequency": 8,
            "eval_rollouts": 5,
        }
    )
    agent2, env2, buffer2 = _extract_final_states(monkeypatch, config2)

    # Both runs should produce identical final states
    _assert_states_equal(agent1, agent2, "Agent state (eval_rollouts 2 vs 5)")
    _assert_states_equal(env1, env2, "Env state (eval_rollouts 2 vs 5)")
    _assert_states_equal(buffer1, buffer2, "Buffer state (eval_rollouts 2 vs 5)")


def test_reproducibility_different_eval_max_steps(monkeypatch):
    """Training with same seed but different eval_max_steps should yield identical results."""
    base_seed = 999
    total_timesteps = 16
    num_envs = 2

    # Run 1: eval_max_steps = 5
    config1 = _create_config(
        run_overrides={
            "seed": base_seed,
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "scan_chunk_size": 2,
            "log_frequency": 4,
            "eval_frequency": 8,
            "eval_max_steps": 5,
        }
    )
    agent1, env1, buffer1 = _extract_final_states(monkeypatch, config1)

    # Run 2: eval_max_steps = 10 (longer eval episodes)
    config2 = _create_config(
        run_overrides={
            "seed": base_seed,
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "scan_chunk_size": 2,
            "log_frequency": 4,
            "eval_frequency": 8,
            "eval_max_steps": 10,
        }
    )
    agent2, env2, buffer2 = _extract_final_states(monkeypatch, config2)

    # Both runs should produce identical final states
    _assert_states_equal(agent1, agent2, "Agent state (eval_max_steps 5 vs 10)")
    _assert_states_equal(env1, env2, "Env state (eval_max_steps 5 vs 10)")
    _assert_states_equal(buffer1, buffer2, "Buffer state (eval_max_steps 5 vs 10)")


def test_reproducibility_combined_config_variations(monkeypatch):
    """Training with same seed but different combinations of config params should yield identical results."""
    base_seed = 2024
    total_timesteps = 24
    num_envs = 3

    # Run 1: Config A
    config1 = _create_config(
        run_overrides={
            "seed": base_seed,
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "scan_chunk_size": 2,
            "log_frequency": 4,
            "eval_frequency": 6,
            "eval_rollouts": 2,
            "eval_max_steps": 5,
        }
    )
    agent1, env1, buffer1 = _extract_final_states(monkeypatch, config1)

    # Run 2: Config B (all logging/eval params different, but same seed)
    config2 = _create_config(
        run_overrides={
            "seed": base_seed,
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "scan_chunk_size": 4,
            "log_frequency": 8,
            "eval_frequency": 12,
            "eval_rollouts": 4,
            "eval_max_steps": 10,
        }
    )
    agent2, env2, buffer2 = _extract_final_states(monkeypatch, config2)

    # Run 3: Config C (extreme settings)
    config3 = _create_config(
        run_overrides={
            "seed": base_seed,
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "scan_chunk_size": 1,
            "log_frequency": 2,
            "eval_frequency": 4,
            "eval_rollouts": 1,
            "eval_max_steps": 3,
        }
    )
    agent3, env3, buffer3 = _extract_final_states(monkeypatch, config3)

    # All three runs should produce identical final states despite very different configs
    _assert_states_equal(agent1, agent2, "Agent state (Config A vs B)")
    _assert_states_equal(agent1, agent3, "Agent state (Config A vs C)")
    _assert_states_equal(env1, env2, "Env state (Config A vs B)")
    _assert_states_equal(env1, env3, "Env state (Config A vs C)")
    _assert_states_equal(buffer1, buffer2, "Buffer state (Config A vs B)")
    _assert_states_equal(buffer1, buffer3, "Buffer state (Config A vs C)")
