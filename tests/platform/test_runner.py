import math
from dataclasses import dataclass
from typing import Any, Callable, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from aion.agents.agent import Agent
from aion.configs.default import AgentConfig, Config, EnvConfig, RunConfig, WandbConfig
from aion.core.replay_buffer import ReplayBuffer, ReplayBufferState
from aion.core.spaces import Box, Space
from aion.core.types import BaseModel
from aion.envs.environment import Environment
from aion.platform import runner
from aion.platform.runner import TrainingEnvState


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
    """Reset the deterministic environment to a zero state and observation."""
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
    """Advance the test environment by one step with deterministic transitions."""
    next_position = state.position + jnp.asarray(params.increment, dtype=jnp.float32)
    next_step = state.step + jnp.array(1, dtype=state.step.dtype)
    done = next_step >= jnp.array(config.max_steps, dtype=next_step.dtype)
    obs = jnp.array([next_position], dtype=jnp.float32)
    reward = state.position * jnp.asarray(params.reward_scalar, dtype=jnp.float32)
    done_flag = jnp.where(done, jnp.array(1.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32))
    next_state = _TestEnvState(position=next_position, step=next_step)
    return obs, next_state, reward, done_flag, {}


def _test_get_action_space(config: _TestEnvConfig) -> Box:
    """Return the bounded one-dimensional Box action space."""
    return Box(low=-1.0, high=1.0, shape=(1,), dtype=jnp.float32)


def _test_get_obs_shape(_config: _TestEnvConfig) -> tuple[int, ...]:
    """Expose the observation shape expected by the environment."""
    return (1,)


def _make_test_env() -> Environment:
    """Instantiate the deterministic JAX environment used in tests."""
    return Environment(
        step=_test_env_step,
        reset=_test_env_reset,
        get_action_space=_test_get_action_space,
        get_obs_shape=_test_get_obs_shape,
        params=_TestEnvParams(),
        config=_TestEnvConfig(),
    )


def _agent_init(_key: chex.PRNGKey, _sample_obs: chex.Array, _params: _TestAgentParams) -> _TestAgentState:
    """Initialize the dummy agent state with a constant marker."""
    return _TestAgentState(marker=jnp.array(0.0, dtype=jnp.float32))


def _agent_select_action(
    _key: chex.PRNGKey,
    _obs: chex.Array,
    _state: _TestAgentState,
    params: _TestAgentParams,
) -> tuple[chex.Array, _TestAgentState]:
    """Emit a zero action and return an unchanged agent state."""
    action_space = cast(Box, params.action_space)
    action = jnp.zeros(action_space.shape, dtype=jnp.float32)
    return action, _state


def _agent_update(
    _key: chex.PRNGKey,
    state: _TestAgentState,
    batch,
    _params: _TestAgentParams,
) -> tuple[_TestAgentState, dict[str, chex.Array]]:
    """Produce a simple loss metric equal to the batch reward mean."""
    rewards = jnp.asarray(batch.reward, dtype=jnp.float32)
    metric = jnp.mean(rewards)
    new_marker = state.marker + jnp.array(1.0, dtype=jnp.float32)
    return _TestAgentState(marker=new_marker), {"loss": metric}


def _make_test_agent(action_space: Space) -> Agent:
    """Create the deterministic agent wrapper used for runner tests."""
    params = _TestAgentParams(action_space=action_space)
    return Agent(params=params, init=_agent_init, select_action=_agent_select_action, update=_agent_update)


def _create_config(*, wandb_enabled: bool = False, run_overrides: dict | None = None) -> Config:
    """Build a small Config object tailored for deterministic test runs."""
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
    """Compute deterministic evaluation return and length from env parameters."""
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
    """Prepare initial keys, states, and functions for exercising train loops."""
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
    """Inject deterministic agent and env factories into the registries."""
    import aion.agents
    import aion.envs

    monkeypatch.setitem(aion.envs.ENV_REGISTRY, "deterministic_env", lambda **_: _make_test_env())
    monkeypatch.setitem(
        aion.agents.AGENT_REGISTRY,
        "deterministic_agent",
        lambda *, action_space, **__: _make_test_agent(action_space),
    )


@pytest.fixture
def wandb_stub(monkeypatch):
    """Provide a stubbed W&B client that captures init, log, and finish calls."""
    stub = _WandbStub()
    monkeypatch.setattr(runner, "wandb", stub)
    monkeypatch.setattr(runner, "wandb_import_error", None, raising=False)
    return stub


@pytest.fixture(scope="module")
def training_setup_factory():
    """Factory fixture returning deterministic training harnesses."""

    def _factory(num_envs: int = 2) -> _TrainingSetup:
        return _create_training_setup(num_envs)

    return _factory


def test_drop_none_filters_empty_entries():
    """Ensure `_drop_none` prunes keys whose values are `None`."""
    data = {"project": "demo", "entity": None, "mode": "offline"}
    assert runner._drop_none(data) == {"project": "demo", "mode": "offline"}


def test_prepare_metrics_host_returns_host_arrays():
    """Verify device metrics become host NumPy arrays with scan truncation."""
    metrics = {"loss": jnp.arange(5, dtype=jnp.float32)}
    result = runner._prepare_metrics_host(metrics, steps_this_chunk=3)
    assert np.allclose(result["loss"], np.array([0.0, 1.0, 2.0], dtype=np.float32))
    assert isinstance(result["loss"], np.ndarray)


def test_prepare_metrics_host_handles_invalid_inputs():
    """Returns an empty dict when metrics input is invalid or chunk is zero."""
    assert runner._prepare_metrics_host({}, 1) == {}
    assert runner._prepare_metrics_host({"loss": jnp.ones(3)}, 0) == {}
    assert runner._prepare_metrics_host("not-a-dict", 2) == {}


def test_build_train_payload_uses_last_value():
    """`_build_train_payload` uses most recent scalar metric for logging."""
    metrics_host = {"loss": np.array([0.2, 0.4], dtype=np.float32)}
    payload = runner._build_train_payload(metrics_host)
    assert payload == {"train/loss": pytest.approx(0.4)}


def test_build_train_payload_expands_vector_metric():
    """Vector metrics are reduced to summary statistics for W&B payloads."""
    metrics_host = {"advantage": np.array([[1.0, 3.0], [2.0, 6.0]], dtype=np.float32)}
    payload = runner._build_train_payload(metrics_host)
    assert payload["train/advantage/mean"] == pytest.approx(4.0)
    assert payload["train/advantage/std"] == pytest.approx(2.0)
    assert payload["train/advantage/max"] == pytest.approx(6.0)
    assert payload["train/advantage/min"] == pytest.approx(2.0)


def test_summarize_metric_supports_scalar_and_bool():
    """Handle scalar values and boolean arrays when summarizing metrics."""
    scalar_result = runner._summarize_metric("test/", "value", np.array(3.5))
    assert scalar_result == {"test/value": pytest.approx(3.5)}

    bool_result = runner._summarize_metric("test/", "done", np.array([True, False]))
    assert bool_result["test/done/mean"] == pytest.approx(0.5)


def test_summarize_metric_rejects_non_numeric():
    """Non-numeric metrics should be ignored by `_summarize_metric`."""
    assert runner._summarize_metric("test/", "value", np.array(["a", "b"])) == {}


def test_tree_select_helpers():
    """Exercise tree masking utilities for scalar and vector masks."""
    mask_scalar = jnp.array(True)
    new_tree = {"value": jnp.array(2.0)}
    old_tree = {"value": jnp.array(-1.0)}
    result = runner._tree_select(mask_scalar, new_tree, old_tree)
    assert np.array(result["value"]) == pytest.approx(2.0)

    mask = jnp.array([True, False], dtype=jnp.bool_)
    new_value = jnp.array([5.0, 10.0])
    old_value = jnp.array([1.0, 1.0])
    expanded = runner._expand_mask(mask, target_ndim=2)
    assert expanded.shape == (2, 1)
    where_result = runner._where_mask(mask, new_value, old_value)
    np.testing.assert_allclose(where_result, np.array([5.0, 1.0]))
    mask_tree_result = runner._mask_tree(mask, {"value": new_value}, {"value": old_value})
    np.testing.assert_allclose(mask_tree_result["value"], np.array([5.0, 1.0]))


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


def test_make_chunk_runner_masks_inactive_iterations(training_setup_factory):
    """`_make_chunk_runner` must keep carry fixed for inactive scan slots."""
    state = training_setup_factory(num_envs=2)
    run_chunk = runner._make_chunk_runner(state.train_step_fn, batch_size=2)
    active_mask = jnp.array([True, False, False])
    (key_out, agent_state_out, env_state_out, buffer_state_out), metrics_history = run_chunk(
        (state.key, state.agent_state, state.training_env_state, state.buffer_state),
        active_mask,
    )

    assert key_out.shape == (2,)
    assert agent_state_out.marker.shape == ()
    assert float(np.array(agent_state_out.marker)) == pytest.approx(1.0)
    assert metrics_history["loss"].shape == (3,)
    assert metrics_history["loss"][1] == pytest.approx(0.0)
    expected_obs = np.array([[2.0], [2.0]], dtype=np.float32)
    assert np.array(env_state_out.obs).shape == state.training_env_state.obs.shape
    assert np.allclose(np.array(env_state_out.obs), expected_obs)
    assert buffer_state_out.position.shape == ()
    assert int(np.array(buffer_state_out.position)) == 2
    assert int(np.array(buffer_state_out.size)) == 2


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
    config = _create_config()
    orig_make_chunk_runner = runner._make_chunk_runner
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

    monkeypatch.setattr(runner, "_make_chunk_runner", instrumented_make_chunk_runner)
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
    wandb_run = runner._maybe_init_wandb(config)
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
    assert runner._maybe_init_wandb(config) is None


def test_maybe_init_wandb_raises_without_package(monkeypatch):
    """Enabled W&B must error if the package import is unavailable."""
    config = _create_config(wandb_enabled=True)
    monkeypatch.setattr(runner, "wandb", None)
    monkeypatch.setattr(runner, "wandb_import_error", ImportError("missing"), raising=False)
    with pytest.raises(RuntimeError):
        runner._maybe_init_wandb(config)


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
