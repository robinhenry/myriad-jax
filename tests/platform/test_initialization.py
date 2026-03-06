"""Tests for shared initialization utilities."""

import jax.numpy as jnp
import pytest
from flax import struct

from myriad.agents.agent import Agent
from myriad.configs.default import AgentConfig, Config, EnvConfig, EvalConfig, EvalRunConfig, RunConfig, WandbConfig
from myriad.core.spaces import Box
from myriad.envs.environment import Environment
from myriad.platform.initialization import get_factory_kwargs, initialize_environment_and_agent


@struct.dataclass
class _EnvConfigWithDt:
    max_steps: int = 5
    dt: float = 0.05


@struct.dataclass
class _EnvParams:
    pass


@struct.dataclass
class _EnvState:
    step: jnp.ndarray


def _env_reset(key, params, config):
    obs = jnp.zeros((2,))
    return obs, _EnvState(step=jnp.array(0))


def _env_step(key, state, action, params, config):
    next_step = state.step + 1
    done = jnp.where(next_step >= config.max_steps, 1.0, 0.0)
    return jnp.zeros((2,)), _EnvState(step=next_step), jnp.array(1.0), done, {}


def _get_action_space(config):
    return Box(low=-1.0, high=1.0, shape=(1,), dtype=jnp.float32)


def _get_obs_shape(config):
    return (2,)


@struct.dataclass
class _AgentParams:
    action_space: Box


@struct.dataclass
class _AgentState:
    pass


def _agent_init(key, obs, params):
    return _AgentState()


def _agent_select_action(key, obs, state, params, deterministic=False):
    return jnp.zeros((1,)), state


def _agent_update(key, state, batch, params):
    return state, {}


@pytest.fixture(autouse=True)
def _register(monkeypatch):
    from myriad.agents import registration as agent_reg
    from myriad.envs import registration as env_reg

    monkeypatch.setitem(
        env_reg._ENV_REGISTRY,
        "test-env-with-dt",
        env_reg.EnvInfo(
            name="test-env-with-dt",
            make_fn=lambda **_: Environment(
                step=_env_step,
                reset=_env_reset,
                get_action_space=_get_action_space,
                get_obs_shape=_get_obs_shape,
                params=_EnvParams(),
                config=_EnvConfigWithDt(),
            ),
            config_cls=_EnvConfigWithDt,
        ),
    )
    monkeypatch.setitem(
        agent_reg._AGENT_REGISTRY,
        "test-agent-no-dt",
        agent_reg.AgentInfo(
            name="test-agent-no-dt",
            make_fn=lambda *, action_space, **__: Agent(
                params=_AgentParams(action_space=action_space),
                init=_agent_init,
                select_action=_agent_select_action,
                update=_agent_update,
            ),
        ),
    )


def _make_config():
    return EvalConfig(
        run=EvalRunConfig(seed=0, eval_rollouts=1, eval_max_steps=5),
        agent=AgentConfig(name="test-agent-no-dt"),
        env=EnvConfig(name="test-env-with-dt"),
        wandb=WandbConfig(enabled=False),
    )


def test_dt_injected_from_env_config():
    """initialize_environment_and_agent should inject dt from env config into agent kwargs."""
    # We verify this doesn't raise — the dt injection path is exercised
    config = _make_config()
    env, agent, action_space = initialize_environment_and_agent(config)
    assert env is not None
    assert agent is not None


def test_get_factory_kwargs_removes_name():
    config = AgentConfig(name="random")
    kwargs = get_factory_kwargs(config)
    assert "name" not in kwargs


def test_epsilon_decay_fraction_resolved(monkeypatch):
    """epsilon_decay_fraction is converted to epsilon_decay_steps at init time."""
    from myriad.agents import registration as agent_reg

    captured_kwargs: dict = {}

    def _capturing_make_fn(*, action_space, **kwargs):
        captured_kwargs.update(kwargs)
        return Agent(
            params=_AgentParams(action_space=action_space),
            init=_agent_init,
            select_action=_agent_select_action,
            update=_agent_update,
        )

    monkeypatch.setitem(
        agent_reg._AGENT_REGISTRY,
        "test-capturing-agent",
        agent_reg.AgentInfo(name="test-capturing-agent", make_fn=_capturing_make_fn),
    )

    config = Config(
        run=RunConfig(steps_per_env=1000),
        agent=AgentConfig(name="test-capturing-agent", epsilon_decay_fraction=0.5),
        env=EnvConfig(name="test-env-with-dt"),
    )

    initialize_environment_and_agent(config)

    assert "epsilon_decay_fraction" not in captured_kwargs
    assert captured_kwargs["epsilon_decay_steps"] == 500


def test_lr_decay_fraction_resolved(monkeypatch):
    """lr_decay_fraction is converted to lr_decay_steps at init time."""
    from myriad.agents import registration as agent_reg

    captured_kwargs: dict = {}

    def _capturing_make_fn(*, action_space, **kwargs):
        captured_kwargs.update(kwargs)
        return Agent(
            params=_AgentParams(action_space=action_space),
            init=_agent_init,
            select_action=_agent_select_action,
            update=_agent_update,
        )

    monkeypatch.setitem(
        agent_reg._AGENT_REGISTRY,
        "test-capturing-agent-lr",
        agent_reg.AgentInfo(name="test-capturing-agent-lr", make_fn=_capturing_make_fn),
    )

    # lr_decay_steps = fraction * (steps_per_env / rollout_steps) * num_epochs * num_minibatches
    # = 0.5 * (1000/10) * 4 * 2 = 0.5 * 100 * 8 = 400
    config = Config(
        run=RunConfig(
            steps_per_env=1000,
            rollout_steps=10,
            eval_frequency=100,
            eval_rollouts=1,
            eval_max_steps=5,
        ),
        agent=AgentConfig(
            name="test-capturing-agent-lr",
            lr_decay_fraction=0.5,
            num_minibatches=2,
            num_epochs=4,
        ),
        env=EnvConfig(name="test-env-with-dt"),
    )

    initialize_environment_and_agent(config)

    assert "lr_decay_fraction" not in captured_kwargs
    assert captured_kwargs["lr_decay_steps"] == 400


def test_frame_stack_n_wraps_environment(monkeypatch):
    """frame_stack_n > 0 should apply the FrameStackWrapper to the environment."""
    from myriad.agents import registration as agent_reg

    monkeypatch.setitem(
        agent_reg._AGENT_REGISTRY,
        "test-agent-frame-stack",
        agent_reg.AgentInfo(
            name="test-agent-frame-stack",
            make_fn=lambda *, action_space, **__: Agent(
                params=_AgentParams(action_space=action_space),
                init=_agent_init,
                select_action=_agent_select_action,
                update=_agent_update,
            ),
        ),
    )

    config = EvalConfig(
        run=EvalRunConfig(seed=0, eval_rollouts=1, eval_max_steps=5),
        agent=AgentConfig(name="test-agent-frame-stack"),
        env=EnvConfig(name="test-env-with-dt", frame_stack_n=3),
        wandb=WandbConfig(enabled=False),
    )

    env, agent, _ = initialize_environment_and_agent(config)

    # Stacked obs shape should be (n_frames * base_obs_dim,) = (3 * 2,) = (6,)
    obs_shape = env.get_obs_shape(env.config)
    assert obs_shape == (6,)
