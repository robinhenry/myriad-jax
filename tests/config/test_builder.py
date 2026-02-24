import pytest

from myriad.configs.builder import create_config, create_eval_config
from myriad.configs.default import Config, EvalConfig, config_to_eval_config


def test_create_config_basic():
    config = create_config(env="cartpole-control", agent="dqn")
    assert isinstance(config, Config)
    assert config.env.name == "cartpole-control"
    assert config.agent.name == "dqn"
    assert config.run.steps_per_env == 1000
    assert config.run.eval_max_steps == 500  # Default from cartpole ControlTaskConfig
    assert config.run.batch_size == 32  # Default from default.py


def test_create_config_overrides():
    config = create_config(
        env="cartpole-control",
        agent="dqn",
        steps_per_env=500,
        eval_max_steps=200,
        learning_rate=1e-4,  # Goes to agent
        num_envs=10,
    )
    assert config.run.steps_per_env == 500
    assert config.run.eval_max_steps == 200
    assert config.agent.learning_rate == 1e-4
    assert config.run.num_envs == 10


def test_create_config_dot_notation():
    config = create_config(
        env="cartpole-control", agent="dqn", **{"agent.hidden_dims": (32, 32), "run.scan_chunk_size": 128}
    )
    assert config.agent.hidden_dims == (32, 32)
    assert config.run.scan_chunk_size == 128


def test_create_eval_config():
    config = create_eval_config(env="cartpole-control", agent="random")
    assert isinstance(config, EvalConfig)
    assert config.env.name == "cartpole-control"
    assert config.agent.name == "random"
    assert config.run.eval_max_steps == 500


def test_on_policy_defaults():
    # PQN should get rollout_steps=2 by default in builder
    config = create_config(env="cartpole-control", agent="pqn")
    assert config.run.rollout_steps == 2


def test_off_policy_defaults():
    # DQN should NOT get rollout_steps
    config = create_config(env="cartpole-control", agent="dqn")
    assert config.run.rollout_steps is None


def test_epsilon_decay_fraction_preserved():
    # epsilon_decay_fraction is stored as-is in the config; resolution happens at
    # initialization time (initialize_environment_and_agent) so both the Python API
    # and YAML/Hydra paths go through a single conversion point.
    config = create_config(
        env="cartpole-control",
        agent="pqn",
        steps_per_env=1000,
        rollout_steps=10,
        epsilon_decay_fraction=0.2,
    )
    assert config.agent.epsilon_decay_fraction == 0.2
    assert not hasattr(config.agent, "epsilon_decay_steps")


def test_total_timesteps():
    config = create_config(env="cartpole-control", agent="dqn", num_envs=4, steps_per_env=500)
    assert config.run.total_timesteps == 2000


def test_kwarg_inferred_to_run_section():
    # buffer_size is a RunConfig field — passing it as a kwarg should infer it to run
    config = create_config(env="cartpole-control", agent="dqn", buffer_size=5000)
    assert config.run.buffer_size == 5000


def test_kwarg_as_nested_dict():
    # Nested dict for a known section key (e.g. wandb={"project": ...})
    config = create_config(env="cartpole-control", agent="dqn", wandb={"project": "test-project"})
    assert config.wandb.project == "test-project"


def test_config_to_eval_config():
    config = create_config(env="cartpole-control", agent="dqn")
    eval_cfg = config_to_eval_config(config)
    assert isinstance(eval_cfg, EvalConfig)
    assert eval_cfg.env.name == "cartpole-control"
    assert eval_cfg.agent.name == "dqn"


def test_on_policy_eval_frequency_alignment_warning():
    # rollout_steps=7 does not divide eval_frequency=100
    with pytest.warns(UserWarning, match="optimal boundary alignment"):
        create_config(env="cartpole-control", agent="pqn", rollout_steps=7, steps_per_env=1000)


def test_scan_chunk_size_rollout_mismatch_warning():
    # scan_chunk_size=15 is not divisible by rollout_steps=10
    with pytest.warns(UserWarning, match="not divisible"):
        create_config(
            env="cartpole-control",
            agent="pqn",
            rollout_steps=10,
            steps_per_env=1000,
            **{"run.scan_chunk_size": 15},
        )
