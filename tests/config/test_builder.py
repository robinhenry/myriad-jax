from myriad.configs.builder import create_config, create_eval_config
from myriad.configs.default import Config, EvalConfig


def test_create_config_basic():
    config = create_config(env="cartpole-control", agent="dqn")
    assert isinstance(config, Config)
    assert config.env.name == "cartpole-control"
    assert config.agent.name == "dqn"
    assert config.run.steps_per_env == 1000
    assert config.run.eval_max_steps == 500  # Default from cartpole ControlTaskConfig
    assert config.agent.batch_size == 32  # Default from default.py


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
