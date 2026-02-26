"""Tests for myriad.platform.seed_eval."""

from myriad.configs.default import AgentConfig, Config, EnvConfig, RunConfig, WandbConfig
from myriad.platform.seed_eval import _make_seed_eval_config


def _base_config(seed: int = 0) -> Config:
    return Config(
        run=RunConfig(
            seed=seed,
            steps_per_env=4,
            num_envs=1,
            batch_size=1,
            buffer_size=4,
            scan_chunk_size=2,
            eval_frequency=1,
            eval_rollouts=1,
            eval_max_steps=1,
        ),
        agent=AgentConfig(name="dummy"),
        env=EnvConfig(name="dummy"),
        wandb=WandbConfig(enabled=True, project="my-project", entity="my-entity"),
    )


def test_make_seed_eval_config_overrides_seed():
    config = _make_seed_eval_config(_base_config(seed=0), seed=42, rank_group="g_rank0", mode="disabled", tags=())
    assert config.run.seed == 42


def test_make_seed_eval_config_sets_wandb_fields():
    config = _make_seed_eval_config(_base_config(), seed=1, rank_group="g_rank2", mode="offline", tags=("foo", "bar"))
    assert config.wandb.group == "g_rank2"
    assert config.wandb.job_type == "seed-eval"
    assert config.wandb.mode == "offline"
    assert list(config.wandb.tags) == ["foo", "bar"]
    assert config.wandb.run_name is None


def test_make_seed_eval_config_preserves_project_and_entity():
    config = _make_seed_eval_config(_base_config(), seed=1, rank_group="g_rank0", mode="disabled", tags=())
    assert config.wandb.project == "my-project"
    assert config.wandb.entity == "my-entity"
