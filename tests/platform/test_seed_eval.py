"""Tests for myriad.platform.seed_eval."""

import warnings
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# run_seed_eval tests
# ---------------------------------------------------------------------------


def test_run_seed_eval_calls_train_for_each_seed_and_rank():
    """run_seed_eval should call train_and_evaluate for every seed × top-k combination."""
    from myriad.platform.seed_eval import run_seed_eval

    mock_wandb_run = MagicMock()

    with (
        patch("myriad.platform.seed_eval.fetch_top_k_runs", return_value=[mock_wandb_run, mock_wandb_run]),
        patch("myriad.platform.seed_eval.config_from_wandb_run", return_value=_base_config()),
        patch("myriad.platform.training.train_and_evaluate") as mock_train,
        patch("myriad.platform.seed_eval.wandb") as mock_wandb,
    ):
        mock_wandb.run = None  # no stale run

        run_seed_eval(
            "ent/proj/sweep1",
            top_k=2,
            seeds=[0, 1],
            metric="eval/return/mean",
            group="test_group",
            maximize=True,
            mode="disabled",
            tags=(),
        )

    # 2 top-k configs × 2 seeds = 4 training calls
    assert mock_train.call_count == 4


def test_run_seed_eval_assigns_correct_rank_groups():
    """run_seed_eval should assign '{group}_rank{rank}' to each config."""
    from myriad.platform.seed_eval import run_seed_eval

    mock_wandb_run = MagicMock()
    captured_configs: list[Config] = []

    def _capture_train(config):
        captured_configs.append(config)

    with (
        patch("myriad.platform.seed_eval.fetch_top_k_runs", return_value=[mock_wandb_run, mock_wandb_run]),
        patch("myriad.platform.seed_eval.config_from_wandb_run", return_value=_base_config()),
        patch("myriad.platform.training.train_and_evaluate", side_effect=_capture_train),
        patch("myriad.platform.seed_eval.wandb") as mock_wandb,
    ):
        mock_wandb.run = None

        run_seed_eval(
            "ent/proj/sweep1",
            top_k=2,
            seeds=[0],
            metric="eval/return/mean",
            group="mygroup",
            maximize=True,
            mode="disabled",
            tags=(),
        )

    assert captured_configs[0].wandb.group == "mygroup_rank0"
    assert captured_configs[1].wandb.group == "mygroup_rank1"


def test_run_seed_eval_warns_and_closes_stale_run():
    """run_seed_eval should warn and call wandb.finish() when a stale run is detected."""
    from myriad.platform.seed_eval import run_seed_eval

    mock_wandb_run = MagicMock()

    with (
        patch("myriad.platform.seed_eval.fetch_top_k_runs", return_value=[mock_wandb_run]),
        patch("myriad.platform.seed_eval.config_from_wandb_run", return_value=_base_config()),
        patch("myriad.platform.training.train_and_evaluate"),
        patch("myriad.platform.seed_eval.wandb") as mock_wandb,
    ):
        mock_wandb.run = MagicMock()  # non-None = stale run

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_seed_eval(
                "ent/proj/sweep1",
                top_k=1,
                seeds=[0],
                metric="eval/return/mean",
                group="test_group",
                maximize=True,
                mode="disabled",
                tags=(),
            )

    mock_wandb.finish.assert_called_once()
    assert any("Stale W&B run" in str(w.message) for w in caught)
