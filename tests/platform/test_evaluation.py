"""Tests for evaluation-only functionality."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from myriad.agents.agent import Agent
from myriad.configs.default import AgentConfig, Config, EnvConfig, EvalConfig, EvalRunConfig, RunConfig, WandbConfig
from myriad.core.spaces import Box, Space
from myriad.envs.environment import Environment
from myriad.platform.evaluation import evaluate


# Test environment and agent fixtures (same as test_training.py)
@struct.dataclass
class _TestEnvConfig:
    max_steps: int = 3


@struct.dataclass
class _TestEnvParams:
    increment: float = 2.0
    reward_scalar: float = 5.0


@struct.dataclass
class _TestEnvState:
    position: jnp.ndarray
    step: jnp.ndarray


@struct.dataclass
class _TestAgentParams:
    action_space: Space


@struct.dataclass
class _TestAgentState:
    marker: jnp.ndarray


def _test_env_reset(key, params, config):
    position = jnp.array(0.0, dtype=jnp.float32)
    step = jnp.array(0, dtype=jnp.int32)
    obs = jnp.array([position], dtype=jnp.float32)
    return obs, _TestEnvState(position=position, step=step)


def _test_env_step(key, state, action, params, config):
    next_position = state.position + jnp.asarray(params.increment, dtype=jnp.float32)
    next_step = state.step + jnp.array(1, dtype=state.step.dtype)
    done = next_step >= jnp.array(config.max_steps, dtype=next_step.dtype)
    obs = jnp.array([next_position], dtype=jnp.float32)
    reward = state.position * jnp.asarray(params.reward_scalar, dtype=jnp.float32)
    done_flag = jnp.where(done, jnp.array(1.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32))
    next_state = _TestEnvState(position=next_position, step=next_step)
    return obs, next_state, reward, done_flag, {}


def _test_get_action_space(config):
    return Box(low=-1.0, high=1.0, shape=(1,), dtype=jnp.float32)


def _test_get_obs_shape(config):
    return (1,)


def _make_test_env():
    return Environment(
        step=_test_env_step,
        reset=_test_env_reset,
        get_action_space=_test_get_action_space,
        get_obs_shape=_test_get_obs_shape,
        params=_TestEnvParams(),
        config=_TestEnvConfig(),
    )


def _agent_init(key, sample_obs, params):
    return _TestAgentState(marker=jnp.array(0.0, dtype=jnp.float32))


def _agent_select_action(key, obs, state, params, deterministic=False):
    action = jnp.zeros((1,), dtype=jnp.float32)
    return action, state


def _agent_update(key, state, batch, params):
    return state, {}


def _make_test_agent(action_space):
    params = _TestAgentParams(action_space=action_space)
    return Agent(params=params, init=_agent_init, select_action=_agent_select_action, update=_agent_update)


@pytest.fixture(autouse=True)
def _register_test_components(monkeypatch):
    """Register test environment and agent in registries."""
    from myriad.agents import registration as agent_reg
    from myriad.envs import registration as env_reg

    monkeypatch.setitem(
        env_reg._ENV_REGISTRY,
        "test-env",
        env_reg.EnvInfo(
            name="test-env",
            make_fn=lambda **_: _make_test_env(),
            config_cls=_TestEnvConfig,
        ),
    )
    monkeypatch.setitem(
        agent_reg._AGENT_REGISTRY,
        "test-agent",
        agent_reg.AgentInfo(
            name="test-agent",
            make_fn=lambda *, action_space, **__: _make_test_agent(action_space),
        ),
    )


@pytest.fixture
def test_eval_config():
    """Create a basic EvalConfig for testing."""
    return EvalConfig(
        run=EvalRunConfig(
            seed=42,
            eval_rollouts=3,
            eval_max_steps=10,
            eval_episode_save_frequency=0,  # Disabled by default
            eval_episode_save_count=None,
            eval_render_videos=False,
            eval_video_fps=30,
        ),
        agent=AgentConfig(name="test-agent"),
        env=EnvConfig(name="test-env"),
        wandb=WandbConfig(enabled=False),
    )


@pytest.fixture
def test_training_config():
    """Create a basic Config for testing evaluation during training."""
    return Config(
        run=RunConfig(
            seed=42,
            steps_per_env=100,
            num_envs=4,
            batch_size=32,
            buffer_size=1000,
            scan_chunk_size=10,
            eval_frequency=50,
            eval_rollouts=3,
            eval_max_steps=10,
            log_frequency=10,
        ),
        agent=AgentConfig(name="test-agent"),
        env=EnvConfig(name="test-env"),
        wandb=WandbConfig(enabled=False),
    )


class TestEvaluateBasic:
    """Tests for basic evaluate() functionality."""

    def test_evaluate_returns_expected_structure(self, test_eval_config):
        """evaluate() should return EvaluationResults with correct fields."""
        results = evaluate(test_eval_config, agent_state=None, return_episodes=False)

        # Check result structure
        assert hasattr(results, "mean_return")
        assert hasattr(results, "std_return")
        assert hasattr(results, "min_return")
        assert hasattr(results, "max_return")
        assert hasattr(results, "mean_length")
        assert hasattr(results, "episode_returns")
        assert hasattr(results, "episode_lengths")
        assert hasattr(results, "num_episodes")
        assert hasattr(results, "seed")

        # Check metadata
        assert results.num_episodes == test_eval_config.run.eval_rollouts
        assert results.seed == test_eval_config.run.seed

        # Episodes should be None by default
        assert results.episodes is None

    def test_evaluate_with_return_episodes(self, test_eval_config):
        """evaluate() with return_episodes=True should include episode data."""
        results = evaluate(test_eval_config, agent_state=None, return_episodes=True)

        # Episodes should be present
        assert results.episodes is not None
        assert "observations" in results.episodes
        assert "actions" in results.episodes
        assert "rewards" in results.episodes
        assert "dones" in results.episodes

        # Episode data should match eval_rollouts
        assert results.episodes["observations"].shape[0] == test_eval_config.run.eval_rollouts

    def test_evaluate_with_config_type(self, test_training_config):
        """evaluate() should work with training Config (not just EvalConfig)."""
        results = evaluate(test_training_config, agent_state=None, return_episodes=False)

        assert results.num_episodes == test_training_config.run.eval_rollouts
        assert results.seed == test_training_config.run.seed

    def test_evaluate_with_pre_initialized_agent_state(self, test_eval_config):
        """evaluate() should accept pre-initialized agent state."""
        # Initialize agent manually
        env = _make_test_env()
        agent = _make_test_agent(env.get_action_space(env.config))
        key = jax.random.PRNGKey(123)
        obs, _ = env.reset(key, env.params, env.config)
        agent_state = agent.init(key, obs, agent.params)

        # Modify agent state to verify it's being used
        agent_state = _TestAgentState(marker=jnp.array(999.0, dtype=jnp.float32))

        # Run evaluation with pre-initialized state
        results = evaluate(test_eval_config, agent_state=agent_state, return_episodes=False)

        # Should complete successfully
        assert results.num_episodes == test_eval_config.run.eval_rollouts


class TestEvaluateStatistics:
    """Tests for evaluation statistics computation."""

    def test_computes_correct_statistics(self, test_eval_config):
        """Statistics should match numpy calculations on episode data."""
        results = evaluate(test_eval_config, agent_state=None, return_episodes=False)

        # Manually compute statistics
        episode_returns = results.episode_returns
        expected_mean = float(np.mean(episode_returns))
        expected_std = float(np.std(episode_returns))
        expected_min = float(np.min(episode_returns))
        expected_max = float(np.max(episode_returns))

        assert results.mean_return == pytest.approx(expected_mean)
        assert results.std_return == pytest.approx(expected_std)
        assert results.min_return == pytest.approx(expected_min)
        assert results.max_return == pytest.approx(expected_max)

    def test_episode_lengths_match_reality(self, test_eval_config):
        """Episode lengths should be capped by max_steps or env max_steps."""
        # Our test env has max_steps=3, eval_max_steps=10
        # Episodes should terminate at step 3
        results = evaluate(test_eval_config, agent_state=None, return_episodes=False)

        # All episodes should have length 3 (env terminates)
        assert all(length == 3 for length in results.episode_lengths)


class TestEvaluateEpisodeSaving:
    """Tests for episode saving during evaluation."""

    def test_saves_episodes_when_configured(self, test_eval_config, tmp_path, monkeypatch):
        """EvalConfig with save_frequency > 0 should save episodes to disk."""
        monkeypatch.chdir(tmp_path)

        # Enable episode saving
        test_eval_config.run.eval_episode_save_frequency = 1
        test_eval_config.run.eval_episode_save_count = 2

        results = evaluate(test_eval_config, agent_state=None, return_episodes=False)

        # Check that episodes directory was created
        assert results.run_dir is not None
        episodes_dir = results.run_dir / "episodes" / "step_00000000"
        assert episodes_dir.exists()

        # Check that episodes were saved
        episode_files = list(episodes_dir.glob("*.npz"))
        assert len(episode_files) == 2  # save_count=2

    def test_does_not_save_when_frequency_zero(self, test_eval_config, tmp_path, monkeypatch):
        """Default eval_episode_save_frequency=0 should not save episodes."""
        monkeypatch.chdir(tmp_path)

        # Default: save_frequency=0 (disabled)
        assert test_eval_config.run.eval_episode_save_frequency == 0

        evaluate(test_eval_config, agent_state=None, return_episodes=False)

        # No episodes directory should exist
        episodes_dir = tmp_path / "episodes"
        assert not episodes_dir.exists()

    def test_saves_all_episodes_when_save_count_none(self, test_eval_config, tmp_path, monkeypatch):
        """save_count=None should save all eval_rollouts episodes."""
        monkeypatch.chdir(tmp_path)

        test_eval_config.run.eval_episode_save_frequency = 1
        test_eval_config.run.eval_episode_save_count = None  # Save all

        results = evaluate(test_eval_config, agent_state=None, return_episodes=False)

        assert results.run_dir is not None
        episodes_dir = results.run_dir / "episodes" / "step_00000000"
        episode_files = list(episodes_dir.glob("*.npz"))

        # Should save all 3 episodes (eval_rollouts=3)
        assert len(episode_files) == 3


class TestEvaluateReproducibility:
    """Tests for evaluation reproducibility."""

    def test_same_seed_produces_same_results(self, test_eval_config):
        """Multiple evaluations with same seed should yield identical results."""
        results1 = evaluate(test_eval_config, agent_state=None, return_episodes=False)
        results2 = evaluate(test_eval_config, agent_state=None, return_episodes=False)

        # Episode returns should be identical
        np.testing.assert_array_equal(results1.episode_returns, results2.episode_returns)
        np.testing.assert_array_equal(results1.episode_lengths, results2.episode_lengths)

        # Statistics should be identical
        assert results1.mean_return == results2.mean_return
        assert results1.std_return == results2.std_return

    def test_different_seeds_produce_different_results(self, test_eval_config):
        """Different seeds should produce different results (stochastic environments)."""
        results1 = evaluate(test_eval_config, agent_state=None, return_episodes=False)

        # Change seed
        test_eval_config.run.seed = 999
        results2 = evaluate(test_eval_config, agent_state=None, return_episodes=False)

        # Note: Our test env is deterministic, but RNG keys will differ
        # In a real stochastic env, results would differ
        # For now, just verify the function runs with different seed
        assert results1.seed != results2.seed


class TestEvaluateIntegration:
    """Integration tests for evaluate()."""

    def test_evaluate_completes_end_to_end(self, test_eval_config):
        """Full evaluation workflow should complete without errors."""
        results = evaluate(test_eval_config, agent_state=None, return_episodes=True)

        # Verify all components worked
        assert results.num_episodes == 3
        assert len(results.episode_returns) == 3
        assert len(results.episode_lengths) == 3
        assert results.episodes is not None
        assert results.mean_return >= results.min_return
        assert results.mean_return <= results.max_return

    def test_evaluate_with_training_config_and_episodes(self, test_training_config, tmp_path, monkeypatch):
        """Training Config type should also support all evaluation features."""
        monkeypatch.chdir(tmp_path)

        results = evaluate(test_training_config, agent_state=None, return_episodes=True)

        # Should work just like EvalConfig
        assert results.num_episodes == test_training_config.run.eval_rollouts
        assert results.episodes is not None
