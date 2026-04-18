"""Tests for opto_hill_1d PhysicsParamsPrior and SysIdTaskParamsPrior."""

import jax
import jax.numpy as jnp
import pytest

from myriad.envs.bio.opto_hill_1d.physics import PhysicsParams, PhysicsParamsPrior
from myriad.envs.bio.opto_hill_1d.tasks.sysid import (
    SysIdTaskParams,
    SysIdTaskParamsPrior,
    make_env,
)


class TestPhysicsParamsPrior:
    def test_sample_returns_physics_params(self):
        prior = PhysicsParamsPrior()
        params = prior.sample(jax.random.PRNGKey(0))
        assert isinstance(params, PhysicsParams)

    def test_sample_all_positive(self):
        prior = PhysicsParamsPrior(k_prod_scale=0.5, K_scale=0.5, n_scale=0.5, k_deg_scale=0.5)
        params = prior.sample(jax.random.PRNGKey(1))
        assert float(params.k_prod) > 0
        assert float(params.K) > 0
        assert float(params.n) > 0
        assert float(params.k_deg) > 0

    def test_sample_all_finite(self):
        prior = PhysicsParamsPrior(k_prod_scale=1.0, k_deg_scale=1.0)
        params = prior.sample(jax.random.PRNGKey(2))
        assert jnp.isfinite(params.k_prod)
        assert jnp.isfinite(params.K)
        assert jnp.isfinite(params.n)
        assert jnp.isfinite(params.k_deg)

    def test_scale_zero_is_deterministic(self):
        prior = PhysicsParamsPrior()
        p1 = prior.sample(jax.random.PRNGKey(0))
        p2 = prior.sample(jax.random.PRNGKey(999))
        assert float(p1.k_prod) == pytest.approx(float(p2.k_prod))
        assert float(p1.K) == pytest.approx(float(p2.K))
        assert float(p1.n) == pytest.approx(float(p2.n))
        assert float(p1.k_deg) == pytest.approx(float(p2.k_deg))

    def test_scale_zero_equals_exp_loc(self):
        prior = PhysicsParamsPrior()
        params = prior.sample(jax.random.PRNGKey(0))
        assert float(params.k_prod) == pytest.approx(5.0, rel=1e-5)
        assert float(params.K) == pytest.approx(0.5, rel=1e-5)
        assert float(params.n) == pytest.approx(2.0, rel=1e-5)
        assert float(params.k_deg) == pytest.approx(0.05, rel=1e-5)

    def test_vmap_produces_distinct_samples(self):
        prior = PhysicsParamsPrior(k_prod_scale=0.5, K_scale=0.5)
        keys = jax.random.split(jax.random.PRNGKey(0), 8)
        batch = jax.vmap(prior.sample)(keys)
        assert not jnp.all(batch.k_prod == batch.k_prod[0])

    def test_vmap_correct_shape(self):
        prior = PhysicsParamsPrior(k_prod_scale=0.3)
        keys = jax.random.split(jax.random.PRNGKey(0), 16)
        batch = jax.vmap(prior.sample)(keys)
        assert batch.k_prod.shape == (16,)
        assert batch.K.shape == (16,)
        assert batch.n.shape == (16,)
        assert batch.k_deg.shape == (16,)


class TestSysIdTaskParamsPrior:
    def test_sample_returns_task_params(self):
        prior = SysIdTaskParamsPrior()
        params = prior.sample(jax.random.PRNGKey(0))
        assert isinstance(params, SysIdTaskParams)

    def test_sample_has_physics(self):
        prior = SysIdTaskParamsPrior()
        params = prior.sample(jax.random.PRNGKey(0))
        assert isinstance(params.physics, PhysicsParams)

    def test_round_trip_deterministic(self):
        prior = SysIdTaskParamsPrior()
        p1 = prior.sample(jax.random.PRNGKey(0))
        p2 = prior.sample(jax.random.PRNGKey(1))
        assert float(p1.physics.k_prod) == pytest.approx(float(p2.physics.k_prod))


class TestMakeEnvWithPrior:
    def test_no_prior_sample_fn_is_none(self):
        env = make_env()
        assert env.sample_params_fn is None

    def test_with_prior_sample_fn_is_set(self):
        env = make_env(k_prod_scale=0.3)
        assert env.sample_params_fn is not None

    def test_sample_fn_returns_task_params(self):
        env = make_env(k_prod_scale=0.3)
        assert env.sample_params_fn is not None
        params = env.sample_params_fn(jax.random.PRNGKey(0))
        assert isinstance(params, SysIdTaskParams)

    def test_explicit_prior_arg(self):
        prior = SysIdTaskParamsPrior(physics=PhysicsParamsPrior(k_prod_scale=0.5))
        env = make_env(params_prior=prior)
        assert env.sample_params_fn is not None

    def test_kwargs_flow_to_physics_params(self):
        """Plain param kwargs override defaults without triggering the prior."""
        env = make_env(k_prod=42.0)
        assert env.sample_params_fn is None
        assert float(env.params.physics.k_prod) == pytest.approx(42.0)

    def test_kwargs_flow_to_task_config(self):
        """Task-level kwargs (max_steps, X_obs_normalizer) flow through make_env."""
        env = make_env(max_steps=100, X_obs_normalizer=50.0)
        assert env.config.max_steps == 100
        assert float(env.config.X_obs_normalizer) == pytest.approx(50.0)

    def test_registered_under_canonical_name(self):
        """make_env('opto-hill-1d-sysid') resolves through the env registry."""
        from myriad.envs import make_env as registry_make_env

        env = registry_make_env("opto-hill-1d-sysid")
        assert env.get_obs_shape(env.config) == (1,)
        action_space = env.get_action_space(env.config)
        assert float(action_space.low) == 0.0
        assert float(action_space.high) == 1.0
