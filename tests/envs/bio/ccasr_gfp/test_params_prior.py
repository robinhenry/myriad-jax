"""Tests for PhysicsParamsPrior and task-level prior structs."""

import jax
import jax.numpy as jnp
import pytest

from myriad.envs.bio.ccasr_gfp.physics import PhysicsParams, PhysicsParamsPrior
from myriad.envs.bio.ccasr_gfp.tasks.sysid import SysIdTaskParams, SysIdTaskParamsPrior, make_env


class TestPhysicsParamsPrior:
    def test_sample_returns_physics_params(self):
        prior = PhysicsParamsPrior()
        key = jax.random.PRNGKey(0)
        params = prior.sample(key)
        assert isinstance(params, PhysicsParams)

    def test_sample_all_positive(self):
        prior = PhysicsParamsPrior(nu_scale=0.5, Kh_scale=0.5, nh_scale=0.5, Kf_scale=0.5, nf_scale=0.5)
        key = jax.random.PRNGKey(1)
        params = prior.sample(key)
        assert float(params.nu) > 0
        assert float(params.Kh) > 0
        assert float(params.nh) > 0
        assert float(params.Kf) > 0
        assert float(params.nf) > 0

    def test_sample_all_finite(self):
        prior = PhysicsParamsPrior(nu_scale=1.0, Kh_scale=1.0)
        key = jax.random.PRNGKey(2)
        params = prior.sample(key)
        assert jnp.isfinite(params.nu)
        assert jnp.isfinite(params.Kh)

    def test_scale_zero_is_deterministic(self):
        """With scale=0, sample must return exp(loc) for every field."""
        prior = PhysicsParamsPrior()  # all scales default to 0
        p1 = prior.sample(jax.random.PRNGKey(0))
        p2 = prior.sample(jax.random.PRNGKey(999))
        assert float(p1.nu) == pytest.approx(float(p2.nu))
        assert float(p1.Kh) == pytest.approx(float(p2.Kh))

    def test_scale_zero_equals_exp_loc(self):
        """scale=0 → param == exp(loc)."""
        prior = PhysicsParamsPrior()
        params = prior.sample(jax.random.PRNGKey(0))
        assert float(params.nu) == pytest.approx(0.01, rel=1e-5)
        assert float(params.Kh) == pytest.approx(90.0, rel=1e-5)

    def test_vmap_produces_distinct_samples(self):
        """jax.vmap over keys must produce N distinct parameter sets."""
        prior = PhysicsParamsPrior(nu_scale=0.5, Kh_scale=0.5)
        keys = jax.random.split(jax.random.PRNGKey(0), 8)
        batch = jax.vmap(prior.sample)(keys)
        # All nus should not be the same
        nus = jnp.array(batch.nu)
        assert not jnp.all(nus == nus[0])

    def test_vmap_correct_shape(self):
        prior = PhysicsParamsPrior(nu_scale=0.3)
        keys = jax.random.split(jax.random.PRNGKey(0), 16)
        batch = jax.vmap(prior.sample)(keys)
        assert batch.nu.shape == (16,)
        assert batch.Kh.shape == (16,)


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
        """No-scale prior must always return the same params."""
        prior = SysIdTaskParamsPrior()
        p1 = prior.sample(jax.random.PRNGKey(0))
        p2 = prior.sample(jax.random.PRNGKey(1))
        assert float(p1.physics.nu) == pytest.approx(float(p2.physics.nu))


class TestMakeEnvWithPrior:
    def test_no_prior_sample_fn_is_none(self):
        env = make_env()
        assert env.sample_params_fn is None

    def test_with_prior_sample_fn_is_set(self):
        env = make_env(nu_scale=0.3)
        assert env.sample_params_fn is not None

    def test_sample_fn_returns_task_params(self):
        env = make_env(nu_scale=0.3)
        assert env.sample_params_fn is not None
        params = env.sample_params_fn(jax.random.PRNGKey(0))
        assert isinstance(params, SysIdTaskParams)

    def test_explicit_prior_arg(self):
        prior = SysIdTaskParamsPrior(physics=PhysicsParamsPrior(nu_scale=0.5))
        env = make_env(params_prior=prior)
        assert env.sample_params_fn is not None
