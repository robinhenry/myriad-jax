"""Integration tests for make_params_batch and domain-randomized training."""

import jax
import jax.numpy as jnp

from myriad.envs import make_env
from myriad.platform.initialization import make_params_batch


class TestMakeParamsBatch:
    def test_no_prior_shape(self):
        """Without a prior, make_params_batch broadcasts env.params to (N, ...)."""
        env = make_env("ccasr-gfp-sysid")
        N = 8
        batch = make_params_batch(env, N, jax.random.PRNGKey(0))
        # Every leaf should have a leading N dimension
        leaves = jax.tree_util.tree_leaves(batch)
        for leaf in leaves:
            assert leaf.shape[0] == N

    def test_no_prior_all_identical(self):
        """Without a prior, every env slice is the same as env.params."""
        env = make_env("ccasr-gfp-sysid")
        N = 4
        batch = make_params_batch(env, N, jax.random.PRNGKey(0))
        # All nu values should be equal
        nus = batch.physics.nu
        assert jnp.all(nus == nus[0])

    def test_with_prior_shape(self):
        """With a prior, make_params_batch returns (N, ...) with N distinct samples."""
        env = make_env("ccasr-gfp-sysid", nu_scale=0.3, Kh_scale=0.2)
        N = 16
        batch = make_params_batch(env, N, jax.random.PRNGKey(42))
        leaves = jax.tree_util.tree_leaves(batch)
        for leaf in leaves:
            assert leaf.shape[0] == N

    def test_with_prior_distinct_params(self):
        """With a prior, parallel envs get distinct parameter values."""
        env = make_env("ccasr-gfp-sysid", nu_scale=0.5, Kh_scale=0.5)
        N = 32
        batch = make_params_batch(env, N, jax.random.PRNGKey(0))
        nus = batch.physics.nu
        # With nonzero scale the samples must not be constant
        assert not jnp.all(nus == nus[0])

    def test_with_prior_all_positive(self):
        """Log-normal prior guarantees strictly positive kinetic params."""
        env = make_env("ccasr-gfp-sysid", nu_scale=1.0, Kh_scale=1.0, nh_scale=1.0)
        batch = make_params_batch(env, 64, jax.random.PRNGKey(7))
        assert jnp.all(batch.physics.nu > 0)
        assert jnp.all(batch.physics.Kh > 0)
        assert jnp.all(batch.physics.nh > 0)


class TestDomainRandomizedStep:
    """Verify that distinct params actually produce distinct rewards after one step."""

    def test_varying_params_vary_dynamics(self):
        """With randomized params each env should diverge from the others over time."""
        env = make_env("ccasr-gfp-sysid", nu_scale=0.5, Kh_scale=0.5)
        N = 8
        key = jax.random.PRNGKey(0)
        key, params_key, reset_key, step_key = jax.random.split(key, 4)

        params_batch = make_params_batch(env, N, params_key)

        reset_keys = jax.random.split(reset_key, N)
        obs_batch, state_batch = jax.vmap(env.reset, in_axes=(0, 0, None))(reset_keys, params_batch, env.config)

        # Run 10 steps
        actions = jnp.ones((N,), dtype=jnp.int32)

        def scan_step(carry, _):
            state, rng = carry
            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, N)
            _, next_state, _, _, info = jax.vmap(env.step, in_axes=(0, 0, 0, 0, None))(
                step_rngs, state, actions, params_batch, env.config
            )
            return (next_state, rng), info["F"]

        (_, _), F_traces = jax.lax.scan(scan_step, (state_batch, step_key), None, length=20)
        # Shape: (20, N) — final GFP values should differ across envs
        final_F = F_traces[-1]
        assert not jnp.all(final_F == final_F[0]), "All envs produced identical F — params may not be varying"
