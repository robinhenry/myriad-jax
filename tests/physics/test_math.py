"""Tests for mathematical functions used in physics models.

This module tests mathematical building blocks like the Hill function,
ensuring they produce correct values and handle edge cases properly.
"""

import jax.numpy as jnp

from myriad.physics import hill_function


class TestHillFunction:
    """Test suite for the Hill function."""

    def test_half_maximal_concentration(self):
        """Test that at x=K, the Hill function returns 0.5."""
        # For any K and n, when x=K, output should be 0.5
        result = hill_function(x=100.0, K=100.0, n=2.0)
        assert jnp.isclose(result, 0.5), f"Expected 0.5 at x=K, got {result}"

        result = hill_function(x=90.0, K=90.0, n=3.6)
        assert jnp.isclose(result, 0.5), f"Expected 0.5 at x=K, got {result}"

    def test_zero_concentration(self):
        """Test that at x=0, the Hill function returns 0."""
        result = hill_function(x=0.0, K=100.0, n=2.0)
        assert jnp.isclose(result, 0.0), f"Expected 0.0 at x=0, got {result}"

    def test_high_concentration(self):
        """Test that at x>>K, the Hill function approaches 1."""
        # When x is much larger than K, output should be close to 1
        result = hill_function(x=1000.0, K=10.0, n=2.0)
        assert result > 0.99, f"Expected >0.99 at x>>K, got {result}"

    def test_monotonicity(self):
        """Test that Hill function is monotonically increasing with x."""
        K = 100.0
        n = 2.0
        x_values = jnp.array([10.0, 50.0, 100.0, 200.0, 500.0])
        results = hill_function(x=x_values, K=K, n=n)

        # Check that each value is >= previous value
        for i in range(1, len(results)):
            assert results[i] >= results[i - 1], f"Hill function not monotonic: {results[i]} < {results[i-1]}"

    def test_bounded_range(self):
        """Test that Hill function output is always in [0, 1]."""
        K = 100.0
        n = 2.0
        x_values = jnp.array([0.0, 1.0, 10.0, 50.0, 100.0, 200.0, 1000.0])
        results = hill_function(x=x_values, K=K, n=n)

        assert jnp.all(results >= 0.0), "Hill function produced negative values"
        assert jnp.all(results <= 1.0), "Hill function produced values > 1.0"

    def test_cooperativity_effect(self):
        """Test that higher Hill coefficient (n) increases steepness."""
        K = 100.0

        # At x=K, all n values should give 0.5
        # But the slope should differ
        result_low = hill_function(x=1.5 * K, K=K, n=1.0)
        result_high = hill_function(x=1.5 * K, K=K, n=4.0)

        # Higher cooperativity should give steeper transition
        # At x=1.5*K, higher n should be closer to 1
        assert result_high > result_low, (
            f"Higher cooperativity should give steeper response: " f"n=4.0 gave {result_high}, n=1.0 gave {result_low}"
        )

    def test_vectorization(self):
        """Test that Hill function works with array inputs."""
        x_values = jnp.array([50.0, 100.0, 150.0])
        K = 100.0
        n = 2.0

        results = hill_function(x=x_values, K=K, n=n)

        # Check shape preservation
        assert results.shape == x_values.shape, f"Shape mismatch: input {x_values.shape}, output {results.shape}"

        # Check individual values are correct
        expected = jnp.array(
            [
                hill_function(x=50.0, K=K, n=n),
                hill_function(x=100.0, K=K, n=n),
                hill_function(x=150.0, K=K, n=n),
            ]
        )
        assert jnp.allclose(results, expected), "Vectorized results differ from scalar"

    def test_ccas_ccar_parameters(self):
        """Test with actual parameters from CcaS-CcaR system."""
        # From the CDC 2025 paper implementation
        H = 100.0
        result_H = hill_function(x=H, K=90.0, n=3.6)

        F = 50.0
        result_F = hill_function(x=F, K=30.0, n=3.6)

        # Just verify they're in valid range and reasonable
        assert 0.0 <= result_H <= 1.0, f"Invalid result for H: {result_H}"
        assert 0.0 <= result_F <= 1.0, f"Invalid result for F: {result_F}"

        # H > K, so should be > 0.5
        assert result_H > 0.5, f"Expected H result > 0.5, got {result_H}"

        # F > K, so should be > 0.5
        assert result_F > 0.5, f"Expected F result > 0.5, got {result_F}"
