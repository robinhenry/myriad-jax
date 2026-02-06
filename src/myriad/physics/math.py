"""Reusable mathematical functions for physics models."""

import jax.numpy as jnp
from jax import Array


def hill_function(
    x: Array,
    K: float,
    n: float,
) -> Array:
    """Compute the Hill function for cooperative binding/regulation.

    The Hill equation models sigmoidal responses in biological systems,
    commonly used for gene regulation, enzyme kinetics, and receptor binding.

    Formula: ``x^n / (K^n + x^n)``

    Args:
        x: Input concentration (molecules, arbitrary units, etc.)
        K: Half-maximal concentration (EC50/IC50). At x=K, output is 0.5.
        n: Hill coefficient controlling cooperativity.
            - n > 1: Positive cooperativity (sigmoidal, ultrasensitive)
            - n = 1: Non-cooperative (hyperbolic, Michaelis-Menten)
            - n < 1: Negative cooperativity (gradual response)

    Returns:
        Hill function value in [0, 1] range.

    Examples:
        >>> # Gene activation with moderate cooperativity
        >>> hill_function(x=100.0, K=90.0, n=3.6)
        Array(0.576, dtype=float32)

        >>> # Vectorized over batch dimension
        >>> concentrations = jnp.array([50.0, 100.0, 150.0])
        >>> hill_function(x=concentrations, K=100.0, n=2.0)
        Array([0.2, 0.5, 0.692], dtype=float32)
    """
    x_powered = jnp.power(x, n)
    K_powered = jnp.power(K, n)
    return x_powered / (K_powered + x_powered)
