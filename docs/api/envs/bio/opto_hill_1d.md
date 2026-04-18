# 1D Optogenetic Hill Circuit

Minimal single-species optogenetic gene circuit. One fluorescent protein `X` driven by a continuous light input `U ∈ [0, 1]` through a Hill production rate and linear degradation. Intended as a transparent sandbox for system-identification experiments.

## Reaction network

Two Gillespie reactions:

- Production: `∅ → X` with rate `k_prod · U^n / (K^n + U^n)`
- Degradation: `X → ∅` with rate `k_deg · X`

Steady state under constant U: `⟨X⟩_ss = k_prod · hill(U, K, n) / k_deg`.

## Action and observation

- Action: `Box(low=0.0, high=1.0, shape=())` — continuous light intensity.
- Observation: shape `(1,)` — `X / X_obs_normalizer`.

## Physics

Stochastic dynamics using the Gillespie algorithm for exact simulation.

```{eval-rst}
.. automodule:: myriad.envs.bio.opto_hill_1d.physics
   :members:
   :undoc-members:
   :show-inheritance:
```

## System Identification Task

Zero-reward task. Parameters `θ* = (k_prod, K, n, k_deg)` live in `SysIdTaskParams` and persist across resets; only the cell state `X` resets between episodes. The inference objective is computed outside the environment.

```{eval-rst}
.. automodule:: myriad.envs.bio.opto_hill_1d.tasks.sysid
   :members:
   :undoc-members:
   :show-inheritance:
```
