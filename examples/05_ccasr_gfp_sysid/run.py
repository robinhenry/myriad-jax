"""CcaS-CcaR system identification — minimal example.

Demonstrates how to use the ``ccasr-gfp-sysid`` environment to collect GFP
fluorescence trajectories from cells with unknown kinetic parameters θ*.

Both sections use the ``evaluate()`` platform entrypoint:

1. Single episode — one cell, 24-hour experiment with an open-loop light
   schedule built directly from a JAX array.  The pre-built agent is passed
   via ``evaluate(config, agent=agent)``, bypassing the config system.

2. Population rollout with domain randomization — 256 cells, each with their
   own θ* sampled from a log-normal prior.  Two extra kwargs (``nu_scale``,
   ``Kh_scale``) opt in; the platform handles per-env param sampling.

Usage:
    cd examples/05_ccasr_gfp_sysid
    python run.py
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from myriad.agents.classical.open_loop import make_agent as make_open_loop_agent
from myriad.configs.builder import create_eval_config
from myriad.envs import make_env
from myriad.platform import evaluate

# ---------------------------------------------------------------------------
# Shared: environment defaults
# ---------------------------------------------------------------------------

_env = make_env("ccasr-gfp-sysid")
F_OBS_NORMALIZER = _env.config.task.F_obs_normalizer
N_STEPS = _env.config.max_steps  # 288 steps = 24 h
PULSE_WIDTH = 72  # 6-hour ON / 6-hour OFF (at 5 min/step)

# ---------------------------------------------------------------------------
# 1.  Single-cell episode — open-loop light schedule built from a JAX array
#
#     open_loop.make_agent takes a raw schedule array — use this for custom
#     signals (step responses, PRBS, multi-sine) that can't be expressed as
#     plain integer kwargs.  Pass the pre-built agent via evaluate(agent=...).
# ---------------------------------------------------------------------------

print("=== Single episode (1 cell, periodic light: ON/OFF every 6 h) ===")

schedule = jnp.array([1] * PULSE_WIDTH + [0] * PULSE_WIDTH, dtype=jnp.int32)
open_loop_agent = make_open_loop_agent(_env.get_action_space(_env.config), schedule)

single_config = create_eval_config(
    "ccasr-gfp-sysid",
    agent="periodic",  # used for logging/metadata only; open_loop_agent runs
    eval_rollouts=1,
    pulse_width=PULSE_WIDTH,
)

single_results = evaluate(single_config, agent=open_loop_agent, return_episodes=True)

assert single_results.episodes is not None
F_single = single_results.episodes["observations"][0, :, 0] * F_OBS_NORMALIZER
U_single = single_results.episodes["actions"][0, :]

print(f"  Steps:   {N_STEPS}")
print(f"  Final F: {F_single[-1]:.1f}  (GFP molecules)")

# ---------------------------------------------------------------------------
# 2.  Population rollout — 256 cells with domain-randomized θ*
#
#     nu_scale / Kh_scale are the only changes vs. the single-cell config.
#     The platform samples a distinct θ* per cell; no manual vmap/scan.
# ---------------------------------------------------------------------------

print("\n=== Population rollout (256 cells, domain-randomized θ*) ===")

N_CELLS = 256

pop_config = create_eval_config(
    "ccasr-gfp-sysid",
    agent="periodic",
    eval_rollouts=N_CELLS,
    pulse_width=PULSE_WIDTH,
    **{"env.nu_scale": 0.2, "env.Kh_scale": 0.2},
)

pop_results = evaluate(pop_config, return_episodes=True)

assert pop_results.episodes is not None
F_pop = pop_results.episodes["observations"][:, :, 0] * F_OBS_NORMALIZER

print(f"  Cells:    {N_CELLS}")
print(f"  F at end: min={F_pop[:, -1].min():.1f}  max={F_pop[:, -1].max():.1f}  mean={F_pop[:, -1].mean():.1f}")
print("  (variation = different θ* drawn from log-normal prior)")

# ---------------------------------------------------------------------------
# 3.  Figure
# ---------------------------------------------------------------------------

from scipy.stats import gaussian_kde  # noqa: E402

time_h = np.arange(N_STEPS) * 5 / 60  # steps → hours

fig, axes = plt.subplots(2, 2, figsize=(10, 6))

ax = axes[0, 0]
ax.plot(time_h, F_single, color="tab:green", lw=1.5)
ax.set_xlabel("Time (h)")
ax.set_ylabel("GFP molecules")
ax.set_title("A  Single cell — GFP trajectory")

ax = axes[0, 1]
ax.step(time_h, U_single, where="post", color="tab:orange", lw=1.5)
ax.set_xlabel("Time (h)")
ax.set_ylabel("Light (U)")
ax.set_ylim(-0.1, 1.2)
ax.set_title("B  Periodic light input (ON/OFF every 6 h)")

ax = axes[1, 0]
ax.plot(time_h, F_pop.T, alpha=0.15, lw=0.8, color="tab:green")
ax.set_xlabel("Time (h)")
ax.set_ylabel("GFP molecules")
ax.set_title(f"C  Population (N={N_CELLS}) — domain-randomized θ*")

ax = axes[1, 1]
final_F = F_pop[:, -1]
kde = gaussian_kde(final_F, bw_method="scott")
x_grid = np.linspace(final_F.min(), final_F.max(), 300)
ax.fill_between(x_grid, kde(x_grid), alpha=0.25, color="tab:green")
ax.plot(x_grid, kde(x_grid), color="tab:green", lw=2)
ax.set_xlabel("GFP at t=24 h")
ax.set_ylabel("Density")
ax.set_title("D  Final GFP distribution (KDE)")

fig.tight_layout()
fig.savefig("sysid_trajectories.png", dpi=120)
print("\nFigure saved to sysid_trajectories.png")
