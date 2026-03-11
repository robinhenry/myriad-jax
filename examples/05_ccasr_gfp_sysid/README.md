# Example 05: CcaS-CcaR System Identification

Minimal demonstration of the `ccasr-gfp-sysid` environment. The task models a
synthetic gene circuit where the agent can switch a light input ON or OFF every
5 minutes and observes GFP fluorescence. CcaSR concentration (H) is hidden;
only GFP (F) is observable.

This is a **system identification** task, not a control task. There is no
reward — the environment just simulates the circuit and returns observations.
The inference algorithm (e.g., SMC, MCMC, neural posterior estimation) is
external and is the actual "agent".

## What the script does

1. **Single episode** — runs one 24-hour experiment (288 steps) with a periodic
   light policy (ON for 6 h, OFF for 6 h). Uses the built-in `periodic` agent
   via the `evaluate()` entrypoint.

2. **Population rollout with domain randomization** — runs 256 cells in parallel,
   each with its own θ\* sampled from a log-normal prior. Two kwargs
   (`nu_scale`, `Kh_scale`) opt into randomization; the platform handles
   per-env parameter sampling automatically.

3. **Figure** — saves a 4-panel PNG showing the single-cell trajectory, the
   light protocol, all 256 population trajectories, and the final GFP
   distribution.

## How to run

```bash
cd examples/05_ccasr_gfp_sysid
python run.py
```

Output is printed to stdout and a figure is saved to `sysid_trajectories.png`.

## Key concepts

| Concept | Where |
|---|---|
| Periodic open-loop policy | `agent="periodic", pulse_width=72` |
| Single-cell rollout | `create_eval_config(..., eval_rollouts=1)` + `evaluate()` |
| Domain randomization | `env.nu_scale=0.2, env.Kh_scale=0.2` kwargs to `create_eval_config` |
| Population rollout | `eval_rollouts=256` — platform vmaps internally |
| Per-env θ\* sampling | `make_params_batch` (called automatically by `evaluate()`) |
| Observation (F only) | `obs.F_normalized` — H is not observable |
| Reward | Always `0.0` — SysID has no reward signal |

## Domain randomization API

Passing `*_scale` kwargs to `create_eval_config` (or `make_env`) opts into
log-normal parameter randomization. Each parallel environment receives an
independently sampled θ\*:

```python
config = create_eval_config(
    "ccasr-gfp-sysid",
    agent="periodic",
    eval_rollouts=256,
    pulse_width=72,
    **{"env.nu_scale": 0.2, "env.Kh_scale": 0.2},
)
results = evaluate(config, return_episodes=True)
# results.episodes["observations"] shape: (256, 288, 1)
```

With `scale=0` (default) all cells share the same θ\* — fully deterministic,
backward compatible.

## Extending this example

- **Bayesian inference** — feed the F trajectories into a likelihood function
  and run SMC or HMC to infer the posterior over (nu, Kh, nh, Kf, nf).
- **Active experiment design** — replace the periodic policy with one that
  maximises expected information gain (mutual information between θ and future
  observations).
- **Scale up** — increase `eval_rollouts` to 1024 or more; the platform scales
  linearly on GPU without code changes.
- **Wider priors** — increase `*_scale` values to explore more of parameter
  space, or add `nh_scale`, `Kf_scale`, `nf_scale` for full-parameter randomization.
