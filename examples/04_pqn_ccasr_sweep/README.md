# Example 04: PQN Hyperparameter Sweep on ccasr-gfp-control

Bayesian hyperparameter search for PQN on the GFP control task, followed by
statistical validation of the top configs. W&B Sweeps manages the search;
Hyperband prunes poor runs early.

## Two-phase workflow

**Phase 1 — hyperparameter search.** All runs use a fixed seed (`seed=0`) so
configs are compared on equal footing. The full Bayesian optimisation budget
is spent on hyperparameters, not on seed variance.

**Phase 2 — seed evaluation.** After the sweep finishes, `myriad seed-eval`
queries W&B for the top-K configs by `eval/return/best` and re-runs each with
multiple seeds. Runs are grouped in W&B (one group per rank) so the dashboard
shows mean ± std across seeds automatically.

## Prerequisites

```bash
pip install myriad-jax
wandb login                         # authenticate once
export WANDB_ENTITY=your-entity     # your W&B username or team
```

## Running the full pipeline

```bash
./examples/04_pqn_ccasr_sweep/run_sweep.sh
```

The script runs both phases end-to-end: creates a sweep, runs a local agent,
then launches seed-eval once the agent finishes. Edit `NUM_ENVS` at the top
of the script to change parallelism.

### Manual / cluster workflow

**Register the sweep without launching an agent** (hand off to a cluster):

```bash
cd examples/04_pqn_ccasr_sweep
SWEEP_ID=$(myriad sweep-create configs/pqn_ccasr_sweep.yaml \
    --base-group pqn_ccasr \
    --levels 1024)
echo "$SWEEP_ID"               # entity/project/sweep_id
wandb agent "$SWEEP_ID"        # run on any machine with GPU access
```

**Run seed-eval after the sweep** (pass the full entity/project/sweep_id):

```bash
myriad seed-eval lugagne-lab/myriad-ccasr-gfp-v1/<SWEEP_ID> \
    --top-k 5 \
    --seeds 0,1,2 \
    --metric eval/return/best \
    --group pqn_ccasr_validated
```

`SWEEP_ID` is printed by `sweep-create` and also shown on the W&B sweep page
URL: `wandb.ai/<entity>/<project>/sweeps/<SWEEP_ID>`.

## Viewing results in W&B

Project: **myriad-ccasr-gfp-v1**

### Phase 1 — hyperparameter search

- Open the **Sweeps** tab and select the sweep.
- Use the **Parallel coordinates** plot to identify which hyperparameters
  correlate with high `eval/return/best`.
- Runs use `group = pqn_ccasr_1024` (or whichever level you set).

### Phase 2 — seed evaluation

Seed-eval runs have `job_type = seed-eval`. To isolate them:

1. In the **Runs** table, filter by `job_type = seed-eval`.
2. Switch to the **Groups** view (button in the top-right of the runs table).
3. Each group (`pqn_ccasr_validated_rank0`, `_rank1`, …) contains one run per
   seed. W&B aggregates them automatically and shows **mean ± std** in charts
   and the runs table.

The group with the highest mean return at rank 0 is your best validated config.

## Understanding the metric floor: -7200.0

Many Phase 1 runs report `eval/return/best ≈ -7200.0`. This is not an error —
it is the **null-policy floor** for this environment:

- `eval_max_steps = 288` steps per episode (≈ 24 h of simulated cell time)
- Reward per step = `-|F - F_target|` where `F_target = 25.0` AU (GFP units)
- If the agent never induces GFP expression (F ≈ 0): reward ≈ -25 per step
- Total: **-25 × 288 = -7200**

Runs stuck at -7200 failed to learn during the sweep — their hyperparameter
configuration prevented convergence (e.g. LR too high/low, insufficient
rollout length). Hyperband prunes the worst of these early; the Bayesian
optimiser steers away from that region of hyperparameter space. Expect 30–60 %
of runs to hit this floor in a broad initial sweep.
