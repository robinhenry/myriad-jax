# Example 04: PQN Hyperparameter Sweep on ccasr-gfp-control

Bayesian hyperparameter search for PQN on the GFP control task, followed by
statistical validation of the top configs. W&B Sweeps manages the search;
Hyperband prunes poor runs early.

## Two-phase workflow

**Phase 1 — hyperparameter search.** All runs use a fixed seed (`seed=0`) so
configs are compared on equal footing. Bayesian optimisation budget is spent
entirely on hyperparameters.

**Phase 2 — seed evaluation.** After the sweep finishes, `myriad seed-eval`
queries W&B for the top-5 configs and re-runs each with seeds 0–2. Runs are
grouped in W&B for mean ± std reporting.

## How to run

```bash
wandb login                         # authenticate once
export WANDB_ENTITY=your-entity     # your W&B username or team

./examples/04_pqn_ccasr_sweep/run_sweep.sh
```

The script runs both phases end-to-end: it creates a sweep, runs a local
agent, then launches seed-eval automatically once the agent finishes. Edit
`NUM_ENVS` at the top of the script to change parallelism level.

To register the sweep without launching an agent (e.g. to hand off to a cluster):

```bash
myriad sweep-create configs/pqn_ccasr_sweep.yaml \
    --base-group pqn_ccasr \
    --levels 1024
# Prints the sweep ID. Then on the worker: wandb agent <sweep_id>
```

Run seed-eval manually after the sweep:

```bash
myriad seed-eval $WANDB_ENTITY/myriad-ccasr/<SWEEP_ID> \
    --top-k 5 --seeds 0,1,2 \
    --metric eval/episode_return/mean \
    --group pqn_ccasr_validated
```

## Viewing results

Open the `myriad-ccasr` project in W&B.

**Phase 1:** Runs are tagged with group `pqn_ccasr_1024`. Use the
parallel-coordinates plot to identify which hyperparameters drive
`eval/episode_return/mean`.

**Phase 2:** Filter by `job_type = seed-eval` to isolate validation runs. In
the Groups view, each rank group (e.g. `pqn_ccasr_validated_rank0`) contains
one run per seed; W&B shows mean ± std automatically.
