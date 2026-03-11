# Example 03b: PQN on CcaSR-GFP Control — 1M envs benchmark

GPU performance benchmark. Trains PQN with 1M parallel environments on
ccasr-gfp-control to verify that training throughput doesn't degrade over time.

This is a quick smoke test, not a full reproduction — use `03_ccasr_gfp_control`
for that.

## How to run

```bash
cd examples/03b_ccasr_gfp_control_1M_envs
./run.sh
```

## What to look for

Compare wall-clock time across runs. The config uses 288 steps per env
(~288M total timesteps at 1M envs) with frame stacking (16 frames).
