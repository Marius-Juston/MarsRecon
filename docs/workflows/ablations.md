# Ablations

`scripts/training/run_ablation.py` orchestrates the loss-component ablation
campaign. It reads a declarative spec (`configs/ablation/loss_dropone.yaml`),
schedules training jobs across GPU lanes, watches for crashes, runs the
test-only phase on the best checkpoints, and produces paper-ready statistical
summaries (paired Wilcoxon + Holm-Bonferroni + hierarchical bootstrap CIs).

## Run

```bash
# Full campaign
uv run python scripts/training/run_ablation.py \
    --spec configs/ablation/loss_dropone.yaml

# Plan only — print the queue and exit
uv run python scripts/training/run_ablation.py \
    --spec configs/ablation/loss_dropone.yaml --dry-run

# Smoke test a single job
uv run python scripts/training/run_ablation.py \
    --spec configs/ablation/loss_dropone.yaml --only L0_s42

# Re-run the aggregator over existing results (no launches)
uv run python scripts/training/run_ablation.py \
    --spec configs/ablation/loss_dropone.yaml --aggregate-only
```

## Spec file shape

See `configs/ablation/loss_dropone.yaml` for the canonical example. Key fields:

- `lanes` — list of `CUDA_VISIBLE_DEVICES` strings (e.g. `["0,1", "2,3"]`).
  One concurrent job per lane.
- `seeds` — reused across every variant for paired comparisons.
- `variants[].overrides` — OmegaConf dotlist; the first variant is the baseline.
- `common_overrides` — applied to every job.
- `watchdog` — crash-detection thresholds (NaN/Inf, val/* stall) and retry
  budgets. See "Fault tolerance" below for what each field controls.

## Outputs

Under `<output_root>` (from the spec):

- `state.json` — per-job status, attempts, resumes, W&B run id, ckpt paths.
- `<job_id>/` — one directory per (variant, seed) with checkpoints, CSVs,
  test_per_patch_*.npz files, and `job.log`.
- `results_table.csv`, `results_{rmse,photo}_track.csv`,
  `results_stats_{rmse,photo}.csv`, `results_bootstrap_{rmse,photo}.csv`,
  `results_cooptimization.csv` — produced by the aggregator.
- `orchestrator.lock`, `orchestrator.heartbeat.json`, `control/lanes.json` —
  see "Fault tolerance".

## Fault tolerance

The campaign is engineered to survive crashes, host reboots, and operational
interruptions without losing checkpoints, W&B continuity, or retry budget.

### Resume-from-checkpoint

State persists in `state.json` (atomic write + fsync on every transition).
Restart the orchestrator after any crash or reboot; it picks up where it left
off. Lightning's `last.ckpt` resumes training; the W&B run continues.

### W&B run continuity

The orchestrator assigns each job a stable W&B run id (`state.wandb_run_id`)
on first launch and exports it as `WANDB_RUN_ID` + `WANDB_RESUME=allow` to
every (re)launch of that job. A single (variant, seed) therefore appears as
one continuous run in W&B across any number of restarts — the step counter
does not reset and the curves stay contiguous.

The test-only phase runs with `WANDB_MODE=disabled`; per-patch `.npz` dumps
are the source of truth for the aggregator.

### Attempts vs resumes

`watchdog.max_attempts` (default `3`) is the budget for **fresh** failures
(e.g. NaN at init, config error). Restarts that find `last.ckpt` increment
`state.resumes` instead (capped at 50 to catch infinite-restart loops). A
host reboot mid-training does not consume `attempts`.

### Graceful shutdown

`SIGINT` / `SIGTERM` to the orchestrator triggers a drain:

1. Stop scheduling new jobs.
2. Send `SIGTERM` to every running child's process group.
3. Wait up to `watchdog.shutdown_grace_seconds` (default `300`) for Lightning
   to flush `last.ckpt`.
4. `SIGKILL` anything still alive, mark jobs `queued` with
   `exit_reason="orchestrator shutdown"`, flush state, return.

No retry budget is consumed; restart the script to resume.

### Dynamic lane control

If a colleague needs GPUs mid-campaign, release a lane without losing W&B
continuity or retry budget:

```bash
# Preempt the running job on lane "2,3" (SIGTERM + grace → last.ckpt → re-queue)
uv run python scripts/training/run_ablation.py --spec ... --disable-lane "2,3"

# Let the current job on "2,3" finish, but don't launch new ones there
uv run python scripts/training/run_ablation.py --spec ... --drain-lane "2,3"

# Give the lane back
uv run python scripts/training/run_ablation.py --spec ... --enable-lane "2,3"

# Show current control file
uv run python scripts/training/run_ablation.py --spec ... --show-control
```

These subcommands mutate `<output_root>/control/lanes.json` atomically and
exit. The running orchestrator re-reads the file each poll tick (~60s) and
applies changes on the next tick. They do **not** require the orchestrator
lock — safe to run from any shell.

### Single-orchestrator lock

`<output_root>/orchestrator.lock` is held with `fcntl.flock` for the lifetime
of the scheduler. A second invocation against the same `output_root` exits
immediately with the offending PID — accidental double-launches cannot race
on `state.json`.

### Liveness

`<output_root>/orchestrator.heartbeat.json` is rewritten each poll tick with
`{pid, hostname, timestamp, running_jobs, queued_jobs, shutting_down}`.
`watch -n 5 cat <output_root>/orchestrator.heartbeat.json` is the easiest way
to monitor a long-running campaign.

## Visualizing ablation results

```bash
uv run python scripts/visualization/paper_training_dynamics.py \
    --project mars-depthfm-ablation \
    --output paper_figures/
```

Produces convergence plots and lunar-Lambert weight-evolution plots from the
W&B history. Pair with the aggregator CSVs for the headline tables.
