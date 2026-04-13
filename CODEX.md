# CODEX Handoff: MarsRecon SatMAE on Olympus Mons

Last updated: 2026-04-13 (America/Chicago)

This file is a practical handoff for the current Stage A SatMAE workflow on the `akshay` branch.
It is meant to let another agent pick up quickly without re-discovering the same issues.

## What this branch is doing

- Primary Stage A model: SatMAE-style ViT-Base masked autoencoder.
- Region: Olympus Mons.
- Data mode: HiRISE `COLOR` channels only.
- Current stable path: build reusable `litData` streams on scratch, then train SatMAE from those streams.
- W&B is now integrated and works online.

This branch is no longer using the old raw patch DataLoader path for serious runs. That path was too slow and too fragile during preprocessing.

## Current branch state

- Branch: `akshay`
- Upstream tracking: `origin/akshay`
- There are uncommitted local changes that matter for this workflow.

Modified / added files for this SatMAE + litData + W&B path:

- `scripts/launch_satmae_olympus.sh`
- `src/clip/build_marsclip_cache.py`
- `src/clip/build_marsclip_litdata.py`
- `src/clip/marsclip_cache.py`
- `src/clip/marsclip_litdata.py`
- `src/clip/marsclip_patches.py`
- `src/clip/marsclip_splits.py`
- `src/clip/train_marsclip_satmae.py`
- `tests/test_marsclip_cache.py`
- `tests/test_marsclip_litdata.py`
- `tests/test_marsclip_patches.py`
- `tests/test_marsclip_splits.py`
- `tests/test_train_marsclip_satmae.py`

## Scratch layout

Stable scratch roots:

- Raw HiRISE data:
  - `/scratch/mars_hirise`
- Stage A reusable assets:
  - `/scratch/marsrecon_runs/stage_a/assets/olympus_color_only_v1`
- Stage A SatMAE runs:
  - `/scratch/marsrecon_runs/stage_a/satmae`

Important note:

- Do not overwrite anything under `/scratch/mars_hirise_dtm`.
  That belongs to a different DTM workflow.

## Current healthy litData cache

The current good cache root is:

- `/scratch/marsrecon_runs/stage_a/assets/olympus_color_only_v1/litdata_cache_v1/6b920fc15bdc92fb`

This cache was built from:

- bbox: `(-136, 12, -124, 24)`
- image size: `64`
- patch size deg: `0.005`
- split manifest:
  - `/scratch/marsrecon_runs/stage_a/assets/olympus_color_only_v1/olympus_full_splits.csv`
- patch records:
  - `/scratch/marsrecon_runs/stage_a/assets/olympus_color_only_v1/olympus_full_patch_records.pkl`
- `color_only = true`
- `dataset_normalize = true`
- `filter_invalid_patches = true`
- `dominant_obs_only = true`
- required splits: `train`, `val`

Counts from `litdata_summary.json`:

- Source train patches: `57,329`
- Kept train patches: `29,722`
- Dropped train patches: `27,607`
- Source val patches: `12,285`
- Kept val patches: `6,449`
- Dropped val patches: `5,836`

This means the current "full dataset" runs are using the full filtered train/val split, not every raw patch in the manifest.

## Current live run

The current long run is active and healthy:

- Run dir:
  - `/scratch/marsrecon_runs/stage_a/satmae/20260413/20260413_095120_olympus-satmae-vit-base-run3-online-litdata-e25`
- W&B project:
  - `akshayn3-auvsl/MarsRecon`
- W&B run URL:
  - `https://wandb.ai/akshayn3-auvsl/MarsRecon/runs/vuvtbxca`

Most recent observed status:

- phase: `training`
- epoch: `0 / 25` at first confirmation, later advanced into validation
- batch size: `64`
- epochs: `25`
- warmup epochs: `2`
- model: `mae_vit_base_patch16`
- norm pix loss: enabled
- GPU: `0`

Quick monitor command:

```bash
watch -n 5 'cat /scratch/marsrecon_runs/stage_a/satmae/20260413/20260413_095120_olympus-satmae-vit-base-run3-online-litdata-e25/progress.json'
```

## Last completed successful run

The first clean full end-to-end success was:

- `/scratch/marsrecon_runs/stage_a/satmae/20260413/20260413_094123_olympus-satmae-vit-base-run1-online-litdata-clean`

Its W&B run:

- `https://wandb.ai/akshayn3-auvsl/MarsRecon/runs/mf45a8i5`

Key metrics from `summary.json`:

- epochs: `2`
- final train loss: `0.37687564596276857`
- best / final val loss: `0.31400009151548147`

Useful artifacts:

- `checkpoints/best_checkpoint.pt`
- `checkpoints/checkpoint.pt`
- `reconstructions/reconstruction_best.png`
- `reconstructions/reconstruction_final.png`
- `history.json`
- `val_history.json`

## How to launch a new run

Use the launcher. It will reuse the good litData cache if present.

Example online run:

```bash
env \
  WANDB_API_KEY='<set this in env, do not hardcode in files>' \
  CUDA_VISIBLE_DEVICES=0 \
  WANDB_MODE=online \
  WANDB_PROJECT=MarsRecon \
  WANDB_ENTITY=akshayn3-auvsl \
  RUN_NAME=olympus_satmae_vit_base_runX \
  EPOCHS=25 \
  WARMUP_EPOCHS=2 \
  USE_LITDATA=1 \
  FILTER_INVALID_PATCHES=1 \
  DOMINANT_OBS_ONLY=1 \
  CACHE_INCLUDE_TEST=0 \
  NUM_WORKERS=24 \
  PREFETCH_FACTOR=8 \
  LITDATA_WORKERS=24 \
  bash scripts/launch_satmae_olympus.sh
```

Important:

- Pass `WANDB_API_KEY` in the environment.
  In this environment, relying on `~/.netrc` was unreliable for the real launcher.
- Keep `USE_LITDATA=1`.
- Keep `FILTER_INVALID_PATCHES=1` and `DOMINANT_OBS_ONLY=1` for apples-to-apples comparison with the successful runs.

## How to check whether a run is truly healthy

Healthy means all of the following are true:

1. `progress.json` exists in the run dir.
2. `progress.json` is updating.
3. GPU memory is allocated to the training process.
4. GPU utilization is non-zero during training.
5. W&B prints a real online run URL, not an offline fallback.

Recommended checks:

```bash
cat /scratch/marsrecon_runs/stage_a/satmae/<run_dir>/progress.json
nvidia-smi
ps -eo pid,ppid,pcpu,pmem,etimes,args | rg 'train_marsclip_satmae|wandb-core'
```

## Known failure modes and fixes

### 1. Raw patch path looked like "training" but never really trained

Cause:

- preprocessing was too slow and mostly hidden
- repeated JP2 open/reproject per patch was the bottleneck

Fix:

- switched to reusable `litData` streams on scratch

### 2. litData cache built but launcher pointed to the wrong cache hash

Cause:

- earlier launcher guessed the cache hash

Fix:

- `scripts/launch_satmae_olympus.sh` now resolves the cache directory by reading `litdata_summary.json`
- it checks metadata and required `_SUCCESS` files

### 3. W&B online silently fell back to offline

Cause:

- online auth was not being carried into the actual launcher environment

Fix:

- pass `WANDB_API_KEY` explicitly in the launch environment
- the trainer now prints when it falls back:
  - `[wandb] Online initialization failed; falling back to offline mode for this run.`

If you see that message on a run that is supposed to be online, stop it early and relaunch with the working key in env.

### 4. litData wrote temporary chunks into `/tmp/chunks`

Cause:

- data optimizer temp paths were not pinned

Fix:

- `build_marsclip_litdata.py` now sets split-local optimizer cache dirs

### 5. `--max-patches` with split manifests could become inconsistent

Cause:

- manifest patch IDs did not always line up with truncated patch-record tables

Fix:

- `align_manifest_to_patch_records(...)` in `src/clip/marsclip_splits.py`

## W&B settings

Current correct values:

- entity: `akshayn3-auvsl`
- project: `MarsRecon`

Do not store the API key in code or docs.

## DDP / Lightning / litData notes

Current state:

- SatMAE trainer is still single-process PyTorch.
- It is not yet DDP.
- It is not yet Lightning.
- The important throughput improvement already in place is `litData`.

If DDP is added later:

- initialize distributed with `init_process_group` before dataloader creation
- use `torchrun`
- do not let world size be invisible to dataset / loader setup

This was called out explicitly as a deadlock hazard by a collaborator.

## What not to change casually

- Do not remove `litData` from the serious run path.
- Do not switch back to the old raw patch DataLoader for full runs.
- Do not change both filtering and model/training hyperparameters at the same time if the goal is comparison.
- Do not overwrite existing scratch run directories.
- Do not touch `/scratch/mars_hirise_dtm`.

## Best next steps after the current long run

1. Inspect reconstruction quality from the long run.
2. Compare the new best checkpoint against the 2-epoch successful run.
3. Optionally build the `test` split stream once the Stage A recipe is locked.
4. If throughput becomes the next bottleneck, consider DDP.
5. If representation quality is the next question, run controlled SatMAE vs custom-MAE comparisons using the same filtered data regime.

## Quick file map

- Main trainer:
  - `src/clip/train_marsclip_satmae.py`
- litData builder:
  - `src/clip/build_marsclip_litdata.py`
- litData dataset bridge:
  - `src/clip/marsclip_litdata.py`
- scratch launcher:
  - `scripts/launch_satmae_olympus.sh`
- split utilities:
  - `src/clip/marsclip_splits.py`
- patch extraction:
  - `src/clip/marsclip_patches.py`

