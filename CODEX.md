# CODEX Handoff: MarsRecon Stage A SatMAE

Last updated: 2026-04-20 (America/Chicago)

This file is a practical handoff for the current Stage A workflow on the `akshay` branch.
It is intentionally short and only keeps things that were re-verified recently.

## Current Reality

- The active Stage A trainer is:
  - `src/clip/train_marsclip_satmae.py`
- The canonical launcher is:
  - `scripts/launch_satmae_olympus.sh`
- Serious runs are scratch-backed and use litData caches.
- The current Mars-specific improvements in the trainer are real:
  - valid-mask-aware masking/loss
  - runtime valid-mask refinement from the image tensor itself
  - suppression of single-band pseudo-valid pixels in `color_only` mode
  - optional spectral dropout
- Spectral dropout is implemented but is still typically run with:
  - `--spectral-dropout-prob 0.0`

## Current Scratch Layout

- raw HiRISE mirror:
  - `/scratch/mars_hirise`
- Stage A assets:
  - `/scratch/marsrecon_runs/stage_a/assets`
- Stage A runs:
  - `/scratch/marsrecon_runs/stage_a/satmae`
- dataset viz defaults:
  - `/scratch/marsrecon_runs/dataset_viz`
- clip viz defaults:
  - `/scratch/marsrecon_runs/clip_viz`
- clip report defaults:
  - `/scratch/marsrecon_runs/clip_reports`

Do not touch:

- `/scratch/mars_hirise_dtm`

## Current Best Completed Run

Best completed Stage A reconstruction run so far:

- run dir:
  - `/scratch/marsrecon_runs/stage_a/satmae/20260420/20260420_035331_olympus-satmae-vit-base-256p8-mr50-e20-validmask-v1`
- W&B:
  - `https://wandb.ai/akshayn3-auvsl/MarsRecon/runs/robbis5a`
- config:
  - `image_size=256`
  - `patch_size_px=8`
  - `mask_ratio=0.5`
  - `epochs=20`
  - `valid_mask_aware=true`
  - `token_min_valid_fraction=0.5`
  - `spectral_dropout_prob=0.0`
- result:
  - best val loss `0.1479`

Interpretation:

- This is the strongest completed run quantitatively so far.
- It is structurally better than the earlier `128/p8` and `128/p4` runs.
- It still leaves some residual green/purple contamination near nodata-like regions.

## Important Completed Runs

### 128/p8/20e baseline

- run dir:
  - `/scratch/marsrecon_runs/stage_a/satmae/20260418/20260418_015410_olympus-satmae-vit-base-128p8-mr50-e20-v1`
- best val loss:
  - `0.1751`

### 128/p8/80e

- run dir:
  - `/scratch/marsrecon_runs/stage_a/satmae/20260418/20260418_033907_olympus-satmae-vit-base-128p8-mr50-e80-underconvergence-test`
- best val loss:
  - `0.1510`

### 128/p4/20e

- run dir:
  - `/scratch/marsrecon_runs/stage_a/satmae/20260420/20260420_000436_olympus-satmae-vit-base-128p4-mr50-e20-v1`
- best val loss:
  - `0.1823`

Interpretation:

- `128/p4` reduced visible block size but was not an overall win.
- `256/p8` plus valid-mask-aware training is the current best direction.

## What the Trainer Supports

Main trainer:

- `src/clip/train_marsclip_satmae.py`

Important CLI knobs that are actually live:

- data:
  - `--image-size`
  - `--patch-size-px`
  - `--patch-size-deg`
  - `--split-manifest`
  - `--patch-records-path`
  - `--litdata-root`
  - `--dataset-normalize`
  - `--dataset-normalization-path`
  - `--filter-invalid-patches`
  - `--dominant-obs-only`
  - `--valid-mask-aware`
  - `--token-min-valid-fraction`
- optimization:
  - `--epochs`
  - `--batch-size`
  - `--accum-iter`
  - `--blr`
  - `--lr`
  - `--min-lr`
  - `--warmup-epochs`
  - `--weight-decay`
- MAE:
  - `--mask-ratio`
  - `--norm-pix-loss`
  - `--spectral-dropout-prob`
  - `--spectral-dropout-max-channels`
- logging:
  - `--wandb-mode`
  - `--wandb-project`
  - `--wandb-entity`
  - `--save-reconstructions`
- init:
  - `--init-checkpoint`
  - `--init-mode`
  - `--init-pos-embed`

## Important Caveats

### 1. litData is the real training path

Do not assume raw patch extraction is acceptable for serious runs.
The current healthy path is scratch-backed litData.

### 2. Recipe-alignment transforms are not active on litData

The trainer has crop/flip transform code, but the litData path bypasses that wrapping.

So:

- the historical `recipealign` run is **not** a clean augmentation ablation

### 3. Previews are better now, but still diagnostic

Saved reconstruction previews are much more faithful than the old ones because they now:

- denormalize from dataset stats
- respect the refined valid mask
- apply a per-channel percentile stretch

Still, treat them as diagnostics rather than a formal metric.

## Current Healthy Caches

### 128 image-size cache

- `/scratch/marsrecon_runs/stage_a/assets/olympus_color_only_v1/litdata_cache_v1/64e082972f642b74`

### 256 image-size cache

- `/scratch/marsrecon_runs/stage_a/assets/olympus_color_only_256_v1/litdata_cache_v1/2815a653281747ba`

That `256` cache is the one used by both the best completed `256/p8` run and the current strict-mask follow-up.

## Current W&B Defaults

- entity:
  - `akshayn3-auvsl`
- project:
  - `MarsRecon`

## Useful Commands

Check the active strict run:

```bash
cat /scratch/marsrecon_runs/stage_a/satmae/20260420/20260420_123744_olympus-satmae-vit-base-256p8-mr50-e20-validmask-v2-strict80/progress.json
```

List the latest Stage A runs:

```bash
find /scratch/marsrecon_runs/stage_a/satmae -maxdepth 3 -name summary.json | sort
```

Launch the standard Olympus workflow:

```bash
bash scripts/launch_satmae_olympus.sh
```

For direct experimental control:

```bash
.venv/bin/python src/clip/train_marsclip_satmae.py --help
```

## Current Repo Changes Worth Preserving

The most important local code changes on this branch are:

- `src/clip/train_marsclip_satmae.py`
  - valid-mask-aware SatMAE path
  - runtime valid-mask refinement
  - optional spectral dropout
- `src/clip/marsclip_patches.py`
  - stricter pixel-valid logic
  - single-band invalidation in `color_only` mode
- `tests/test_train_marsclip_satmae.py`
- `tests/test_marsclip_patches.py`

## Practical Recommendation

Before choosing another architecture sweep, first compare the current strict-mask run against:

- the completed `256/p8 validmask-v1` run

The key question is whether the stricter mask logic reduces residual green contamination without damaging the stronger structural reconstruction that `256/p8` already achieved.
