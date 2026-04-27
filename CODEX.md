# CODEX Handoff: MarsRecon Stage A + Stage B

Last updated: 2026-04-26 (America/Chicago)

This file is a practical handoff for the active workflows on the `akshay`
branch. It is intentionally short and only keeps things that have been
re-verified against on-disk artifacts.

The repo currently has two active stages:

- **Stage A**: SatMAE pretraining on Mars HiRISE color patches.
- **Stage B**: image-text embedding alignment on top of a frozen Stage A
  encoder. The text-only path is treated as **Stage B0** (a baseline harness),
  not as the final Stage B architecture.

Run this repo from `.venv` via direct `python` invocations. There are no
run-specific shell wrappers under `src/stage_b`; all canonical commands below
call the trainers and evaluators directly.

## Scratch Layout

- raw HiRISE mirror:
  - `/scratch/mars_hirise`
- Stage A assets:
  - `/scratch/marsrecon_runs/stage_a/assets`
- Stage A runs:
  - `/scratch/marsrecon_runs/stage_a/satmae`
- Stage B runs:
  - `/scratch/marsrecon_runs/stage_b/text_mae_align`
- dataset viz / clip viz / clip reports defaults:
  - `/scratch/marsrecon_runs/dataset_viz`
  - `/scratch/marsrecon_runs/clip_viz`
  - `/scratch/marsrecon_runs/clip_reports`

Do not touch:

- `/scratch/mars_hirise_dtm`

W&B defaults for both stages:

- entity: `akshayn3-auvsl`
- project: `MarsRecon`

## Stage A: SatMAE pretraining

### Active trainer

- `src/clip/train_marsclip_satmae.py`

### Current best completed run

- run dir:
  - `/scratch/marsrecon_runs/stage_a/satmae/20260420/20260420_123744_olympus-satmae-vit-base-256p8-mr50-e20-validmask-v2-strict80`
- W&B:
  - `https://wandb.ai/akshayn3-auvsl/MarsRecon/runs/a8mfnetk`
- config:
  - `image_size=256`
  - `patch_size_px=8`
  - `mask_ratio=0.5`
  - `epochs=20`
  - `valid_mask_aware=true`
  - `token_min_valid_fraction=0.8`
  - `spectral_dropout_prob=0.0`
- best val loss:
  - `0.1342`

### Stage A checkpoint used by Stage B

The Stage B trainers consume this Stage A checkpoint as their frozen visual
backbone:

```
/scratch/marsrecon_runs/stage_a/satmae/20260421/20260421_012519_olympus-satmae-vit-base-256p8-mr50-e20-validmask-v3-softweight-exp2-graypreview-v1/checkpoints/best_checkpoint.pt
```

### Other important Stage A runs

- earlier `256/p8` baseline (`token_min_valid_fraction=0.5`):
  - `/scratch/marsrecon_runs/stage_a/satmae/20260420/20260420_035331_olympus-satmae-vit-base-256p8-mr50-e20-validmask-v1`
  - best val loss `0.1479`
- `128/p8/20e` baseline:
  - `/scratch/marsrecon_runs/stage_a/satmae/20260418/20260418_015410_olympus-satmae-vit-base-128p8-mr50-e20-v1`
  - best val loss `0.1751`
- `128/p8/80e`:
  - `/scratch/marsrecon_runs/stage_a/satmae/20260418/20260418_033907_olympus-satmae-vit-base-128p8-mr50-e80-underconvergence-test`
  - best val loss `0.1510`
- `128/p4/20e`:
  - `/scratch/marsrecon_runs/stage_a/satmae/20260420/20260420_000436_olympus-satmae-vit-base-128p4-mr50-e20-v1`
  - best val loss `0.1823`

Interpretation:

- `256/p8` plus valid-mask-aware training is the current best direction.
- Tightening token validity helped more than another patch-size sweep.

### Live trainer features

- valid-mask-aware MAE masking and loss
- runtime valid-mask refinement from the image tensor itself
- single-band invalidation in `color_only` mode
- optional spectral dropout (typically run with `--spectral-dropout-prob 0.0`)

### Caveats

- litData is the real training path. Raw patch extraction is not appropriate
  for serious runs.
- Crop/flip transforms exist in code but are bypassed on the litData path, so
  the historical `recipealign` run is not a clean augmentation ablation.
- Saved reconstruction previews now denormalize, respect the refined valid
  mask, and percentile-stretch per channel; treat them as diagnostics, not as
  a formal metric.

### Healthy caches

- 128 image-size cache:
  - `/scratch/marsrecon_runs/stage_a/assets/olympus_color_only_v1/litdata_cache_v1/64e082972f642b74`
- 256 image-size cache:
  - `/scratch/marsrecon_runs/stage_a/assets/olympus_color_only_256_v1/litdata_cache_v1/2815a653281747ba`

The `256` cache is the one used by both `256/p8` valid-mask-aware runs.

### Useful Stage A commands

Inspect the current best summary:

```bash
cat /scratch/marsrecon_runs/stage_a/satmae/20260420/20260420_123744_olympus-satmae-vit-base-256p8-mr50-e20-validmask-v2-strict80/summary.json
```

List the latest Stage A summaries:

```bash
find /scratch/marsrecon_runs/stage_a/satmae -maxdepth 3 -name summary.json | sort
```

See all live Stage A CLI knobs:

```bash
.venv/bin/python src/clip/train_marsclip_satmae.py --help
```

## Stage B: image-text alignment (B0 baseline, B1a roadmap)

Stage B was reorganized from `src/stage2` into `src/stage_b`. The folder now
contains exactly three Python modules and no shell wrappers:

- `src/stage_b/align_text_mae_embeddings.py` — Stage B0 trainer
- `src/stage_b/evaluate_text_mae_alignment.py` — held-out retrieval evaluator
- `src/stage_b/T5_encoder.py` — frozen T5 encoder with HF/T5 local cache fallback

Stage B0 is the **text-only baseline**. The next architecture target is
**Stage B1a**: paired local/global crops sharing the frozen Stage A backbone,
text + location context targets, geometry/viewing features, and a CACo-style
local/global consistency loss. B0 is intentionally treated as an ablation
harness against B1a, not as the final Stage B.

### Stage B0 trainer features

- Stage-A-style run layout under `/scratch/marsrecon_runs/stage_b/text_mae_align/YYYYMMDD/...`:
  - `startup_progress.json`, `progress.json`, `history.json`,
    `summary.json`, `run_config.json`,
    `checkpoints/checkpoint.pt`, rolling `checkpoint_epoch_NNNN.pt`,
    and `best_checkpoint.pt`.
- W&B online/offline integration with early init (so a run is visible during
  cache warmup) and a canonical step metric (`trainer/global_step`).
- Split-manifest-aware train/val using
  `/scratch/marsrecon_runs/stage_a/assets/olympus_color_only_v1/olympus_full_splits.csv`.
- Stage-A-style warmup + cosine LR schedule.
- Per-epoch validation retrieval and **best-checkpoint selection by
  `val/alignment_score`** (mean of image→text and text→image MRR), not by
  train loss.
- Frozen text-embedding cache so repeated rationales aren't re-encoded.
- Frozen MAE image-feature cache so the per-step cost is just two linear
  projections + a 256×256 logits matmul.
- **False-negative-aware contrastive loss**: with only 244 unique rationale
  strings across 81,899 patches, every batch contains many same-text samples;
  off-diagonal same-text entries are masked out of the InfoNCE softmax.
- **CLIP-style stability rails**: `logit_scale.exp()` clamped at `100` (and the
  parameter clamped after every step), gradient clipping at `1.0`, and bf16
  AMP autocast on CUDA.
- **Robust shutdown**: SIGINT/SIGTERM are converted into a controlled
  shutdown that writes `checkpoints/interrupted_checkpoint.pt`, flips
  `progress.json` to `status: "aborted"`, and finishes the W&B run.
- **Cached-feature loaders use single-process settings**: when train/val are
  driven by `FrozenImageFeatureDataset`, the loaders use `num_workers=0`,
  `pin_memory=False`, `persistent_workers=False`. Multi-worker settings still
  apply to the cache warmup path.

### Stage B0 runs of record

#### 2026-04-23 baseline (online)

- run dir:
  - `/scratch/marsrecon_runs/stage_b/text_mae_align/20260423/20260423_234340_olympus-text-mae-online-v1`
- W&B:
  - `https://wandb.ai/akshayn3-auvsl/MarsRecon/runs/y9wzkaz7`
- 5 epochs, train loss `2.1253 → 1.8504` monotonic.
- Selection was still by train loss (pre-validation-checkpointing); useful as
  the B0 reference baseline.
- Balanced retrieval probes improved over training:
  - 244-way, 1 sample/text: R@1 `4.5% → 6.6%`, R@10 `30.3% → 40.2%`, median rank `21 → 15`.
  - 976-sample, 4 samples/text: R@1 `4.3% → 6.6%`, R@10 `10.3% → 17.1%`, median rank `85 → 53`.

#### 2026-04-26 split-val cosine (crashed)

- run dir:
  - `/scratch/marsrecon_runs/stage_b/text_mae_align/20260426/20260426_172729_olympus-stage-b0-splitval-cosine-v1`
- W&B run id: `cjybkq24`
- Status: process terminated mid-epoch 3 with no Python traceback (consistent
  with external SIGKILL/preemption). `summary.json` and `checkpoints/` were
  never written. `progress.json` is stuck at `status: "running"`,
  `step_in_epoch=1350/3584`. `history.json` only has epochs 1 and 2.
- Observed val signal during the 2 completed epochs was flat:
  `val/alignment_score = 0.05646 → 0.05548`.
- **Do not** treat this as a fair B0 readout — it never finished and it ran
  before the loss correctness, scaling, and stability fixes below.

#### 2026-04-26 optimized B0 (in progress)

- run dir:
  - `/scratch/marsrecon_runs/stage_b/text_mae_align/20260426/20260426_214338_olympus-stage-b0-bs256-fnmask-bf16-v1`
- W&B:
  - `https://wandb.ai/akshayn3-auvsl/MarsRecon/runs/erwozhp9`
- Designed to be the first apples-to-apples best B0 readout. Differences from
  the crashed 2026-04-26 run:
  - false-negative-mask in the contrastive loss (correctness fix for the 244
    unique-text vocabulary)
  - `batch_size=256` and `val_batch_size=256` (cached features make this free)
  - `epochs=30`, `warmup_epochs=2`, `learning_rate=1e-4` (unchanged), cosine
    decay to `min_lr=1e-6`
  - bf16 AMP autocast, gradient clipping `1.0`, logit-scale clamp at `100`
  - same Stage A backbone, same split manifest, same val sampling
    (`val_max_patches=4096`)
- Linear projector kept; B0 architecture is intentionally unchanged so it
  remains a clean ablation against the eventual B1a model.

### Canonical Stage B0 invocations

There are no shell wrappers; call the Python modules directly. The two
recipes below are the canonical "train a fresh B0 run" and "evaluate a
checkpoint on val" flows.

Train an optimized B0 run:

```bash
.venv/bin/python src/stage_b/align_text_mae_embeddings.py \
  --mae-checkpoint "/scratch/marsrecon_runs/stage_a/satmae/20260421/20260421_012519_olympus-satmae-vit-base-256p8-mr50-e20-validmask-v3-softweight-exp2-graypreview-v1/checkpoints/best_checkpoint.pt" \
  --root "/scratch/mars_hirise" \
  --bbox -136 12 -124 24 \
  --patch-size-deg 0.005 --image-size 256 --patch-size-px 8 \
  --patch-records-path "/scratch/marsrecon_runs/stage_a/assets/olympus_color_only_v1/olympus_full_patch_records.pkl" \
  --split-manifest "/scratch/marsrecon_runs/stage_a/assets/olympus_color_only_v1/olympus_full_splits.csv" \
  --batch-size 256 --val-batch-size 256 \
  --epochs 30 --warmup-epochs 2 --learning-rate 1e-4 --min-lr 1e-6 \
  --embed-dim 256 \
  --cache-image-embeddings --image-cache-batch-size 64 \
  --cache-text-embeddings \
  --val-max-patches 4096 --val-every 1 \
  --num-workers 8 --pin-memory --prefetch-factor 4 --persistent-workers \
  --use-amp --amp-dtype bf16 --grad-clip-norm 1.0 \
  --false-negative-mask --logit-scale-max 100.0 \
  --checkpoint-every 1 --progress-log-interval 10 \
  --out-root "/scratch/marsrecon_runs/stage_b/text_mae_align" \
  --wandb-mode online --wandb-project MarsRecon --wandb-entity akshayn3-auvsl \
  --run-name "olympus-stage-b0-bs256-fnmask-bf16-v1" \
  --wandb-run-name "olympus-stage-b0-bs256-fnmask-bf16-v1"
```

Notes:

- `--num-workers/--pin-memory/--prefetch-factor/--persistent-workers` apply
  to cache warmup. Once cached features are built, the train/val loaders
  automatically downgrade to single-process settings (`num_workers=0`).
- For a more conservative LR rerun, swap in `--learning-rate 5e-5`.
- For an offline-friendly variant, use `--wandb-mode offline` (or
  `--wandb-mode disabled` to skip W&B entirely).

Evaluate a Stage B0 checkpoint on the held-out val split:

```bash
.venv/bin/python src/stage_b/evaluate_text_mae_alignment.py \
  --alignment-checkpoint "/scratch/marsrecon_runs/stage_b/text_mae_align/20260426/20260426_214338_olympus-stage-b0-bs256-fnmask-bf16-v1/checkpoints/best_checkpoint.pt" \
  --root "/scratch/mars_hirise" \
  --bbox -136 12 -124 24 \
  --patch-size-deg 0.005 --image-size 256 --patch-size-px 8 \
  --patch-records-path "/scratch/marsrecon_runs/stage_a/assets/olympus_color_only_v1/olympus_full_patch_records.pkl" \
  --split-manifest "/scratch/marsrecon_runs/stage_a/assets/olympus_color_only_v1/olympus_full_splits.csv" \
  --holdout-split val \
  --max-patches 4096 \
  --batch-size 256 --num-workers 8 --pin-memory --prefetch-factor 4 \
  --output-json /tmp/stage_b_eval_val.json
```

Run with `--max-patches 0` (or omit) and the larger holdout to get the full
val readout once the run completes. To get embeddings out for downstream
analysis, add `--export-embeddings --embeddings-out path/to/embeddings.pt`.

## Roadmap

The current text-only aligner is **Stage B0**. The next milestone is
**Stage B1a** with:

- paired local/global crops centered on the same patch
- shared frozen Stage A visual backbone for both views
- text + location-context targets
- geometry / viewing features included from the start
- CACo-style local/global consistency loss
- validation metrics for image↔text, text↔image, and local↔global

Existing pieces that B1a should reuse:

- `src/clip/marsclip_patches.py` already emits location and `viewing_features`.
- The older `src/clip/marsclip_model.py` has a geo-context tower scaffold.
- The Stage B0 trainer now provides reusable infrastructure: W&B,
  split-aware train/val, checkpoint selection by validation metric,
  scratch run layout, image/text caches, and graceful shutdown.

Do not over-optimize the text-only B0 path as if it were the final Stage B.

## Repo changes worth preserving

Stage A:

- `src/clip/train_marsclip_satmae.py`
  - valid-mask-aware SatMAE path
  - runtime valid-mask refinement
  - optional spectral dropout
- `src/clip/marsclip_patches.py`
  - stricter pixel-valid logic
  - single-band invalidation in `color_only` mode
- `tests/test_train_marsclip_satmae.py`
- `tests/test_marsclip_patches.py`

Stage B:

- `src/stage_b/align_text_mae_embeddings.py`
  - split-manifest-aware train/val and balanced val sampling
  - frozen text and frozen image feature caches
  - cached-feature loaders forced to single-process
  - false-negative-aware InfoNCE with logit-scale clamp
  - bf16 AMP autocast and gradient clipping
  - SIGINT/SIGTERM monitor that writes an interrupted checkpoint
  - W&B integration with canonical `trainer/global_step`
- `src/stage_b/evaluate_text_mae_alignment.py`
  - split-manifest-aware held-out retrieval evaluation
  - optional embeddings export
- `src/stage_b/T5_encoder.py`
  - HF/T5 local cache fallback
