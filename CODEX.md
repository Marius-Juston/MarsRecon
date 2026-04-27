# CODEX Handoff: MarsRecon Stage A + Stage B

Last updated: 2026-04-26 (America/Chicago)

This file is a practical handoff for the active workflows in the repo. The
two stages currently live on different branches:

- Stage A (SatMAE) work landed on `akshay`.
- Stage B (text/MAE alignment, B0 baseline, B1a roadmap) lives on `jay`,
  which is the branch this file is maintained on.

It is intentionally short and only keeps things that have been re-verified
against on-disk artifacts.

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

### Stage B0+ knobs (added 2026-04-26)

The trainer now exposes a small set of orthogonal CLI flags so the same
script covers both B0 and the B0+ ablations without needing forks. Defaults
preserve the original B0 behavior, so existing checkpoints and recipes still
work.

- `--projector-type {linear,mlp}` (default `linear`).
  - For `mlp`: `--projector-hidden-dim` (default `768`),
    `--projector-depth` (default `2`, total `Linear` layers including the
    output projection), `--projector-dropout` (default `0.0`). LayerNorm +
    GELU between hidden layers.
- `--image-pool {cls,mean_patch,cls_plus_mean}` (default `cls`).
  - `cls_plus_mean` concatenates CLS and mean-patch pooled SatMAE
    outputs, doubling the image-feature dim (768 → 1536). The frozen image
    cache rebuilds at the requested pool, so swapping pools forces a
    one-time warmup.
- `--loss-type {infonce,prototype}` (default `infonce`).
  - `prototype` scores each image against all 244 unique text-prototype
    embeddings (computed by passing the cached unique-text features
    through the text projector each step). Requires
    `--cache-text-embeddings`; ignores `--false-negative-mask` because it
    operates on a fixed 244-way classifier.
- `--label-smoothing FLOAT` (default `0.0`).
  - Smoothing is masked-aware: when `--false-negative-mask` is on, the
    smoothing distribution is computed only over valid (non-masked)
    columns, so masking + smoothing coexist without producing `inf` loss.
- `--balanced-sampler / --no-balanced-sampler` (default off).
  - Class-stratified order over the global text-class id. Each contiguous
    window of `num_classes` (244 today) emits at most one sample per class,
    so any batch ≥ 244 covers every class. Disables `shuffle` because the
    sampler already produces an interleaved permutation.
- `--ema-decay FLOAT` (default `0.0`, disabled).
  - EMA over the aligner's parameters and buffers. Validation and the
    `best_checkpoint.pt` use the EMA weights; rolling per-epoch
    checkpoints save the live weights and the EMA shadow side-by-side
    (`ema_state` field in the checkpoint).

Checkpoints now also persist `aligner_config`, `image_pool`, `loss_type`, and
`projector_type`, so the evaluator can reconstruct any of these
architectures without extra flags. The evaluator gained
`--use-ema/--no-use-ema` (default on); when EMA weights are present they are
applied automatically, falling back to the live weights otherwise.

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

#### 2026-04-26 optimized B0 (completed)

- run dir:
  - `/scratch/marsrecon_runs/stage_b/text_mae_align/20260426/20260426_214338_olympus-stage-b0-bs256-fnmask-bf16-v1`
- W&B:
  - `https://wandb.ai/akshayn3-auvsl/MarsRecon/runs/erwozhp9`
- First apples-to-apples best B0 readout. Differences from the crashed
  2026-04-26 run:
  - false-negative-mask in the contrastive loss (correctness fix for the 244
    unique-text vocabulary)
  - `batch_size=256` and `val_batch_size=256` (cached features make this free)
  - `epochs=30`, `warmup_epochs=2`, `learning_rate=1e-4` (unchanged), cosine
    decay to `min_lr=1e-6`
  - bf16 AMP autocast, gradient clipping `1.0`, logit-scale clamp at `100`
  - same Stage A backbone, same split manifest, same val sampling
    (`val_max_patches=4096`)
- Linear projector kept; B0 architecture intentionally unchanged so it remains
  a clean ablation against the upcoming B0+/proto runs and the eventual B1a.
- Final state (from `summary.json` + `eval_val_4096.json`):
  - status `completed`, 30/30 epochs.
  - Best `val/alignment_score = 0.13958` at epoch 28 (selection metric).
  - Final train loss `4.098` — large because the loss is the symmetric
    InfoNCE *over a 256-way batch with same-text duplicates masked out*, so
    the floor is closer to `log(244) ≈ 5.5` than `log(256) ≈ 5.55`. Train
    loss alone is not the right success metric here.
  - Held-out val (4,096 patches), best checkpoint:
    - image→text: R@1 0.146, R@10 0.146, MRR 0.156, median rank 139.
    - text→image: R@1 0.040, R@5 0.177, R@10 0.313, MRR 0.123, median rank 25.
- Takeaway: validation retrieval improved over the 2026-04-23 baseline
  on text→image, but i2t plateaued. Next step is to add capacity (MLP
  projector), regularization (label smoothing, EMA), batch composition
  (class-balanced sampler), and an architectural ablation on pooling /
  loss type — see "B0+ follow-up runs" below.

#### 2026-04-26 B0+ follow-up runs (completed parallel ablations)

Two follow-up runs completed cleanly in parallel. Both kept the Stage A
backbone, split manifest, bf16 + grad-clip + logit-scale-clamp stability
rails, and the same 4,096-patch split-manifest val protocol. They differ in
the variables they swept:

- **Run A — `olympus-stage-b0plus-mlp-bal-ema-ls-v1`** (regularize + capacity):
  - same data and pooling as the previous best (CLS, InfoNCE).
  - `--projector-type mlp --projector-hidden-dim 768 --projector-depth 2`
  - `--balanced-sampler` (class-stratified ordering over the 244 rationales)
  - `--ema-decay 0.999` (EMA-tracked aligner used for val + best checkpoint)
  - `--label-smoothing 0.05` (mask-aware so it stays finite under FN-mask)
  - longer schedule: `--epochs 60 --warmup-epochs 3 --learning-rate 5e-5
    --min-lr 1e-7`.
  - GPU: `cuda:0`.
  - run dir:
    `/scratch/marsrecon_runs/stage_b/text_mae_align/20260426/20260426_230004_olympus-stage-b0plus-mlp-bal-ema-ls-v1`.
  - W&B: `https://wandb.ai/akshayn3-auvsl/MarsRecon/runs/sm7ud0n9`.
  - status: completed, 60/60 epochs.
  - best checkpoint: epoch 50 by `val/alignment_score = 0.53046`.
  - standalone val eval (`eval_val_4096.json`, best checkpoint):
    image→text R@1 0.379, R@10 0.387, MRR 0.394, median rank 19;
    text→image R@1 0.509, R@10 0.939, MRR 0.667, median rank 1.
  - full holdout evals, best checkpoint:
    - full val (`eval_val_full.json`, 12,285 patches): image→text R@1
      0.349, R@10 0.357, MRR 0.357, median rank 73; text→image R@1
      0.514, R@10 0.936, MRR 0.659, median rank 1.
    - full test (`eval_test_full.json`, 12,285 patches): image→text R@1
      0.335, R@10 0.345, MRR 0.343, median rank 73; text→image R@1
      0.539, R@10 0.929, MRR 0.673, median rank 1.
  - critical read: this is the new B0 run of record by a very large margin.
    It strongly suggests the cheap B0+ levers (capacity + balanced sampler +
    EMA + smoothing + longer lower-LR schedule) solved most of the B0
    underfitting problem. It does **not** prove open-vocabulary semantic
    alignment; the task is still a 244-rationale classification-like
    retrieval problem.
- **Run B — `olympus-stage-b0-proto-clsmean-mlp-v1`** (architecture ablation):
  - `--image-pool cls_plus_mean` (concat CLS + mean-patch pooled SatMAE
    output, image-feature dim 1536). The image cache is rebuilt at this
    pool config, so the run pays a one-time warmup cost.
  - `--loss-type prototype` — image is classified against all 244
    text-prototype embeddings instead of a 256-way contrastive subset; this
    removes the small-vocabulary false-negative problem at the source.
  - same MLP projector + balanced sampler + EMA + label smoothing as Run A.
  - same 60-epoch / 5e-5 / cosine schedule as Run A.
  - GPU: `cuda:1`.
  - run dir:
    `/scratch/marsrecon_runs/stage_b/text_mae_align/20260426/20260426_230008_olympus-stage-b0-proto-clsmean-mlp-v1`.
  - W&B: `https://wandb.ai/akshayn3-auvsl/MarsRecon/runs/84q7ssq1`.
  - status: completed, 60/60 epochs.
  - best checkpoint: epoch 60 by `val/alignment_score = 0.30946`.
  - standalone val eval (`eval_val_4096.json`, best checkpoint):
    image→text R@1 0.192, R@10 0.192, MRR 0.204, median rank 81;
    text→image R@1 0.238, R@10 0.778, MRR 0.415, median rank 3.
  - critical read: prototype loss + CLS+mean pooling is better than the
    linear B0 baseline but far worse than Run A. Treat this path as a useful
    negative/partial ablation. Do not make it the next default.

Comparison against the previous optimized linear B0 (`erwozhp9`,
`eval_val_4096_recheck.json`): image→text MRR improved from 0.156 to 0.394
with Run A; text→image MRR improved from 0.123 to 0.667; median ranks
improved from 139/25 to 19/1. That is too large to ignore, but the win is
still on the 244-rationale B0 harness, not the final B1a target.

Recommended next action:

- Freeze Run A as the B0+ run of record and evaluate it on the full val split
  and test split before any architectural claims.
- Do **not** spend another broad run on prototype loss yet; if revisited,
  isolate variables (`cls_plus_mean` with InfoNCE, `prototype` with CLS) so
  the failure can be attributed cleanly.
- Move implementation effort to Stage B1a next: paired local/global crops,
  location/context inputs, geometry/viewing features, and local/global
  consistency. B0+ has now done its job as a strong baseline harness.

### Canonical Stage B0 invocations

There are no shell wrappers; call the Python modules directly. The recipes
below are the canonical "train a fresh B0 run", "B0+ regularize + capacity",
"B0 prototype ablation", and "evaluate a checkpoint on val" flows.

Train the original optimized B0 (linear projector, InfoNCE, CLS pool):

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

Train **Run A** (B0+ regularize + capacity, MLP / balanced / EMA / smooth /
60 epochs):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python src/stage_b/align_text_mae_embeddings.py \
  --mae-checkpoint "/scratch/marsrecon_runs/stage_a/satmae/20260421/20260421_012519_olympus-satmae-vit-base-256p8-mr50-e20-validmask-v3-softweight-exp2-graypreview-v1/checkpoints/best_checkpoint.pt" \
  --root "/scratch/mars_hirise" \
  --bbox -136 12 -124 24 \
  --patch-size-deg 0.005 --image-size 256 --patch-size-px 8 \
  --patch-records-path "/scratch/marsrecon_runs/stage_a/assets/olympus_color_only_v1/olympus_full_patch_records.pkl" \
  --split-manifest "/scratch/marsrecon_runs/stage_a/assets/olympus_color_only_v1/olympus_full_splits.csv" \
  --batch-size 256 --val-batch-size 256 \
  --epochs 60 --warmup-epochs 3 --learning-rate 5e-5 --min-lr 1e-7 \
  --embed-dim 256 \
  --projector-type mlp --projector-hidden-dim 768 --projector-depth 2 \
  --balanced-sampler --ema-decay 0.999 --label-smoothing 0.05 \
  --image-pool cls --loss-type infonce --false-negative-mask \
  --cache-image-embeddings --image-cache-batch-size 64 \
  --cache-text-embeddings \
  --val-max-patches 4096 --val-every 1 \
  --num-workers 8 --pin-memory --prefetch-factor 4 --persistent-workers \
  --use-amp --amp-dtype bf16 --grad-clip-norm 1.0 --logit-scale-max 100.0 \
  --checkpoint-every 5 --progress-log-interval 10 \
  --out-root "/scratch/marsrecon_runs/stage_b/text_mae_align" \
  --wandb-mode online --wandb-project MarsRecon --wandb-entity akshayn3-auvsl \
  --run-name "olympus-stage-b0plus-mlp-bal-ema-ls-v1" \
  --wandb-run-name "olympus-stage-b0plus-mlp-bal-ema-ls-v1"
```

Train **Run B** (CLS+mean pool, prototype loss, MLP, balanced, EMA, 60
epochs):

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python src/stage_b/align_text_mae_embeddings.py \
  --mae-checkpoint "/scratch/marsrecon_runs/stage_a/satmae/20260421/20260421_012519_olympus-satmae-vit-base-256p8-mr50-e20-validmask-v3-softweight-exp2-graypreview-v1/checkpoints/best_checkpoint.pt" \
  --root "/scratch/mars_hirise" \
  --bbox -136 12 -124 24 \
  --patch-size-deg 0.005 --image-size 256 --patch-size-px 8 \
  --patch-records-path "/scratch/marsrecon_runs/stage_a/assets/olympus_color_only_v1/olympus_full_patch_records.pkl" \
  --split-manifest "/scratch/marsrecon_runs/stage_a/assets/olympus_color_only_v1/olympus_full_splits.csv" \
  --batch-size 256 --val-batch-size 256 \
  --epochs 60 --warmup-epochs 3 --learning-rate 5e-5 --min-lr 1e-7 \
  --embed-dim 256 \
  --projector-type mlp --projector-hidden-dim 768 --projector-depth 2 \
  --balanced-sampler --ema-decay 0.999 --label-smoothing 0.05 \
  --image-pool cls_plus_mean --loss-type prototype \
  --cache-image-embeddings --image-cache-batch-size 64 \
  --cache-text-embeddings \
  --val-max-patches 4096 --val-every 1 \
  --num-workers 8 --pin-memory --prefetch-factor 4 --persistent-workers \
  --use-amp --amp-dtype bf16 --grad-clip-norm 1.0 --logit-scale-max 100.0 \
  --checkpoint-every 5 --progress-log-interval 10 \
  --out-root "/scratch/marsrecon_runs/stage_b/text_mae_align" \
  --wandb-mode online --wandb-project MarsRecon --wandb-entity akshayn3-auvsl \
  --run-name "olympus-stage-b0-proto-clsmean-mlp-v1" \
  --wandb-run-name "olympus-stage-b0-proto-clsmean-mlp-v1"
```

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

Stage B (lives on the `jay` branch):

- `src/stage_b/align_text_mae_embeddings.py`
  - split-manifest-aware train/val and balanced val sampling
  - frozen text and frozen image feature caches
  - cached-feature loaders forced to single-process
  - false-negative-aware InfoNCE with logit-scale clamp
  - bf16 AMP autocast and gradient clipping
  - SIGINT/SIGTERM monitor that writes an interrupted checkpoint
  - W&B integration with canonical `trainer/global_step`
  - **B0+ knobs**: linear/MLP projector, CLS / mean-patch / CLS+mean
    image pooling, InfoNCE / prototype loss, mask-aware label smoothing,
    class-balanced sampler, EMA-tracked aligner weights (used for
    validation and the best checkpoint).
- `src/stage_b/evaluate_text_mae_alignment.py`
  - split-manifest-aware held-out retrieval evaluation
  - optional embeddings export
  - reconstructs MLP / image-pool / EMA architectures from checkpoint
    metadata; `--use-ema/--no-use-ema` toggles EMA-vs-live weights.
- `src/stage_b/T5_encoder.py`
  - HF/T5 local cache fallback
