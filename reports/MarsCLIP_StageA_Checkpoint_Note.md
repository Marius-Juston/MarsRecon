# MarsCLIP Stage A Checkpoint Note

This note records the Stage A training runs used to select a reusable MAE checkpoint.

## Environment

- Date: `2026-03-29`
- Preferred Stage A runtime: `GPU`
- GPUs verified on host:
  - `nvidia-smi` reported four `NVIDIA RTX 6000 Ada Generation` GPUs
  - PyTorch reported `cuda_available=True` and `device_count=4` outside the sandbox
- Preferred training run device:
  - `CUDA_VISIBLE_DEVICES=0`
  - effective device: `cuda:0`

One nuance: early in-sandbox checks reported no GPU visibility. The actual training and reporting commands that matter for Stage A selection were run outside the sandbox, where CUDA was available. The first run below was a CPU validation run. A later CUDA-backed run on the same machine superseded it and is now the preferred Stage A checkpoint.

## CPU Validation Run

Output directory:

- `/tmp/marsclip_artifacts/stage_a/training/cpu_validation`

Command:

```bash
.venv/bin/python src/train_marsclip_mae.py \
  --root /scratch/mars_hirise \
  --bbox -136 12 -124 24 \
  --out-dir /tmp/marsclip_artifacts/stage_a/training/cpu_validation \
  --image-size 64 \
  --patch-size-deg 0.005 \
  --patch-size-px 16 \
  --max-patches 64 \
  --batch-size 4 \
  --num-steps 40 \
  --learning-rate 1e-4 \
  --weight-decay 1e-2 \
  --mask-ratio 0.75 \
  --preview-items 4 \
  --encoder-dim 64 \
  --encoder-depth 2 \
  --encoder-heads 4 \
  --decoder-dim 32 \
  --decoder-depth 1 \
  --decoder-heads 4 \
  --checkpoint-every 10 \
  --preview-every 20 \
  --device cpu
```

Key outputs:

- `/tmp/marsclip_artifacts/stage_a/training/cpu_validation/checkpoint.pt`
- `/tmp/marsclip_artifacts/stage_a/training/cpu_validation/best_checkpoint.pt`
- `/tmp/marsclip_artifacts/stage_a/training/cpu_validation/history.json`
- `/tmp/marsclip_artifacts/stage_a/training/cpu_validation/loss_curve.png`
- `/tmp/marsclip_artifacts/stage_a/training/cpu_validation/preview_step_000020.png`
- `/tmp/marsclip_artifacts/stage_a/training/cpu_validation/preview_step_000040.png`
- `/tmp/marsclip_artifacts/stage_a/training/cpu_validation/preview_final.png`

## Loss Summary

From `/tmp/marsclip_artifacts/stage_a/training/cpu_validation/history.json`:

- Step `1`: `0.3556479215621948`
- Step `10`: `0.3118983805179596`
- Step `20`: `0.2847018837928772`
- Step `30`: `0.26112452149391174`
- Step `40`: `0.23192331194877625`

Run summary from `/tmp/marsclip_artifacts/stage_a/training/cpu_validation/summary.json`:

- Initial loss: `0.3556479215621948`
- Final loss: `0.23192331194877625`
- Best loss: `0.23192331194877625`
- Best step: `40`

This is the first Stage A run where the loss curve is clearly meaningful rather than a one-step smoke artifact.

## CPU Checkpoint Comparison

Two embedding sanity reports were generated on the same Olympus Mons subset:

- Early checkpoint:
  - checkpoint: `/tmp/marsclip_artifacts/stage_a/training/cpu_validation/checkpoint_step_000010.pt`
  - report dir: `/tmp/marsclip_artifacts/stage_a/embedding/cpu_step10`
- Selected checkpoint:
  - checkpoint: `/tmp/marsclip_artifacts/stage_a/training/cpu_validation/best_checkpoint.pt`
  - report dir: `/tmp/marsclip_artifacts/stage_a/embedding/cpu_best`

Artifacts in each report:

- `embeddings.pt`
- `embedding_metadata.json`
- `nearest_neighbors.png`
- `embedding_scatter.png`
- `embedding_summary.json`

Observed report summaries:

- Step `10` mean first-neighbor similarity: `0.9999764561653137`
- Best checkpoint mean first-neighbor similarity: `0.9999743700027466`

The nearest-neighbor similarity metric is nearly saturated on this tiny sanity subset, so it is not strong enough on its own to choose a checkpoint. It is still useful as a qualitative sanity artifact.

## Selection Rule

Stage A checkpoint selection rule for this repo:

1. Choose the checkpoint with the minimum observed Stage A reconstruction loss.
2. Confirm that its reconstruction preview and embedding sanity artifacts are not obviously degenerate.
3. Prefer the earlier checkpoint only if the qualitative embedding artifacts are clearly better despite a worse training loss.

For the CPU validation run, that rule selects:

- `/tmp/marsclip_artifacts/stage_a/training/cpu_validation/best_checkpoint.pt`

## CUDA-Backed Run

Output directory:

- `/tmp/marsclip_artifacts/stage_a/training/gpu_selected`

Command:

```bash
env CUDA_VISIBLE_DEVICES=0 .venv/bin/python src/train_marsclip_mae.py \
  --root /scratch/mars_hirise \
  --bbox -136 12 -124 24 \
  --out-dir /tmp/marsclip_artifacts/stage_a/training/gpu_selected \
  --image-size 64 \
  --patch-size-deg 0.005 \
  --patch-size-px 16 \
  --max-patches 128 \
  --batch-size 8 \
  --num-steps 80 \
  --learning-rate 1e-4 \
  --weight-decay 1e-2 \
  --mask-ratio 0.75 \
  --preview-items 4 \
  --encoder-dim 64 \
  --encoder-depth 2 \
  --encoder-heads 4 \
  --decoder-dim 32 \
  --decoder-depth 1 \
  --decoder-heads 4 \
  --checkpoint-every 20 \
  --preview-every 40 \
  --device auto
```

CUDA visibility was verified outside the sandbox:

- Host GPUs visible via `nvidia-smi`
- PyTorch reported `cuda_available=True` and `device_count=4`
- During training, `nvidia-smi` showed the Stage A job using GPU `0`

Key outputs:

- `/tmp/marsclip_artifacts/stage_a/training/gpu_selected/checkpoint.pt`
- `/tmp/marsclip_artifacts/stage_a/training/gpu_selected/best_checkpoint.pt`
- `/tmp/marsclip_artifacts/stage_a/training/gpu_selected/history.json`
- `/tmp/marsclip_artifacts/stage_a/training/gpu_selected/loss_curve.png`
- `/tmp/marsclip_artifacts/stage_a/training/gpu_selected/preview_step_000040.png`
- `/tmp/marsclip_artifacts/stage_a/training/gpu_selected/preview_step_000080.png`
- `/tmp/marsclip_artifacts/stage_a/training/gpu_selected/preview_final.png`
- `/tmp/marsclip_artifacts/stage_a/embedding/gpu_best/nearest_neighbors.png`
- `/tmp/marsclip_artifacts/stage_a/embedding/gpu_best/embedding_scatter.png`

Loss summary:

- Step `1`: `0.3595007061958313`
- Step `20`: `0.3163852095603943`
- Step `40`: `0.2706204950809479`
- Step `60`: `0.21238133311271667`
- Step `79`: `0.16813473403453827`
- Step `80`: `0.16833315789699554`

Run summary:

- Batch size: `8`
- Initial loss: `0.3595007061958313`
- Final loss: `0.16833315789699554`
- Best loss: `0.16813473403453827`
- Best step: `79`

Embedding sanity report:

- checkpoint: `/tmp/marsclip_artifacts/stage_a/training/gpu_selected/best_checkpoint.pt`
- report dir: `/tmp/marsclip_artifacts/stage_a/embedding/gpu_best`
- mean first-neighbor similarity: `0.9999740123748779`

## Final Decision

Stage A is considered complete, and the preferred checkpoint is now the CUDA-backed checkpoint:

- `/tmp/marsclip_artifacts/stage_a/training/gpu_selected/best_checkpoint.pt`

What that means:

- We have a reusable pretrained vision encoder checkpoint.
- We have a training history and loss curve from a meaningful run.
- We have checkpoint comparison artifacts.
- We have an explicit checkpoint-selection rule recorded for the final report.
- We have verified that the repo can use the machine's GPUs outside the sandbox.

The next step is Stage B1:

- paired local/global crop sampling
- reuse of the Stage A encoder
- workflow-aligned location / geometry / text context path
