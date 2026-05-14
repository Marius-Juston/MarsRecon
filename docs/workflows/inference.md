# Inference

## Single-image DTM prediction

```bash
PYTHONPATH=src uv run python scripts/inference/inference.py \
    --ckpt /path/to/best.ckpt \
    --image /path/to/PSP_XXXXXX_XXXX_RED.JP2 \
    --output /path/to/predicted_dtm.tif
```

The script runs an **Euler ODE integration** of the flow-matching velocity field, with
optional **ensemble averaging** over multiple stochastic starts for variance reduction.

## Pre-computing VAE latents

For larger-scale batched inference, latents can be pre-computed:

```bash
PYTHONPATH=src uv run python scripts/inference/precompute_latents.py \
    --root /scratch/mars_hirise --output /scratch/latents
```

This script uses the legacy filesystem dataset at `scripts/inference/dtm_dataset.py`.

## Tile-based reconstruction

For inference on full strips, predictions are emitted per-patch and then merged with
Huber-IRLS overlap blending + cosine taper:

```bash
PYTHONPATH=src uv run python scripts/reconstruction/surface_blend.py \
    --pred-dir /path/to/per_patch_preds \
    --output /path/to/strip_dtm.tif
```

See `scripts/inference/inference.py` for the CLI surface and
[`depth_fm.training.lightning_module`](../reference/depth_fm/training/lightning_module.md)
for the underlying model wrapping.
