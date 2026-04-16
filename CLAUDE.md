# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MarsRecon is a Python geospatial dataset manager and deep-learning training framework for NASA's HiRISE (High Resolution Imaging Science Experiment) Mars imagery. It provides:

- **Dataset layer**: TorchGeo `GeoDataset` wrappers for HiRISE RDR images and DTM stereo pairs, with spatiotemporal indexing, async PDS downloading, and radiometric calibration.
- **DepthFM training pipeline**: Flow-matching monocular depth estimation adapted for Mars DTMs, using PyTorch Lightning on multi-GPU hardware.
- **MarsCLIP**: Tri-modal CLIP model for Mars imagery (image + elevation + text).

## Setup & Commands

This project uses `uv` as the package manager (Python 3.12 required):

```bash
uv sync                        # Install runtime dependencies
uv sync --extra dev            # Also install pytest and test deps

uv run pytest tests/ -v -n auto                   # Run all unit tests
uv run pytest tests/ -m "not integration" -n auto # Skip tests needing real data
uv run pytest tests/ --cov=src --cov-report=term-missing -n auto

# Run the RDR imagery pipeline (downloads + samples + plots)
uv run python src/dataset/mars_hirise.py

# Run the DTM pipeline (downloads stereo pairs + samples + plots)
uv run python src/dataset/mars_hirise_dtm.py --bbox -120 -30 150 30

# Pre-convert JP2/IMG files to COG GeoTIFF (run once before training)
uv run python -m src.dataset.preprocessing --root /scratch/mars_hirise --workers 4

# Pre-build LitData streaming chunks (fastest I/O path for training)
PYTHONPATH=src uv run python -m depth_fm.build_litdata --config configs/train_hirise.yaml --workers 96

# Launch training (4× A6000, torchrun DDP)
bash scripts/launch_train.sh --config configs/train_hirise.yaml --n_runs 1

# Data inspection / visualization (no training)
bash scripts/launch_train.sh --config configs/train_hirise.yaml --view_thumbnails
bash scripts/launch_train.sh --config configs/train_hirise.yaml --all_viz

# OmegaConf overrides are passed through directly
bash scripts/launch_train.sh training.per_gpu_batch_size=2 training.max_steps=500
```

## Source Layout

```
src/
  dataset/
    mars_hirise_base.py   — MarsHiRISEBase: shared download, indexing, sampling infra
    mars_hirise.py        — MarsHiRISE: RDR single-image dataset
    mars_hirise_dtm.py    — MarsHiRISEDTM: stereo DTM + orthoimage dataset
    hirise_sampler.py     — HiRISEGeoSampler: strip-aware patch sampler
    preprocessing.py      — JP2→COG conversion + geographic train/test split
  depth_fm/
    train_lightning.py    — Training entry point (torchrun target)
    lightning_module.py   — DepthFMLightningModule: train/val/test steps, EMA
    depthfm_adapter.py    — DepthFMHiRISEAdapterCached: map-style wrapper for DTM pairs
    litdata_datamodule.py — LitData StreamingDataset path (faster I/O)
    build_litdata.py      — Preprocess DTM patches → LitData binary chunks
    model.py              — DepthFM backbone (UNet + VAE)
    losses.py             — Velocity, normals, multi-scale gradient, photometric losses
    metrics.py            — RMSE, abs_rel, δ₁, normal angular error, photo consistency
    visualization.py      — Publication-quality figure helpers
    scalers.py            — DTM normalization strategies
  clip/
    marsclip_model.py     — Tri-modal CLIP architecture
    marsclip_dataset.py   — Dataset for CLIP training
    train_marsclip_mae.py — MAE pre-training entry point
depth-fm/                 — Original DepthFM reference implementation (upstream)
configs/                  — OmegaConf YAML configs (train_hirise.yaml is primary)
scripts/
  launch_train.sh         — Wrapper that sets NCCL/CUDA env vars and calls torchrun
```

## Architecture

### Dataset Layer: `MarsHiRISEBase` (abstract)

`mars_hirise_base.py` is the shared foundation. Both `MarsHiRISE` and `MarsHiRISEDTM` extend it. Key shared responsibilities:
- Async PDS index download (`RDRCUMINDEX.TAB` / `DTMCUMINDEX.TAB`)
- Raster-based footprint extraction via `_run_footprint_extraction()` (ProcessPoolExecutor)
- Spatial index construction → `self.index` (GeoDataFrame with `pd.IntervalIndex` datetime axis)
- COG sidecar preference (`prefer_cog()`)
- Global and regional coverage plots

### `MarsHiRISEDTM` (primary training data source)

Index is one row per stereo pair. Each row stores paths to:
- `dtm_path` — `.IMG` float32 elevation raster (1 DN = 1 m)
- `left_red_path` / `right_red_path` — RED orthoimage JP2s (1-band)
- `left_irb_path` / `right_irb_path` — IRB orthoimage JP2s (3-band: NIR, RED, BG)

`__getitem__` returns `{"elevation": (1,H,W), "left_red": (1,H,W), ..., "bounds", "crs"}`. Elevation nodata is `NaN`; orthos are calibrated to I/F [0, 1] via PDS3 `.LBL` scaling factors.

Build-time validation: pairs missing either L or R ortho are dropped; pairs where ortho footprint overlaps DTM by <75% are logged and dropped.

### `HiRISEGeoSampler`

HiRISE strips are thin rotated parallelograms. `HiRISEGeoSampler` pre-computes a grid of patch centres that actually intersect each strip's convex-hull polygon, then samples uniformly from that set each epoch. Supports train/val/test geographic splits (by longitude or latitude) and k-fold cross-validation.

```python
sampler = HiRISEGeoSampler(
    dataset, size=0.018, length=None,
    units=Units.CRS,
    split_fractions=(0.8, 0.1, 0.1),
    split_method="geographic",
    split_axis="longitude",
    split="train",
)
```

### DepthFM Training Pipeline

**Data flow**:
1. `MarsHiRISEDTM` + `HiRISEGeoSampler` → raw patches
2. `DepthFMHiRISEAdapterCached` normalizes elevation (`"relative"` = patch-centred / p98 scale; `"global"` = log-normal stats) and formats images to `[-1, 1]` float32. Each `__getitem__` randomly picks left or right ortho (stereo augmentation).
3. `LitData StreamingDataset` (preferred) or cached GDAL adapter (fallback)

**`DepthFMLightningModule`**:
- Flow-matching training: noisy elevation → velocity prediction
- Losses: velocity MSE + normals + multi-scale gradient + photometric (Lunar-Lambert rendering)
- Dual checkpoint strategy: best RMSE and best photometric consistency tracked separately
- EMA via `FasterEMAWeightAveraging`
- Round-robin figure distribution across DDP ranks to avoid rank-0 stalls

**Training config** (`configs/train_hirise.yaml`): all hyperparameters; OmegaConf dotlist overrides supported from CLI.

**Data loading priority** in `build_dataloaders()`:
1. LitData `StreamingDataset` (pre-built binary chunks, fastest)
2. `DepthFMHiRISEAdapterCached` fallback (live GDAL reads)

### COG Conversion

`preprocessing.jp2_to_cog()` writes a sidecar `.tif` with 512×512 internal tiles. Both dataset classes auto-prefer the `.tif` when present via `prefer_cog()`. Run once before training; required for good random-access I/O performance on large JP2s.

## Mars Coordinate System

- `self.crs`: IAU 2000 Mars geographic CRS — `+proj=longlat +a=3396190 +b=3376200`
- All longitudes normalized to `[-180°, 180°]`
- Each JP2/IMG uses its own per-observation Equirectangular projection; `rasterio.warp.transform_bounds` handles reprojection into the common geographic hub automatically

## Key Parameters

| Parameter            | Description                                                                   |
|----------------------|-------------------------------------------------------------------------------|
| `root`               | Local storage dir (`/scratch/mars_hirise_dtm` for DTM)                        |
| `bbox`               | `(lon_min, lat_min, lon_max, lat_max)` — **use this** before downloading DTMs |
| `include_ortho`      | Load orthoimage alongside elevation (DTM dataset)                             |
| `ortho_type`         | `"RED"` (1-band) and/or `"IRB"` (3-band)                                      |
| `ortho_scale`        | Resolution letter `"A"`–`"D"` (A=0.25m finest); `None` picks finest available |
| `normalize_elevation`| Z-score normalise elevation using external stats JSON                          |
| `download`           | Fetch from PDS. Unfiltered DTM download is >10 TB — always use `bbox`/`target`|
| `reuse_cache`        | Reuse cached `.gpkg` spatial index                                             |

## Logging

Configured via `logger_config.json`:
- stdout: WARNING+
- stderr: ERROR+
- `app.log`: DEBUG+ (full trace)

## Tests

```bash
uv run pytest tests/ -m "not integration" --cov=src --cov-report=term-missing -n auto
```

Integration tests require real data at `/scratch/mars_hirise` and are excluded from the unit suite via `-m "not integration"`.
