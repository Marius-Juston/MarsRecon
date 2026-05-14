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
PYTHONPATH=src uv run python -m dataset.core.rdr

# Run the DTM pipeline (downloads stereo pairs + samples + plots)
PYTHONPATH=src uv run python -m dataset.core.dtm --bbox -120 -30 150 30

# Pre-convert JP2/IMG files to COG GeoTIFF (run once before training)
PYTHONPATH=src uv run python -m dataset.preprocessing.cog_conversion --root /scratch/mars_hirise --workers 4

# Pre-build LitData streaming chunks (fastest I/O path for training)
PYTHONPATH=src uv run python scripts/training/build_litdata.py --config configs/train_hirise.yaml --workers 96

# Launch training (4× A6000, torchrun DDP)
bash scripts/training/launch_train.sh --config configs/train_hirise.yaml --n_runs 1

# Data inspection / visualization (no training)
bash scripts/training/launch_train.sh --config configs/train_hirise.yaml --view_thumbnails
bash scripts/training/launch_train.sh --config configs/train_hirise.yaml --all_viz

# OmegaConf overrides are passed through directly
bash scripts/training/launch_train.sh training.per_gpu_batch_size=2 training.max_steps=500
```

`PYTHONPATH=src` is required because the repo treats `src/` as the import root; `tests/conftest.py` adds it automatically for pytest, but ad-hoc scripts need it set explicitly.

## Source Layout

```
src/
  dataset/                       # public API: from dataset import MarsHiRISE, MarsHiRISEDTM, HiRISEGeoSampler
    core/
      base.py                    — MarsHiRISEBase: shared download, indexing, footprint, viz infra
      rdr.py                     — MarsHiRISE: RDR single-image dataset
      dtm.py                     — MarsHiRISEDTM: stereo DTM + orthoimage dataset
    sampling/
      sampler.py                 — HiRISEGeoSampler: strip-aware patch sampler with geographic splits
      geometry.py                — Valid-center region + patch-packing for the "optimal" sampler mode
    preprocessing/
      cog_conversion.py          — JP2/IMG → Cloud-Optimized GeoTIFF conversion
    stats/
      compute_stats.py           — Multi-GPU Welford accumulator over live GDAL reads
      compute_stats_litdata.py   — Fast NumPy-only path over pre-built LitData chunks
    validation/
      sampling_diagnostics.py    — Diagnostic PNGs for strip geometry, sampler coverage, calibration

  depth_fm/                      # public API: from depth_fm import MarsDepthFM, DepthFMLightningModule
    models/
      mars_depthfm.py            — MarsDepthFM wrapper + `build_model()` + `load_sd21_backend()`
      experimental.py            — DebugUNet, ModulatedMicroFlowNet (optional backbones)
      unet/                      — CompVis LDM UNetModel (upstream code, treat as frozen)
    training/
      train_lightning.py         — Entry point: `torchrun -m depth_fm.training.train_lightning`
      lightning_module.py        — DepthFMLightningModule: train/val/test, EMA, dual checkpoint
    data/
      adapter.py                 — DepthFMHiRISEAdapterCached: map-style wrapper over MarsHiRISEDTM
      datamodule.py              — Lightning DataModule + LitData StreamingDataset
      scalers.py                 — Elevation normalization strategies (Global Fixed / Global Log / Adaptive)
      image_processing/          — Algorithms used by the adapter:
        mask_ops.py              —   erode_valid_mask
        void_filling.py          —   nearest-neighbor / smart-diffusion / kriging / GMRF fillers
        seam_detection.py        —   SeamResult + detect_seam_artifact + is_tin_artifact + helpers
        sun_vector.py            —   estimate_sun_vector_irls / _ols
        terrain.py               —   compute_topographic_residual
    objectives/
      losses.py                  — PhotoclinometricLoss, AbsoluteDepthLoss, LaplacianLoss, OrdinalRankingLoss, CombinedLoss
      metrics.py                 — DTMMetrics, affine_align, compute_depth_metrics, compute_photo_consistency
    flow/
      noise.py                   — Flow-matching noise schedule (cosine_alpha_bar, q_sample)
    viz/
      train_viz.py               — Publication-quality figures (triptychs, error maps, normals)
      debug_viz.py               — Training-side analysis viz; imported wholesale by train_lightning.py

  clip/                          — Tri-modal CLIP model + MAE pretraining

configs/                         — OmegaConf YAML configs (train_hirise.yaml is primary)
depth-fm/                        — Original DepthFM reference implementation (upstream)

scripts/                         — Entry-point scripts (not part of the importable library)
  training/
    launch_train.sh              — Wrapper that sets NCCL/CUDA env vars and calls torchrun
    run_ablation.py              — Loss-component ablation orchestrator
    build_litdata.py             — Build LitData chunks in processed training format (sun-vector, residuals)
    build_litdata_raw.py         — Build LitData chunks in raw format (elevation/bounds/CRS only)
  inference/
    inference.py                 — Single-image DTM prediction (Euler ODE, optional ensemble)
    precompute_latents.py        — Pre-compute VAE latents to disk (uses dtm_dataset.py)
    dtm_dataset.py               — Legacy filesystem dataset used only by precompute_latents.py
  visualization/
    paper_training_dynamics.py   — W&B convergence + lunar-Lambert weight evolution
    qualitative_results.py       — Publication panel (ortho | pred | GT | error | normals | render)
    hirise_targeting_analysis.py — HiRISE rationale analysis (wordclouds, science themes, maps)
    sampler_comparison.py        — Simple vs. optimal sampler comparison + benchmarks
    dataset_histograms.py        — Overlayed per-channel histograms
    dataset_statistics.py        — CV/DRCR LaTeX table from dataset_stats.json
    normalization_analysis.py    — Linear vs. log normalization clipping metrics + figures
  architecture/
    marsclip_diagram.py          — MarsCLIP architecture graphviz
    depthfm_pipeline_diagram.py  — DepthFM pipeline graphviz with embedded image nodes
  reconstruction/
    surface_blend.py             — Huber-IRLS overlap + cosine-taper blending of patch predictions
  common/
    io_utils.py                  — load_geotiff, load_json, load_csv
    viz_utils.py                 — apply_paper_style, save_fig
```

## Where to look (LLM routing table)

When working on common tasks, this table is the fast path:

| Task                                    | File                                                   |
|-----------------------------------------|--------------------------------------------------------|
| Add/modify a loss                       | `src/depth_fm/objectives/losses.py`                    |
| Add/modify a metric                     | `src/depth_fm/objectives/metrics.py`                   |
| Change training loop / Lightning step   | `src/depth_fm/training/lightning_module.py`            |
| Change training entry point / CLI       | `src/depth_fm/training/train_lightning.py`             |
| Add/change a model backbone             | `src/depth_fm/models/mars_depthfm.py`                  |
| Try an experimental backbone            | `src/depth_fm/models/experimental.py`                  |
| Change normalization strategy           | `src/depth_fm/data/scalers.py`                         |
| Change data adapter / GDAL reads        | `src/depth_fm/data/adapter.py`                         |
| Void filling (kriging / GMRF / diffusion)| `src/depth_fm/data/image_processing/void_filling.py`  |
| Seam / TIN artifact detection           | `src/depth_fm/data/image_processing/seam_detection.py` |
| Sun-vector estimation                   | `src/depth_fm/data/image_processing/sun_vector.py`     |
| Change LitData streaming                | `src/depth_fm/data/datamodule.py`                      |
| Flow-matching noise schedule            | `src/depth_fm/flow/noise.py`                           |
| Training-side analysis viz              | `src/depth_fm/viz/debug_viz.py`                        |
| Publication figures                     | `src/depth_fm/viz/train_viz.py`                        |
| Change sampling / split logic           | `src/dataset/sampling/sampler.py`                      |
| Change DTM dataset semantics            | `src/dataset/core/dtm.py`                              |
| Change RDR dataset semantics            | `src/dataset/core/rdr.py`                              |
| Change PDS download / footprint         | `src/dataset/core/base.py`                             |
| JP2 → COG conversion                    | `src/dataset/preprocessing/cog_conversion.py`          |
| Dataset statistics                      | `src/dataset/stats/compute_stats.py`                   |
| Pre-build training chunks               | `scripts/training/build_litdata.py`                    |
| Inference on a single image             | `scripts/inference/inference.py`                       |
| New publication figure                  | `scripts/visualization/`                               |
| Reproduce ablation                      | `scripts/training/run_ablation.py`                     |

## Architecture

### Dataset Layer: `MarsHiRISEBase` (abstract)

`src/dataset/core/base.py` is the shared foundation. Both `MarsHiRISE` (RDR) and `MarsHiRISEDTM` (DTM) extend it. Key shared responsibilities:
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

HiRISE strips are thin rotated parallelograms. `HiRISEGeoSampler` (in `src/dataset/sampling/sampler.py`) pre-computes a grid of patch centres that actually intersect each strip's convex-hull polygon, then samples uniformly from that set each epoch. Supports train/val/test geographic splits (by longitude or latitude) and k-fold cross-validation.

```python
from dataset import HiRISEGeoSampler
from torchgeo.samplers import Units

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
2. `DepthFMHiRISEAdapterCached` (`src/depth_fm/data/adapter.py`) normalizes elevation (`"relative"` = patch-centred / p98 scale; `"global"` = log-normal stats) and formats images to `[-1, 1]` float32. Each `__getitem__` randomly picks left or right ortho (stereo augmentation).
3. `LitData StreamingDataset` (preferred) or cached GDAL adapter (fallback) — both in `src/depth_fm/data/datamodule.py`.

**`DepthFMLightningModule`** (`src/depth_fm/training/lightning_module.py`):
- Flow-matching training: noisy elevation → velocity prediction
- Losses: velocity MSE + normals + multi-scale gradient + photometric (Lunar-Lambert rendering)
- Dual checkpoint strategy: best RMSE and best photometric consistency tracked separately
- EMA via `FasterEMAWeightAveraging`
- Round-robin figure distribution across DDP ranks to avoid rank-0 stalls

**Training config** (`configs/train_hirise.yaml`): all hyperparameters; OmegaConf dotlist overrides supported from CLI.

**Data loading priority** in `build_dataloaders()` (`src/depth_fm/training/train_lightning.py`):
1. LitData `StreamingDataset` (pre-built binary chunks, fastest)
2. `DepthFMHiRISEAdapterCached` fallback (live GDAL reads)

**Visualization split**: heavy training-side viz (XGBoost failure prediction, residual analysis, Pareto plots) lives in `src/depth_fm/viz/debug_viz.py` and is imported wholesale by `train_lightning.py` via `from depth_fm.viz.debug_viz import *`. Publication figures live in `src/depth_fm/viz/train_viz.py`.

### COG Conversion

`dataset.preprocessing.cog_conversion.jp2_to_cog()` writes a sidecar `.tif` with 512×512 internal tiles. Both dataset classes auto-prefer the `.tif` when present via `prefer_cog()`. Run once before training; required for good random-access I/O performance on large JP2s.

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

## Refactor notes (load-bearing for LLM agents)

Several files exceed 1000 LOC and carry an inline `[REFACTOR NOTE]` comment at the top suggesting a future split:

- `src/depth_fm/training/lightning_module.py` (1982 LOC)
- `src/depth_fm/objectives/losses.py` (1226 LOC)
- `src/depth_fm/viz/debug_viz.py` (3382 LOC)
- `src/dataset/core/base.py` (1369 LOC)

These were left intact during the structural refactor; treat them as candidates if you're already touching them for an unrelated reason. Don't proactively split unless asked.

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

**Known broken after refactor:** ~48 unit tests in `test_mars_hirise_unit.py`, `test_preprocessing.py`, `test_download.py`, `test_sampler.py`, `test_depthfm.py` use string-based `mock.patch("dataset.X.Y")` and `caplog.at_level(logger="dataset.X")` that still reference pre-refactor module paths. The paths need updating to `dataset.core.base`, `dataset.preprocessing.cog_conversion`, `dataset.sampling.sampler`, `depth_fm.training.lightning_module`, etc. Functionality is unchanged — only the test patches reference stale names.
