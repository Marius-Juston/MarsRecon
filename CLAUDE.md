# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MarsRecon is a Python geospatial dataset manager for NASA's HiRISE (High Resolution Imaging Science Experiment) Mars imagery. It wraps TorchGeo's `GeoDataset` to provide spatiotemporal indexing, async downloading from NASA's Planetary Data System (PDS), radiometric calibration, and PyTorch DataLoader integration.

## Setup & Commands

This project uses `uv` as the package manager (Python 3.12 required):

```bash
uv sync                        # Install runtime dependencies
uv sync --extra dev            # Also install pytest and test deps

uv run pytest tests/ -v                    # Run all unit tests
uv run pytest tests/ -m "not integration" # Skip tests that need real data on disk
uv run pytest tests/ --cov=src --cov-report=term-missing  # With coverage

uv run python src/temp.py      # Run main pipeline (downloads + samples + plots)

# Pre-convert JP2 files to COG GeoTIFF for faster training (run once)
uv run python -m src.preprocessing --root /scratch/mars_hirise --workers 4
```

## Architecture

Source files:
- **`src/temp.py`** — active implementation; `MarsHiRISE` dataset class
- **`src/mars_hirise.py`** — original (reference only)
- **`src/hirise_sampler.py`** — `HiRISEGeoSampler`: strip-aware sampler
- **`src/preprocessing.py`** — COG conversion pipeline and geographic train/test split

### Core Class: `MarsHiRISE(GeoDataset)`

The dataset lifecycle:
1. `_verify()` — checks local files; triggers download pipeline if missing
2. `_download_index()` — fetches `RDRCUMINDEX.LBL` + `RDRCUMINDEX.TAB` from NASA PDS
3. `_load_index()` — parses PDS index into a pandas DataFrame, normalizes longitudes from [0°,360°] to [-180°,180°]
4. `_build_spatial_index()` — groups by product ID, creates a GeoDataFrame with spatial geometry + temporal interval; cached to `spatial_cache.gpkg`
5. `_download_images()` — concurrent async JP2 download using 8 worker processes × 2 concurrent requests (tuned for NASA server limits)
6. `__getitem__(index)` — spatiotemporal slice → calls `_load_tile()` → returns `{"image": Tensor, "crs": ..., "bbox": ...}`

### Sampling: `HiRISEGeoSampler` (`src/hirise_sampler.py`)

HiRISE strips are long, thin, rotated parallelograms. `RandomGeoSampler` samples within axis-aligned bounding boxes, causing 60–90% of patches to be empty. `HiRISEGeoSampler` solves this by:
1. Reading actual polygon footprints from the spatial index (built from `CORNER1-4` coordinates)
2. Pre-computing a regular grid of patch centres that are confirmed to intersect each strip polygon
3. Randomly sampling from this pre-computed set each epoch

```python
from hirise_sampler import HiRISEGeoSampler
sampler = HiRISEGeoSampler(dataset, size=0.005, length=500, units=Units.CRS)
```

### COG Conversion (`src/preprocessing.py`)

JP2 files (200MB–2.5GB) are slow for random windowed reads. `jp2_to_cog()` creates a sidecar `.tif` (same stem) in COG format with 512×512 internal tiles. `MarsHiRISE._prefer_cog()` automatically uses the sidecar when it exists. Geographic train/test split via `geographic_split(dataset.index)`.

### Download Infrastructure

- `_download_file()` — async HTTP with exponential backoff retry
- `_download_many()` / `_worker_process()` — ProcessPoolExecutor wrapping asyncio event loops
- Halts if free disk space drops below 100 GB threshold

### Radiometric Calibration (`_ProductMeta`)

Parses PDS3 `.LBL` label files to extract per-product `SCALING_FACTOR` and `OFFSET` for DN→I/F conversion. Falls back to hardcoded defaults when metadata is unavailable.

## Mars Coordinate System

- CRS: IAU 2000 Mars geographic (lon/lat decimal degrees)
- Ellipsoid: a=3,396,190m, b=3,376,200m
- All longitude values normalized to [-180°, 180°]

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `root` | Local storage directory (default: `/scratch/mars_hirise`) |
| `bbox` | (lon_min, lat_min, lon_max, lat_max) bounding box filter |
| `target` | Optional keyword filter on product names (e.g., `"Olympus"`) |
| `channels` | List from `["NEAR-INFRARED", "RED", "BLUE-GREEN"]` |
| `download` | Fetch missing files from NASA PDS if `True` |
| `reuse_cache` | Reuse cached `spatial_cache.gpkg` if `True` |

## Data Products

- `_COLOR.JP2` — 3-band mosaic (NIR, RED, BG)
- `_RED.JP2` — Single RED band, higher quality (higher TDI accumulation)

## Logging

Configured via `logger_config.json`:
- stdout: WARNING+ only
- stderr: ERROR+
- `app.log`: DEBUG+ (full trace)
