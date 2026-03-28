# MarsRecon

A [TorchGeo](https://torchgeo.readthedocs.io/)-based PyTorch dataset for NASA HiRISE Mars imagery, designed for crater segmentation and other geospatial deep-learning tasks.

## Overview

HiRISE (High Resolution Imaging Science Experiment) aboard the Mars Reconnaissance Orbiter produces the highest-resolution images of Mars available (~25 cm/pixel RED, ~50 cm/pixel colour). This library wraps the HiRISE RDR (Reduced Data Records) hosted on the NASA PDS Imaging Node as a `GeoDataset` compatible with TorchGeo's samplers and data loaders.

**Key features:**

- Strip-aware sampling — `HiRISEGeoSampler` pre-grids valid patch centres within actual HiRISE strip polygons, avoiding 60–90% of empty-pixel patches that result from bounding-box sampling.
- Automatic radiometric calibration — `I/F = DN × SCALING_FACTOR + OFFSET`, clipped to `[0, 1]`, using per-product `.LBL` files.
- Spatial index with polygon footprints — corner-coordinate polygons intersected with each JP2's reprojected bounds for accurate footprints.
- COG-ready — `src/preprocessing.py` converts JP2 files to Cloud Optimised GeoTIFF for 10–100× faster random-access reads.
- Geographic train/test split — longitude- or latitude-blocked splits prevent spatial leakage.

## Requirements

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Real HiRISE data under a local mirror (see [Data](#data))

## Installation

```bash
uv sync           # installs main + dev dependencies from pyproject.toml
```

## Data

HiRISE RDR products are served from the NASA PDS Imaging Node. The dataset expects files mirroring the PDS hierarchy under `root` (default `/scratch/mars_hirise`):

```
<root>/
    RDRCUMINDEX.LBL
    RDRCUMINDEX.TAB
    MROHR_0001/
        DATA/
            PSP/
                ORB_001400_001499/
                    PSP_001430_1780/
                        PSP_001430_1780_COLOR.JP2
                        PSP_001430_1780_COLOR.LBL
                        PSP_001430_1780_RED.JP2
                        PSP_001430_1780_RED.LBL
```

Pass `download=True` on first use to auto-download files. Mirrors via `rsync` or `wget -r` are fully compatible.

## Quick start

```python
from torch.utils.data import DataLoader
from src.temp import MarsHiRISE
from src.hirise_sampler import HiRISEGeoSampler
from torchgeo.samplers import Units

# Olympus Mons region, all three colour channels
dataset = MarsHiRISE(
    bbox=(-136, 12, -124, 24),
    channels=["NEAR-INFRARED", "RED", "BLUE-GREEN"],
    download=False,
    reuse_cache=True,
)

# Strip-aware sampler: every patch guaranteed to intersect real data
sampler = HiRISEGeoSampler(dataset, size=0.005, length=200, units=Units.CRS)
loader = DataLoader(dataset, sampler=sampler)

for sample in loader:
    image = sample["image"]   # (C, H, W) float32 in [0, 1]
    print(image.shape)
    break
```

`size=0.005` degrees ≈ 593 pixels ≈ 296 m at the equator.

## Channels

| Channel | Source file | Notes |
|---|---|---|
| `NEAR-INFRARED` | `_COLOR.JP2` band 1 | ~900 nm |
| `RED` | `_COLOR.JP2` band 2 or `_RED.JP2` | RED.JP2 used when only RED requested (higher fidelity) |
| `BLUE-GREEN` | `_COLOR.JP2` band 3 | ~500 nm |

When `_COLOR.JP2` is absent for an observation, NIR and BG channels are zero-filled; the RED channel falls back to `_RED.JP2`.

## Sampler

`HiRISEGeoSampler` in `src/hirise_sampler.py` replaces TorchGeo's `RandomGeoSampler` for HiRISE data:

```python
sampler = HiRISEGeoSampler(
    dataset,
    size=0.005,           # patch size in degrees (or pixels with units=Units.PIXELS)
    length=500,           # patches per epoch; defaults to total valid centres
    stride=0.003,         # centre-to-centre spacing (default = size, non-overlapping)
    units=Units.CRS,
)
```

At construction, it pre-computes a regular grid of candidate centres for each strip polygon and keeps only those whose corresponding patch intersects the polygon (not just its bounding box). Iteration draws uniformly from this set — O(1) per sample.

## COG conversion (optional, recommended)

Converting JP2 files to Cloud Optimised GeoTIFF dramatically speeds up random-window reads:

```bash
python -m src.preprocessing --root /scratch/mars_hirise --workers 4
```

The dataset transparently prefers `.tif` COG sidecars when they exist alongside `.JP2` files.

## Running

```bash
# Generate coverage map + sample patches (saves to Figures/)
uv run python src/mars_hirise.py
```

## Tests

```bash
# Unit tests (no HiRISE data required)
uv run pytest tests/ -m "not integration" -v

# With coverage
uv run pytest tests/ -m "not integration" --cov=src --cov-report=term-missing

# Preprocessing-only coverage (100%)
uv run pytest tests/test_preprocessing.py --cov=preprocessing --cov-report=term-missing

# Integration tests (require real data at /scratch/mars_hirise)
uv run pytest tests/ -m integration -v
```

### Test suite overview

| Test file | What it covers |
|---|---|
| `test_preprocessing.py` | All of `src/preprocessing.py` — 100% line coverage; 73 tests |
| `test_download.py` | Async download helpers, retry logic, disk-space guard |
| `test_spatial_index.py` | Spatial index construction (Cases A–D), GeoPackage cache |
| `test_dataset.py` | `MarsHiRISE.__getitem__`, tile loading, radiometric calibration |
| `test_sampler.py` | `HiRISEGeoSampler` grid pre-computation and epoch sampling |
| `test_lbl_parsing.py` | PDS3 `.LBL` label parsing and `_ProductMeta` extraction |
| `test_coordinates.py` | Longitude normalisation and CRS helpers |
| `conftest.py` | Shared fixtures: `mars_crs`, `strip_polygon`, `synthetic_lbl`, `synthetic_corner_row` |
| `helpers.py` | `make_mock_dataset()` — minimal `GeoSampler`-compatible mock |

### `test_preprocessing.py` in detail

The file uses **synthetic data only** — no real HiRISE files are required.

| Test class | Functions under test | Key scenarios |
|---|---|---|
| `TestAvailableMemoryBytes` | `_available_memory_bytes` | `/proc/meminfo` read, sysconf fallback, 8 GiB hard fallback, missing `MemAvailable` line |
| `TestSafeWorkerCount` | `_safe_worker_count` | Ample/tight RAM, never-below-1 floor, never-exceeds-requested cap, largest-files-first sampling, warning logged when capped |
| `TestIsCorruptJp2Error` | `_is_corrupt_jp2_error` | All recognised corrupt tokens, case-insensitivity, non-corrupt I/O errors |
| `TestFilterMaker` | `filter_maker` | Records at/below/above the configured level |
| `TestWorkerInit` | `_worker_init` | SIGINT restored to `SIG_DFL`, `basicConfig` called at INFO |
| `TestJp2ToCog` | `jp2_to_cog` | Success path, skip-existing, overwrite, corrupted JP2 (delete+None), ungeoreferenced input (no spurious warnings) |
| `TestJp2ToCogErrorPaths` | `jp2_to_cog` | Non-corrupt `RasterioIOError` (source preserved), generic `Exception` (source preserved, COG/tmp cleaned up) |
| `TestConvertAll` | `convert_all` | Counts accuracy, corrupt-file deletion, valid conversion, pre-existing COG skip, empty directory |
| `TestConvertAllExtra` | `convert_all` | `skipped` counter (COG older than JP2), `overwrite=True` reconverts, worker future exception → `failed`, `KeyboardInterrupt` → `SystemExit(130)` + `shutdown(cancel_futures=True)` |
| `TestGeographicSplit` | `geographic_split` | Sizes sum, no overlap, test-fraction accuracy, reproducibility, different seeds differ, longitude/latitude axes |
| `TestGeographicSplitExtra` | `geographic_split` | Returns `GeoDataFrame`, CRS preserved, small/large `test_fraction` (block-count `max(5, ...)` boundary), `n_test_blocks ≥ 1` guarantee |
| `TestCLI` | `__main__` block | Empty-root run, `--overwrite` flag, `basicConfig` fallback when `logger_config.json` absent |

## Architecture

| File                    | Purpose |
|-------------------------|---|
| `src/mars_hirise.py`    | `MarsHiRISE` — main `GeoDataset` subclass; index loading, spatial index, tile loading, radiometric calibration |
| `src/hirise_sampler.py` | `HiRISEGeoSampler` — strip-polygon-aware geospatial sampler |
| `src/preprocessing.py`  | JP2 → COG conversion pipeline; geographic train/test split |
| `tests/conftest.py`     | Shared fixtures: `mars_crs`, `strip_polygon`, `synthetic_lbl`, `synthetic_corner_row` |
| `tests/helpers.py`      | `make_mock_dataset()` — minimal GeoSampler-compatible mock |

### CRS design

The dataset CRS is the Mars IAU 2000 geographic CRS (`+proj=longlat +a=3396190 +b=3376200`). Each HiRISE JP2 has its own per-observation Equirectangular projection; rasterio reprojects into the common geographic CRS at load time. `self.res` is a scalar float in degrees/pixel (`1.0 / 118_502.26` ≈ 8.44 × 10⁻⁶ °/px at HiRISE native resolution).

### Spatial index

Built once at dataset construction and cached as a GeoPackage (`spatial_cache{suffix}_v2.gpkg`; suffix encodes any `target`/`bbox` filters). For each observation:

1. Compute `corners_polygon ∩ JP2_bounds` (accurate strip polygon clipped to the JP2's actual coverage) — **Case A**
2. Fall back to JP2 bounding box if no corners or degenerate intersection — **Case B**
3. Fall back to cumulative index min/max bbox if no JP2 file available — **Cases C/D**

A `_SPATIAL_TOL = 1e-5°` tolerance absorbs floating-point rounding between the index geometry and rasterio's recomputed bounds at load time.
