# MarsRecon

A [TorchGeo](https://torchgeo.readthedocs.io/)-based PyTorch dataset for NASA HiRISE Mars imagery, designed for crater
segmentation and other geospatial deep-learning tasks.

# ISSUES

## When I implemented the GlobalLogScaler for the DTM instead of the linear scaler, I forgot to descale the features for the Lunar-Lambert renderings!!! To be aware, I am unsure about the impacts of this effect.

## Overview

HiRISE (High Resolution Imaging Science Experiment) aboard the Mars Reconnaissance Orbiter produces the
highest-resolution images of Mars available (~25 cm/pixel RED, ~50 cm/pixel colour). This library wraps the HiRISE RDR (
Reduced Data Records) hosted on the NASA PDS Imaging Node as a `GeoDataset` compatible with TorchGeo's samplers and data
loaders.

**Key features:**

- Strip-aware sampling — `HiRISEGeoSampler` pre-grids valid patch centres within actual HiRISE strip polygons, avoiding
  60–90% of empty-pixel patches that result from bounding-box sampling.
- Automatic radiometric calibration — `I/F = DN × SCALING_FACTOR + OFFSET`, clipped to `[0, 1]`, using per-product
  `.LBL` files.
- Spatial index with polygon footprints — convex hull of non-zero pixels intersected with each JP2's reprojected bounds
  for accurate footprints.
- COG-ready — `src/dataset/preprocessing.py` converts JP2 files to Cloud Optimised GeoTIFF for 10–100× faster
  random-access reads.
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

HiRISE RDR products are served from the NASA PDS Imaging Node. The dataset expects files mirroring the PDS hierarchy
under `root` (default `/scratch/mars_hirise`):

```
<root>/
    RDRCUMINDEX.LBL
    RDRCUMINDEX.TAB
    images/
        PSP_001430_1780_COLOR.JP2
        PSP_001430_1780_COLOR.LBL
        PSP_001430_1780_RED.JP2
        PSP_001430_1780_RED.LBL
```

Pass `download=True` on first use to auto-download files. Mirrors via `rsync` or `wget -r` are fully compatible.

## Quick start

```python
from torch.utils.data import DataLoader
from dataset.mars_hirise import MarsHiRISE
from dataset.hirise_sampler import HiRISEGeoSampler
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
    image = sample["image"]  # (C, H, W) float32 in [0, 1]
    print(image.shape)
    break
```

`size=0.005` degrees ≈ 593 pixels ≈ 296 m at the equator.

## Channels

| Channel         | Source file                       | Notes                                                     |
|-----------------|-----------------------------------|-----------------------------------------------------------|
| `NEAR-INFRARED` | `_COLOR.JP2` band 1               | ~900 nm                                                   |
| `RED`           | `_COLOR.JP2` band 2 or `_RED.JP2` | `_RED.JP2` used when only RED requested (higher fidelity) |
| `BLUE-GREEN`    | `_COLOR.JP2` band 3               | ~500 nm                                                   |

When `_COLOR.JP2` is absent for an observation, available channels fall back to `_RED.JP2` where possible.

## Sampler

`HiRISEGeoSampler` in `src/dataset/hirise_sampler.py` replaces TorchGeo's `RandomGeoSampler` for HiRISE data:

```python
sampler = HiRISEGeoSampler(
    dataset,
    size=0.005,           # patch size in degrees (or pixels with units=Units.PIXELS)
    length=500,           # patches per epoch; defaults to total valid centres
    stride=0.003,         # centre-to-centre spacing (default = size, non-overlapping)
    units=Units.CRS,
)
```

At construction, it pre-computes a regular grid of candidate centres for each strip polygon and keeps only those whose
corresponding patch intersects the polygon (not just its bounding box). Iteration draws uniformly from this set — O(1)
per sample.

## Dataset statisitics

```bash
time PYTHONPATH=src uv run -m src.dataset.compute_dataset_stats
time PYTHONPATH=src uv run -m src.dataset.compute_dataset_stats --dtm
```

## COG conversion (optional, recommended)

Converting JP2 files to Cloud Optimised GeoTIFF dramatically speeds up random-window reads:

```bash
time PYTHONPATH=src uv run python -m src.dataset.preprocessing --root /scratch/mars_hirise --workers 4
time PYTHONPATH=src uv run python -m src.dataset.preprocessing --root /scratch/mars_hirise_dtm --workers 4
```

The dataset transparently prefers `.tif` COG sidecars when they exist alongside `.JP2` files.

## LitData

To have even faster dataset throughput you can convert the information for the liData format

```bash
PYTHONPATH=src uv run -m src.depth_fm.build_litdata
```

The sun view is probably doing to be wrong due to the GPU vs CPU computation, this can be validated using

```bash
bash scripts/launch_train.sh configs/train_hirise.yaml 1 4 --view_loss_physics 
```

this is an important thing to run otherwise you will not have the correct losses.

## Running

```bash
# Generate coverage map + sample patches (saves to Figures/)
uv run python src/dataset/mars_hirise.py
```

## Tests

```bash
# Unit tests (no HiRISE data required)
uv run pytest tests/ -m "not integration" -v -n auto

# With coverage
uv run pytest tests/ -m "not integration" --cov=src --cov-report=term-missing -n auto

# Integration tests (require real data at /scratch/mars_hirise)
uv run pytest tests/ -m integration -v -n auto
```

### Coverage

All three library modules are at **100% line coverage** across 274 unit tests:

| Module                  | Statements | Coverage |
|-------------------------|------------|----------|
| `src/mars_hirise.py`    | 828        | 100%     |
| `src/hirise_sampler.py` | 80         | 100%     |
| `src/preprocessing.py`  | 161        | 100%     |

### Test suite overview

| Test file                  | Tests | What it covers                                                                                  |
|----------------------------|-------|-------------------------------------------------------------------------------------------------|
| `test_mars_hirise_unit.py` | 100   | `MarsHiRISE` — dataset init, spatial index, tile loading, plotting, download pipeline, `main()` |
| `test_preprocessing.py`    | 73    | All of `src/dataset/preprocessing.py` — JP2→COG conversion, geographic split, CLI               |
| `test_download.py`         | 22    | Async download helpers, retry logic, disk-space guard, stop-event handling                      |
| `test_sampler.py`          | 23    | `HiRISEGeoSampler` grid pre-computation, stride, pixel units, reproducibility                   |
| `test_coordinates.py`      | 16    | Longitude normalisation and CRS helpers                                                         |
| `test_dataset.py`          | 18    | `MarsHiRISE.__getitem__`, tile loading, radiometric calibration                                 |
| `test_spatial_index.py`    | 14    | Spatial index construction (Cases A–D), GeoPackage cache                                        |
| `test_lbl_parsing.py`      | 12    | PDS3 `.LBL` label parsing and `_ProductMeta` extraction                                         |
| `conftest.py`              | —     | Shared fixtures: `mars_crs`, `strip_polygon`, `synthetic_lbl`, `synthetic_corner_row`           |
| `helpers.py`               | —     | `make_mock_dataset()` — minimal `GeoSampler`-compatible mock                                    |

### `test_preprocessing.py` in detail

The file uses **synthetic data only** — no real HiRISE files are required.

| Test class                 | Functions under test      | Key scenarios                                                                                                                                                                      |
|----------------------------|---------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `TestAvailableMemoryBytes` | `_available_memory_bytes` | `/proc/meminfo` read, sysconf fallback, 8 GiB hard fallback, missing `MemAvailable` line                                                                                           |
| `TestSafeWorkerCount`      | `_safe_worker_count`      | Ample/tight RAM, never-below-1 floor, never-exceeds-requested cap, largest-files-first sampling, warning logged when capped                                                        |
| `TestIsCorruptJp2Error`    | `_is_corrupt_jp2_error`   | All recognised corrupt tokens, case-insensitivity, non-corrupt I/O errors                                                                                                          |
| `TestFilterMaker`          | `filter_maker`            | Records at/below/above the configured level                                                                                                                                        |
| `TestWorkerInit`           | `_worker_init`            | SIGINT restored to `SIG_DFL`, `basicConfig` called at INFO                                                                                                                         |
| `TestJp2ToCog`             | `jp2_to_cog`              | Success path, skip-existing, overwrite, corrupted JP2 (delete+None), ungeoreferenced input (no spurious warnings)                                                                  |
| `TestJp2ToCogErrorPaths`   | `jp2_to_cog`              | Non-corrupt `RasterioIOError` (source preserved), generic `Exception` (source preserved, COG/tmp cleaned up)                                                                       |
| `TestConvertAll`           | `convert_all`             | Counts accuracy, corrupt-file deletion, valid conversion, pre-existing COG skip, empty directory                                                                                   |
| `TestConvertAllExtra`      | `convert_all`             | `skipped` counter (COG older than JP2), `overwrite=True` reconverts, worker future exception → `failed`, `KeyboardInterrupt` → `SystemExit(130)` + `shutdown(cancel_futures=True)` |
| `TestGeographicSplit`      | `geographic_split`        | Sizes sum, no overlap, test-fraction accuracy, reproducibility, different seeds differ, longitude/latitude axes                                                                    |
| `TestGeographicSplitExtra` | `geographic_split`        | Returns `GeoDataFrame`, CRS preserved, small/large `test_fraction` (block-count `max(5, ...)` boundary), `n_test_blocks ≥ 1` guarantee                                             |
| `TestCLI`                  | `__main__` block          | Empty-root run, `--overwrite` flag, `basicConfig` fallback when `logger_config.json` absent                                                                                        |

## Architecture

| File                       | Purpose                                                                                                        |
|----------------------------|----------------------------------------------------------------------------------------------------------------|
| `src/mars_hirise.py`       | `MarsHiRISE` — main `GeoDataset` subclass; index loading, spatial index, tile loading, radiometric calibration |
| `src/hirise_sampler.py`    | `HiRISEGeoSampler` — strip-polygon-aware geospatial sampler                                                    |
| `src/preprocessing.py`     | JP2 → COG conversion pipeline; geographic train/test split                                                     |
| `src/validate_sampling.py` | Diagnostic script for visualising sampler hit-rate (not a library module)                                      |
| `tests/conftest.py`        | Shared fixtures: `mars_crs`, `strip_polygon`, `synthetic_lbl`, `synthetic_corner_row`                          |
| `tests/helpers.py`         | `make_mock_dataset()` — minimal GeoSampler-compatible mock                                                     |

### CRS design

The dataset CRS is the Mars IAU 2000 geographic CRS (`+proj=longlat +a=3396190 +b=3376200`). Each HiRISE JP2 has its own
per-observation Equirectangular projection; rasterio reprojects into the common geographic CRS at load time. `self.res`
is a scalar float in degrees/pixel (`1.0 / 118_502.26` ≈ 8.44 × 10⁻⁶ °/px at HiRISE native resolution).

### Spatial index

Built once at dataset construction and cached as a GeoPackage (`spatial_cache{suffix}_v3.gpkg`; suffix encodes any
`target`/`bbox` filters). For each observation the geometry is determined in priority order:

1. **Case A** — convex hull of non-zero pixels extracted from the JP2 (most accurate)
2. **Case B** — JP2 bounding box when the hull has too few non-zero pixels
3. **Case C** — cumulative index min/max bbox when no JP2 file is available
4. **Case D** — corner-coordinate polygon from the PDS index as a last resort

A `_SPATIAL_TOL = 1e-5°` tolerance absorbs floating-point rounding between the index geometry and rasterio's recomputed
bounds at load time.

## Additional Vizualiation

To generate the full `marsrise_dataset.mmd` MarsHiRISE dataset architecture flow diagram as,

```bash
npm install -g @mermaid-js/mermaid-cli
mmdc -i reports/marsrise_dataset.mmd -o architecture.pdf -b transparent -f
mmdc -i reports/preprocessing.mmd -o preprocessing.pdf -b transparent -f
```

To generate the overview of the MarsRecond architecutre overview:

```bash
uv sync --optional viz
uv run scripts/marsrecon_architecture.py
```

To generate the LaTeX tables for the dataset statistics

```bash
uv run scripts/generate_dataset_stats_table.py dataset_stats/image/dataset_stats.json
```

To generate the validations for the sampling:

```bash
PYTHONPATH=src uv run python src/dataset/validate_sampling.py --root /scratch/mars_hirise --bbox -136 12 -124 24 --patch-size 0.005 --n-thumbnails 16 --n-hist-patches 30 --out validation/
2026
```

