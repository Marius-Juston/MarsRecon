"""HiRISE preprocessing utilities.

Provides:

* :func:`jp2_to_cog` — Convert a single JP2 to a Cloud-Optimized GeoTIFF
  (COG) sidecar.  COGs use internal 512×512 tiling so rasterio can decompress
  only the tiles that overlap a small query window, giving 10–100× faster
  random-access reads compared to JPEG2000 during ML training.

* :func:`convert_all` — Batch-convert all JP2 files under a root directory
  using a process pool.

* :func:`geographic_split` — Split a spatial index GeoDataFrame into train /
  test sets along a geographic axis to prevent spatial data leakage for crater
  segmentation models.

CLI usage::

    # Convert all JP2s to COG GeoTIFFs (run once before training)
    uv run python -m src.preprocessing --root /scratch/mars_hirise --workers 4

    # Overwrite existing .tif sidecars
    uv run python -m src.preprocessing --root /scratch/mars_hirise --workers 4 --overwrite
"""

import concurrent.futures
import gc
import logging
import multiprocessing
import os
import pathlib
import signal
import warnings
from collections.abc import Iterator
from typing import Callable

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.enums
import rasterio.shutil

from dataset.mars_hirise_base import MARS_PROJECTED_CRS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COG creation settings
# ---------------------------------------------------------------------------

#: Overview levels built into each COG for multi-scale access.
_OVERVIEW_LEVELS: list[int] = [2, 4, 8, 16]

#: Resampling method used when building overviews.
_OVERVIEW_RESAMPLING = rasterio.enums.Resampling.average

#: Base rasterio profile applied to every COG output (integer / uint imagery).
_COG_CREATION_OPTIONS: dict = {
    "driver": "GTiff",
    "compress": "deflate",
    "predictor": 2,  # horizontal differencing — good for imagery
    "tiled": True,
    "blockxsize": 512,
    "blockysize": 512,
    "copy_src_overviews": True,
    "bigtiff": "IF_SAFER",
}

#: COG profile for DTM .IMG files (float32 elevation data).
#: Uses predictor=3 (floating-point differencing) instead of predictor=2
#: (horizontal integer differencing) for better compression of float rasters.
_COG_CREATION_OPTIONS_FLOAT: dict = {
    **_COG_CREATION_OPTIONS,
    "predictor": 3,  # floating-point differencing — better for float32 DTMs
}

#: HiRISE DTMs use IEEE float32 minimum as nodata sentinel.
_DTM_NODATA: float = -3.4028226550889045e+38


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _available_memory_bytes() -> int:
    """Return available (not just free) system RAM in bytes.

    Tries ``/proc/meminfo`` first (Linux), then falls back to the POSIX
    ``sysconf`` interface.  Returns 8 GiB when neither source is readable so
    that callers can still make a conservative decision.
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB → bytes
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError):
        return 8 * 1024 ** 3  # conservative fallback: 8 GiB


def _safe_worker_count(jp2_files: list[pathlib.Path], requested: int) -> int:
    """Return the largest worker count that fits in available RAM.

    Each worker fully decompresses one JP2 at a time.  HiRISE JP2 files
    compress at roughly 3-6× so we use 5× the on-disk size as a conservative
    per-worker memory estimate, then reserve 25 % of available RAM for the OS
    and GDAL internals.

    The returned value is clamped to ``[1, requested]``.
    """
    if not jp2_files:
        return requested

    # Sample up to 8 files (sorted by size, worst-case first) to estimate
    # average decompressed footprint.
    sample = sorted(jp2_files, key=lambda p: p.stat().st_size, reverse=True)[:8]
    avg_on_disk = sum(p.stat().st_size for p in sample) / len(sample)

    # JP2000 → raw numpy is typically 3-6× the compressed file size for
    # HiRISE imagery; 5× is a conservative (safe) upper bound.
    per_worker_bytes = avg_on_disk * 5

    available = _available_memory_bytes()
    usable = available * 0.75  # keep 25 % headroom

    safe = max(1, int(usable / per_worker_bytes))
    chosen = min(safe, requested)

    if chosen < requested:
        logger.warning(
            "Capping workers at %d (requested %d): estimated %.1f GiB per JP2, "
            "%.1f GiB usable RAM.",
            chosen,
            requested,
            per_worker_bytes / 1024 ** 3,
            usable / 1024 ** 3,
        )
    else:
        logger.debug(
            "Worker count %d fits within %.1f GiB usable RAM "
            "(%.1f GiB estimated per JP2).",
            chosen,
            usable / 1024 ** 3,
            per_worker_bytes / 1024 ** 3,
        )
    return chosen


# ---------------------------------------------------------------------------
# Per-file conversion
# ---------------------------------------------------------------------------
def filter_maker(level: str) -> Callable:
    numeric = getattr(logging, level)

    def _filter(record: logging.LogRecord) -> bool:
        return record.levelno <= numeric

    return _filter


def _is_corrupt_jp2_error(exc: Exception) -> bool:
    """Return True when *exc* indicates a corrupted JPEG2000 bitstream.

    Distinguishes genuine decode failures from transient I/O problems such as
    disk-full or permission errors, which should not delete the source file.
    """
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "opj_decode",
            "ireadblock",
            "read failed",
            "tile part length",
            "not recognized as being in a supported file format",
        )
    )


def _cog_path(jp2_path: pathlib.Path) -> pathlib.Path:
    """Return the expected COG sidecar path for *jp2_path* (same stem, .tif)."""
    return jp2_path.with_suffix(".tif")


def jp2_to_cog(jp2_path: pathlib.Path, overwrite: bool = False) -> pathlib.Path | None:
    """Convert a single HiRISE JP2 to a Cloud-Optimized GeoTIFF sidecar.

    The COG is written alongside the source JP2 with the same stem and a
    ``.tif`` extension.  :meth:`~temp.MarsHiRISE.prefer_cog` will
    automatically use it when it exists, bypassing the slower JP2 path.

    The conversion proceeds in two passes:

    1. Write a temporary intermediate GeoTIFF so that overviews can be built
       on a writeable dataset (rasterio requires this).
    2. Copy the intermediate file to the final COG path with
       ``copy_src_overviews=True`` to embed the overviews efficiently.

    Args:
        jp2_path: Path to the source JPEG2000 file.
        overwrite: If ``False`` (default), skip files that already have a
            ``.tif`` sidecar.

    Returns:
        Path to the output COG on success, or ``None`` if conversion failed.
    """
    cog = _cog_path(jp2_path)

    if cog.exists() and not overwrite:
        logger.debug("COG sidecar already exists, skipping: %s", cog.name)
        return cog

    tmp = cog.with_suffix(".tmp.tif")
    try:
        # Some HiRISE JP2s carry no embedded geotransform; rasterio emits
        # NotGeoreferencedWarning on open (identity matrix assumed) and again
        # on the intermediate write.  Both are expected — we copy whatever
        # spatial metadata is present — so suppress them for the whole pass.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*geotransform.*|.*identity matrix.*",
            )
            with rasterio.open(jp2_path) as src:
                profile = src.profile.copy()
                profile.update(
                    driver="GTiff",
                    tiled=True,
                    blockxsize=512,
                    blockysize=512,
                    bigtiff="IF_SAFER",  # switch to 64-bit offsets when >4 GiB
                )
                # The intermediate is a scratch file deleted in `finally` —
                # compressing it with deflate is the main conversion bottleneck
                # (CPU-intensive on gigabytes of data that are immediately
                # discarded).  Compression is applied only in the final
                # rasterio.shutil.copy call below.
                for key in ("lossless", "quality", "compress", "predictor"):
                    profile.pop(key, None)

                logger.debug("Writing intermediate GeoTIFF for %s …", jp2_path.name)
                with rasterio.open(tmp, "w", **profile) as dst:
                    for band_idx in src.indexes:
                        band_data = src.read(band_idx)
                        dst.write(band_data, band_idx)
                        del band_data  # release decompressed array before next band
                    dst.build_overviews(_OVERVIEW_LEVELS, _OVERVIEW_RESAMPLING)
                    dst.update_tags(
                        ns="rio_overview", resampling=_OVERVIEW_RESAMPLING.name
                    )

        # Source JP2 is now closed; release any lingering references before the
        # second pass so the decompressed pixel data can be reclaimed.
        gc.collect()

        # Second pass: copy to final COG with overviews embedded.
        # Use _COG_CREATION_OPTIONS directly — it already carries compress,
        # predictor, tiling, and copy_src_overviews.  Deriving opts from
        # `profile` would omit compression because we stripped it above.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*geotransform.*|.*identity matrix.*",
            )
            rasterio.shutil.copy(tmp, cog, **_COG_CREATION_OPTIONS)

        logger.info("COG written: %s", cog.name)
        return cog

    except rasterio.errors.RasterioIOError as exc:
        if _is_corrupt_jp2_error(exc):
            logger.warning(
                "Corrupted JP2 detected — deleting so it can be re-downloaded: %s",
                jp2_path.name,
            )
            jp2_path.unlink(missing_ok=True)
        else:
            logger.error("COG conversion failed for %s: %s", jp2_path.name, exc)
        cog.unlink(missing_ok=True)
        return None
    except Exception as exc:
        logger.error("COG conversion failed for %s: %s", jp2_path.name, exc)
        cog.unlink(missing_ok=True)
        return None

    finally:
        tmp.unlink(missing_ok=True)


def _img_cog_path(img_path: pathlib.Path) -> pathlib.Path:
    """Return the expected COG sidecar path for a DTM .IMG file."""
    return img_path.with_suffix(".tif")


def img_to_cog(img_path: pathlib.Path, overwrite: bool = False) -> pathlib.Path | None:
    """Convert a single HiRISE DTM .IMG (PDS3 float32) to a Cloud-Optimized GeoTIFF.

    DTM ``.IMG`` files are flat binary rasters with attached PDS3 labels.
    Unlike JP2 orthoimages, they are not compressed, so the conversion is
    mainly about adding internal 512×512 tiling and overviews for fast
    random-access reads during ML training.

    Float32 elevation data uses ``predictor=3`` (floating-point differencing)
    for better deflate compression.  The nodata sentinel
    ``-3.4028226550889045e+38`` is preserved in the output GeoTIFF metadata.

    Args:
        img_path: Path to the PDS3 ``.IMG`` file.
        overwrite: If ``False`` (default), skip files that already have a
            ``.tif`` sidecar.

    Returns:
        Path to the output COG on success, or ``None`` if conversion failed.
    """
    cog = _img_cog_path(img_path)

    if cog.exists() and not overwrite:
        logger.debug("COG sidecar already exists, skipping: %s", cog.name)
        return cog

    tmp = cog.with_suffix(".tmp.tif")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*geotransform.*|.*identity matrix.*",
            )
            with rasterio.open(img_path) as src:
                profile = src.profile.copy()
                profile.update(
                    driver="GTiff",
                    tiled=True,
                    blockxsize=512,
                    blockysize=512,
                    bigtiff="IF_SAFER",
                )
                # Strip JP2-specific keys that don't apply
                for key in ("lossless", "quality", "compress", "predictor"):
                    profile.pop(key, None)

                # Ensure float32 dtype for elevation data
                if profile.get("dtype") is None:
                    profile["dtype"] = "float32"

                # Preserve nodata
                src_nodata = src.nodata
                if src_nodata is None:
                    # HiRISE DTMs use IEEE float32 min as nodata
                    src_nodata = _DTM_NODATA
                profile["nodata"] = src_nodata

                logger.debug(
                    "Writing intermediate GeoTIFF for DTM %s "
                    "(dtype=%s, %dx%d, %d band(s)) …",
                    img_path.name,
                    profile.get("dtype"),
                    src.width,
                    src.height,
                    src.count,
                )
                with rasterio.open(tmp, "w", **profile) as dst:
                    for band_idx in src.indexes:
                        band_data = src.read(band_idx)
                        dst.write(band_data, band_idx)
                        del band_data
                    dst.build_overviews(_OVERVIEW_LEVELS, _OVERVIEW_RESAMPLING)
                    dst.update_tags(
                        ns="rio_overview", resampling=_OVERVIEW_RESAMPLING.name
                    )

        gc.collect()

        # Second pass: copy to final COG with float predictor
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*geotransform.*|.*identity matrix.*",
            )
            rasterio.shutil.copy(tmp, cog, **_COG_CREATION_OPTIONS_FLOAT)

        logger.info("DTM COG written: %s", cog.name)
        return cog

    except rasterio.errors.RasterioIOError as exc:
        logger.error("DTM COG conversion failed for %s: %s", img_path.name, exc)
        cog.unlink(missing_ok=True)
        return None
    except Exception as exc:
        logger.error("DTM COG conversion failed for %s: %s", img_path.name, exc)
        cog.unlink(missing_ok=True)
        return None
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------


def _worker_init() -> None:
    """Initialise each spawned worker process.

    Two responsibilities:

    1. **SIGINT**: Python's ``ProcessPoolExecutor`` installs ``SIG_IGN`` for
       SIGINT in spawned workers, which means Ctrl+C reaches the parent (raising
       ``KeyboardInterrupt``) but workers keep running.  Restoring ``SIG_DFL``
       lets the OS kill workers immediately when the terminal sends SIGINT to the
       foreground process group.

    2. **Logging**: ``spawn`` starts a fresh Python interpreter, so the parent's
       ``dictConfig`` (from ``logger_config.json``) is never applied in workers.
       The root logger therefore defaults to ``WARNING``, silently dropping all
       ``logger.info()`` calls — including the "COG written" confirmation.
       ``basicConfig`` at ``INFO`` routes worker log records to stderr (which is
       inherited from the parent at the OS level) so progress is visible on the
       terminal without unsafe concurrent writes to ``app.log``.
    """
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  [worker] %(message)s",
    )


def _iter_jp2_files(root: pathlib.Path) -> Iterator[pathlib.Path]:
    """Yield all JP2 files under *root* (case-insensitive extension match)."""
    for pattern in ("*.JP2", "*.jp2"):
        yield from root.rglob(pattern)


def _iter_img_files(root: pathlib.Path) -> Iterator[pathlib.Path]:
    """Yield all HiRISE DTM .IMG files under *root*.

    Only matches files whose names start with ``DTE`` (the PDS naming
    convention for HiRISE DTMs: ``DTEEC_…``, ``DTEED_…``, etc.) to avoid
    picking up non-DTM .IMG files that may exist in the same tree.
    """
    for pattern in ("*.IMG", "*.img"):
        for p in root.rglob(pattern):
            if p.stem.upper().startswith("DTE"):
                yield p


def convert_all(
        root: pathlib.Path,
        workers: int = 4,
        overwrite: bool = False,
        skip_jp2: bool = False,
        skip_dtm: bool = False,
) -> dict[str, int]:
    """Convert all JP2 and DTM .IMG files under *root* to COG GeoTIFF sidecars.

    Uses a :class:`~concurrent.futures.ProcessPoolExecutor` to parallelise
    the CPU-bound conversion.  Each worker calls :func:`jp2_to_cog` or
    :func:`img_to_cog`.

    .. note::
        Large JP2 files (up to 2.5 GB) are fully decompressed in memory during
        the intermediate write step.  The actual worker count is automatically
        capped by :func:`_safe_worker_count` based on available RAM and the
        estimated decompressed size of the JP2 files found under *root*; the
        ``workers`` argument is therefore treated as an upper bound.

    Args:
        root: Dataset root directory containing JP2 and/or IMG files.
        workers: Number of parallel conversion processes.
        overwrite: Re-convert files that already have a ``.tif`` sidecar.
        skip_jp2: Skip JP2 orthoimage conversion (only process DTM .IMG).
        skip_dtm: Skip DTM .IMG conversion (only process JP2 orthoimages).

    Returns:
        Dict with keys ``"converted"``, ``"skipped"``, and ``"failed"``.
    """
    # Collect files to convert
    tasks: list[tuple[pathlib.Path, Callable]] = []

    if not skip_jp2:
        jp2_files = list(_iter_jp2_files(root))
        logger.info("Found %d JP2 file(s) under %s.", len(jp2_files), root)
        tasks.extend((p, jp2_to_cog) for p in jp2_files)
    else:
        jp2_files = []

    if not skip_dtm:
        img_files = list(_iter_img_files(root))
        logger.info("Found %d DTM .IMG file(s) under %s.", len(img_files), root)
        tasks.extend((p, img_to_cog) for p in img_files)
    else:
        img_files = []

    if not tasks:
        logger.warning("No files found to convert under %s.", root)
        return {"converted": 0, "skipped": 0, "failed": 0}

    counts: dict[str, int] = {"converted": 0, "skipped": 0, "failed": 0}

    # Memory-safe worker count is computed from JP2s (the larger files);
    # IMG files are typically smaller in memory since they're already raw.
    size_reference = jp2_files if jp2_files else img_files
    actual_workers = _safe_worker_count(size_reference, workers)

    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=actual_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init,
        max_tasks_per_child=1,
    )
    future_to_path = {
        pool.submit(convert_fn, p, overwrite): p
        for p, convert_fn in tasks
    }
    try:
        for future in concurrent.futures.as_completed(future_to_path):
            src_path = future_to_path[future]
            try:
                result = future.result()
                if result is None:
                    counts["failed"] += 1
                else:
                    # Check if the COG sidecar was freshly written
                    cog = src_path.with_suffix(".tif")
                    if cog.exists() and cog.stat().st_mtime >= src_path.stat().st_mtime:
                        counts["converted"] += 1
                    else:
                        counts["skipped"] += 1
            except Exception as exc:
                logger.error("Worker error for %s: %s", src_path.name, exc)
                counts["failed"] += 1
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        logger.warning("Interrupted — %d converted so far.", counts["converted"])
        raise SystemExit(130)
    else:
        pool.shutdown(wait=True)

    logger.info(
        "COG conversion complete — converted: %d, skipped: %d, failed: %d.",
        counts["converted"],
        counts["skipped"],
        counts["failed"],
    )
    return counts


# ---------------------------------------------------------------------------
# Geographic train / test split
# ---------------------------------------------------------------------------


def geographic_split(
        index: gpd.GeoDataFrame,
        test_fraction: float = 0.2,
        split_axis: str = "longitude",
        seed: int = 42,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Split a spatial index into geographically separated train and test sets.

    Assigns observations to blocks along the chosen axis, then randomly assigns
    whole blocks to train or test.  This keeps geographically adjacent
    observations on the same side of the split, preventing spatial data leakage
    in crater segmentation models (a model should not see craters immediately
    adjacent to its test craters during training).

    Args:
        index: The :attr:`~temp.MarsHiRISE.index` GeoDataFrame.
        test_fraction: Fraction of observations to place in the test set.
        split_axis: ``"longitude"`` (default) or ``"latitude"``.
        seed: Random seed for reproducible block shuffling.

    Returns:
        ``(train_gdf, test_gdf)`` tuple of GeoDataFrames with the same schema
        as *index*.

    Example::

        from preprocessing import geographic_split

        dataset = MarsHiRISE(bbox=..., ...)
        train_idx, test_idx = geographic_split(dataset.index, test_fraction=0.2)
    """
    # Project to a planar CRS before computing centroids to avoid the
    # "Geometry is in a geographic CRS" UserWarning from geopandas.
    projected = index.to_crs(MARS_PROJECTED_CRS)
    centroids = projected.geometry.centroid.to_crs(index.crs)
    coords: np.ndarray = (
        centroids.x.to_numpy() if split_axis == "longitude"
        else centroids.y.to_numpy()
    )

    # Divide the coordinate range into ~(1/test_fraction) equally-populated
    # blocks, then assign a random subset of blocks to the test set.
    n_blocks = max(5, int(round(1.0 / test_fraction)))
    edges = np.percentile(coords, np.linspace(0.0, 100.0, n_blocks + 1))
    # digitize assigns each point to a block 0 … n_blocks-1
    block_ids = np.digitize(coords, edges[1:-1])

    rng = np.random.default_rng(seed)
    unique_blocks = np.unique(block_ids)
    rng.shuffle(unique_blocks)

    n_test_blocks = max(1, int(round(len(unique_blocks) * test_fraction)))
    test_block_set = set(unique_blocks[:n_test_blocks].tolist())

    test_mask = np.isin(block_ids, list(test_block_set))
    train_gdf = index.iloc[~test_mask]
    test_gdf = index.iloc[test_mask]

    logger.info(
        "Geographic split (%s axis): %d train, %d test observations.",
        split_axis,
        len(train_gdf),
        len(test_gdf),
    )
    return train_gdf, test_gdf


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import logging.config
    import json

    _parser = argparse.ArgumentParser(
        description="Convert HiRISE JP2 and DTM .IMG files to Cloud-Optimized GeoTIFF sidecars."
    )
    _parser.add_argument(
        "--root",
        required=True,
        type=pathlib.Path,
        help="Dataset root directory containing JP2 and/or IMG files.",
    )
    _parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel conversion processes (default: 4). "
             "Use 1 on memory-constrained machines.",
    )
    _parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-convert files that already have a .tif sidecar.",
    )
    _parser.add_argument(
        "--skip-jp2",
        action="store_true",
        help="Skip JP2 orthoimage conversion (only process DTM .IMG files).",
    )
    _parser.add_argument(
        "--skip-dtm",
        action="store_true",
        help="Skip DTM .IMG conversion (only process JP2 orthoimages).",
    )
    _args = _parser.parse_args()

    # Try loading the project logger config; fall back to basicConfig.
    _config_path = pathlib.Path(__file__).parent.parent / "logger_config.json"
    if _config_path.exists():
        with open(_config_path) as _fh:
            logging.config.dictConfig(json.load(_fh))
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(message)s",
        )

    convert_all(
        _args.root,
        workers=_args.workers,
        overwrite=_args.overwrite,
        skip_jp2=_args.skip_jp2,
        skip_dtm=_args.skip_dtm,
    )