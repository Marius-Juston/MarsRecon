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
import logging
import pathlib
from collections.abc import Iterator
from typing import Callable

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.enums
import rasterio.shutil

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COG creation settings
# ---------------------------------------------------------------------------

#: Overview levels built into each COG for multi-scale access.
_OVERVIEW_LEVELS: list[int] = [2, 4, 8, 16]

#: Resampling method used when building overviews.
_OVERVIEW_RESAMPLING = rasterio.enums.Resampling.average

#: Base rasterio profile applied to every COG output.
_COG_CREATION_OPTIONS: dict = {
    "driver": "GTiff",
    "compress": "deflate",
    "predictor": 2,       # horizontal differencing — good for imagery
    "tiled": True,
    "blockxsize": 512,
    "blockysize": 512,
    "copy_src_overviews": True,
}


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
    ``.tif`` extension.  :meth:`~temp.MarsHiRISE._prefer_cog` will
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
        with rasterio.open(jp2_path) as src:
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                compress="deflate",
                predictor=2,
                tiled=True,
                blockxsize=512,
                blockysize=512,
            )
            # Remove JP2-specific keys that GTiff doesn't understand.
            for key in ("lossless", "quality"):
                profile.pop(key, None)

            logger.debug("Writing intermediate GeoTIFF for %s …", jp2_path.name)
            with rasterio.open(tmp, "w", **profile) as dst:
                for band_idx in src.indexes:
                    dst.write(src.read(band_idx), band_idx)
                dst.build_overviews(_OVERVIEW_LEVELS, _OVERVIEW_RESAMPLING)
                dst.update_tags(
                    ns="rio_overview", resampling=_OVERVIEW_RESAMPLING.name
                )

        # Second pass: copy to final COG with overviews embedded.
        # rasterio.shutil.copy takes GDAL creation options only — strip
        # dataset-metadata keys (dtype, width, height, …) that are valid in a
        # rasterio profile dict but are not GTiff creation options.
        _PROFILE_META_KEYS = frozenset(
            ("dtype", "nodata", "width", "height", "count", "crs", "transform", "driver")
        )
        cog_creation_opts = {
            k: v for k, v in profile.items() if k not in _PROFILE_META_KEYS
        }
        cog_creation_opts["copy_src_overviews"] = True
        rasterio.shutil.copy(tmp, cog, driver="GTiff", **cog_creation_opts)

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


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------


def _iter_jp2_files(root: pathlib.Path) -> Iterator[pathlib.Path]:
    """Yield all JP2 files under *root* (case-insensitive extension match)."""
    for pattern in ("*.JP2", "*.jp2"):
        yield from root.rglob(pattern)


def convert_all(
    root: pathlib.Path,
    workers: int = 4,
    overwrite: bool = False,
) -> dict[str, int]:
    """Convert all JP2 files under *root* to COG GeoTIFF sidecars.

    Uses a :class:`~concurrent.futures.ProcessPoolExecutor` to parallelise
    the CPU-bound conversion.  Each worker calls :func:`jp2_to_cog`.

    .. note::
        Large JP2 files (up to 2.5 GB) are fully decompressed in memory during
        the intermediate write step.  With ``workers=4`` this may require
        ~10 GB of RAM.  Use ``workers=1`` on memory-constrained machines.

    Args:
        root: Dataset root directory containing JP2 files.
        workers: Number of parallel conversion processes.
        overwrite: Re-convert files that already have a ``.tif`` sidecar.

    Returns:
        Dict with keys ``"converted"``, ``"skipped"``, and ``"failed"``.
    """
    jp2_files = list(_iter_jp2_files(root))
    logger.info("Found %d JP2 file(s) under %s.", len(jp2_files), root)

    counts: dict[str, int] = {"converted": 0, "skipped": 0, "failed": 0}

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_path = {
            pool.submit(jp2_to_cog, p, overwrite): p for p in jp2_files
        }
        for future in concurrent.futures.as_completed(future_to_path):
            jp2_path = future_to_path[future]
            try:
                result = future.result()
                if result is None:
                    counts["failed"] += 1
                else:
                    cog = _cog_path(jp2_path)
                    # A freshly converted COG is newer than its source JP2.
                    if cog.exists() and cog.stat().st_mtime >= jp2_path.stat().st_mtime:
                        counts["converted"] += 1
                    else:
                        counts["skipped"] += 1
            except Exception as exc:
                logger.error("Worker error for %s: %s", jp2_path.name, exc)
                counts["failed"] += 1

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
    centroids = index.geometry.centroid
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
        description="Convert HiRISE JP2 files to Cloud-Optimized GeoTIFF sidecars."
    )
    _parser.add_argument(
        "--root",
        required=True,
        type=pathlib.Path,
        help="Dataset root directory containing JP2 files.",
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

    convert_all(_args.root, workers=_args.workers, overwrite=_args.overwrite)
