# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Abstract base for Mars HiRISE dataset variants (RDR and DTM)."""

from __future__ import annotations

import asyncio
import json
import logging
import logging.config
import math
import multiprocessing
import os
import pathlib
import random
import re
import shutil
import threading
from abc import abstractmethod
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from os import PathLike

import aiohttp
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pdr
import rasterio
import torch
from aiohttp import ClientConnectorError, ClientResponseError
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from shapely import MultiPoint
from shapely.geometry import Polygon, box
from torchgeo.datasets.errors import DatasetNotFoundError
from torchgeo.datasets.geo import GeoDataset
from torchgeo.datasets.utils import GeoSlice, Path, Sample, download_url
from tqdm import tqdm

CONFIG = "logger_config.json"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mars-specific CRS
# ---------------------------------------------------------------------------
# The dataset CRS must be the Mars IAU 2000 *geographic* CRS (lon/lat degrees).
#
# Why geographic, not projected?
#   The cumulative index bounding-box columns are in decimal degrees, so they
#   must be stored with a degree-unit CRS — assigning degree values to a
#   projected (metre-unit) CRS silently mislabels them, causing the
#   reprojection destination window to land ~214 m from the prime meridian
#   instead of at the correct surface location, producing black tiles.
#
#   Each JP2 uses its own per-observation Equirectangular projection
#   (CENTER_LATITUDE differs per image), so there is no single projected CRS
#   that correctly represents all files.  The geographic CRS is the natural
#   common hub: rasterio reads each file's embedded CRS and reprojects into
#   geographic degrees automatically.
#
# Ellipsoid from HIRISE_RDR_SIS.PDF §3.5.1:
#   equatorial radius  a = 3,396,190 m
#   polar radius       b = 3,376,200 m
MARS_GEOGRAPHIC_CRS = CRS.from_proj4(
    "+proj=longlat +a=3396190 +b=3376200 +no_defs"
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_MIN_FREE_BYTES: int = 100 * (1024 ** 3)  # 100 GB

# Tolerance (degrees) added to the JP2 bounds early-exit check in
# _load_from_jp2.  Absorbs floating-point rounding between the index geometry
# (computed at dataset construction time) and the bounds rasterio reports at
# load time.  1e-5° ≈ 0.6 m on Mars — far below HiRISE pixel size (~0.25 m).
_SPATIAL_TOL: float = 1e-5

# ---------------------------------------------------------------------------
# Per-product LBL metadata
# ---------------------------------------------------------------------------
_DEFAULT_SCALING_FACTOR: float = 2.37936949017414e-04
_DEFAULT_OFFSET: float = 0.037954361744101
_DEFAULT_SAMPLE_BITS: int = 16
_EFFECTIVE_BIT_DEPTH: int = 10


@dataclass
class ProductMeta:
    """Radiometric metadata parsed from a per-product PDS3 LBL file.

    Shared across RDR and DTM ortho products.  The defaults correspond to
    representative COLOR RDR label values; DTM ortho images should parse
    their own LBL which typically has 8-bit samples and different offsets.
    """

    scaling_factor: float = _DEFAULT_SCALING_FACTOR
    offset: float = _DEFAULT_OFFSET
    sample_bits: int = _DEFAULT_SAMPLE_BITS
    effective_max_dn: int = (1 << _EFFECTIVE_BIT_DEPTH) - 1
    filter_names: list[str] = field(
        default_factory=lambda: ["NEAR-INFRARED", "RED", "BLUE-GREEN"]
    )
    bands: int = 3

    @classmethod
    def from_lbl(cls, lbl_path: pathlib.Path) -> ProductMeta:
        """Parse a PDS3 LBL and return a populated instance."""
        obj = cls()
        if not lbl_path.exists():
            logger.debug("LBL not found, using defaults: %s", lbl_path)
            return obj
        try:
            text = lbl_path.read_text(errors="replace")
        except OSError as exc:
            logger.warning("Could not read LBL %s: %s", lbl_path, exc)
            return obj

        def _float(pat: str) -> float | None:
            m = re.search(pat, text, re.MULTILINE)
            return float(m.group(1)) if m else None

        def _int(pat: str) -> int | None:
            m = re.search(pat, text, re.MULTILINE)
            return int(m.group(1)) if m else None

        if (v := _float(r"^\s*SCALING_FACTOR\s*=\s*([\d.eE+\-]+)")) is not None:
            obj.scaling_factor = v
        if (v := _float(r"^\s*OFFSET\s*=\s*([\d.eE+\-]+)")) is not None:
            obj.offset = v
        if (v := _int(r"^\s*SAMPLE_BITS\s*=\s*(\d+)")) is not None:
            obj.sample_bits = v
        if (v := _int(r"^\s*BANDS\s*=\s*(\d+)")) is not None:
            obj.bands = v
        if m := re.search(r"SAMPLE_BIT_MASK\s*=\s*2#([01]+)#", text):
            obj.effective_max_dn = (1 << m.group(1).count("1")) - 1
        if m := re.search(r"FILTER_NAME\s*=\s*\(([^)]+)\)", text, re.DOTALL):
            names = [s.strip().strip('"').strip("'") for s in m.group(1).split(",")]
            if names:
                obj.filter_names = names
        elif m := re.search(r'FILTER_NAME\s*=\s*"?(\w[\w-]*)"?', text):
            obj.filter_names = [m.group(1)]

        return obj


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def corners_to_polygon(row: pd.Series) -> Polygon | None:
    """Build a Shapely Polygon from CORNER1–4 lat/lon columns in *row*.

    Longitudes are normalised from PDS [0°, 360°] to [−180°, 180°].
    Returns ``None`` on missing/degenerate data.
    """
    try:
        coords = [
            (
                ((float(row[f"CORNER{i}_LONGITUDE"]) + 180.0) % 360.0) - 180.0,
                float(row[f"CORNER{i}_LATITUDE"]),
            )
            for i in (1, 2, 3, 4)
        ]
    except (KeyError, ValueError, TypeError):
        return None

    if any(math.isnan(lon) or math.isnan(lat) for lon, lat in coords):
        return None

    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if (poly.is_valid and not poly.is_empty) else None


def extract_footprint(
        file_path: str | PathLike | None,
        mars_crs: rasterio.crs.CRS = MARS_GEOGRAPHIC_CRS,
        nodata_test: Callable | None = None,
) -> tuple[list[tuple[float, float]] | None, tuple[float, float, float, float] | None]:
    """Extract the convex hull of valid pixels from a raster.

    Reads band 1 at the coarsest available overview level so even multi-GB
    images resolve to a few hundred pixels.

    Thread-safe: each call opens its own file handle.

    Args:
        file_path: Path to the raster file (JP2, IMG, or COG).
        mars_crs: Target geographic CRS to reproject vertices into.
        nodata_test: Optional callable ``(data_array) -> bool_mask`` that
            returns a boolean mask of *valid* pixels.  Defaults to
            ``data > 0``.

    Returns:
        ``(hull_coords, file_bounds)`` — hull_coords is a list of
        ``(lon, lat)`` vertices, or ``None`` on failure.  file_bounds is a
        ``(west, south, east, north)`` fallback.
    """
    from rasterio.warp import transform as warp_transform

    if file_path is None:
        return None, None

    if isinstance(file_path, str):
        path = pathlib.Path(file_path)
    else:
        path = file_path

    cog = path.with_suffix(".tif")
    actual = cog if cog.exists() else path
    if not actual.exists():
        return None, None

    try:
        with rasterio.open(actual) as src:
            src_crs = src.crs
            if src_crs is None:
                return None, None

            # File bounds (always — cheap fallback)
            try:
                fl, fb, fr, ft = transform_bounds(src_crs, mars_crs, *src.bounds)
                fl = ((fl + 180.0) % 360.0) - 180.0
                fr = ((fr + 180.0) % 360.0) - 180.0
                if not (-180.0 <= fl < fr <= 180.0 and -90.0 <= fb < ft <= 90.0):
                    file_bounds = None
                else:
                    file_bounds = (fl, fb, fr, ft)
            except Exception:
                file_bounds = None

            # Read band 1 at coarsest overview
            ovrs = src.overviews(1)
            factor = max(ovrs) if ovrs else max(1, min(src.height, src.width) // 500)
            oh = max(1, src.height // factor)
            ow = max(1, src.width // factor)
            data = src.read(1, out_shape=(oh, ow))

            if nodata_test is not None:
                valid_mask = nodata_test(data)
            else:
                valid_mask = data > 0

            ys, xs = np.where(valid_mask)
            if len(xs) < 3:
                return None, file_bounds

            step = max(1, len(xs) // 4000)
            hull = MultiPoint(
                list(zip(xs[::step].tolist(), ys[::step].tolist()))
            ).convex_hull
            if hull.is_empty:
                return None, file_bounds

            hull_px = np.array(hull.exterior.coords)
            ovr_tf = rasterio.transform.from_bounds(*src.bounds, ow, oh)
            src_xs, src_ys = rasterio.transform.xy(
                ovr_tf, hull_px[:, 1].tolist(), hull_px[:, 0].tolist()
            )
            geo_xs, geo_ys = warp_transform(
                src_crs, mars_crs, list(src_xs), list(src_ys)
            )
            geo_xs = [((x + 180.0) % 360.0) - 180.0 for x in geo_xs]

            return list(zip(geo_xs, geo_ys)), file_bounds

    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Async download helpers
# ---------------------------------------------------------------------------


def filter_maker(level: str) -> Callable:
    numeric = getattr(logging, level)

    def _filter(record: logging.LogRecord) -> bool:
        return record.levelno <= numeric

    return _filter


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(config_path: str = CONFIG) -> None:
    with open(config_path) as fh:
        logging.config.dictConfig(json.load(fh))


async def _download_file(
        session: aiohttp.ClientSession,
        url: str,
        path: pathlib.Path,
        stop_event: threading.Event,
        max_retries: int = 8,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
) -> None:
    if stop_event.is_set():
        return
    parent = path.parent if path.parent.exists() else pathlib.Path("/")
    if shutil.disk_usage(parent).free < _MIN_FREE_BYTES:
        if not stop_event.is_set():
            logger.critical("Free disk space below threshold. Halting downloads.")
            stop_event.set()
        return
    if path.exists():
        logger.debug("Already downloaded, skipping: %s", path)
        return

    tmp_path = path.with_name(f".tmp.{path.name}")

    for attempt in range(max_retries + 1):
        if stop_event.is_set():
            return
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                expected_bytes: int | None = None
                cl = resp.headers.get("Content-Length")
                if cl is not None:
                    try:
                        expected_bytes = int(cl)
                    except ValueError:
                        pass

                path.parent.mkdir(parents=True, exist_ok=True)
                bytes_written = 0
                with open(tmp_path, "wb") as fh:
                    async for chunk in resp.content.iter_chunked(1 << 20):
                        if stop_event.is_set():
                            logger.error("Halted mid-write: %s", path.name)
                            tmp_path.unlink(missing_ok=True)
                            return
                        await asyncio.to_thread(fh.write, chunk)
                        bytes_written += len(chunk)

                if expected_bytes is not None and bytes_written != expected_bytes:
                    logger.warning(
                        "Truncated response for %s: expected %d bytes, got %d. "
                        "Will retry.",
                        path.name, expected_bytes, bytes_written,
                    )
                    tmp_path.unlink(missing_ok=True)
                else:
                    tmp_path.rename(path)
                    return

        except ClientResponseError as exc:
            tmp_path.unlink(missing_ok=True)
            if exc.status in {429, 503, 504}:
                logger.warning(
                    "Server saturated (%s) for %s. Attempt %d/%d.",
                    exc.status, url, attempt + 1, max_retries,
                )
            else:
                logger.error(
                    "Fatal HTTP %s for %s: %s", exc.status, url, exc.message
                )
                return
        except (ClientConnectorError, asyncio.TimeoutError) as exc:
            tmp_path.unlink(missing_ok=True)
            logger.warning(
                "Connection error for %s. Attempt %d/%d. %s",
                url, attempt + 1, max_retries, exc,
            )
        except Exception as exc:  # noqa: BLE001
            tmp_path.unlink(missing_ok=True)
            logger.error("Unexpected failure downloading %s: %s", url, exc)
            return

        if attempt < max_retries:
            delay = random.uniform(0, min(base_delay * (2 ** attempt), max_delay))
            logger.info("Back-off %.2fs before retrying %s", delay, url)
            await asyncio.sleep(delay)
        else:
            logger.error("Max retries (%d) exhausted for %s.", max_retries, url)


async def _download_many(
        tasks: list[tuple[str, pathlib.Path]],
        concurrency: int,
        stop_event: threading.Event,
) -> None:
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        await asyncio.gather(
            *(_download_file(session, url, path, stop_event) for url, path in tasks)
        )


def _worker_process(
        tasks: list[tuple[str, pathlib.Path]],
        concurrency_per_process: int,
        stop_event: threading.Event,
) -> None:
    asyncio.run(_download_many(tasks, concurrency_per_process, stop_event))


# ---------------------------------------------------------------------------
# Shared reprojection helper
# ---------------------------------------------------------------------------


def reproject_band(
        src_dataset: rasterio.DatasetReader,
        band_idx: int,
        dst_crs: rasterio.crs.CRS,
        dst_transform: rasterio.Affine,
        out_h: int,
        out_w: int,
        *,
        src_nodata: float | None = None,
        dst_nodata: float = 0.0,
        resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Reproject a single band from an open rasterio dataset.

    Returns an ``(out_h, out_w)`` float32 array.
    """
    src_crs = src_dataset.crs
    if src_crs is None:
        src_crs = dst_crs

    dest = np.empty((out_h, out_w), dtype=np.float32)

    kwargs: dict = dict(
        source=rasterio.band(src_dataset, band_idx),
        destination=dest,
        src_transform=src_dataset.transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
        dst_nodata=dst_nodata,
    )
    if src_nodata is not None:
        kwargs["src_nodata"] = src_nodata

    reproject(**kwargs)
    return dest


def check_overlap(
        src_dataset: rasterio.DatasetReader,
        dst_crs: rasterio.crs.CRS,
        x: slice,
        y: slice,
        tol: float = _SPATIAL_TOL,
) -> bool:
    """Return True if the file bounds overlap the query window.

    Absorbs floating-point imprecision and normalises longitudes.
    """
    src_crs = src_dataset.crs
    if src_crs is None:
        return True  # can't check — assume overlap
    try:
        fl, fb, fr, ft = transform_bounds(src_crs, dst_crs, *src_dataset.bounds)
        fl = ((fl + 180.0) % 360.0) - 180.0
        fr = ((fr + 180.0) % 360.0) - 180.0
        if fl <= fr:
            if (
                    fr + tol < x.start
                    or fl - tol > x.stop
                    or ft + tol < y.start
                    or fb - tol > y.stop
            ):
                return False
    except Exception:
        pass
    return True


# ---------------------------------------------------------------------------
# Abstract base dataset
# ---------------------------------------------------------------------------


class MarsHiRISEBase(GeoDataset):
    """Abstract base for Mars HiRISE-derived datasets.

    Provides the common skeleton for index downloading/parsing, spatial
    filtering, parallel footprint extraction, async image downloading,
    coverage visualisation, and the verify/cache lifecycle.

    Subclasses must implement the hooks marked ``@abstractmethod``.
    """

    url: str = "https://hirise-pds.lpl.arizona.edu/PDS"
    mars_crs: CRS = MARS_GEOGRAPHIC_CRS

    # Subclasses override these class attributes.
    _INDEX_STEM: str = ""  # e.g. "RDRCUMINDEX" or "DTMCUMINDEX"
    _INDEX_TABLE_KEY: str = "RDR_INDEX_TABLE"  # key pdr gives the table

    def __init__(
            self,
            root: Path,
            *,
            split: str = "train",
            target: str | None = None,
            transforms: Callable[[Sample], Sample] | None = None,
            download: bool = False,
            bbox: tuple[float, float, float, float] | None = None,
            checksum: bool = False,
            reuse_cache: bool = True,
            res: float = 1.0 / 118_502.26464032,
    ) -> None:
        super(GeoDataset, self).__init__()

        self.root = pathlib.Path(root)
        self.bbox = bbox
        self.split = split
        self.target = target
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.reuse_cache = reuse_cache

        self.res: float = res
        self._crs = MARS_GEOGRAPHIC_CRS

        self.index: gpd.GeoDataFrame | None = None
        self._raw_index: pd.DataFrame | None = None

        self._verify()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index) if self.index else 0

    @abstractmethod
    def __getitem__(self, index: GeoSlice) -> Sample:
        ...

    @abstractmethod
    def plot(self, sample: Sample, **kwargs) -> Figure:
        ...

    # ------------------------------------------------------------------
    # Verify / index loading
    # ------------------------------------------------------------------

    def _verify(self) -> None:
        """Template: ensure dataset is usable, downloading if requested.

        Subclasses should call ``super()._verify()`` or use this directly.
        They may override :meth:`_post_download_verify` for extra checks.
        """
        self.root.mkdir(parents=True, exist_ok=True)

        lbl = self.root / f"{self._INDEX_STEM}.LBL"
        if not lbl.exists():
            if not self.download:
                raise DatasetNotFoundError(self)
            self._download_index()

        self._load_index()

        rebuilt = False
        self._build_spatial_index()

        if self.download:
            changes = self._download_images()
            if changes:
                self._build_spatial_index(force_rebuild=True)
                rebuilt = True

        if rebuilt or not self.reuse_cache or not self.spatial_index_cache.exists():
            self._save_cache()

        self._post_download_verify()

    def _post_download_verify(self) -> None:
        """Hook for subclass-specific post-download sanity checks.

        Default implementation samples a few rows and warns if no data files
        are found.  Subclasses may override to check different column names.
        """
        pass

    def _download_index(self) -> None:
        index_url = f"{self.url}/INDEX"
        for ext in (".LBL", ".TAB"):
            fname = self._INDEX_STEM + ext
            download_url(f"{index_url}/{fname}", str(self.root), fname)

    def _load_index(self) -> None:
        """Parse cumulative index and apply target / bbox filters."""
        lbl_path = self.root / f"{self._INDEX_STEM}.LBL"
        if not lbl_path.exists():
            raise DatasetNotFoundError(self)

        data = pdr.read(str(lbl_path))
        data.load("all")
        df: pd.DataFrame = data[self._INDEX_TABLE_KEY]
        logger.info("Loaded cumulative index (%s): %d rows.", self._INDEX_STEM, len(df))

        # --- text filter ---
        if self.target is not None:
            str_cols = df.select_dtypes(include=["object", "string"])
            mask = str_cols.apply(
                lambda col: col.str.contains(
                    self.target, na=False, regex=False, case=False
                )
            ).any(axis=1)
            df = df[mask].reset_index(drop=True)
            if df.empty:
                logger.warning(
                    "target='%s' matched no rows in %s.",
                    self.target, self._INDEX_STEM,
                )
            else:
                logger.info(
                    "After text filter '%s': %d rows.", self.target, len(df)
                )

        # --- spatial / bbox filter ---
        if self.bbox is not None:
            lon_min, lat_min, lon_max, lat_max = self.bbox

            def _norm(lon: pd.Series) -> pd.Series:
                return ((lon + 180.0) % 360.0) - 180.0

            obs_lon_min = _norm(df["MINIMUM_LONGITUDE"].astype(float))
            obs_lon_max = _norm(df["MAXIMUM_LONGITUDE"].astype(float))
            obs_lat_min = df["MINIMUM_LATITUDE"].astype(float)
            obs_lat_max = df["MAXIMUM_LATITUDE"].astype(float)

            overlap = (
                    (obs_lon_max >= lon_min)
                    & (obs_lon_min <= lon_max)
                    & (obs_lat_max >= lat_min)
                    & (obs_lat_min <= lat_max)
            )
            df = df[overlap].reset_index(drop=True)
            logger.info("After bbox filter %s: %d rows.", self.bbox, len(df))

        self._raw_index = df

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _pds_local_path(self, spec: str) -> pathlib.Path:
        """Return the flat local file path under ``<root>/images/``."""
        rel = pathlib.PurePosixPath(spec.strip())
        return self.root / "images" / pathlib.Path(rel).name

    def _pds_local_stem(self, spec: str) -> pathlib.Path:
        return self._pds_local_path(spec).with_suffix("")

    # ------------------------------------------------------------------
    # Spatial index — cache management
    # ------------------------------------------------------------------

    @property
    def spatial_index_cache(self) -> pathlib.Path:
        """Path to the GeoPackage cache file.

        Incorporates target, bbox, and subclass-specific suffix parts to
        avoid stale-cache collisions when configuration changes.
        """

        def _fmt(v: object) -> str:
            if isinstance(v, float) and v.is_integer():
                return str(int(v))
            return str(v)

        parts: list[str] = []
        if self.target:
            parts.append(self.target)
        if self.bbox:
            parts.append("_".join(_fmt(v) for v in self.bbox))
        parts.extend(self._cache_suffix_parts())
        suffix = f"_{'_'.join(parts)}" if parts else ""
        return self.root / f"spatial_cache{suffix}_{self._cache_version()}.gpkg"

    def _cache_suffix_parts(self) -> list[str]:
        """Extra tokens appended to the cache filename.

        Override in subclasses to bust cache on config change (e.g. ortho
        type selection).
        """
        return []

    def _cache_version(self) -> str:
        """Cache version tag.  Bump when the schema changes."""
        return "v3"

    def _save_cache(self) -> None:
        self.spatial_index_cache.parent.mkdir(parents=True, exist_ok=True)
        cache_df = self.index.copy()
        cache_df["t_start"] = self.index.index.left.astype(str)
        cache_df["t_stop"] = self.index.index.right.astype(str)
        cache_df.reset_index(drop=True).to_file(self.spatial_index_cache)

    def _try_load_cache(self) -> bool:
        """Try to restore :attr:`index` from cache.  Returns success flag."""
        if not (
                self.reuse_cache
                and self.spatial_index_cache.exists()
                and self.index is None
        ):
            return False

        logger.info(
            "Loading spatial index from cache (%s).", self.spatial_index_cache
        )
        gdf = gpd.read_file(self.spatial_index_cache)

        # Detect legacy bbox-only cache.
        sample_geoms = gdf.geometry.iloc[: min(5, len(gdf))]
        is_legacy = all(g.equals(box(*g.bounds)) for g in sample_geoms)
        if is_legacy:
            logger.info("Legacy bbox cache detected at %s; rebuilding.", self.spatial_index_cache)
            return False

        t_start = pd.to_datetime(gdf.pop("t_start"), utc=True)
        t_stop = pd.to_datetime(gdf.pop("t_stop"), utc=True)
        gdf.index = pd.IntervalIndex.from_arrays(
            t_start, t_stop, closed="both", name="datetime"
        )
        self.index = gdf
        self._log_index_extent()
        return True

    def _log_index_extent(self) -> None:
        total_bounds = self.index["geometry"].bounds
        minx, miny = total_bounds[["minx", "miny"]].min()
        maxx, maxy = total_bounds[["maxx", "maxy"]].max()
        logger.info(
            "Spatial index: %d entries, lon [%.2f, %.2f] lat [%.2f, %.2f].",
            len(self.index), minx, maxx, miny, maxy,
        )

    # ------------------------------------------------------------------
    # Spatial index — parallel footprint extraction
    # ------------------------------------------------------------------

    def _run_footprint_extraction(
            self,
            file_paths: list[str | None],
            *,
            nodata_test: Callable | None = None,
    ) -> list[tuple[list[tuple[float, float]] | None, tuple | None]]:
        """Run :func:`extract_footprint` in parallel for a list of paths.

        Returns a list aligned with *file_paths*.
        """
        mars_crs_rio = rasterio.crs.CRS.from_user_input(self.mars_crs)
        n_total = len(file_paths)
        n_with_files = sum(1 for p in file_paths if p is not None)
        n_workers = min(os.cpu_count() or 4, 32)

        logger.info(
            "Extracting footprints: %d entries (%d with files), %d threads …",
            n_total, n_with_files, n_workers,
        )

        results: list[
            tuple[list[tuple[float, float]] | None, tuple | None]
        ] = [(None, None)] * n_total

        done_count = 0
        lock = threading.Lock()

        def _do(idx: int) -> None:
            nonlocal done_count
            results[idx] = extract_footprint(
                file_paths[idx], mars_crs_rio, nodata_test
            )
            with lock:
                done_count += 1
                if done_count % 50 == 0 or done_count == n_with_files:
                    logger.info(
                        "  footprint extraction: %d / %d", done_count, n_with_files
                    )

        indices = [i for i, p in enumerate(file_paths) if p is not None]

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_do, i) for i in indices]
            for f in tqdm(
                    as_completed(futs),
                    total=len(futs),
                    desc="Extracting footprints",
                    unit="file",
            ):
                f.result()

        return results

    def _geometry_from_footprint_result(
            self,
            hull_coords: list[tuple[float, float]] | None,
            file_bounds: tuple | None,
            ref_row: pd.Series,
    ) -> Polygon | None:
        """Build the best available geometry for one index row.

        Tries, in order: convex hull → file bbox → corner polygon → index
        min/max bbox.  Returns ``None`` only for antimeridian-crossing tiles.
        """
        geom = None

        # A: convex hull of non-nodata pixels
        if hull_coords is not None:
            candidate = Polygon(hull_coords)
            if not candidate.is_valid:
                candidate = candidate.buffer(0)
            if candidate.is_valid and not candidate.is_empty and candidate.area > 1e-12:
                geom = candidate

        # B: file bounding box
        if geom is None and file_bounds is not None:
            geom = box(*file_bounds)

        # C: cumulative-index corners
        if geom is None:
            geom = corners_to_polygon(ref_row)

        # D: cumulative-index min/max bbox
        if geom is None:
            lon_min = ((float(ref_row["MINIMUM_LONGITUDE"]) + 180.0) % 360.0) - 180.0
            lon_max = ((float(ref_row["MAXIMUM_LONGITUDE"]) + 180.0) % 360.0) - 180.0
            if lon_min > lon_max:
                return None  # antimeridian
            lat_min = float(ref_row["MINIMUM_LATITUDE"])
            lat_max = float(ref_row["MAXIMUM_LATITUDE"])
            geom = box(lon_min, lat_min, lon_max, lat_max)

        return geom

    # ------------------------------------------------------------------
    # Abstract spatial index hook
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_spatial_index(self, force_rebuild: bool = False) -> None:
        ...

    # ------------------------------------------------------------------
    # Download orchestration
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_download_tasks(self) -> list[tuple[str, pathlib.Path]]:
        ...

    def _download_images(self) -> bool:
        tasks = self._build_download_tasks()
        if not tasks:
            logger.info("Everything is already downloaded.")
            return False
        logger.info("Scheduling %d file downloads.", len(tasks))
        active_processes = 8
        concurrency_per_process = 2
        chunk_size = max(1, (len(tasks) + active_processes - 1) // active_processes)
        chunks = [
            tasks[i * chunk_size: (i + 1) * chunk_size]
            for i in range(active_processes)
        ]
        manager = multiprocessing.Manager()
        stop_event = manager.Event()
        with ProcessPoolExecutor(max_workers=active_processes) as pool:
            futures = [
                pool.submit(
                    _worker_process, chunk, concurrency_per_process, stop_event
                )
                for chunk in chunks
                if chunk
            ]
            for fut in futures:
                fut.result()
        if stop_event.is_set():
            logger.warning("Download halted early to preserve disk space.")
        return True

    # ------------------------------------------------------------------
    # Tile helpers
    # ------------------------------------------------------------------

    @staticmethod
    def prefer_cog(path: pathlib.Path | None) -> pathlib.Path | None:
        """Return COG sidecar (.tif) for *path* if it exists, else *path*."""
        if path is None:
            return None
        cog = path.with_suffix(".tif")
        return cog if cog.exists() else path

    @staticmethod
    def merge_tiles(tiles: list[torch.Tensor]) -> torch.Tensor:
        """Mosaic co-registered tiles with a first-non-zero-wins strategy."""
        if len(tiles) == 1:
            return tiles[0]

        max_h = max(t.shape[1] for t in tiles)
        max_w = max(t.shape[2] for t in tiles)
        n_ch = max(t.shape[0] for t in tiles)

        if not all(t.shape[0] == n_ch for t in tiles):
            logger.warning(
                "merge_tiles: channel mismatch (%d tiles, max %d ch) — padding.",
                len(tiles), n_ch,
            )
            padded = []
            for t in tiles:
                if t.shape[0] < n_ch:
                    pad = torch.zeros(
                        n_ch - t.shape[0], t.shape[1], t.shape[2], dtype=t.dtype
                    )
                    t = torch.cat([t, pad])
                padded.append(t)
            tiles = padded

        merged = torch.zeros((n_ch, max_h, max_w), dtype=torch.float32)
        for tile in tiles:
            h, w = tile.shape[1], tile.shape[2]
            empty = merged[:, :h, :w] == 0.0
            merged[:, :h, :w][empty] = tile[empty]
        return merged

    # ------------------------------------------------------------------
    # Coverage visualisation
    # ------------------------------------------------------------------

    def plot_coverage(
            self,
            resolution: float | None = None,
            show_count: bool = True,
            suptitle: str | None = None,
    ) -> Figure:
        """Visualise the spatial coverage of all entries in the index.

        The plot is cropped to the actual extent of the entries, with
        a count heatmap plus per-entry bounding boxes.
        """
        if self.index is None or len(self.index) == 0:
            raise RuntimeError("Spatial index is empty — nothing to plot.")

        all_bounds = self.index.geometry.bounds
        lon_min = float(all_bounds["minx"].min())
        lon_max = float(all_bounds["maxx"].max())
        lat_min = float(all_bounds["miny"].min())
        lat_max = float(all_bounds["maxy"].max())

        lon_span = lon_max - lon_min or 1.0
        lat_span = lat_max - lat_min or 1.0
        margin_lon = lon_span * 0.05
        margin_lat = lat_span * 0.05
        lon_min -= margin_lon
        lon_max += margin_lon
        lat_min -= margin_lat
        lat_max += margin_lat

        if resolution is None:
            resolution = min(lon_span, lat_span) / 20.0

        coverage, lon_edges, lat_edges = self._coverage_grid(
            resolution=resolution,
            lon_min=lon_min, lon_max=lon_max,
            lat_min=lat_min, lat_max=lat_max,
        )
        lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
        lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2

        n_obs = len(self.index)
        covered_cells = int(np.count_nonzero(coverage))
        total_cells = coverage.size
        pct_covered = 100.0 * covered_cells / total_cells if total_cells else 0.0
        median_overlap = (
            float(np.median(coverage[coverage > 0])) if covered_cells else 0.0
        )
        max_overlap = int(coverage.max())

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(
            2, 2,
            width_ratios=[4, 1], height_ratios=[1, 4],
            hspace=0.05, wspace=0.05,
        )
        ax_main = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

        display = np.ma.masked_equal(coverage, 0)
        norm = LogNorm(vmin=1, vmax=max(max_overlap, 1)) if show_count else None
        im = ax_main.pcolormesh(
            lon_edges, lat_edges, display,
            cmap="YlOrRd", norm=norm, shading="flat",
            rasterized=True, zorder=1,
        )
        ax_main.pcolormesh(
            lon_edges, lat_edges,
            np.ma.masked_not_equal(coverage, 0),
            cmap="Greys", vmin=0, vmax=1,
            shading="flat", rasterized=True, zorder=0,
        )

        cmap_boxes = matplotlib.colormaps.get_cmap("tab20")
        rects = []
        for geom in self.index.geometry:
            b = geom.bounds
            rects.append(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1]))

        box_colors = [cmap_boxes(i % 20) for i in range(n_obs)]
        pc = PatchCollection(
            rects,
            facecolors=[(r, g, b, 0.08) for r, g, b, _ in box_colors],
            edgecolors=[(r, g, b, 0.85) for r, g, b, _ in box_colors],
            linewidths=0.6, zorder=2,
        )
        ax_main.add_collection(pc)

        ax_main.set_xlim(lon_min, lon_max)
        ax_main.set_ylim(lat_min, lat_max)
        ax_main.set_xlabel("Longitude (°E, normalised to [−180, 180])")
        ax_main.set_ylabel("Latitude (°)")

        def _nice_ticks(lo: float, hi: float, n: int = 6) -> np.ndarray:
            span = hi - lo
            raw_step = span / n
            magnitude = 10 ** np.floor(np.log10(raw_step))
            for candidate in [1, 2, 2.5, 5, 10]:
                step = candidate * magnitude
                if span / step <= n + 1:
                    break
            start = np.ceil(lo / step) * step
            return np.arange(start, hi + step * 0.5, step)

        ax_main.set_xticks(_nice_ticks(lon_min, lon_max))
        ax_main.set_yticks(_nice_ticks(lat_min, lat_max))
        ax_main.grid(color="white", linewidth=0.3, alpha=0.5, zorder=3)

        if show_count:
            cbar = fig.colorbar(im, ax=ax_main, pad=0.01, fraction=0.025)
            cbar.set_label("Entries per cell (log scale)", fontsize=8)

        legend_handles = [
            Patch(facecolor="#bbbbbb", label="No coverage"),
            Patch(facecolor=plt.cm.YlOrRd(0.15), label="Low overlap"),
            Patch(facecolor=plt.cm.YlOrRd(0.85), label="High overlap"),
        ]
        ax_main.legend(
            handles=legend_handles, loc="lower left",
            fontsize=7, framealpha=0.75,
        )

        ax_top.bar(
            lon_centers, coverage.sum(axis=0),
            width=resolution, color="steelblue", alpha=0.8, linewidth=0,
        )
        ax_top.set_ylabel("Sum", fontsize=8)
        ax_top.tick_params(labelbottom=False, labelsize=7)
        ax_top.grid(axis="y", linewidth=0.4, alpha=0.5)

        ax_right.barh(
            lat_centers, coverage.sum(axis=1),
            height=resolution, color="steelblue", alpha=0.8, linewidth=0,
        )
        ax_right.set_xlabel("Sum", fontsize=8)
        ax_right.tick_params(labelleft=False, labelsize=7)
        ax_right.grid(axis="x", linewidth=0.4, alpha=0.5)

        summary = (
            f"Entries : {n_obs:,}\n"
            f"Cells covered : {pct_covered:.1f}%  "
            f"({covered_cells:,} / {total_cells:,}  @  {resolution:.4f}°/cell)\n"
            f"Overlap — median : {median_overlap:.1f}×   max : {max_overlap}×"
        )
        ax_main.text(
            0.01, 0.99, summary,
            transform=ax_main.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.80),
            zorder=4,
        )

        title = suptitle or (
            f"{type(self).__name__} coverage — {n_obs:,} entries"
        )
        if self.target:
            title += f"  (filter: '{self.target}')"
        fig.suptitle(title, fontsize=12, y=1.005)

        return fig

    def _coverage_grid(
            self,
            resolution: float = 1.0,
            lon_min: float = -180.0,
            lon_max: float = 180.0,
            lat_min: float = -90.0,
            lat_max: float = 90.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lon_edges = np.arange(lon_min, lon_max + resolution, resolution)
        lat_edges = np.arange(lat_min, lat_max + resolution, resolution)
        n_lon = len(lon_edges) - 1
        n_lat = len(lat_edges) - 1
        coverage = np.zeros((n_lat, n_lon), dtype=np.int32)

        for geom in self.index.geometry:
            b = geom.bounds
            i_lon_lo = max(0, int(np.floor((b[0] - lon_min) / resolution)))
            i_lon_hi = min(n_lon, int(np.ceil((b[2] - lon_min) / resolution)))
            i_lat_lo = max(0, int(np.floor((b[1] - lat_min) / resolution)))
            i_lat_hi = min(n_lat, int(np.ceil((b[3] - lat_min) / resolution)))
            if i_lon_lo < i_lon_hi and i_lat_lo < i_lat_hi:
                coverage[i_lat_lo:i_lat_hi, i_lon_lo:i_lon_hi] += 1

        return coverage, lon_edges, lat_edges
