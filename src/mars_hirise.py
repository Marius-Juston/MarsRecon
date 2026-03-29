# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""MarsHiRISE dataset."""

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
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Literal

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

logger = logging.getLogger(__name__)

CONFIG = "logger_config.json"

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
# Channel / band constants
# ---------------------------------------------------------------------------
# HiRISE RDR products per observation:
#
#   _COLOR.JP2  Three-band mosaic, in-file band order:
#       band 1 -> NEAR-INFRARED  (~900 nm)
#       band 2 -> RED            (~700 nm)
#       band 3 -> BLUE-GREEN     (~500 nm)
#
#   _RED.JP2    Single-band full-strip:
#       band 1 -> RED            (~700 nm)  [more TDI lines = higher quality]

ALL_CHANNELS: tuple[str, ...] = ("NEAR-INFRARED", "RED", "BLUE-GREEN")
_COLOR_CHANNELS: tuple[str, ...] = ("NEAR-INFRARED", "RED", "BLUE-GREEN")
_RED_CHANNELS: tuple[str, ...] = ("RED",)

_COLOR_BAND: dict[str, int] = {"NEAR-INFRARED": 1, "RED": 2, "BLUE-GREEN": 3}
_RED_BAND: dict[str, int] = {"RED": 1}

_MIN_FREE_BYTES: int = 100 * (1024 ** 3)  # 100 GB

# Fallback DN -> I/F coefficients (representative COLOR LBL values).
_DEFAULT_SCALING_FACTOR: float = 2.37936949017414e-04
_DEFAULT_OFFSET: float = 0.037954361744101
_DEFAULT_SAMPLE_BITS: int = 16
_EFFECTIVE_BIT_DEPTH: int = 10  # from SAMPLE_BIT_MASK = 2#0000001111111111#


# ---------------------------------------------------------------------------
# Per-product LBL metadata
# ---------------------------------------------------------------------------


@dataclass
class _ProductMeta:
    """Radiometric metadata parsed from a per-product PDS3 LBL file."""

    scaling_factor: float = _DEFAULT_SCALING_FACTOR
    offset: float = _DEFAULT_OFFSET
    sample_bits: int = _DEFAULT_SAMPLE_BITS
    effective_max_dn: int = (1 << _EFFECTIVE_BIT_DEPTH) - 1
    filter_names: list[str] = field(default_factory=lambda: list(_COLOR_CHANNELS))
    bands: int = 3

    @classmethod
    def from_lbl(cls, lbl_path: pathlib.Path) -> "_ProductMeta":
        """Parse a PDS3 LBL and return a populated instance, defaulting on failure."""
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

        return obj


# ---------------------------------------------------------------------------
# Async download helpers
# ---------------------------------------------------------------------------


def filter_maker(level: str) -> Callable:
    numeric = getattr(logging, level)

    def _filter(record: logging.LogRecord) -> bool:
        return record.levelno <= numeric

    return _filter


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

    # Write to a hidden temp file in the same directory; rename to the final
    # path only on success.  This makes the download atomic: a partial file
    # from a previous interrupted run is invisible to the "already exists"
    # guard above and is simply overwritten on the next attempt.
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

                # Validate Content-Length if the server provided one.
                if expected_bytes is not None and bytes_written != expected_bytes:
                    logger.warning(
                        "Truncated response for %s: expected %d bytes, got %d. "
                        "Will retry.",
                        path.name, expected_bytes, bytes_written,
                    )
                    tmp_path.unlink(missing_ok=True)
                    # Treat as a retriable error — fall through to back-off.
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
                logger.error("Fatal HTTP %s for %s: %s", exc.status, url, exc.message)
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
# Constants
# ---------------------------------------------------------------------------

# Tolerance (degrees) added to the JP2 bounds early-exit check in
# _load_from_jp2.  Absorbs floating-point rounding between the index geometry
# (computed at dataset construction time) and the bounds rasterio reports at
# load time.  1e-5° ≈ 0.6 m on Mars — far below HiRISE pixel size (~0.25 m).
_SPATIAL_TOL: float = 1e-5


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _corners_to_polygon(row: "pd.Series") -> Polygon | None:
    """Build a Shapely Polygon from CORNER1-4 lat/lon columns in *row*.

    Longitudes are normalised from the PDS [0°, 360°] convention to
    [−180°, 180°].  Returns ``None`` if any coordinate is NaN, a column is
    missing, or the resulting polygon is degenerate.
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
        poly = poly.buffer(0)  # standard fix for self-intersecting rings
    return poly if (poly.is_valid and not poly.is_empty) else None


def _extract_footprint(
        file_path: str | None,
        mars_crs: "rasterio.crs.CRS",
) -> tuple[list[tuple[float, float]] | None, tuple[float, float, float, float] | None]:
    """Extract the convex hull of non-zero pixels from a JP2/COG.

    Reads band 1 at the coarsest available overview level so even
    multi-GB images resolve to a few hundred pixels.

    Thread-safe: each call opens its own file handle.

    Returns:
        ``(hull_coords, file_bounds)`` — *hull_coords* is a list of
        ``(lon, lat)`` vertices for the convex hull, or ``None`` on
        failure.  *file_bounds* is a ``(west, south, east, north)``
        fallback when the hull cannot be computed but the file is
        readable.
    """
    from rasterio.warp import transform as warp_transform, transform_bounds
    from shapely.geometry import MultiPoint

    if file_path is None:
        return None, None

    path = pathlib.Path(file_path)
    cog = path.with_suffix(".tif")
    actual = cog if cog.exists() else path
    if not actual.exists():
        return None, None

    try:
        with rasterio.open(actual) as src:
            src_crs = src.crs
            if src_crs is None:
                return None, None

            # ── File bounds (always computed — cheap fallback) ────
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

            # ── Read band 1 at coarsest overview ─────────────────
            ovrs = src.overviews(1)
            factor = max(ovrs) if ovrs else max(1, min(src.height, src.width) // 500)
            oh = max(1, src.height // factor)
            ow = max(1, src.width // factor)

            data = src.read(1, out_shape=(oh, ow))

            ys, xs = np.where(data > 0)
            if len(xs) < 3:
                return None, file_bounds

            # ── Convex hull in pixel space ───────────────────────
            step = max(1, len(xs) // 4000)
            hull = MultiPoint(
                list(zip(xs[::step].tolist(), ys[::step].tolist()))
            ).convex_hull
            if hull.is_empty:
                return None, file_bounds

            hull_px = np.array(hull.exterior.coords)

            # ── Pixel → source CRS → geographic CRS ─────────────
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
# Dataset
# ---------------------------------------------------------------------------

Channel = Literal["NEAR-INFRARED", "RED", "BLUE-GREEN"]


class MarsHiRISE(GeoDataset):
    """Mars HiRISE Reduced Data Records (RDR) dataset.

    `HiRISE <https://pds-imaging.jpl.nasa.gov/volumes/mro.html>`__ is the
    High Resolution Imaging Science Experiment aboard the Mars Reconnaissance
    Orbiter (MRO).  This dataset wraps the RDR products — radiometrically-
    corrected images resampled to a standard map projection — hosted on the
    NASA PDS Imaging Node.

    Directory layout
    ~~~~~~~~~~~~~~~~
    Files are stored preserving the PDS directory hierarchy under *root*::

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

    This mirrors the structure of the PDS server and is compatible with
    ``rsync`` / ``wget -r`` mirrors.  Existing mirrors require no
    reorganisation.

    Each HiRISE observation produces two JP2 product files:

    ``_COLOR.JP2``
        Three-band mosaic — in-file band order:
        ``NEAR-INFRARED`` (band 1), ``RED`` (band 2), ``BLUE-GREEN`` (band 3).

    ``_RED.JP2``
        Single-band full-strip ``RED`` image.  Uses more TDI lines; highest-
        quality RED source when colour context is not needed.

    Channel selection
    ~~~~~~~~~~~~~~~~~
    * **Only** ``"RED"`` requested → ``_RED.JP2`` (higher fidelity).
    * Any ``"NEAR-INFRARED"`` or ``"BLUE-GREEN"`` → ``_COLOR.JP2``.

    Radiometric calibration
    ~~~~~~~~~~~~~~~~~~~~~~~
    ``I/F = DN * SCALING_FACTOR + OFFSET``, clipped to ``[0, 1]``.

    Sampler units
    ~~~~~~~~~~~~~
    ``self.crs`` is a geographic CRS; ``self.res`` is in **degrees/pixel**.
    Pass ``size`` to :class:`~torchgeo.samplers.RandomGeoSampler` in degrees
    (e.g. ``size=0.01`` ≈ 1 185 px ≈ 593 m at the equator).

    Dataset homepage:
        https://hirise-pds.lpl.arizona.edu/PDS/AAREADME.TXT

    .. versionadded:: 0.7
    """

    url: str = "https://hirise-pds.lpl.arizona.edu/PDS"
    _INDEX_STEM: str = "RDRCUMINDEX"
    INDEX_CACHE = 'spatial_cache.gpkg'

    all_channels: tuple[str, ...] = ALL_CHANNELS
    mars_crs: CRS = MARS_GEOGRAPHIC_CRS

    def __init__(
            self,
            root: Path = "/scratch/mars_hirise",
            split: str = "train",
            target: str | None = None,
            channels: list[Channel] | None = None,
            transforms: Callable[[Sample], Sample] | None = None,
            download: bool = False,
            bbox: tuple[float, float, float, float] | None = None,
            checksum: bool = False,
            reuse_cache: bool = True
    ) -> None:
        """Initialise the dataset.

        Args:
            root: Root directory.  Must contain the PDS index files and the
                observation JP2/LBL files in the PDS directory structure, or
                ``download=True`` must be set to fetch them.
            split: Dataset split — informational until train/val/test manifests
                are added.
            target: Optional case-insensitive substring filter on all character
                columns of the cumulative index (e.g. ``"Olympus"``).
            channels: Which channels to include.  Valid values:
                ``"NEAR-INFRARED"``, ``"RED"``, ``"BLUE-GREEN"``.  Output
                tensor band order always follows :attr:`all_channels`.
                Defaults to all three.
            transforms: Optional callable applied to each :class:`Sample`.
            download: Fetch index and images from the PDS server if absent.
            checksum: Verify checksums after download (not yet implemented).

        Raises:
            ValueError: If *channels* contains an unrecognised name.
            DatasetNotFoundError: If index files are absent and
                ``download=False``, or if no JP2 files exist under *root*
                after downloading/filtering.
        """
        super(GeoDataset, self).__init__()

        self.bbox = bbox
        if channels is None:
            self.channels: list[str] = list(ALL_CHANNELS)
        else:
            invalid = set(channels) - set(ALL_CHANNELS)
            if invalid:
                raise ValueError(
                    f"Invalid channel(s): {invalid}. "
                    f"Valid choices are: {ALL_CHANNELS}"
                )
            self.channels = [c for c in ALL_CHANNELS if c in set(channels)]

        self.root = pathlib.Path(root)
        self.split = split
        self.target = target
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self.reuse_cache = reuse_cache

        # Geographic Mars CRS — units are decimal degrees.
        # Native HiRISE resolution: 1 / 118 502.26 pix/deg ≈ 8.44e-6 deg/pix
        self.res: float = 1.0 / 118_502.26464032
        self._crs = MARS_GEOGRAPHIC_CRS

        self.index: gpd.GeoDataFrame | None = None
        self._raw_index: pd.DataFrame | None = None

        self._verify()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: GeoSlice) -> Sample:
        """Return an image patch for the given spatiotemporal slice.

        Args:
            index: ``[xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres]``
                in Mars geographic degrees / UTC datetimes.

        Returns:
            Sample dict with ``"image"`` ``(C, H, W)`` float32 in ``[0,1]``,
            ``"bounds"`` tensor, and ``"crs"`` WKT string.

        Raises:
            IndexError: No observations overlap *index*, or all matching JP2s
                are absent.  The latter triggers a more informative message
                that points to ``download=True`` or a manual mirror.
        """
        x, y, t = self._disambiguate_slice(index)

        query_geom = box(x.start, y.start, x.stop, y.stop)
        # interval = pd.Interval(t.start, t.stop)
        #
        # time_mask = self.index.index.overlaps(interval)
        # candidates: gpd.GeoDataFrame = self.index.iloc[time_mask]
        # candidates = candidates[candidates.geometry.intersects(query_geom)]

        candidates = self.index[self.index.geometry.intersects(query_geom)]

        if candidates.empty:
            raise IndexError(
                f"No MarsHiRISE observations found for slice {index}."
            )

        tiles: list[torch.Tensor] = []
        for _, row in candidates.iterrows():
            # GeoPackage stores None as NaN on round-trip; guard against both.
            cp = row["color_path"]
            rp = row["red_path"]
            tile = self._load_tile(
                color_path=pathlib.Path(cp) if isinstance(cp, str) else None,
                red_path=pathlib.Path(rp) if isinstance(rp, str) else None,
                x=x,
                y=y,
            )
            if tile is not None:
                tiles.append(tile)

        if not tiles:
            n = len(candidates)
            raise IndexError(
                f"{n} observation(s) matched slice {index} spatially/temporally, "
                f"but no image data could be loaded. Possible causes:\n"
                f"  1. JP2 files are absent under '{self.root}' — run with download=True.\n"
                f"  2. The JP2 files exist but their reprojected bounds don't overlap the "
                f"query (check DEBUG logs for 'doesn't overlap' messages).\n"
                f"  3. RasterioIOError on open — check WARNING logs."
            )

        image = self._merge_tiles(tiles)
        sample: Sample = {
            "image": image,
            "bounds": self._slice_to_tensor(index),
            "crs": self.crs.to_wkt(),
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def plot(self, sample, show_titles=True, suptitle=None) -> Figure:
        image: torch.Tensor = sample["image"]
        if image.ndim == 4:
            image = image[0]

        ch = self.channels
        if set(ch) >= {"NEAR-INFRARED", "RED", "BLUE-GREEN"}:
            idx = [ch.index("NEAR-INFRARED"), ch.index("RED"), ch.index("BLUE-GREEN")]
            rgb = image[idx]
            # If only one channel has non-zero data (e.g. COLOR file absent),
            # fall back to grayscale rather than a misleadingly coloured render.
            nonzero = [(rgb[i] > 0).any().item() for i in range(3)]
            if sum(nonzero) == 1:
                active_i = nonzero.index(True)
                active_name = ["NEAR-INFRARED", "RED", "BLUE-GREEN"][active_i]
                img_np = rgb[active_i].numpy()
                cmap = "grey"
                title = f"MarsHiRISE — {active_name} (COLOR file unavailable)"
            else:
                img_np = rgb.permute(1, 2, 0).numpy()
                cmap = None
                title = "MarsHiRISE — false colour (NIR→R, RED→G, BG→B)"
        else:
            img_np = image[0].numpy()
            cmap = "grey"
            title = f"MarsHiRISE — {ch[0]}"

        # Percentile stretch — only over non-zero (data) pixels
        img_out = img_np.copy()
        if img_out.ndim == 3:
            for c in range(img_out.shape[2]):
                band = img_out[..., c]
                data_pixels = band[band > 1e-6]
                if len(data_pixels) > 0:
                    p2, p98 = np.percentile(data_pixels, [2, 98])
                    if p98 > p2:
                        img_out[..., c] = np.clip((band - p2) / (p98 - p2), 0, 1)
                        img_out[..., c][band == 0] = 0  # keep nodata black
        else:
            data_pixels = img_out[img_out > 0]
            if len(data_pixels) > 0:
                p2, p98 = np.percentile(data_pixels, [2, 98])
                if p98 > p2:
                    img_out = np.clip((img_out - p2) / (p98 - p2), 0, 1)
                    img_out[img_np == 0] = 0

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_out, cmap=cmap, interpolation="nearest")
        ax.axis("off")
        if show_titles:
            ax.set_title(title)
        if suptitle is not None:
            fig.suptitle(suptitle)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Verification / index loading
    # ------------------------------------------------------------------

    def _verify(self) -> None:
        """Ensure the dataset is usable, downloading files if requested.

        Raises:
            DatasetNotFoundError: If the index is absent (and download=False),
                or if no JP2 files at all are found after index loading.
        """
        self.root.mkdir(parents=True, exist_ok=True)

        # --- index files ---
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

        # In _verify, replace the save block:
        if rebuilt or not self.reuse_cache or not self.spatial_index_cache.exists():
            self.spatial_index_cache.parent.mkdir(parents=True, exist_ok=True)
            cache_df = self.index.copy()
            cache_df["t_start"] = self.index.index.left.astype(str)
            cache_df["t_stop"] = self.index.index.right.astype(str)
            cache_df.reset_index(drop=True).to_file(self.spatial_index_cache)

        # --- sanity-check: at least one JP2 must exist ---
        # Check a sample of observations rather than all of them to keep
        # startup fast on large datasets.
        sample_size = min(20, len(self.index))
        sample_rows = self.index.sample(n=sample_size, random_state=0)
        found = 0
        for _, row in sample_rows.iterrows():
            for col in ("color_path", "red_path"):
                p = row[col]
                if p is not None and pathlib.Path(p).exists():
                    found += 1
                    break

        if found == 0:
            if not self.download:
                logger.warning(
                    "No JP2 files found under '%s' for any of %d sampled "
                    "observations. Either run with download=True to fetch images "
                    "or point 'root' at an existing PDS mirror that preserves the "
                    "original directory structure:\n"
                    "  <root>/MROHR_XXXX/DATA/PSP/ORB_XXXXXX_XXXXXX/"
                    "<OBS_ID>/<OBS_ID>_COLOR.JP2\n"
                    "  <root>/MROHR_XXXX/DATA/PSP/ORB_XXXXXX_XXXXXX/"
                    "<OBS_ID>/<OBS_ID>_RED.JP2",
                    self.root,
                    sample_size,
                )
            else:
                logger.warning(
                    "Download completed but no JP2 files found on disk. "
                    "Check network connectivity and available disk space."
                )

    def _download_index(self) -> None:
        index_url = f"{self.url}/INDEX"
        for ext in (".LBL", ".TAB"):
            fname = self._INDEX_STEM + ext
            download_url(f"{index_url}/{fname}", str(self.root), fname)

    def _load_index(self) -> None:
        lbl_path = self.root / f"{self._INDEX_STEM}.LBL"
        if not lbl_path.exists():
            raise DatasetNotFoundError(self)

        data = pdr.read(str(lbl_path))
        data.load("all")
        df: pd.DataFrame = data["RDR_INDEX_TABLE"]
        logger.info("Loaded cumulative index: %d rows.", len(df))

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
                    "target='%s' matched no rows in the cumulative index. "
                    "Note: the HiRISE index does not contain place names. "
                    "Use bbox=(lon_min, lat_min, lon_max, lat_max) to filter "
                    "by geographic region instead.",
                    self.target,
                )
            else:
                logger.info("After text filter '%s': %d rows.", self.target, len(df))

        # Spatial/bbox filter — normalise PDS 0-360 longitudes first
        if self.bbox is not None:
            lon_min, lat_min, lon_max, lat_max = self.bbox

            def _norm(lon: pd.Series) -> pd.Series:
                return ((lon + 180.0) % 360.0) - 180.0

            obs_lon_min = _norm(df["MINIMUM_LONGITUDE"].astype(float))
            obs_lon_max = _norm(df["MAXIMUM_LONGITUDE"].astype(float))
            obs_lat_min = df["MINIMUM_LATITUDE"].astype(float)
            obs_lat_max = df["MAXIMUM_LATITUDE"].astype(float)

            # Keep observations whose bounding box overlaps the query bbox
            overlap = (
                    (obs_lon_max >= lon_min) & (obs_lon_min <= lon_max) &
                    (obs_lat_max >= lat_min) & (obs_lat_min <= lat_max)
            )
            df = df[overlap].reset_index(drop=True)
            logger.info(
                "After bbox filter %s: %d rows.", self.bbox, len(df)
            )

        self._raw_index = df

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _pds_local_path(self, spec: str) -> pathlib.Path:
        # Normalise separators (PDS uses forward slashes on all platforms)
        rel = pathlib.PurePosixPath(spec.strip())
        return self.root / 'images' / pathlib.Path(rel).name

    def _pds_local_stem(self, spec: str) -> pathlib.Path:
        """Return the local path stem (without extension) for *spec*."""
        jp2 = self._pds_local_path(spec)
        return jp2.with_suffix("")

    # ------------------------------------------------------------------
    # Spatial index
    # ------------------------------------------------------------------

    def _extract_data_footprint(
            self,
            jp2_path: pathlib.Path,
    ) -> "Polygon | None":
        """Return the convex hull of non-zero pixels in *jp2_path*.

        Reads at the coarsest available overview level so even multi-GB
        images resolve to a few hundred pixels.  The convex hull is
        computed in pixel coordinates (cheap) then only the hull vertices
        are reprojected to the dataset's geographic CRS.

        Returns ``None`` if the file cannot be opened or contains no data.
        """
        from rasterio.warp import transform as warp_transform

        path = self._prefer_cog(jp2_path)
        if path is None or not path.exists():
            return None

        dst_crs = rasterio.crs.CRS.from_user_input(self.mars_crs)

        try:
            with rasterio.open(path) as src:
                src_crs = src.crs
                if src_crs is None:
                    return None

                # Pick the coarsest overview — or downsample manually.
                overviews = src.overviews(1)  # list of reduction factors
                if overviews:
                    factor = max(overviews)
                else:
                    # No overviews (raw JP2) — target ~500 px on the
                    # shorter axis so we don't decompress the full image.
                    factor = max(1, min(src.height, src.width) // 500)

                out_h = max(1, src.height // factor)
                out_w = max(1, src.width // factor)

                # Read a single band at reduced resolution.
                data = src.read(1, out_shape=(out_h, out_w))

                # Non-zero pixel coordinates.
                ys, xs = np.where(data > 0)
                if len(xs) < 3:
                    return None

                # ── Convex hull in pixel space (fast) ─────────────────
                # Subsample to at most ~4000 points for the hull input;
                # shapely's convex_hull is O(n log n) so this is fine.
                step = max(1, len(xs) // 4000)
                xs_sub = xs[::step]
                ys_sub = ys[::step]

                hull_shapely = MultiPoint(
                    list(zip(xs_sub.tolist(), ys_sub.tolist()))
                ).convex_hull

                if hull_shapely.is_empty:
                    return None

                # Extract hull vertices (typically 4–20 points).
                hull_px = np.array(hull_shapely.exterior.coords)
                hx = hull_px[:, 0]  # pixel x (column)
                hy = hull_px[:, 1]  # pixel y (row)

                # ── Pixel → source CRS ───────────────────────────────
                # Build an affine for the reduced-resolution grid.
                ovr_transform = rasterio.transform.from_bounds(
                    *src.bounds, out_w, out_h
                )
                # rasterio.transform.xy wants (row, col)
                src_xs, src_ys = rasterio.transform.xy(
                    ovr_transform, hy.tolist(), hx.tolist()
                )

                # ── Source CRS → geographic CRS ──────────────────────
                geo_xs, geo_ys = warp_transform(
                    src_crs, dst_crs, list(src_xs), list(src_ys)
                )

                # Normalise longitudes to [−180, 180].
                geo_xs = [((x + 180.0) % 360.0) - 180.0 for x in geo_xs]

                footprint = Polygon(zip(geo_xs, geo_ys))
                if not footprint.is_valid:
                    footprint = footprint.buffer(0)
                return footprint if (footprint.is_valid and not footprint.is_empty) else None

        except Exception as exc:
            logger.debug("Could not extract footprint from %s: %s", jp2_path, exc)
            return None

    @property
    def spatial_index_cache(self) -> pathlib.Path:
        parts = []
        if self.target:
            parts.append(self.target)
        if self.bbox:
            parts.append(
                f"{self.bbox[0]}_{self.bbox[1]}_{self.bbox[2]}_{self.bbox[3]}"
            )
        suffix = f"_{'_'.join(parts)}" if parts else ""
        return self.root / f"spatial_cache{suffix}_v3.gpkg"

    def _build_spatial_index(self, force_rebuild: bool = False) -> None:
        """Build the GeoDataFrame spatial index from the cumulative PDS index.

        Groups the one-row-per-JP2 table by observation ID so each row in
        :attr:`index` represents one unique observation and carries both
        ``color_path`` and ``red_path`` (either may be ``None`` if absent).

        Geometries are degree-valued strip polygons built from the four corner
        coordinates (CORNER1-4) in the cumulative index.  This accurately
        represents the long, thin, rotated parallelogram of each HiRISE pass
        and avoids sampling patches in the empty corners of the bounding box.
        Falls back to JP2-derived bounds, then to cumulative-index min/max
        bounds (as an axis-aligned rectangle), when corner data are unavailable.

        Longitudes are normalised from [0°, 360°] to [−180°, 180°];
        antimeridian-crossing tiles are skipped.
        """
        # ── Cache check (unchanged) ──────────────────────────────────
        if (
                self.reuse_cache
                and not force_rebuild
                and self.spatial_index_cache.exists()
                and self.index is None
        ):
            logger.info(
                "Loading spatial index from cache (%s).", self.spatial_index_cache
            )
            gdf = gpd.read_file(self.spatial_index_cache)

            # Detect legacy bbox-only cache.
            sample_geoms = gdf.geometry.iloc[: min(5, len(gdf))]
            is_legacy = all(g.equals(box(*g.bounds)) for g in sample_geoms)
            if is_legacy:
                logger.info(
                    "Legacy bbox cache detected at %s; rebuilding with "
                    "strip polygon footprints.",
                    self.spatial_index_cache,
                )
            else:
                t_start = pd.to_datetime(gdf.pop("t_start"), utc=True)
                t_stop = pd.to_datetime(gdf.pop("t_stop"), utc=True)
                gdf.index = pd.IntervalIndex.from_arrays(
                    t_start, t_stop, closed="both", name="datetime"
                )
                self.index = gdf
                total_bounds = self.index["geometry"].bounds
                minx, miny = total_bounds[["minx", "miny"]].min()
                maxx, maxy = total_bounds[["maxx", "maxy"]].max()
                logger.info(
                    "Spatial index: %d observations, "
                    "lon [%.2f, %.2f] lat [%.2f, %.2f].",
                    len(self.index), minx, maxx, miny, maxy,
                )
                return

        # ── Phase 1: group products by observation (fast) ────────────
        df = self._raw_index.copy()
        pid = df["PRODUCT_ID"].str.strip()
        df["_product_type"] = pid.str.extract(r"_(COLOR|RED)\s*$", expand=False)
        df["_obs_id"] = pid.str.replace(r"_(COLOR|RED)\s*$", "", regex=True)
        df["_local_path"] = df["FILE_NAME_SPECIFICATION"].apply(
            lambda s: str(self._pds_local_path(s))
        )

        # Build a list of (obs_id, color_path, red_path, ref_row) and
        # pick the best file for footprint extraction.
        obs_list: list[tuple[str, str | None, str | None, str | None, "pd.Series"]] = []
        for obs_id, grp in df.groupby("_obs_id", sort=False):
            ref = grp.iloc[0]
            color_rows = grp[grp["_product_type"] == "COLOR"]
            red_rows = grp[grp["_product_type"] == "RED"]
            cp = color_rows.iloc[0]["_local_path"] if not color_rows.empty else None
            rp = red_rows.iloc[0]["_local_path"] if not red_rows.empty else None

            # Pick the first file that exists on disk (or has a COG).
            fp_path = None
            for p in (cp, rp):
                if p is not None:
                    pp = pathlib.Path(p)
                    if pp.exists() or pp.with_suffix(".tif").exists():
                        fp_path = p
                        break

            obs_list.append((str(obs_id), cp, rp, fp_path, ref))

        # ── Phase 2: parallel footprint extraction (the slow part) ───
        mars_crs = rasterio.crs.CRS.from_user_input(self.mars_crs)
        n_workers = min(os.cpu_count() or 4, 32)
        n_total = len(obs_list)
        n_with_files = sum(1 for _, _, _, fp, _ in obs_list if fp is not None)

        logger.info(
            "Extracting strip footprints: %d observations "
            "(%d with files on disk), %d threads ...",
            n_total, n_with_files, n_workers,
        )

        # Results indexed by position in obs_list.
        fp_results: list[
            tuple[list[tuple[float, float]] | None, tuple | None]
        ] = [(None, None)] * n_total

        done_count = 0
        lock = threading.Lock()

        def _do_extract(idx: int) -> None:
            nonlocal done_count
            _, _, _, fp_path, _ = obs_list[idx]
            fp_results[idx] = _extract_footprint(fp_path, mars_crs)
            with lock:
                done_count += 1
                if done_count % 50 == 0 or done_count == n_with_files:
                    logger.info(
                        "  footprint extraction: %d / %d done",
                        done_count, n_with_files,
                    )

        # Only submit work for observations that have a file on disk.
        indices_with_files = [
            i for i, (_, _, _, fp, _) in enumerate(obs_list) if fp is not None
        ]

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_do_extract, i) for i in indices_with_files]
            for f in tqdm(
                    as_completed(futs),
                    total=len(futs),
                    desc="Extracting footprints",
                    unit="strip",
            ):
                f.result()

        # ── Phase 3: assemble records (fast) ─────────────────────────
        records: list[dict] = []
        n_polygon = 0
        n_jp2_bbox = 0
        n_index_bbox = 0

        for i, (obs_id, cp, rp, _, ref) in enumerate(obs_list):
            hull_coords, file_bounds = fp_results[i]

            geom = None

            # Case A: convex hull of non-zero pixels.
            if hull_coords is not None:
                candidate = Polygon(hull_coords)
                if not candidate.is_valid:
                    candidate = candidate.buffer(0)
                if candidate.is_valid and not candidate.is_empty and candidate.area > 1e-12:
                    geom = candidate
                    n_polygon += 1

            # Case B: JP2/COG bounding box.
            if geom is None and file_bounds is not None:
                geom = box(*file_bounds)
                n_jp2_bbox += 1

            # Case C: cumulative-index min/max bbox.
            if geom is None:
                lon_min = ((float(ref["MINIMUM_LONGITUDE"]) + 180.0) % 360.0) - 180.0
                lon_max = ((float(ref["MAXIMUM_LONGITUDE"]) + 180.0) % 360.0) - 180.0
                if lon_min > lon_max:
                    logger.warning(
                        "Observation %s straddles antimeridian. Skipping.", obs_id
                    )
                    continue
                lat_min = float(ref["MINIMUM_LATITUDE"])
                lat_max = float(ref["MAXIMUM_LATITUDE"])
                geom = box(lon_min, lat_min, lon_max, lat_max)
                n_index_bbox += 1

            records.append(
                {
                    "obs_id": obs_id,
                    "color_path": cp,
                    "red_path": rp,
                    "geometry": geom,
                    "t_start": ref["START_TIME"],
                    "t_stop": ref["STOP_TIME"],
                }
            )

        logger.info(
            "Footprint sources — polygon: %d, JP2 bbox: %d, index bbox: %d.",
            n_polygon, n_jp2_bbox, n_index_bbox,
        )

        # ── Phase 4: build GeoDataFrame (unchanged) ──────────────────
        obs_df = pd.DataFrame(records)
        if obs_df.empty:
            raise DatasetNotFoundError(self)

        t_start = pd.to_datetime(obs_df["t_start"], utc=True, errors="coerce")
        t_stop = pd.to_datetime(obs_df["t_stop"], utc=True, errors="coerce")
        t_stop = t_stop.fillna(t_start)

        geometries = gpd.GeoSeries(obs_df["geometry"].tolist(), crs=self.mars_crs)

        self.index = gpd.GeoDataFrame(
            {
                "obs_id": obs_df["obs_id"].values,
                "color_path": obs_df["color_path"].values,
                "red_path": obs_df["red_path"].values,
            },
            index=pd.IntervalIndex.from_arrays(
                t_start, t_stop, closed="both", name="datetime"
            ),
            geometry=geometries.values,
            crs=self.mars_crs,
        )

        total_bounds = self.index["geometry"].bounds
        minx, miny = total_bounds[["minx", "miny"]].min()
        maxx, maxy = total_bounds[["maxx", "maxy"]].max()
        logger.info(
            "Spatial index: %d observations, "
            "lon [%.2f, %.2f] lat [%.2f, %.2f].",
            len(self.index), minx, maxx, miny, maxy,
        )

    def _read_jp2_bounds(
            self, jp2_path: pathlib.Path
    ) -> tuple[float, float, float, float] | None:
        if not jp2_path.exists():
            return None
        dst_crs = rasterio.crs.CRS.from_user_input(self.mars_crs)
        try:
            with rasterio.open(jp2_path) as src:
                src_crs = src.crs
                if src_crs is None:
                    logger.debug(
                        "JP2 has no embedded CRS; cannot derive bounds from file: %s",
                        jp2_path.name,
                    )
                    return None  # don't risk treating metre coords as degrees
                fl, fb, fr, ft = transform_bounds(src_crs, dst_crs, *src.bounds)
                # Normalise longitude to [−180, 180]; latitude needs no wrapping.
                fl = ((fl + 180.0) % 360.0) - 180.0
                fr = ((fr + 180.0) % 360.0) - 180.0
                # Sanity-check: reject anything outside valid geographic ranges.
                if not (-180.0 <= fl < fr <= 180.0 and -90.0 <= fb < ft <= 90.0):
                    logger.debug(
                        "Out-of-range bounds from %s (lon=[%.3f,%.3f] lat=[%.3f,%.3f]); "
                        "falling back to index.",
                        jp2_path.name, fl, fr, fb, ft,
                    )
                    return None
                return fl, fb, fr, ft
        except Exception as exc:
            logger.debug("Could not read bounds from %s: %s", jp2_path, exc)
            return None

    # ------------------------------------------------------------------
    # Image download
    # ------------------------------------------------------------------

    def _build_download_tasks(self) -> list[tuple[str, pathlib.Path]]:
        """Return ``(remote_url, local_path)`` pairs for all JP2 and LBL files.

        Remote URL: ``<self.url>/<PDS_spec_without_ext><ext>``
        Local path: ``<root>/<PDS_spec_without_ext><ext>``  (full PDS tree)
        """
        tasks: list[tuple[str, pathlib.Path]] = []
        for spec in self._raw_index["FILE_NAME_SPECIFICATION"]:
            spec = spec.strip()
            stem_str = spec[: -len(".JP2")]  # remote path without extension
            local_stem = self._pds_local_stem(spec)
            for ext in (".JP2", ".LBL"):
                remote = f"{self.url}/{stem_str}{ext}"
                local = local_stem.with_suffix(ext)

                if not local.exists():
                    tasks.append((remote, local))
        return tasks

    def _download_images(self) -> bool:
        tasks = self._build_download_tasks()
        if not tasks:
            logger.info("Everything is already downloaded")
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
                pool.submit(_worker_process, chunk, concurrency_per_process, stop_event)
                for chunk in chunks if chunk
            ]
            for fut in futures:
                fut.result()
        if stop_event.is_set():
            logger.warning("Download halted early to preserve disk space.")

        return True

    # ------------------------------------------------------------------
    # Tile loading
    # ------------------------------------------------------------------

    @staticmethod
    def _prefer_cog(path: pathlib.Path | None) -> pathlib.Path | None:
        """Return the COG sidecar (.tif) for *path* if it exists, else *path* itself.

        When preprocessing.py converts JP2 files to Cloud-Optimized GeoTIFFs,
        it creates a sidecar ``<stem>.tif`` next to the original JP2.  Loading
        from the COG is significantly faster for small random-access patches
        because its internal 512×512 tiling avoids decompressing large JP2
        codeblocks.
        """
        if path is None:
            return None
        cog = path.with_suffix(".tif")
        return cog if cog.exists() else path

    def _load_tile(
            self,
            color_path: pathlib.Path | None,
            red_path: pathlib.Path | None,
            x: slice,
            y: slice,
    ) -> torch.Tensor | None:
        """Load and assemble the requested channels from one observation."""
        # Prefer pre-converted COG sidecars for faster windowed reads.
        color_path = self._prefer_cog(color_path)
        red_path = self._prefer_cog(red_path)

        requested = set(self.channels)
        color_available = color_path is not None and color_path.exists()
        red_available = red_path is not None and red_path.exists()

        only_red = requested == {"RED"}
        use_red_file = only_red and red_available

        from_color: set[str] = set() if use_red_file else requested.copy()
        from_red: set[str] = {"RED"} if use_red_file else set()

        if from_color and not color_available:
            salvageable = from_color & set(_RED_CHANNELS)
            lost = from_color - salvageable
            if lost:
                logger.warning(
                    "COLOR unavailable (%s); cannot provide channels %s.",
                    color_path, lost,
                )
            from_red |= salvageable
            from_color = set()

        if from_red and not red_available:
            logger.warning("RED file unavailable: %s", red_path)
            from_red = set()

        if not from_color and not from_red:
            return None

        band_arrays: dict[str, np.ndarray] = {}

        if from_color and color_available:
            lbl = color_path.with_suffix(".LBL")
            meta = _ProductMeta.from_lbl(lbl)
            if meta.filter_names and len(meta.filter_names) == meta.bands:
                color_band_map = {
                    name: idx + 1 for idx, name in enumerate(meta.filter_names)
                }
            else:
                color_band_map = _COLOR_BAND.copy()
            band_map = {
                ch: color_band_map[ch]
                for ch in from_color
                if ch in color_band_map
            }
            band_arrays.update(self._load_from_jp2(color_path, band_map, meta, x, y))

        if from_red and red_available:
            lbl = red_path.with_suffix(".LBL")
            meta = _ProductMeta.from_lbl(lbl)
            meta.filter_names = ["RED"]
            meta.bands = 1
            band_map = {ch: _RED_BAND[ch] for ch in from_red if ch in _RED_BAND}
            band_arrays.update(self._load_from_jp2(red_path, band_map, meta, x, y))

        if not band_arrays:
            return None

        # Determine output spatial shape from first loaded band.
        first = next(iter(band_arrays.values()))
        out_h, out_w = first.shape

        # Build channel tensors in self.channels order.
        # Missing channels (e.g. NIR/BG when COLOR file is absent) are filled
        # with zeros so every tile always has exactly len(self.channels) channels.
        tensors = []
        for ch in self.channels:
            if ch in band_arrays:
                tensors.append(torch.from_numpy(band_arrays[ch]))
            else:
                tensors.append(torch.zeros(out_h, out_w, dtype=torch.float32))
        return torch.stack(tensors)

    def _load_from_jp2(
            self,
            jp2_path: pathlib.Path,
            band_map: dict[str, int],
            meta: _ProductMeta,
            x: slice,
            y: slice,
    ) -> dict[str, np.ndarray]:
        """Reproject bands from a JP2 into geographic degrees and calibrate.

        Each HiRISE JP2 has its own Equirectangular projection with a unique
        CENTER_LATITUDE.  rasterio reads ``src.crs`` from the file and
        reprojects into the geographic Mars CRS automatically.  The destination
        grid is defined by the query slice (``x``, ``y``) in degrees and the
        native HiRISE resolution.
        """
        result: dict[str, np.ndarray] = {}

        if not band_map:
            return result

        out_w = max(1, int(round((x.stop - x.start) / (x.step or self.res))))
        out_h = max(1, int(round((y.stop - y.start) / (y.step or self.res))))

        dst_transform = rasterio.transform.from_bounds(
            x.start, y.start, x.stop, y.stop, out_w, out_h
        )
        dst_crs = rasterio.crs.CRS.from_user_input(self.mars_crs)

        try:
            with rasterio.open(jp2_path) as src:
                src_crs = src.crs
                if src_crs is None:
                    logger.warning(
                        "JP2 has no embedded CRS, assuming dataset CRS: %s",
                        jp2_path,
                    )
                    src_crs = dst_crs

                # Early-exit if the file doesn't overlap the query window.
                # _SPATIAL_TOL absorbs floating-point imprecision between the
                # index geometry (built at dataset construction time) and the
                # bounds that rasterio recomputes here at load time.
                try:
                    fl, fb, fr, ft = transform_bounds(src_crs, dst_crs, *src.bounds)
                    # Normalize to [-180, 180] to match the index/query convention.
                    # HiRISE JP2s use an equirectangular with lon_0=180, so transform_bounds
                    # may return 0-360 values; without this, the check produces false negatives.
                    fl = ((fl + 180.0) % 360.0) - 180.0
                    fr = ((fr + 180.0) % 360.0) - 180.0
                    # Only apply the check when bounds are non-inverted (i.e., no antimeridian wrap).
                    if fl <= fr:
                        _tol = _SPATIAL_TOL
                        if (
                                fr + _tol < x.start
                                or fl - _tol > x.stop
                                or ft + _tol < y.start
                                or fb - _tol > y.stop
                        ):
                            logger.debug(
                                "Query [%.4f,%.4f,%.4f,%.4f] doesn't overlap "
                                "file bounds [%.4f,%.4f,%.4f,%.4f]: %s",
                                x.start, y.start, x.stop, y.stop,
                                fl, fb, fr, ft, jp2_path.name,
                            )
                            return result
                except Exception:
                    pass

                for ch_name, band_idx in band_map.items():
                    dest = np.empty((out_h, out_w), dtype=np.float32)
                    try:
                        reproject(
                            source=rasterio.band(src, band_idx),
                            destination=dest,
                            src_transform=src.transform,
                            src_crs=src_crs,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.bilinear,
                            dst_nodata=0.0,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Reprojection failed for %s band %d: %s",
                            jp2_path.name, band_idx, exc,
                        )
                        continue
                    nodata_mask = dest == 0.0

                    dest *= meta.scaling_factor
                    dest += meta.offset
                    np.clip(dest, 0.0, 1.0, out=dest)
                    dest[nodata_mask] = 0.0

                    result[ch_name] = dest

        except rasterio.errors.RasterioIOError as exc:
            logger.warning("Could not open %s: %s", jp2_path, exc)

        return result

    @staticmethod
    def _merge_tiles(tiles: list[torch.Tensor]) -> torch.Tensor:
        """Mosaic co-registered tiles with a first-non-zero-wins strategy."""
        if len(tiles) == 1:
            return tiles[0]

        max_h = max(t.shape[1] for t in tiles)
        max_w = max(t.shape[2] for t in tiles)
        n_ch = max(t.shape[0] for t in tiles)

        # Pad any tile that has fewer channels than the maximum (e.g. due to a
        # partially-unavailable COLOR file).  This should not happen after
        # _load_tile zero-fills missing channels, but guard here for safety.
        if not all(t.shape[0] == n_ch for t in tiles):
            logger.warning(
                "_merge_tiles: channel count mismatch across %d tiles "
                "(max %d channels) — zero-padding narrower tiles.",
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

    def plot_coverage(
            self,
            resolution: float | None = None,
            show_count: bool = True,
            suptitle: str | None = None,
    ) -> Figure:
        """Visualise the spatial coverage of all observations in the index.

        The plot is cropped to the **actual extent** of the observations rather
        than the full Mars surface, so sparse targeted datasets (e.g. filtered
        by ``target="Olympus"``) are shown at a useful zoom level.

        Each observation bounding box is drawn as a semi-transparent rectangle
        with a unique colour, making it easy to see individual footprints and
        where they overlap.  A count heatmap is rendered underneath to show
        the cumulative overlap density.

        Args:
            resolution: Coverage-grid cell size in **degrees**.  Defaults to
                1/20th of the narrower extent dimension, which gives ~20 cells
                across the tightest axis regardless of how zoomed-in the
                dataset is.
            show_count: Draw a colourbar for the count heatmap.
            suptitle: Optional figure-level title.

        Returns:
            A :class:`~matplotlib.figure.Figure` with a main panel (heatmap +
            bounding boxes), a longitudinal marginal on top, a latitudinal
            marginal on the right, and a text summary.
        """

        if self.index is None or len(self.index) == 0:
            raise RuntimeError("Spatial index is empty — nothing to plot.")

        # ── Derive plot extent from the actual observations ───────────
        all_bounds = self.index.geometry.bounds  # DataFrame: minx miny maxx maxy
        lon_min = float(all_bounds["minx"].min())
        lon_max = float(all_bounds["maxx"].max())
        lat_min = float(all_bounds["miny"].min())
        lat_max = float(all_bounds["maxy"].max())

        # Add a small margin (5 % of each span) so edge boxes aren't clipped.
        lon_span = lon_max - lon_min or 1.0
        lat_span = lat_max - lat_min or 1.0
        margin_lon = lon_span * 0.05
        margin_lat = lat_span * 0.05
        lon_min -= margin_lon
        lon_max += margin_lon
        lat_min -= margin_lat
        lat_max += margin_lat

        # Default resolution: ~20 cells across the narrower axis.
        if resolution is None:
            resolution = min(lon_span, lat_span) / 20.0

        # ── Build coverage grid scoped to actual extent ───────────────
        coverage, lon_edges, lat_edges = self._coverage_grid(
            resolution=resolution,
            lon_min=lon_min, lon_max=lon_max,
            lat_min=lat_min, lat_max=lat_max,
        )
        lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
        lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2

        # ── Summary statistics ────────────────────────────────────────
        n_obs = len(self.index)
        covered_cells = int(np.count_nonzero(coverage))
        total_cells = coverage.size
        pct_covered = 100.0 * covered_cells / total_cells if total_cells else 0.0
        median_overlap = (
            float(np.median(coverage[coverage > 0])) if covered_cells else 0.0
        )
        max_overlap = int(coverage.max())

        # ── Layout ────────────────────────────────────────────────────
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(
            2, 2,
            width_ratios=[4, 1],
            height_ratios=[1, 4],
            hspace=0.05,
            wspace=0.05,
        )
        ax_main = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

        # ── Heatmap (count, log-scaled) ───────────────────────────────
        display = np.ma.masked_equal(coverage, 0)
        norm = LogNorm(vmin=1, vmax=max(max_overlap, 1)) if show_count else None
        im = ax_main.pcolormesh(
            lon_edges, lat_edges, display,
            cmap="YlOrRd",
            norm=norm,
            shading="flat",
            rasterized=True,
            zorder=1,
        )
        # Grey out zero-coverage cells.
        ax_main.pcolormesh(
            lon_edges, lat_edges,
            np.ma.masked_not_equal(coverage, 0),
            cmap="Greys", vmin=0, vmax=1,
            shading="flat", rasterized=True, zorder=0,
        )

        # ── Individual bounding boxes, each a unique colour ───────────
        # Use a high-contrast qualitative cycle; wrap around if n_obs > n_colors.
        cmap_boxes = matplotlib.colormaps.get_cmap("tab20")  # type: ignore[attr-defined]
        rects = []
        for i, geom in enumerate(self.index.geometry):
            b = geom.bounds  # (minx, miny, maxx, maxy)
            w = b[2] - b[0]
            h = b[3] - b[1]
            rects.append(Rectangle((b[0], b[1]), w, h))

        # Draw as a PatchCollection so matplotlib batches the render.
        box_colors = [cmap_boxes(i % 20) for i in range(n_obs)]
        pc = PatchCollection(
            rects,
            facecolors=[(r, g, b, 0.08) for r, g, b, _ in box_colors],  # very transparent fill
            edgecolors=[(r, g, b, 0.85) for r, g, b, _ in box_colors],  # solid edge
            linewidths=0.6,
            zorder=2,
        )
        ax_main.add_collection(pc)

        ax_main.set_xlim(lon_min, lon_max)
        ax_main.set_ylim(lat_min, lat_max)
        ax_main.set_xlabel("Longitude (°E, normalised to [−180, 180])")
        ax_main.set_ylabel("Latitude (°)")

        # Tick density: ~6 ticks per axis, rounded to a nice interval.
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
            cbar.set_label("Observations per cell (log scale)", fontsize=8)

        legend_handles = [
            Patch(facecolor="#bbbbbb", label="No coverage"),
            Patch(facecolor=plt.cm.YlOrRd(0.15), label="Low overlap"),  # type: ignore[attr-defined]
            Patch(facecolor=plt.cm.YlOrRd(0.85), label="High overlap"),  # type: ignore[attr-defined]
        ]
        ax_main.legend(
            handles=legend_handles, loc="lower left",
            fontsize=7, framealpha=0.75,
        )

        # ── Top marginal: longitudinal distribution ───────────────────
        ax_top.bar(
            lon_centers, coverage.sum(axis=0),
            width=resolution, color="steelblue", alpha=0.8, linewidth=0,
        )
        ax_top.set_ylabel("Obs.\nsum", fontsize=8)
        ax_top.tick_params(labelbottom=False, labelsize=7)
        ax_top.grid(axis="y", linewidth=0.4, alpha=0.5)

        # ── Right marginal: latitudinal distribution ──────────────────
        ax_right.barh(
            lat_centers, coverage.sum(axis=1),
            height=resolution, color="steelblue", alpha=0.8, linewidth=0,
        )
        ax_right.set_xlabel("Obs.\nsum", fontsize=8)
        ax_right.tick_params(labelleft=False, labelsize=7)
        ax_right.grid(axis="x", linewidth=0.4, alpha=0.5)

        # ── Summary text ──────────────────────────────────────────────
        summary = (
            f"Observations : {n_obs:,}\n"
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

        title = suptitle or f"MarsHiRISE coverage — {n_obs:,} observations"
        if self.target:
            title += f"  (filter: '{self.target}')"
        fig.suptitle(title, fontsize=12, y=1.005)
        fig.tight_layout()
        return fig

    def _coverage_grid(
            self,
            resolution: float = 1.0,
            lon_min: float = -180.0,
            lon_max: float = 180.0,
            lat_min: float = -90.0,
            lat_max: float = 90.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a raw observation-count grid scoped to a bounding extent.

        Args:
            resolution: Grid cell size in degrees.
            lon_min: Left edge of the grid (degrees).
            lon_max: Right edge of the grid (degrees).
            lat_min: Bottom edge of the grid (degrees).
            lat_max: Top edge of the grid (degrees).

        Returns:
            ``(coverage, lon_edges, lat_edges)`` where ``coverage`` is an
            ``(n_lat, n_lon)`` int32 array of observation counts per cell.
        """
        lon_edges = np.arange(lon_min, lon_max + resolution, resolution)
        lat_edges = np.arange(lat_min, lat_max + resolution, resolution)
        n_lon = len(lon_edges) - 1
        n_lat = len(lat_edges) - 1
        coverage = np.zeros((n_lat, n_lon), dtype=np.int32)

        for geom in self.index.geometry:
            b = geom.bounds  # (minx, miny, maxx, maxy)
            i_lon_lo = max(0, int(np.floor((b[0] - lon_min) / resolution)))
            i_lon_hi = min(n_lon, int(np.ceil((b[2] - lon_min) / resolution)))
            i_lat_lo = max(0, int(np.floor((b[1] - lat_min) / resolution)))
            i_lat_hi = min(n_lat, int(np.ceil((b[3] - lat_min) / resolution)))
            if i_lon_lo < i_lon_hi and i_lat_lo < i_lat_hi:
                coverage[i_lat_lo:i_lat_hi, i_lon_lo:i_lon_hi] += 1

        return coverage, lon_edges, lat_edges


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(config_path: str = CONFIG) -> None:
    with open(config_path) as fh:
        logging.config.dictConfig(json.load(fh))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    setup_logging()

    torch.manual_seed(42)
    np.random.seed(42)

    from torch.utils.data import DataLoader

    from hirise_sampler import HiRISEGeoSampler

    # dataset = MarsHiRISE(
    #     target="Olympus",
    #     channels=["NEAR-INFRARED", "RED", "BLUE-GREEN"],
    #     download=True,  # set False if you already have a local mirror
    #     reuse_cache=True
    # )

    dataset = MarsHiRISE(
        bbox=(-136, 12, -124, 24),  # Olympus Mons extent from the CTX metadata
        channels=["NEAR-INFRARED", "RED", "BLUE-GREEN"],
        download=True,
        reuse_cache=True,
    )

    fig = dataset.plot_coverage()
    fig.savefig("coverage.png")

    logger.info("saved fig")

    # HiRISEGeoSampler pre-computes a grid of valid patch centers within each
    # strip polygon, so every yielded patch is guaranteed to intersect real data.
    # size= is in degrees (units of self.crs = geographic Mars CRS).
    # 0.005 deg ≈ 593 pixels ≈ 296 m at the equator.
    sampler = HiRISEGeoSampler(dataset, size=0.005, length=200, units=Units.CRS)
    dataloader = DataLoader(dataset, sampler=sampler)

    output_path = pathlib.Path("Figures")

    output_path.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(dataloader):
        if i >= 10:
            break

        fig = dataset.plot(sample)
        fig.savefig(output_path / f"output{i}.png")
        logger.info("Saved fig output%d.png", i)

        plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    main()
