# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""MarsHiRISE dataset."""
import asyncio
import json
import logging
import logging.config
import multiprocessing
import pathlib
import random
import re
import shutil
import threading
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import field, dataclass
from typing import Literal

import aiohttp
import geopandas as gpd
import numpy as np
import pandas as pd
import pdr
import rasterio
import torch
from aiohttp import ClientConnectorError, ClientResponseError
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject
from shapely.geometry import box
from torch.utils.data import DataLoader
from torchgeo.datasets.geo import GeoDataset
from torchgeo.datasets.utils import GeoSlice, Path, Sample, download_url
from torchgeo.samplers import RandomGeoSampler

Channel = Literal["NEAR-INFRARED", "RED", "BLUE-GREEN"]

ALL_CHANNELS: tuple[Channel, ...] = ("NEAR-INFRARED", "RED", "BLUE-GREEN")

# In-file band ordering for each product type.
_COLOR_CHANNELS: tuple[Channel, ...] = ("NEAR-INFRARED", "RED", "BLUE-GREEN")
_RED_CHANNELS: tuple[Channel, ...] = ("RED",)

# rasterio band indices (1-based) within each product file.
_COLOR_BAND: dict[Channel, int] = {"NEAR-INFRARED": 1, "RED": 2, "BLUE-GREEN": 3}
_RED_BAND: dict[Channel, int] = {"RED": 1}


# /* The source image data definition. */
# OBJECT = UNCOMPRESSED_FILE
#     FILE_NAME    = "ESP_027409_1960_UNFILTERED_COLOR.IMG"
#     RECORD_TYPE  = FIXED_LENGTH
#     RECORD_BYTES = 15642 <BYTES>
#     FILE_RECORDS = 134613
#     ^IMAGE       = "ESP_027409_1960_UNFILTERED_COLOR.IMG"
#     OBJECT = IMAGE
#         DESCRIPTION                = "HiRISE projected and mosaicked product"
#         LINES                      = 44871
#         LINE_SAMPLES               = 7821
#         BANDS                      = 3
#         SAMPLE_TYPE                = MSB_UNSIGNED_INTEGER
#         SAMPLE_BITS                = 16
#         SAMPLE_BIT_MASK            = 2#0000001111111111#
#         /* NOTE: The conversion from DN to I/F (intensity/flux) is: */
#         /* I/F = (DN * SCALING_FACTOR) + OFFSET                     */
#         /* I/F is defined as the ratio of the observed radiance and */
#         /* the radiance of a 100% lambertian reflector with the sun */
#         /* and camera orthogonal to the observing surface.          */
#         SCALING_FACTOR             = 2.37936949017414e-04
#         OFFSET                     = 0.037954361744101
#         BAND_STORAGE_TYPE          = BAND_SEQUENTIAL
#         CORE_NULL                  = 0
#         CORE_LOW_REPR_SATURATION   = 1
#         CORE_LOW_INSTR_SATURATION  = 2
#         CORE_HIGH_REPR_SATURATION  = 1023
#         CORE_HIGH_INSTR_SATURATION = 1022
#         CENTER_FILTER_WAVELENGTH   = (900 <NM>, 700 <NM>, 500 <NM>)
#         MRO:MINIMUM_STRETCH        = (3, 3, 3)
#         MRO:MAXIMUM_STRETCH        = (1021, 1021, 1021)
#         FILTER_NAME                = ("NEAR-INFRARED", "RED", "BLUE-GREEN")
#     END_OBJECT = IMAGE
# END_OBJECT = UNCOMPRESSED_FILE
# Default DN -> I/F conversion coefficients (from a representative COLOR LBL).
# Per-product LBL files are preferred; these are fallbacks only.
_DEFAULT_SCALING_FACTOR: float = 2.37936949017414e-04
_DEFAULT_OFFSET: float = 0.037954361744101
_DEFAULT_SAMPLE_BITS: int = 16
# Effective bit depth from SAMPLE_BIT_MASK = 2#0000001111111111# (10 set bits)
_EFFECTIVE_BIT_DEPTH: int = 10

# Mars 2000 geographic CRS: uses the IAU 2000 Mars ellipsoid.
# a = 3,396,190 m  (equatorial radius)
# b = 3,376,200 m  (polar radius)
# This is the standard used by HiRISE / MRO PDS products.
# https://hirise-pds.lpl.arizona.edu/PDS/DOCUMENT/HIRISE_RDR_SIS.PDF (3.5.1 Equirectangular Projection page 15)
MARS_CRS = CRS.from_proj4("+proj=longlat +a=3396190 +b=3376200 +no_defs")



# Define our minimum disk size: 100 GB in bytes
E_MIN_BYTES = 100 * (1024 ** 3)



logger = logging.getLogger(__name__)

CONFIG = 'logger_config.json'


def filter_maker(level: str):
    level = getattr(logging, level)

    def _filter(record: logging.LogRecord) -> bool:
        return record.levelno <= level

    return _filter




async def _download_file(session: aiohttp.ClientSession,
                         url: str,
                         path: pathlib.Path,
                         stop_event : threading.Event,
                         max_retries: int = 8,
                         base_delay: float = 1.0,
                         max_delay: float = 60.0) -> None:
    if stop_event.is_set():
        return

    disk_usage = shutil.disk_usage(path.parent if path.parent.exists() else "/")
    if disk_usage.free < E_MIN_BYTES:
        if not stop_event.is_set():
            logger.critical(
                f"CRITICAL: Available disk space {disk_usage.free / (1024 ** 3):.2f}GB fell below threshold. Halting system.")
            stop_event.set()  # Trip the global kill switch
        return

    if path.exists():
        logger.warning(f"File {path} already exists. Skipping download.")
        return

    attempt = 0
    while attempt <= max_retries:
        if stop_event.is_set():
            break

        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                path.parent.mkdir(parents=True, exist_ok=True)

                chunk_size = 1 << 20

                with open(path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(chunk_size):
                        if stop_event.is_set():
                            logger.error(f"System halted. Aborting inflight write for {path.name}")
                            return

                        await asyncio.to_thread(f.write, chunk)

            # If successful, break the loop and return
            return

        except ClientResponseError as e:
            # 429: Too Many Requests, 503: Service Unavailable, 504: Gateway Timeout
            if e.status in {429, 503, 504}:
                logger.warning(f"Server saturated ({e.status}) for {url}. Attempt {attempt + 1}/{max_retries}.")
            else:
                # Fatal error (e.g., 404 Not Found, 403 Forbidden). Do not retry.
                logger.error(f"Fatal HTTP {e.status} for {url}: {e.message}")
                return

        except (ClientConnectorError, asyncio.TimeoutError) as e:
            # Handle socket drops and timeouts which also occur during saturation
            logger.warning(f"Connection dropped for {url}. Attempt {attempt + 1}/{max_retries}. Error: {e}")

        except Exception as e:
            logger.error(f"Unexpected failure downloading {url}: {e}")
            return

        # Calculate Jittered Exponential Backoff
        attempt += 1
        if attempt <= max_retries:
            upper_bound = min(base_delay * (2 ** attempt), max_delay)
            sleep_time = random.uniform(0, upper_bound)

            logger.info(f"Backing off for {sleep_time:.2f} seconds before retrying {url}")
            await asyncio.sleep(sleep_time)
        else:
            logger.error(f"Max retries ({max_retries}) exhausted for {url}. File skipped.")


async def _download_many(tasks: list[tuple[str, pathlib.Path]],
                         concurrency: int,
                         stop_event: threading.Event) -> None:
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        await asyncio.gather(
            *(_download_file(session, url, path, stop_event) for url, path in tasks)
        )


def _worker_process(tasks: list[tuple[str, pathlib.Path]],
                    concurrency_per_process: int,
                    stop_event: threading.Event) -> None:
    asyncio.run(_download_many(tasks, concurrency_per_process, stop_event))


@dataclass
class _ProductMeta:
    """Radiometric metadata parsed from a per-product PDS3 LBL file."""

    scaling_factor: float = _DEFAULT_SCALING_FACTOR
    offset: float = _DEFAULT_OFFSET
    sample_bits: int = _DEFAULT_SAMPLE_BITS
    effective_max_dn: int = (1 << _EFFECTIVE_BIT_DEPTH) - 1  # 1023 for 10-bit
    filter_names: list[str] = field(default_factory=lambda: list(_COLOR_CHANNELS))
    bands: int = 3

    @classmethod
    def from_lbl(cls, lbl_path: pathlib.Path) -> "_ProductMeta":
        """Parse a PDS3 product LBL and return a populated instance.

        Falls back gracefully to defaults for any field that cannot be parsed,
        so that a partially-written or missing LBL never raises an exception.

        Args:
            lbl_path: Path to the companion ``.LBL`` file.

        Returns:
            A :class:`_ProductMeta` populated from the LBL, or with defaults.
        """
        obj = cls()
        if not lbl_path.exists():
            logger.debug("LBL not found, using defaults: %s", lbl_path)
            return obj

        try:
            text = lbl_path.read_text(errors="replace")
        except OSError as exc:
            logger.warning("Could not read LBL %s: %s", lbl_path, exc)
            return obj

        def _float(pattern: str) -> float | None:
            m = re.search(pattern, text, re.MULTILINE)
            return float(m.group(1)) if m else None

        def _int(pattern: str) -> int | None:
            m = re.search(pattern, text, re.MULTILINE)
            return int(m.group(1)) if m else None

        sf = _float(r"^\s*SCALING_FACTOR\s*=\s*([\d.eE+\-]+)")
        if sf is not None:
            obj.scaling_factor = sf

        off = _float(r"^\s*OFFSET\s*=\s*([\d.eE+\-]+)")
        if off is not None:
            obj.offset = off

        sb = _int(r"^\s*SAMPLE_BITS\s*=\s*(\d+)")
        if sb is not None:
            obj.sample_bits = sb

        nb = _int(r"^\s*BANDS\s*=\s*(\d+)")
        if nb is not None:
            obj.bands = nb

        # SAMPLE_BIT_MASK — derive effective_max_dn from the number of set bits.
        m_mask = re.search(r"SAMPLE_BIT_MASK\s*=\s*2#([01]+)#", text)
        if m_mask:
            n_set = m_mask.group(1).count("1")
            obj.effective_max_dn = (1 << n_set) - 1

        # FILTER_NAME list, e.g. ("NEAR-INFRARED", "RED", "BLUE-GREEN")
        m_filt = re.search(r"FILTER_NAME\s*=\s*\(([^)]+)\)", text, re.DOTALL)
        if m_filt:
            names = [
                s.strip().strip('"').strip("'")
                for s in m_filt.group(1).split(",")
            ]
            if names:
                obj.filter_names = names

        return obj


# We want to use
# TODO should actually make this into a GeoDataset instead of a NonGeoDataset as an example https://github.com/torchgeo/torchgeo/blob/main/torchgeo/datasets/eddmaps.py
class MarsHiRISE(GeoDataset):
    """Mars HiRISE Experiment Data Records dataset.

    HiRISE <https://pds-imaging.jpl.nasa.gov/volumes/mro.html> is a large dataset containing high resolution
    images of the Mars surface. This dataset focuses specifically on the RDR products which are
    radiometrically-corrected images resampled to a standard map projection. They are formatted and organized
    according to the standards of the PDS.

    The dataset is indexed spatially using the bounding-box columns in the cumulative RDR index (``RDRCUMINDEX.TAB``)
    and temporally using the ``START_TIME`` / ``STOP_TIME`` columns.  Coordinates are expressed in the Mars IAU 2000
    geographic CRS (semi-major axis 3,396,190 m; semi-minor axis 3,376,200 m,
    from <https://hirise-pds.lpl.arizona.edu/PDS/DOCUMENT/HIRISE_RDR_SIS.PDF>) because Mars uses a different ellipsoid
    than Earth and standard EPSG codes (which assume GRS-80 / WGS-84) cannot be used.

    Dataset homepage:
        https://hirise-pds.lpl.arizona.edu/PDS/AAREADME.TXT

    .. note::
        Images are stored as JPEG-2000 (.JP2) files and can be very large.  Set ``download=False`` and point ``root``
        at a pre-existing mirror to avoid re-downloading.
    """

    url: str = 'https://hirise-pds.lpl.arizona.edu/PDS'

    rdr_name = 'RDRCUMINDEX'

    mars_crs: CRS = MARS_CRS

    all_channels: tuple[str, ...] = ALL_CHANNELS

    def __init__(
            self,
            root: Path = '/scratch/mars_hirise',
            split: str = 'train',
            target: str | None = None,
            transforms: Callable[[Sample], Sample] | None = None,
            download: bool = False,
            checksum: bool = False,
            channels: list[Literal['RED', 'NEAR-INFRARED', 'BLUE-GREEN']] | None = None,
    ) -> None:
        super(GeoDataset).__init__()

        self.channels = channels
        self.target = target
        self.root = pathlib.Path(root)
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self.map_scale = 118502.26464032 # PIX/DEG
        self.map_resolution = 0.5 # METERS/PIXEL

        self.res = 1/self.map_scale # m / pix

        self._raw_index: pd.DataFrame | None = None

        self._verify()

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """

        return len(self.index)

    def _verify(self):
        self.root.mkdir(parents=True, exist_ok=True)

        if self.download:
            self._download_index()

        self._load_index()
        self._build_spatial_index()

        if self.download:
            self._download_images()

    def _load_index(self):
        lbl_path = self.root / f"{self.rdr_name}.LBL"

        if not lbl_path.exists():
            raise FileNotFoundError(
                f"Index label not found at {lbl_path}. "
                "Run with download=True or place the index files manually."
            )

        data = pdr.read(str(lbl_path))
        data.load('all')

        df: pd.DataFrame = data['RDR_INDEX_TABLE']

        logger.info(f"Loaded all cumulative data of size {len(df)}")

        if self.target is not None:
            string_cols = df.select_dtypes(include=["object", "string"])
            mask = string_cols.apply(
                lambda col: col.str.contains(self.target, na=False, regex=False, case=False)
            ).any(axis=1)
            df = df[mask].reset_index(drop=True)

            logger.info(
                "After filtering for '%s': %d rows.", self.target, len(df)
            )

        self._raw_index = df

    def _download_index(self) -> None:
        index_url = f"{self.url}/INDEX"

        for path in [".LBL", ".TAB"]:
            full_name = self.rdr_name + path

            download_url(f'{index_url}/{full_name}', str(self.root), full_name)

    def _build_spatial_index(self) -> None:
        """Build the GeoDataFrame spatial index from the raw PDS index table.

        The cumulative index has **one row per JP2 file**, so COLOR and RED
        products for the same observation appear as separate rows.  This
        method groups them by observation ID (derived by stripping the
        ``_COLOR`` / ``_RED`` suffix from ``PRODUCT_ID``) so that each row
        in :attr:`index` represents one unique observation and carries both
        ``color_path`` and ``red_path`` columns.

        Coordinate normalisation
        ~~~~~~~~~~~~~~~~~~~~~~~~
        HiRISE stores longitudes as east-positive in ``[0°, 360°]``.  These
        are normalised to ``[−180°, 180°]``.  Tiles that straddle the
        antimeridian (``lon_min > lon_max`` after normalisation) are skipped
        with a warning.
        """
        df = self._raw_index.copy()

        # Derive observation ID and product type from PRODUCT_ID.
        # PRODUCT_ID looks like "ESP_027409_1960_COLOR" or "ESP_027409_1960_RED".
        pid = df["PRODUCT_ID"].str.strip()
        df["_product_type"] = pid.str.extract(r"_(COLOR|RED)\s*$", expand=False)
        df["_obs_id"] = pid.str.replace(r"_(COLOR|RED)\s*$", "", regex=True)

        # Map FILE_NAME_SPECIFICATION to local paths.
        def _local(spec: str) -> str:
            return str(self.root / "images" / pathlib.Path(spec.strip()).name)

        df["_local_path"] = df["FILE_NAME_SPECIFICATION"].apply(_local)

        # Group by observation ID, collecting COLOR and RED paths per row.
        records: list[dict] = []
        for obs_id, grp in df.groupby("_obs_id", sort=False):
            ref = grp.iloc[0]

            color_rows = grp[grp["_product_type"] == "COLOR"]
            red_rows = grp[grp["_product_type"] == "RED"]

            color_path = (
                color_rows.iloc[0]["_local_path"] if not color_rows.empty else None
            )
            red_path = (
                red_rows.iloc[0]["_local_path"] if not red_rows.empty else None
            )

            # Normalise longitudes [0, 360] -> [-180, 180]
            lon_min = float(ref["MINIMUM_LONGITUDE"])
            lon_max = float(ref["MAXIMUM_LONGITUDE"])
            if lon_min > 180.0:
                lon_min -= 360.0
            if lon_max > 180.0:
                lon_max -= 360.0

            lat_min = float(ref["MINIMUM_LATITUDE"])
            lat_max = float(ref["MAXIMUM_LATITUDE"])

            if lon_min > lon_max:
                logger.warning(
                    "Observation %s straddles the antimeridian "
                    "(lon_min=%.4f > lon_max=%.4f after normalisation). Skipping.",
                    obs_id, lon_min, lon_max,
                )
                continue

            records.append({
                "obs_id": str(obs_id),
                "color_path": color_path,
                "red_path": red_path,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "t_start": ref["START_TIME"],
                "t_stop": ref["STOP_TIME"],
            })

        obs_df = pd.DataFrame(records)

        if obs_df.empty:
            raise RuntimeError(
                "Spatial index is empty after grouping observations. "
                "Check that the index files are present and the target "
                "filter is not too restrictive."
            )

        t_start = pd.to_datetime(obs_df["t_start"], utc=True, errors="coerce")
        t_stop = pd.to_datetime(obs_df["t_stop"], utc=True, errors="coerce")
        t_stop = t_stop.fillna(t_start)

        geometries = gpd.GeoSeries(
            [
                box(row.lon_min, row.lat_min, row.lon_max, row.lat_max)
                for row in obs_df.itertuples()
            ],
            crs=self.mars_crs,
        )

        time_intervals = pd.IntervalIndex.from_arrays(
            t_start, t_stop, closed="both", name="datetime"
        )

        self.index = gpd.GeoDataFrame(
            {
                "obs_id": obs_df["obs_id"].values,
                "color_path": obs_df["color_path"].values,
                "red_path": obs_df["red_path"].values,
            },
            index=time_intervals,
            geometry=geometries.values,
            crs=self.mars_crs,
        )

        logger.info(
            "Spatial index built: %d unique observations, "
            "lon [%.2f, %.2f]  lat [%.2f, %.2f].",
            len(self.index),
            obs_df["lon_min"].min(),
            obs_df["lon_max"].max(),
            obs_df["lat_min"].min(),
            obs_df["lat_max"].max(),
        )

    def _build_download_tasks(self):
        tasks: list[tuple[str, pathlib.Path]] = []

        for spec in self._raw_index["FILE_NAME_SPECIFICATION"]:
            spec = spec.strip()

            stem = pathlib.Path(spec[: -len(".JP2")])

            for suffix in [".JP2", ".LBL"]:
                # Download only the small metadata for now
                # for suffix in [".LBL", ]:
                url = f"{self.url}/{stem}{suffix}"
                local = self.root / "images" / pathlib.Path(stem).with_suffix(suffix).name
                tasks.append((url, local))

        return tasks

    def _download_images(self, subdir='images') -> None:
        task_list = self._build_download_tasks()

        if not task_list:
            return

        total_tasks = len(task_list)

        logger.info(f"Downloading {len(task_list)} images to {subdir}")

        # Need to finetune for the specific server
        active_processes = 8
        concurrency_per_process = 2

        chunk_size = max(1, (total_tasks + active_processes - 1) // active_processes)
        task_chunks = [
            task_list[i * chunk_size:(i + 1) * chunk_size]
            for i in range(active_processes)
        ]

        logger.info(f"Distributing payload across {active_processes} processes.")

        manager = multiprocessing.Manager()
        global_stop_event = manager.Event()

        with ProcessPoolExecutor(max_workers=active_processes) as executor:
            futures = [
                executor.submit(_worker_process, chunk, concurrency_per_process, global_stop_event)
                for chunk in task_chunks if chunk
            ]

            for future in futures:
                future.result()

        if global_stop_event.is_set():
            logger.warning("Download strictly terminated to prevent disk space.")

    def _load_tile(
            self,
            color_path: pathlib.Path | None,
            red_path: pathlib.Path | None,
            x: slice,
            y: slice,
    ) -> torch.Tensor | None:
        """Load and assemble requested channels from one observation.

        Channel routing
        ~~~~~~~~~~~~~~~
        Which product file(s) are opened depends on ``self.channels`` and
        on-disk availability:

        * **Only RED requested + RED.JP2 on disk** → open ``_RED.JP2``
          (band 1).  This path provides the highest-fidelity RED data because
          the standalone RED product uses more TDI accumulation lines.
        * **Any NIR or BG requested, OR RED requested alongside them** →
          open ``_COLOR.JP2`` and extract the required bands.  The RED band
          from ``_COLOR.JP2`` is used in this case to keep all channels
          spatially co-registered within a single file open.
        * **Fallback**: if the preferred file is missing, attempt the other.
          Channels that are genuinely unavailable (e.g. NIR when only a
          ``_RED.JP2`` exists) are logged as warnings and omitted.

        Radiometric calibration
        ~~~~~~~~~~~~~~~~~~~~~~~
        ``I/F = DN * SCALING_FACTOR + OFFSET`` using coefficients from
        the companion ``.LBL`` file (defaults used when absent).  Values
        are clipped to ``[0, 1]``.

        Reprojection
        ~~~~~~~~~~~~
        Each JP2 uses an Equirectangular projection with a per-observation
        ``CENTER_LATITUDE``.  ``rasterio.warp.reproject`` transforms every
        band into the common :attr:`mars_crs` geographic CRS so that tiles
        from different observations can be mosaicked without further
        alignment.

        Args:
            color_path: Local path to ``_COLOR.JP2``, or ``None``.
            red_path:   Local path to ``_RED.JP2``, or ``None``.
            x: Longitude slice ``(start, stop, step)`` in Mars decimal degrees.
            y: Latitude  slice ``(start, stop, step)`` in Mars decimal degrees.

        Returns:
            ``(C, H, W)`` float32 tensor in canonical channel order, or
            ``None`` if no usable data could be loaded.
        """
        requested = set(self.channels)

        color_available = color_path is not None and color_path.exists()
        red_available = red_path is not None and red_path.exists()

        # Prefer _RED.JP2 only when it is the exclusive request.
        only_red = requested == {"RED"}
        use_red_file_for_red = only_red and red_available

        # Channels to load from each product.
        from_color: set[str] = set() if use_red_file_for_red else requested.copy()
        from_red: set[str] = {"RED"} if use_red_file_for_red else set()

        # Fallback: redirect to whichever file is available.
        if from_color and not color_available:
            salvageable = from_color & set(_RED_CHANNELS)  # only "RED" can come from RED file
            lost = from_color - salvageable
            if lost:
                logger.warning(
                    "COLOR file unavailable (%s); cannot provide channels %s.",
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

        # Load from _COLOR.JP2
        if from_color and color_available:
            lbl = color_path.with_suffix(".LBL")
            meta = _ProductMeta.from_lbl(lbl)

            # Rebuild band map from LBL if the filter list differs from the default.
            if meta.filter_names and len(meta.filter_names) == meta.bands:
                color_band_map = {
                    name: idx + 1
                    for idx, name in enumerate(meta.filter_names)
                }
            else:
                color_band_map = _COLOR_BAND.copy()

            band_map = {
                ch: color_band_map[ch]
                for ch in from_color
                if ch in color_band_map
            }
            band_arrays.update(
                self._load_from_jp2(color_path, band_map, meta, x, y)
            )

        # Load from _RED.JP2
        if from_red and red_available:
            lbl = red_path.with_suffix(".LBL")
            meta = _ProductMeta.from_lbl(lbl)
            meta.filter_names = ["RED"]
            meta.bands = 1

            band_map = {ch: _RED_BAND[ch] for ch in from_red if ch in _RED_BAND}
            band_arrays.update(
                self._load_from_jp2(red_path, band_map, meta, x, y)
            )

        if not band_arrays:
            return None

        # Stack in canonical channel order.
        tensors = [
            torch.from_numpy(band_arrays[ch])
            for ch in self.channels
            if ch in band_arrays
        ]

        return torch.stack(tensors, dim=0) if tensors else None  # (C, H, W)

    def _load_from_jp2(
            self,
            jp2_path: pathlib.Path,
            band_map: dict[str, int],
            meta: _ProductMeta,
            x: slice,
            y: slice,
    ) -> dict[str, np.ndarray]:
        """Open a JP2 and return calibrated, reprojected 2-D arrays.

        Args:
            jp2_path: Path to the ``.JP2`` file.
            band_map: ``{channel_name: 1-based_rasterio_band_index}``.
            meta: Radiometric metadata for this product.
            x: Longitude query slice in Mars decimal degrees.
            y: Latitude  query slice in Mars decimal degrees.

        Returns:
            ``{channel_name: (H, W) float32 ndarray}`` for every channel in
            *band_map* that was successfully read.
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
                for ch_name, band_idx in band_map.items():
                    dest = np.zeros((out_h, out_w), dtype=np.float32)

                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=dest,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear,
                    )

                    # DN -> I/F reflectance
                    dest = dest * meta.scaling_factor + meta.offset

                    # Clip to valid reflectance range
                    dest = np.clip(dest, 0.0, 1.0)

                    result[ch_name] = dest

        except rasterio.errors.RasterioIOError as exc:
            logger.warning("Could not open %s: %s", jp2_path, exc)

        return result

    @staticmethod
    def _merge_tiles(tiles: list[torch.Tensor]) -> torch.Tensor:
        """Merge a list of co-registered tiles into one tensor.

        Uses a simple "first non-zero wins" strategy: tiles are stacked and
        the first finite, non-zero value in depth order is selected for each
        pixel.  This is fast and avoids blending artifacts at tile boundaries.

        Args:
            tiles: List of ``(C, H, W)`` float32 tensors, all the same shape.

        Returns:
            Merged ``(C, H, W)`` float32 tensor.
        """
        if len(tiles) == 1:
            return tiles[0]

        # Broadcast tiles to the same spatial shape (take the maximum).
        max_h = max(t.shape[1] for t in tiles)
        max_w = max(t.shape[2] for t in tiles)
        n_ch = tiles[0].shape[0]

        merged = torch.zeros((n_ch, max_h, max_w), dtype=torch.float32)

        for tile in tiles:
            h, w = tile.shape[1], tile.shape[2]
            empty = merged[:, :h, :w] == 0.0
            merged[:, :h, :w][empty] = tile[empty]


        return merged

    def __getitem__(self, index: GeoSlice) -> Sample:
        """Return an image patch for the given spatiotemporal slice.

        Spatial coordinates are in **Mars decimal degrees** (longitude,
        latitude) in the IAU 2000 geographic CRS.

        Args:
            index: ``[xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres]``

        Returns:
            A :class:`~torchgeo.datasets.utils.Sample` dict:

            ``"image"``
                ``torch.Tensor`` of shape ``(C, H, W)``, dtype ``float32``,
                values in ``[0, 1]``.  ``C == len(self.channels)``; band
                order follows :attr:`all_channels`.

            ``"bounds"``
                6-element ``float64`` tensor
                ``[xmin, xmax, xres, ymin, ymax, yres]``.

            ``"crs"``
                WKT string of the Mars IAU 2000 CRS.

        Raises:
            IndexError: If no observations overlap *index* or all matching
                JP2 files are absent from disk.
        """
        x, y, t = self._disambiguate_slice(index)

        query_geom = box(x.start, y.start, x.stop, y.stop)
        interval = pd.Interval(t.start, t.stop)

        time_mask = self.index.index.overlaps(interval)
        candidates: gpd.GeoDataFrame = self.index.iloc[time_mask]
        candidates = candidates[candidates.geometry.intersects(query_geom)]

        if candidates.empty:
            raise IndexError(
                f"No MarsHiRISE observations found for slice {index}."
            )

        tiles: list[torch.Tensor] = []
        for _, row in candidates.iterrows():
            tile = self._load_tile(
                color_path=(
                    pathlib.Path(row["color_path"])
                    if row["color_path"] is not None else None
                ),
                red_path=(
                    pathlib.Path(row["red_path"])
                    if row["red_path"] is not None else None
                ),
                x=x,
                y=y,
            )
            if tile is not None:
                tiles.append(tile)

        if not tiles:
            raise IndexError(
                f"All candidate JP2 files for slice {index} are missing from disk."
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

    def plot(
            self,
            sample: Sample,
            show_titles: bool = True,
            suptitle: str | None = None,
    ) -> Figure:
        """Visualise an image patch returned by :meth:`__getitem__`.

        Rendering strategy:

        * All three channels present → false-colour composite
          (NIR→R, RED→G, BG→B).
        * Single channel → greyscale.
        * Two channels → first channel in greyscale.

        Args:
            sample: A sample returned by :meth:`__getitem__`.
            show_titles: Whether to display a title above the image.
            suptitle: Optional figure-level title.

        Returns:
            A :class:`~matplotlib.figure.Figure` with the rendered patch.
        """
        image: torch.Tensor = sample["image"]  # (C, H, W) float32 in [0, 1]
        ch = self.channels

        if set(ch) >= {"NEAR-INFRARED", "RED", "BLUE-GREEN"}:
            nir_i = ch.index("NEAR-INFRARED")
            red_i = ch.index("RED")
            bg_i = ch.index("BLUE-GREEN")
            img_np = image[[nir_i, red_i, bg_i]].permute(1, 2, 0).numpy()
            cmap = None
            title = "MarsHiRISE — false colour (NIR→R, RED→G, BG→B)"
        elif image.shape[0] >= 3:
            img_np = image[:3].permute(1, 2, 0).numpy()
            cmap = None
            title = f"MarsHiRISE — channels {ch[:3]}"
        else:
            img_np = image[0].numpy()
            cmap = "grey"
            title = f"MarsHiRISE — {ch[0]}"

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_np, cmap=cmap, interpolation="nearest")
        ax.axis("off")

        if show_titles:
            ax.set_title(title)
        if suptitle is not None:
            fig.suptitle(suptitle)

        fig.tight_layout()
        return fig


def setup_logging(config_path: str = CONFIG) -> None:
    with open(config_path, 'r') as f:
        logging.config.dictConfig(json.load(f))


def main():
    setup_logging()

    # # import urllib.request
    # # urllib.request.urlretrieve("https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX.LBL", "RDRCUMINDEX.LBL")
    # # urllib.request.urlretrieve("https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX.TAB", "RDRCUMINDEX.TAB")
    #
    # data = pdr.read("RDRCUMINDEX.LBL")
    # data.load('all')
    #
    # print(data)
    #
    # get_data_column_index = "FILE_NAME_SPECIFICATION"
    # index_table: pd.DataFrame = data['RDR_INDEX_TABLE']
    #
    # file_names = index_table["FILE_NAME_SPECIFICATION"]
    #
    # full_path = MarsHiRISE.url + file_names
    #
    # print(full_path)
    #
    # val = index_table['RATIONALE_DESC'].unique()
    # print(val)

    keyword = 'Olympus'

    dataset = MarsHiRISE(target=keyword)

    sampler = RandomGeoSampler(dataset, size=100, length=200)
    dataloader = DataLoader(dataset, sampler=sampler)

    val = iter(dataloader)

    for i, sample in dataloader:

        fig = dataset.plot(sample)
        fig.savefig(f"output{i}.png")

        plt.close(fig)


if __name__ == '__main__':
    main()
