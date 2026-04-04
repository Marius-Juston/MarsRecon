# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""MarsHiRISE dataset."""

from __future__ import annotations

import json
import logging
import pathlib
from collections.abc import Callable
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from shapely.geometry import box
from torchgeo.datasets.errors import DatasetNotFoundError
from torchgeo.datasets.utils import GeoSlice, Path, Sample
from torchgeo.samplers import Units
from torchvision.transforms import Normalize

from dataset.mars_hirise_base import (
    MarsHiRISEBase,
    ProductMeta,
    check_overlap,
    reproject_band, setup_logging,
)

logger = logging.getLogger(__name__)

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

Channel = Literal["NEAR-INFRARED", "RED", "BLUE-GREEN"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MarsHiRISE(MarsHiRISEBase):
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
            images/
                PSP_001430_1780_COLOR.JP2
                PSP_001430_1780_COLOR.LBL
                PSP_001430_1780_RED.JP2
                PSP_001430_1780_RED.LBL

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

    _INDEX_STEM: str = "RDRCUMINDEX"

    all_channels: tuple[str, ...] = ALL_CHANNELS

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
            reuse_cache: bool = True,
            normalize: bool = False,
            normalization_path: str | None = None,
    ) -> None:
        """Initialise the dataset.

        Args:
            root: Root directory.  Must contain the PDS index files and the
                observation JP2/LBL files, or ``download=True`` must be set.
            split: Dataset split — informational.
            target: Optional case-insensitive substring filter on all character
                columns of the cumulative index (e.g. ``"Olympus"``).
            channels: Which channels to include.  Valid values:
                ``"NEAR-INFRARED"``, ``"RED"``, ``"BLUE-GREEN"``.  Output
                tensor band order always follows :attr:`all_channels`.
                Defaults to all three.
            transforms: Optional callable applied to each :class:`Sample`.
            download: Fetch index and images from the PDS server if absent.
            bbox: ``(lon_min, lat_min, lon_max, lat_max)`` bounding box filter
                in degrees ([-180, 180] longitude convention).
            checksum: Verify checksums after download (not yet implemented).
            reuse_cache: Reuse cached spatial index if present.
            normalize: If ``True``, apply per-channel z-score normalisation
                using statistics from *normalization_path*.
            normalization_path: Path to a JSON file with ``"channels"``,
                ``"mean"``, and ``"std"`` keys.

        Raises:
            ValueError: If *channels* contains an unrecognised name.
            DatasetNotFoundError: If index files are absent and
                ``download=False``.
        """
        # ── Channel validation (before super().__init__ triggers _verify) ──
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

        # ── Normalisation ──
        self.normalize = normalize
        self.normalization_path = normalization_path
        self._normalizer = None

        if self.normalize:
            path = pathlib.Path(self.normalization_path or "")
            if not self.normalization_path or not path.exists():
                raise ValueError(
                    f"normalization_path must be a valid path to "
                    f"dataset_stats.json when normalize=True, currently "
                    f"pointing to {path.absolute()}"
                )
            with open(self.normalization_path) as f:
                stats = json.load(f)

            stat_channels = stats["channels"]
            mean_dict = dict(zip(stat_channels, stats["mean"]))
            std_dict = dict(zip(stat_channels, stats["std"]))
            try:
                mean = [mean_dict[ch] for ch in self.channels]
                std = [std_dict[ch] for ch in self.channels]
            except KeyError as e:
                raise ValueError(
                    f"Channel {e} not found in normalization stats."
                )

            self._normalizer = Normalize(mean=mean, std=std)

        # Native HiRISE RDR resolution: 1 / 118 502.26 pix/deg ≈ 8.44e-6 deg/pix
        super().__init__(
            root,
            split=split,
            target=target,
            transforms=transforms,
            download=download,
            bbox=bbox,
            checksum=checksum,
            reuse_cache=reuse_cache,
            res=1.0 / 118_502.26464032,
        )

    # ------------------------------------------------------------------
    # Cache version (matches original v3 for backward compat)
    # ------------------------------------------------------------------

    def _cache_version(self) -> str:
        return "v3"

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, index: GeoSlice) -> Sample:
        """Return an image patch for the given spatiotemporal slice.

        Args:
            index: ``[xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres]``
                in Mars geographic degrees / UTC datetimes.

        Returns:
            Sample dict with ``"image"`` ``(C, H, W)`` float32 in ``[0,1]``,
            ``"bounds"`` tensor, and ``"crs"`` WKT string.

        Raises:
            IndexError: No observations overlap *index*, or all matching
                JP2s are absent.
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
                f"{n} observation(s) matched slice {index} spatially/"
                f"temporally, but no image data could be loaded. Possible "
                f"causes:\n"
                f"  1. JP2 files are absent under '{self.root}' — run with "
                f"download=True.\n"
                f"  2. The JP2 files exist but their reprojected bounds "
                f"don't overlap the query (check DEBUG logs).\n"
                f"  3. RasterioIOError on open — check WARNING logs."
            )

        image = self.merge_tiles(tiles)

        # Apply normalization but preserve the 0.0 nodata pixels
        if self.normalize and self._normalizer is not None:
            nodata_mask = image == 0.0
            image = self._normalizer(image)
            image[nodata_mask] = 0.0

        sample: Sample = {
            "image": image,
            "bounds": self._slice_to_tensor(index),
            "crs": self.crs.to_wkt(),
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    # ------------------------------------------------------------------
    # plot
    # ------------------------------------------------------------------

    def plot(
            self,
            sample: Sample,
            show_titles: bool = True,
            suptitle: str | None = None,
            eps: float = 1e-8,
            **kwargs,
    ) -> Figure:
        """Visualise an RDR image sample with percentile stretch."""
        eps = abs(eps)

        image: torch.Tensor = sample["image"]
        if image.ndim == 4:
            image = image[0]

        ch = self.channels
        if set(ch) >= {"NEAR-INFRARED", "RED", "BLUE-GREEN"}:
            idx = [
                ch.index("NEAR-INFRARED"),
                ch.index("RED"),
                ch.index("BLUE-GREEN"),
            ]
            rgb = image[idx]
            # If only one channel has non-zero data (e.g. COLOR file
            # absent), fall back to grayscale.
            nonzero = [(rgb[i] > 0).any().item() for i in range(3)]
            if sum(nonzero) == 1:
                active_i = nonzero.index(True)
                active_name = ["NEAR-INFRARED", "RED", "BLUE-GREEN"][active_i]
                img_np = rgb[active_i].numpy()
                cmap = "grey"
                title = (
                    f"MarsHiRISE — {active_name} (COLOR file unavailable)"
                )
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
                mask = np.logical_or(band > eps, band < -eps)
                data_pixels = band[mask]
                if len(data_pixels) > 0:
                    p2, p98 = np.percentile(data_pixels, [2, 98])
                    if p98 > p2:
                        img_out[..., c] = np.clip(
                            (band - p2) / (p98 - p2), 0, 1
                        )
                        img_out[..., c][np.logical_not(mask)] = 0
        else:
            mask = np.logical_or(img_out > eps, img_out < -eps)
            data_pixels = img_out[mask]
            if len(data_pixels) > 0:
                p2, p98 = np.percentile(data_pixels, [2, 98])
                if p98 > p2:
                    img_out = np.clip(
                        (img_out - p2) / (p98 - p2), 0, 1
                    )
                    img_out[np.logical_not(mask)] = 0

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
    # Post-download verification
    # ------------------------------------------------------------------

    def _post_download_verify(self) -> None:
        """Check that at least some JP2 files exist on disk."""
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
                    "observations. Either run with download=True to fetch "
                    "images or point 'root' at an existing PDS mirror.",
                    self.root, sample_size,
                )
            else:
                logger.warning(
                    "Download completed but no JP2 files found on disk. "
                    "Check network connectivity and available disk space."
                )

    # ------------------------------------------------------------------
    # Spatial index
    # ------------------------------------------------------------------

    def _build_spatial_index(self, force_rebuild: bool = False) -> None:
        """Build a per-observation GeoDataFrame from the cumulative index.

        Groups the one-row-per-JP2 table by observation ID so each row in
        :attr:`index` carries both ``color_path`` and ``red_path`` (either
        may be ``None``).

        Geometries are strip polygons from the four corner coordinates
        (CORNER1-4); falls back to JP2-derived bounds, then to index
        min/max bounding box.
        """
        if not force_rebuild and self._try_load_cache():
            return

        # ── Phase 1: group products by observation ───────────────────
        df = self._raw_index.copy()
        pid = df["PRODUCT_ID"].str.strip()
        df["_product_type"] = pid.str.extract(
            r"_(COLOR|RED)\s*$", expand=False
        )
        df["_obs_id"] = pid.str.replace(
            r"_(COLOR|RED)\s*$", "", regex=True
        )
        df["_local_path"] = df["FILE_NAME_SPECIFICATION"].apply(
            lambda s: str(self._pds_local_path(s))
        )

        obs_list: list[
            tuple[str, str | None, str | None, pd.Series]
        ] = []
        fp_paths: list[str | None] = []

        for obs_id, grp in df.groupby("_obs_id", sort=False):
            ref = grp.iloc[0]
            color_rows = grp[grp["_product_type"] == "COLOR"]
            red_rows = grp[grp["_product_type"] == "RED"]
            cp = (
                color_rows.iloc[0]["_local_path"]
                if not color_rows.empty
                else None
            )
            rp = (
                red_rows.iloc[0]["_local_path"]
                if not red_rows.empty
                else None
            )

            # Pick the first file that exists on disk for footprint.
            fp_path = None
            for p in (cp, rp):
                if p is not None:
                    pp = pathlib.Path(p)
                    if pp.exists() or pp.with_suffix(".tif").exists():
                        fp_path = p
                        break

            obs_list.append((str(obs_id), cp, rp, ref))
            fp_paths.append(fp_path)

        # ── Phase 2: parallel footprint extraction ───────────────────
        fp_results = self._run_footprint_extraction(fp_paths)

        # ── Phase 3: assemble records ────────────────────────────────
        records: list[dict] = []

        for i, (obs_id, cp, rp, ref) in enumerate(obs_list):
            hull_coords, file_bounds = fp_results[i]

            geom = self._geometry_from_footprint_result(
                hull_coords, file_bounds, ref
            )
            if geom is None:
                logger.warning(
                    "Observation %s straddles antimeridian. Skipping.",
                    obs_id,
                )
                continue

            records.append({
                "obs_id": obs_id,
                "color_path": cp,
                "red_path": rp,
                "geometry": geom,
                "t_start": ref["START_TIME"],
                "t_stop": ref["STOP_TIME"],
            })

        # ── Phase 4: build GeoDataFrame ──────────────────────────────
        obs_df = pd.DataFrame(records)
        if obs_df.empty:
            raise DatasetNotFoundError(self)

        t_start = pd.to_datetime(
            obs_df["t_start"], utc=True, errors="coerce"
        )
        t_stop = pd.to_datetime(
            obs_df["t_stop"], utc=True, errors="coerce"
        )
        t_stop = t_stop.fillna(t_start)

        geometries = gpd.GeoSeries(
            obs_df["geometry"].tolist(), crs=self.mars_crs
        )

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
        self._log_index_extent()

    # ------------------------------------------------------------------
    # Download tasks
    # ------------------------------------------------------------------

    def _build_download_tasks(self) -> list[tuple[str, pathlib.Path]]:
        """Return ``(remote_url, local_path)`` pairs for JP2 and LBL files."""
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

    # ------------------------------------------------------------------
    # Tile loading
    # ------------------------------------------------------------------

    def _load_tile(
            self,
            color_path: pathlib.Path | None,
            red_path: pathlib.Path | None,
            x: slice,
            y: slice,
    ) -> torch.Tensor | None:
        """Load and assemble the requested channels from one observation."""
        color_path = self.prefer_cog(color_path)
        red_path = self.prefer_cog(red_path)

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
            meta = ProductMeta.from_lbl(lbl)
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

        if from_red and red_available:
            lbl = red_path.with_suffix(".LBL")
            meta = ProductMeta.from_lbl(lbl)
            meta.filter_names = ["RED"]
            meta.bands = 1
            band_map = {
                ch: _RED_BAND[ch] for ch in from_red if ch in _RED_BAND
            }
            band_arrays.update(
                self._load_from_jp2(red_path, band_map, meta, x, y)
            )

        if not band_arrays:
            return None

        first = next(iter(band_arrays.values()))
        out_h, out_w = first.shape

        # Build channel tensors in self.channels order.
        # Missing channels are filled with zeros.
        tensors = []
        for ch in self.channels:
            if ch in band_arrays:
                tensors.append(torch.from_numpy(band_arrays[ch]))
            else:
                tensors.append(
                    torch.zeros(out_h, out_w, dtype=torch.float32)
                )
        return torch.stack(tensors)

    def _load_from_jp2(
            self,
            jp2_path: pathlib.Path,
            band_map: dict[str, int],
            meta: ProductMeta,
            x: slice,
            y: slice,
    ) -> dict[str, np.ndarray]:
        """Reproject bands from a JP2 into geographic degrees and calibrate.

        Each HiRISE JP2 has its own Equirectangular projection with a unique
        CENTER_LATITUDE.  rasterio reprojects into the geographic Mars CRS.
        """
        result: dict[str, np.ndarray] = {}
        if not band_map:
            return result

        out_w = max(1, int(round(
            (x.stop - x.start) / (x.step or self.res)
        )))
        out_h = max(1, int(round(
            (y.stop - y.start) / (y.step or self.res)
        )))

        dst_transform = rasterio.transform.from_bounds(
            x.start, y.start, x.stop, y.stop, out_w, out_h
        )
        dst_crs = rasterio.crs.CRS.from_user_input(self.mars_crs)

        try:
            with rasterio.open(jp2_path) as src:
                if not check_overlap(src, dst_crs, x, y):
                    logger.debug(
                        "Query doesn't overlap file bounds: %s",
                        jp2_path.name,
                    )
                    return result

                for ch_name, band_idx in band_map.items():
                    try:
                        dest = reproject_band(
                            src, band_idx, dst_crs, dst_transform,
                            out_h, out_w, dst_nodata=0.0,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Reprojection failed for %s band %d: %s",
                            jp2_path.name, band_idx, exc,
                        )
                        continue

                    # Radiometric calibration: I/F = DN * scaling + offset
                    nodata_mask = dest == 0.0
                    dest *= meta.scaling_factor
                    dest += meta.offset
                    np.clip(dest, 0.0, 1.0, out=dest)
                    dest[nodata_mask] = 0.0

                    result[ch_name] = dest

        except rasterio.errors.RasterioIOError as exc:
            logger.warning("Could not open %s: %s", jp2_path, exc)

        return result


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv=None) -> None:  # pragma: no cover
    import sys
    _SRC = pathlib.Path(__file__).parent.parent / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    setup_logging()

    from torch.utils.data import DataLoader

    from dataset.hirise_sampler import HiRISEGeoSampler
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a sample test on the main HiRISE dataset"
    )
    parser.add_argument(
        "-ol", "--olympus", action=argparse.BooleanOptionalAction,
        help="Whether to use the 'Olympus' target",
        default=False
    )
    parser.add_argument(
        "-l", "--length", type=int, default=-1,
        help="Number of samples to have",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42,
        help="seed",
    )

    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    normalization_path = "dataset_stats/dataset_stats.json"

    if args.olympus:
        dataset = MarsHiRISE(
            target="Olympus",
            channels=["NEAR-INFRARED", "RED", "BLUE-GREEN"],
            download=True,
            reuse_cache=True,
            normalize=True,
            normalization_path=normalization_path,
        )
    else:
        dataset = MarsHiRISE(
            bbox=(-136, 12, -124, 24),  # Olympus Mons extent from the CTX metadata
            channels=["NEAR-INFRARED", "RED", "BLUE-GREEN"],
            download=True,
            reuse_cache=True,
            normalize=True,
            normalization_path=normalization_path,
        )

    output_path = pathlib.Path("Figures")
    output_path.mkdir(parents=True, exist_ok=True)

    fig = dataset.plot_coverage()
    fig.savefig(output_path / "coverage.png", bbox_inches='tight')
    logger.info("saved fig")

    sampler = HiRISEGeoSampler(
        dataset, size=0.005,
        length=None if args.length <= 0 else args.length,
        units=Units.CRS,
    )

    logger.info("Number of samples: %d", len(sampler))
    dataloader = DataLoader(
        dataset, sampler=sampler,
        num_workers=10, multiprocessing_context="spawn", prefetch_factor=4,
    )
    logger.info("Number of data-loader: %d", len(dataloader))

    for i, sample in enumerate(dataloader):
        output_path.mkdir(parents=True, exist_ok=True)
        fig = dataset.plot(sample)
        fig.savefig(output_path / f"output{i}.png")
        logger.info("Saved fig output%d.png", i)
        plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    main()
