# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""MarsHiRISE DTM (Digital Terrain Model) dataset."""

from __future__ import annotations

import json
import logging
import pathlib
import re
from collections.abc import Callable
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely.geometry
import shapely.ops
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import box
from torchgeo.datasets.errors import DatasetNotFoundError
from torchgeo.datasets.utils import GeoSlice, Path, Sample
from torchgeo.samplers import Units

from dataset.mars_hirise_base import (
    MarsHiRISEBase,
    ProductMeta,
    check_overlap,
    reproject_band, setup_logging,
)

logger = logging.getLogger(__name__)

DEFAULT_HIRISE_DTM_VIZ_OUT = pathlib.Path("/scratch/marsrecon_runs/dataset_viz/hirise_dtm")

# ---------------------------------------------------------------------------
# DTM-specific constants
# ---------------------------------------------------------------------------

# DATA_TYPE column values in DTMCUMINDEX.TAB (trimmed / upper-cased):
DTM_DATA_TYPES = frozenset({"DTM"})
ORTHO_DATA_TYPES = frozenset({"ORTHOIMAGE"})

# Orthoimage colour-content tags and their in-file band mappings:
IRB_CHANNELS: tuple[str, ...] = ("NEAR-INFRARED", "RED", "BLUE-GREEN")
RED_CHANNELS: tuple[str, ...] = ("RED",)

_IRB_BAND: dict[str, int] = {"NEAR-INFRARED": 1, "RED": 2, "BLUE-GREEN": 3}
_RED_BAND: dict[str, int] = {"RED": 1}

# HiRISE DTMs use IEEE float32 minimum as nodata.
_DTM_NODATA: float = -3.4028226550889045e+38
_EPS = 1e-6

# DTM naming convention (from https://www.uahirise.org/dtm/about.php):
#   PRODUCT_ID = aabcd_xxxxxx_xxxx_yyyyyy_yyyy_Vnn
#
# Orthoimage naming convention:
#   PRODUCT_ID = XSP_xxxxxx_xxxx_CCC_S_NN_ORTHO
_ORTHO_PATTERN = re.compile(
    r"(\w+_\d+_\d+)_(RED|IRB)_([A-Z])_(\d+)_ORTHO\s*$"
)

OrthoType = Literal["RED", "IRB"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MarsHiRISEDTM(MarsHiRISEBase):
    """Mars HiRISE Digital Terrain Model (DTM) dataset.

    `HiRISE DTMs <https://www.uahirise.org/dtm/about.php>`__ are derived
    from stereo pairs of HiRISE observations.  Each DTM set consists of:

    * A **DTM** (``.IMG``) — 32-bit floating-point raster where each pixel
      value is an areoid elevation (metres) or planetary radius (metres).
      ``1 DN = 1 m``.
    * **Orthoimages** (``.JP2``) — the left and right stereo observations
      orthorectified onto the DTM, in RED (1-band) and/or IRB (3-band:
      near-IR, RED, blue-green) colour content, at one or more resolutions
      (A = 0.25 m, B = 0.5 m, C = 1.0 m, D = 2.0 m).

    This class indexes the PDS DTM cumulative index
    (``DTMCUMINDEX.TAB``) and groups all products by **stereo pair**.

    Directory layout
    ~~~~~~~~~~~~~~~~
    ::

        <root>/
            DTMCUMINDEX.LBL
            DTMCUMINDEX.TAB
            images/
                DTEEC_011265_1560_011331_1560_U01.IMG
                ESP_011265_1560_RED_A_01_ORTHO.JP2
                ESP_011265_1560_RED_A_01_ORTHO.LBL
                ...

    Sample dict
    ~~~~~~~~~~~
    ``__getitem__`` returns a dict containing:

    * ``"elevation"`` — ``(1, H, W)`` float32 in metres.  Nodata = ``NaN``.
    * ``"left_red"``  — ``(1, H, W)`` float32 I/F [0, 1] (if requested)
    * ``"right_red"`` — same, for the right observation
    * ``"left_irb"``  — ``(3, H, W)`` float32 I/F [0, 1] (if requested)
    * ``"right_irb"`` — same, for the right observation
    * ``"bounds"`` tensor, ``"crs"`` WKT string

    Sampler units
    ~~~~~~~~~~~~~
    ``self.crs`` is geographic; ``self.res`` is in degrees/pixel
    (~1.69e-5 deg/px ≈ 1 m at the equator by default).

    Dataset homepage:
        https://www.uahirise.org/dtm/about.php

    .. versionadded:: 0.8
    """

    _INDEX_STEM: str = "DTMCUMINDEX"

    def __init__(
            self,
            root: Path = "/scratch/mars_hirise_dtm",
            *,
            split: str = "train",
            target: str | None = None,
            include_ortho: bool = True,
            ortho_type: OrthoType | list[OrthoType] = "RED",
            ortho_scale: str | None = None,
            transforms: Callable[[Sample], Sample] | None = None,
            download: bool = False,
            bbox: tuple[float, float, float, float] | None = None,
            checksum: bool = False,
            reuse_cache: bool = True,
            normalize_elevation: bool = False,
            elevation_stats_path: str | None = None,
    ) -> None:
        """Initialise the dataset.

        Args:
            root: Root directory containing PDS index files and data.
            split: Dataset split — informational.
            target: Case-insensitive substring filter on character columns
                of the cumulative index (e.g. ``"Eberswalde"``).
            include_ortho: If ``True``, load orthoimage patches alongside
                elevation.
            ortho_type: Which orthoimage colour content to load:
                ``"RED"`` (1-band) and/or ``"IRB"`` (3-band).
            ortho_scale: Preferred scale letter (``"A"``–``"D"``).  If
                ``None``, the finest resolution available is used.
            transforms: Optional transform applied to each sample.
            download: Fetch from PDS if absent.
            bbox: ``(lon_min, lat_min, lon_max, lat_max)`` in degrees.
            checksum: Verify checksums (not yet implemented).
            reuse_cache: Reuse cached spatial index if present.
            normalize_elevation: Z-score normalise elevation using stats.
            elevation_stats_path: JSON file with ``"mean"`` and ``"std"``.

        Raises:
            ValueError: If ortho_type contains an unrecognised value.
            DatasetNotFoundError: If index is absent and ``download=False``.
        """
        # ── Ortho configuration (before super().__init__ calls _verify) ──
        self.include_ortho = include_ortho
        if isinstance(ortho_type, str):
            ortho_type = [ortho_type]
        invalid = set(ortho_type) - {"RED", "IRB"}
        if invalid:
            raise ValueError(
                f"Invalid ortho_type(s): {invalid}. Valid: 'RED', 'IRB'"
            )
        self.ortho_types: list[str] = list(ortho_type)
        self.ortho_scale = ortho_scale.upper() if ortho_scale else None

        # ── Elevation normalisation ──
        self.normalize_elevation = normalize_elevation
        self._elev_mean: float | None = None
        self._elev_std: float | None = None

        if self.normalize_elevation:
            path = pathlib.Path(elevation_stats_path or "")
            if not elevation_stats_path or not path.exists():
                raise ValueError(
                    f"elevation_stats_path must point to a valid stats JSON "
                    f"when normalize_elevation=True (got: {path.absolute()})"
                )
            with open(path) as f:
                stats = json.load(f)
            self._elev_mean = float(stats["mean"])
            self._elev_std = float(stats["std"])

        # DTM typical resolution ≈ 1 m/pix → ~1/(59 251) deg/pix at equator.
        super().__init__(
            root,
            split=split,
            target=target,
            transforms=transforms,
            download=download,
            bbox=bbox,
            checksum=checksum,
            reuse_cache=reuse_cache,
            res=1.0 / 59_251.13,
        )

    # ------------------------------------------------------------------
    # Cache key customisation
    # ------------------------------------------------------------------

    def _cache_suffix_parts(self) -> list[str]:
        parts: list[str] = []
        if self.include_ortho:
            parts.append("_".join(sorted(self.ortho_types)))
            if self.ortho_scale:
                parts.append(f"s{self.ortho_scale}")
        return parts

    def _cache_version(self) -> str:
        return "dtm_v1"

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, index: GeoSlice) -> Sample:
        """Return an elevation (and optionally orthoimage) patch.

        Returns:
            Sample dict — see class docstring for keys.

        Raises:
            IndexError: No stereo pairs overlap *index* or no data loaded.
        """
        x, y, t = self._disambiguate_slice(index)

        query_geom = box(x.start, y.start, x.stop, y.stop)
        candidates = self.index[self.index.geometry.intersects(query_geom)]

        if candidates.empty:
            raise IndexError(
                f"No MarsHiRISEDTM stereo pairs found for slice {index}."
            )

        elevation_tiles: list[torch.Tensor] = []
        ortho_tiles: dict[str, list[torch.Tensor]] = {
            f"{side}_{color.lower()}": []
            for side in ("left", "right")
            for color in self.ortho_types
        }

        for _, row in candidates.iterrows():
            # --- elevation ---
            dtm_path = row.get("dtm_path")
            if isinstance(dtm_path, str):
                tile = self._load_dtm_tile(pathlib.Path(dtm_path), x, y)
                if tile is not None:
                    elevation_tiles.append(tile)

            # --- ortho images ---
            if self.include_ortho:
                for side in ("left", "right"):
                    for otype in self.ortho_types:
                        col = f"{side}_{otype.lower()}_path"
                        p = row.get(col)
                        if not isinstance(p, str):
                            continue
                        tile = self._load_ortho_tile(
                            pathlib.Path(p), otype, x, y
                        )
                        if tile is not None:
                            ortho_tiles[f"{side}_{otype.lower()}"].append(tile)

        if not elevation_tiles:
            n = len(candidates)
            raise IndexError(
                f"{n} stereo pair(s) matched slice {index} spatially, but "
                f"no elevation data could be loaded.  Run with download=True "
                f"or check WARNING logs."
            )

        sample: Sample = {
            "elevation": self._merge_elevation_tiles(elevation_tiles),
            "bounds": self._slice_to_tensor(index),
            "crs": self.crs.to_wkt(),
        }

        if self.normalize_elevation and self._elev_mean is not None:
            elev = sample["elevation"]
            nodata_mask = torch.isnan(elev)
            elev = (elev - self._elev_mean) / self._elev_std
            elev[nodata_mask] = float("nan")
            sample["elevation"] = elev

        for key, tiles in ortho_tiles.items():
            if tiles:
                sample[key] = self.merge_tiles(tiles)

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
            **kwargs,
    ) -> Figure:
        """Visualise a sample: elevation + ortho panels."""
        has_elev = "elevation" in sample
        ortho_keys = [
            k for k in ("left_red", "right_red", "left_irb", "right_irb")
            if k in sample
        ]
        n_panels = max(int(has_elev) + len(ortho_keys), 1)

        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
        if n_panels == 1:
            axes = [axes]

        panel = 0

        # ── Elevation ─────────────────────────────────────────────────
        if has_elev:
            ax = axes[panel]
            elev = sample["elevation"]
            if elev.ndim == 4:
                elev = elev[0]
            elev_np = elev[0].numpy().copy()
            valid = np.isfinite(elev_np)
            if valid.any():
                vmin, vmax = np.nanpercentile(elev_np[valid], [2, 98])
                im = ax.imshow(
                    np.where(valid, elev_np, np.nan),
                    cmap="terrain", vmin=vmin, vmax=vmax,
                    interpolation="nearest",
                )
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             label="Elevation (m)")
            else:
                ax.imshow(elev_np, cmap="terrain", interpolation="nearest")
            ax.axis("off")
            if show_titles:
                ax.set_title("Elevation (DTM)")
            panel += 1

        # ── Ortho panels ─────────────────────────────────────────────
        for key in ortho_keys:
            ax = axes[panel]
            img = sample[key]
            if img.ndim == 4:
                img = img[0]

            if img.shape[0] >= 3:
                img_np = img[:3].permute(1, 2, 0).numpy().copy()
                cmap = None
            else:
                img_np = img[0].numpy().copy()
                cmap = "grey"

            ax.imshow(
                self._percentile_stretch(img_np),
                cmap=cmap, interpolation="nearest",
            )
            ax.axis("off")
            if show_titles:
                ax.set_title(f"Ortho — {key.replace('_', ' ').title()}")
            panel += 1

        if suptitle is not None:
            fig.suptitle(suptitle)
        fig.tight_layout()
        return fig

    def plot3d(
            self,
            sample: Sample,
            show_titles: bool = True,
            suptitle: str | None = None,
            **kwargs,
    ) -> Figure:
        """Visualise a sample: 3D elevation panel."""
        has_elev = "elevation" in sample

        fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "3d"})
        ax: Axes3D

        if has_elev:
            elev = sample["elevation"]
            if elev.ndim == 4:
                elev = elev[0]
            elev_np = elev[0].numpy().copy()
            valid = np.isfinite(elev_np)

            x, y = np.meshgrid(range(elev_np.shape[1]), range(elev_np.shape[0]))

            if valid.any():
                stride = max(1, min(elev_np.shape[0], elev_np.shape[1]) // 100)

                im = ax.plot_surface(
                    np.where(valid, x, np.nan),
                    np.where(valid, y, np.nan),
                    np.where(valid, elev_np, np.nan),
                    cmap="terrain",
                    rstride=stride,
                    cstride=stride,
                    linewidth=0,
                    antialiased=False,
                )
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             label="Elevation (m)")

            ax.axis("off")
            if show_titles:
                ax.set_title("3D Elevation (DTM)")

        if suptitle is not None:
            fig.suptitle(suptitle)
        fig.tight_layout()
        return fig

    @staticmethod
    def _percentile_stretch(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Per-channel 2–98 percentile stretch over non-zero pixels."""
        out = img.copy()
        if out.ndim == 3:
            for c in range(out.shape[2]):
                band = out[..., c]
                mask = np.abs(band) > eps
                data = band[mask]
                if len(data) > 0:
                    p2, p98 = np.percentile(data, [2, 98])
                    if p98 > p2:
                        out[..., c] = np.clip((band - p2) / (p98 - p2), 0, 1)
                        out[..., c][~mask] = 0
        else:
            mask = np.abs(out) > eps
            data = out[mask]
            if len(data) > 0:
                p2, p98 = np.percentile(data, [2, 98])
                if p98 > p2:
                    out = np.clip((out - p2) / (p98 - p2), 0, 1)
                    out[~mask] = 0
        return out

    # ------------------------------------------------------------------
    # Post-download verification hook
    # ------------------------------------------------------------------

    def _post_download_verify(self) -> None:
        sample_size = min(20, len(self.index))
        sample_rows = self.index.sample(n=sample_size, random_state=0)
        found = 0
        for _, row in sample_rows.iterrows():
            p = row.get("dtm_path")
            if p is not None and pathlib.Path(p).exists():
                found += 1
        if found == 0:
            if not self.download:
                logger.warning(
                    "No DTM .IMG files found under '%s' for any of %d "
                    "sampled stereo pairs. Either run with download=True "
                    "to fetch data or point 'root' at an existing PDS "
                    "mirror that preserves the directory structure.\n"
                    "  <root>/images/DTEEC_xxxxxx_xxxx_yyyyyy_yyyy_Vnn.IMG",
                    self.root, sample_size,
                )
            else:
                logger.warning(
                    "Download completed but no DTM .IMG files found on "
                    "disk. Check network connectivity and available disk "
                    "space.  Note: HiRISE DTM files are typically "
                    "100 MB–1 GB each; an unfiltered download may exceed "
                    "10 TB.  Use --bbox or --target to limit scope."
                )

    def _get_ortho_overlap(self, dtm_geom: shapely.geometry.Polygon, ortho_path: str) -> float:
        """Calculate the area intersection ratio of an Orthoimage against the DTM footprint."""
        try:
            actual_path = self.prefer_cog(pathlib.Path(ortho_path))
            if not actual_path or not actual_path.exists():
                return 0.0

            with rasterio.open(actual_path) as src:
                src_crs = src.crs
                if src_crs is None:
                    return 1.0  # Assume it overlaps if we can't project

                fl, fb, fr, ft = rasterio.warp.transform_bounds(src_crs, self.mars_crs, *src.bounds)

                # Normalize to [-180, 180]
                fl_norm = ((fl + 180.0) % 360.0) - 180.0
                fr_norm = ((fr + 180.0) % 360.0) - 180.0

                if fl_norm > fr_norm:
                    # Antimeridian crossing: split into two bounding boxes
                    ortho_geom = shapely.ops.unary_union([
                        box(fl_norm, fb, 180.0, ft),
                        box(-180.0, fb, fr_norm, ft)
                    ])
                else:
                    ortho_geom = box(fl_norm, fb, fr_norm, ft)

                if dtm_geom.area == 0:
                    return 0.0

                intersection = dtm_geom.intersection(ortho_geom)
                return intersection.area / dtm_geom.area

        except Exception as e:
            logger.debug("Failed overlap check for %s: %s", ortho_path, e)
            return 0.0

    # ------------------------------------------------------------------
    # Spatial index
    # ------------------------------------------------------------------

    def _build_spatial_index(self, force_rebuild: bool = False) -> None:
        """Construct GeoDataFrame with exactly one row per DTM stereo pair.

        Uses raster-based footprint extraction (DTM + orthos) and unions
        all valid footprints per stereo pair.
        """
        if not force_rebuild and self._try_load_cache():
            return

        df = self._raw_index.copy()

        # ──────────────────────────────────────────────────────────────
        # Normalize fields
        # ──────────────────────────────────────────────────────────────
        df["_data_type"] = df["DATA_TYPE"].str.strip().str.upper()
        df["LEFT_OBSERVATION_ID"] = df["LEFT_OBSERVATION_ID"].astype(str).str.strip()
        df["RIGHT_OBSERVATION_ID"] = df["RIGHT_OBSERVATION_ID"].astype(str).str.strip()

        df["_local_path"] = df["FILE_NAME_SPECIFICATION"].apply(
            lambda s: str(self._pds_local_path(s))
        )

        # Parse ortho metadata
        df["_ortho_color"] = None
        df["_ortho_scale"] = None
        df["_ortho_obs_id"] = None

        for i, row in df.iterrows():
            m = _ORTHO_PATTERN.search(str(row["PRODUCT_ID"]).strip())
            if m:
                df.at[i, "_ortho_obs_id"] = m.group(1)
                df.at[i, "_ortho_color"] = m.group(2)
                df.at[i, "_ortho_scale"] = m.group(3)

        # ──────────────────────────────────────────────────────────────
        # Step 1: Extract DTM rows (authoritative pairs)
        # ──────────────────────────────────────────────────────────────
        dtm_df = df[df["_data_type"].isin(DTM_DATA_TYPES)].copy()

        if dtm_df.empty:
            raise DatasetNotFoundError(self)

        dtm_df["_pair_key"] = (
                dtm_df["LEFT_OBSERVATION_ID"] + "__" +
                dtm_df["RIGHT_OBSERVATION_ID"]
        )

        # ──────────────────────────────────────────────────────────────
        # Step 2: Build obs_id → list[pair_key] mapping
        # ──────────────────────────────────────────────────────────────
        obs_to_pairs: dict[str, list[str]] = {}

        for _, row in dtm_df.iterrows():
            L = row["LEFT_OBSERVATION_ID"]
            R = row["RIGHT_OBSERVATION_ID"]
            key = row["_pair_key"]

            if key not in obs_to_pairs.setdefault(L, []):
                obs_to_pairs[L].append(key)
            if key not in obs_to_pairs.setdefault(R, []):
                obs_to_pairs[R].append(key)

        # ──────────────────────────────────────────────────────────────
        # Step 3: Assign pair_key to ALL rows (Exploded Orthos)
        # ──────────────────────────────────────────────────────────────
        exploded_rows = []

        for _, row in df.iterrows():
            if row["_data_type"] in DTM_DATA_TYPES:
                row_copy = row.copy()
                row_copy["_pair_key"] = row_copy["LEFT_OBSERVATION_ID"] + "__" + row_copy["RIGHT_OBSERVATION_ID"]
                exploded_rows.append(row_copy)
            else:
                # Map ortho rows via observation ID → multiple pair_keys
                obs_id = row.get("_ortho_obs_id")
                if obs_id in obs_to_pairs:
                    for p_key in obs_to_pairs[obs_id]:
                        row_copy = row.copy()
                        row_copy["_pair_key"] = p_key
                        exploded_rows.append(row_copy)

        df = pd.DataFrame(exploded_rows)

        # ──────────────────────────────────────────────────────────────
        # Step 4: Build records + DTM authoritative footprint paths
        # ──────────────────────────────────────────────────────────────
        pair_records: list[dict] = []
        fp_paths_flat: list[str | None] = []
        pair_slices: list[tuple[int, int]] = []

        for pair_key, grp in df.groupby("_pair_key", sort=False):

            dtm_rows = grp[grp["_data_type"].isin(DTM_DATA_TYPES)]
            if dtm_rows.empty:
                continue

            dr = dtm_rows.iloc[0]

            left_id = dr["LEFT_OBSERVATION_ID"]
            right_id = dr["RIGHT_OBSERVATION_ID"]

            rec = dict(
                pair_key=pair_key,
                left_obs_id=left_id,
                right_obs_id=right_id,
                dtm_path=dr["_local_path"],
                dtm_product_id=str(dr["PRODUCT_ID"]).strip(),
                data_elevation_type=str(dr["_data_type"]),
                map_scale=None,
                rationale_desc=str(dr.get("RATIONALE_DESC", "")).strip(),
                left_red_path=None,
                right_red_path=None,
                left_irb_path=None,
                right_irb_path=None,
                _ref_row=dr,
            )

            try:
                rec["map_scale"] = float(dr["MAP_SCALE"])
            except (ValueError, TypeError):
                pass

            # Assign orthos
            ortho_rows = grp[
                grp["_data_type"].isin({d.upper() for d in ORTHO_DATA_TYPES})
            ]

            for _, orow in ortho_rows.iterrows():
                self._assign_ortho_path(rec, orow, left_id, right_id)

            # Strict Ortho Validation: Ensure the pair has both Left and Right
            # data for ALL requested ortho types. Exclude the pair if incomplete.
            if self.include_ortho:
                missing_required_ortho = False
                for otype in self.ortho_types:
                    color_key = otype.lower()
                    if rec.get(f"left_{color_key}_path") is None or rec.get(f"right_{color_key}_path") is None:
                        missing_required_ortho = True
                        break

                if missing_required_ortho:
                    continue  # Skip this stereo pair entirely

            # Only use the DTM to compute the footprint. Orthoimages contain
            # unreliable padding that generates "ghost" geometries.
            dtm_p = rec.get("dtm_path")
            if dtm_p is not None and pathlib.Path(dtm_p).exists():
                paths = [dtm_p]
            else:
                paths = []

            start = len(fp_paths_flat)
            fp_paths_flat.extend(paths)
            end = len(fp_paths_flat)

            pair_slices.append((start, end))
            pair_records.append(rec)

        # ──────────────────────────────────────────────────────────────
        # Step 5: Run footprint extraction (UNCHANGED)
        # ──────────────────────────────────────────────────────────────
        def _dtm_valid(data: np.ndarray) -> np.ndarray:
            return np.isfinite(data) & (data > -1e30)

        fp_results_flat = self._run_footprint_extraction(
            fp_paths_flat,
            nodata_test=_dtm_valid
        )

        # ──────────────────────────────────────────────────────────────
        # Step 6: Assemble geometries (DTM Authoritative & Overlap Verif)
        # ──────────────────────────────────────────────────────────────
        records: list[dict] = []

        for i, rec in enumerate(pair_records):
            start, end = pair_slices[i]
            pair_fp_results = fp_results_flat[start:end]

            ref = rec.pop("_ref_row")

            # Default to None, meaning we rely on index metadata if no file exists
            hull_coords = None
            file_bounds = None

            # If we successfully extracted a footprint from the DTM, use it
            if pair_fp_results:
                hull_coords, file_bounds = pair_fp_results[0]

            geom = self._geometry_from_footprint_result(
                hull_coords, file_bounds, ref
            )

            if geom is None or geom.is_empty:
                logger.warning("Could not generate valid geometry for %s", rec["pair_key"])
                continue

            # --- NEW: Orthoimage overlap verification (>75%) ---
            if self.include_ortho:
                ortho_validation = False
                for otype in self.ortho_types:
                    color_key = otype.lower()
                    for side in ("left", "right"):
                        ortho_path = rec.get(f"{side}_{color_key}_path")

                        if ortho_path is not None and (ortho_path := pathlib.Path(ortho_path)).exists():
                            overlap_ratio = self._get_ortho_overlap(geom, ortho_path)

                            if overlap_ratio < 0.75:
                                logger.error(
                                    "Misalignment detected! DTM '%s' and Ortho '%s' "
                                    "only overlap by %.1f%%. Dropping pair from index.",
                                    rec["dtm_product_id"],
                                    pathlib.Path(ortho_path).name,
                                    overlap_ratio * 100
                                )
                                ortho_validation = True
                                break

                            lbl_path = ortho_path.with_suffix(".LBL")
                            meta = ProductMeta.from_lbl(lbl_path)

                            if abs(meta.offset) < _EPS and abs(meta.scaling_factor - 1) < _EPS:
                                logger.error(
                                    "Incorrect scaling_factor and offset! DTM '%s''s associated Ortho '%s' "
                                    "does not have valid offset (%0.3f) and scaling factors (%0.3f). Dropping pair from index.",
                                    rec["dtm_product_id"],
                                    pathlib.Path(ortho_path).name,
                                    meta.offset,
                                    meta.scaling_factor
                                )
                                ortho_validation = True
                                break

                    if ortho_validation:
                        break

                # If any assigned ortho fails the overlap check, drop the entire pair
                if ortho_validation:
                    continue

            rec["geometry"] = geom

            # Time handling
            _epoch_str = "2006-01-01T00:00:00"
            rec["t_start"] = (
                ref["START_TIME"] if "START_TIME" in ref.index else _epoch_str
            )
            rec["t_stop"] = (
                ref["STOP_TIME"] if "STOP_TIME" in ref.index else _epoch_str
            )

            records.append(rec)

        # ──────────────────────────────────────────────────────────────
        # Step 7: Build GeoDataFrame
        # ──────────────────────────────────────────────────────────────
        obs_df = pd.DataFrame(records)
        if obs_df.empty:
            raise DatasetNotFoundError(self)

        _epoch = pd.Timestamp("2006-01-01", tz="UTC")

        t_start = pd.to_datetime(
            obs_df["t_start"], utc=True, errors="coerce"
        ).fillna(_epoch)

        t_stop = pd.to_datetime(
            obs_df["t_stop"], utc=True, errors="coerce"
        ).fillna(t_start)

        geometries = gpd.GeoSeries(
            obs_df["geometry"].tolist(), crs=self.mars_crs
        )

        data_cols = {
            col: obs_df[col].values
            for col in obs_df.columns
            if col not in ("geometry", "t_start", "t_stop")
        }

        self.index = gpd.GeoDataFrame(
            data_cols,
            index=pd.IntervalIndex.from_arrays(
                t_start, t_stop, closed="both", name="datetime"
            ),
            geometry=geometries.values,
            crs=self.mars_crs,
        )

        self._log_index_extent()

    # ------------------------------------------------------------------
    # Spatial index helpers
    # ------------------------------------------------------------------

    def _assign_ortho_path(
            self,
            rec: dict,
            orow: pd.Series,
            left_id: str,
            right_id: str,
    ) -> None:
        """Assign an ortho row's path to the correct slot in *rec*."""
        color = orow["_ortho_color"]
        scale = orow["_ortho_scale"]
        obs_id = str(orow["_ortho_obs_id"] or "").strip()
        dt = str(orow["_data_type"]).strip().upper()

        if color not in ("RED", "IRB"):
            return

        # Determine left vs right from DATA_TYPE or observation ID
        if "LEFT" in dt:
            side = "left"
        elif "RIGHT" in dt:
            side = "right"
        elif obs_id == left_id:
            side = "left"
        elif obs_id == right_id:
            side = "right"
        else:
            return

        col_key = f"{side}_{color.lower()}_path"
        scale_key = f"{col_key}_scale"

        current_path = rec.get(col_key)
        current_scale = rec.get(scale_key)

        # Enforce exact scale preference if defined, otherwise prioritize finest resolution
        if self.ortho_scale:
            if scale == self.ortho_scale:
                rec[col_key] = orow["_local_path"]
                rec[scale_key] = scale
            elif current_path is None:
                rec[col_key] = orow["_local_path"]
                rec[scale_key] = scale
        else:
            # Scale notation ranges from A (finest) to D (coarsest)
            if current_path is None or (current_scale and scale < current_scale):
                rec[col_key] = orow["_local_path"]
                rec[scale_key] = scale

    @staticmethod
    def _pick_footprint_file(rec: dict) -> str | None:
        """Return the first file that exists on disk for footprint extraction."""
        for col in ("dtm_path", "left_red_path", "right_red_path",
                    "left_irb_path", "right_irb_path"):
            p = rec.get(col)
            if p is not None:
                pp = pathlib.Path(p)
                if pp.exists() or pp.with_suffix(".tif").exists():
                    return p
        return None

    # ------------------------------------------------------------------
    # Download tasks
    # ------------------------------------------------------------------

    def _build_download_tasks(self) -> list[tuple[str, pathlib.Path]]:
        tasks: list[tuple[str, pathlib.Path]] = []
        for _, row in self._raw_index.iterrows():
            spec = row["FILE_NAME_SPECIFICATION"].strip()
            local = self._pds_local_path(spec)

            if not local.exists():
                tasks.append((f"{self.url}/{spec}", local))

            # For ortho JP2s, also fetch the companion LBL.
            # (DTM .IMG files have attached PDS3 labels — no separate LBL.)
            if spec.upper().endswith(".JP2"):
                lbl_spec = spec[:-4] + ".LBL"
                lbl_local = self._pds_local_path(lbl_spec)
                if not lbl_local.exists():
                    tasks.append(
                        (f"{self.url}/{lbl_spec}", lbl_local)
                    )
        return tasks

    # ------------------------------------------------------------------
    # Tile loading — DTM elevation
    # ------------------------------------------------------------------

    def _load_dtm_tile(
            self,
            dtm_path: pathlib.Path,
            x: slice,
            y: slice,
    ) -> torch.Tensor | None:
        """Load and reproject a DTM .IMG patch.

        Returns ``(1, H, W)`` float32 with elevation in metres;
        nodata pixels are ``NaN``.
        """
        dtm_path = self.prefer_cog(dtm_path)
        if dtm_path is None or not dtm_path.exists():
            return None

        out_w = max(1, int(round((x.stop - x.start) / (x.step or self.res))))
        out_h = max(1, int(round((y.stop - y.start) / (y.step or self.res))))

        dst_transform = rasterio.transform.from_bounds(
            x.start, y.start, x.stop, y.stop, out_w, out_h
        )
        dst_crs = rasterio.crs.CRS.from_user_input(self.mars_crs)

        try:
            with rasterio.open(dtm_path) as src:
                if not check_overlap(src, dst_crs, x, y):
                    logger.debug(
                        "Query doesn't overlap DTM bounds: %s", dtm_path.name
                    )
                    return None

                src_nodata = src.nodata if src.nodata is not None else _DTM_NODATA

                dest = reproject_band(
                    src, 1, dst_crs, dst_transform, out_h, out_w,
                    src_nodata=src_nodata,
                    dst_nodata=float("nan"),
                )
                return torch.from_numpy(dest).unsqueeze(0)

        except rasterio.errors.RasterioIOError as exc:
            logger.warning("Could not open DTM %s: %s", dtm_path, exc)
            return None

    # ------------------------------------------------------------------
    # Tile loading — orthoimages
    # ------------------------------------------------------------------

    def _load_ortho_tile(
            self,
            jp2_path: pathlib.Path,
            color: str,
            x: slice,
            y: slice,
    ) -> torch.Tensor | None:
        """Load and reproject an orthoimage JP2 patch.

        Returns ``(C, H, W)`` float32 in I/F [0, 1].
        """
        jp2_path = self.prefer_cog(jp2_path)
        if jp2_path is None or not jp2_path.exists():
            return None

        lbl_path = jp2_path.with_suffix(".LBL")
        meta = ProductMeta.from_lbl(lbl_path)

        band_map = _IRB_BAND.copy() if color == "IRB" else _RED_BAND.copy()

        out_w = max(1, int(round((x.stop - x.start) / (x.step or self.res))))
        out_h = max(1, int(round((y.stop - y.start) / (y.step or self.res))))

        dst_transform = rasterio.transform.from_bounds(
            x.start, y.start, x.stop, y.stop, out_w, out_h
        )
        dst_crs = rasterio.crs.CRS.from_user_input(self.mars_crs)

        try:
            with rasterio.open(jp2_path) as src:
                if not check_overlap(src, dst_crs, x, y):
                    return None

                bands: list[np.ndarray] = []
                for ch_name, band_idx in band_map.items():
                    if band_idx > src.count:
                        logger.warning(
                            "Band %d absent in %s (%d bands); filling zeros.",
                            band_idx, jp2_path.name, src.count,
                        )
                        bands.append(np.zeros((out_h, out_w), dtype=np.float32))
                        continue

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
                        bands.append(np.zeros((out_h, out_w), dtype=np.float32))
                        continue

                    # Radiometric calibration: I/F = DN * scaling + offset
                    nodata_mask = dest == 0.0
                    dest *= meta.scaling_factor
                    dest += meta.offset
                    np.clip(dest, 0.0, 1.0, out=dest)
                    dest[nodata_mask] = 0.0
                    bands.append(dest)

                return torch.from_numpy(np.stack(bands))

        except rasterio.errors.RasterioIOError as exc:
            logger.warning("Could not open ortho %s: %s", jp2_path, exc)
            return None

    # ------------------------------------------------------------------
    # Elevation tile merging
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_elevation_tiles(tiles: list[torch.Tensor]) -> torch.Tensor:
        """Mosaic elevation tiles with first-valid-wins (NaN = nodata)."""
        if len(tiles) == 1:
            return tiles[0]

        max_h = max(t.shape[1] for t in tiles)
        max_w = max(t.shape[2] for t in tiles)

        merged = torch.full((1, max_h, max_w), float("nan"), dtype=torch.float32)
        for tile in tiles:
            h, w = tile.shape[1], tile.shape[2]
            empty = torch.isnan(merged[:, :h, :w])
            valid = ~torch.isnan(tile)
            merged[:, :h, :w][empty & valid] = tile[empty & valid]
        return merged


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv=None) -> None:  # pragma: no cover
    setup_logging()

    import argparse

    parser = argparse.ArgumentParser(
        description="Run a sample test on the HiRISE DTM dataset"
    )
    parser.add_argument("-t", "--target", type=str, default=None)
    parser.add_argument(
        "--bbox", type=float, nargs=4, default=None,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
    )
    parser.add_argument(
        "--ortho-type", type=str, nargs="+", default=["RED"],
        choices=["RED", "IRB"],
    )
    parser.add_argument("--no-ortho", action="store_true")
    parser.add_argument("-l", "--length", type=int, default=-1)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument(
        "-d", action=argparse.BooleanOptionalAction,
        help="Whether to generate 3D visualisation plots of the surface",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_HIRISE_DTM_VIZ_OUT,
        help=(
            "Directory for generated DTM coverage/sample figures "
            f"(default: {DEFAULT_HIRISE_DTM_VIZ_OUT})"
        ),
    )

    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    bbox_tuple = tuple(args.bbox) if args.bbox else None

    if args.bbox:
        logger.info("Loading bbox: %s", str(args.bbox))
    elif args.target:
        logger.info("Loading target: %s", args.target)
    else:
        logger.info("Loading all DTMs available")

    # Warn if no filter is set — unfiltered DTM download is ~10+ TB.
    if args.target is None and bbox_tuple is None:
        logger.warning(
            "No --target or --bbox filter specified.  The full DTM index "
            "contains ~11,600 products totalling >10 TB.  Using a default "
            "bbox for Eberswalde Crater as a demo.  Pass --bbox or "
            "--target to override."
        )

        bbox_tuple = (-150, 15, -90, 70)

    dataset = MarsHiRISEDTM(
        target=args.target,
        bbox=bbox_tuple,
        include_ortho=not args.no_ortho,
        ortho_type=args.ortho_type,
        download=True,
        reuse_cache=True,
    )

    output_path = args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    fig = dataset.plot_coverage()
    fig.savefig(output_path / "dtm_coverage.png", bbox_inches='tight')
    logger.info("Saved coverage figure.")

    from dataset.hirise_sampler import HiRISEGeoSampler

    sampler = HiRISEGeoSampler(
        dataset, size=0.01,
        length=None if args.length <= 0 else args.length,
        units=Units.CRS,
    )

    logger.info("Number of samples: %d", len(sampler))

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset, sampler=sampler,
        num_workers=4, multiprocessing_context="spawn", prefetch_factor=10,
    )

    output_path_3d = output_path / '3d'

    for i, sample in enumerate(dataloader):
        if args.d:
            output_path_3d.mkdir(parents=True, exist_ok=True)
            fig = dataset.plot3d(sample)
            fig.savefig(output_path_3d / f"dtm_output{i}_3d.png")
            logger.info("Saved dtm_output%d_3d.png", i)
            plt.close(fig)

        output_path_flat = output_path / 'flat'
        output_path_flat.mkdir(parents=True, exist_ok=True)
        fig = dataset.plot(sample)
        fig.savefig(output_path_flat / f"dtm_output{i}.png")
        logger.info("Saved dtm_output%d.png", i)
        plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    main()
