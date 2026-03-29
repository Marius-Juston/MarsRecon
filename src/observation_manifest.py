"""Observation-level manifest builder for MarsCLIP-style pretraining.

This module provides the first implementation slice for a tri-modal Mars
pretraining pipeline:

* one record per HiRISE observation,
* one preferred COLOR image path per record,
* observation text from ``RATIONALE_DESC``, and
* geospatial / viewing metadata derived from the cumulative index.

The builder intentionally operates at observation level rather than tile level,
because HiRISE text metadata are observation-level descriptions.
"""

from __future__ import annotations

import logging
import pathlib
import re
from typing import Any

import pandas as pd
import pdr

logger = logging.getLogger(__name__)

_INDEX_STEM = "RDRCUMINDEX"
_PRODUCT_RE = re.compile(r"_(COLOR|RED)\s*$")


def _normalize_longitude(value: float | int) -> float:
    """Convert PDS longitudes from ``[0, 360]`` to ``[-180, 180]``."""
    return ((float(value) + 180.0) % 360.0) - 180.0


def _normalize_longitude_series(series: pd.Series) -> pd.Series:
    """Vectorized longitude normalization."""
    return ((series.astype(float) + 180.0) % 360.0) - 180.0


def _clean_text(value: Any) -> str:
    """Collapse repeated whitespace in free-text fields."""
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def _apply_bbox_filter(
    df: pd.DataFrame,
    bbox: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    """Filter a cumulative-index table using the dataset bbox semantics."""
    if bbox is None:
        return df.copy()

    lon_min, lat_min, lon_max, lat_max = bbox
    obs_lon_min = _normalize_longitude_series(df["MINIMUM_LONGITUDE"])
    obs_lon_max = _normalize_longitude_series(df["MAXIMUM_LONGITUDE"])
    obs_lat_min = df["MINIMUM_LATITUDE"].astype(float)
    obs_lat_max = df["MAXIMUM_LATITUDE"].astype(float)

    overlap = (
        (obs_lon_max >= lon_min)
        & (obs_lon_min <= lon_max)
        & (obs_lat_max >= lat_min)
        & (obs_lat_min <= lat_max)
    )
    return df[overlap].copy()


def _choose_image_path(
    jp2_path: pathlib.Path,
    tif_path: pathlib.Path,
    has_local_jp2: bool,
    has_local_tif: bool,
    prefer_cog: bool,
) -> pathlib.Path | None:
    """Choose the preferred local image path for one observation."""
    if prefer_cog and has_local_tif:
        return tif_path
    if has_local_jp2:
        return jp2_path
    if has_local_tif:
        return tif_path
    return None


def _row_to_record(
    row: pd.Series,
    root: pathlib.Path,
    prefer_cog: bool,
) -> dict[str, object]:
    """Convert one COLOR index row into an observation manifest record."""
    images_root = root / "images"
    product_id = str(row["PRODUCT_ID"]).strip()
    file_name = str(row["FILE_NAME_SPECIFICATION"]).strip().split("/")[-1]

    jp2_path = images_root / file_name
    tif_path = images_root / f"{product_id}.tif"
    has_local_jp2 = jp2_path.exists()
    has_local_tif = tif_path.exists()
    image_path = _choose_image_path(
        jp2_path=jp2_path,
        tif_path=tif_path,
        has_local_jp2=has_local_jp2,
        has_local_tif=has_local_tif,
        prefer_cog=prefer_cog,
    )

    min_lon = _normalize_longitude(row["MINIMUM_LONGITUDE"])
    max_lon = _normalize_longitude(row["MAXIMUM_LONGITUDE"])
    if min_lon > max_lon:
        raise ValueError("Observation straddles the antimeridian.")

    min_lat = float(row["MINIMUM_LATITUDE"])
    max_lat = float(row["MAXIMUM_LATITUDE"])
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat

    stereo_flag = str(row["STEREO_FLAG"]).strip().upper()

    return {
        "obs_id": str(row["OBSERVATION_ID"]).strip(),
        "product_id": product_id,
        "image_path": str(image_path) if image_path is not None else None,
        "image_format": image_path.suffix.lower().lstrip(".") if image_path else None,
        "color_jp2_path": str(jp2_path) if has_local_jp2 else None,
        "color_tif_path": str(tif_path) if has_local_tif else None,
        "has_local_jp2": has_local_jp2,
        "has_local_tif": has_local_tif,
        "has_local_image": has_local_jp2 or has_local_tif,
        "rationale_desc": _clean_text(row["RATIONALE_DESC"]),
        "start_time": pd.to_datetime(row["START_TIME"], utc=True, errors="coerce"),
        "stop_time": pd.to_datetime(row["STOP_TIME"], utc=True, errors="coerce"),
        "min_lon": min_lon,
        "max_lon": max_lon,
        "min_lat": min_lat,
        "max_lat": max_lat,
        "centroid_lon": (min_lon + max_lon) / 2.0,
        "centroid_lat": (min_lat + max_lat) / 2.0,
        "lon_span_deg": lon_span,
        "lat_span_deg": lat_span,
        "bbox_area_deg2": lon_span * lat_span,
        "image_lines": int(row["IMAGE_LINES"]),
        "line_samples": int(row["LINE_SAMPLES"]),
        "map_scale": float(row["MAP_SCALE"]),
        "map_resolution": float(row["MAP_RESOLUTION"]),
        "emission_angle": float(row["EMISSION_ANGLE"]),
        "incidence_angle": float(row["INCIDENCE_ANGLE"]),
        "phase_angle": float(row["PHASE_ANGLE"]),
        "local_time": float(row["LOCAL_TIME"]),
        "solar_longitude": float(row["SOLAR_LONGITUDE"]),
        "sub_solar_azimuth": float(row["SUB_SOLAR_AZIMUTH"]),
        "north_azimuth": float(row["NORTH_AZIMUTH"]),
        "spacecraft_altitude": float(row["SPACECRAFT_ALTITUDE"]),
        "stereo_flag": stereo_flag,
        "is_stereo": stereo_flag == "YES",
        "projection_center_latitude": float(row["PROJECTION_CENTER_LATITUDE"]),
        "projection_center_longitude": _normalize_longitude(
            row["PROJECTION_CENTER_LONGITUDE"]
        ),
        "has_near_infrared": True,
        "has_red": True,
        "has_blue_green": True,
        "channel_count": 3,
    }


def build_observation_manifest_from_index(
    df: pd.DataFrame,
    root: pathlib.Path | str,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    require_local_image: bool = True,
    prefer_cog: bool = True,
) -> pd.DataFrame:
    """Build one COLOR observation record per observation from an index table."""
    root_path = pathlib.Path(root)

    working = df.copy()
    working["PRODUCT_ID"] = working["PRODUCT_ID"].astype(str).str.strip()
    working["OBSERVATION_ID"] = working["OBSERVATION_ID"].astype(str).str.strip()
    working["_product_type"] = working["PRODUCT_ID"].str.extract(
        _PRODUCT_RE, expand=False
    )
    working = working[working["_product_type"] == "COLOR"].copy()
    working = _apply_bbox_filter(working, bbox)
    working = working.drop_duplicates("OBSERVATION_ID", keep="first").copy()

    records: list[dict[str, object]] = []
    for _, row in working.iterrows():
        try:
            record = _row_to_record(row, root_path, prefer_cog=prefer_cog)
        except ValueError:
            logger.warning(
                "Observation %s straddles antimeridian. Skipping.",
                row["OBSERVATION_ID"],
            )
            continue

        if require_local_image and not record["has_local_image"]:
            continue
        records.append(record)

    manifest = pd.DataFrame.from_records(records)
    if manifest.empty:
        return manifest

    # Keep optional local-path fields as Python ``None`` rather than pandas
    # NaN so downstream dataset code can use simple ``is None`` checks.
    nullable_object_cols = [
        "image_path",
        "image_format",
        "color_jp2_path",
        "color_tif_path",
    ]
    for col in nullable_object_cols:
        manifest[col] = manifest[col].astype(object)
        manifest[col] = manifest[col].where(pd.notna(manifest[col]), None)
    manifest = manifest.sort_values("obs_id").reset_index(drop=True)
    return manifest


def build_observation_manifest(
    root: pathlib.Path | str,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    require_local_image: bool = True,
    prefer_cog: bool = True,
) -> pd.DataFrame:
    """Load the HiRISE cumulative index from disk and build a COLOR manifest."""
    root_path = pathlib.Path(root)
    lbl_path = root_path / f"{_INDEX_STEM}.LBL"
    data = pdr.read(str(lbl_path))
    data.load("all")
    index_df: pd.DataFrame = data["RDR_INDEX_TABLE"]

    return build_observation_manifest_from_index(
        index_df,
        root=root_path,
        bbox=bbox,
        require_local_image=require_local_image,
        prefer_cog=prefer_cog,
    )
