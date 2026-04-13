"""Patch-level MarsCLIP dataset aligned to the Stage A workflow."""

from __future__ import annotations

import pathlib
import re
from ast import literal_eval
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd
import torch
from shapely.geometry import box as shapely_box
from torch.utils.data import Dataset
from torchgeo.samplers import Units

from dataset.hirise_sampler import HiRISEGeoSampler, _to_tuple
from dataset.mars_hirise import ALL_CHANNELS, MarsHiRISE
from clip.marsclip_dataset import (
    GEO_FEATURE_NAMES,
    VIEWING_FEATURE_NAMES,
    _build_geo_features,
    _build_viewing_features,
)
from clip.observation_manifest import _clean_text, _normalize_longitude
from clip.rationale_cache import merge_rationale_cache

PATCH_SCALE_FEATURE_NAMES: tuple[str, ...] = (
    "map_scale",
    "map_resolution",
    "patch_lon_span_deg",
    "patch_lat_span_deg",
    "patch_area_deg2",
    "dominant_overlap_fraction",
    "source_obs_count",
    "patch_aspect_ratio",
)

DEFAULT_PATCH_VALID_FRACTION = 0.5

_PRODUCT_RE = re.compile(r"_(COLOR|RED)\s*$")


def _normalize_obs_id_set(obs_ids: Sequence[str]) -> set[str]:
    """Normalize a sequence of observation ids into a string set."""
    return {str(obs_id) for obs_id in obs_ids}


def _filter_observation_metadata_to_color(
    observation_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only observations with full three-band COLOR support."""
    required = {"obs_id", "has_near_infrared", "has_blue_green"}
    missing = required.difference(observation_metadata.columns)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(
            f"observation_metadata is missing required COLOR filter columns: {missing_names}"
        )

    mask = (
        observation_metadata["has_near_infrared"].astype(bool)
        & observation_metadata["has_blue_green"].astype(bool)
    )
    filtered = observation_metadata.loc[mask].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No COLOR-capable observations remain after color_only filtering.")
    return filtered


def _filter_geo_dataset_to_obs_ids(
    geo_dataset: MarsHiRISE | Any,
    *,
    allowed_obs_ids: set[str],
) -> None:
    """Restrict a GeoDataset in-place to a set of allowed observation ids."""
    if hasattr(geo_dataset, "index") and getattr(geo_dataset, "index") is not None:
        index = geo_dataset.index.copy()
        if "obs_id" not in index.columns:
            raise ValueError("geo_dataset.index must include an 'obs_id' column for color filtering.")
        index["obs_id"] = index["obs_id"].astype(str)
        geo_dataset.index = index.loc[index["obs_id"].isin(allowed_obs_ids)].copy()
        if len(geo_dataset.index) == 0:
            raise ValueError("No spatial-index observations remain after color_only filtering.")

    if hasattr(geo_dataset, "_raw_index") and getattr(geo_dataset, "_raw_index") is not None:
        raw_index = geo_dataset._raw_index.copy()
        if "PRODUCT_ID" not in raw_index.columns:
            raise ValueError("geo_dataset._raw_index must include a 'PRODUCT_ID' column for color filtering.")
        product_ids = raw_index["PRODUCT_ID"].astype(str).str.strip()
        obs_ids = product_ids.map(_observation_id)
        geo_dataset._raw_index = raw_index.loc[obs_ids.isin(allowed_obs_ids)].copy()


def _filter_patch_records_to_color(
    patch_records: pd.DataFrame,
    *,
    allowed_obs_ids: set[str],
) -> pd.DataFrame:
    """Keep only patch records whose contributing observations are COLOR-capable."""
    required = {"dominant_obs_id", "contributing_obs_ids", "has_near_infrared", "has_blue_green"}
    missing = required.difference(patch_records.columns)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(
            f"patch_records is missing required COLOR filter columns: {missing_names}"
        )

    def _all_allowed(contributing_obs_ids: object) -> bool:
        if isinstance(contributing_obs_ids, Sequence) and not isinstance(contributing_obs_ids, str):
            return all(str(obs_id) in allowed_obs_ids for obs_id in contributing_obs_ids)
        return False

    mask = (
        patch_records["dominant_obs_id"].astype(str).isin(allowed_obs_ids)
        & patch_records["has_near_infrared"].astype(bool)
        & patch_records["has_blue_green"].astype(bool)
        & patch_records["contributing_obs_ids"].map(_all_allowed)
    )
    filtered = patch_records.loc[mask].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No patch records remain after color_only filtering.")
    return filtered


def _normalize_image_size(value: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize an image size specification to ``(height, width)``."""
    if isinstance(value, int):
        return value, value
    return int(value[0]), int(value[1])


def _product_type(product_id: str) -> str | None:
    """Extract the HiRISE product suffix (``COLOR`` or ``RED``)."""
    match = _PRODUCT_RE.search(str(product_id).strip())
    return match.group(1) if match else None


def _observation_id(product_id: str) -> str:
    """Strip the product suffix from a HiRISE product id."""
    return _PRODUCT_RE.sub("", str(product_id).strip())


def _build_patch_scale_features(
    patch_row: pd.Series,
    obs_row: pd.Series,
) -> torch.Tensor:
    """Build patch-level scale features from one patch record and observation row."""
    patch_lon = max(1e-12, float(patch_row["patch_lon_span_deg"]))
    patch_lat = max(1e-12, float(patch_row["patch_lat_span_deg"]))
    values = [
        float(obs_row["map_scale"]),
        float(obs_row["map_resolution"]),
        patch_lon,
        patch_lat,
        float(patch_row["patch_area_deg2"]),
        float(patch_row["dominant_overlap_fraction"]),
        float(patch_row["source_obs_count"]),
        patch_lon / patch_lat,
    ]
    return torch.tensor(values, dtype=torch.float32)


def build_patch_observation_metadata(
    geo_dataset: MarsHiRISE | Any,
    *,
    rationale_cache: pd.DataFrame | pathlib.Path | str | None = None,
) -> pd.DataFrame:
    """Build one metadata row per observation for patch-level MarsCLIP samples."""
    raw_index = getattr(geo_dataset, "_raw_index", None)
    if raw_index is None:
        raise ValueError("geo_dataset must expose a populated _raw_index DataFrame.")

    working = raw_index.copy()
    working["PRODUCT_ID"] = working["PRODUCT_ID"].astype(str).str.strip()
    working["_obs_id"] = working["PRODUCT_ID"].map(_observation_id)
    working["_product_type"] = working["PRODUCT_ID"].map(_product_type)

    records: list[dict[str, object]] = []
    for obs_id, group in working.groupby("_obs_id", sort=False):
        color_rows = group[group["_product_type"] == "COLOR"]
        red_rows = group[group["_product_type"] == "RED"]

        primary = color_rows.iloc[0] if not color_rows.empty else group.iloc[0]
        has_color = not color_rows.empty
        has_red = has_color or not red_rows.empty
        stereo_flag = str(primary.get("STEREO_FLAG", "")).strip().upper()

        records.append(
            {
                "obs_id": str(obs_id),
                "product_id": str(primary["PRODUCT_ID"]).strip(),
                "rationale_desc": _clean_text(primary.get("RATIONALE_DESC", "")),
                "start_time": pd.to_datetime(primary.get("START_TIME"), utc=True, errors="coerce"),
                "stop_time": pd.to_datetime(primary.get("STOP_TIME"), utc=True, errors="coerce"),
                "min_lon": _normalize_longitude(primary.get("MINIMUM_LONGITUDE", 0.0)),
                "max_lon": _normalize_longitude(primary.get("MAXIMUM_LONGITUDE", 0.0)),
                "min_lat": float(primary.get("MINIMUM_LATITUDE", 0.0)),
                "max_lat": float(primary.get("MAXIMUM_LATITUDE", 0.0)),
                "centroid_lon": (
                    _normalize_longitude(primary.get("MINIMUM_LONGITUDE", 0.0))
                    + _normalize_longitude(primary.get("MAXIMUM_LONGITUDE", 0.0))
                )
                / 2.0,
                "centroid_lat": (
                    float(primary.get("MINIMUM_LATITUDE", 0.0))
                    + float(primary.get("MAXIMUM_LATITUDE", 0.0))
                )
                / 2.0,
                "lon_span_deg": (
                    _normalize_longitude(primary.get("MAXIMUM_LONGITUDE", 0.0))
                    - _normalize_longitude(primary.get("MINIMUM_LONGITUDE", 0.0))
                ),
                "lat_span_deg": float(primary.get("MAXIMUM_LATITUDE", 0.0))
                - float(primary.get("MINIMUM_LATITUDE", 0.0)),
                "bbox_area_deg2": (
                    (
                        _normalize_longitude(primary.get("MAXIMUM_LONGITUDE", 0.0))
                        - _normalize_longitude(primary.get("MINIMUM_LONGITUDE", 0.0))
                    )
                    * (
                        float(primary.get("MAXIMUM_LATITUDE", 0.0))
                        - float(primary.get("MINIMUM_LATITUDE", 0.0))
                    )
                ),
                "image_lines": int(primary.get("IMAGE_LINES", 0)),
                "line_samples": int(primary.get("LINE_SAMPLES", 0)),
                "map_scale": float(primary.get("MAP_SCALE", 0.0)),
                "map_resolution": float(primary.get("MAP_RESOLUTION", 0.0)),
                "emission_angle": float(primary.get("EMISSION_ANGLE", 0.0)),
                "incidence_angle": float(primary.get("INCIDENCE_ANGLE", 0.0)),
                "phase_angle": float(primary.get("PHASE_ANGLE", 0.0)),
                "local_time": float(primary.get("LOCAL_TIME", 0.0)),
                "solar_longitude": float(primary.get("SOLAR_LONGITUDE", 0.0)),
                "sub_solar_azimuth": float(primary.get("SUB_SOLAR_AZIMUTH", 0.0)),
                "north_azimuth": float(primary.get("NORTH_AZIMUTH", 0.0)),
                "spacecraft_altitude": float(primary.get("SPACECRAFT_ALTITUDE", 0.0)),
                "stereo_flag": stereo_flag,
                "is_stereo": stereo_flag == "YES",
                "has_near_infrared": has_color,
                "has_red": has_red,
                "has_blue_green": has_color,
                "channel_count": 3 if has_color else 1,
            }
        )

    metadata = pd.DataFrame.from_records(records).sort_values("obs_id").reset_index(drop=True)
    if metadata.empty:  # pragma: no cover
        raise ValueError("No observation metadata could be built from the cumulative index.")  # pragma: no cover

    if rationale_cache is not None:
        metadata = merge_rationale_cache(metadata, rationale_cache)
    else:
        metadata["rationale_expanded"] = None
        metadata["has_rationale_expanded"] = False

    return metadata


def _deduplicate_centers(
    centers: Sequence[tuple[float, float, pd.Interval]],
    *,
    precision: int = 12,
) -> list[tuple[float, float, pd.Interval]]:
    """Drop duplicate patch centers created by overlapping strip footprints."""
    unique: dict[tuple[float, float], tuple[float, float, pd.Interval]] = {}
    for cx, cy, interval in centers:
        key = (round(float(cx), precision), round(float(cy), precision))
        unique.setdefault(key, (float(cx), float(cy), interval))
    return list(unique.values())


def _select_centers(
    centers: list[tuple[float, float, pd.Interval]],
    *,
    max_patches: int | None = None,
    generator: torch.Generator | None = None,
) -> list[tuple[float, float, pd.Interval]]:
    """Optionally truncate or subsample a deterministic list of patch centers."""
    if max_patches is None or max_patches >= len(centers):
        return centers
    if generator is None:
        return centers[:max_patches]

    order = torch.randperm(len(centers), generator=generator).tolist()[:max_patches]
    return [centers[i] for i in order]


def build_patch_records(
    geo_dataset: MarsHiRISE | Any,
    *,
    size: float | tuple[float, float] = 0.005,
    stride: float | tuple[float, float] | None = None,
    min_geometry_overlap: float = 0.5,
    observation_metadata: pd.DataFrame | None = None,
    centers: Sequence[tuple[float, float, pd.Interval]] | None = None,
    max_patches: int | None = None,
    generator: torch.Generator | None = None,
) -> pd.DataFrame:
    """Build deterministic patch records from HiRISE sampler centers."""
    if observation_metadata is None:
        observation_metadata = build_patch_observation_metadata(geo_dataset)
    obs_lookup = observation_metadata.set_index("obs_id", drop=False)

    if centers is None:
        sampler = HiRISEGeoSampler(
            geo_dataset,
            size=size,
            stride=stride,
            split="all",
            units=Units.CRS,
            min_overlap=min_geometry_overlap,
        )
        centers = sampler._centers

    size_h, size_w = _to_tuple(size)
    half_h = size_h / 2.0
    half_w = size_w / 2.0

    unique_centers = _deduplicate_centers(list(centers))
    selected_centers = _select_centers(
        unique_centers,
        max_patches=max_patches,
        generator=generator,
    )

    records: list[dict[str, object]] = []
    try:
        spatial_index = geo_dataset.index.sindex
    except Exception:  # pragma: no cover
        spatial_index = None  # pragma: no cover

    for patch_idx, (cx, cy, interval) in enumerate(selected_centers):
        patch_geom = shapely_box(cx - half_w, cy - half_h, cx + half_w, cy + half_h)
        if spatial_index is None:  # pragma: no cover
            candidates = geo_dataset.index  # pragma: no cover
        else:
            candidate_positions = list(spatial_index.intersection(patch_geom.bounds))
            if not candidate_positions:
                continue
            candidates = geo_dataset.index.iloc[candidate_positions]
        candidates = candidates[candidates.geometry.intersects(patch_geom)]
        if candidates.empty:
            continue

        overlaps: list[tuple[str, float]] = []
        for _, row in candidates.iterrows():
            overlap_area = row.geometry.intersection(patch_geom).area
            if overlap_area > 0:
                overlaps.append((str(row["obs_id"]), float(overlap_area)))

        if not overlaps:
            continue

        overlaps.sort(key=lambda item: item[1], reverse=True)
        total_overlap = sum(area for _, area in overlaps)
        dominant_obs_id = overlaps[0][0]
        dominant_meta = obs_lookup.loc[dominant_obs_id]

        contributing_obs_ids = tuple(obs_id for obs_id, _ in overlaps)
        overlap_fractions = tuple(area / total_overlap for _, area in overlaps)
        contributing_rationales = tuple(
            str(obs_lookup.loc[obs_id]["rationale_desc"])
            for obs_id in contributing_obs_ids
            if obs_id in obs_lookup.index
        )

        records.append(
            {
                "patch_id": f"patch_{patch_idx:06d}",
                "x_start": cx - half_w,
                "x_stop": cx + half_w,
                "y_start": cy - half_h,
                "y_stop": cy + half_h,
                "t_start": pd.Timestamp(interval.left),
                "t_stop": pd.Timestamp(interval.right),
                "min_lon": cx - half_w,
                "max_lon": cx + half_w,
                "min_lat": cy - half_h,
                "max_lat": cy + half_h,
                "centroid_lon": cx,
                "centroid_lat": cy,
                "patch_lon_span_deg": size_w,
                "patch_lat_span_deg": size_h,
                "patch_area_deg2": patch_geom.area,
                "dominant_obs_id": dominant_obs_id,
                "contributing_obs_ids": contributing_obs_ids,
                "contributing_rationales": contributing_rationales,
                "overlap_fractions": overlap_fractions,
                "dominant_overlap_fraction": overlap_fractions[0],
                "source_obs_count": len(contributing_obs_ids),
                "has_near_infrared": any(
                    bool(obs_lookup.loc[obs_id]["has_near_infrared"])
                    for obs_id in contributing_obs_ids
                ),
                "has_red": any(
                    bool(obs_lookup.loc[obs_id]["has_red"])
                    for obs_id in contributing_obs_ids
                ),
                "has_blue_green": any(
                    bool(obs_lookup.loc[obs_id]["has_blue_green"])
                    for obs_id in contributing_obs_ids
                ),
                "rationale_raw": str(dominant_meta["rationale_desc"]),
            }
        )

    patch_records = pd.DataFrame.from_records(records)
    if patch_records.empty:
        raise ValueError("No patch records could be built from the requested sampler settings.")

    return patch_records.reset_index(drop=True)


def summarize_patch_records(patch_records: pd.DataFrame) -> dict[str, float]:
    """Compute a few lightweight summary stats for a patch manifest."""
    if patch_records.empty:
        return {
            "num_patches": 0,
            "num_unique_dominant_obs": 0,
            "mean_source_obs_count": 0.0,
            "mean_dominant_overlap_fraction": 0.0,
        }

    return {
        "num_patches": float(len(patch_records)),
        "num_unique_dominant_obs": float(patch_records["dominant_obs_id"].nunique()),
        "mean_source_obs_count": float(patch_records["source_obs_count"].mean()),
        "mean_dominant_overlap_fraction": float(
            patch_records["dominant_overlap_fraction"].mean()
        ),
    }


def save_patch_records(
    patch_records: pd.DataFrame,
    path: pathlib.Path | str,
) -> pathlib.Path:
    """Persist patch records for reuse across training runs."""
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        patch_records.to_pickle(out)
    elif suffix == ".parquet":
        patch_records.to_parquet(out, index=False)
    else:
        patch_records.to_csv(out, index=False)
    return out


def load_patch_records(path: pathlib.Path | str) -> pd.DataFrame:
    """Load cached patch records from disk."""
    source = pathlib.Path(path)
    suffix = source.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        patch_records = pd.read_pickle(source)
    elif suffix == ".parquet":
        patch_records = pd.read_parquet(source)
    else:
        patch_records = pd.read_csv(source)
        tuple_columns = ("contributing_obs_ids", "contributing_rationales", "overlap_fractions")
        for column in tuple_columns:
            if column in patch_records.columns:
                patch_records[column] = patch_records[column].map(literal_eval)
    return patch_records


def summarize_patch_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize sampled Stage A patches using realized valid-mask statistics."""
    if not samples:
        return {
            "num_samples": 0,
            "num_valid_patches": 0,
            "valid_patch_fraction": 0.0,
            "mean_overall_valid_fraction": 0.0,
            "min_overall_valid_fraction": 0.0,
            "max_overall_valid_fraction": 0.0,
            "mean_source_obs_count": 0.0,
            "mean_dominant_overlap_fraction": 0.0,
            "mean_band_valid_fraction": [0.0, 0.0, 0.0],
            "min_valid_fraction_threshold": DEFAULT_PATCH_VALID_FRACTION,
        }

    overall_valid = []
    is_valid = []
    source_counts = []
    dominant_overlap = []
    band_valid = []
    thresholds = []
    for sample in samples:
        metadata = sample["metadata"]
        overall_valid.append(float(metadata["overall_valid_fraction"]))
        is_valid.append(bool(metadata["is_patch_valid"]))
        source_counts.append(float(metadata["source_obs_count"]))
        dominant_overlap.append(float(metadata["dominant_overlap_fraction"]))
        band_valid.append(metadata["band_valid_fraction"].detach().cpu().float())
        thresholds.append(float(metadata["min_valid_fraction"]))

    mean_band_valid = torch.stack(band_valid, dim=0).mean(dim=0).tolist()
    return {
        "num_samples": int(len(samples)),
        "num_valid_patches": int(sum(is_valid)),
        "valid_patch_fraction": float(sum(is_valid) / len(samples)),
        "mean_overall_valid_fraction": float(sum(overall_valid) / len(samples)),
        "min_overall_valid_fraction": float(min(overall_valid)),
        "max_overall_valid_fraction": float(max(overall_valid)),
        "mean_source_obs_count": float(sum(source_counts) / len(source_counts)),
        "mean_dominant_overlap_fraction": float(
            sum(dominant_overlap) / len(dominant_overlap)
        ),
        "mean_band_valid_fraction": [float(x) for x in mean_band_valid],
        "min_valid_fraction_threshold": float(thresholds[0]),
    }


class MarsCLIPPatchDataset(Dataset):
    """Patch-level dataset bridging MarsHiRISE sampling to the MarsCLIP workflow."""

    def __init__(
        self,
        geo_dataset: MarsHiRISE | Any | None = None,
        *,
        root: pathlib.Path | str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        patch_size: float | tuple[float, float] = 0.005,
        stride: float | tuple[float, float] | None = None,
        image_size: int | tuple[int, int] = 224,
        min_geometry_overlap: float = 0.5,
        min_valid_fraction: float = DEFAULT_PATCH_VALID_FRACTION,
        max_patches: int | None = None,
        generator: torch.Generator | None = None,
        color_only: bool = False,
        rationale_cache: pd.DataFrame | pathlib.Path | str | None = None,
        observation_metadata: pd.DataFrame | None = None,
        patch_records: pd.DataFrame | None = None,
        dataset_normalize: bool = False,
        dataset_normalization_path: pathlib.Path | str | None = None,
        use_dominant_obs_only: bool = False,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        if geo_dataset is None:
            if root is None:
                raise ValueError("Either geo_dataset or root must be provided.")
            geo_dataset = MarsHiRISE(  # pragma: no cover
                root=root,  # pragma: no cover
                bbox=bbox,  # pragma: no cover
                channels=list(ALL_CHANNELS),  # pragma: no cover
                download=False,  # pragma: no cover
                normalize=bool(dataset_normalize),  # pragma: no cover
                normalization_path=dataset_normalization_path,  # pragma: no cover
            )  # pragma: no cover

        if observation_metadata is None:
            observation_metadata = build_patch_observation_metadata(
                geo_dataset,
                rationale_cache=rationale_cache,
            )
        else:
            observation_metadata = observation_metadata.copy()
            if rationale_cache is not None:
                observation_metadata = merge_rationale_cache(
                    observation_metadata,
                    rationale_cache,
                )
            elif "rationale_expanded" not in observation_metadata.columns:
                observation_metadata["rationale_expanded"] = None
                observation_metadata["has_rationale_expanded"] = False

        if color_only:
            observation_metadata = _filter_observation_metadata_to_color(observation_metadata)
            allowed_obs_ids = _normalize_obs_id_set(observation_metadata["obs_id"].tolist())
            _filter_geo_dataset_to_obs_ids(geo_dataset, allowed_obs_ids=allowed_obs_ids)
        else:
            allowed_obs_ids = set()

        if patch_records is None:
            patch_records = build_patch_records(
                geo_dataset,
                size=patch_size,
                stride=stride,
                min_geometry_overlap=min_geometry_overlap,
                observation_metadata=observation_metadata,
                max_patches=max_patches,
                generator=generator,
            )
        else:
            patch_records = patch_records.copy()
        if color_only:
            patch_records = _filter_patch_records_to_color(
                patch_records,
                allowed_obs_ids=allowed_obs_ids,
            )
        if max_patches is not None and max_patches < len(patch_records):
            if generator is None:
                patch_records = patch_records.iloc[:max_patches].copy()
            else:
                selected = torch.randperm(len(patch_records), generator=generator).tolist()[
                    :max_patches
                ]
                patch_records = patch_records.iloc[selected].copy()

        if patch_records.empty:
            raise ValueError("Patch record table is empty.")

        self.geo_dataset = geo_dataset
        self.patch_records = patch_records.reset_index(drop=True)
        self.observation_metadata = observation_metadata.set_index("obs_id", drop=False)
        self.image_size = _normalize_image_size(image_size)
        self.patch_size = _to_tuple(patch_size)
        self.min_valid_fraction = float(min_valid_fraction)
        self.color_only = bool(color_only)
        self.dataset_normalize = bool(dataset_normalize)
        self.dataset_normalization_path = (
            str(dataset_normalization_path) if dataset_normalization_path is not None else None
        )
        self.use_dominant_obs_only = bool(use_dominant_obs_only)
        self._obs_index_by_id = None
        if self.use_dominant_obs_only:
            if not hasattr(self.geo_dataset, "index") or getattr(self.geo_dataset, "index") is None:
                raise ValueError("geo_dataset must expose an index to use dominant_obs_only mode.")
            obs_index = self.geo_dataset.index.copy()
            if "obs_id" not in obs_index.columns:
                raise ValueError("geo_dataset.index must include an 'obs_id' column.")
            obs_index["obs_id"] = obs_index["obs_id"].astype(str)
            self._obs_index_by_id = obs_index.set_index("obs_id", drop=False)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.patch_records)

    def _load_patch_sample(
        self,
        patch_row: pd.Series,
        x_step: float,
        y_step: float,
    ) -> dict[str, Any]:
        x_slice = slice(float(patch_row["x_start"]), float(patch_row["x_stop"]), x_step)
        y_slice = slice(float(patch_row["y_start"]), float(patch_row["y_stop"]), y_step)
        t_slice = slice(pd.Timestamp(patch_row["t_start"]), pd.Timestamp(patch_row["t_stop"]), 1)

        if self.use_dominant_obs_only and self._obs_index_by_id is not None:
            obs_id = str(patch_row["dominant_obs_id"])
            obs_entry = self._obs_index_by_id.loc[obs_id]
            cp = obs_entry["color_path"]
            rp = obs_entry["red_path"]
            tile = self.geo_dataset._load_tile(
                color_path=pathlib.Path(cp) if isinstance(cp, str) else None,
                red_path=pathlib.Path(rp) if isinstance(rp, str) else None,
                x=x_slice,
                y=y_slice,
            )
            if tile is not None:
                image = tile
                if self.geo_dataset.normalize and self.geo_dataset._normalizer is not None:
                    nodata_mask = image == 0.0
                    image = self.geo_dataset._normalizer(image)
                    image[nodata_mask] = 0.0
                return {
                    "image": image,
                    "bounds": self.geo_dataset._slice_to_tensor((x_slice, y_slice, t_slice)),
                    "crs": self.geo_dataset.crs.to_wkt(),
                    "loaded_from_dominant_obs_only": True,
                }

        sample = self.geo_dataset[(x_slice, y_slice, t_slice)]
        sample["loaded_from_dominant_obs_only"] = False
        return sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        patch_row = self.patch_records.iloc[index]
        obs_row = self.observation_metadata.loc[str(patch_row["dominant_obs_id"])]

        out_h, out_w = self.image_size
        x_step = (float(patch_row["x_stop"]) - float(patch_row["x_start"])) / float(out_w)
        y_step = (float(patch_row["y_stop"]) - float(patch_row["y_start"])) / float(out_h)

        sample = self._load_patch_sample(patch_row, x_step=x_step, y_step=y_step)

        image: torch.Tensor = sample["image"]
        valid_mask = (image > 1e-6).any(dim=0)
        band_valid_fraction = (image > 1e-6).float().mean(dim=(1, 2))
        overall_valid_fraction = float(valid_mask.float().mean())
        band_presence_mask = torch.tensor(
            [
                bool(patch_row["has_near_infrared"]),
                bool(patch_row["has_red"]),
                bool(patch_row["has_blue_green"]),
            ],
            dtype=torch.bool,
        )

        rationale_expanded = obs_row.get("rationale_expanded")
        if pd.isna(rationale_expanded):
            rationale_expanded = None

        out: dict[str, Any] = {
            "image": image,
            "valid_mask": valid_mask,
            "rationale_raw": str(obs_row["rationale_desc"]),
            "rationale_expanded": rationale_expanded,
            "location": torch.tensor(
                [float(patch_row["centroid_lon"]), float(patch_row["centroid_lat"])],
                dtype=torch.float32,
            ),
            "geo_features": _build_geo_features(patch_row),
            "scale_features": _build_patch_scale_features(patch_row, obs_row),
            "metadata": {
                "patch_id": str(patch_row["patch_id"]),
                "obs_id": str(patch_row["dominant_obs_id"]),
                "dominant_obs_id": str(patch_row["dominant_obs_id"]),
                "product_id": str(obs_row["product_id"]),
                "bounds": sample["bounds"],
                "crs": sample["crs"],
                "patch_bounds": (
                    float(patch_row["x_start"]),
                    float(patch_row["y_start"]),
                    float(patch_row["x_stop"]),
                    float(patch_row["y_stop"]),
                ),
                "viewing_features": _build_viewing_features(obs_row),
                "viewing_feature_names": VIEWING_FEATURE_NAMES,
                "geo_feature_names": GEO_FEATURE_NAMES,
                "scale_feature_names": PATCH_SCALE_FEATURE_NAMES,
                "band_presence_mask": band_presence_mask,
                "band_valid_fraction": band_valid_fraction,
                "overall_valid_fraction": overall_valid_fraction,
                "is_patch_valid": overall_valid_fraction >= self.min_valid_fraction,
                "min_valid_fraction": self.min_valid_fraction,
                "contributing_obs_ids": tuple(patch_row["contributing_obs_ids"]),
                "contributing_rationales": tuple(patch_row["contributing_rationales"]),
                "overlap_fractions": tuple(patch_row["overlap_fractions"]),
                "source_obs_count": int(patch_row["source_obs_count"]),
                "dominant_overlap_fraction": float(patch_row["dominant_overlap_fraction"]),
                "patch_size": self.patch_size,
                "image_size": self.image_size,
                "start_time": obs_row["start_time"],
                "stop_time": obs_row["stop_time"],
                "loaded_from_dominant_obs_only": bool(
                    sample.get("loaded_from_dominant_obs_only", False)
                ),
                "has_rationale_expanded": bool(obs_row.get("has_rationale_expanded", False)),
                "expansion_model": obs_row.get("expansion_model"),
                "prompt_version": obs_row.get("prompt_version"),
            },
        }
        if self.transforms is not None:
            out = self.transforms(out)
        return out
