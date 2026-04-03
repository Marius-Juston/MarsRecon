"""Paired local/global crop dataset and batching utilities for multimodal alignment."""

from __future__ import annotations

import pathlib
from ast import literal_eval
from collections.abc import Callable
from typing import Any

import pandas as pd
import torch
from shapely.geometry import box as shapely_box
from torch.utils.data import Dataset

from dataset.hirise_sampler import _to_tuple
from dataset.mars_hirise import ALL_CHANNELS, MarsHiRISE
from clip.marsclip_dataset import (
    GEO_FEATURE_NAMES,
    VIEWING_FEATURE_NAMES,
    _build_geo_features,
    _build_viewing_features,
)
from clip.marsclip_patches import (
    DEFAULT_PATCH_VALID_FRACTION,
    PATCH_SCALE_FEATURE_NAMES,
    _filter_geo_dataset_to_obs_ids,
    _filter_observation_metadata_to_color,
    _filter_patch_records_to_color,
    _normalize_image_size,
    _normalize_obs_id_set,
    build_patch_observation_metadata,
    build_patch_records,
)
from clip.marsclip_text import SimpleTextTokenizer, compose_rationale_text
from clip.rationale_cache import merge_rationale_cache


def _build_scale_features(
    obs_row: pd.Series,
    *,
    lon_span_deg: float,
    lat_span_deg: float,
    dominant_overlap_fraction: float,
    source_obs_count: int,
) -> torch.Tensor:
    """Build patch-scale features for either the local or global crop."""
    lon_span = max(1e-12, float(lon_span_deg))
    lat_span = max(1e-12, float(lat_span_deg))
    values = [
        float(obs_row["map_scale"]),
        float(obs_row["map_resolution"]),
        lon_span,
        lat_span,
        float(lon_span * lat_span),
        float(dominant_overlap_fraction),
        float(source_obs_count),
        float(lon_span / lat_span),
    ]
    return torch.tensor(values, dtype=torch.float32)


def _compute_crop_quality(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Compute valid-mask and valid fractions for one crop image."""
    valid_mask = (image > 1e-6).any(dim=0)
    band_valid_fraction = (image > 1e-6).float().mean(dim=(1, 2))
    overall_valid_fraction = float(valid_mask.float().mean())
    return valid_mask, band_valid_fraction, overall_valid_fraction


def _crop_request(
    *,
    x_start: float,
    x_stop: float,
    y_start: float,
    y_stop: float,
    t_start: pd.Timestamp,
    t_stop: pd.Timestamp,
    image_size: tuple[int, int],
) -> tuple[slice, slice, slice]:
    """Build a GeoDataset request tuple for one crop."""
    out_h, out_w = image_size
    x_step = (float(x_stop) - float(x_start)) / float(out_w)
    y_step = (float(y_stop) - float(y_start)) / float(out_h)
    return (
        slice(float(x_start), float(x_stop), x_step),
        slice(float(y_start), float(y_stop), y_step),
        slice(pd.Timestamp(t_start), pd.Timestamp(t_stop), 1),
    )


def _compute_overlap_metadata(
    geo_dataset: MarsHiRISE | Any,
    crop_geom: Any,
    *,
    obs_lookup: pd.DataFrame,
) -> dict[str, object] | None:
    """Summarize how a crop overlaps the available strip geometries."""
    try:
        spatial_index = geo_dataset.index.sindex
    except Exception:
        spatial_index = None

    if spatial_index is None:
        candidates = geo_dataset.index
    else:
        candidate_positions = list(spatial_index.intersection(crop_geom.bounds))
        if not candidate_positions:
            return None
        candidates = geo_dataset.index.iloc[candidate_positions]

    candidates = candidates[candidates.geometry.intersects(crop_geom)]
    if candidates.empty:
        return None

    overlaps: list[tuple[str, float]] = []
    for _, row in candidates.iterrows():
        overlap_area = row.geometry.intersection(crop_geom).area
        if overlap_area > 0:
            overlaps.append((str(row["obs_id"]), float(overlap_area)))

    if not overlaps:
        return None

    overlaps.sort(key=lambda item: item[1], reverse=True)
    total_overlap = sum(area for _, area in overlaps)
    contributing_obs_ids = tuple(obs_id for obs_id, _ in overlaps)
    overlap_fractions = tuple(area / total_overlap for _, area in overlaps)
    contributing_rationales = tuple(
        str(obs_lookup.loc[obs_id]["rationale_desc"])
        for obs_id in contributing_obs_ids
        if obs_id in obs_lookup.index
    )

    return {
        "dominant_obs_id": contributing_obs_ids[0],
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
    }


def _filter_paired_records_to_color(
    paired_records: pd.DataFrame,
    *,
    allowed_obs_ids: set[str],
) -> pd.DataFrame:
    """Keep only paired records whose local/global observations are COLOR-capable."""
    required = {
        "dominant_obs_id",
        "global_dominant_obs_id",
        "contributing_obs_ids",
        "global_contributing_obs_ids",
        "has_near_infrared",
        "has_blue_green",
        "global_has_near_infrared",
        "global_has_blue_green",
    }
    missing = required.difference(paired_records.columns)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(
            f"paired_records is missing required COLOR filter columns: {missing_names}"
        )

    def _all_allowed(values: object) -> bool:
        if isinstance(values, tuple | list):
            return all(str(value) in allowed_obs_ids for value in values)
        return False

    mask = (
        paired_records["dominant_obs_id"].astype(str).isin(allowed_obs_ids)
        & paired_records["global_dominant_obs_id"].astype(str).isin(allowed_obs_ids)
        & paired_records["has_near_infrared"].astype(bool)
        & paired_records["has_blue_green"].astype(bool)
        & paired_records["global_has_near_infrared"].astype(bool)
        & paired_records["global_has_blue_green"].astype(bool)
        & paired_records["contributing_obs_ids"].map(_all_allowed)
        & paired_records["global_contributing_obs_ids"].map(_all_allowed)
    )
    filtered = paired_records.loc[mask].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No paired crop records remain after color_only filtering.")
    return filtered


def build_paired_crop_records(
    geo_dataset: MarsHiRISE | Any,
    *,
    patch_records: pd.DataFrame,
    observation_metadata: pd.DataFrame,
    global_scale_factor: float = 3.0,
    require_same_dominant_obs: bool = True,
    min_global_dominant_overlap: float = 0.5,
) -> pd.DataFrame:
    """Augment Stage A patch records with paired multiscale global-crop metadata."""
    if global_scale_factor <= 1.0:
        raise ValueError("global_scale_factor must be greater than 1.0.")

    obs_lookup = observation_metadata.set_index("obs_id", drop=False)
    records: list[dict[str, object]] = []
    for _, patch_row in patch_records.iterrows():
        global_lon_span = float(patch_row["patch_lon_span_deg"]) * float(global_scale_factor)
        global_lat_span = float(patch_row["patch_lat_span_deg"]) * float(global_scale_factor)
        half_w = global_lon_span / 2.0
        half_h = global_lat_span / 2.0
        cx = float(patch_row["centroid_lon"])
        cy = float(patch_row["centroid_lat"])

        crop_geom = shapely_box(cx - half_w, cy - half_h, cx + half_w, cy + half_h)
        overlap = _compute_overlap_metadata(
            geo_dataset,
            crop_geom,
            obs_lookup=obs_lookup,
        )
        if overlap is None:
            continue
        if float(overlap["dominant_overlap_fraction"]) < float(min_global_dominant_overlap):
            continue
        if require_same_dominant_obs and str(overlap["dominant_obs_id"]) != str(
            patch_row["dominant_obs_id"]
        ):
            continue

        records.append(
            {
                **patch_row.to_dict(),
                "global_x_start": cx - half_w,
                "global_x_stop": cx + half_w,
                "global_y_start": cy - half_h,
                "global_y_stop": cy + half_h,
                "global_min_lon": cx - half_w,
                "global_max_lon": cx + half_w,
                "global_min_lat": cy - half_h,
                "global_max_lat": cy + half_h,
                "global_patch_lon_span_deg": global_lon_span,
                "global_patch_lat_span_deg": global_lat_span,
                "global_patch_area_deg2": float(global_lon_span * global_lat_span),
                "global_scale_factor": float(global_scale_factor),
                "global_dominant_obs_id": str(overlap["dominant_obs_id"]),
                "global_contributing_obs_ids": tuple(overlap["contributing_obs_ids"]),
                "global_contributing_rationales": tuple(overlap["contributing_rationales"]),
                "global_overlap_fractions": tuple(overlap["overlap_fractions"]),
                "global_dominant_overlap_fraction": float(
                    overlap["dominant_overlap_fraction"]
                ),
                "global_source_obs_count": int(overlap["source_obs_count"]),
                "global_has_near_infrared": bool(overlap["has_near_infrared"]),
                "global_has_red": bool(overlap["has_red"]),
                "global_has_blue_green": bool(overlap["has_blue_green"]),
            }
        )

    paired_records = pd.DataFrame.from_records(records)
    if paired_records.empty:
        raise ValueError("No valid paired crop records could be built for paired multiscale alignment.")
    return paired_records.reset_index(drop=True)


class MarsCLIPPairedCropDataset(Dataset):
    """Dataset returning aligned local/global crops plus shared context."""

    def __init__(
        self,
        geo_dataset: MarsHiRISE | Any | None = None,
        *,
        root: pathlib.Path | str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        patch_size: float | tuple[float, float] = 0.005,
        global_scale_factor: float = 3.0,
        stride: float | tuple[float, float] | None = None,
        image_size: int | tuple[int, int] = 224,
        min_geometry_overlap: float = 0.5,
        min_global_dominant_overlap: float = 0.5,
        min_valid_fraction: float = DEFAULT_PATCH_VALID_FRACTION,
        max_patches: int | None = None,
        generator: torch.Generator | None = None,
        color_only: bool = False,
        rationale_cache: pd.DataFrame | pathlib.Path | str | None = None,
        observation_metadata: pd.DataFrame | None = None,
        patch_records: pd.DataFrame | None = None,
        paired_records: pd.DataFrame | None = None,
        require_same_dominant_obs: bool = True,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        if geo_dataset is None:
            if root is None:
                raise ValueError("Either geo_dataset or root must be provided.")
            geo_dataset = MarsHiRISE(
                root=root,
                bbox=bbox,
                channels=list(ALL_CHANNELS),
                download=False,
            )

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

        if paired_records is None:
            paired_records = build_paired_crop_records(
                geo_dataset,
                patch_records=patch_records,
                observation_metadata=observation_metadata,
                global_scale_factor=global_scale_factor,
                require_same_dominant_obs=require_same_dominant_obs,
                min_global_dominant_overlap=min_global_dominant_overlap,
            )
        else:
            paired_records = paired_records.copy()
        if color_only:
            paired_records = _filter_paired_records_to_color(
                paired_records,
                allowed_obs_ids=allowed_obs_ids,
            )

        if paired_records.empty:
            raise ValueError("Paired crop record table is empty.")

        self.geo_dataset = geo_dataset
        self.patch_records = patch_records.reset_index(drop=True)
        self.paired_records = paired_records.reset_index(drop=True)
        self.observation_metadata = observation_metadata.set_index("obs_id", drop=False)
        self.image_size = _normalize_image_size(image_size)
        self.patch_size = _to_tuple(patch_size)
        self.global_scale_factor = float(global_scale_factor)
        self.global_patch_size = (
            self.patch_size[0] * self.global_scale_factor,
            self.patch_size[1] * self.global_scale_factor,
        )
        self.min_valid_fraction = float(min_valid_fraction)
        self.color_only = bool(color_only)
        self.require_same_dominant_obs = bool(require_same_dominant_obs)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.paired_records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair_row = self.paired_records.iloc[index]
        local_obs_row = self.observation_metadata.loc[str(pair_row["dominant_obs_id"])]
        global_obs_row = self.observation_metadata.loc[str(pair_row["global_dominant_obs_id"])]

        local_sample = self.geo_dataset[
            _crop_request(
                x_start=float(pair_row["x_start"]),
                x_stop=float(pair_row["x_stop"]),
                y_start=float(pair_row["y_start"]),
                y_stop=float(pair_row["y_stop"]),
                t_start=pd.Timestamp(pair_row["t_start"]),
                t_stop=pd.Timestamp(pair_row["t_stop"]),
                image_size=self.image_size,
            )
        ]
        global_sample = self.geo_dataset[
            _crop_request(
                x_start=float(pair_row["global_x_start"]),
                x_stop=float(pair_row["global_x_stop"]),
                y_start=float(pair_row["global_y_start"]),
                y_stop=float(pair_row["global_y_stop"]),
                t_start=pd.Timestamp(pair_row["t_start"]),
                t_stop=pd.Timestamp(pair_row["t_stop"]),
                image_size=self.image_size,
            )
        ]

        local_image: torch.Tensor = local_sample["image"]
        global_image: torch.Tensor = global_sample["image"]

        local_valid_mask, local_band_valid_fraction, local_overall_valid_fraction = (
            _compute_crop_quality(local_image)
        )
        global_valid_mask, global_band_valid_fraction, global_overall_valid_fraction = (
            _compute_crop_quality(global_image)
        )

        local_band_presence_mask = torch.tensor(
            [
                bool(pair_row["has_near_infrared"]),
                bool(pair_row["has_red"]),
                bool(pair_row["has_blue_green"]),
            ],
            dtype=torch.bool,
        )
        global_band_presence_mask = torch.tensor(
            [
                bool(pair_row["global_has_near_infrared"]),
                bool(pair_row["global_has_red"]),
                bool(pair_row["global_has_blue_green"]),
            ],
            dtype=torch.bool,
        )

        rationale_expanded = local_obs_row.get("rationale_expanded")
        if pd.isna(rationale_expanded):
            rationale_expanded = None

        local_scale_features = _build_scale_features(
            local_obs_row,
            lon_span_deg=float(pair_row["patch_lon_span_deg"]),
            lat_span_deg=float(pair_row["patch_lat_span_deg"]),
            dominant_overlap_fraction=float(pair_row["dominant_overlap_fraction"]),
            source_obs_count=int(pair_row["source_obs_count"]),
        )
        global_scale_features = _build_scale_features(
            global_obs_row,
            lon_span_deg=float(pair_row["global_patch_lon_span_deg"]),
            lat_span_deg=float(pair_row["global_patch_lat_span_deg"]),
            dominant_overlap_fraction=float(pair_row["global_dominant_overlap_fraction"]),
            source_obs_count=int(pair_row["global_source_obs_count"]),
        )

        out: dict[str, Any] = {
            "local_image": local_image,
            "local_valid_mask": local_valid_mask,
            "global_image": global_image,
            "global_valid_mask": global_valid_mask,
            "rationale_raw": str(local_obs_row["rationale_desc"]),
            "rationale_expanded": rationale_expanded,
            "location": torch.tensor(
                [float(pair_row["centroid_lon"]), float(pair_row["centroid_lat"])],
                dtype=torch.float32,
            ),
            "geo_features": _build_geo_features(pair_row),
            "local_scale_features": local_scale_features,
            "global_scale_features": global_scale_features,
            "metadata": {
                "pair_id": str(pair_row["patch_id"]),
                "patch_id": str(pair_row["patch_id"]),
                "obs_id": str(pair_row["dominant_obs_id"]),
                "dominant_obs_id": str(pair_row["dominant_obs_id"]),
                "global_dominant_obs_id": str(pair_row["global_dominant_obs_id"]),
                "product_id": str(local_obs_row["product_id"]),
                "global_product_id": str(global_obs_row["product_id"]),
                "local_bounds": local_sample["bounds"],
                "global_bounds": global_sample["bounds"],
                "crs": local_sample["crs"],
                "viewing_features": _build_viewing_features(local_obs_row),
                "global_viewing_features": _build_viewing_features(global_obs_row),
                "viewing_feature_names": VIEWING_FEATURE_NAMES,
                "geo_feature_names": GEO_FEATURE_NAMES,
                "scale_feature_names": PATCH_SCALE_FEATURE_NAMES,
                "local_band_presence_mask": local_band_presence_mask,
                "global_band_presence_mask": global_band_presence_mask,
                "local_band_valid_fraction": local_band_valid_fraction,
                "global_band_valid_fraction": global_band_valid_fraction,
                "local_overall_valid_fraction": local_overall_valid_fraction,
                "global_overall_valid_fraction": global_overall_valid_fraction,
                "is_local_valid": local_overall_valid_fraction >= self.min_valid_fraction,
                "is_global_valid": global_overall_valid_fraction >= self.min_valid_fraction,
                "is_pair_valid": (
                    local_overall_valid_fraction >= self.min_valid_fraction
                    and global_overall_valid_fraction >= self.min_valid_fraction
                ),
                "min_valid_fraction": self.min_valid_fraction,
                "contributing_obs_ids": tuple(pair_row["contributing_obs_ids"]),
                "contributing_rationales": tuple(pair_row["contributing_rationales"]),
                "overlap_fractions": tuple(pair_row["overlap_fractions"]),
                "source_obs_count": int(pair_row["source_obs_count"]),
                "dominant_overlap_fraction": float(pair_row["dominant_overlap_fraction"]),
                "global_contributing_obs_ids": tuple(pair_row["global_contributing_obs_ids"]),
                "global_contributing_rationales": tuple(
                    pair_row["global_contributing_rationales"]
                ),
                "global_overlap_fractions": tuple(pair_row["global_overlap_fractions"]),
                "global_source_obs_count": int(pair_row["global_source_obs_count"]),
                "global_dominant_overlap_fraction": float(
                    pair_row["global_dominant_overlap_fraction"]
                ),
                "patch_size": self.patch_size,
                "global_patch_size": self.global_patch_size,
                "image_size": self.image_size,
                "global_scale_factor": self.global_scale_factor,
                "start_time": local_obs_row["start_time"],
                "stop_time": local_obs_row["stop_time"],
                "has_rationale_expanded": bool(local_obs_row.get("has_rationale_expanded", False)),
                "expansion_model": local_obs_row.get("expansion_model"),
                "prompt_version": local_obs_row.get("prompt_version"),
            },
        }
        if self.transforms is not None:
            out = self.transforms(out)
        return out


class MarsCLIPPairedBatchCollator:
    """Batch paired local/global crop samples and tokenize their shared text."""

    def __init__(
        self,
        tokenizer: SimpleTextTokenizer,
        *,
        text_mode: str = "raw_plus_expanded",
        max_length: int = 64,
    ) -> None:
        self.tokenizer = tokenizer
        self.text_mode = text_mode
        self.max_length = max_length

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [
            compose_rationale_text(
                sample["rationale_raw"],
                sample.get("rationale_expanded"),
                text_mode=self.text_mode,
            )
            for sample in samples
        ]
        input_ids, attention_mask = self.tokenizer.batch_encode(
            texts,
            max_length=self.max_length,
        )

        local_images = torch.stack([sample["local_image"] for sample in samples])
        local_valid_masks = torch.stack([sample["local_valid_mask"] for sample in samples])
        global_images = torch.stack([sample["global_image"] for sample in samples])
        global_valid_masks = torch.stack([sample["global_valid_mask"] for sample in samples])
        locations = torch.stack([sample["location"] for sample in samples])
        geo_features = torch.stack([sample["geo_features"] for sample in samples])
        local_scale_features = torch.stack(
            [sample["local_scale_features"] for sample in samples]
        )
        global_scale_features = torch.stack(
            [sample["global_scale_features"] for sample in samples]
        )
        viewing_features = torch.stack(
            [sample["metadata"]["viewing_features"] for sample in samples]
        )
        local_band_presence_mask = torch.stack(
            [sample["metadata"]["local_band_presence_mask"] for sample in samples]
        ).to(torch.float32)
        global_band_presence_mask = torch.stack(
            [sample["metadata"]["global_band_presence_mask"] for sample in samples]
        ).to(torch.float32)
        local_band_valid_fraction = torch.stack(
            [sample["metadata"]["local_band_valid_fraction"] for sample in samples]
        )
        global_band_valid_fraction = torch.stack(
            [sample["metadata"]["global_band_valid_fraction"] for sample in samples]
        )
        local_overall_valid_fraction = torch.tensor(
            [
                [float(sample["metadata"]["local_overall_valid_fraction"])]
                for sample in samples
            ],
            dtype=torch.float32,
        )
        global_overall_valid_fraction = torch.tensor(
            [
                [float(sample["metadata"]["global_overall_valid_fraction"])]
                for sample in samples
            ],
            dtype=torch.float32,
        )
        local_quality_features = torch.cat(
            [
                local_band_presence_mask,
                local_band_valid_fraction,
                local_overall_valid_fraction,
            ],
            dim=1,
        )
        global_quality_features = torch.cat(
            [
                global_band_presence_mask,
                global_band_valid_fraction,
                global_overall_valid_fraction,
            ],
            dim=1,
        )

        return {
            "local_image": local_images,
            "local_valid_mask": local_valid_masks,
            "global_image": global_images,
            "global_valid_mask": global_valid_masks,
            "location": locations,
            "geo_features": geo_features,
            "local_scale_features": local_scale_features,
            "global_scale_features": global_scale_features,
            "viewing_features": viewing_features,
            "local_quality_features": local_quality_features,
            "global_quality_features": global_quality_features,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "rationale_raw": [sample["rationale_raw"] for sample in samples],
            "rationale_expanded": [sample.get("rationale_expanded") for sample in samples],
            "text": texts,
            "metadata": [sample["metadata"] for sample in samples],
        }


def save_paired_crop_records(
    paired_records: pd.DataFrame,
    path: pathlib.Path | str,
) -> pathlib.Path:
    """Persist paired multiscale crop records for reuse across runs."""
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        paired_records.to_pickle(out)
    elif suffix == ".parquet":
        paired_records.to_parquet(out, index=False)
    else:
        paired_records.to_csv(out, index=False)
    return out


def load_paired_crop_records(path: pathlib.Path | str) -> pd.DataFrame:
    """Load cached paired multiscale crop records from disk."""
    source = pathlib.Path(path)
    suffix = source.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        paired_records = pd.read_pickle(source)
    elif suffix == ".parquet":
        paired_records = pd.read_parquet(source)
    else:
        paired_records = pd.read_csv(source)
        tuple_columns = (
            "contributing_obs_ids",
            "contributing_rationales",
            "overlap_fractions",
            "global_contributing_obs_ids",
            "global_contributing_rationales",
            "global_overlap_fractions",
        )
        for column in tuple_columns:
            if column in paired_records.columns:
                paired_records[column] = paired_records[column].map(literal_eval)
    return paired_records
