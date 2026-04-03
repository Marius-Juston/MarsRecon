"""Observation-level dataset for MarsCLIP-style pretraining."""

from __future__ import annotations

import math
import pathlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.enums import Resampling
from torch.utils.data import Dataset

from dataset.mars_hirise_base import ProductMeta
from clip.observation_manifest import build_observation_manifest
from clip.rationale_cache import merge_rationale_cache

GEO_FEATURE_NAMES: tuple[str, ...] = (
    "centroid_lon_sin",
    "centroid_lon_cos",
    "centroid_lat_sin",
    "centroid_lat_cos",
    "min_lon_norm",
    "max_lon_norm",
    "min_lat_norm",
    "max_lat_norm",
)

SCALE_FEATURE_NAMES: tuple[str, ...] = (
    "map_scale",
    "map_resolution",
    "log_image_lines",
    "log_line_samples",
    "lon_span_deg",
    "lat_span_deg",
    "bbox_area_deg2",
    "line_sample_aspect",
)

VIEWING_FEATURE_NAMES: tuple[str, ...] = (
    "emission_angle_norm",
    "incidence_angle_norm",
    "phase_angle_norm",
    "local_time_sin",
    "local_time_cos",
    "solar_longitude_sin",
    "solar_longitude_cos",
    "sub_solar_azimuth_sin",
    "sub_solar_azimuth_cos",
    "north_azimuth_sin",
    "north_azimuth_cos",
    "spacecraft_altitude_km",
    "is_stereo",
)


def _cyclic_features(value: float, period: float) -> tuple[float, float]:
    """Encode a periodic scalar using sine/cosine features."""
    angle = 2.0 * math.pi * (float(value) / period)
    return math.sin(angle), math.cos(angle)


def _build_geo_features(row: pd.Series) -> torch.Tensor:
    """Create location features from one observation manifest row."""
    lon_sin, lon_cos = _cyclic_features(float(row["centroid_lon"]) + 180.0, 360.0)
    lat_sin, lat_cos = _cyclic_features(float(row["centroid_lat"]) + 90.0, 180.0)
    values = [
        lon_sin,
        lon_cos,
        lat_sin,
        lat_cos,
        float(row["min_lon"]) / 180.0,
        float(row["max_lon"]) / 180.0,
        float(row["min_lat"]) / 90.0,
        float(row["max_lat"]) / 90.0,
    ]
    return torch.tensor(values, dtype=torch.float32)


def _build_scale_features(row: pd.Series) -> torch.Tensor:
    """Create scale / footprint features from one observation manifest row."""
    image_lines = max(1.0, float(row["image_lines"]))
    line_samples = max(1.0, float(row["line_samples"]))
    values = [
        float(row["map_scale"]),
        float(row["map_resolution"]),
        math.log1p(image_lines),
        math.log1p(line_samples),
        float(row["lon_span_deg"]),
        float(row["lat_span_deg"]),
        float(row["bbox_area_deg2"]),
        line_samples / image_lines,
    ]
    return torch.tensor(values, dtype=torch.float32)


def _build_viewing_features(row: pd.Series) -> torch.Tensor:
    """Create viewing / acquisition features from one observation manifest row."""
    local_time_sin, local_time_cos = _cyclic_features(float(row["local_time"]), 24.0)
    solar_lon_sin, solar_lon_cos = _cyclic_features(
        float(row["solar_longitude"]), 360.0
    )
    sub_solar_sin, sub_solar_cos = _cyclic_features(
        float(row["sub_solar_azimuth"]), 360.0
    )
    north_sin, north_cos = _cyclic_features(float(row["north_azimuth"]), 360.0)
    values = [
        float(row["emission_angle"]) / 180.0,
        float(row["incidence_angle"]) / 180.0,
        float(row["phase_angle"]) / 180.0,
        local_time_sin,
        local_time_cos,
        solar_lon_sin,
        solar_lon_cos,
        sub_solar_sin,
        sub_solar_cos,
        north_sin,
        north_cos,
        float(row["spacecraft_altitude"]),
        1.0 if bool(row["is_stereo"]) else 0.0,
    ]
    return torch.tensor(values, dtype=torch.float32)


def _load_color_thumbnail(
    image_path: pathlib.Path,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a calibrated COLOR thumbnail and associated quality masks."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    lbl_path = image_path.with_suffix(".LBL")
    meta = ProductMeta.from_lbl(lbl_path)

    with rasterio.open(image_path) as src:
        band_count = min(max(1, src.count), 3)
        data = src.read(
            indexes=list(range(1, band_count + 1)),
            out_shape=(band_count, image_size, image_size),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

    nodata_mask = data == 0.0
    data *= meta.scaling_factor
    data += meta.offset
    np.clip(data, 0.0, 1.0, out=data)
    data[nodata_mask] = 0.0

    if band_count < 3:
        padded = np.zeros((3, image_size, image_size), dtype=np.float32)
        padded[:band_count] = data
        data = padded

    image = torch.from_numpy(data)
    valid_mask = (image > 1e-6).any(dim=0)

    band_presence_mask = torch.zeros(3, dtype=torch.bool)
    band_presence_mask[:band_count] = True
    band_valid_fraction = (image > 1e-6).float().mean(dim=(1, 2))

    return image, valid_mask, band_presence_mask, band_valid_fraction


class MarsCLIPDataset(Dataset):
    """Observation-level dataset returning MarsCLIP v1 inputs."""

    def __init__(
        self,
        manifest: pd.DataFrame | None = None,
        *,
        root: pathlib.Path | str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        image_size: int = 224,
        require_local_image: bool = True,
        prefer_cog: bool = True,
        rationale_cache: pd.DataFrame | pathlib.Path | str | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        if manifest is None:
            if root is None:
                raise ValueError("Either manifest or root must be provided.")
            manifest = build_observation_manifest(
                root,
                bbox=bbox,
                require_local_image=require_local_image,
                prefer_cog=prefer_cog,
            )
        else:
            manifest = manifest.copy()
            if require_local_image and "has_local_image" in manifest.columns:
                mask = manifest["has_local_image"].fillna(False).astype(bool)
                manifest = manifest[mask].copy()
            if require_local_image and "image_path" in manifest.columns:
                manifest = manifest[manifest["image_path"].notna()].copy()

        if manifest.empty:
            raise ValueError("Manifest is empty after filtering.")

        if rationale_cache is not None:
            manifest = merge_rationale_cache(manifest, rationale_cache)
        elif "rationale_expanded" not in manifest.columns:
            manifest = manifest.copy()
            manifest["rationale_expanded"] = None
            manifest["has_rationale_expanded"] = False

        self.manifest = manifest.reset_index(drop=True)
        self.image_size = int(image_size)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        image_path_str = row["image_path"]
        if image_path_str is None:
            raise FileNotFoundError(
                f"Observation {row['obs_id']} has no local image_path."
            )

        image_path = pathlib.Path(str(image_path_str))
        image, valid_mask, band_presence_mask, band_valid_fraction = _load_color_thumbnail(
            image_path=image_path,
            image_size=self.image_size,
        )

        geo_features = _build_geo_features(row)
        scale_features = _build_scale_features(row)
        viewing_features = _build_viewing_features(row)
        overall_valid_fraction = float(valid_mask.float().mean())
        rationale_expanded = row.get("rationale_expanded")
        if pd.isna(rationale_expanded):
            rationale_expanded = None

        sample: dict[str, Any] = {
            "image": image,
            "valid_mask": valid_mask,
            "rationale_raw": str(row["rationale_desc"]),
            "rationale_expanded": rationale_expanded,
            "geo_features": geo_features,
            "scale_features": scale_features,
            "metadata": {
                "obs_id": str(row["obs_id"]),
                "product_id": str(row["product_id"]),
                "image_path": str(image_path),
                "image_format": row["image_format"],
                "start_time": row["start_time"],
                "stop_time": row["stop_time"],
                "viewing_features": viewing_features,
                "viewing_feature_names": VIEWING_FEATURE_NAMES,
                "geo_feature_names": GEO_FEATURE_NAMES,
                "scale_feature_names": SCALE_FEATURE_NAMES,
                "band_presence_mask": band_presence_mask,
                "band_valid_fraction": band_valid_fraction,
                "overall_valid_fraction": overall_valid_fraction,
                "channel_count": int(row.get("channel_count", 3)),
                "image_size": self.image_size,
                "has_local_image": bool(row.get("has_local_image", True)),
                "has_rationale_expanded": bool(row.get("has_rationale_expanded", False)),
                "expansion_model": row.get("expansion_model"),
                "prompt_version": row.get("prompt_version"),
            },
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
