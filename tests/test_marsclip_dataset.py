"""Tests for the observation-level MarsCLIP dataset."""

from __future__ import annotations

import pathlib
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import rasterio

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch

from clip.marsclip_dataset import (
    GEO_FEATURE_NAMES,
    SCALE_FEATURE_NAMES,
    VIEWING_FEATURE_NAMES,
    MarsCLIPDataset,
    _build_geo_features,
    _build_scale_features,
    _build_viewing_features,
    _load_color_thumbnail,
)

_MARS_RCRS = rasterio.crs.CRS.from_proj4("+proj=longlat +a=3396190 +b=3376200 +no_defs")


def _write_lbl(path: pathlib.Path, scaling_factor: float = 1e-3, offset: float = 0.0) -> None:
    path.write_text(
        "\n".join(
            [
                "PDS_VERSION_ID = PDS3",
                f"SCALING_FACTOR = {scaling_factor}",
                f"OFFSET = {offset}",
                "SAMPLE_BITS = 16",
                "SAMPLE_BIT_MASK = 2#0000001111111111#",
                "BANDS = 3",
                'FILTER_NAME = ("NEAR-INFRARED", "RED", "BLUE-GREEN")',
                "END",
            ]
        )
    )


def _write_tiff(path: pathlib.Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = rasterio.transform.from_bounds(-131.0, 18.0, -130.0, 19.0, data.shape[2], data.shape[1])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=data.shape[0],
        dtype=str(data.dtype),
        width=data.shape[2],
        height=data.shape[1],
        crs=_MARS_RCRS,
        transform=transform,
    ) as dst:
        dst.write(data)


def _manifest_row(image_path: pathlib.Path | None) -> pd.DataFrame:
    image_format = image_path.suffix.lower().lstrip(".") if image_path else None
    return pd.DataFrame(
        [
            {
                "obs_id": "OBS_A",
                "product_id": "OBS_A_COLOR",
                "image_path": str(image_path) if image_path else None,
                "image_format": image_format,
                "color_jp2_path": None,
                "color_tif_path": str(image_path) if image_path and image_format == "tif" else None,
                "has_local_jp2": False,
                "has_local_tif": bool(image_path and image_format == "tif"),
                "has_local_image": image_path is not None,
                "rationale_desc": "Olympus Mons lava channels",
                "start_time": pd.Timestamp("2007-01-01T00:00:00Z"),
                "stop_time": pd.Timestamp("2007-01-01T00:01:00Z"),
                "min_lon": -136.0,
                "max_lon": -124.0,
                "min_lat": 10.0,
                "max_lat": 12.0,
                "centroid_lon": -130.0,
                "centroid_lat": 11.0,
                "lon_span_deg": 12.0,
                "lat_span_deg": 2.0,
                "bbox_area_deg2": 24.0,
                "image_lines": 4,
                "line_samples": 4,
                "map_scale": 0.5,
                "map_resolution": 1000.0,
                "emission_angle": 7.0,
                "incidence_angle": 74.0,
                "phase_angle": 78.0,
                "local_time": 15.0,
                "solar_longitude": 90.0,
                "sub_solar_azimuth": 180.0,
                "north_azimuth": 270.0,
                "spacecraft_altitude": 300.0,
                "stereo_flag": "YES",
                "is_stereo": True,
                "projection_center_latitude": 11.0,
                "projection_center_longitude": -130.0,
                "has_near_infrared": True,
                "has_red": True,
                "has_blue_green": True,
                "channel_count": 3,
            }
        ]
    )


def test_feature_builders_return_expected_lengths():
    row = _manifest_row(None).iloc[0]

    geo = _build_geo_features(row)
    scale = _build_scale_features(row)
    viewing = _build_viewing_features(row)

    assert geo.shape == (len(GEO_FEATURE_NAMES),)
    assert scale.shape == (len(SCALE_FEATURE_NAMES),)
    assert viewing.shape == (len(VIEWING_FEATURE_NAMES),)
    assert scale[-1] == pytest.approx(1.0)


def test_dataset_returns_expected_keys_and_calibrated_thumbnail(tmp_path):
    path = tmp_path / "images" / "OBS_A_COLOR.tif"
    data = np.ones((3, 4, 4), dtype=np.uint16) * 500
    data[:, :2, :2] = 0
    _write_tiff(path, data)
    _write_lbl(path.with_suffix(".LBL"), scaling_factor=1e-3, offset=0.0)

    ds = MarsCLIPDataset(manifest=_manifest_row(path), image_size=4)
    sample = ds[0]

    assert set(sample) == {
        "image",
        "valid_mask",
        "rationale_raw",
        "rationale_expanded",
        "geo_features",
        "scale_features",
        "metadata",
    }
    assert sample["image"].shape == (3, 4, 4)
    assert sample["image"].dtype == torch.float32
    assert sample["valid_mask"].shape == (4, 4)
    assert sample["valid_mask"].dtype == torch.bool
    assert sample["rationale_raw"] == "Olympus Mons lava channels"
    assert sample["rationale_expanded"] is None
    assert sample["geo_features"].shape == (len(GEO_FEATURE_NAMES),)
    assert sample["scale_features"].shape == (len(SCALE_FEATURE_NAMES),)

    assert torch.all(sample["image"][:, :2, :2] == 0.0)
    assert torch.allclose(sample["image"][:, 3, 3], torch.tensor([0.5, 0.5, 0.5]))
    assert not bool(sample["valid_mask"][0, 0])
    assert bool(sample["valid_mask"][3, 3])

    md = sample["metadata"]
    assert md["obs_id"] == "OBS_A"
    assert md["image_path"] == str(path)
    assert md["viewing_features"].shape == (len(VIEWING_FEATURE_NAMES),)
    assert md["band_presence_mask"].tolist() == [True, True, True]
    assert torch.allclose(md["band_valid_fraction"], torch.tensor([0.75, 0.75, 0.75]))
    assert md["overall_valid_fraction"] == pytest.approx(0.75)
    assert not md["has_rationale_expanded"]


def test_dataset_pads_narrower_band_count_to_three(tmp_path):
    path = tmp_path / "images" / "OBS_A_COLOR.tif"
    data = np.ones((1, 4, 4), dtype=np.uint16) * 500
    _write_tiff(path, data)
    _write_lbl(path.with_suffix(".LBL"), scaling_factor=1e-3, offset=0.0)

    ds = MarsCLIPDataset(manifest=_manifest_row(path), image_size=4)
    sample = ds[0]

    assert sample["image"].shape == (3, 4, 4)
    assert torch.allclose(sample["image"][0], torch.full((4, 4), 0.5))
    assert torch.all(sample["image"][1:] == 0.0)
    assert sample["metadata"]["band_presence_mask"].tolist() == [True, False, False]
    assert torch.allclose(
        sample["metadata"]["band_valid_fraction"],
        torch.tensor([1.0, 0.0, 0.0]),
    )
    assert sample["metadata"]["overall_valid_fraction"] == pytest.approx(1.0)


def test_dataset_raises_when_image_missing():
    ds = MarsCLIPDataset(manifest=_manifest_row(None), require_local_image=False)
    with pytest.raises(FileNotFoundError, match="has no local image_path"):
        _ = ds[0]


def test_dataset_can_build_manifest_from_root(tmp_path):
    manifest = _manifest_row(tmp_path / "images" / "OBS_A_COLOR.tif")
    with patch("clip.marsclip_dataset.build_observation_manifest", return_value=manifest) as mock_build:
        ds = MarsCLIPDataset(root=tmp_path)

    assert len(ds) == 1
    mock_build.assert_called_once()


def test_dataset_merges_optional_rationale_cache(tmp_path):
    path = tmp_path / "images" / "OBS_A_COLOR.tif"
    data = np.ones((3, 4, 4), dtype=np.uint16) * 500
    _write_tiff(path, data)
    _write_lbl(path.with_suffix(".LBL"), scaling_factor=1e-3, offset=0.0)

    cache = pd.DataFrame(
        [
            {
                "obs_id": "OBS_A",
                "rationale_raw": "Olympus Mons lava channels",
                "rationale_expanded": "Expanded geology text",
                "expansion_model": "mock-llm",
                "prompt_version": "v1",
                "prompt_template": "tmpl",
                "expansion_status": "ok",
                "expansion_error": None,
                "expansion_timestamp": pd.Timestamp("2026-03-29T00:00:00Z"),
            }
        ]
    )

    ds = MarsCLIPDataset(
        manifest=_manifest_row(path),
        image_size=4,
        rationale_cache=cache,
    )
    sample = ds[0]

    assert sample["rationale_expanded"] == "Expanded geology text"
    assert sample["metadata"]["has_rationale_expanded"]
    assert sample["metadata"]["expansion_model"] == "mock-llm"
    assert sample["metadata"]["prompt_version"] == "v1"


@pytest.mark.integration
def test_real_dataset_returns_observation_sample():
    ds = MarsCLIPDataset(
        root="/scratch/mars_hirise",
        bbox=(-136.0, 12.0, -124.0, 24.0),
        image_size=64,
    )
    sample = ds[0]

    assert sample["image"].shape == (3, 64, 64)
    assert sample["valid_mask"].shape == (64, 64)
    assert sample["rationale_raw"]
    assert sample["metadata"]["overall_valid_fraction"] > 0.0


def test_load_color_thumbnail_raises_for_nonexistent_path(tmp_path):
    missing = tmp_path / "missing.tif"
    with pytest.raises(FileNotFoundError, match="Image not found"):
        _load_color_thumbnail(missing, image_size=4)


def test_dataset_raises_when_both_manifest_and_root_are_none():
    with pytest.raises(ValueError, match="Either manifest or root must be provided"):
        MarsCLIPDataset(manifest=None, root=None)


def test_dataset_raises_when_manifest_empty_after_filtering():
    manifest = _manifest_row(None)
    manifest["has_local_image"] = False
    with pytest.raises(ValueError, match="Manifest is empty after filtering"):
        MarsCLIPDataset(manifest=manifest, require_local_image=True)


def test_dataset_applies_transforms_to_sample(tmp_path):
    path = tmp_path / "images" / "OBS_A_COLOR.tif"
    data = np.ones((3, 4, 4), dtype=np.uint16) * 500
    _write_tiff(path, data)
    _write_lbl(path.with_suffix(".LBL"))

    transformed_keys = []

    def _add_key(sample):
        sample["extra_key"] = "added"
        transformed_keys.append(True)
        return sample

    ds = MarsCLIPDataset(manifest=_manifest_row(path), image_size=4, transforms=_add_key)
    sample = ds[0]

    assert "extra_key" in sample
    assert sample["extra_key"] == "added"
    assert len(transformed_keys) == 1
