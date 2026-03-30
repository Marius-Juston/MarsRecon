"""Tests for the patch-level MarsCLIP Stage A dataset bridge."""

from __future__ import annotations

import pathlib
import sys
import types

import pandas as pd
import pytest
import torch
from shapely.geometry import box

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from helpers import make_mock_dataset
from marsclip_dataset import GEO_FEATURE_NAMES, VIEWING_FEATURE_NAMES
from marsclip_patches import (
    DEFAULT_PATCH_VALID_FRACTION,
    PATCH_SCALE_FEATURE_NAMES,
    MarsCLIPPatchDataset,
    build_patch_observation_metadata,
    build_patch_records,
    load_patch_records,
    save_patch_records,
    summarize_patch_records,
    summarize_patch_samples,
)


def _raw_index_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "PRODUCT_ID": "OBS_A_COLOR",
                "RATIONALE_DESC": " Olympus  Mons   lava channels ",
                "START_TIME": "2007-01-01T00:00:00Z",
                "STOP_TIME": "2007-01-01T00:01:00Z",
                "MINIMUM_LONGITUDE": 224.0,
                "MAXIMUM_LONGITUDE": 236.0,
                "MINIMUM_LATITUDE": 10.0,
                "MAXIMUM_LATITUDE": 12.0,
                "IMAGE_LINES": 100,
                "LINE_SAMPLES": 200,
                "MAP_SCALE": 0.5,
                "MAP_RESOLUTION": 1000.0,
                "EMISSION_ANGLE": 7.0,
                "INCIDENCE_ANGLE": 74.0,
                "PHASE_ANGLE": 78.0,
                "LOCAL_TIME": 15.0,
                "SOLAR_LONGITUDE": 90.0,
                "SUB_SOLAR_AZIMUTH": 180.0,
                "NORTH_AZIMUTH": 270.0,
                "SPACECRAFT_ALTITUDE": 300.0,
                "STEREO_FLAG": "YES",
            },
            {
                "PRODUCT_ID": "OBS_A_RED",
                "RATIONALE_DESC": "Olympus Mons lava channels",
                "START_TIME": "2007-01-01T00:00:00Z",
                "STOP_TIME": "2007-01-01T00:01:00Z",
                "MINIMUM_LONGITUDE": 224.0,
                "MAXIMUM_LONGITUDE": 236.0,
                "MINIMUM_LATITUDE": 10.0,
                "MAXIMUM_LATITUDE": 12.0,
                "IMAGE_LINES": 100,
                "LINE_SAMPLES": 200,
                "MAP_SCALE": 0.25,
                "MAP_RESOLUTION": 1000.0,
                "EMISSION_ANGLE": 7.0,
                "INCIDENCE_ANGLE": 74.0,
                "PHASE_ANGLE": 78.0,
                "LOCAL_TIME": 15.0,
                "SOLAR_LONGITUDE": 90.0,
                "SUB_SOLAR_AZIMUTH": 180.0,
                "NORTH_AZIMUTH": 270.0,
                "SPACECRAFT_ALTITUDE": 300.0,
                "STEREO_FLAG": "YES",
            },
            {
                "PRODUCT_ID": "OBS_B_RED",
                "RATIONALE_DESC": "Red only strip",
                "START_TIME": "2007-01-02T00:00:00Z",
                "STOP_TIME": "2007-01-02T00:01:00Z",
                "MINIMUM_LONGITUDE": 224.0,
                "MAXIMUM_LONGITUDE": 230.0,
                "MINIMUM_LATITUDE": 11.0,
                "MAXIMUM_LATITUDE": 13.0,
                "IMAGE_LINES": 90,
                "LINE_SAMPLES": 180,
                "MAP_SCALE": 0.25,
                "MAP_RESOLUTION": 1000.0,
                "EMISSION_ANGLE": 8.0,
                "INCIDENCE_ANGLE": 73.0,
                "PHASE_ANGLE": 79.0,
                "LOCAL_TIME": 14.0,
                "SOLAR_LONGITUDE": 91.0,
                "SUB_SOLAR_AZIMUTH": 181.0,
                "NORTH_AZIMUTH": 270.0,
                "SPACECRAFT_ALTITUDE": 301.0,
                "STEREO_FLAG": "NO",
            },
        ]
    )


def test_build_patch_observation_metadata_prefers_color_and_keeps_red_only():
    fake = types.SimpleNamespace(_raw_index=_raw_index_rows())

    metadata = build_patch_observation_metadata(fake)
    metadata = metadata.set_index("obs_id")

    assert set(metadata.index) == {"OBS_A", "OBS_B"}

    obs_a = metadata.loc["OBS_A"]
    assert obs_a["rationale_desc"] == "Olympus Mons lava channels"
    assert bool(obs_a["has_near_infrared"])
    assert bool(obs_a["has_red"])
    assert bool(obs_a["has_blue_green"])
    assert int(obs_a["channel_count"]) == 3
    assert float(obs_a["map_scale"]) == pytest.approx(0.5)

    obs_b = metadata.loc["OBS_B"]
    assert not bool(obs_b["has_near_infrared"])
    assert bool(obs_b["has_red"])
    assert not bool(obs_b["has_blue_green"])
    assert int(obs_b["channel_count"]) == 1


def test_build_patch_records_tracks_dominant_observation_and_overlap(mars_crs):
    geometries = [
        box(-1.0, 0.0, 1.0, 2.0),
        box(0.0, 0.0, 2.0, 2.0),
    ]
    dataset = make_mock_dataset(geometries, mars_crs)
    interval = dataset.index.index[0]

    observation_metadata = pd.DataFrame(
        [
            {
                "obs_id": "obs_0",
                "product_id": "OBS0_RED",
                "rationale_desc": "Left strip",
                "rationale_expanded": None,
                "has_rationale_expanded": False,
                "has_near_infrared": False,
                "has_red": True,
                "has_blue_green": False,
                "channel_count": 1,
                "map_scale": 0.25,
                "map_resolution": 1000.0,
                "emission_angle": 1.0,
                "incidence_angle": 2.0,
                "phase_angle": 3.0,
                "local_time": 4.0,
                "solar_longitude": 5.0,
                "sub_solar_azimuth": 6.0,
                "north_azimuth": 270.0,
                "spacecraft_altitude": 300.0,
                "start_time": pd.Timestamp("2007-01-01T00:00:00Z"),
                "stop_time": pd.Timestamp("2007-01-01T00:01:00Z"),
                "stereo_flag": "NO",
                "is_stereo": False,
            },
            {
                "obs_id": "obs_1",
                "product_id": "OBS1_COLOR",
                "rationale_desc": "Right strip",
                "rationale_expanded": None,
                "has_rationale_expanded": False,
                "has_near_infrared": True,
                "has_red": True,
                "has_blue_green": True,
                "channel_count": 3,
                "map_scale": 0.5,
                "map_resolution": 1000.0,
                "emission_angle": 1.0,
                "incidence_angle": 2.0,
                "phase_angle": 3.0,
                "local_time": 4.0,
                "solar_longitude": 5.0,
                "sub_solar_azimuth": 6.0,
                "north_azimuth": 270.0,
                "spacecraft_altitude": 300.0,
                "start_time": pd.Timestamp("2007-01-01T00:00:00Z"),
                "stop_time": pd.Timestamp("2007-01-01T00:01:00Z"),
                "stereo_flag": "NO",
                "is_stereo": False,
            },
        ]
    )

    records = build_patch_records(
        dataset,
        size=1.0,
        centers=[(0.25, 1.0, interval)],
        observation_metadata=observation_metadata,
    )

    assert len(records) == 1
    row = records.iloc[0]
    assert row["dominant_obs_id"] == "obs_0"
    assert row["contributing_obs_ids"] == ("obs_0", "obs_1")
    assert row["source_obs_count"] == 2
    assert row["rationale_raw"] == "Left strip"
    assert bool(row["has_near_infrared"])
    assert bool(row["has_red"])
    assert bool(row["has_blue_green"])
    assert row["dominant_overlap_fraction"] == pytest.approx(1.0 / 1.75)

    summary = summarize_patch_records(records)
    assert summary["num_patches"] == pytest.approx(1.0)
    assert summary["num_unique_dominant_obs"] == pytest.approx(1.0)


class _FakeGeoDataset:
    def __init__(self, image: torch.Tensor):
        self.image = image

    def __getitem__(self, _: object) -> dict[str, object]:
        return {
            "image": self.image.clone(),
            "bounds": torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32),
            "crs": "mars",
        }


def test_patch_dataset_returns_expected_keys_and_quality_flags():
    image = torch.zeros(3, 4, 4, dtype=torch.float32)
    image[1, :2, :2] = 0.4
    fake = _FakeGeoDataset(image)

    observation_metadata = pd.DataFrame(
        [
            {
                "obs_id": "obs_0",
                "product_id": "OBS0_RED",
                "rationale_desc": "Left strip",
                "rationale_expanded": "Expanded left strip context",
                "has_rationale_expanded": True,
                "expansion_model": "mock-llm",
                "prompt_version": "v1",
                "map_scale": 0.25,
                "map_resolution": 1000.0,
                "emission_angle": 1.0,
                "incidence_angle": 2.0,
                "phase_angle": 3.0,
                "local_time": 4.0,
                "solar_longitude": 5.0,
                "sub_solar_azimuth": 6.0,
                "north_azimuth": 270.0,
                "spacecraft_altitude": 300.0,
                "start_time": pd.Timestamp("2007-01-01T00:00:00Z"),
                "stop_time": pd.Timestamp("2007-01-01T00:01:00Z"),
                "stereo_flag": "NO",
                "is_stereo": False,
            }
        ]
    )
    patch_records = pd.DataFrame(
        [
            {
                "patch_id": "patch_000000",
                "x_start": -1.0,
                "x_stop": 1.0,
                "y_start": 0.0,
                "y_stop": 2.0,
                "t_start": pd.Timestamp("2007-01-01T00:00:00Z"),
                "t_stop": pd.Timestamp("2007-01-01T00:01:00Z"),
                "min_lon": -1.0,
                "max_lon": 1.0,
                "min_lat": 0.0,
                "max_lat": 2.0,
                "centroid_lon": 0.0,
                "centroid_lat": 1.0,
                "patch_lon_span_deg": 2.0,
                "patch_lat_span_deg": 2.0,
                "patch_area_deg2": 4.0,
                "dominant_obs_id": "obs_0",
                "contributing_obs_ids": ("obs_0",),
                "contributing_rationales": ("Left strip",),
                "overlap_fractions": (1.0,),
                "dominant_overlap_fraction": 1.0,
                "source_obs_count": 1,
                "has_near_infrared": False,
                "has_red": True,
                "has_blue_green": False,
                "rationale_raw": "Left strip",
            }
        ]
    )

    ds = MarsCLIPPatchDataset(
        geo_dataset=fake,
        patch_records=patch_records,
        observation_metadata=observation_metadata,
        image_size=4,
        min_valid_fraction=0.5,
    )
    sample = ds[0]

    assert set(sample) == {
        "image",
        "valid_mask",
        "rationale_raw",
        "rationale_expanded",
        "location",
        "geo_features",
        "scale_features",
        "metadata",
    }
    assert sample["image"].shape == (3, 4, 4)
    assert sample["valid_mask"].dtype == torch.bool
    assert sample["location"].shape == (2,)
    assert sample["geo_features"].shape == (len(GEO_FEATURE_NAMES),)
    assert sample["scale_features"].shape == (len(PATCH_SCALE_FEATURE_NAMES),)
    assert sample["rationale_raw"] == "Left strip"
    assert sample["rationale_expanded"] == "Expanded left strip context"

    metadata = sample["metadata"]
    assert metadata["obs_id"] == "obs_0"
    assert metadata["patch_id"] == "patch_000000"
    assert metadata["viewing_features"].shape == (len(VIEWING_FEATURE_NAMES),)
    assert metadata["band_presence_mask"].tolist() == [False, True, False]
    assert torch.allclose(
        metadata["band_valid_fraction"],
        torch.tensor([0.0, 0.25, 0.0]),
    )
    assert metadata["overall_valid_fraction"] == pytest.approx(0.25)
    assert not metadata["is_patch_valid"]
    assert metadata["contributing_obs_ids"] == ("obs_0",)
    assert metadata["overlap_fractions"] == (1.0,)


def test_patch_dataset_color_only_filters_red_only_observations(mars_crs):
    geometries = [
        box(-1.0, 0.0, 1.0, 2.0),
        box(0.0, 0.0, 2.0, 2.0),
    ]
    dataset = make_mock_dataset(geometries, mars_crs)
    dataset.index["obs_id"] = ["OBS_A", "OBS_B"]
    dataset._raw_index = _raw_index_rows()

    image = torch.full((3, 4, 4), 0.25, dtype=torch.float32)

    def _getitem(_: object) -> dict[str, object]:
        return {
            "image": image.clone(),
            "bounds": torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32),
            "crs": "mars",
        }

    dataset.__getitem__ = _getitem  # type: ignore[attr-defined]

    patch_records = pd.DataFrame(
        [
            {
                "patch_id": "patch_color",
                "x_start": -1.0,
                "x_stop": 1.0,
                "y_start": 0.0,
                "y_stop": 2.0,
                "t_start": pd.Timestamp("2007-01-01T00:00:00Z"),
                "t_stop": pd.Timestamp("2007-01-01T00:01:00Z"),
                "min_lon": -1.0,
                "max_lon": 1.0,
                "min_lat": 0.0,
                "max_lat": 2.0,
                "centroid_lon": 0.0,
                "centroid_lat": 1.0,
                "patch_lon_span_deg": 2.0,
                "patch_lat_span_deg": 2.0,
                "patch_area_deg2": 4.0,
                "dominant_obs_id": "OBS_A",
                "contributing_obs_ids": ("OBS_A",),
                "contributing_rationales": ("Olympus Mons lava channels",),
                "overlap_fractions": (1.0,),
                "dominant_overlap_fraction": 1.0,
                "source_obs_count": 1,
                "has_near_infrared": True,
                "has_red": True,
                "has_blue_green": True,
                "rationale_raw": "Olympus Mons lava channels",
            },
            {
                "patch_id": "patch_red_only",
                "x_start": 0.0,
                "x_stop": 2.0,
                "y_start": 0.0,
                "y_stop": 2.0,
                "t_start": pd.Timestamp("2007-01-02T00:00:00Z"),
                "t_stop": pd.Timestamp("2007-01-02T00:01:00Z"),
                "min_lon": 0.0,
                "max_lon": 2.0,
                "min_lat": 0.0,
                "max_lat": 2.0,
                "centroid_lon": 1.0,
                "centroid_lat": 1.0,
                "patch_lon_span_deg": 2.0,
                "patch_lat_span_deg": 2.0,
                "patch_area_deg2": 4.0,
                "dominant_obs_id": "OBS_B",
                "contributing_obs_ids": ("OBS_B",),
                "contributing_rationales": ("Red only strip",),
                "overlap_fractions": (1.0,),
                "dominant_overlap_fraction": 1.0,
                "source_obs_count": 1,
                "has_near_infrared": False,
                "has_red": True,
                "has_blue_green": False,
                "rationale_raw": "Red only strip",
            },
        ]
    )

    ds = MarsCLIPPatchDataset(
        geo_dataset=dataset,
        patch_records=patch_records,
        image_size=4,
        color_only=True,
    )

    assert len(ds) == 1
    assert ds.patch_records["patch_id"].tolist() == ["patch_color"]
    assert set(ds.observation_metadata.index.tolist()) == {"OBS_A"}
    assert set(ds.geo_dataset.index["obs_id"].astype(str).tolist()) == {"OBS_A"}


def test_summarize_patch_samples_reports_validity_threshold():
    samples = [
        {
            "metadata": {
                "overall_valid_fraction": 0.25,
                "is_patch_valid": False,
                "source_obs_count": 1,
                "dominant_overlap_fraction": 1.0,
                "band_valid_fraction": torch.tensor([0.0, 0.25, 0.0]),
                "min_valid_fraction": DEFAULT_PATCH_VALID_FRACTION,
            }
        },
        {
            "metadata": {
                "overall_valid_fraction": 0.75,
                "is_patch_valid": True,
                "source_obs_count": 2,
                "dominant_overlap_fraction": 0.8,
                "band_valid_fraction": torch.tensor([0.75, 0.75, 0.5]),
                "min_valid_fraction": DEFAULT_PATCH_VALID_FRACTION,
            }
        },
    ]

    summary = summarize_patch_samples(samples)

    assert summary["num_samples"] == 2
    assert summary["num_valid_patches"] == 1
    assert summary["valid_patch_fraction"] == pytest.approx(0.5)
    assert summary["mean_overall_valid_fraction"] == pytest.approx(0.5)
    assert summary["mean_source_obs_count"] == pytest.approx(1.5)
    assert summary["mean_dominant_overlap_fraction"] == pytest.approx(0.9)
    assert summary["mean_band_valid_fraction"] == pytest.approx([0.375, 0.5, 0.25])
    assert summary["min_valid_fraction_threshold"] == pytest.approx(
        DEFAULT_PATCH_VALID_FRACTION
    )


def test_patch_records_round_trip_pickle(tmp_path):
    patch_records = pd.DataFrame(
        [
            {
                "patch_id": "patch_000000",
                "dominant_obs_id": "OBS_A",
                "contributing_obs_ids": ("OBS_A", "OBS_B"),
                "contributing_rationales": ("A", "B"),
                "overlap_fractions": (0.6, 0.4),
                "has_near_infrared": True,
                "has_red": True,
                "has_blue_green": True,
            }
        ]
    )

    path = save_patch_records(patch_records, tmp_path / "patch_records.pkl")
    loaded = load_patch_records(path)

    assert loaded.to_dict(orient="records") == patch_records.to_dict(orient="records")


@pytest.mark.integration
def test_real_patch_dataset_returns_stage_a_sample():
    from mars_hirise import MarsHiRISE

    geo = MarsHiRISE(
        bbox=(-136.0, 12.0, -124.0, 24.0),
        channels=["NEAR-INFRARED", "RED", "BLUE-GREEN"],
        download=False,
    )
    observation_metadata = build_patch_observation_metadata(geo)

    first_geom = geo.index.geometry.iloc[0]
    minx, miny, maxx, maxy = first_geom.bounds
    center_x = (minx + maxx) / 2.0
    center_y = (miny + maxy) / 2.0
    interval = geo.index.index[0]

    patch_records = build_patch_records(
        geo,
        size=0.005,
        centers=[(center_x, center_y, interval)],
        observation_metadata=observation_metadata,
    )

    ds = MarsCLIPPatchDataset(
        geo_dataset=geo,
        patch_records=patch_records,
        observation_metadata=observation_metadata,
        image_size=64,
    )
    sample = ds[0]

    assert sample["image"].shape == (3, 64, 64)
    assert sample["valid_mask"].shape == (64, 64)
    assert sample["metadata"]["patch_id"].startswith("patch_")
    assert sample["metadata"]["overall_valid_fraction"] > 0.0
