"""Tests for the patch-level MarsCLIP Stage A dataset bridge."""

from __future__ import annotations

import pathlib
import sys
import types

import pandas as pd
import pytest
import torch
from shapely.geometry import Polygon, box

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from helpers import make_mock_dataset
from clip.marsclip_dataset import GEO_FEATURE_NAMES, VIEWING_FEATURE_NAMES
from clip.marsclip_patches import (
    DEFAULT_PATCH_VALID_FRACTION,
    PATCH_SCALE_FEATURE_NAMES,
    MarsCLIPPatchDataset,
    _filter_observation_metadata_to_color,
    _filter_geo_dataset_to_obs_ids,
    _filter_patch_records_to_color,
    _normalize_image_size,
    _select_centers,
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


def _minimal_obs_metadata(obs_id: str = "obs_0") -> pd.DataFrame:
    """Return a one-row observation_metadata DataFrame with all required fields."""
    return pd.DataFrame([{
        "obs_id": obs_id,
        "product_id": f"{obs_id.upper()}_COLOR",
        "rationale_desc": "Test obs",
        "rationale_expanded": None,
        "has_rationale_expanded": False,
        "has_near_infrared": True,
        "has_red": True,
        "has_blue_green": True,
        "channel_count": 3,
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
        "start_time": pd.Timestamp("2007-01-01T00:00:00Z"),
        "stop_time": pd.Timestamp("2007-01-01T00:01:00Z"),
        "stereo_flag": "NO",
        "is_stereo": False,
    }])


def _minimal_patch_records(obs_id: str = "obs_0") -> pd.DataFrame:
    """Return a one-row patch_records DataFrame."""
    return pd.DataFrame([{
        "patch_id": "patch_000000",
        "x_start": -0.5,
        "x_stop": 0.5,
        "y_start": 0.5,
        "y_stop": 1.5,
        "t_start": pd.Timestamp("2006-01-01T00:00:00Z"),
        "t_stop": pd.Timestamp("2006-01-02T00:00:00Z"),
        "min_lon": -0.5,
        "max_lon": 0.5,
        "min_lat": 0.5,
        "max_lat": 1.5,
        "centroid_lon": 0.0,
        "centroid_lat": 1.0,
        "patch_lon_span_deg": 1.0,
        "patch_lat_span_deg": 1.0,
        "patch_area_deg2": 1.0,
        "dominant_obs_id": obs_id,
        "contributing_obs_ids": (obs_id,),
        "contributing_rationales": ("Test obs",),
        "overlap_fractions": (1.0,),
        "dominant_overlap_fraction": 1.0,
        "source_obs_count": 1,
        "has_near_infrared": True,
        "has_red": True,
        "has_blue_green": True,
        "rationale_raw": "Test obs",
    }])


# ---------------------------------------------------------------------------
# _filter_observation_metadata_to_color
# ---------------------------------------------------------------------------

def test_filter_observation_metadata_to_color_raises_missing_columns():
    df = pd.DataFrame([{"obs_id": "OBS_A"}])
    with pytest.raises(ValueError, match="missing required COLOR filter columns"):
        _filter_observation_metadata_to_color(df)


def test_filter_observation_metadata_to_color_raises_no_color_rows():
    df = pd.DataFrame([{
        "obs_id": "OBS_A",
        "has_near_infrared": False,
        "has_blue_green": False,
    }])
    with pytest.raises(ValueError, match="No COLOR-capable observations remain"):
        _filter_observation_metadata_to_color(df)


# ---------------------------------------------------------------------------
# _filter_geo_dataset_to_obs_ids
# ---------------------------------------------------------------------------

def test_filter_geo_dataset_to_obs_ids_raises_missing_obs_id_column(mars_crs):
    geometries = [box(-1.0, 0.0, 1.0, 2.0)]
    dataset = make_mock_dataset(geometries, mars_crs)
    # Remove obs_id column from index
    dataset.index = dataset.index.drop(columns=["obs_id"])
    with pytest.raises(ValueError, match="must include an 'obs_id' column"):
        _filter_geo_dataset_to_obs_ids(dataset, allowed_obs_ids={"OBS_A"})


def test_filter_geo_dataset_to_obs_ids_raises_empty_after_filter(mars_crs):
    geometries = [box(-1.0, 0.0, 1.0, 2.0)]
    dataset = make_mock_dataset(geometries, mars_crs)
    with pytest.raises(ValueError, match="No spatial-index observations remain"):
        _filter_geo_dataset_to_obs_ids(dataset, allowed_obs_ids={"NONEXISTENT"})


def test_filter_geo_dataset_to_obs_ids_raises_missing_product_id_in_raw_index(mars_crs):
    """Line 89: _raw_index present but missing PRODUCT_ID column."""
    geometries = [box(-1.0, 0.0, 1.0, 2.0)]
    dataset = make_mock_dataset(geometries, mars_crs)
    dataset._raw_index = pd.DataFrame([{"OBS_ID": "obs_0"}])  # no PRODUCT_ID column
    with pytest.raises(ValueError, match="must include a 'PRODUCT_ID' column"):
        _filter_geo_dataset_to_obs_ids(dataset, allowed_obs_ids={"obs_0"})


# ---------------------------------------------------------------------------
# _filter_patch_records_to_color
# ---------------------------------------------------------------------------

def test_filter_patch_records_to_color_raises_missing_columns():
    df = pd.DataFrame([{"patch_id": "p0"}])
    with pytest.raises(ValueError, match="missing required COLOR filter columns"):
        _filter_patch_records_to_color(df, allowed_obs_ids={"OBS_A"})


def test_filter_patch_records_to_color_returns_false_for_non_sequence():
    """_all_allowed returns False when contributing_obs_ids is not a Sequence."""
    df = pd.DataFrame([{
        "dominant_obs_id": "OBS_A",
        "contributing_obs_ids": 42,  # not a Sequence
        "has_near_infrared": True,
        "has_blue_green": True,
    }])
    with pytest.raises(ValueError, match="No patch records remain"):
        _filter_patch_records_to_color(df, allowed_obs_ids={"OBS_A"})


def test_filter_patch_records_to_color_raises_empty_after_filter():
    df = pd.DataFrame([{
        "dominant_obs_id": "OBS_B",
        "contributing_obs_ids": ("OBS_B",),
        "has_near_infrared": True,
        "has_blue_green": True,
    }])
    with pytest.raises(ValueError, match="No patch records remain"):
        _filter_patch_records_to_color(df, allowed_obs_ids={"OBS_A"})


# ---------------------------------------------------------------------------
# _normalize_image_size
# ---------------------------------------------------------------------------

def test_normalize_image_size_accepts_tuple():
    assert _normalize_image_size((128, 256)) == (128, 256)


def test_normalize_image_size_accepts_int():
    assert _normalize_image_size(224) == (224, 224)


# ---------------------------------------------------------------------------
# build_patch_observation_metadata edge cases
# ---------------------------------------------------------------------------

def test_build_patch_observation_metadata_raises_when_no_raw_index():
    fake = types.SimpleNamespace()  # no _raw_index attribute
    with pytest.raises(ValueError, match="must expose a populated _raw_index"):
        build_patch_observation_metadata(fake)


def test_build_patch_observation_metadata_raises_for_none_raw_index_value():
    """Raises when _raw_index attribute is None (not just missing)."""
    fake = types.SimpleNamespace(_raw_index=None)
    with pytest.raises(ValueError, match="must expose a populated _raw_index"):
        build_patch_observation_metadata(fake)


def test_build_patch_observation_metadata_applies_rationale_cache():
    from clip.rationale_cache import DEFAULT_PROMPT_TEMPLATE
    fake = types.SimpleNamespace(_raw_index=_raw_index_rows().iloc[[0]].copy())
    cache = pd.DataFrame([{
        "obs_id": "OBS_A",
        "rationale_raw": "Olympus Mons lava channels",
        "rationale_expanded": "Expanded description",
        "expansion_model": "mock-llm",
        "prompt_version": "v1",
        "prompt_template": DEFAULT_PROMPT_TEMPLATE,
        "expansion_status": "ok",
        "expansion_error": None,
        "expansion_timestamp": pd.Timestamp("2026-03-29T00:00:00Z"),
    }])
    metadata = build_patch_observation_metadata(fake, rationale_cache=cache)
    obs_a = metadata.loc[metadata["obs_id"] == "OBS_A"].iloc[0]
    assert obs_a["rationale_expanded"] == "Expanded description"
    assert bool(obs_a["has_rationale_expanded"])


# ---------------------------------------------------------------------------
# _select_centers
# ---------------------------------------------------------------------------

def test_select_centers_returns_all_when_below_max():
    centers = [(0.0, 0.0, None), (1.0, 1.0, None)]
    result = _select_centers(centers, max_patches=5)
    assert result == centers


def test_select_centers_truncates_without_generator():
    centers = [(float(i), float(i), None) for i in range(10)]
    result = _select_centers(centers, max_patches=3)
    assert result == centers[:3]


def test_select_centers_uses_generator_for_random_selection():
    centers = [(float(i), float(i), None) for i in range(10)]
    gen = torch.Generator().manual_seed(42)
    result = _select_centers(centers, max_patches=3, generator=gen)
    assert len(result) == 3
    assert all(c in centers for c in result)


# ---------------------------------------------------------------------------
# build_patch_records: loop continue paths
# ---------------------------------------------------------------------------

def test_build_patch_records_raises_empty_when_centers_outside_geometries(mars_crs):
    """Line 339 (no candidate_positions) and line 409 (empty records raise)."""
    geometries = [box(-1.0, 0.0, 1.0, 2.0)]
    dataset = make_mock_dataset(geometries, mars_crs)
    interval = dataset.index.index[0]
    obs_meta = _minimal_obs_metadata("obs_0")

    with pytest.raises(ValueError, match="No patch records could be built"):
        build_patch_records(
            dataset,
            size=0.1,
            centers=[(100.0, 100.0, interval)],  # far outside all geometries
            observation_metadata=obs_meta,
        )


def test_build_patch_records_skips_patches_with_no_geometry_intersection(mars_crs):
    """Line 343 (candidates.empty after geometry filter): triangle geometry."""
    # Triangle: vertices at (0,0), (2,0), (0,2). Area only below the hypotenuse.
    # A patch at (1.5, 1.5) is inside the bounding box [0,2]x[0,2] but outside the triangle.
    triangle = Polygon([(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)])
    dataset = make_mock_dataset([triangle], mars_crs)
    interval = dataset.index.index[0]
    obs_meta = _minimal_obs_metadata("obs_0")

    # Patch from (1.5, 1.5) to (2.5, 2.5) — bbox overlaps triangle but actual geometry doesn't
    # (all points in that patch have x+y > 2 which is outside the triangle)
    with pytest.raises(ValueError, match="No patch records could be built"):
        build_patch_records(
            dataset,
            size=1.0,
            centers=[(2.0, 2.0, interval)],  # center at (2,2), patch from (1.5,1.5) to (2.5,2.5)
            observation_metadata=obs_meta,
        )


def test_build_patch_records_skips_zero_area_overlaps(mars_crs):
    """Line 352 (overlaps empty after area check): touching-edge geometry."""
    # box(0,0,1,1) and a patch from (1,0) to (2,1): they share edge at x=1 (area=0)
    geom = box(0.0, 0.0, 1.0, 1.0)
    dataset = make_mock_dataset([geom], mars_crs)
    interval = dataset.index.index[0]
    obs_meta = _minimal_obs_metadata("obs_0")

    # Patch centered at (1.5, 0.5): box(1.0, 0.0, 2.0, 1.0) — touches geom at x=1 (area=0)
    with pytest.raises(ValueError, match="No patch records could be built"):
        build_patch_records(
            dataset,
            size=1.0,
            centers=[(1.5, 0.5, interval)],
            observation_metadata=obs_meta,
        )


def test_build_patch_records_builds_metadata_when_observation_metadata_none(mars_crs):
    """Line 302: build_patch_records calls build_patch_observation_metadata."""
    geometries = [box(-1.0, 0.0, 1.0, 2.0)]
    dataset = make_mock_dataset(geometries, mars_crs)
    # obs_id in GDF is "obs_0" → product must decode to "obs_0"
    dataset._raw_index = pd.DataFrame([{
        "PRODUCT_ID": "obs_0_COLOR",
        "RATIONALE_DESC": "Test obs",
        "START_TIME": "2007-01-01T00:00:00Z",
        "STOP_TIME": "2007-01-01T00:01:00Z",
        "MINIMUM_LONGITUDE": 224.0,
        "MAXIMUM_LONGITUDE": 225.0,
        "MINIMUM_LATITUDE": 10.0,
        "MAXIMUM_LATITUDE": 11.0,
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
        "STEREO_FLAG": "NO",
    }])
    interval = dataset.index.index[0]

    records = build_patch_records(
        dataset,
        size=1.0,
        centers=[(0.0, 1.0, interval)],
        # observation_metadata=None → triggers line 302
    )

    assert not records.empty
    assert records.iloc[0]["dominant_obs_id"] == "obs_0"


def test_build_patch_records_uses_sampler_when_centers_none(mars_crs):
    """Lines 306-313: build_patch_records uses HiRISEGeoSampler when centers=None."""
    geometries = [box(-1.0, 0.0, 1.0, 2.0)]
    dataset = make_mock_dataset(geometries, mars_crs)
    obs_meta = _minimal_obs_metadata("obs_0")

    records = build_patch_records(
        dataset,
        size=0.5,
        observation_metadata=obs_meta,
        # centers=None → HiRISEGeoSampler creates centers
    )

    assert not records.empty


# ---------------------------------------------------------------------------
# summarize_patch_records edge case
# ---------------------------------------------------------------------------

def test_summarize_patch_records_returns_zeros_for_empty():
    result = summarize_patch_records(pd.DataFrame())
    assert result["num_patches"] == 0
    assert result["num_unique_dominant_obs"] == 0
    assert result["mean_source_obs_count"] == pytest.approx(0.0)
    assert result["mean_dominant_overlap_fraction"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# save/load patch_records — parquet
# ---------------------------------------------------------------------------

def test_patch_records_round_trip_parquet(tmp_path):
    patch_records = pd.DataFrame([{
        "patch_id": "patch_000000",
        "dominant_obs_id": "OBS_A",
        "x_start": -0.5,
        "x_stop": 0.5,
        "dominant_overlap_fraction": 1.0,
    }])
    path = tmp_path / "patch_records.parquet"
    out = save_patch_records(patch_records, path)
    loaded = load_patch_records(out)
    assert loaded.iloc[0]["patch_id"] == "patch_000000"
    assert loaded.iloc[0]["dominant_obs_id"] == "OBS_A"


def test_patch_records_round_trip_csv(tmp_path):
    """Lines 447 (CSV save) and 460-464 (CSV load with literal_eval)."""
    patch_records = pd.DataFrame([{
        "patch_id": "patch_000000",
        "dominant_obs_id": "OBS_A",
        "contributing_obs_ids": ("OBS_A",),
        "contributing_rationales": ("Test",),
        "overlap_fractions": (1.0,),
        "dominant_overlap_fraction": 1.0,
    }])
    path = tmp_path / "patch_records.csv"
    out = save_patch_records(patch_records, path)
    loaded = load_patch_records(out)
    assert loaded.iloc[0]["patch_id"] == "patch_000000"
    assert loaded.iloc[0]["contributing_obs_ids"] == ("OBS_A",)


# ---------------------------------------------------------------------------
# summarize_patch_samples — empty list
# ---------------------------------------------------------------------------

def test_summarize_patch_samples_empty_list():
    result = summarize_patch_samples([])
    assert result["num_samples"] == 0
    assert result["num_valid_patches"] == 0
    assert result["valid_patch_fraction"] == pytest.approx(0.0)
    assert result["mean_band_valid_fraction"] == [0.0, 0.0, 0.0]
    assert result["min_valid_fraction_threshold"] == pytest.approx(DEFAULT_PATCH_VALID_FRACTION)


# ---------------------------------------------------------------------------
# MarsCLIPPatchDataset.__init__ edge cases
# ---------------------------------------------------------------------------

def test_patch_dataset_raises_when_both_geo_dataset_and_root_are_none():
    with pytest.raises(ValueError, match="Either geo_dataset or root must be provided"):
        MarsCLIPPatchDataset(geo_dataset=None, root=None)


def test_patch_dataset_merges_rationale_cache_with_explicit_observation_metadata():
    from clip.rationale_cache import DEFAULT_PROMPT_TEMPLATE
    image = torch.zeros(3, 4, 4, dtype=torch.float32)
    fake = _FakeGeoDataset(image)
    # Provide obs_meta WITHOUT rationale_expanded — merge_rationale_cache adds it
    obs_meta = _minimal_obs_metadata("obs_0").drop(columns=["rationale_expanded", "has_rationale_expanded"])
    cache = pd.DataFrame([{
        "obs_id": "obs_0",
        "rationale_raw": "Test obs",
        "rationale_expanded": "Expanded via cache",
        "expansion_model": "mock-llm",
        "prompt_version": "v1",
        "prompt_template": DEFAULT_PROMPT_TEMPLATE,
        "expansion_status": "ok",
        "expansion_error": None,
        "expansion_timestamp": pd.Timestamp("2026-03-29T00:00:00Z"),
    }])

    ds = MarsCLIPPatchDataset(
        geo_dataset=fake,
        observation_metadata=obs_meta,
        patch_records=_minimal_patch_records("obs_0"),
        rationale_cache=cache,  # triggers line 556
        image_size=4,
    )
    assert len(ds) == 1
    assert ds.observation_metadata.loc["obs_0"]["rationale_expanded"] == "Expanded via cache"


def test_patch_dataset_adds_rationale_expanded_when_column_missing():
    """Line 561-562: observation_metadata without rationale_expanded column."""
    image = torch.zeros(3, 4, 4, dtype=torch.float32)
    fake = _FakeGeoDataset(image)
    obs_meta = _minimal_obs_metadata("obs_0").drop(columns=["rationale_expanded", "has_rationale_expanded"])

    ds = MarsCLIPPatchDataset(
        geo_dataset=fake,
        observation_metadata=obs_meta,
        patch_records=_minimal_patch_records("obs_0"),
        image_size=4,
    )
    assert "rationale_expanded" in ds.observation_metadata.columns
    assert ds.observation_metadata.loc["obs_0"]["rationale_expanded"] is None
    assert not bool(ds.observation_metadata.loc["obs_0"]["has_rationale_expanded"])


def test_patch_dataset_raises_when_patch_records_empty():
    """Line 590: empty patch_records raises ValueError."""
    image = torch.zeros(3, 4, 4, dtype=torch.float32)
    fake = _FakeGeoDataset(image)
    with pytest.raises(ValueError, match="Patch record table is empty"):
        MarsCLIPPatchDataset(
            geo_dataset=fake,
            observation_metadata=_minimal_obs_metadata("obs_0"),
            patch_records=pd.DataFrame(),  # empty
            image_size=4,
        )


def test_patch_dataset_builds_patch_records_when_not_provided(mars_crs):
    """Line 572: MarsCLIPPatchDataset builds patch_records internally."""
    geometries = [box(-1.0, 0.0, 1.0, 2.0)]
    dataset = make_mock_dataset(geometries, mars_crs)
    obs_meta = _minimal_obs_metadata("obs_0")

    ds = MarsCLIPPatchDataset(
        geo_dataset=dataset,
        observation_metadata=obs_meta,
        patch_size=0.5,
        image_size=4,
        # patch_records=None → triggers build_patch_records (line 572)
    )
    assert len(ds) > 0


# ---------------------------------------------------------------------------
# MarsCLIPPatchDataset.__getitem__ — transforms
# ---------------------------------------------------------------------------

def test_patch_dataset_applies_transform(mars_crs):
    """Line 685: transforms function is applied to sample."""
    image = torch.full((3, 4, 4), 0.3, dtype=torch.float32)
    fake = _FakeGeoDataset(image)
    transformed = []

    def _add_key(sample: dict) -> dict:
        sample["extra"] = "added"
        transformed.append(True)
        return sample

    ds = MarsCLIPPatchDataset(
        geo_dataset=fake,
        observation_metadata=_minimal_obs_metadata("obs_0"),
        patch_records=_minimal_patch_records("obs_0"),
        image_size=4,
        transforms=_add_key,
    )
    sample = ds[0]

    assert "extra" in sample
    assert sample["extra"] == "added"
    assert len(transformed) == 1


@pytest.mark.integration
def test_real_patch_dataset_returns_stage_a_sample():
    from dataset.mars_hirise import MarsHiRISE

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
