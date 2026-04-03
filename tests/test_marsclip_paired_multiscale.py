"""Tests for the paired local/global crop dataset foundation."""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest
import torch
from shapely.geometry import box

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from helpers import make_mock_dataset
from clip.marsclip_dataset import GEO_FEATURE_NAMES, VIEWING_FEATURE_NAMES
from clip.marsclip_paired_multiscale import (
    MarsCLIPPairedBatchCollator,
    MarsCLIPPairedCropDataset,
    build_paired_crop_records,
    load_paired_crop_records,
    save_paired_crop_records,
)
from clip.marsclip_text import SimpleTextTokenizer


class _SamplingGeoDataset:
    def __init__(self, index: pd.DataFrame, raw_index: pd.DataFrame):
        self.index = index
        self._raw_index = raw_index

    def __getitem__(self, query: object) -> dict[str, object]:
        x_slice, y_slice, _ = query
        out_w = int(round((float(x_slice.stop) - float(x_slice.start)) / float(x_slice.step)))
        out_h = int(round((float(y_slice.stop) - float(y_slice.start)) / float(y_slice.step)))
        image = torch.zeros(3, out_h, out_w, dtype=torch.float32)
        span = float(x_slice.stop) - float(x_slice.start)
        if span > 1.5:
            image[:, 1:, :] = 0.6
        else:
            image[1, : out_h // 2, :] = 0.4
        return {
            "image": image,
            "bounds": torch.tensor(
                [float(x_slice.start), float(y_slice.start), float(x_slice.stop), float(y_slice.stop)],
                dtype=torch.float32,
            ),
            "crs": "mars",
        }


def _raw_index_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "PRODUCT_ID": "OBS_A_COLOR",
                "RATIONALE_DESC": "Olympus Mons lava channels",
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


def test_build_paired_crop_records_drops_edge_cases_with_weak_global_overlap(mars_crs):
    geometries = [
        box(-1.0, 0.0, 1.0, 4.0),
        box(1.0, 0.0, 3.0, 4.0),
    ]
    dataset = make_mock_dataset(geometries, mars_crs)
    dataset.index["obs_id"] = ["obs_0", "obs_1"]
    interval = dataset.index.index[0]

    observation_metadata = pd.DataFrame(
        [
            {
                "obs_id": "obs_0",
                "product_id": "OBS0_COLOR",
                "rationale_desc": "Interior strip",
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
                "has_near_infrared": True,
                "has_red": True,
                "has_blue_green": True,
            },
            {
                "obs_id": "obs_1",
                "product_id": "OBS1_COLOR",
                "rationale_desc": "Neighbor strip",
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
                "has_near_infrared": True,
                "has_red": True,
                "has_blue_green": True,
            },
        ]
    )
    patch_records = pd.DataFrame(
        [
            {
                "patch_id": "patch_keep",
                "x_start": -0.5,
                "x_stop": 0.5,
                "y_start": 1.5,
                "y_stop": 2.5,
                "t_start": pd.Timestamp(interval.left),
                "t_stop": pd.Timestamp(interval.right),
                "min_lon": -0.5,
                "max_lon": 0.5,
                "min_lat": 1.5,
                "max_lat": 2.5,
                "centroid_lon": 0.0,
                "centroid_lat": 2.0,
                "patch_lon_span_deg": 1.0,
                "patch_lat_span_deg": 1.0,
                "patch_area_deg2": 1.0,
                "dominant_obs_id": "obs_0",
                "contributing_obs_ids": ("obs_0",),
                "contributing_rationales": ("Interior strip",),
                "overlap_fractions": (1.0,),
                "dominant_overlap_fraction": 1.0,
                "source_obs_count": 1,
                "has_near_infrared": True,
                "has_red": True,
                "has_blue_green": True,
                "rationale_raw": "Interior strip",
            },
            {
                "patch_id": "patch_drop",
                "x_start": 0.3,
                "x_stop": 1.3,
                "y_start": 1.5,
                "y_stop": 2.5,
                "t_start": pd.Timestamp(interval.left),
                "t_stop": pd.Timestamp(interval.right),
                "min_lon": 0.3,
                "max_lon": 1.3,
                "min_lat": 1.5,
                "max_lat": 2.5,
                "centroid_lon": 0.8,
                "centroid_lat": 2.0,
                "patch_lon_span_deg": 1.0,
                "patch_lat_span_deg": 1.0,
                "patch_area_deg2": 1.0,
                "dominant_obs_id": "obs_0",
                "contributing_obs_ids": ("obs_0", "obs_1"),
                "contributing_rationales": ("Interior strip", "Neighbor strip"),
                "overlap_fractions": (0.7, 0.3),
                "dominant_overlap_fraction": 0.7,
                "source_obs_count": 2,
                "has_near_infrared": True,
                "has_red": True,
                "has_blue_green": True,
                "rationale_raw": "Interior strip",
            },
        ]
    )

    paired = build_paired_crop_records(
        dataset,
        patch_records=patch_records,
        observation_metadata=observation_metadata,
        global_scale_factor=5.0,
        min_global_dominant_overlap=0.51,
    )

    assert paired["patch_id"].tolist() == ["patch_keep"]
    assert paired.iloc[0]["global_patch_lon_span_deg"] == pytest.approx(5.0)
    assert paired.iloc[0]["global_dominant_obs_id"] == "obs_0"


def test_paired_multiscale_dataset_returns_local_and_global_crops(mars_crs):
    base = make_mock_dataset([box(-2.0, 0.0, 2.0, 4.0)], mars_crs)
    base.index["obs_id"] = ["OBS_A"]
    geo_dataset = _SamplingGeoDataset(base.index, _raw_index_rows().iloc[[0]].copy())

    observation_metadata = pd.DataFrame(
        [
            {
                "obs_id": "OBS_A",
                "product_id": "OBS_A_COLOR",
                "rationale_desc": "Olympus Mons lava channels",
                "rationale_expanded": "Expanded Olympus geology",
                "has_rationale_expanded": True,
                "expansion_model": "mock-llm",
                "prompt_version": "v1",
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
                "stereo_flag": "YES",
                "is_stereo": True,
                "has_near_infrared": True,
                "has_red": True,
                "has_blue_green": True,
            }
        ]
    )
    patch_records = pd.DataFrame(
        [
            {
                "patch_id": "pair_000000",
                "x_start": -0.5,
                "x_stop": 0.5,
                "y_start": 1.5,
                "y_stop": 2.5,
                "t_start": pd.Timestamp("2007-01-01T00:00:00Z"),
                "t_stop": pd.Timestamp("2007-01-01T00:01:00Z"),
                "min_lon": -0.5,
                "max_lon": 0.5,
                "min_lat": 1.5,
                "max_lat": 2.5,
                "centroid_lon": 0.0,
                "centroid_lat": 2.0,
                "patch_lon_span_deg": 1.0,
                "patch_lat_span_deg": 1.0,
                "patch_area_deg2": 1.0,
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
            }
        ]
    )
    paired_records = build_paired_crop_records(
        geo_dataset,
        patch_records=patch_records,
        observation_metadata=observation_metadata,
        global_scale_factor=3.0,
    )

    ds = MarsCLIPPairedCropDataset(
        geo_dataset=geo_dataset,
        patch_records=patch_records,
        paired_records=paired_records,
        observation_metadata=observation_metadata,
        image_size=4,
        min_valid_fraction=0.5,
    )
    sample = ds[0]

    assert set(sample) == {
        "local_image",
        "local_valid_mask",
        "global_image",
        "global_valid_mask",
        "rationale_raw",
        "rationale_expanded",
        "location",
        "geo_features",
        "local_scale_features",
        "global_scale_features",
        "metadata",
    }
    assert sample["local_image"].shape == (3, 4, 4)
    assert sample["global_image"].shape == (3, 4, 4)
    assert sample["local_valid_mask"].dtype == torch.bool
    assert sample["global_valid_mask"].dtype == torch.bool
    assert sample["geo_features"].shape == (len(GEO_FEATURE_NAMES),)
    assert sample["local_scale_features"].shape == (8,)
    assert sample["global_scale_features"].shape == (8,)
    assert sample["rationale_expanded"] == "Expanded Olympus geology"

    metadata = sample["metadata"]
    assert metadata["patch_id"] == "pair_000000"
    assert metadata["global_scale_factor"] == pytest.approx(3.0)
    assert metadata["viewing_features"].shape == (len(VIEWING_FEATURE_NAMES),)
    assert metadata["local_band_presence_mask"].tolist() == [True, True, True]
    assert metadata["global_band_presence_mask"].tolist() == [True, True, True]
    assert metadata["local_overall_valid_fraction"] == pytest.approx(0.5)
    assert metadata["global_overall_valid_fraction"] == pytest.approx(0.75)
    assert metadata["is_local_valid"]
    assert metadata["is_global_valid"]
    assert metadata["is_pair_valid"]


def test_paired_multiscale_dataset_color_only_filters_red_only_pairs(mars_crs):
    base = make_mock_dataset(
        [box(-2.0, 0.0, 0.0, 4.0), box(0.0, 0.0, 2.0, 4.0)],
        mars_crs,
    )
    base.index["obs_id"] = ["OBS_A", "OBS_B"]
    geo_dataset = _SamplingGeoDataset(base.index, _raw_index_rows())

    patch_records = pd.DataFrame(
        [
            {
                "patch_id": "pair_color",
                "x_start": -1.5,
                "x_stop": -0.5,
                "y_start": 1.5,
                "y_stop": 2.5,
                "t_start": pd.Timestamp("2007-01-01T00:00:00Z"),
                "t_stop": pd.Timestamp("2007-01-01T00:01:00Z"),
                "min_lon": -1.5,
                "max_lon": -0.5,
                "min_lat": 1.5,
                "max_lat": 2.5,
                "centroid_lon": -1.0,
                "centroid_lat": 2.0,
                "patch_lon_span_deg": 1.0,
                "patch_lat_span_deg": 1.0,
                "patch_area_deg2": 1.0,
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
                "patch_id": "pair_red_only",
                "x_start": 0.5,
                "x_stop": 1.5,
                "y_start": 1.5,
                "y_stop": 2.5,
                "t_start": pd.Timestamp("2007-01-02T00:00:00Z"),
                "t_stop": pd.Timestamp("2007-01-02T00:01:00Z"),
                "min_lon": 0.5,
                "max_lon": 1.5,
                "min_lat": 1.5,
                "max_lat": 2.5,
                "centroid_lon": 1.0,
                "centroid_lat": 2.0,
                "patch_lon_span_deg": 1.0,
                "patch_lat_span_deg": 1.0,
                "patch_area_deg2": 1.0,
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
    paired_records = pd.DataFrame(
        [
            {
                **patch_records.iloc[0].to_dict(),
                "global_x_start": -2.0,
                "global_x_stop": 0.0,
                "global_y_start": 1.0,
                "global_y_stop": 3.0,
                "global_min_lon": -2.0,
                "global_max_lon": 0.0,
                "global_min_lat": 1.0,
                "global_max_lat": 3.0,
                "global_patch_lon_span_deg": 2.0,
                "global_patch_lat_span_deg": 2.0,
                "global_patch_area_deg2": 4.0,
                "global_scale_factor": 2.0,
                "global_dominant_obs_id": "OBS_A",
                "global_contributing_obs_ids": ("OBS_A",),
                "global_contributing_rationales": ("Olympus Mons lava channels",),
                "global_overlap_fractions": (1.0,),
                "global_dominant_overlap_fraction": 1.0,
                "global_source_obs_count": 1,
                "global_has_near_infrared": True,
                "global_has_red": True,
                "global_has_blue_green": True,
            },
            {
                **patch_records.iloc[1].to_dict(),
                "global_x_start": 0.0,
                "global_x_stop": 2.0,
                "global_y_start": 1.0,
                "global_y_stop": 3.0,
                "global_min_lon": 0.0,
                "global_max_lon": 2.0,
                "global_min_lat": 1.0,
                "global_max_lat": 3.0,
                "global_patch_lon_span_deg": 2.0,
                "global_patch_lat_span_deg": 2.0,
                "global_patch_area_deg2": 4.0,
                "global_scale_factor": 2.0,
                "global_dominant_obs_id": "OBS_B",
                "global_contributing_obs_ids": ("OBS_B",),
                "global_contributing_rationales": ("Red only strip",),
                "global_overlap_fractions": (1.0,),
                "global_dominant_overlap_fraction": 1.0,
                "global_source_obs_count": 1,
                "global_has_near_infrared": False,
                "global_has_red": True,
                "global_has_blue_green": False,
            },
        ]
    )

    ds = MarsCLIPPairedCropDataset(
        geo_dataset=geo_dataset,
        patch_records=patch_records,
        paired_records=paired_records,
        image_size=4,
        color_only=True,
    )

    assert len(ds) == 1
    assert ds.paired_records["patch_id"].tolist() == ["pair_color"]
    assert set(ds.observation_metadata.index.tolist()) == {"OBS_A"}
    assert set(ds.geo_dataset.index["obs_id"].astype(str).tolist()) == {"OBS_A"}


def test_paired_multiscale_collator_batches_paired_samples(mars_crs):
    base = make_mock_dataset([box(-2.0, 0.0, 2.0, 4.0)], mars_crs)
    base.index["obs_id"] = ["OBS_A"]
    geo_dataset = _SamplingGeoDataset(base.index, _raw_index_rows().iloc[[0]].copy())
    observation_metadata = pd.DataFrame(
        [
            {
                "obs_id": "OBS_A",
                "product_id": "OBS_A_COLOR",
                "rationale_desc": "Olympus Mons lava channels",
                "rationale_expanded": "Expanded Olympus geology",
                "has_rationale_expanded": True,
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
                "stereo_flag": "YES",
                "is_stereo": True,
                "has_near_infrared": True,
                "has_red": True,
                "has_blue_green": True,
            }
        ]
    )
    patch_records = pd.DataFrame(
        [
            {
                "patch_id": "pair_000000",
                "x_start": -0.5,
                "x_stop": 0.5,
                "y_start": 1.5,
                "y_stop": 2.5,
                "t_start": pd.Timestamp("2007-01-01T00:00:00Z"),
                "t_stop": pd.Timestamp("2007-01-01T00:01:00Z"),
                "min_lon": -0.5,
                "max_lon": 0.5,
                "min_lat": 1.5,
                "max_lat": 2.5,
                "centroid_lon": 0.0,
                "centroid_lat": 2.0,
                "patch_lon_span_deg": 1.0,
                "patch_lat_span_deg": 1.0,
                "patch_area_deg2": 1.0,
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
            }
        ]
    )
    paired_records = build_paired_crop_records(
        geo_dataset,
        patch_records=patch_records,
        observation_metadata=observation_metadata,
        global_scale_factor=3.0,
    )
    ds = MarsCLIPPairedCropDataset(
        geo_dataset=geo_dataset,
        patch_records=patch_records,
        paired_records=paired_records,
        observation_metadata=observation_metadata,
        image_size=4,
        min_valid_fraction=0.5,
    )

    sample = ds[0]
    tokenizer = SimpleTextTokenizer.build(
        [sample["rationale_raw"], sample["rationale_expanded"] or ""]
    )
    collator = MarsCLIPPairedBatchCollator(tokenizer, max_length=16)
    batch = collator([sample, sample])

    assert batch["local_image"].shape == (2, 3, 4, 4)
    assert batch["global_image"].shape == (2, 3, 4, 4)
    assert batch["local_valid_mask"].shape == (2, 4, 4)
    assert batch["global_valid_mask"].shape == (2, 4, 4)
    assert batch["location"].shape == (2, 2)
    assert batch["geo_features"].shape == (2, len(GEO_FEATURE_NAMES))
    assert batch["local_scale_features"].shape == (2, 8)
    assert batch["global_scale_features"].shape == (2, 8)
    assert batch["viewing_features"].shape == (2, len(VIEWING_FEATURE_NAMES))
    assert batch["local_quality_features"].shape == (2, 7)
    assert batch["global_quality_features"].shape == (2, 7)
    assert batch["input_ids"].shape == (2, 16)
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["text"][0].startswith("Olympus Mons lava channels")


def test_paired_crop_record_cache_roundtrip(tmp_path, mars_crs):
    base = make_mock_dataset([box(-2.0, 0.0, 2.0, 4.0)], mars_crs)
    base.index["obs_id"] = ["OBS_A"]
    geo_dataset = _SamplingGeoDataset(base.index, _raw_index_rows().iloc[[0]].copy())

    observation_metadata = pd.DataFrame(
        [
            {
                "obs_id": "OBS_A",
                "product_id": "OBS_A_COLOR",
                "rationale_desc": "Olympus Mons lava channels",
                "rationale_expanded": None,
                "has_rationale_expanded": False,
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
                "has_near_infrared": True,
                "has_red": True,
                "has_blue_green": True,
            }
        ]
    )
    patch_records = pd.DataFrame(
        [
            {
                "patch_id": "pair_000000",
                "x_start": -0.5,
                "x_stop": 0.5,
                "y_start": 1.5,
                "y_stop": 2.5,
                "t_start": pd.Timestamp("2007-01-01T00:00:00Z"),
                "t_stop": pd.Timestamp("2007-01-01T00:01:00Z"),
                "min_lon": -0.5,
                "max_lon": 0.5,
                "min_lat": 1.5,
                "max_lat": 2.5,
                "centroid_lon": 0.0,
                "centroid_lat": 2.0,
                "patch_lon_span_deg": 1.0,
                "patch_lat_span_deg": 1.0,
                "patch_area_deg2": 1.0,
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
            }
        ]
    )
    paired_records = build_paired_crop_records(
        geo_dataset,
        patch_records=patch_records,
        observation_metadata=observation_metadata,
        global_scale_factor=3.0,
    )

    cache_path = tmp_path / "paired_records.pkl"
    save_paired_crop_records(paired_records, cache_path)
    restored = load_paired_crop_records(cache_path)

    assert restored["patch_id"].tolist() == paired_records["patch_id"].tolist()
    assert restored.iloc[0]["global_contributing_obs_ids"] == paired_records.iloc[0][
        "global_contributing_obs_ids"
    ]
