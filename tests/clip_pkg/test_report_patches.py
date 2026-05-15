"""Tests for Stage A1 patch report generation."""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import torch

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from clip.marsclip_patches import DEFAULT_PATCH_VALID_FRACTION, MarsCLIPPatchDataset
from clip.report_marsclip_patches import save_patch_report


class _FakeGeoDataset:
    def __init__(self, image: torch.Tensor):
        self.image = image

    def __getitem__(self, _: object) -> dict[str, object]:
        return {
            "image": self.image.clone(),
            "bounds": torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32),
            "crs": "mars",
        }


def test_save_patch_report_writes_summary_and_preview(tmp_path):
    image = torch.zeros(3, 4, 4, dtype=torch.float32)
    image[:, :3, :3] = 0.5

    observation_metadata = pd.DataFrame(
        [
            {
                "obs_id": "obs_0",
                "product_id": "OBS0_COLOR",
                "rationale_desc": "Left strip",
                "rationale_expanded": None,
                "has_rationale_expanded": False,
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
                "has_near_infrared": True,
                "has_red": True,
                "has_blue_green": True,
                "rationale_raw": "Left strip",
            }
        ]
    )

    ds = MarsCLIPPatchDataset(
        geo_dataset=_FakeGeoDataset(image),
        patch_records=patch_records,
        observation_metadata=observation_metadata,
        image_size=4,
        min_valid_fraction=DEFAULT_PATCH_VALID_FRACTION,
    )

    outputs = save_patch_report(ds, tmp_path, preview_items=1, report_items=1)

    assert outputs["preview_path"].exists()
    assert outputs["summary_path"].exists()
    summary = outputs["summary"]
    assert summary["num_patches"] == 1.0
    assert summary["num_samples"] == 1
    assert summary["num_valid_patches"] == 1
    assert summary["min_valid_fraction_threshold"] == DEFAULT_PATCH_VALID_FRACTION
