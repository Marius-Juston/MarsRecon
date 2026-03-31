"""Tests for observation-level MarsCLIP manifest building."""

from __future__ import annotations

import pathlib
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from clip.observation_manifest import (
    _apply_bbox_filter,
    _choose_image_path,
    _clean_text,
    _normalize_longitude,
    build_observation_manifest,
    build_observation_manifest_from_index,
)


def _make_pdr_mock(df: pd.DataFrame) -> MagicMock:
    mock_data = MagicMock()
    mock_data.__getitem__ = MagicMock(return_value=df)
    return mock_data


def _synthetic_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "OBSERVATION_ID": [
                "OBS_A",
                "OBS_A",
                "OBS_B",
                "OBS_C",
                "OBS_D",
            ],
            "PRODUCT_ID": [
                "OBS_A_COLOR",
                "OBS_A_RED",
                "OBS_B_COLOR",
                "OBS_C_COLOR",
                "OBS_D_COLOR",
            ],
            "FILE_NAME_SPECIFICATION": [
                "VOL/OBS_A/OBS_A_COLOR.JP2",
                "VOL/OBS_A/OBS_A_RED.JP2",
                "VOL/OBS_B/OBS_B_COLOR.JP2",
                "VOL/OBS_C/OBS_C_COLOR.JP2",
                "VOL/OBS_D/OBS_D_COLOR.JP2",
            ],
            "RATIONALE_DESC": [
                "  Olympus   Mons   lava   channels  ",
                "ignored red row",
                "Karzok Crater on Olympus Mons",
                "Missing local image",
                "Crosses antimeridian",
            ],
            "START_TIME": [
                "2007-01-01T00:00:00",
                "2007-01-01T00:00:00",
                "2007-01-02T00:00:00",
                "2007-01-03T00:00:00",
                "2007-01-04T00:00:00",
            ],
            "STOP_TIME": [
                "2007-01-01T00:01:00",
                "2007-01-01T00:01:00",
                "2007-01-02T00:01:00",
                "2007-01-03T00:01:00",
                "2007-01-04T00:01:00",
            ],
            "MINIMUM_LONGITUDE": [224.0, 224.0, 229.5, 240.0, 170.0],
            "MAXIMUM_LONGITUDE": [236.0, 236.0, 230.5, 241.0, 190.0],
            "MINIMUM_LATITUDE": [10.0, 10.0, 18.0, 21.0, 18.0],
            "MAXIMUM_LATITUDE": [12.0, 12.0, 19.0, 22.0, 19.0],
            "IMAGE_LINES": [1000, 1000, 2000, 1500, 1200],
            "LINE_SAMPLES": [300, 300, 500, 400, 350],
            "MAP_SCALE": [0.5, 0.5, 0.8, 1.1, 0.9],
            "MAP_RESOLUTION": [1000.0, 1000.0, 900.0, 800.0, 750.0],
            "EMISSION_ANGLE": [7.0, 7.0, 8.0, 9.0, 10.0],
            "INCIDENCE_ANGLE": [74.0, 74.0, 70.0, 68.0, 66.0],
            "PHASE_ANGLE": [78.0, 78.0, 75.0, 72.0, 69.0],
            "LOCAL_TIME": [15.2, 15.2, 14.3, 13.4, 12.5],
            "SOLAR_LONGITUDE": [92.8, 92.8, 95.0, 97.0, 99.0],
            "SUB_SOLAR_AZIMUTH": [223.7, 223.7, 210.0, 205.0, 200.0],
            "NORTH_AZIMUTH": [270.0, 270.0, 271.0, 272.0, 273.0],
            "SPACECRAFT_ALTITUDE": [300.0, 300.0, 350.0, 400.0, 450.0],
            "STEREO_FLAG": ["YES", "YES", "NO", "NO", "NO"],
            "PROJECTION_CENTER_LATITUDE": [11.0, 11.0, 18.5, 21.5, 18.5],
            "PROJECTION_CENTER_LONGITUDE": [230.0, 230.0, 230.0, 240.5, 180.0],
        }
    )


def _touch(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_normalize_longitude_wraps_pds_range():
    assert _normalize_longitude(224.0) == pytest.approx(-136.0)
    assert _normalize_longitude(180.0) == pytest.approx(-180.0)
    assert _normalize_longitude(360.0) == pytest.approx(0.0)


def test_clean_text_collapses_repeated_whitespace():
    assert _clean_text("  Olympus   Mons \n lava  ") == "Olympus Mons lava"


def test_apply_bbox_filter_matches_dataset_semantics():
    df = _synthetic_index()
    filtered = _apply_bbox_filter(df, bbox=(-137.0, 9.0, -125.0, 13.0))
    assert filtered["OBSERVATION_ID"].tolist() == ["OBS_A", "OBS_A"]


def test_build_manifest_prefers_cog_and_derives_fields(tmp_path):
    df = _synthetic_index()
    _touch(tmp_path / "images" / "OBS_A_COLOR.JP2")
    _touch(tmp_path / "images" / "OBS_A_COLOR.tif")
    _touch(tmp_path / "images" / "OBS_B_COLOR.JP2")

    manifest = build_observation_manifest_from_index(df, tmp_path)

    assert manifest["obs_id"].tolist() == ["OBS_A", "OBS_B"]

    obs_a = manifest.loc[manifest["obs_id"] == "OBS_A"].iloc[0]
    assert obs_a["image_format"] == "tif"
    assert obs_a["image_path"] == str(tmp_path / "images" / "OBS_A_COLOR.tif")
    assert obs_a["color_jp2_path"] == str(tmp_path / "images" / "OBS_A_COLOR.JP2")
    assert obs_a["color_tif_path"] == str(tmp_path / "images" / "OBS_A_COLOR.tif")
    assert bool(obs_a["has_local_jp2"])
    assert bool(obs_a["has_local_tif"])
    assert bool(obs_a["has_local_image"])
    assert obs_a["rationale_desc"] == "Olympus Mons lava channels"
    assert obs_a["min_lon"] == pytest.approx(-136.0)
    assert obs_a["max_lon"] == pytest.approx(-124.0)
    assert obs_a["centroid_lon"] == pytest.approx(-130.0)
    assert obs_a["centroid_lat"] == pytest.approx(11.0)
    assert obs_a["lon_span_deg"] == pytest.approx(12.0)
    assert obs_a["lat_span_deg"] == pytest.approx(2.0)
    assert obs_a["bbox_area_deg2"] == pytest.approx(24.0)
    assert obs_a["image_lines"] == 1000
    assert obs_a["line_samples"] == 300
    assert obs_a["map_scale"] == pytest.approx(0.5)
    assert obs_a["map_resolution"] == pytest.approx(1000.0)
    assert obs_a["emission_angle"] == pytest.approx(7.0)
    assert obs_a["incidence_angle"] == pytest.approx(74.0)
    assert obs_a["phase_angle"] == pytest.approx(78.0)
    assert obs_a["local_time"] == pytest.approx(15.2)
    assert obs_a["solar_longitude"] == pytest.approx(92.8)
    assert obs_a["sub_solar_azimuth"] == pytest.approx(223.7)
    assert obs_a["north_azimuth"] == pytest.approx(270.0)
    assert obs_a["spacecraft_altitude"] == pytest.approx(300.0)
    assert obs_a["stereo_flag"] == "YES"
    assert bool(obs_a["is_stereo"])
    assert obs_a["projection_center_latitude"] == pytest.approx(11.0)
    assert obs_a["projection_center_longitude"] == pytest.approx(-130.0)
    assert bool(obs_a["has_near_infrared"])
    assert bool(obs_a["has_red"])
    assert bool(obs_a["has_blue_green"])
    assert obs_a["channel_count"] == 3
    assert obs_a["start_time"] == pd.Timestamp("2007-01-01T00:00:00Z")
    assert obs_a["stop_time"] == pd.Timestamp("2007-01-01T00:01:00Z")

    obs_b = manifest.loc[manifest["obs_id"] == "OBS_B"].iloc[0]
    assert obs_b["image_format"] == "jp2"
    assert obs_b["image_path"] == str(tmp_path / "images" / "OBS_B_COLOR.JP2")
    assert obs_b["color_tif_path"] is None
    assert not bool(obs_b["has_local_tif"])


def test_build_manifest_uses_jp2_when_cog_preference_disabled(tmp_path):
    df = _synthetic_index().iloc[:1].copy()
    _touch(tmp_path / "images" / "OBS_A_COLOR.JP2")
    _touch(tmp_path / "images" / "OBS_A_COLOR.tif")

    manifest = build_observation_manifest_from_index(
        df,
        tmp_path,
        prefer_cog=False,
    )

    assert manifest.loc[0, "image_path"] == str(tmp_path / "images" / "OBS_A_COLOR.JP2")
    assert manifest.loc[0, "image_format"] == "jp2"


def test_build_manifest_bbox_filters_observations(tmp_path):
    df = _synthetic_index()
    _touch(tmp_path / "images" / "OBS_A_COLOR.JP2")
    _touch(tmp_path / "images" / "OBS_B_COLOR.JP2")

    manifest = build_observation_manifest_from_index(
        df,
        tmp_path,
        bbox=(-137.0, 9.0, -125.0, 13.0),
    )

    assert manifest["obs_id"].tolist() == ["OBS_A"]


def test_build_manifest_can_keep_rows_without_local_image(tmp_path):
    df = _synthetic_index().iloc[[2, 3]].copy()
    _touch(tmp_path / "images" / "OBS_B_COLOR.JP2")

    manifest = build_observation_manifest_from_index(
        df,
        tmp_path,
        require_local_image=False,
    )

    assert manifest["obs_id"].tolist() == ["OBS_B", "OBS_C"]
    obs_c = manifest.loc[manifest["obs_id"] == "OBS_C"].iloc[0]
    assert obs_c["image_path"] is None
    assert obs_c["image_format"] is None
    assert obs_c["color_jp2_path"] is None
    assert obs_c["color_tif_path"] is None
    assert not bool(obs_c["has_local_image"])


def test_build_manifest_skips_antimeridian_rows(tmp_path, caplog):
    df = _synthetic_index().iloc[[4]].copy()
    with caplog.at_level("WARNING", logger="clip.observation_manifest"):
        manifest = build_observation_manifest_from_index(
            df,
            tmp_path,
            require_local_image=False,
        )
    assert manifest.empty
    assert "antimeridian" in caplog.text.lower()


def test_build_observation_manifest_loads_index_from_disk(tmp_path):
    df = _synthetic_index().iloc[[0]].copy()
    _touch(tmp_path / "RDRCUMINDEX.LBL")
    _touch(tmp_path / "images" / "OBS_A_COLOR.JP2")
    mock_pdr = _make_pdr_mock(df)

    with patch("clip.observation_manifest.pdr.read", return_value=mock_pdr):
        manifest = build_observation_manifest(tmp_path)

    assert manifest["obs_id"].tolist() == ["OBS_A"]
    mock_pdr.load.assert_called_once_with("all")


@pytest.mark.integration
def test_real_olympus_color_manifest_counts():
    root = pathlib.Path("/scratch/mars_hirise")
    manifest = build_observation_manifest(
        root,
        bbox=(-136.0, 12.0, -124.0, 24.0),
    )

    assert len(manifest) == 419
    assert manifest["obs_id"].nunique() == 419
    assert manifest["image_format"].isin(["tif", "jp2"]).all()
    assert manifest["rationale_desc"].str.len().gt(0).all()


def test_choose_image_path_returns_tif_when_prefer_cog_false_and_only_tif_available(tmp_path):
    jp2_path = tmp_path / "obs.jp2"
    tif_path = tmp_path / "obs.tif"
    result = _choose_image_path(
        jp2_path=jp2_path,
        tif_path=tif_path,
        has_local_jp2=False,
        has_local_tif=True,
        prefer_cog=False,
    )
    assert result == tif_path
