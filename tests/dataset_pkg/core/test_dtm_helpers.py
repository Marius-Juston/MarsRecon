"""Unit tests for :mod:`dataset.core.dtm` helpers.

Focus is on pure / quasi-pure methods that don't require a full
``MarsHiRISEDTM`` instance with PDS data — the static
``_percentile_stretch``, ``_pick_footprint_file``, ``_merge_elevation_tiles``,
init-time validation, cache key helpers, and ``_assign_ortho_path``.

Tests that would need real .IMG / .JP2 files are marked integration and
excluded from the default unit suite.
"""

from __future__ import annotations

import json
import math
import pathlib
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from dataset.core.dtm import MarsHiRISEDTM


# ---------------------------------------------------------------------------
# _percentile_stretch — static, pure numpy
# ---------------------------------------------------------------------------


class TestPercentileStretch:
    def test_uniform_image_returns_copy_unchanged(self):
        img = np.full((4, 4), 0.5, dtype=np.float32)
        out = MarsHiRISEDTM._percentile_stretch(img)
        # p2 == p98 → no stretch applied; output is just a copy
        np.testing.assert_array_equal(out, img)
        # Must be a copy, not the input
        assert out is not img

    def test_2d_stretch_maps_to_unit_interval(self):
        rng = np.random.default_rng(0)
        img = rng.uniform(0.1, 0.9, size=(32, 32)).astype(np.float32)
        out = MarsHiRISEDTM._percentile_stretch(img)
        # Stretch must keep values inside [0, 1]
        assert out.min() >= 0.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_nodata_zero_pixels_remain_zero_after_stretch(self):
        img = np.full((8, 8), 0.0, dtype=np.float32)
        img[2:4, 2:4] = np.linspace(0.2, 0.8, 4).reshape(2, 2)
        out = MarsHiRISEDTM._percentile_stretch(img)
        # Border zero pixels remain zero (nodata mask: |x| <= eps)
        assert out[0, 0] == 0.0
        assert out[7, 7] == 0.0

    def test_3d_per_channel_stretch(self):
        rng = np.random.default_rng(7)
        img = rng.uniform(0.05, 0.95, size=(16, 16, 3)).astype(np.float32)
        out = MarsHiRISEDTM._percentile_stretch(img)
        # All channels independently stretched into [0, 1]
        for c in range(3):
            assert out[..., c].min() >= 0.0 - 1e-6
            assert out[..., c].max() <= 1.0 + 1e-6

    def test_all_zero_3d_returns_unchanged(self):
        img = np.zeros((4, 4, 3), dtype=np.float32)
        out = MarsHiRISEDTM._percentile_stretch(img)
        np.testing.assert_array_equal(out, img)


# ---------------------------------------------------------------------------
# _pick_footprint_file — first-existing-file-wins selector
# ---------------------------------------------------------------------------


class TestPickFootprintFile:
    def test_returns_none_when_all_paths_missing(self, tmp_path):
        rec = {
            "dtm_path": str(tmp_path / "ghost_dtm.img"),
            "left_red_path": str(tmp_path / "ghost_lred.jp2"),
            "right_red_path": None,
        }
        assert MarsHiRISEDTM._pick_footprint_file(rec) is None

    def test_prefers_dtm_path_when_exists(self, tmp_path):
        dtm = tmp_path / "real.img"
        dtm.write_bytes(b"")
        lred = tmp_path / "real_lred.jp2"
        lred.write_bytes(b"")
        rec = {
            "dtm_path": str(dtm),
            "left_red_path": str(lred),
        }
        assert MarsHiRISEDTM._pick_footprint_file(rec) == str(dtm)

    def test_falls_back_to_ortho_when_dtm_missing(self, tmp_path):
        # dtm doesn't exist but left_red does
        ortho = tmp_path / "lred.jp2"
        ortho.write_bytes(b"")
        rec = {
            "dtm_path": str(tmp_path / "missing.img"),
            "left_red_path": str(ortho),
            "right_red_path": None,
        }
        assert MarsHiRISEDTM._pick_footprint_file(rec) == str(ortho)

    def test_cog_sidecar_counts_as_existing(self, tmp_path):
        # Underlying .IMG missing but .tif sidecar present
        sidecar = tmp_path / "real.tif"
        sidecar.write_bytes(b"")
        rec = {"dtm_path": str(tmp_path / "real.img")}
        assert MarsHiRISEDTM._pick_footprint_file(rec) == str(tmp_path / "real.img")

    def test_none_values_are_skipped(self, tmp_path):
        rec = {k: None for k in (
            "dtm_path", "left_red_path", "right_red_path",
            "left_irb_path", "right_irb_path",
        )}
        assert MarsHiRISEDTM._pick_footprint_file(rec) is None


# ---------------------------------------------------------------------------
# _merge_elevation_tiles — first-valid-wins mosaic
# ---------------------------------------------------------------------------


class TestMergeElevationTiles:
    def test_single_tile_returned_as_is(self):
        t = torch.full((1, 4, 4), 100.0)
        out = MarsHiRISEDTM._merge_elevation_tiles([t])
        assert out is t

    def test_two_tiles_first_valid_wins(self):
        t1 = torch.full((1, 4, 4), 100.0)
        t2 = torch.full((1, 4, 4), 200.0)
        out = MarsHiRISEDTM._merge_elevation_tiles([t1, t2])
        # t1 fills first → those positions stay 100.0
        assert torch.all(out == 100.0)

    def test_nan_in_first_tile_filled_by_second(self):
        t1 = torch.full((1, 4, 4), float("nan"))
        t2 = torch.full((1, 4, 4), 50.0)
        out = MarsHiRISEDTM._merge_elevation_tiles([t1, t2])
        assert torch.all(out == 50.0)

    def test_different_shapes_use_max_dims(self):
        t1 = torch.full((1, 4, 4), 10.0)
        t2 = torch.full((1, 6, 6), 20.0)
        out = MarsHiRISEDTM._merge_elevation_tiles([t1, t2])
        assert out.shape == (1, 6, 6)
        # t1 region: 10.0 (first valid)
        assert torch.all(out[:, :4, :4] == 10.0)
        # Outside t1's region: filled by t2
        assert out[0, 5, 5].item() == 20.0


# ---------------------------------------------------------------------------
# Init-time validation — does not require touching any PDS data
# ---------------------------------------------------------------------------


class TestInitValidation:
    def test_invalid_ortho_type_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid ortho_type"):
            with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
                MarsHiRISEDTM(root=str(tmp_path), ortho_type="MAGENTA")

    def test_missing_stats_path_raises_when_normalize_requested(self, tmp_path):
        with pytest.raises(ValueError, match="elevation_stats_path"):
            with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
                MarsHiRISEDTM(
                    root=str(tmp_path),
                    normalize_elevation=True,
                    elevation_stats_path=None,
                )

    def test_nonexistent_stats_path_raises(self, tmp_path):
        with pytest.raises(ValueError, match="elevation_stats_path"):
            with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
                MarsHiRISEDTM(
                    root=str(tmp_path),
                    normalize_elevation=True,
                    elevation_stats_path=str(tmp_path / "ghost.json"),
                )

    def test_valid_stats_path_loads_mean_std(self, tmp_path):
        stats = tmp_path / "stats.json"
        stats.write_text(json.dumps({"mean": 1234.5, "std": 67.8}))
        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(
                root=str(tmp_path),
                normalize_elevation=True,
                elevation_stats_path=str(stats),
            )
        assert math.isclose(ds._elev_mean, 1234.5)
        assert math.isclose(ds._elev_std, 67.8)

    @pytest.mark.parametrize("ortho_type", ["RED", "IRB", ["RED", "IRB"]])
    def test_valid_ortho_types_accepted(self, tmp_path, ortho_type):
        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(root=str(tmp_path), ortho_type=ortho_type)
        expected = [ortho_type] if isinstance(ortho_type, str) else ortho_type
        assert ds.ortho_types == expected

    def test_ortho_scale_uppercased(self, tmp_path):
        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(root=str(tmp_path), ortho_scale="b")
        assert ds.ortho_scale == "B"

    def test_ortho_scale_none_kept_none(self, tmp_path):
        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(root=str(tmp_path), ortho_scale=None)
        assert ds.ortho_scale is None


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------


class TestCacheKeyHelpers:
    def test_cache_version_constant(self, tmp_path):
        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(root=str(tmp_path))
        assert ds._cache_version() == "dtm_v1"

    def test_cache_suffix_no_ortho_returns_empty(self, tmp_path):
        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(root=str(tmp_path), include_ortho=False)
        assert ds._cache_suffix_parts() == []

    def test_cache_suffix_with_ortho_and_no_scale(self, tmp_path):
        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(
                root=str(tmp_path), include_ortho=True, ortho_type=["IRB", "RED"]
            )
        # Sorted alphabetically: IRB, RED
        assert ds._cache_suffix_parts() == ["IRB_RED"]

    def test_cache_suffix_with_ortho_and_scale(self, tmp_path):
        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(
                root=str(tmp_path), include_ortho=True,
                ortho_type="RED", ortho_scale="A",
            )
        assert ds._cache_suffix_parts() == ["RED", "sA"]


# ---------------------------------------------------------------------------
# _assign_ortho_path — slot resolution logic
# ---------------------------------------------------------------------------


def _make_ds(tmp_path, ortho_scale=None):
    with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
        ds = MarsHiRISEDTM(root=str(tmp_path), ortho_scale=ortho_scale)
    return ds


def _orow(color="RED", scale="A", obs_id="ESP_001", data_type="ORTHOIMAGE-LEFT",
          local_path="/tmp/foo.jp2"):
    return pd.Series({
        "_ortho_color": color,
        "_ortho_scale": scale,
        "_ortho_obs_id": obs_id,
        "_data_type": data_type,
        "_local_path": local_path,
    })


class TestAssignOrthoPath:
    def test_invalid_color_is_skipped(self, tmp_path):
        ds = _make_ds(tmp_path)
        rec = {}
        ds._assign_ortho_path(rec, _orow(color="MAGENTA"), "L1", "R1")
        assert rec == {}

    def test_left_data_type_assigns_left_slot(self, tmp_path):
        ds = _make_ds(tmp_path)
        rec = {}
        orow = _orow(data_type="ORTHO-LEFT-RDR", color="RED",
                     local_path="/data/x.jp2")
        ds._assign_ortho_path(rec, orow, "L1", "R1")
        assert rec["left_red_path"] == "/data/x.jp2"
        assert rec["left_red_path_scale"] == "A"

    def test_right_data_type_assigns_right_slot(self, tmp_path):
        ds = _make_ds(tmp_path)
        rec = {}
        orow = _orow(data_type="ORTHO-RIGHT", color="IRB",
                     local_path="/data/r.jp2")
        ds._assign_ortho_path(rec, orow, "L1", "R1")
        assert rec["right_irb_path"] == "/data/r.jp2"

    def test_obs_id_match_disambiguates_side_when_data_type_silent(self, tmp_path):
        ds = _make_ds(tmp_path)
        rec = {}
        orow = _orow(data_type="ORTHOIMAGE", color="RED", obs_id="L1",
                     local_path="/data/x.jp2")
        ds._assign_ortho_path(rec, orow, "L1", "R1")
        assert rec["left_red_path"] == "/data/x.jp2"

    def test_unrelated_obs_id_skipped(self, tmp_path):
        ds = _make_ds(tmp_path)
        rec = {}
        orow = _orow(data_type="ORTHOIMAGE", color="RED", obs_id="OTHER")
        ds._assign_ortho_path(rec, orow, "L1", "R1")
        assert rec == {}

    def test_exact_scale_preference_wins(self, tmp_path):
        ds = _make_ds(tmp_path, ortho_scale="B")
        rec = {"left_red_path": "/old.jp2", "left_red_path_scale": "C"}
        # New row with scale=B (preferred) → overwrites
        orow = _orow(data_type="ORTHO-LEFT", color="RED", scale="B",
                     local_path="/new.jp2")
        ds._assign_ortho_path(rec, orow, "L1", "R1")
        assert rec["left_red_path"] == "/new.jp2"
        assert rec["left_red_path_scale"] == "B"

    def test_finer_scale_wins_when_no_preference(self, tmp_path):
        ds = _make_ds(tmp_path)  # no ortho_scale preference
        rec = {"left_red_path": "/old.jp2", "left_red_path_scale": "C"}
        # Scale A (finer) replaces C
        orow = _orow(data_type="ORTHO-LEFT", color="RED", scale="A",
                     local_path="/finer.jp2")
        ds._assign_ortho_path(rec, orow, "L1", "R1")
        assert rec["left_red_path"] == "/finer.jp2"
        assert rec["left_red_path_scale"] == "A"

    def test_coarser_scale_kept_when_finer_already_set(self, tmp_path):
        ds = _make_ds(tmp_path)
        rec = {"left_red_path": "/finer.jp2", "left_red_path_scale": "A"}
        orow = _orow(data_type="ORTHO-LEFT", color="RED", scale="D",
                     local_path="/coarser.jp2")
        ds._assign_ortho_path(rec, orow, "L1", "R1")
        assert rec["left_red_path"] == "/finer.jp2"

    def test_first_assignment_when_no_current_path(self, tmp_path):
        ds = _make_ds(tmp_path, ortho_scale="A")
        rec = {}
        # New row scale ≠ preferred but rec has no current path → assigned
        orow = _orow(data_type="ORTHO-LEFT", color="RED", scale="C",
                     local_path="/x.jp2")
        ds._assign_ortho_path(rec, orow, "L1", "R1")
        assert rec["left_red_path"] == "/x.jp2"


# ---------------------------------------------------------------------------
# _post_download_verify — warning on missing files
# ---------------------------------------------------------------------------


class TestPostDownloadVerify:
    @staticmethod
    def _make_index(tmp_path, paths):
        # geopandas-free GeoDataFrame mock — only iterrows / sample / len are used
        return pd.DataFrame({"dtm_path": paths})

    def test_no_files_no_download_logs_help_message(self, tmp_path, caplog):
        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(root=str(tmp_path))
        ds.index = self._make_index(tmp_path, [str(tmp_path / "ghost.img")] * 3)
        ds.download = False
        import logging
        with caplog.at_level(logging.WARNING, logger="dataset.core.dtm"):
            ds._post_download_verify()
        assert "No DTM .IMG files found" in caplog.text

    def test_no_files_after_download_logs_disk_warning(self, tmp_path, caplog):
        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(root=str(tmp_path))
        ds.index = self._make_index(tmp_path, [str(tmp_path / "ghost.img")] * 2)
        ds.download = True
        import logging
        with caplog.at_level(logging.WARNING, logger="dataset.core.dtm"):
            ds._post_download_verify()
        assert "Download completed but no DTM" in caplog.text

    def test_existing_files_silent(self, tmp_path, caplog):
        real = tmp_path / "real.img"
        real.write_bytes(b"")
        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(root=str(tmp_path))
        ds.index = self._make_index(tmp_path, [str(real)] * 5)
        ds.download = False
        import logging
        with caplog.at_level(logging.WARNING, logger="dataset.core.dtm"):
            ds._post_download_verify()
        # No warning emitted
        assert "No DTM" not in caplog.text
        assert "Download completed" not in caplog.text


# ---------------------------------------------------------------------------
# _get_ortho_overlap — exception path
# ---------------------------------------------------------------------------


class TestGetOrthoOverlap:
    def test_nonexistent_path_returns_zero(self, tmp_path):
        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(root=str(tmp_path))
        from shapely.geometry import box
        dtm_geom = box(-131, 18, -130, 19)
        overlap = ds._get_ortho_overlap(dtm_geom, str(tmp_path / "ghost.jp2"))
        assert overlap == 0.0

    def test_zero_dtm_area_returns_zero(self, tmp_path):
        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(root=str(tmp_path))
        from shapely.geometry import Point
        zero_area = Point(0, 0)  # has area=0
        # Even if path doesn't exist, the early-return on missing path triggers
        overlap = ds._get_ortho_overlap(zero_area, str(tmp_path / "ghost.jp2"))
        assert overlap == 0.0

    def test_exception_returns_zero(self, tmp_path):
        """A bogus file that rasterio can't open should yield 0.0, not crash."""
        bad = tmp_path / "bad.jp2"
        bad.write_bytes(b"not a jp2")
        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(root=str(tmp_path))
        from shapely.geometry import box
        overlap = ds._get_ortho_overlap(box(-131, 18, -130, 19), str(bad))
        assert overlap == 0.0
