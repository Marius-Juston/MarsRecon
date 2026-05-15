"""Targeted unit tests for low-level helpers in dataset.core.base.

Closes the remaining coverage gaps:
  * reproject_band src_nodata branch (base.py:553)
  * MarsHiRISEBase._geometry_from_footprint_result fallbacks
    (base.py:940 invalid-hull buffer(0), 946 file-bbox fallback)
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.crs import CRS as RasterioCRS
from shapely.geometry import Polygon

from dataset.core.base import MarsHiRISEBase, reproject_band
from dataset.core.rdr import MarsHiRISE

_MARS_RCRS = RasterioCRS.from_proj4(
    "+proj=longlat +a=3396190 +b=3376200 +no_defs"
)


@pytest.fixture
def open_src(tmp_path):
    p = tmp_path / "src.tif"
    transform = rasterio.transform.from_bounds(-131.0, 18.0, -130.0, 19.0, 16, 16)
    data = np.ones((1, 16, 16), dtype=np.float32) * 7.0
    with rasterio.open(
        p, "w", driver="GTiff", count=1, dtype="float32",
        width=16, height=16, crs=_MARS_RCRS, transform=transform,
    ) as dst:
        dst.write(data)
    src = rasterio.open(p)
    yield src
    src.close()


class TestReprojectBand:
    def test_src_nodata_passed_through(self, open_src):
        """src_nodata not None → kwargs gets src_nodata (base.py:553)."""
        dst_tf = rasterio.transform.from_bounds(-131.0, 18.0, -130.0, 19.0, 8, 8)
        out = reproject_band(
            open_src, 1, _MARS_RCRS, dst_tf, 8, 8,
            src_nodata=-9999.0,
        )
        assert out.shape == (8, 8)
        assert out.dtype == np.float32

    def test_src_crs_none_uses_dst(self, tmp_path):
        """src_dataset.crs None → src_crs = dst_crs (base.py:537-538)."""
        p = tmp_path / "nocrs.tif"
        transform = rasterio.transform.from_bounds(-131.0, 18.0, -130.0, 19.0, 8, 8)
        with rasterio.open(
            p, "w", driver="GTiff", count=1, dtype="float32",
            width=8, height=8, transform=transform,
        ) as dst:
            dst.write(np.ones((1, 8, 8), dtype=np.float32))
        with rasterio.open(p) as src:
            dst_tf = rasterio.transform.from_bounds(-131.0, 18.0, -130.0, 19.0, 4, 4)
            out = reproject_band(src, 1, _MARS_RCRS, dst_tf, 4, 4)
        assert out.shape == (4, 4)


def _ref_row():
    return pd.Series(
        {
            "CORNER1_LATITUDE": -5.0, "CORNER1_LONGITUDE": 224.0,
            "CORNER2_LATITUDE": -5.0, "CORNER2_LONGITUDE": 236.0,
            "CORNER3_LATITUDE": 5.0, "CORNER3_LONGITUDE": 236.0,
            "CORNER4_LATITUDE": 5.0, "CORNER4_LONGITUDE": 224.0,
            "MINIMUM_LATITUDE": -5.0, "MAXIMUM_LATITUDE": 5.0,
            "MINIMUM_LONGITUDE": 224.0, "MAXIMUM_LONGITUDE": 236.0,
        }
    )


class TestBaseDefaults:
    @pytest.fixture
    def ds(self, tmp_path):
        with patch.object(MarsHiRISE, "_verify", return_value=None):
            return MarsHiRISE(root=tmp_path)

    def test_cache_suffix_parts_default_empty(self, ds):
        # RDR does not override _cache_suffix_parts
        assert ds._cache_suffix_parts() == []

    def test_base_cache_version_default(self, ds):
        """Base default _cache_version ("v3"); subclasses override (base.py:813)."""
        assert MarsHiRISEBase._cache_version(ds) == "v3"

    def test_fractional_bbox_value_str_branch(self, ds):
        """Non-integer-valued float in bbox → str(v) branch (base.py:792)."""
        ds.target = None
        ds.bbox = (-136.5, 12.25, -124.0, 24.0)
        name = ds.spatial_index_cache.name
        assert "-136.5" in name
        assert "12.25" in name

    def test_base_post_download_verify_default_is_noop(self, ds):
        """Base default _post_download_verify is a no-op (base.py:699)."""
        assert MarsHiRISEBase._post_download_verify(ds) is None

    def test_try_load_cache_false_when_reuse_disabled(self, ds):
        """reuse_cache False short-circuits to False (base.py:834)."""
        ds.reuse_cache = False
        ds.index = None
        assert ds._try_load_cache() is False


class TestGeometryFromFootprintResult:
    @pytest.fixture
    def ds(self, tmp_path):
        with patch.object(MarsHiRISE, "_verify", return_value=None):
            return MarsHiRISE(root=tmp_path)

    def test_valid_hull_used_directly(self, ds):
        coords = [(-136.0, -5.0), (-124.0, -5.0), (-124.0, 5.0), (-136.0, 5.0)]
        geom = ds._geometry_from_footprint_result(coords, None, _ref_row())
        assert isinstance(geom, Polygon)
        assert geom.area == pytest.approx(120.0)

    def test_invalid_hull_repaired_with_buffer0(self, ds):
        """Self-intersecting bowtie hull → buffer(0) repair (base.py:940)."""
        bowtie = [(0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (0.0, 2.0)]
        geom = ds._geometry_from_footprint_result(bowtie, None, _ref_row())
        assert geom is not None
        assert geom.is_valid

    def test_degenerate_hull_falls_back_to_file_bounds(self, ds):
        """Collinear hull (zero area) → file bbox fallback (base.py:946)."""
        collinear = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        fb = (-136.0, -5.0, -124.0, 5.0)
        geom = ds._geometry_from_footprint_result(collinear, fb, _ref_row())
        assert isinstance(geom, Polygon)
        assert geom.bounds == pytest.approx((-136.0, -5.0, -124.0, 5.0))

    def test_no_hull_no_bounds_falls_back_to_corners(self, ds):
        geom = ds._geometry_from_footprint_result(None, None, _ref_row())
        assert isinstance(geom, Polygon)
        assert geom.is_valid

    def test_antimeridian_returns_none(self, ds):
        row = _ref_row()
        # Drop corner cols so corners_to_polygon returns None, then make
        # min/max straddle the antimeridian (lon_min > lon_max).
        for i in (1, 2, 3, 4):
            del row[f"CORNER{i}_LONGITUDE"]
        row["MINIMUM_LONGITUDE"] = 170.0
        row["MAXIMUM_LONGITUDE"] = 190.0  # → -170 after normalisation
        geom = ds._geometry_from_footprint_result(None, None, row)
        assert geom is None
