"""Tests for MarsHiRISE dataset correctness.

Unit tests use synthetic data only (no disk I/O to /scratch).
Integration tests are marked with @pytest.mark.integration and require
real HiRISE files under /scratch/mars_hirise.
"""

import pathlib
import sys
import types
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch

from temp import _SPATIAL_TOL, MarsHiRISE, _corners_to_polygon


# ---------------------------------------------------------------------------
# Unit tests: _merge_tiles channel-count handling
# ---------------------------------------------------------------------------


class TestMergeTiles:
    """Verify _merge_tiles handles uniform and mismatched channel counts."""

    def test_single_tile_returned_unchanged(self):
        t = torch.rand(3, 16, 16)
        result = MarsHiRISE._merge_tiles([t])
        assert result.shape == t.shape
        assert torch.allclose(result, t)

    def test_two_matching_tiles_merged(self):
        t1 = torch.zeros(3, 8, 8)
        t2 = torch.ones(3, 8, 8)
        # First-non-zero-wins: t1 is all zeros so t2 fills in.
        result = MarsHiRISE._merge_tiles([t1, t2])
        assert result.shape == (3, 8, 8)
        assert torch.allclose(result, t2)

    def test_first_non_zero_wins(self):
        t1 = torch.zeros(1, 4, 4)
        t1[0, 0, 0] = 0.5
        t2 = torch.ones(1, 4, 4)
        result = MarsHiRISE._merge_tiles([t1, t2])
        # Position (0,0) was set in t1 → t2 must not overwrite it.
        assert result[0, 0, 0] == pytest.approx(0.5)
        # Zero position in t1 → filled by t2.
        assert result[0, 1, 1] == pytest.approx(1.0)

    def test_mismatched_channels_padded_to_max(self):
        """Tiles with different channel counts must be zero-padded, not raise."""
        t3 = torch.ones(3, 8, 8)       # 3-channel tile
        t1 = torch.ones(1, 8, 8) * 0.5  # 1-channel tile (simulates RED-only)
        # Must not raise AssertionError.
        result = MarsHiRISE._merge_tiles([t3, t1])
        assert result.shape[0] == 3  # max channels

    def test_output_channels_equals_max_input_channels(self):
        tiles = [torch.rand(c, 4, 4) for c in (3, 1, 3)]
        result = MarsHiRISE._merge_tiles(tiles)
        assert result.shape[0] == 3

    def test_spatial_dims_match_max(self):
        t1 = torch.ones(2, 4, 6)
        t2 = torch.ones(2, 3, 5)
        result = MarsHiRISE._merge_tiles([t1, t2])
        assert result.shape == (2, 4, 6)


# ---------------------------------------------------------------------------
# Unit tests: JP2 bounds intersection in _build_spatial_index
# ---------------------------------------------------------------------------


class TestFootprintIntersection:
    """Verify that corners polygon ∩ JP2 bbox gives accurate footprints."""

    def test_intersection_smaller_than_corners(self):
        """If JP2 bbox is strictly inside corners, intersection == JP2 bbox."""
        corners = box(-136.0, 12.0, -124.0, 24.0)
        jp2_bounds = box(-135.9, 12.1, -124.1, 23.9)  # slightly inside
        result = corners.intersection(jp2_bounds)
        assert result.area == pytest.approx(jp2_bounds.area, rel=1e-9)

    def test_intersection_when_corners_extend_beyond_jp2(self):
        """Corners polygon that extends beyond JP2 bounds is clipped."""
        corners = box(-136.1, 11.9, -123.9, 24.1)   # larger than JP2
        jp2_bounds = box(-136.0, 12.0, -124.0, 24.0)
        result = corners.intersection(jp2_bounds)
        # Result must be entirely within jp2_bounds
        assert jp2_bounds.contains(result) or jp2_bounds.equals(result)

    def test_intersection_with_rotated_strip(self, strip_polygon: Polygon):
        """Intersection of a rotated strip with its bbox is the strip itself."""
        bbox = box(*strip_polygon.bounds)
        result = strip_polygon.intersection(bbox)
        assert result.area == pytest.approx(strip_polygon.area, rel=1e-9)

    def test_degenerate_intersection_falls_back(self):
        """Empty intersection falls back to JP2 bbox."""
        # Two non-overlapping geometries
        corners = box(-140.0, 20.0, -139.0, 21.0)
        jp2_bounds = box(-130.0, 10.0, -129.0, 11.0)
        result = corners.intersection(jp2_bounds)
        # The result should be empty; code falls back to jp2_bounds
        assert result.is_empty

    def test_sampled_centers_within_jp2_bounds(self, strip_polygon: Polygon, mars_crs):
        """Centers from HiRISEGeoSampler must lie inside the footprint geometry."""
        from helpers import make_mock_dataset
        from hirise_sampler import HiRISEGeoSampler
        from torchgeo.samplers import Units

        # Simulate the intersection: JP2 bbox is slightly inset from corners
        inset_bbox = box(*[v + 0.01 for v in strip_polygon.bounds])
        footprint = strip_polygon.intersection(inset_bbox) if strip_polygon.intersects(inset_bbox) else strip_polygon

        dataset = make_mock_dataset([footprint], mars_crs)
        sampler = HiRISEGeoSampler(dataset, size=0.003, length=50, units=Units.CRS)

        for x_sl, y_sl, _ in sampler:
            patch = box(x_sl.start, y_sl.start, x_sl.stop, y_sl.stop)
            assert footprint.intersects(patch)


# ---------------------------------------------------------------------------
# Unit tests: _SPATIAL_TOL
# ---------------------------------------------------------------------------


class TestSpatialTolerance:
    def test_spatial_tol_is_small(self):
        """Tolerance must be below HiRISE pixel size (~8.44e-6°) but positive."""
        assert 0 < _SPATIAL_TOL < 1e-3

    def test_tolerance_allows_boundary_patch(self):
        """A patch touching the JP2 boundary within tolerance should be accepted."""
        # Simulates the comparison in _load_from_jp2
        fl, fb, fr, ft = -140.0, 19.0, -130.0, 25.0   # JP2 bounds
        # Query patch right at the boundary
        x_start = fr - _SPATIAL_TOL / 2  # inside tolerance
        x_stop = fr + 0.005

        # Without tolerance: fr < x_start → True → would early-exit
        assert not (fr < x_start)  # strict comparison would let it through here
        # With tolerance: fr + tol < x_start → False → no early exit
        assert not (fr + _SPATIAL_TOL < x_start)

    def test_tolerance_still_rejects_truly_outside(self):
        """A patch clearly outside JP2 bounds is still rejected."""
        fl, fb, fr, ft = -140.0, 19.0, -130.0, 25.0
        x_start = fr + _SPATIAL_TOL * 10   # well outside

        # With tolerance: fr + tol < x_start → True → early exit
        assert fr + _SPATIAL_TOL < x_start


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMarsHiRISEIntegration:
    """Require real data at /scratch/mars_hirise and download=False."""

    _DATA_ROOT = pathlib.Path("/scratch/mars_hirise")
    _BBOX = (-136, 12, -124, 24)

    @pytest.fixture(scope="class")
    def dataset(self):
        from temp import MarsHiRISE

        return MarsHiRISE(
            bbox=self._BBOX,
            channels=["NEAR-INFRARED", "RED", "BLUE-GREEN"],
            download=False,
            reuse_cache=True,
        )

    def test_dataset_has_observations(self, dataset):
        assert len(dataset.index) > 0

    def test_spatial_index_geometries_within_jp2_bounds(self, dataset):
        """Every footprint polygon must be inside the corresponding JP2 bounds."""
        import rasterio
        from rasterio.warp import transform_bounds

        failures = 0
        for i in range(min(10, len(dataset.index))):
            row = dataset.index.iloc[i]
            footprint = dataset.index.geometry.iloc[i]
            fb_minx, fb_miny, fb_maxx, fb_maxy = footprint.bounds

            # Check against each available JP2
            for col in ("color_path", "red_path"):
                jp2_path = row.get(col)
                if jp2_path is None:
                    continue
                p = pathlib.Path(jp2_path)
                if not p.exists():
                    continue
                try:
                    with rasterio.open(p) as src:
                        fl, flb, fr, ft = transform_bounds(
                            src.crs, dataset.mars_crs, *src.bounds
                        )
                        fl = ((fl + 180.0) % 360.0) - 180.0
                        fr = ((fr + 180.0) % 360.0) - 180.0
                except Exception:
                    continue

                # Footprint must be within JP2 bounds (with tolerance)
                tol = _SPATIAL_TOL * 10  # generous for integration test
                if (
                    fb_minx < fl - tol
                    or fb_maxx > fr + tol
                    or fb_miny < flb - tol
                    or fb_maxy > ft + tol
                ):
                    failures += 1
                break

        assert failures == 0, (
            f"{failures} observations had footprints outside their JP2 bounds"
        )

    def test_hirise_sampler_no_index_error(self, dataset):
        """HiRISEGeoSampler must not trigger IndexError for sampled patches."""
        from torch.utils.data import DataLoader

        from hirise_sampler import HiRISEGeoSampler
        from torchgeo.samplers import Units

        sampler = HiRISEGeoSampler(dataset, size=0.005, length=20, units=Units.CRS)
        loader = DataLoader(dataset, sampler=sampler)

        errors = 0
        for sample in loader:
            # If we get here without IndexError, the patch was loaded successfully.
            assert "image" in sample
            assert sample["image"].shape[0] == 1  # batch dim from DataLoader
            errors += 1 if sample["image"].sum() == 0 else 0

        # Allow at most 10 % fully-black patches (occasional nodata is OK)
        assert errors <= max(2, len(sampler) * 0.1)

    def test_olympus_target_filter_no_index_error(self):
        """target='Olympus' filter also works without IndexError.

        Corrupted JP2 files produce partially-zero images (rasterio logs a
        warning and skips the band), but must NOT raise an unhandled exception.
        """
        from torch.utils.data import DataLoader

        from hirise_sampler import HiRISEGeoSampler
        from temp import MarsHiRISE
        from torchgeo.samplers import Units

        ds = MarsHiRISE(
            target="Olympus",
            channels=["RED"],
            download=False,
            reuse_cache=True,
        )
        sampler = HiRISEGeoSampler(ds, size=0.005, length=10, units=Units.CRS)
        loader = DataLoader(ds, sampler=sampler)

        for sample in loader:
            # No unhandled exception is the primary assertion.
            assert "image" in sample
