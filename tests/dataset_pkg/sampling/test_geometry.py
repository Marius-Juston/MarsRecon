"""Unit tests for :mod:`dataset.sampling.geometry`.

Covers ``as_patch_size``, ``generate_valid_center_region`` (feasible,
infeasible, validation, degenerate), and both packers
(``pack_patches_independent_strips`` / ``pack_patches_grid``) including
empty-region and overlap-validation edge cases.
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon, box

from dataset.sampling.geometry import (
    as_patch_size,
    generate_valid_center_region,
    pack_patches_grid,
    pack_patches_independent_strips,
)


# ---------------------------------------------------------------------------
# as_patch_size
# ---------------------------------------------------------------------------


class TestAsPatchSize:
    def test_scalar_int(self):
        assert as_patch_size(4) == (4.0, 4.0)

    def test_scalar_float(self):
        assert as_patch_size(2.5) == (2.5, 2.5)

    def test_tuple(self):
        assert as_patch_size((3, 7)) == (3.0, 7.0)


# ---------------------------------------------------------------------------
# generate_valid_center_region
# ---------------------------------------------------------------------------


class TestGenerateValidCenterRegion:
    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError, match="overlap_percentage"):
            generate_valid_center_region(box(0, 0, 10, 10), 1.0, 1.5)

    def test_nonpositive_patch_raises(self):
        with pytest.raises(ValueError, match="patch_size must be positive"):
            generate_valid_center_region(box(0, 0, 10, 10), 0.0, 0.5)

    def test_feasible_returns_polygon(self):
        # Large target, small patch, modest overlap → non-empty valid region
        region = generate_valid_center_region(box(0, 0, 100, 100), 10.0, 0.5)
        assert isinstance(region, Polygon)
        assert not region.is_empty
        # Centre of the target must be a valid centre.
        assert region.contains(region.centroid)

    def test_infeasible_returns_empty(self):
        # Patch far larger than the target at 100% overlap → impossible
        region = generate_valid_center_region(box(0, 0, 1, 1), 100.0, 1.0)
        assert region.is_empty

    def test_rotated_strip(self, strip_polygon):
        region = generate_valid_center_region(strip_polygon, 0.01, 0.5)
        assert isinstance(region, Polygon)
        # Either a valid sub-region or empty — must never raise.
        if not region.is_empty:
            assert strip_polygon.buffer(1e-6).contains(region.centroid)

    def test_tuple_patch_size(self):
        region = generate_valid_center_region(box(0, 0, 100, 100), (10.0, 20.0), 0.4)
        assert isinstance(region, Polygon)


# ---------------------------------------------------------------------------
# pack_patches_independent_strips
# ---------------------------------------------------------------------------


class TestPackIndependentStrips:
    def test_empty_region_returns_empty(self):
        assert pack_patches_independent_strips(Polygon(), 1.0) == []

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError, match="patch_overlap must be in"):
            pack_patches_independent_strips(box(0, 0, 10, 10), 1.0, patch_overlap=1.0)

    def test_packs_centres_in_square(self):
        pts = pack_patches_independent_strips(box(0, 0, 10, 10), 2.0, phase_steps=4)
        assert len(pts) > 0
        for x, y in pts:
            assert 0 <= x <= 10 and 0 <= y <= 10

    def test_overlap_increases_count(self):
        few = pack_patches_independent_strips(box(0, 0, 20, 20), 4.0, 0.0, phase_steps=4)
        many = pack_patches_independent_strips(box(0, 0, 20, 20), 4.0, 0.5, phase_steps=4)
        assert len(many) >= len(few)

    def test_thin_region_still_returns_list(self):
        # Very thin sliver — exercises the segment-iteration branches
        sliver = box(0, 0, 50, 0.5)
        pts = pack_patches_independent_strips(sliver, 1.0, phase_steps=3)
        assert isinstance(pts, list)


# ---------------------------------------------------------------------------
# pack_patches_grid
# ---------------------------------------------------------------------------


class TestPackGrid:
    def test_empty_region_returns_empty(self):
        assert pack_patches_grid(Polygon(), 1.0) == []

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError, match="patch_overlap must be in"):
            pack_patches_grid(box(0, 0, 10, 10), 1.0, patch_overlap=1.5)

    def test_grid_centres_inside_region(self):
        region = box(0, 0, 10, 10)
        pts = pack_patches_grid(region, 2.0, phase_steps=3)
        assert len(pts) > 0
        for x, y in pts:
            assert region.contains_properly(
                __import__("shapely").geometry.Point(x, y)
            ) or region.touches(__import__("shapely").geometry.Point(x, y)) or region.contains(
                __import__("shapely").geometry.Point(x, y)
            )

    def test_grid_on_rotated_strip(self, strip_polygon):
        region = generate_valid_center_region(strip_polygon, 0.01, 0.4)
        pts = pack_patches_grid(region, 0.01, phase_steps=3)
        assert isinstance(pts, list)
