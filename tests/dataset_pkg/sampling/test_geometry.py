"""Unit tests for :mod:`dataset.sampling.geometry`.

Covers ``as_patch_size``, ``generate_valid_center_region`` (feasible,
infeasible, validation, degenerate), and both packers
(``pack_patches_independent_strips`` / ``pack_patches_grid``) including
empty-region and overlap-validation edge cases.
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Point, Polygon, box

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

    def test_tall_region_vertical_sweep_wins(self):
        """A tall narrow region → vertical sweep yields the best packing.

        Exercises the vertical-sweep `best_points` update (geometry.py:263-264).
        """
        tall = box(0.0, 0.0, 2.0, 60.0)
        pts = pack_patches_independent_strips(
            tall, patch_size=1.0, patch_overlap=0.0, phase_steps=4
        )
        assert len(pts) > 10
        for x, y in pts:
            assert 0.0 <= x <= 2.0 and 0.0 <= y <= 60.0


class TestGenerateValidCenterRegionRayEdges:
    """brentq / buffer(0) / degenerate-boundary branches (152-163)."""

    _SQ = staticmethod(lambda: box(0.0, 0.0, 20.0, 20.0))

    def test_brentq_value_error_rays_skipped(self, monkeypatch):
        """Rays with no sign change raise ValueError → skipped (152-154)."""
        import scipy.optimize as so
        from dataset.sampling import geometry as G

        orig = so.brentq
        state = {"n": 0}

        def flaky(*a, **k):
            state["n"] += 1
            if state["n"] % 3 == 0:
                raise ValueError("no sign change on this ray")
            return orig(*a, **k)

        monkeypatch.setattr(G, "brentq", flaky)
        region = generate_valid_center_region(
            self._SQ(), patch_size=1.0, overlap_percentage=0.3,
            extra_rays_per_edge=6,
        )
        assert region.geom_type == "Polygon"
        assert not region.is_empty

    def test_self_intersecting_boundary_repaired_with_buffer0(self, monkeypatch):
        """Bowtie boundary → invalid → buffer(0) repair (geometry.py:161)."""
        import numpy as np
        from dataset.sampling import geometry as G

        # 4 angles ordered so the resulting ring self-intersects (bowtie).
        monkeypatch.setattr(
            G, "_get_optimized_angles",
            lambda *a, **k: np.array([0.0, np.pi, np.pi / 2, 3 * np.pi / 2]),
        )
        rs = iter([2.0, 2.0, 2.0, 2.0])
        monkeypatch.setattr(G, "brentq", lambda *a, **k: next(rs))
        region = generate_valid_center_region(
            self._SQ(), patch_size=1.0, overlap_percentage=0.3,
        )
        assert region.geom_type == "Polygon"
        assert region.is_valid
        assert not region.is_empty

    def test_degenerate_collinear_boundary_returns_empty(self, monkeypatch):
        """Collinear boundary → buffer(0) empty → Polygon() (geometry.py:163)."""
        import numpy as np
        from dataset.sampling import geometry as G

        monkeypatch.setattr(
            G, "_get_optimized_angles",
            lambda *a, **k: np.array([0.0, 0.0, 0.0]),
        )
        rs = iter([1.0, 2.0, 3.0])
        monkeypatch.setattr(G, "brentq", lambda *a, **k: next(rs))
        region = generate_valid_center_region(
            self._SQ(), patch_size=1.0, overlap_percentage=0.3,
        )
        assert region.is_empty

    def test_fewer_than_three_boundary_points_returns_empty(self, monkeypatch):
        """All rays skipped → <3 points → Polygon() (geometry.py:156-157)."""
        from dataset.sampling import geometry as G

        def always_fail(*a, **k):
            raise ValueError("no root")

        monkeypatch.setattr(G, "brentq", always_fail)
        region = generate_valid_center_region(
            box(0.0, 0.0, 20.0, 20.0), patch_size=1.0,
            overlap_percentage=0.3,
        )
        assert region.is_empty


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
            assert region.intersects(Point(x, y))

    def test_grid_on_rotated_strip(self, strip_polygon):
        region = generate_valid_center_region(strip_polygon, 0.01, 0.4)
        pts = pack_patches_grid(region, 0.01, phase_steps=3)
        assert isinstance(pts, list)
