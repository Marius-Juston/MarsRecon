"""Tests for spatial index building helpers.

Covers:
- _corners_to_polygon: polygon construction, longitude normalisation, NaN
  handling, missing columns, and the geometry-is-smaller-than-bbox property.
"""

import pandas as pd
import pytest
from shapely.geometry import Polygon, box

# src/ is on sys.path via conftest.py
from mars_hirise import _corners_to_polygon


# ---------------------------------------------------------------------------
# _corners_to_polygon
# ---------------------------------------------------------------------------


class TestCornersToPolygon:
    # ---- Basic construction ------------------------------------------------

    def test_returns_polygon(self):
        row = _make_row(0.0, 0.0, 10.0, 10.0)
        result = _corners_to_polygon(row)
        assert isinstance(result, Polygon)

    def test_polygon_is_valid(self):
        row = _make_row(0.0, 0.0, 10.0, 10.0)
        poly = _corners_to_polygon(row)
        assert poly is not None and poly.is_valid

    def test_polygon_is_not_empty(self):
        row = _make_row(0.0, 0.0, 10.0, 10.0)
        poly = _corners_to_polygon(row)
        assert poly is not None and not poly.is_empty

    # ---- Longitude normalisation -------------------------------------------

    def test_pds_longitude_224_normalises_to_minus136(self):
        # PDS 224° → -136°
        row = _make_row(-5.0, 224.0, 5.0, 236.0)
        poly = _corners_to_polygon(row)
        assert poly is not None
        minx, _, maxx, _ = poly.bounds
        assert minx == pytest.approx(-136.0, abs=0.01)
        assert maxx == pytest.approx(-124.0, abs=0.01)

    def test_negative_longitudes_pass_through_unchanged(self):
        row = _make_row(-5.0, -50.0, 5.0, -40.0)
        poly = _corners_to_polygon(row)
        assert poly is not None
        minx, _, maxx, _ = poly.bounds
        assert minx == pytest.approx(-50.0, abs=0.01)
        assert maxx == pytest.approx(-40.0, abs=0.01)

    def test_longitude_265_normalises_to_minus95(self):
        # CORNER1/4 at 265° → -95°, CORNER2/3 at 275° → -85°
        row = pd.Series({
            "CORNER1_LATITUDE": -5.0, "CORNER1_LONGITUDE": 265.0,
            "CORNER2_LATITUDE": -5.0, "CORNER2_LONGITUDE": 275.0,
            "CORNER3_LATITUDE": 5.0, "CORNER3_LONGITUDE": 275.0,
            "CORNER4_LATITUDE": 5.0, "CORNER4_LONGITUDE": 265.0,
        })
        poly = _corners_to_polygon(row)
        assert poly is not None
        minx, _, maxx, _ = poly.bounds
        assert minx == pytest.approx(-95.0, abs=0.1)
        assert maxx == pytest.approx(-85.0, abs=0.1)

    # ---- NaN / missing column handling ------------------------------------

    def test_nan_latitude_returns_none(self):
        row = _make_row(float("nan"), 0.0, 10.0, 10.0)
        assert _corners_to_polygon(row) is None

    def test_nan_longitude_returns_none(self):
        row = pd.Series(
            {
                "CORNER1_LATITUDE": 0.0, "CORNER1_LONGITUDE": float("nan"),
                "CORNER2_LATITUDE": 0.0, "CORNER2_LONGITUDE": 10.0,
                "CORNER3_LATITUDE": 10.0, "CORNER3_LONGITUDE": 10.0,
                "CORNER4_LATITUDE": 10.0, "CORNER4_LONGITUDE": 0.0,
            }
        )
        assert _corners_to_polygon(row) is None

    def test_missing_corner_column_returns_none(self):
        row = pd.Series({"CORNER1_LATITUDE": 0.0, "CORNER1_LONGITUDE": 0.0})
        assert _corners_to_polygon(row) is None

    def test_empty_series_returns_none(self):
        assert _corners_to_polygon(pd.Series(dtype=float)) is None

    # ---- Geometry properties -----------------------------------------------

    def test_rotated_strip_smaller_than_bbox(self, strip_polygon: Polygon):
        """A rotated strip polygon should occupy less area than its bbox."""
        bbox_area = box(*strip_polygon.bounds).area
        assert strip_polygon.area < bbox_area * 0.95

    def test_axis_aligned_box_equals_bbox(self):
        row = _make_row(-5.0, -50.0, 5.0, -40.0)
        poly = _corners_to_polygon(row)
        assert poly is not None
        # For a perfectly axis-aligned rectangle the polygon area ≈ bbox area.
        bbox_area = box(*poly.bounds).area
        assert poly.area == pytest.approx(bbox_area, rel=1e-6)

    def test_four_vertices(self):
        row = _make_row(0.0, 0.0, 10.0, 10.0)
        poly = _corners_to_polygon(row)
        # Shapely closes the ring, so exterior coords = 5 (4 + repeated first).
        assert len(list(poly.exterior.coords)) == 5

    # ---- synthetic_corner_row fixture ---------------------------------------

    def test_synthetic_corner_row_produces_valid_polygon(
            self, synthetic_corner_row: pd.Series
    ):
        poly = _corners_to_polygon(synthetic_corner_row)
        assert poly is not None and poly.is_valid and not poly.is_empty


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_row(lat1: float, lon1: float, lat3: float, lon3: float) -> pd.Series:
    """Build a minimal row with four corners forming an axis-aligned rectangle.

    CORNER1 = bottom-left, CORNER2 = bottom-right,
    CORNER3 = top-right, CORNER4 = top-left.
    """
    return pd.Series(
        {
            "CORNER1_LATITUDE": lat1, "CORNER1_LONGITUDE": lon1,
            "CORNER2_LATITUDE": lat1, "CORNER2_LONGITUDE": lon3,
            "CORNER3_LATITUDE": lat3, "CORNER3_LONGITUDE": lon3,
            "CORNER4_LATITUDE": lat3, "CORNER4_LONGITUDE": lon1,
        }
    )
