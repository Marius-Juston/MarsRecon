"""Tests for Mars geographic coordinate transformations.

Covers:
- Longitude normalisation from PDS [0°, 360°] to [-180°, 180°]
- Mars IAU 2000 CRS properties
"""

import pandas as pd
import pytest
from pyproj import CRS


# ---------------------------------------------------------------------------
# Helpers — mirrors the normalisation logic in temp.py
# ---------------------------------------------------------------------------


def _norm_lon(lon: float) -> float:
    """Normalise a single longitude value to [-180°, 180°]."""
    return ((lon + 180.0) % 360.0) - 180.0


def _norm_lon_series(s: pd.Series) -> pd.Series:
    return ((s + 180.0) % 360.0) - 180.0


# ---------------------------------------------------------------------------
# Longitude normalisation
# ---------------------------------------------------------------------------


class TestLongitudeNormalization:
    def test_zero_stays_zero(self):
        assert _norm_lon(0.0) == pytest.approx(0.0)

    def test_90_east_unchanged(self):
        assert _norm_lon(90.0) == pytest.approx(90.0)

    def test_180_maps_to_minus_180(self):
        # 180° and -180° are the same meridian; the formula produces -180.
        assert _norm_lon(180.0) == pytest.approx(-180.0)

    def test_270_maps_to_minus_90(self):
        assert _norm_lon(270.0) == pytest.approx(-90.0)

    def test_360_maps_to_zero(self):
        assert _norm_lon(360.0) == pytest.approx(0.0)

    def test_already_negative(self):
        assert _norm_lon(-45.0) == pytest.approx(-45.0)

    def test_pds_western_hemisphere(self):
        # 224° in PDS convention → western hemisphere
        assert _norm_lon(224.0) == pytest.approx(-136.0)

    def test_pds_eastern_limit(self):
        # 236° → -124°
        assert _norm_lon(236.0) == pytest.approx(-124.0)

    def test_output_always_in_range(self):
        for lon in range(0, 361, 10):
            result = _norm_lon(float(lon))
            assert -180.0 <= result <= 180.0, f"Out of range for input {lon}: {result}"

    def test_series_vectorised(self):
        lons = pd.Series([0.0, 90.0, 180.0, 270.0, 360.0])
        result = _norm_lon_series(lons)
        expected = pd.Series([0.0, 90.0, -180.0, -90.0, 0.0])
        pd.testing.assert_series_equal(result, expected)

    def test_series_pds_values(self):
        lons = pd.Series([224.0, 236.0, 269.5])
        result = _norm_lon_series(lons)
        expected = pd.Series([-136.0, -124.0, -90.5])
        pd.testing.assert_series_equal(result, expected)


# ---------------------------------------------------------------------------
# Mars CRS properties
# ---------------------------------------------------------------------------


class TestMarsCRS:
    def test_is_geographic(self, mars_crs: CRS):
        assert mars_crs.is_geographic

    def test_axis_units_are_degrees(self, mars_crs: CRS):
        unit = mars_crs.axis_info[0].unit_name.lower()
        assert "degree" in unit

    def test_semi_major_axis(self, mars_crs: CRS):
        assert mars_crs.ellipsoid.semi_major_metre == pytest.approx(3_396_190.0, rel=1e-4)

    def test_semi_minor_axis(self, mars_crs: CRS):
        assert mars_crs.ellipsoid.semi_minor_metre == pytest.approx(3_376_200.0, rel=1e-4)

    def test_is_not_earth(self, mars_crs: CRS):
        earth = CRS.from_epsg(4326)
        assert mars_crs.ellipsoid.semi_major_metre != pytest.approx(
            earth.ellipsoid.semi_major_metre, rel=1e-3
        )
