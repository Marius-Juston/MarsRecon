"""Shared pytest fixtures for MarsRecon tests.

All fixtures are pure Python / in-memory; none require real HiRISE data.
"""

import math
import pathlib
import sys
import textwrap

import pandas as pd
import pytest
from pyproj import CRS
from shapely.geometry import Polygon

# Make src/ and tests/ importable without a package install.
_ROOT = pathlib.Path(__file__).parent
_SRC = _ROOT.parent / "src"
for _p in (_SRC, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------------
# CRS
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mars_crs() -> CRS:
    """Mars IAU 2000 geographic CRS (lon/lat decimal degrees)."""
    return CRS.from_proj4("+proj=longlat +a=3396190 +b=3376200 +no_defs")


# ---------------------------------------------------------------------------
# Geometry fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def strip_polygon() -> Polygon:
    """Synthetic HiRISE-like strip: ~0.05° wide × 2.7° long, tilted ~15°.

    A real HiRISE pass over the Olympus Mons region might be ~6 km wide and
    300 km long.  At Mars geographic scale that's roughly 0.1° × 5°.  This
    fixture uses a slightly smaller strip so unit tests run quickly while still
    exercising the rotated-polygon geometry.
    """
    angle = math.radians(15.0)
    cx, cy = -130.0, 18.5
    half_w, half_h = 0.025, 1.35  # degrees — width/height half-extents
    corners = [
        (
            cx + (-half_w) * math.cos(angle) - (-half_h) * math.sin(angle),
            cy + (-half_w) * math.sin(angle) + (-half_h) * math.cos(angle),
        ),
        (
            cx + (half_w) * math.cos(angle) - (-half_h) * math.sin(angle),
            cy + (half_w) * math.sin(angle) + (-half_h) * math.cos(angle),
        ),
        (
            cx + (half_w) * math.cos(angle) - (half_h) * math.sin(angle),
            cy + (half_w) * math.sin(angle) + (half_h) * math.cos(angle),
        ),
        (
            cx + (-half_w) * math.cos(angle) - (half_h) * math.sin(angle),
            cy + (-half_w) * math.sin(angle) + (half_h) * math.cos(angle),
        ),
    ]
    return Polygon(corners)


# ---------------------------------------------------------------------------
# LBL fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_lbl(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a minimal PDS3 LBL file to *tmp_path* and return its path.

    Values are chosen to be easily verifiable in tests:

    * SCALING_FACTOR = 2.5e-04
    * OFFSET = 0.04
    * SAMPLE_BITS = 16
    * SAMPLE_BIT_MASK = 2#0000001111111111# → effective_max_dn = 1023
    * BANDS = 3
    * FILTER_NAME = ("NEAR-INFRARED", "RED", "BLUE-GREEN")
    """
    content = textwrap.dedent("""\
        PDS_VERSION_ID = PDS3
        SCALING_FACTOR = 2.5e-04
        OFFSET = 0.04
        SAMPLE_BITS = 16
        SAMPLE_BIT_MASK = 2#0000001111111111#
        BANDS = 3
        FILTER_NAME = ("NEAR-INFRARED", "RED", "BLUE-GREEN")
        END
    """)
    lbl_file = tmp_path / "PSP_001430_1780_COLOR.LBL"
    lbl_file.write_text(content)
    return lbl_file


# ---------------------------------------------------------------------------
# Index row fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_corner_row() -> pd.Series:
    """A :class:`pandas.Series` mimicking one row of :attr:`_raw_index`.

    Uses PDS longitude convention [0°, 360°]:
      CORNER1 = (175°, -5°)  → normalised (-185°+360° = 175° → -185° … wait)

    Actually let us use a simple rectangle in the western hemisphere:
      CORNER1 = (224°, -5°)  → normalised to -136°  (224 - 360 = -136)
      CORNER2 = (236°, -5°)  → normalised to -124°
      CORNER3 = (236°,  5°)  → normalised to -124°
      CORNER4 = (224°,  5°)  → normalised to -136°
    """
    return pd.Series(
        {
            "CORNER1_LATITUDE": -5.0,
            "CORNER1_LONGITUDE": 224.0,
            "CORNER2_LATITUDE": -5.0,
            "CORNER2_LONGITUDE": 236.0,
            "CORNER3_LATITUDE": 5.0,
            "CORNER3_LONGITUDE": 236.0,
            "CORNER4_LATITUDE": 5.0,
            "CORNER4_LONGITUDE": 224.0,
            "MINIMUM_LATITUDE": -5.0,
            "MAXIMUM_LATITUDE": 5.0,
            "MINIMUM_LONGITUDE": 224.0,
            "MAXIMUM_LONGITUDE": 236.0,
        }
    )
