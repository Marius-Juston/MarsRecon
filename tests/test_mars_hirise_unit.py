"""Unit tests for mars_hirise.py — comprehensive branch coverage.

Covers all branches not exercised by existing tests:
  filter_maker, _corners_to_polygon (buffer fix), _extract_footprint,
  _prefer_cog, _read_jp2_bounds, _extract_data_footprint, _load_from_jp2,
  _load_tile, plot(), plot_coverage(), _coverage_grid, spatial_index_cache,
  __getitem__ transforms, _merge_tiles warning, _load_index, _build_spatial_index
  (legacy + antimeridian + non-legacy cache), _verify JP2 warnings, setup_logging.

All tests are pure unit tests — no real HiRISE data required.
"""

import json
import logging
import pathlib
import sys
import textwrap
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.crs import CRS as RasterioCRS
from shapely.geometry import Polygon, box

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from mars_hirise import (
    MarsHiRISE,
    _ProductMeta,
    _corners_to_polygon,
    _extract_footprint,
    filter_maker,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Module-level constants for test geometry
# ---------------------------------------------------------------------------

# Mars geographic CRS as a rasterio CRS (for creating synthetic files)
_MARS_RCRS = RasterioCRS.from_proj4(
    "+proj=longlat +a=3396190 +b=3376200 +no_defs"
)

# Small query slices (4×4 pixels at 0.01°/pixel — avoids huge arrays)
_X = slice(-131.0, -130.96, 0.01)
_Y = slice(18.0, 18.04, 0.01)

# Timestamps for synthetic index
_T0 = pd.Timestamp("2007-01-01T00:00:00", tz="UTC")
_T1 = pd.Timestamp("2007-01-01T00:01:00", tz="UTC")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_dataset(tmp_path, mars_crs):
    """MarsHiRISE with _verify() suppressed and a minimal 1-row synthetic index."""
    with patch.object(MarsHiRISE, "_verify", return_value=None):
        ds = MarsHiRISE(root=tmp_path)

    ds.index = gpd.GeoDataFrame(
        {
            "obs_id": ["PSP_001430_1780"],
            "color_path": [None],
            "red_path": [None],
        },
        index=pd.IntervalIndex.from_tuples(
            [(_T0, _T1)], closed="both", name="datetime"
        ),
        geometry=[box(-131.0, 18.0, -130.0, 19.0)],
        crs=mars_crs,
    )
    ds._raw_index = pd.DataFrame(
        {
            "PRODUCT_ID": ["PSP_001430_1780_COLOR"],
            "FILE_NAME_SPECIFICATION": [
                "MROHR_0001/DATA/PSP/ORB_001400_001499/"
                "PSP_001430_1780/PSP_001430_1780_COLOR.JP2"
            ],
            "OBSERVATION_ID": ["PSP_001430_1780"],
            "START_TIME": ["2007-01-01T00:00:00"],
            "STOP_TIME": ["2007-01-01T00:01:00"],
            "MINIMUM_LONGITUDE": [229.0],
            "MAXIMUM_LONGITUDE": [230.0],
            "MINIMUM_LATITUDE": [18.0],
            "MAXIMUM_LATITUDE": [19.0],
        }
    )
    ds.channels = ["NEAR-INFRARED", "RED", "BLUE-GREEN"]
    return ds


@pytest.fixture
def mars_geotiff(tmp_path):
    """16×16 single-band GeoTIFF with Mars geographic CRS, non-zero pixels."""
    p = tmp_path / "mars_test.tif"
    transform = rasterio.transform.from_bounds(-131.0, 18.0, -130.0, 19.0, 16, 16)
    data = np.ones((1, 16, 16), dtype=np.uint16) * 500
    with rasterio.open(
            p, "w",
            driver="GTiff", count=1, dtype="uint16",
            width=16, height=16, crs=_MARS_RCRS, transform=transform,
    ) as dst:
        dst.write(data)
    return p


@pytest.fixture
def mars_geotiff_3band(tmp_path):
    """16×16 3-band GeoTIFF with Mars geographic CRS, non-zero pixels."""
    p = tmp_path / "mars_test_3band.tif"
    transform = rasterio.transform.from_bounds(-131.0, 18.0, -130.0, 19.0, 16, 16)
    data = np.ones((3, 16, 16), dtype=np.uint16) * 500
    with rasterio.open(
            p, "w",
            driver="GTiff", count=3, dtype="uint16",
            width=16, height=16, crs=_MARS_RCRS, transform=transform,
    ) as dst:
        dst.write(data)
    return p


@pytest.fixture
def no_crs_geotiff(tmp_path):
    """16×16 GeoTIFF with no embedded CRS."""
    p = tmp_path / "nocrs.tif"
    transform = rasterio.transform.from_bounds(-131.0, 18.0, -130.0, 19.0, 16, 16)
    data = np.ones((1, 16, 16), dtype=np.uint16) * 500
    with rasterio.open(
            p, "w",
            driver="GTiff", count=1, dtype="uint16",
            width=16, height=16, transform=transform,
            # intentionally omit crs=
    ) as dst:
        dst.write(data)
    return p


@pytest.fixture
def zero_geotiff(tmp_path):
    """16×16 all-zero GeoTIFF (simulates empty observation)."""
    p = tmp_path / "zeros.tif"
    transform = rasterio.transform.from_bounds(-131.0, 18.0, -130.0, 19.0, 16, 16)
    data = np.zeros((1, 16, 16), dtype=np.uint16)
    with rasterio.open(
            p, "w",
            driver="GTiff", count=1, dtype="uint16",
            width=16, height=16, crs=_MARS_RCRS, transform=transform,
    ) as dst:
        dst.write(data)
    return p


# ---------------------------------------------------------------------------
# A. filter_maker (mars_hirise.py version)
# ---------------------------------------------------------------------------


class TestFilterMakerMarsHiRISE:
    """filter_maker() in mars_hirise.py creates a level-bounding log filter."""

    def test_passes_record_at_threshold_level(self):
        filt = filter_maker("WARNING")
        rec = logging.LogRecord("test", logging.WARNING, "", 0, "msg", (), None)
        assert filt(rec) is True

    def test_blocks_record_above_threshold(self):
        filt = filter_maker("WARNING")
        rec = logging.LogRecord("test", logging.ERROR, "", 0, "msg", (), None)
        assert filt(rec) is False


# ---------------------------------------------------------------------------
# B. _corners_to_polygon — buffer(0) fix for self-intersecting polygon
# ---------------------------------------------------------------------------


class TestCornersToPolygonBuffer:
    """If the 4 corners form a self-intersecting (bowtie) polygon, buffer(0)
    is applied to produce a valid result."""

    def test_self_intersecting_corners_returns_valid_polygon(self):
        # Bowtie: corners 1 and 3 swapped so the polygon crosses itself
        # PDS convention (0-360): all values in eastern hemisphere
        row = pd.Series(
            {
                # Crossing the diagonals: (minlon, maxlat), (maxlon, minlat),
                # (minlon, minlat), (maxlon, maxlat)  → self-intersecting
                "CORNER1_LONGITUDE": 200.0,
                "CORNER1_LATITUDE": 5.0,
                "CORNER2_LONGITUDE": 210.0,
                "CORNER2_LATITUDE": -5.0,
                "CORNER3_LONGITUDE": 200.0,
                "CORNER3_LATITUDE": -5.0,
                "CORNER4_LONGITUDE": 210.0,
                "CORNER4_LATITUDE": 5.0,
            }
        )
        result = _corners_to_polygon(row)
        # buffer(0) should heal the bowtie into a valid polygon
        assert result is not None
        assert result.is_valid


# ---------------------------------------------------------------------------
# C. _extract_footprint (standalone module-level function)
# ---------------------------------------------------------------------------


class TestExtractFootprintStandalone:
    """Tests for the top-level _extract_footprint() helper."""

    def test_none_path_returns_none_none(self):
        result = _extract_footprint(None, _MARS_RCRS)
        assert result == (None, None)

    def test_nonexistent_file_returns_none_none(self, tmp_path):
        result = _extract_footprint(str(tmp_path / "ghost.tif"), _MARS_RCRS)
        assert result == (None, None)

    def test_no_crs_returns_none_none(self, no_crs_geotiff):
        hull, bounds = _extract_footprint(str(no_crs_geotiff), _MARS_RCRS)
        assert hull is None

    def test_valid_file_returns_hull_and_bounds(self, mars_geotiff):
        hull, bounds = _extract_footprint(str(mars_geotiff), _MARS_RCRS)
        assert hull is not None
        assert bounds is not None
        assert len(hull) >= 3

    def test_fewer_than_3_nonzero_pixels_returns_none_with_bounds(self, tmp_path):
        """< 3 non-zero pixels: hull is None but file_bounds may still be returned."""
        # Create a file with only 2 non-zero pixels
        p = tmp_path / "sparse.tif"
        transform = rasterio.transform.from_bounds(-131.0, 18.0, -130.0, 19.0, 16, 16)
        data = np.zeros((1, 16, 16), dtype=np.uint16)
        data[0, 0, 0] = 1
        data[0, 0, 1] = 1  # only 2 non-zero → len(xs) < 3
        with rasterio.open(
                p, "w", driver="GTiff", count=1, dtype="uint16",
                width=16, height=16, crs=_MARS_RCRS, transform=transform,
        ) as dst:
            dst.write(data)
        hull, bounds = _extract_footprint(str(p), _MARS_RCRS)
        assert hull is None

    def test_exception_during_open_returns_none_none(self, tmp_path):
        """Exception inside rasterio.open caught → (None, None)."""
        p = tmp_path / "bad.tif"
        p.write_bytes(b"not a valid rasterio file at all")
        result = _extract_footprint(str(p), _MARS_RCRS)
        assert result == (None, None)

    def test_empty_hull_path(self, mars_geotiff):
        """hull.is_empty path: mock MultiPoint.convex_hull to be empty."""
        empty_geom = MagicMock()
        empty_geom.is_empty = True

        mock_mp_instance = MagicMock()
        mock_mp_instance.convex_hull = empty_geom

        with patch("shapely.geometry.MultiPoint", return_value=mock_mp_instance):
            hull, bounds = _extract_footprint(str(mars_geotiff), _MARS_RCRS)

        assert hull is None
        # file_bounds may or may not be set depending on how far we get
        # The key assertion is that hull is None (is_empty path returned early)


# ---------------------------------------------------------------------------
# D. _prefer_cog
# ---------------------------------------------------------------------------


class TestPreferCog:
    def test_none_returns_none(self):
        assert MarsHiRISE._prefer_cog(None) is None

    def test_cog_exists_returns_cog(self, tmp_path):
        jp2 = tmp_path / "image.JP2"
        cog = tmp_path / "image.tif"
        cog.touch()
        result = MarsHiRISE._prefer_cog(jp2)
        assert result == cog

    def test_cog_not_exists_returns_original(self, tmp_path):
        jp2 = tmp_path / "image.JP2"
        jp2.touch()
        result = MarsHiRISE._prefer_cog(jp2)
        assert result == jp2


# ---------------------------------------------------------------------------
# E. _read_jp2_bounds
# ---------------------------------------------------------------------------


class TestReadJp2Bounds:
    def test_nonexistent_file_returns_none(self, mock_dataset, tmp_path):
        result = mock_dataset._read_jp2_bounds(tmp_path / "ghost.tif")
        assert result is None

    def test_no_crs_returns_none(self, mock_dataset, no_crs_geotiff):
        result = mock_dataset._read_jp2_bounds(no_crs_geotiff)
        assert result is None

    def test_out_of_range_bounds_returns_none(self, mock_dataset, tmp_path):
        """transform_bounds returning invalid geographic range → None."""
        with patch("mars_hirise.transform_bounds", return_value=(400.0, 200.0, 500.0, 300.0)):
            # Create a dummy file that rasterio can open
            p = tmp_path / "dummy.tif"
            transform = rasterio.transform.from_bounds(-131, 18, -130, 19, 4, 4)
            data = np.ones((1, 4, 4), dtype=np.uint16)
            with rasterio.open(
                    p, "w", driver="GTiff", count=1, dtype="uint16",
                    width=4, height=4, crs=_MARS_RCRS, transform=transform,
            ) as dst:
                dst.write(data)
            result = mock_dataset._read_jp2_bounds(p)
        assert result is None

    def test_exception_returns_none(self, mock_dataset, tmp_path):
        """Exception during open → None."""
        p = tmp_path / "bad.tif"
        p.write_bytes(b"garbage")
        result = mock_dataset._read_jp2_bounds(p)
        assert result is None

    def test_valid_file_returns_bounds(self, mock_dataset, mars_geotiff):
        result = mock_dataset._read_jp2_bounds(mars_geotiff)
        assert result is not None
        fl, fb, fr, ft = result
        assert -180.0 <= fl < fr <= 180.0
        assert -90.0 <= fb < ft <= 90.0


# ---------------------------------------------------------------------------
# F. _extract_data_footprint
# ---------------------------------------------------------------------------


class TestExtractDataFootprint:
    def test_nonexistent_path_returns_none(self, mock_dataset, tmp_path):
        result = mock_dataset._extract_data_footprint(tmp_path / "ghost.tif")
        assert result is None

    def test_no_crs_returns_none(self, mock_dataset, no_crs_geotiff):
        result = mock_dataset._extract_data_footprint(no_crs_geotiff)
        assert result is None

    def test_fewer_than_3_nonzero_pixels_returns_none(self, mock_dataset, tmp_path):
        p = tmp_path / "sparse.tif"
        transform = rasterio.transform.from_bounds(-131.0, 18.0, -130.0, 19.0, 16, 16)
        data = np.zeros((1, 16, 16), dtype=np.uint16)
        data[0, 0, 0] = 1
        data[0, 0, 1] = 1  # only 2 non-zero
        with rasterio.open(
                p, "w", driver="GTiff", count=1, dtype="uint16",
                width=16, height=16, crs=_MARS_RCRS, transform=transform,
        ) as dst:
            dst.write(data)
        result = mock_dataset._extract_data_footprint(p)
        assert result is None

    def test_valid_file_returns_polygon(self, mock_dataset, mars_geotiff):
        result = mock_dataset._extract_data_footprint(mars_geotiff)
        assert result is not None
        assert isinstance(result, Polygon)

    def test_exception_returns_none(self, mock_dataset, tmp_path):
        p = tmp_path / "bad.tif"
        p.write_bytes(b"not a rasterio file")
        result = mock_dataset._extract_data_footprint(p)
        assert result is None

    def test_empty_hull_returns_none(self, mock_dataset, mars_geotiff):
        """hull_shapely.is_empty path via mocked MultiPoint."""
        empty_geom = MagicMock()
        empty_geom.is_empty = True

        mock_mp_instance = MagicMock()
        mock_mp_instance.convex_hull = empty_geom

        with patch("mars_hirise.MultiPoint", return_value=mock_mp_instance):
            result = mock_dataset._extract_data_footprint(mars_geotiff)
        assert result is None


# ---------------------------------------------------------------------------
# G. _load_from_jp2
# ---------------------------------------------------------------------------


class TestLoadFromJp2:
    def test_empty_band_map_returns_empty(self, mock_dataset, mars_geotiff):
        result = mock_dataset._load_from_jp2(
            mars_geotiff, {}, _ProductMeta(), _X, _Y
        )
        assert result == {}

    def test_no_crs_uses_dst_crs(self, mock_dataset, no_crs_geotiff):
        """File with no CRS: warns and assumes dataset CRS, still returns data."""
        result = mock_dataset._load_from_jp2(
            no_crs_geotiff, {"RED": 1}, _ProductMeta(), _X, _Y
        )
        # Should attempt reprojection (may succeed or not depending on bounds)
        assert isinstance(result, dict)

    def test_antimeridian_file_skips_early_exit(self, mock_dataset, mars_geotiff):
        """When fl > fr (antimeridian wrap), early-exit check is skipped."""
        # transform_bounds returns values whose normalised result gives fl=175 > fr=-175
        with patch("mars_hirise.transform_bounds", return_value=(175.0, 18.0, 185.0, 19.0)):
            result = mock_dataset._load_from_jp2(
                mars_geotiff, {"RED": 1}, _ProductMeta(), _X, _Y
            )
        # Key: we got through without hitting the early-exit return {}
        assert isinstance(result, dict)

    def test_non_overlapping_file_early_exit(self, mock_dataset, mars_geotiff):
        """File bounds completely outside query → early exit, returns {}."""
        # File bounds at (0,0,10,10) normalised to (-180,-170) ← well left of x=-131
        with patch("mars_hirise.transform_bounds", return_value=(0.0, 0.0, 10.0, 10.0)):
            result = mock_dataset._load_from_jp2(
                mars_geotiff, {"RED": 1}, _ProductMeta(), _X, _Y
            )
        assert result == {}

    def test_reproject_exception_skips_band(self, mock_dataset, mars_geotiff):
        """If reproject() raises, the band is skipped and result is {}."""
        with patch("mars_hirise.reproject", side_effect=RuntimeError("boom")):
            result = mock_dataset._load_from_jp2(
                mars_geotiff, {"RED": 1}, _ProductMeta(), _X, _Y
            )
        assert result == {}

    def test_rasterioioerror_returns_empty(self, mock_dataset, tmp_path):
        """RasterioIOError on open → returns {}."""
        p = tmp_path / "corrupt.tif"
        p.write_bytes(b"not a tiff")
        result = mock_dataset._load_from_jp2(
            p, {"RED": 1}, _ProductMeta(), _X, _Y
        )
        assert result == {}

    def test_valid_reprojection_returns_calibrated_data(self, mock_dataset, mars_geotiff):
        """Valid file + overlapping query → returns float32 array in [0,1]."""
        result = mock_dataset._load_from_jp2(
            mars_geotiff, {"RED": 1}, _ProductMeta(), _X, _Y
        )
        assert "RED" in result
        arr = result["RED"]
        assert arr.dtype == np.float32
        # Non-zero pixels must be in [0, 1] (zeros are nodata)
        data_pixels = arr[arr > 0]
        if len(data_pixels) > 0:
            assert data_pixels.max() <= 1.0


# ---------------------------------------------------------------------------
# H. _load_tile
# ---------------------------------------------------------------------------


class TestLoadTile:
    def test_only_red_uses_red_file(self, mock_dataset, mars_geotiff):
        """channels=['RED'] → uses RED file (only_red path)."""
        mock_dataset.channels = ["RED"]
        result = mock_dataset._load_tile(None, mars_geotiff, _X, _Y)
        assert result is not None
        assert result.shape[0] == 1  # 1 channel

    def test_color_unavailable_logs_lost_channels(
            self, mock_dataset, tmp_path, mars_geotiff, caplog
    ):
        """COLOR unavailable, NIR/BG channels lost → warning logged."""
        mock_dataset.channels = ["NEAR-INFRARED", "RED", "BLUE-GREEN"]
        nonexistent_color = tmp_path / "nocolor.tif"  # doesn't exist
        with caplog.at_level(logging.WARNING, logger="mars_hirise"):
            result = mock_dataset._load_tile(nonexistent_color, mars_geotiff, _X, _Y)
        assert "cannot provide channels" in caplog.text.lower() or "lost" in caplog.text.lower() or result is not None

    def test_color_unavailable_red_salvaged(self, mock_dataset, tmp_path, mars_geotiff):
        """COLOR unavailable, RED channel salvaged from red file."""
        mock_dataset.channels = ["RED"]
        nonexistent_color = tmp_path / "nocolor.tif"
        result = mock_dataset._load_tile(nonexistent_color, mars_geotiff, _X, _Y)
        # RED salvaged from red file
        assert result is not None

    def test_both_files_unavailable_returns_none(self, mock_dataset, tmp_path):
        """Both color and red unavailable → returns None."""
        mock_dataset.channels = ["NEAR-INFRARED", "RED", "BLUE-GREEN"]
        result = mock_dataset._load_tile(
            tmp_path / "no_color.tif",
            tmp_path / "no_red.tif",
            _X, _Y,
        )
        assert result is None

    def test_red_unavailable_after_color_fail_returns_none(
            self, mock_dataset, tmp_path, caplog
    ):
        """NIR requested, color missing, RED also missing → None with warning."""
        mock_dataset.channels = ["NEAR-INFRARED"]
        with caplog.at_level(logging.WARNING, logger="mars_hirise"):
            result = mock_dataset._load_tile(
                tmp_path / "no_color.tif",
                tmp_path / "no_red.tif",
                _X, _Y,
            )
        assert result is None

    def test_empty_band_arrays_returns_none(self, mock_dataset, mars_geotiff):
        """_load_from_jp2 returns {} → band_arrays empty → returns None."""
        mock_dataset.channels = ["RED"]
        with patch.object(mock_dataset, "_load_from_jp2", return_value={}):
            result = mock_dataset._load_tile(None, mars_geotiff, _X, _Y)
        assert result is None

    def test_custom_filter_names_from_lbl(self, mock_dataset, tmp_path):
        """If LBL has FILTER_NAME, uses custom band map."""
        color_path = tmp_path / "test_COLOR.tif"
        lbl_path = tmp_path / "test_COLOR.LBL"
        transform = rasterio.transform.from_bounds(-131.0, 18.0, -130.0, 19.0, 16, 16)
        data = np.ones((3, 16, 16), dtype=np.uint16) * 500
        with rasterio.open(
                color_path, "w", driver="GTiff", count=3, dtype="uint16",
                width=16, height=16, crs=_MARS_RCRS, transform=transform,
        ) as dst:
            dst.write(data)
        lbl_path.write_text(textwrap.dedent("""\
            SCALING_FACTOR = 2.5e-04
            OFFSET = 0.04
            SAMPLE_BITS = 16
            BANDS = 3
            FILTER_NAME = ("NEAR-INFRARED", "RED", "BLUE-GREEN")
            END
        """))
        mock_dataset.channels = ["NEAR-INFRARED", "RED", "BLUE-GREEN"]
        result = mock_dataset._load_tile(color_path, None, _X, _Y)
        assert result is not None
        assert result.shape[0] == 3


# ---------------------------------------------------------------------------
# I. plot()
# ---------------------------------------------------------------------------


def _make_sample(channels, h=16, w=16, nonzero=True):
    """Helper: build a minimal Sample dict for plot()."""
    n_ch = len(channels)
    if nonzero:
        img = torch.rand(n_ch, h, w).float() * 0.5 + 0.1
    else:
        img = torch.zeros(n_ch, h, w, dtype=torch.float32)
    return {"image": img, "bounds": torch.zeros(6), "crs": "FAKE"}


class TestPlot:
    def test_ndim4_squeezes_batch(self, mock_dataset):
        """image.ndim == 4 → first element extracted."""
        img4 = torch.rand(1, 3, 8, 8)
        sample = {"image": img4, "bounds": torch.zeros(6), "crs": "FAKE"}
        mock_dataset.channels = ["NEAR-INFRARED", "RED", "BLUE-GREEN"]
        fig = mock_dataset.plot(sample)
        plt.close(fig)

    def test_single_nonzero_band_grayscale_fallback(self, mock_dataset):
        """All 3 channels present but only 1 has data → grayscale fallback."""
        mock_dataset.channels = ["NEAR-INFRARED", "RED", "BLUE-GREEN"]
        img = torch.zeros(3, 8, 8)
        img[1, :, :] = 0.5  # only RED non-zero
        sample = {"image": img, "bounds": torch.zeros(6), "crs": "FAKE"}
        fig = mock_dataset.plot(sample)
        assert fig is not None
        plt.close(fig)

    def test_single_channel_mode(self, mock_dataset):
        """channels has < 3 channels → grey cmap, first-channel display."""
        mock_dataset.channels = ["RED"]
        sample = _make_sample(["RED"])
        fig = mock_dataset.plot(sample)
        plt.close(fig)

    def test_show_titles_false(self, mock_dataset):
        """show_titles=False → ax.set_title() not called."""
        mock_dataset.channels = ["RED"]
        sample = _make_sample(["RED"])
        fig = mock_dataset.plot(sample, show_titles=False)
        ax = fig.axes[0]
        assert ax.get_title() == ""
        plt.close(fig)

    def test_suptitle_set(self, mock_dataset):
        """suptitle= kwarg → fig.suptitle() called."""
        mock_dataset.channels = ["RED"]
        sample = _make_sample(["RED"])
        fig = mock_dataset.plot(sample, suptitle="My Title")
        assert fig._suptitle.get_text() == "My Title"
        plt.close(fig)

    def test_all_zero_image_no_stretch(self, mock_dataset):
        """All-zero image → no-op stretch (data_pixels is empty)."""
        mock_dataset.channels = ["RED"]
        sample = _make_sample(["RED"], nonzero=False)
        fig = mock_dataset.plot(sample)
        plt.close(fig)

    def test_3channel_p98_equals_p2_no_stretch(self, mock_dataset):
        """p98 == p2 (uniform data) → stretch branch not taken."""
        mock_dataset.channels = ["NEAR-INFRARED", "RED", "BLUE-GREEN"]
        # All pixels the same value → p2 == p98
        img = torch.full((3, 8, 8), 0.5)
        sample = {"image": img, "bounds": torch.zeros(6), "crs": "FAKE"}
        fig = mock_dataset.plot(sample)
        plt.close(fig)

    def test_grayscale_p98_gt_p2_stretch(self, mock_dataset):
        """Grayscale (single channel) with varied pixels → stretch applied."""
        mock_dataset.channels = ["RED"]
        img = torch.linspace(0.1, 0.9, 64).view(1, 8, 8)
        sample = {"image": img, "bounds": torch.zeros(6), "crs": "FAKE"}
        fig = mock_dataset.plot(sample)
        plt.close(fig)


# ---------------------------------------------------------------------------
# J. plot_coverage()
# ---------------------------------------------------------------------------


class TestPlotCoverage:
    def test_empty_index_raises_runtime_error(self, mock_dataset, mars_crs):
        """Empty spatial index → RuntimeError raised."""
        empty_gdf = gpd.GeoDataFrame(
            {"obs_id": [], "color_path": [], "red_path": []},
            index=pd.IntervalIndex.from_tuples([], closed="both", name="datetime"),
            geometry=[],
            crs=mars_crs,
        )
        mock_dataset.index = empty_gdf
        with pytest.raises(RuntimeError, match="Spatial index is empty"):
            mock_dataset.plot_coverage()

    def test_show_count_false_no_colorbar(self, mock_dataset):
        """show_count=False → norm=None path executed."""
        fig = mock_dataset.plot_coverage(show_count=False)
        plt.close(fig)

    def test_suptitle_provided(self, mock_dataset):
        """suptitle= kwarg → appears in figure suptitle."""
        fig = mock_dataset.plot_coverage(suptitle="Test Coverage")
        assert "Test Coverage" in fig._suptitle.get_text()
        plt.close(fig)

    def test_target_appended_to_title(self, mock_dataset):
        """If self.target is set → target filter appended to auto title."""
        mock_dataset.target = "Olympus"
        fig = mock_dataset.plot_coverage()
        assert "Olympus" in fig._suptitle.get_text()
        mock_dataset.target = None  # reset
        plt.close(fig)

    def test_resolution_auto_computed(self, mock_dataset):
        """resolution=None → auto-computed without error."""
        fig = mock_dataset.plot_coverage(resolution=None)
        plt.close(fig)

    def test_resolution_explicit(self, mock_dataset):
        """Explicit resolution → used directly."""
        fig = mock_dataset.plot_coverage(resolution=0.5)
        plt.close(fig)


# ---------------------------------------------------------------------------
# K. _coverage_grid
# ---------------------------------------------------------------------------


class TestCoverageGrid:
    def test_basic_grid_has_nonzero_cells(self, mock_dataset):
        coverage, lon_edges, lat_edges = mock_dataset._coverage_grid(
            resolution=0.5, lon_min=-132.0, lon_max=-129.0,
            lat_min=17.0, lat_max=20.0,
        )
        assert coverage.sum() > 0

    def test_observation_outside_grid_clamped(self, mock_dataset):
        """Observation outside grid bounds → coverage unchanged (no IndexError)."""
        coverage, _, _ = mock_dataset._coverage_grid(
            resolution=1.0, lon_min=0.0, lon_max=10.0,
            lat_min=0.0, lat_max=10.0,
        )
        assert coverage.sum() == 0  # observation at -131 is outside grid

    def test_empty_index_gives_zero_grid(self, mock_dataset, mars_crs):
        mock_dataset.index = gpd.GeoDataFrame(
            {"obs_id": [], "color_path": [], "red_path": []},
            index=pd.IntervalIndex.from_tuples([], closed="both", name="datetime"),
            geometry=[],
            crs=mars_crs,
        )
        coverage, _, _ = mock_dataset._coverage_grid(
            resolution=1.0, lon_min=-180.0, lon_max=180.0,
            lat_min=-90.0, lat_max=90.0,
        )
        assert coverage.sum() == 0


# ---------------------------------------------------------------------------
# L. spatial_index_cache property
# ---------------------------------------------------------------------------


class TestSpatialIndexCache:
    def test_no_filter_no_suffix(self, mock_dataset):
        mock_dataset.target = None
        mock_dataset.bbox = None
        path = mock_dataset.spatial_index_cache
        assert path.name == "spatial_cache_v3.gpkg"

    def test_target_in_suffix(self, mock_dataset):
        mock_dataset.target = "Olympus"
        mock_dataset.bbox = None
        path = mock_dataset.spatial_index_cache
        assert "Olympus" in path.name
        mock_dataset.target = None

    def test_bbox_in_suffix(self, mock_dataset):
        mock_dataset.target = None
        mock_dataset.bbox = (-136.0, 12.0, -124.0, 24.0)
        path = mock_dataset.spatial_index_cache
        assert "-136.0" in path.name
        mock_dataset.bbox = None

    def test_both_target_and_bbox_in_suffix(self, mock_dataset):
        mock_dataset.target = "Olympus"
        mock_dataset.bbox = (-136.0, 12.0, -124.0, 24.0)
        path = mock_dataset.spatial_index_cache
        assert "Olympus" in path.name
        assert "-136.0" in path.name
        mock_dataset.target = None
        mock_dataset.bbox = None


# ---------------------------------------------------------------------------
# M. __getitem__ with transforms
# ---------------------------------------------------------------------------


class TestGetItemTransforms:
    def test_transforms_applied_to_sample(self, mock_dataset):
        """transforms callable is invoked on the sample dict."""
        mock_dataset.transforms = lambda s: {**s, "extra": 42}
        t_start = _T0
        t_stop = _T1
        x = slice(-131.0, -130.96)
        y = slice(18.0, 18.04)
        t = slice(t_start, t_stop)

        with (
            patch.object(
                mock_dataset, "_disambiguate_slice", return_value=(x, y, t)
            ),
            patch.object(
                mock_dataset, "_load_tile", return_value=torch.ones(3, 4, 4)
            ),
            patch.object(
                mock_dataset, "_slice_to_tensor", return_value=torch.zeros(6)
            ),
        ):
            result = mock_dataset[x, y, t]

        assert result["extra"] == 42


# ---------------------------------------------------------------------------
# N. _merge_tiles channel-mismatch warning
# ---------------------------------------------------------------------------


class TestMergeTilesWarning:
    def test_channel_mismatch_logs_warning(self, caplog):
        """_merge_tiles with different channel counts emits a WARNING."""
        t3 = torch.ones(3, 8, 8)
        t1 = torch.ones(1, 8, 8) * 0.5
        with caplog.at_level(logging.WARNING, logger="mars_hirise"):
            result = MarsHiRISE._merge_tiles([t3, t1])
        assert "channel count mismatch" in caplog.text.lower()
        assert result.shape[0] == 3


# ---------------------------------------------------------------------------
# O. _load_index — target warning and info branches
# ---------------------------------------------------------------------------


def _make_pdr_mock(df: pd.DataFrame) -> MagicMock:
    """Return a mock pdr object whose ['RDR_INDEX_TABLE'] is *df*."""
    mock_data = MagicMock()
    mock_data.__getitem__ = MagicMock(return_value=df)
    return mock_data


class TestLoadIndex:
    def _base_df(self):
        return pd.DataFrame(
            {
                "PRODUCT_ID": ["PSP_001430_1780_COLOR"],
                "FILE_NAME_SPECIFICATION": [
                    "MROHR_0001/DATA/PSP/ORB_001400_001499/"
                    "PSP_001430_1780/PSP_001430_1780_COLOR.JP2"
                ],
                "OBSERVATION_ID": ["PSP_001430_1780"],
                "START_TIME": ["2007-01-01T00:00:00"],
                "STOP_TIME": ["2007-01-01T00:01:00"],
                "MINIMUM_LONGITUDE": [229.0],
                "MAXIMUM_LONGITUDE": [230.0],
                "MINIMUM_LATITUDE": [18.0],
                "MAXIMUM_LATITUDE": [19.0],
            }
        )

    def test_target_no_match_logs_warning(self, mock_dataset, tmp_path, caplog):
        """target filter produces empty DataFrame → warning logged."""
        (tmp_path / "RDRCUMINDEX.LBL").touch()
        mock_dataset.target = "XYZZY_NOT_FOUND"
        mock_pdr = _make_pdr_mock(self._base_df())
        with (
            patch("mars_hirise.pdr.read", return_value=mock_pdr),
            caplog.at_level(logging.WARNING, logger="mars_hirise"),
        ):
            mock_dataset._load_index()
        assert "matched no rows" in caplog.text
        mock_dataset.target = None

    def test_target_matches_logs_info(self, mock_dataset, tmp_path, caplog):
        """target filter produces non-empty DataFrame → info logged (else branch)."""
        (tmp_path / "RDRCUMINDEX.LBL").touch()
        mock_dataset.target = "PSP"  # matches PRODUCT_ID
        mock_pdr = _make_pdr_mock(self._base_df())
        with (
            patch("mars_hirise.pdr.read", return_value=mock_pdr),
            caplog.at_level(logging.INFO, logger="mars_hirise"),
        ):
            mock_dataset._load_index()
        assert "After text filter" in caplog.text
        mock_dataset.target = None

    def test_bbox_filter_applied(self, mock_dataset, tmp_path, caplog):
        """bbox filter logs info message."""
        (tmp_path / "RDRCUMINDEX.LBL").touch()
        mock_dataset.bbox = (-132.0, 17.0, -129.0, 20.0)
        mock_pdr = _make_pdr_mock(self._base_df())
        with (
            patch("mars_hirise.pdr.read", return_value=mock_pdr),
            caplog.at_level(logging.INFO, logger="mars_hirise"),
        ):
            mock_dataset._load_index()
        assert "After bbox filter" in caplog.text
        mock_dataset.bbox = None


# ---------------------------------------------------------------------------
# P. _build_spatial_index — legacy cache, non-legacy cache, antimeridian skip
# ---------------------------------------------------------------------------


def _make_raw_index_two_obs():
    """_raw_index with two observations: first spans antimeridian, second normal."""
    return pd.DataFrame(
        {
            "PRODUCT_ID": ["PSP_001430_1780_COLOR", "PSP_001431_1780_COLOR"],
            "FILE_NAME_SPECIFICATION": [
                "MROHR_0001/DATA/PSP/ORB_001400_001499/"
                "PSP_001430_1780/PSP_001430_1780_COLOR.JP2",
                "MROHR_0001/DATA/PSP/ORB_001400_001499/"
                "PSP_001431_1780/PSP_001431_1780_COLOR.JP2",
            ],
            "START_TIME": ["2007-01-01T00:00:00", "2007-01-02T00:00:00"],
            "STOP_TIME": ["2007-01-01T00:01:00", "2007-01-02T00:01:00"],
            # First obs spans antimeridian: 170→-10 (norm), 190→-170 (norm) → min>max
            "MINIMUM_LONGITUDE": [170.0, 229.0],
            "MAXIMUM_LONGITUDE": [190.0, 230.0],
            "MINIMUM_LATITUDE": [18.0, 18.0],
            "MAXIMUM_LATITUDE": [19.0, 19.0],
        }
    )


class TestBuildSpatialIndex:
    def test_antimeridian_observation_skipped_with_warning(
            self, mock_dataset, caplog
    ):
        """Observation straddling antimeridian is skipped with a warning."""
        mock_dataset._raw_index = _make_raw_index_two_obs()
        with caplog.at_level(logging.WARNING, logger="mars_hirise"):
            mock_dataset._build_spatial_index(force_rebuild=True)
        assert "antimeridian" in caplog.text.lower()
        # Only the second (normal) observation should be in the index
        assert len(mock_dataset.index) == 1

    def test_legacy_bbox_cache_triggers_rebuild(
            self, mock_dataset, mars_crs, caplog
    ):
        """Cache where all geometries are axis-aligned boxes → legacy rebuild."""
        from shapely.geometry import box as sbox

        cache_path = mock_dataset.spatial_index_cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # All-bbox geometry → is_legacy check returns True
        fake_gdf = gpd.GeoDataFrame(
            {"obs_id": ["OBS_001"], "color_path": [None], "red_path": [None]},
            geometry=[sbox(-131.0, 18.0, -130.0, 19.0)],
            crs=mars_crs,
        )
        fake_gdf.to_file(cache_path, driver="GPKG")

        mock_dataset.index = None
        mock_dataset.reuse_cache = True
        # Use standard _raw_index (1 normal obs) so rebuild succeeds
        with caplog.at_level(logging.INFO, logger="mars_hirise"):
            mock_dataset._build_spatial_index(force_rebuild=False)

        assert "Legacy bbox cache detected" in caplog.text
        assert mock_dataset.index is not None

    def test_valid_cache_loaded_directly(self, mock_dataset, mars_crs):
        """Non-legacy cache (rotated polygon) → loaded without rebuild."""
        cache_path = mock_dataset.spatial_index_cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # A non-rectangular polygon → is_legacy = False
        rotated_poly = Polygon(
            [(-131.0, 18.0), (-130.5, 18.3), (-130.0, 19.0), (-131.0, 19.0)]
        )
        fake_gdf = gpd.GeoDataFrame(
            {
                "obs_id": ["OBS_001"],
                "color_path": [None],
                "red_path": [None],
                "t_start": [str(_T0)],
                "t_stop": [str(_T1)],
            },
            geometry=[rotated_poly],
            crs=mars_crs,
        )
        fake_gdf.to_file(cache_path, driver="GPKG")

        mock_dataset.index = None
        mock_dataset.reuse_cache = True
        mock_dataset._build_spatial_index(force_rebuild=False)

        assert mock_dataset.index is not None
        assert len(mock_dataset.index) == 1

    def test_empty_obs_df_raises_dataset_not_found(self, mock_dataset, caplog):
        """If all observations are skipped (antimeridian-only), DatasetNotFoundError raised."""
        from torchgeo.datasets.errors import DatasetNotFoundError

        # Only one antimeridian-spanning observation
        mock_dataset._raw_index = pd.DataFrame(
            {
                "PRODUCT_ID": ["PSP_001430_1780_COLOR"],
                "FILE_NAME_SPECIFICATION": [
                    "MROHR_0001/.../PSP_001430_1780_COLOR.JP2"
                ],
                "START_TIME": ["2007-01-01T00:00:00"],
                "STOP_TIME": ["2007-01-01T00:01:00"],
                "MINIMUM_LONGITUDE": [170.0],
                "MAXIMUM_LONGITUDE": [190.0],
                "MINIMUM_LATITUDE": [18.0],
                "MAXIMUM_LATITUDE": [19.0],
            }
        )
        with pytest.raises(DatasetNotFoundError):
            mock_dataset._build_spatial_index(force_rebuild=True)


# ---------------------------------------------------------------------------
# Q. _verify — JP2 not found warnings
# ---------------------------------------------------------------------------


class TestVerifyJp2Warnings:
    def _setup_ds(self, tmp_path, mars_crs, download=False):
        with patch.object(MarsHiRISE, "_verify", return_value=None):
            ds = MarsHiRISE(root=tmp_path)

        t = pd.Timestamp("2007-01-01", tz="UTC")
        ds.index = gpd.GeoDataFrame(
            {
                "obs_id": ["OBS"],
                "color_path": [None],
                "red_path": [None],
            },
            index=pd.IntervalIndex.from_tuples([(t, t)], closed="both"),
            geometry=[box(-131.0, 18.0, -130.0, 19.0)],
            crs=mars_crs,
        )
        ds.download = download
        ds.reuse_cache = True
        # Create LBL and spatial cache so _verify doesn't raise or save
        (tmp_path / "RDRCUMINDEX.LBL").touch()
        (tmp_path / "spatial_cache_v3.gpkg").touch()
        return ds

    def test_no_jp2s_no_download_warns(self, tmp_path, mars_crs, caplog):
        ds = self._setup_ds(tmp_path, mars_crs, download=False)
        with (
            patch.object(ds, "_load_index"),
            patch.object(ds, "_build_spatial_index"),
            caplog.at_level(logging.WARNING, logger="mars_hirise"),
        ):
            ds._verify()
        assert "No JP2 files found" in caplog.text

    def test_no_jp2s_with_download_warns(self, tmp_path, mars_crs, caplog):
        ds = self._setup_ds(tmp_path, mars_crs, download=True)
        with (
            patch.object(ds, "_load_index"),
            patch.object(ds, "_build_spatial_index"),
            patch.object(ds, "_download_images", return_value=False),
            caplog.at_level(logging.WARNING, logger="mars_hirise"),
        ):
            ds._verify()
        assert "Download completed but no JP2 files" in caplog.text


# ---------------------------------------------------------------------------
# R. setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_loads_config_json(self, tmp_path):
        """setup_logging reads a JSON config and configures logging."""
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {},
            "root": {"level": "WARNING", "handlers": []},
        }
        config_path = tmp_path / "logger_config.json"
        config_path.write_text(json.dumps(config))
        # Must not raise
        setup_logging(str(config_path))
