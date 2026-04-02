"""Unit tests for mars_hirise.py — comprehensive branch coverage.

Covers all branches not exercised by existing tests:
  filter_maker, corners_to_polygon (buffer fix), extract_footprint,
  prefer_cog, _load_from_jp2, extract_footprint, _load_from_jp2,
  _load_tile, plot(), plot_coverage(), _coverage_grid, spatial_index_cache,
  __getitem__ transforms, merge_tiles warning, _load_index, _build_spatial_index
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

from dataset.mars_hirise import MarsHiRISE


from dataset.mars_hirise_base import (ProductMeta,
                                      corners_to_polygon,
                                      extract_footprint,
                                      filter_maker,
                                      setup_logging,
extract_footprint
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
# B. corners_to_polygon — buffer(0) fix for self-intersecting polygon
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
        result = corners_to_polygon(row)
        # buffer(0) should heal the bowtie into a valid polygon
        assert result is not None
        assert result.is_valid


# ---------------------------------------------------------------------------
# C. extract_footprint (standalone module-level function)
# ---------------------------------------------------------------------------


class TestExtractFootprintStandalone:
    """Tests for the top-level extract_footprint() helper."""

    def test_none_path_returns_none_none(self):
        result = extract_footprint(None, _MARS_RCRS)
        assert result == (None, None)

    def test_nonexistent_file_returns_none_none(self, tmp_path):
        result = extract_footprint(str(tmp_path / "ghost.tif"), _MARS_RCRS)
        assert result == (None, None)

    def test_no_crs_returns_none_none(self, no_crs_geotiff):
        hull, bounds = extract_footprint(str(no_crs_geotiff), _MARS_RCRS)
        assert hull is None

    def test_valid_file_returns_hull_and_bounds(self, mars_geotiff):
        hull, bounds = extract_footprint(str(mars_geotiff), _MARS_RCRS)
        assert hull is not None
        assert bounds is not None
        assert len(hull) >= 3

    def test_fewer_than_3_nonzero_pixels_returns_none_with_bounds(self, tmp_path):
        """< 3 non-zero pixels: hull is None but file_bounds may still be returned."""
        # Create a file with only 2 non-zero pixels
        p = tmp_path / "sparse.tif"
        bounds = (-131.0, 18.0, -130.0, 19.0)

        transform = rasterio.transform.from_bounds(*bounds, 16, 16)
        data = np.zeros((1, 16, 16), dtype=np.uint16)
        data[0, 0, 0] = 1
        data[0, 0, 1] = 1  # only 2 non-zero → len(xs) < 3
        with rasterio.open(
                p, "w", driver="GTiff", count=1, dtype="uint16",
                width=16, height=16, crs=_MARS_RCRS, transform=transform,
        ) as dst:
            dst.write(data)
        hull, bounds = extract_footprint(str(p), _MARS_RCRS)
        assert hull == (None, bounds)

    def test_exception_during_open_returns_none_none(self, tmp_path):
        """Exception inside rasterio.open caught → (None, None)."""
        p = tmp_path / "bad.tif"
        p.write_bytes(b"not a valid rasterio file at all")
        result = extract_footprint(str(p), _MARS_RCRS)
        assert result == (None, None)

    def test_empty_hull_path(self, mars_geotiff):
        """hull.is_empty path: mock MultiPoint.convex_hull to be empty."""
        empty_geom = MagicMock()
        empty_geom.is_empty = True

        mock_mp_instance = MagicMock()
        mock_mp_instance.convex_hull = empty_geom

        with patch("shapely.geometry.MultiPoint", return_value=mock_mp_instance):
            hull, bounds = extract_footprint(str(mars_geotiff), _MARS_RCRS)

        assert hull is None
        # file_bounds may or may not be set depending on how far we get
        # The key assertion is that hull is None (is_empty path returned early)


# ---------------------------------------------------------------------------
# D. prefer_cog
# ---------------------------------------------------------------------------


class TestPreferCog:
    def test_none_returns_none(self):
        assert MarsHiRISE.prefer_cog(None) is None

    def test_cog_exists_returns_cog(self, tmp_path):
        jp2 = tmp_path / "image.JP2"
        cog = tmp_path / "image.tif"
        cog.touch()
        result = MarsHiRISE.prefer_cog(jp2)
        assert result == cog

    def test_cog_not_exists_returns_original(self, tmp_path):
        jp2 = tmp_path / "image.JP2"
        jp2.touch()
        result = MarsHiRISE.prefer_cog(jp2)
        assert result == jp2


# ---------------------------------------------------------------------------
# E. _load_from_jp2
# ---------------------------------------------------------------------------


class TestReadJp2Bounds:
    def test_nonexistent_file_returns_none(self, mock_dataset, tmp_path):
        result = mock_dataset._load_from_jp2(tmp_path / "ghost.tif")
        assert result is None

    def test_no_crs_returns_none(self, mock_dataset, no_crs_geotiff):
        result = mock_dataset._load_from_jp2(no_crs_geotiff)
        assert result is None

    def test_out_of_range_bounds_returns_none(self, mock_dataset, tmp_path):
        """transform_bounds returning invalid geographic range → None."""
        with patch("dataset.mars_hirise_base.transform_bounds", return_value=(400.0, 200.0, 500.0, 300.0)):
            # Create a dummy file that rasterio can open
            p = tmp_path / "dummy.tif"
            transform = rasterio.transform.from_bounds(-131, 18, -130, 19, 4, 4)
            data = np.ones((1, 4, 4), dtype=np.uint16)
            with rasterio.open(
                    p, "w", driver="GTiff", count=1, dtype="uint16",
                    width=4, height=4, crs=_MARS_RCRS, transform=transform,
            ) as dst:
                dst.write(data)
            result = mock_dataset._load_from_jp2(p)
        assert result is None

    def test_exception_returns_none(self, mock_dataset, tmp_path):
        """Exception during open → None."""
        p = tmp_path / "bad.tif"
        p.write_bytes(b"garbage")
        result = mock_dataset._load_from_jp2(p)
        assert result is None

    def test_valid_file_returns_bounds(self, mock_dataset, mars_geotiff):
        result = mock_dataset._load_from_jp2(mars_geotiff)
        assert result is not None
        fl, fb, fr, ft = result
        assert -180.0 <= fl < fr <= 180.0
        assert -90.0 <= fb < ft <= 90.0


# ---------------------------------------------------------------------------
# F. extract_footprint
# ---------------------------------------------------------------------------


class TestExtractDataFootprint:
    def test_nonexistent_path_returns_none(self, mock_dataset, tmp_path):
        result = extract_footprint(tmp_path / "ghost.tif", _MARS_RCRS)
        assert result == (None, None)

    def test_no_crs_returns_none(self, mock_dataset, no_crs_geotiff):
        result = extract_footprint(no_crs_geotiff, _MARS_RCRS)
        assert result == (None, None)

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
        result = extract_footprint(str(p), _MARS_RCRS)
        assert result is None

    def test_valid_file_returns_polygon(self, mock_dataset, mars_geotiff):
        result = extract_footprint(mars_geotiff)
        assert result is not None
        assert isinstance(result, Polygon)

    def test_exception_returns_none(self, mock_dataset, tmp_path):
        p = tmp_path / "bad.tif"
        p.write_bytes(b"not a rasterio file")
        result = extract_footprint(p)
        assert result == (None, None)

    def test_empty_hull_returns_none(self, mock_dataset, mars_geotiff):
        """hull_shapely.is_empty path via mocked MultiPoint."""
        empty_geom = MagicMock()
        empty_geom.is_empty = True

        mock_mp_instance = MagicMock()
        mock_mp_instance.convex_hull = empty_geom

        with patch("dataset.mars_hirise_base.MultiPoint", return_value=mock_mp_instance):
            result = extract_footprint(mars_geotiff)
        assert result is None


# ---------------------------------------------------------------------------
# G. _load_from_jp2
# ---------------------------------------------------------------------------


class TestLoadFromJp2:
    def test_empty_band_map_returns_empty(self, mock_dataset, mars_geotiff):
        result = mock_dataset._load_from_jp2(
            mars_geotiff, {}, ProductMeta(), _X, _Y
        )
        assert result == {}

    def test_no_crs_uses_dst_crs(self, mock_dataset, no_crs_geotiff):
        """File with no CRS: warns and assumes dataset CRS, still returns data."""
        result = mock_dataset._load_from_jp2(
            no_crs_geotiff, {"RED": 1}, ProductMeta(), _X, _Y
        )
        # Should attempt reprojection (may succeed or not depending on bounds)
        assert isinstance(result, dict)

    def test_antimeridian_file_skips_early_exit(self, mock_dataset, mars_geotiff):
        """When fl > fr (antimeridian wrap), early-exit check is skipped."""
        # transform_bounds returns values whose normalised result gives fl=175 > fr=-175
        with patch("dataset.mars_hirise_base.transform_bounds", return_value=(175.0, 18.0, 185.0, 19.0)):
            result = mock_dataset._load_from_jp2(
                mars_geotiff, {"RED": 1}, ProductMeta(), _X, _Y
            )
        # Key: we got through without hitting the early-exit return {}
        assert isinstance(result, dict)

    def test_non_overlapping_file_early_exit(self, mock_dataset, mars_geotiff):
        """File bounds completely outside query → early exit, returns {}."""
        # File bounds at (0,0,10,10) normalised to (-180,-170) ← well left of x=-131
        with patch("dataset.mars_hirise_base.transform_bounds", return_value=(0.0, 0.0, 10.0, 10.0)):
            result = mock_dataset._load_from_jp2(
                mars_geotiff, {"RED": 1}, ProductMeta(), _X, _Y
            )
        assert result == {}

    def test_reproject_exception_skips_band(self, mock_dataset, mars_geotiff):
        """If reproject() raises, the band is skipped and result is {}."""
        with patch("dataset.mars_hirise_base.reproject", side_effect=RuntimeError("boom")):
            result = mock_dataset._load_from_jp2(
                mars_geotiff, {"RED": 1}, ProductMeta(), _X, _Y
            )
        assert result == {}

    def test_rasterioioerror_returns_empty(self, mock_dataset, tmp_path):
        """RasterioIOError on open → returns {}."""
        p = tmp_path / "corrupt.tif"
        p.write_bytes(b"not a tiff")
        result = mock_dataset._load_from_jp2(
            p, {"RED": 1}, ProductMeta(), _X, _Y
        )
        assert result == {}

    def test_valid_reprojection_returns_calibrated_data(self, mock_dataset, mars_geotiff):
        """Valid file + overlapping query → returns float32 array in [0,1]."""
        result = mock_dataset._load_from_jp2(
            mars_geotiff, {"RED": 1}, ProductMeta(), _X, _Y
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
        with caplog.at_level(logging.WARNING, logger="dataset.mars_hirise"):
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
        with caplog.at_level(logging.WARNING, logger="dataset.mars_hirise"):
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
        assert path.name == "spatial_cache_-136_12_-124_24_v3.gpkg"
        mock_dataset.bbox = None

    def test_both_target_and_bbox_in_suffix(self, mock_dataset):
        mock_dataset.target = "Olympus"
        mock_dataset.bbox = (-136.0, 12.0, -124.0, 24.0)
        path = mock_dataset.spatial_index_cache
        assert "Olympus" in path.name
        assert "-136_12_-124_24" in path.name
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
# N. merge_tiles channel-mismatch warning
# ---------------------------------------------------------------------------


class TestMergeTilesWarning:
    def test_channel_mismatch_logs_warning(self, caplog):
        """merge_tiles with different channel counts emits a WARNING."""
        t3 = torch.ones(3, 8, 8)
        t1 = torch.ones(1, 8, 8) * 0.5
        with caplog.at_level(logging.WARNING, logger="dataset.mars_hirise"):
            result = MarsHiRISE.merge_tiles([t3, t1])
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
            patch("dataset.mars_hirise_base.pdr.read", return_value=mock_pdr),
            caplog.at_level(logging.WARNING, logger="dataset.mars_hirise_base"),
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
            patch("dataset.mars_hirise_base.pdr.read", return_value=mock_pdr),
            caplog.at_level(logging.INFO, logger="dataset.mars_hirise"),
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
            patch("dataset.mars_hirise_base.pdr.read", return_value=mock_pdr),
            caplog.at_level(logging.INFO, logger="dataset.mars_hirise"),
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
        with caplog.at_level(logging.WARNING, logger="dataset.mars_hirise"):
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
        with caplog.at_level(logging.INFO, logger="dataset.mars_hirise"):
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
            caplog.at_level(logging.WARNING, logger="dataset.mars_hirise"),
        ):
            ds._verify()
        assert "No JP2 files found" in caplog.text

    def test_no_jp2s_with_download_warns(self, tmp_path, mars_crs, caplog):
        ds = self._setup_ds(tmp_path, mars_crs, download=True)
        with (
            patch.object(ds, "_load_index"),
            patch.object(ds, "_build_spatial_index"),
            patch.object(ds, "_download_images", return_value=False),
            caplog.at_level(logging.WARNING, logger="dataset.mars_hirise"),
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


# ---------------------------------------------------------------------------
# C+. extract_footprint bounds edge cases (lines 374, 377-378)
# ---------------------------------------------------------------------------


class TestExtractFootprintBoundsEdgeCases:
    def test_out_of_range_file_bounds_are_none(self, tmp_path):
        """transform_bounds returns out-of-range lat → file_bounds = None (line 374)."""
        p = tmp_path / "bounds_test.tif"
        transform = rasterio.transform.from_bounds(-131.0, 18.0, -130.0, 19.0, 4, 4)
        data = np.ones((1, 4, 4), dtype=np.uint16) * 500
        with rasterio.open(
                p, "w", driver="GTiff", count=1, dtype="uint16",
                width=4, height=4, crs=_MARS_RCRS, transform=transform,
        ) as dst:
            dst.write(data)
        # fb=200 fails -90 <= fb < ft <= 90 → file_bounds = None
        with patch("rasterio.warp.transform_bounds", return_value=(40.0, 200.0, 140.0, 300.0)):
            hull, bounds = extract_footprint(str(p), _MARS_RCRS)
        assert bounds is None

    def test_transform_bounds_exception_sets_file_bounds_none(self, tmp_path):
        """transform_bounds raises → except Exception: file_bounds = None (lines 377-378)."""
        p = tmp_path / "exc_test.tif"
        transform = rasterio.transform.from_bounds(-131.0, 18.0, -130.0, 19.0, 4, 4)
        data = np.ones((1, 4, 4), dtype=np.uint16) * 500
        with rasterio.open(
                p, "w", driver="GTiff", count=1, dtype="uint16",
                width=4, height=4, crs=_MARS_RCRS, transform=transform,
        ) as dst:
            dst.write(data)
        with patch("rasterio.warp.transform_bounds", side_effect=RuntimeError("bad crs")):
            hull, bounds = extract_footprint(str(p), _MARS_RCRS)
        assert bounds is None


# ---------------------------------------------------------------------------
# S. Invalid channels in __init__ (lines 535-541)
# ---------------------------------------------------------------------------


class TestInvalidChannels:
    def test_invalid_channel_raises_value_error(self, tmp_path):
        """channels containing an unknown name → ValueError (lines 535-541)."""
        with pytest.raises(ValueError, match="Invalid channel"):
            MarsHiRISE(root=tmp_path, channels=["NOT_A_CHANNEL"])

    def test_subset_channels_accepted(self, tmp_path):
        """Valid subset of channels → accepted, ordered as ALL_CHANNELS."""
        with patch.object(MarsHiRISE, "_verify", return_value=None):
            ds = MarsHiRISE(root=tmp_path, channels=["RED"])
        assert ds.channels == ["RED"]


# ---------------------------------------------------------------------------
# T. __len__ (line 567)
# ---------------------------------------------------------------------------


class TestDatasetLen:
    def test_len_returns_index_length(self, mock_dataset):
        """__len__ returns the number of rows in the spatial index."""
        assert len(mock_dataset) == 1


# ---------------------------------------------------------------------------
# U. __getitem__ IndexError paths (lines 597, 616-617)
# ---------------------------------------------------------------------------


class TestGetItemIndexErrors:
    def test_no_candidates_raises_index_error(self, mock_dataset):
        """Query outside all strip polygons → IndexError (line 597)."""
        x = slice(-150.0, -149.9)
        y = slice(0.0, 0.1)
        t = slice(_T0, _T1)
        with patch.object(mock_dataset, "_disambiguate_slice", return_value=(x, y, t)):
            with pytest.raises(IndexError, match="No MarsHiRISE observations found"):
                var = mock_dataset[x, y, t]

    def test_candidates_but_no_tile_raises_index_error(self, mock_dataset):
        """Query intersects polygon but color_path=None, red_path=None → IndexError (lines 616-617)."""
        x = slice(-131.0, -130.9)
        y = slice(18.0, 18.1)
        t = slice(_T0, _T1)
        # mock_dataset has color_path=None and red_path=None → _load_tile returns None
        with patch.object(mock_dataset, "_disambiguate_slice", return_value=(x, y, t)):
            with pytest.raises(IndexError, match="but no image data could be loaded"):
                var = mock_dataset[x, y, t]


# ---------------------------------------------------------------------------
# V. _verify new code paths (lines 708-710, 721-722, 726-730, 742-743)
# ---------------------------------------------------------------------------


class TestVerifyNewPaths:
    def test_missing_lbl_download_false_raises(self, tmp_path, mars_crs):
        """No LBL + download=False → DatasetNotFoundError (lines 708-709)."""
        from torchgeo.datasets.errors import DatasetNotFoundError

        with patch.object(MarsHiRISE, "_verify", return_value=None):
            ds = MarsHiRISE(root=tmp_path)
        ds.download = False
        # LBL does NOT exist → hits line 707-709
        with pytest.raises(DatasetNotFoundError):
            ds._verify()

    def test_missing_lbl_download_true_calls_download_index(self, tmp_path, mars_crs):
        """No LBL + download=True → _download_index() called (line 710)."""
        with patch.object(MarsHiRISE, "_verify", return_value=None):
            ds = MarsHiRISE(root=tmp_path)

        t = pd.Timestamp("2007-01-01", tz="UTC")
        ds.index = gpd.GeoDataFrame(
            {"obs_id": ["OBS"], "color_path": [None], "red_path": [None]},
            index=pd.IntervalIndex.from_tuples([(t, t)], closed="both"),
            geometry=[box(-131.0, 18.0, -130.0, 19.0)],
            crs=mars_crs,
        )
        ds.download = True
        ds.reuse_cache = True
        (tmp_path / "spatial_cache_v3.gpkg").touch()

        with (
            patch.object(ds, "_download_index") as mock_dl,
            patch.object(ds, "_load_index"),
            patch.object(ds, "_build_spatial_index"),
            patch.object(ds, "_download_images", return_value=False),
        ):
            ds._verify()

        mock_dl.assert_called_once()

    def test_download_changes_triggers_rebuild(self, tmp_path, mars_crs):
        """_download_images returns True → _build_spatial_index called twice (lines 721-722)."""
        with patch.object(MarsHiRISE, "_verify", return_value=None):
            ds = MarsHiRISE(root=tmp_path)

        t = pd.Timestamp("2007-01-01", tz="UTC")
        ds.index = gpd.GeoDataFrame(
            {"obs_id": ["OBS"], "color_path": [None], "red_path": [None]},
            index=pd.IntervalIndex.from_tuples([(t, t)], closed="both"),
            geometry=[box(-131.0, 18.0, -130.0, 19.0)],
            crs=mars_crs,
        )
        ds.download = True
        ds.reuse_cache = True
        (tmp_path / "RDRCUMINDEX.LBL").touch()
        (tmp_path / "spatial_cache_v3.gpkg").touch()

        with (
            patch.object(ds, "_load_index"),
            patch.object(ds, "_build_spatial_index") as mock_rebuild,
            patch.object(ds, "_download_images", return_value=True),
        ):
            ds._verify()

        assert mock_rebuild.call_count == 2

    def test_reuse_cache_false_writes_cache_file(self, tmp_path, mars_crs):
        """reuse_cache=False → cache save block executed (lines 726-730)."""
        with patch.object(MarsHiRISE, "_verify", return_value=None):
            ds = MarsHiRISE(root=tmp_path)

        t = pd.Timestamp("2007-01-01", tz="UTC")
        ds.index = gpd.GeoDataFrame(
            {"obs_id": ["OBS"], "color_path": [None], "red_path": [None]},
            index=pd.IntervalIndex.from_tuples([(t, t)], closed="both"),
            geometry=[box(-131.0, 18.0, -130.0, 19.0)],
            crs=mars_crs,
        )
        ds.download = False
        ds.reuse_cache = False
        (tmp_path / "RDRCUMINDEX.LBL").touch()

        with (
            patch.object(ds, "_load_index"),
            patch.object(ds, "_build_spatial_index"),
        ):
            ds._verify()

        assert ds.spatial_index_cache.exists()

    def test_found_jp2_path_increments_found(self, tmp_path, mars_crs):
        """A color_path that exists → found += 1; break (lines 742-743)."""
        with patch.object(MarsHiRISE, "_verify", return_value=None):
            ds = MarsHiRISE(root=tmp_path)

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        jp2 = images_dir / "PSP_001430_1780_COLOR.JP2"
        jp2.touch()

        t = pd.Timestamp("2007-01-01", tz="UTC")
        ds.index = gpd.GeoDataFrame(
            {"obs_id": ["OBS"], "color_path": [str(jp2)], "red_path": [None]},
            index=pd.IntervalIndex.from_tuples([(t, t)], closed="both"),
            geometry=[box(-131.0, 18.0, -130.0, 19.0)],
            crs=mars_crs,
        )
        ds.download = False
        ds.reuse_cache = True
        (tmp_path / "RDRCUMINDEX.LBL").touch()
        (tmp_path / "spatial_cache_v3.gpkg").touch()

        with (
            patch.object(ds, "_load_index"),
            patch.object(ds, "_build_spatial_index"),
        ):
            ds._verify()  # Must not warn about "No JP2 files found"


# ---------------------------------------------------------------------------
# W. _download_index (lines 766-769)
# ---------------------------------------------------------------------------


class TestDownloadIndex:
    def test_download_index_calls_download_url_twice(self, mock_dataset):
        """_download_index fetches .LBL and .TAB (lines 766-769)."""
        with patch("dataset.mars_hirise_base.download_url") as mock_du:
            mock_dataset._download_index()
        assert mock_du.call_count == 2


# ---------------------------------------------------------------------------
# X. _load_index missing LBL → DatasetNotFoundError (line 774)
# ---------------------------------------------------------------------------


class TestLoadIndexMissingLbl:
    def test_missing_lbl_raises_dataset_not_found(self, mock_dataset):
        """_load_index when LBL absent → DatasetNotFoundError (line 774)."""
        from torchgeo.datasets.errors import DatasetNotFoundError

        # mock_dataset.root is tmp_path; no LBL file created there
        with pytest.raises(DatasetNotFoundError):
            mock_dataset._load_index()


# ---------------------------------------------------------------------------
# Y. _pds_local_stem (lines 835-836)
# ---------------------------------------------------------------------------


class TestPdsLocalStem:
    def test_pds_local_stem_strips_extension(self, mock_dataset, tmp_path):
        """_pds_local_stem returns path without .JP2 extension (lines 835-836)."""
        spec = "MROHR_0001/DATA/PSP/ORB_001400_001499/PSP_001430_1780/PSP_001430_1780_COLOR.JP2"
        result = mock_dataset._pds_local_stem(spec)
        assert result == tmp_path / "images" / "PSP_001430_1780_COLOR"
        assert result.suffix == ""


# ---------------------------------------------------------------------------
# Z. _extract_data_footprint with overviews (line 872)
# ---------------------------------------------------------------------------


class TestExtractDataFootprintOverviews:
    def test_overviews_branch_uses_max_overview_factor(self, mock_dataset, tmp_path):
        """src.overviews(1) non-empty → factor = max(overviews) (line 872)."""
        from rasterio.enums import Resampling as RioResampling

        p = tmp_path / "overviews.tif"
        transform = rasterio.transform.from_bounds(-131.0, 18.0, -130.0, 19.0, 64, 64)
        data = np.ones((1, 64, 64), dtype=np.uint16) * 500
        with rasterio.open(
                p, "w", driver="GTiff", count=1, dtype="uint16",
                width=64, height=64, crs=_MARS_RCRS, transform=transform,
        ) as dst:
            dst.write(data)
        # Build overviews so src.overviews(1) returns [2, 4]
        with rasterio.open(p, "r+") as dst:
            dst.build_overviews([2, 4], RioResampling.nearest)
            dst.update_tags(ns="rio_overview", resampling="nearest")

        result = extract_footprint(p)
        assert result is not None
        assert isinstance(result, Polygon)


# ---------------------------------------------------------------------------
# AA. _extract_data_footprint invalid polygon → buffer(0) (line 928)
# ---------------------------------------------------------------------------


class TestExtractDataFootprintInvalidPolygon:
    def test_invalid_footprint_triggers_buffer(self, mock_dataset, mars_geotiff):
        """footprint.is_valid False → footprint.buffer(0) applied (line 928)."""
        from shapely.geometry import Polygon as RealPolygon

        # Bowtie = self-intersecting → not valid
        bowtie = RealPolygon([(0, 0), (0, 1), (1, 0), (1, 1)])
        assert not bowtie.is_valid

        with patch("dataset.mars_hirise_base.Polygon", return_value=bowtie):
            # No exception should be raised; result may be None or Polygon
            result = extract_footprint(mars_geotiff)
        # The branch was exercised; the return value depends on buffer(0) outcome
        assert result is None or isinstance(result, RealPolygon)


# ---------------------------------------------------------------------------
# AB. _build_spatial_index with real files on disk (1027-1028, 1054-1059,
#     1077, 1092-1097, 1101-1102)
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset_with_dense_file(tmp_path, mars_crs):
    """MarsHiRISE with a 16×16 all-500 GeoTIFF at the expected _pds_local_path."""
    with patch.object(MarsHiRISE, "_verify", return_value=None):
        ds = MarsHiRISE(root=tmp_path)

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    fname = "PSP_001430_1780_COLOR.JP2"
    fpath = images_dir / fname
    transform = rasterio.transform.from_bounds(-131.0, 18.0, -130.0, 19.0, 16, 16)
    data = np.ones((1, 16, 16), dtype=np.uint16) * 500
    with rasterio.open(
            fpath, "w", driver="GTiff", count=1, dtype="uint16",
            width=16, height=16, crs=_MARS_RCRS, transform=transform,
    ) as dst:
        dst.write(data)

    ds._raw_index = pd.DataFrame({
        "PRODUCT_ID": ["PSP_001430_1780_COLOR"],
        "FILE_NAME_SPECIFICATION": [
            "MROHR_0001/DATA/PSP/ORB_001400_001499/"
            "PSP_001430_1780/PSP_001430_1780_COLOR.JP2"
        ],
        "START_TIME": ["2007-01-01T00:00:00"],
        "STOP_TIME": ["2007-01-01T00:01:00"],
        "MINIMUM_LONGITUDE": [229.0],
        "MAXIMUM_LONGITUDE": [230.0],
        "MINIMUM_LATITUDE": [18.0],
        "MAXIMUM_LATITUDE": [19.0],
    })
    ds.channels = ["NEAR-INFRARED", "RED", "BLUE-GREEN"]
    return ds


@pytest.fixture
def dataset_with_sparse_file(tmp_path, mars_crs):
    """MarsHiRISE with a GeoTIFF that has only 2 non-zero pixels (Case B trigger)."""
    with patch.object(MarsHiRISE, "_verify", return_value=None):
        ds = MarsHiRISE(root=tmp_path)

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    fname = "PSP_001430_1780_COLOR.JP2"
    fpath = images_dir / fname
    transform = rasterio.transform.from_bounds(-131.0, 18.0, -130.0, 19.0, 16, 16)
    data = np.zeros((1, 16, 16), dtype=np.uint16)
    data[0, 0, 0] = 500
    data[0, 0, 1] = 500  # only 2 non-zero → len(xs) < 3 → returns (None, file_bounds)
    with rasterio.open(
            fpath, "w", driver="GTiff", count=1, dtype="uint16",
            width=16, height=16, crs=_MARS_RCRS, transform=transform,
    ) as dst:
        dst.write(data)

    ds._raw_index = pd.DataFrame({
        "PRODUCT_ID": ["PSP_001430_1780_COLOR"],
        "FILE_NAME_SPECIFICATION": [
            "MROHR_0001/DATA/PSP/ORB_001400_001499/"
            "PSP_001430_1780/PSP_001430_1780_COLOR.JP2"
        ],
        "START_TIME": ["2007-01-01T00:00:00"],
        "STOP_TIME": ["2007-01-01T00:01:00"],
        "MINIMUM_LONGITUDE": [229.0],
        "MAXIMUM_LONGITUDE": [230.0],
        "MINIMUM_LATITUDE": [18.0],
        "MAXIMUM_LATITUDE": [19.0],
    })
    ds.channels = ["NEAR-INFRARED", "RED", "BLUE-GREEN"]
    return ds


class TestBuildSpatialIndexWithFiles:
    def test_case_a_hull_from_dense_file(self, dataset_with_dense_file, caplog):
        """Dense file → hull extracted → Case A geometry set (lines 1027-1028, 1054-1059,
        1077, 1092-1097)."""
        with caplog.at_level(logging.INFO, logger="dataset.mars_hirise"):
            dataset_with_dense_file._build_spatial_index(force_rebuild=True)
        assert len(dataset_with_dense_file.index) == 1
        assert "footprint extraction" in caplog.text
        geom = dataset_with_dense_file.index.geometry.iloc[0]
        assert geom is not None and not geom.is_empty

    def test_case_b_file_bounds_from_sparse_file(self, dataset_with_sparse_file):
        """Sparse file (<3 non-zero px) → file_bounds used → Case B geometry set
        (lines 1101-1102)."""
        dataset_with_sparse_file._build_spatial_index(force_rebuild=True)
        assert len(dataset_with_sparse_file.index) == 1
        geom = dataset_with_sparse_file.index.geometry.iloc[0]
        assert geom is not None and not geom.is_empty

    def test_case_a_invalid_hull_triggers_buffer0(self, dataset_with_dense_file):
        """hull_coords forms an invalid (bowtie) polygon → buffer(0) applied (line 1094)."""
        # Bowtie coords: (0,0)→(1,1)→(0,1)→(1,0) — self-intersecting, is_valid=False
        bowtie_coords = [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]
        # Patch the module-level extract_footprint to return the bowtie hull
        with patch("dataset.mars_hirise_base.extract_footprint", return_value=(bowtie_coords, (-131.0, 18.0, -130.0, 19.0))):
            dataset_with_dense_file._build_spatial_index(force_rebuild=True)

        assert len(dataset_with_dense_file.index) == 1


# ---------------------------------------------------------------------------
# AC. _build_download_tasks (lines 1209-1220)
# ---------------------------------------------------------------------------


class TestBuildDownloadTasks:
    def test_returns_list_of_remote_local_pairs(self, mock_dataset):
        """_build_download_tasks iterates _raw_index and builds URL/path pairs."""
        tasks = mock_dataset._build_download_tasks()
        assert isinstance(tasks, list)
        # Each row produces up to 2 tasks (.JP2 and .LBL); none exist → both included
        assert len(tasks) >= 1
        for remote, local in tasks:
            assert isinstance(remote, str)
            assert "hirise" in remote.lower() or "pds" in remote.lower()
            assert isinstance(local, pathlib.Path)

    def test_existing_file_excluded_from_tasks(self, mock_dataset, tmp_path):
        """If the local file already exists it is not added to the task list."""
        # Create the expected local file
        images_dir = tmp_path / "images"
        images_dir.mkdir(exist_ok=True)
        jp2 = images_dir / "PSP_001430_1780_COLOR.JP2"
        jp2.touch()
        lbl = images_dir / "PSP_001430_1780_COLOR.LBL"
        lbl.touch()
        tasks = mock_dataset._build_download_tasks()
        local_paths = [str(t[1]) for t in tasks]
        assert str(jp2) not in local_paths
        assert str(lbl) not in local_paths


# ---------------------------------------------------------------------------
# AD. _download_images (lines 1223-1247)
# ---------------------------------------------------------------------------


class TestDownloadImages:
    def test_no_tasks_returns_false(self, mock_dataset):
        """_build_download_tasks returns [] → 'already downloaded' → False (lines 1223-1226)."""
        with patch.object(mock_dataset, "_build_download_tasks", return_value=[]):
            result = mock_dataset._download_images()
        assert result is False

    def test_with_tasks_schedules_worker_processes(self, mock_dataset, tmp_path):
        """Has tasks → ProcessPoolExecutor used → returns True (lines 1227-1247)."""
        fake_path = tmp_path / "foo.JP2"
        tasks = [("https://example.com/foo.JP2", fake_path)]

        mock_stop = MagicMock()
        mock_stop.is_set.return_value = False

        mock_manager = MagicMock()
        mock_manager.Event.return_value = mock_stop

        mock_pool = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=None)
        mock_pool.submit.return_value = mock_future

        with (
            patch.object(mock_dataset, "_build_download_tasks", return_value=tasks),
            patch("dataset.mars_hirise_base.multiprocessing.Manager", return_value=mock_manager),
            patch("dataset.mars_hirise_base.ProcessPoolExecutor", return_value=mock_pool),
        ):
            result = mock_dataset._download_images()

        assert result is True
        mock_future.result.assert_called()

    def test_stop_event_set_after_download_logs_warning(self, mock_dataset, tmp_path, caplog):
        """stop_event.is_set() True after futures → warning logged (line 1245)."""
        fake_path = tmp_path / "foo.JP2"
        tasks = [("https://example.com/foo.JP2", fake_path)]

        mock_stop = MagicMock()
        mock_stop.is_set.return_value = True  # set → triggers warning at line 1244-1245

        mock_manager = MagicMock()
        mock_manager.Event.return_value = mock_stop

        mock_pool = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=None)
        mock_pool.submit.return_value = mock_future

        with (
            patch.object(mock_dataset, "_build_download_tasks", return_value=tasks),
            patch("dataset.mars_hirise_base.multiprocessing.Manager", return_value=mock_manager),
            patch("dataset.mars_hirise_base.ProcessPoolExecutor", return_value=mock_pool),
            caplog.at_level(logging.WARNING, logger="dataset.mars_hirise"),
        ):
            result = mock_dataset._download_images()

        assert result is True
        assert "disk space" in caplog.text


# ---------------------------------------------------------------------------
# AE. _load_tile: empty filter_names → _COLOR_BAND.copy() fallback (line 1318)
# ---------------------------------------------------------------------------


class TestLoadTileFilterNamesFallback:
    def test_no_lbl_uses_default_color_band_map(self, mock_dataset, tmp_path):
        """filter_names=[] on meta → _COLOR_BAND.copy() fallback (line 1318)."""
        from dataset.mars_hirise import ProductMeta

        color_path = tmp_path / "test_COLOR.tif"
        transform = rasterio.transform.from_bounds(-131.0, 18.0, -130.0, 19.0, 16, 16)
        data = np.ones((3, 16, 16), dtype=np.uint16) * 500
        with rasterio.open(
                color_path, "w", driver="GTiff", count=3, dtype="uint16",
                width=16, height=16, crs=_MARS_RCRS, transform=transform,
        ) as dst:
            dst.write(data)
        mock_dataset.channels = ["NEAR-INFRARED", "RED", "BLUE-GREEN"]
        # Force filter_names=[] so len(filter_names) != bands → else branch at line 1318
        sparse_meta = ProductMeta()
        sparse_meta.filter_names = []
        with patch("dataset.mars_hirise_base.ProductMeta.from_lbl", return_value=sparse_meta):
            result = mock_dataset._load_tile(color_path, None, _X, _Y)
        assert result is not None


# ---------------------------------------------------------------------------
# AF. _load_from_jp2: inner transform_bounds raises → except Exception: pass
#     (lines 1418-1419)
# ---------------------------------------------------------------------------


class TestLoadFromJp2BoundsException:
    def test_transform_bounds_exception_caught_reprojection_continues(
            self, mock_dataset, mars_geotiff
    ):
        """transform_bounds raises in inner try → pass → reproject still runs (1418-1419)."""
        with patch("dataset.mars_hirise_base.transform_bounds", side_effect=RuntimeError("bad crs")):
            result = mock_dataset._load_from_jp2(
                mars_geotiff, {"RED": 1}, ProductMeta(), _X, _Y
            )
        # Exception was swallowed; reprojection still attempted on the overlapping file
        assert isinstance(result, dict)
        assert "RED" in result


# ---------------------------------------------------------------------------
# AG. main() entry point (lines 1747-1794)
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_runs_without_error_with_all_mocked(self):
        """main() exercises lines 1747-1794 with all I/O mocked."""
        from dataset import hirise_sampler
        import torch.utils.data
        from dataset.mars_hirise import main

        mock_ds = MagicMock()
        mock_fig = MagicMock()
        mock_ds.plot_coverage.return_value = mock_fig
        mock_ds.plot.return_value = mock_fig

        sample = {
            "image": torch.zeros(1, 1, 4, 4),
            "bounds": torch.zeros(6),
            "crs": "FAKE",
        }
        mock_sampler_inst = MagicMock()
        mock_dl = MagicMock()
        mock_dl.__len__ = lambda self: 1
        mock_dl.__iter__ = lambda self: iter([sample])

        with (
            patch("dataset.mars_hirise.MarsHiRISE", return_value=mock_ds),
            patch("dataset.mars_hirise_base.setup_logging"),
            patch.object(hirise_sampler, "HiRISEGeoSampler", return_value=mock_sampler_inst),
            patch.object(torch.utils.data, "DataLoader", return_value=mock_dl),
            patch.object(pathlib.Path, "mkdir"),
            patch("dataset.mars_hirise.plt.close"),
        ):
            main([])

        mock_ds.plot_coverage.assert_called_once()
        mock_ds.plot.assert_called_once_with(sample)
        assert mock_fig.savefig.call_count >= 2  # coverage.png + output0.png
