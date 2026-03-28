"""Tests for src/preprocessing.py.

Covers jp2_to_cog, convert_all, and geographic_split using synthetic data only.
No real HiRISE files are required.
"""

import pathlib
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from pyproj import CRS
from rasterio.transform import from_bounds
from shapely.geometry import box

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from preprocessing import _is_corrupt_jp2_error, convert_all, geographic_split, jp2_to_cog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_jp2(tmp_path: pathlib.Path) -> pathlib.Path:
    """A tiny valid GeoTIFF saved with a .JP2 extension.

    rasterio opens files by content, not extension, so this acts as a
    stand-in for a real JP2 in conversion tests.
    """
    p = tmp_path / "PSP_001430_1780_RED.JP2"
    transform = from_bounds(-130.0, 18.0, -129.9, 18.1, 64, 64)
    mars_crs = CRS.from_proj4("+proj=longlat +a=3396190 +b=3376200 +no_defs")
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": 64,
        "height": 64,
        "count": 1,
        "crs": rasterio.crs.CRS.from_user_input(mars_crs),
        "transform": transform,
    }
    with rasterio.open(p, "w", **profile) as dst:
        dst.write(np.random.rand(64, 64).astype(np.float32), 1)
    return p


@pytest.fixture
def corrupted_jp2(tmp_path: pathlib.Path) -> pathlib.Path:
    """A file with a .JP2 extension containing garbage bytes."""
    p = tmp_path / "PSP_002000_1780_RED.JP2"
    p.write_bytes(b"THIS IS NOT A VALID JP2 FILE\x00\x01\x02")
    return p


@pytest.fixture
def split_index(tmp_path: pathlib.Path) -> gpd.GeoDataFrame:
    """A synthetic GeoDataFrame with 100 observations spread across longitudes."""
    np.random.seed(0)
    lons = np.linspace(-180.0, 180.0, 100)
    lats = np.zeros(100)
    geometries = [box(lon - 0.5, lat - 0.5, lon + 0.5, lat + 0.5) for lon, lat in zip(lons, lats)]
    mars_crs = CRS.from_proj4("+proj=longlat +a=3396190 +b=3376200 +no_defs")
    t_starts = pd.to_datetime(["2006-01-01"] * 100, utc=True)
    t_stops = pd.to_datetime(["2006-01-02"] * 100, utc=True)
    interval_index = pd.IntervalIndex.from_arrays(t_starts, t_stops, closed="both")
    return gpd.GeoDataFrame(
        {"obs_id": [f"obs_{i}" for i in range(100)]},
        geometry=geometries,
        crs=mars_crs,
        index=interval_index,
    )


# ---------------------------------------------------------------------------
# _is_corrupt_jp2_error
# ---------------------------------------------------------------------------


class TestIsCorruptJp2Error:
    def test_opj_decode(self):
        assert _is_corrupt_jp2_error(Exception("opj_decode() failed"))

    def test_ireadblock(self):
        assert _is_corrupt_jp2_error(Exception("IReadBlock failed at X offset 0"))

    def test_read_failed(self):
        assert _is_corrupt_jp2_error(Exception("Read failed. See previous exception"))

    def test_tile_part_length(self):
        assert _is_corrupt_jp2_error(Exception("Tile part length size inconsistent"))

    def test_case_insensitive(self):
        assert _is_corrupt_jp2_error(Exception("OPJ_DECODE() FAILED"))

    def test_unrecognised_format(self):
        assert _is_corrupt_jp2_error(
            Exception("foo.JP2: not recognized as being in a supported file format.")
        )

    def test_disk_full_not_corrupt(self):
        assert not _is_corrupt_jp2_error(OSError("No space left on device"))

    def test_permission_error_not_corrupt(self):
        assert not _is_corrupt_jp2_error(PermissionError("Permission denied"))

    def test_generic_error_not_corrupt(self):
        assert not _is_corrupt_jp2_error(RuntimeError("something else"))


# ---------------------------------------------------------------------------
# jp2_to_cog
# ---------------------------------------------------------------------------


class TestJp2ToCog:
    def test_success_creates_cog(self, valid_jp2):
        result = jp2_to_cog(valid_jp2)
        assert result is not None
        cog = valid_jp2.with_suffix(".tif")
        assert cog.exists()

    def test_success_return_value_is_cog_path(self, valid_jp2):
        result = jp2_to_cog(valid_jp2)
        assert result == valid_jp2.with_suffix(".tif")

    def test_success_removes_tmp_file(self, valid_jp2):
        jp2_to_cog(valid_jp2)
        assert not valid_jp2.with_suffix(".tmp.tif").exists()

    def test_source_jp2_preserved_on_success(self, valid_jp2):
        jp2_to_cog(valid_jp2)
        assert valid_jp2.exists()

    def test_cog_is_readable_geotiff(self, valid_jp2):
        jp2_to_cog(valid_jp2)
        cog = valid_jp2.with_suffix(".tif")
        with rasterio.open(cog) as src:
            assert src.driver == "GTiff"
            assert src.count == 1

    def test_skip_existing_cog(self, valid_jp2):
        jp2_to_cog(valid_jp2)
        cog = valid_jp2.with_suffix(".tif")
        mtime_before = cog.stat().st_mtime
        time.sleep(0.05)
        jp2_to_cog(valid_jp2, overwrite=False)
        assert cog.stat().st_mtime == pytest.approx(mtime_before, abs=0.01)

    def test_overwrite_flag_reconverts(self, valid_jp2):
        jp2_to_cog(valid_jp2)
        cog = valid_jp2.with_suffix(".tif")
        mtime_before = cog.stat().st_mtime
        time.sleep(0.05)
        jp2_to_cog(valid_jp2, overwrite=True)
        assert cog.stat().st_mtime >= mtime_before

    def test_corrupted_jp2_returns_none(self, corrupted_jp2):
        result = jp2_to_cog(corrupted_jp2)
        assert result is None

    def test_corrupted_jp2_deletes_source(self, corrupted_jp2):
        jp2_to_cog(corrupted_jp2)
        assert not corrupted_jp2.exists()

    def test_corrupted_jp2_no_cog_left(self, corrupted_jp2):
        jp2_to_cog(corrupted_jp2)
        assert not corrupted_jp2.with_suffix(".tif").exists()

    def test_corrupted_jp2_no_tmp_left(self, corrupted_jp2):
        jp2_to_cog(corrupted_jp2)
        assert not corrupted_jp2.with_suffix(".tmp.tif").exists()


# ---------------------------------------------------------------------------
# convert_all
# ---------------------------------------------------------------------------


class TestConvertAll:
    def test_counts_correct(self, valid_jp2, corrupted_jp2):
        root = valid_jp2.parent
        counts = convert_all(root, workers=1)
        assert counts["converted"] + counts["failed"] == 2

    def test_corrupted_files_deleted(self, valid_jp2, corrupted_jp2):
        convert_all(valid_jp2.parent, workers=1)
        assert not corrupted_jp2.exists()

    def test_valid_file_converted(self, valid_jp2, corrupted_jp2):
        convert_all(valid_jp2.parent, workers=1)
        assert valid_jp2.with_suffix(".tif").exists()

    def test_skipped_when_cog_exists(self, valid_jp2, tmp_path):
        # Pre-create the COG sidecar.
        cog = valid_jp2.with_suffix(".tif")
        cog.write_bytes(b"placeholder")
        counts = convert_all(tmp_path, workers=1)
        # The pre-existing sidecar means it should be skipped.
        assert counts["failed"] == 0

    def test_empty_directory_returns_zero_counts(self, tmp_path):
        counts = convert_all(tmp_path, workers=1)
        assert counts == {"converted": 0, "skipped": 0, "failed": 0}


# ---------------------------------------------------------------------------
# geographic_split
# ---------------------------------------------------------------------------


class TestGeographicSplit:
    def test_sizes_sum_to_total(self, split_index):
        train, test = geographic_split(split_index)
        assert len(train) + len(test) == len(split_index)

    def test_no_overlap(self, split_index):
        train, test = geographic_split(split_index)
        train_ids = set(train["obs_id"])
        test_ids = set(test["obs_id"])
        assert train_ids.isdisjoint(test_ids)

    def test_test_fraction_approximately_correct(self, split_index):
        train, test = geographic_split(split_index, test_fraction=0.2)
        actual = len(test) / len(split_index)
        assert abs(actual - 0.2) < 0.15

    def test_reproducible(self, split_index):
        train1, test1 = geographic_split(split_index, seed=42)
        train2, test2 = geographic_split(split_index, seed=42)
        assert list(train1["obs_id"]) == list(train2["obs_id"])
        assert list(test1["obs_id"]) == list(test2["obs_id"])

    def test_different_seeds_differ(self, split_index):
        _, test1 = geographic_split(split_index, seed=1)
        _, test2 = geographic_split(split_index, seed=999)
        # With 100 observations it is overwhelmingly likely the splits differ.
        assert set(test1["obs_id"]) != set(test2["obs_id"])

    def test_longitude_axis(self, split_index):
        train, test = geographic_split(split_index, split_axis="longitude", seed=0)
        assert len(train) > 0 and len(test) > 0

    def test_latitude_axis(self, split_index):
        # Rebuild index with lat variation so a latitude split is meaningful.
        geoms = [box(-1.0, lat - 0.5, 1.0, lat + 0.5) for lat in np.linspace(-60, 60, 100)]
        mars_crs = CRS.from_proj4("+proj=longlat +a=3396190 +b=3376200 +no_defs")
        t_starts = pd.to_datetime(["2006-01-01"] * 100, utc=True)
        t_stops = pd.to_datetime(["2006-01-02"] * 100, utc=True)
        idx = gpd.GeoDataFrame(
            {"obs_id": [f"obs_{i}" for i in range(100)]},
            geometry=geoms,
            crs=mars_crs,
            index=pd.IntervalIndex.from_arrays(t_starts, t_stops, closed="both"),
        )
        train, test = geographic_split(idx, split_axis="latitude", seed=0)
        assert len(train) + len(test) == 100
