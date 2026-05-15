"""Tests for src/dataset/preprocessing.py.

Covers jp2_to_cog, convert_all, _available_memory_bytes,
and _safe_worker_count using synthetic data only.
No real HiRISE files are required.
"""

import concurrent.futures as _cf
import logging
import pathlib
import signal
import sys
import time
import unittest.mock as mock

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

from dataset.preprocessing.cog_conversion import (
    _available_memory_bytes,
    _is_corrupt_jp2_error,
    _safe_worker_count,
    _worker_init,
    _iter_img_files,
    convert_all,
    filter_maker,
    img_to_cog,
    jp2_to_cog,
)


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
def ungeoreferenced_jp2(tmp_path: pathlib.Path) -> pathlib.Path:
    """A tiny GeoTIFF with no CRS or geotransform (identity matrix), saved as .JP2."""
    p = tmp_path / "PSP_001430_1780_RED_nogeo.JP2"
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": 64,
        "height": 64,
        "count": 1,
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
# _available_memory_bytes
# ---------------------------------------------------------------------------

_PROC_MEMINFO = (
    "MemTotal:       32768000 kB\n"
    "MemFree:         4096000 kB\n"
    "MemAvailable:   16384000 kB\n"  # 16 GiB available
    "Buffers:         1024000 kB\n"
)


class TestAvailableMemoryBytes:
    def test_reads_proc_meminfo(self):
        with mock.patch("builtins.open", mock.mock_open(read_data=_PROC_MEMINFO)):
            result = _available_memory_bytes()
        assert result == 16384000 * 1024  # 16 GiB in bytes

    def test_returns_positive_integer(self):
        # Against the real system — just sanity-check the type and sign.
        result = _available_memory_bytes()
        assert isinstance(result, int)
        assert result > 0

    def test_falls_back_to_sysconf_on_oserror(self):
        page_size = 4096
        phys_pages = 1024 * 1024  # 4 GiB

        def fake_sysconf(name):
            return page_size if "PAGE_SIZE" in str(name) else phys_pages

        with mock.patch("builtins.open", side_effect=OSError("no proc")):
            with mock.patch("os.sysconf", side_effect=fake_sysconf):
                result = _available_memory_bytes()
        assert result == phys_pages * page_size

    def test_falls_back_to_8gib_when_both_fail(self):
        with mock.patch("builtins.open", side_effect=OSError):
            with mock.patch("os.sysconf", side_effect=ValueError):
                result = _available_memory_bytes()
        assert result == 8 * 1024 ** 3

    def test_missing_memavailable_line_falls_back(self):
        # /proc/meminfo exists but has no MemAvailable line → open succeeds
        # but the loop never matches → function falls through to sysconf.
        minimal = "MemTotal: 32768000 kB\nMemFree: 4096000 kB\n"
        with mock.patch("builtins.open", mock.mock_open(read_data=minimal)):
            with mock.patch("os.sysconf", side_effect=ValueError):
                result = _available_memory_bytes()
        assert result == 8 * 1024 ** 3


# ---------------------------------------------------------------------------
# _safe_worker_count
# ---------------------------------------------------------------------------


def _fake_jp2s(tmp_path: pathlib.Path, sizes_bytes: list[int]) -> list[pathlib.Path]:
    """Create zero-byte placeholder paths whose stat().st_size is mocked."""
    files = []
    for i, _ in enumerate(sizes_bytes):
        p = tmp_path / f"fake_{i}.JP2"
        p.touch()
        files.append(p)
    return files


def _patch_sizes(files: list[pathlib.Path], sizes_bytes: list[int]):
    """Return a context manager that patches stat() on each file with a fake size."""
    size_map = {str(p): s for p, s in zip(files, sizes_bytes)}
    real_stat = pathlib.Path.stat

    def fake_stat(self, **kwargs):
        key = str(self)
        if key in size_map:
            result = real_stat(self, **kwargs)
            result = mock.MagicMock(wraps=result)
            result.st_size = size_map[key]
            return result
        return real_stat(self, **kwargs)

    return mock.patch.object(pathlib.Path, "stat", fake_stat)


class TestSafeWorkerCount:
    def test_empty_list_returns_requested(self):
        assert _safe_worker_count([], 8) == 8

    def test_returns_requested_when_memory_is_ample(self, tmp_path):
        # 4 × 100 MiB files; decompressed estimate = 500 MiB each.
        # 64 GiB available → all 4 workers fit easily.
        mb100 = 100 * 1024 ** 2
        files = _fake_jp2s(tmp_path, [mb100] * 4)
        with _patch_sizes(files, [mb100] * 4):
            with mock.patch("dataset.preprocessing.cog_conversion._available_memory_bytes", return_value=64 * 1024 ** 3):
                result = _safe_worker_count(files, 4)
        assert result == 4

    def test_caps_workers_when_memory_is_tight(self, tmp_path):
        # 4 × 1 GiB files → 5 GiB decompressed each.
        # Only 6 GiB usable → safe = floor(6 / 5) = 1 worker.
        gib1 = 1024 ** 3
        files = _fake_jp2s(tmp_path, [gib1] * 4)
        usable = 6 * gib1
        available = int(usable / 0.75)
        with _patch_sizes(files, [gib1] * 4):
            with mock.patch("dataset.preprocessing.cog_conversion._available_memory_bytes", return_value=available):
                result = _safe_worker_count(files, 4)
        assert result == 1

    def test_result_never_below_one(self, tmp_path):
        # Even if a single file exceeds all available RAM, return at least 1.
        gib4 = 4 * 1024 ** 3
        files = _fake_jp2s(tmp_path, [gib4])
        with _patch_sizes(files, [gib4]):
            with mock.patch("dataset.preprocessing.cog_conversion._available_memory_bytes", return_value=1024 ** 3):
                result = _safe_worker_count(files, 8)
        assert result >= 1

    def test_result_never_exceeds_requested(self, tmp_path):
        mb10 = 10 * 1024 ** 2
        files = _fake_jp2s(tmp_path, [mb10] * 100)
        with _patch_sizes(files, [mb10] * 100):
            with mock.patch("dataset.preprocessing.cog_conversion._available_memory_bytes", return_value=512 * 1024 ** 3):
                result = _safe_worker_count(files, 3)
        assert result <= 3

    def test_samples_largest_files_first(self, tmp_path):
        # 20 small (10 MiB) + 4 large (1 GiB) files.
        # Sorted descending, sample = top-8 = [4 × 1 GiB, 4 × 10 MiB].
        # avg = (4×1 GiB + 4×10 MiB) / 8 ≈ 517 MiB → per-worker = 2.52 GiB.
        # With 10 GiB usable → safe = floor(10 / 2.52) = 3.
        # If sorted ascending (worst case: all small) avg ≈ 10 MiB →
        # per-worker = 50 MiB → safe = floor(10 GiB / 50 MiB) = 204.
        # The large-first sort must produce a strictly smaller cap.
        mb10 = 10 * 1024 ** 2
        gib1 = 1024 ** 3
        (tmp_path / "s").mkdir()
        (tmp_path / "l").mkdir()
        small = _fake_jp2s(tmp_path / "s", [mb10] * 20)
        large = _fake_jp2s(tmp_path / "l", [gib1] * 4)
        all_files = small + large
        all_sizes = [mb10] * 20 + [gib1] * 4
        usable = 10 * gib1
        available = int(usable / 0.75)
        with _patch_sizes(all_files, all_sizes):
            with mock.patch("dataset.preprocessing.cog_conversion._available_memory_bytes", return_value=available):
                result_largest_first = _safe_worker_count(all_files, 8)

        # If we sampled only the 20 small files the cap would be 200+.
        # Sampling the largest first must cap well below the request of 8.
        assert result_largest_first < 8
        # Specifically the 4 large files dominate the 8-file sample enough
        # that we get ≤ 4 workers (not the 200+ we'd get from only small files).
        assert result_largest_first <= 4

    def test_logs_warning_when_capped(self, tmp_path, caplog):
        import logging
        gib1 = 1024 ** 3
        files = _fake_jp2s(tmp_path, [gib1] * 2)
        available = int((1 * gib1) / 0.75)  # forces cap to 1 worker
        with _patch_sizes(files, [gib1] * 2):
            with mock.patch("dataset.preprocessing.cog_conversion._available_memory_bytes", return_value=available):
                with caplog.at_level(logging.WARNING, logger="dataset.preprocessing"):
                    _safe_worker_count(files, 4)
        assert any("Capping workers" in r.message for r in caplog.records)

    def test_no_warning_when_not_capped(self, tmp_path, caplog):
        import logging
        mb10 = 10 * 1024 ** 2
        files = _fake_jp2s(tmp_path, [mb10])
        with _patch_sizes(files, [mb10]):
            with mock.patch("dataset.preprocessing.cog_conversion._available_memory_bytes", return_value=512 * 1024 ** 3):
                with caplog.at_level(logging.WARNING, logger="dataset.preprocessing"):
                    _safe_worker_count(files, 2)
        assert not any("Capping workers" in r.message for r in caplog.records)


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

    def test_ungeoreferenced_jp2_no_warning(self, ungeoreferenced_jp2):
        """jp2_to_cog must not surface NotGeoreferencedWarning to the caller."""
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            jp2_to_cog(ungeoreferenced_jp2)
        geo_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
               and ("geotransform" in str(w.message).lower()
                    or "identity matrix" in str(w.message).lower())
        ]
        assert geo_warnings == [], f"Unexpected warnings: {geo_warnings}"

    def test_ungeoreferenced_jp2_still_produces_cog(self, ungeoreferenced_jp2):
        """Conversion succeeds even when the source has no geotransform."""
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            result = jp2_to_cog(ungeoreferenced_jp2)
        assert result is not None
        assert result.exists()


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

    def test_skip_jp2_only_processes_img(self, valid_jp2, dtm_img):
        # skip_jp2=True → JP2 list short-circuited; only the .IMG converts
        counts = convert_all(valid_jp2.parent, workers=1, skip_jp2=True)
        assert counts["converted"] >= 1
        assert dtm_img.with_suffix(".tif").exists()
        assert not valid_jp2.with_suffix(".tif").exists()

    def test_skip_dtm_only_processes_jp2(self, valid_jp2, dtm_img):
        counts = convert_all(valid_jp2.parent, workers=1, skip_dtm=True)
        assert counts["converted"] >= 1
        assert valid_jp2.with_suffix(".tif").exists()
        assert not dtm_img.with_suffix(".tif").exists()


# ---------------------------------------------------------------------------
# img_to_cog
# ---------------------------------------------------------------------------


@pytest.fixture
def dtm_img(tmp_path: pathlib.Path) -> pathlib.Path:
    """A tiny float32 GeoTIFF saved with a DTM-style .IMG name."""
    p = tmp_path / "DTEEC_001234_1780_005678_1780_A01.IMG"
    transform = from_bounds(-130.0, 18.0, -129.9, 18.1, 32, 32)
    mars_crs = CRS.from_proj4("+proj=longlat +a=3396190 +b=3376200 +no_defs")
    with rasterio.open(
        p, "w", driver="GTiff", dtype="float32", width=32, height=32,
        count=1, crs=rasterio.crs.CRS.from_user_input(mars_crs),
        transform=transform, nodata=-3.4028226550889045e+38,
    ) as dst:
        dst.write(np.full((32, 32), 1500.0, dtype=np.float32), 1)
    return p


class TestImgToCog:
    def test_success_creates_cog(self, dtm_img):
        out = img_to_cog(dtm_img)
        assert out is not None
        assert out.exists()
        assert out.suffix == ".tif"

    def test_cog_is_readable_geotiff(self, dtm_img):
        out = img_to_cog(dtm_img)
        with rasterio.open(out) as src:
            assert src.count == 1
            assert src.dtypes[0] == "float32"

    def test_skip_existing_cog(self, dtm_img):
        cog = dtm_img.with_suffix(".tif")
        cog.write_bytes(b"placeholder")
        out = img_to_cog(dtm_img, overwrite=False)
        assert out == cog
        # untouched placeholder
        assert cog.read_bytes() == b"placeholder"

    def test_overwrite_reconverts(self, dtm_img):
        cog = dtm_img.with_suffix(".tif")
        cog.write_bytes(b"placeholder")
        out = img_to_cog(dtm_img, overwrite=True)
        assert out is not None
        with rasterio.open(out) as src:
            assert src.count == 1

    def test_corrupt_img_returns_none(self, tmp_path):
        bad = tmp_path / "DTEEC_bad.IMG"
        bad.write_bytes(b"not a raster at all")
        out = img_to_cog(bad)
        assert out is None
        assert not bad.with_suffix(".tif").exists()

    def test_img_without_nodata_uses_dtm_sentinel(self, tmp_path):
        """src.nodata is None → falls back to the HiRISE float32 sentinel."""
        p = tmp_path / "DTEEC_nonodata.IMG"
        transform = from_bounds(-130.0, 18.0, -129.9, 18.1, 16, 16)
        mars_crs = CRS.from_proj4("+proj=longlat +a=3396190 +b=3376200 +no_defs")
        with rasterio.open(
            p, "w", driver="GTiff", dtype="float32", width=16, height=16,
            count=1, crs=rasterio.crs.CRS.from_user_input(mars_crs),
            transform=transform,  # no nodata=
        ) as dst:
            dst.write(np.full((16, 16), 1500.0, dtype=np.float32), 1)
        out = img_to_cog(p)
        assert out is not None
        with rasterio.open(out) as src:
            assert src.nodata == pytest.approx(-3.4028226550889045e+38)

    def test_generic_exception_returns_none(self, dtm_img):
        """A non-RasterioIOError during conversion is caught → None."""
        with mock.patch(
            "dataset.preprocessing.cog_conversion.rasterio.open",
            side_effect=RuntimeError("boom"),
        ):
            out = img_to_cog(dtm_img)
        assert out is None


class TestIterImgFiles:
    def test_only_dte_prefixed_files_yielded(self, tmp_path):
        (tmp_path / "DTEEC_001.IMG").write_bytes(b"x")
        (tmp_path / "RDR_other.IMG").write_bytes(b"x")
        (tmp_path / "dteec_lower.img").write_bytes(b"x")
        found = {p.name for p in _iter_img_files(tmp_path)}
        assert "DTEEC_001.IMG" in found
        assert "dteec_lower.img" in found
        assert "RDR_other.IMG" not in found


# ---------------------------------------------------------------------------
# filter_maker
# ---------------------------------------------------------------------------


class TestFilterMaker:
    def test_returns_callable(self):
        f = filter_maker("WARNING")
        assert callable(f)

    def test_passes_record_at_level(self):
        f = filter_maker("WARNING")
        record = logging.LogRecord("test", logging.WARNING, "", 0, "msg", (), None)
        assert f(record) is True

    def test_passes_record_below_level(self):
        f = filter_maker("WARNING")
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        assert f(record) is True

    def test_blocks_record_above_level(self):
        f = filter_maker("WARNING")
        record = logging.LogRecord("test", logging.ERROR, "", 0, "msg", (), None)
        assert f(record) is False

    def test_info_level_blocks_warning(self):
        f = filter_maker("INFO")
        record = logging.LogRecord("test", logging.WARNING, "", 0, "msg", (), None)
        assert f(record) is False

    def test_info_level_passes_debug(self):
        f = filter_maker("INFO")
        record = logging.LogRecord("test", logging.DEBUG, "", 0, "msg", (), None)
        assert f(record) is True


# ---------------------------------------------------------------------------
# _worker_init
# ---------------------------------------------------------------------------


class TestWorkerInit:
    def test_restores_sigint_to_default(self):
        with mock.patch("signal.signal") as mock_signal:
            with mock.patch("logging.basicConfig"):
                _worker_init()
        mock_signal.assert_called_once_with(signal.SIGINT, signal.SIG_DFL)

    def test_configures_logging_at_info(self):
        with mock.patch("signal.signal"):
            with mock.patch("logging.basicConfig") as mock_basic:
                _worker_init()
        mock_basic.assert_called_once()
        assert mock_basic.call_args[1]["level"] == logging.INFO


# ---------------------------------------------------------------------------
# jp2_to_cog — additional error-path tests
# ---------------------------------------------------------------------------


class TestJp2ToCogErrorPaths:
    def test_non_corrupt_rasterio_error_preserves_source(self, tmp_path):
        """A non-corrupt RasterioIOError (e.g. permission denied) must not delete source."""
        jp2 = tmp_path / "PSP_001430_1780_RED.JP2"
        jp2.touch()
        err = rasterio.errors.RasterioIOError("Permission denied: cannot open file")
        with mock.patch("rasterio.open", side_effect=err):
            result = jp2_to_cog(jp2)
        assert result is None
        assert jp2.exists()

    def test_non_corrupt_rasterio_error_no_cog_left(self, tmp_path):
        """Non-corrupt RasterioIOError must not leave a partial COG."""
        jp2 = tmp_path / "PSP_001430_1780_RED.JP2"
        jp2.touch()
        err = rasterio.errors.RasterioIOError("Permission denied: cannot open file")
        with mock.patch("rasterio.open", side_effect=err):
            jp2_to_cog(jp2)
        assert not jp2.with_suffix(".tif").exists()

    def test_generic_exception_returns_none(self, tmp_path):
        """Non-rasterio exceptions return None without deleting the source JP2."""
        jp2 = tmp_path / "PSP_001430_1780_RED.JP2"
        jp2.touch()
        with mock.patch("rasterio.open", side_effect=MemoryError("OOM")):
            result = jp2_to_cog(jp2)
        assert result is None
        assert jp2.exists()

    def test_generic_exception_no_cog_left(self, tmp_path):
        """Generic exceptions must not leave a partial COG."""
        jp2 = tmp_path / "PSP_001430_1780_RED.JP2"
        jp2.touch()
        with mock.patch("rasterio.open", side_effect=RuntimeError("unexpected")):
            jp2_to_cog(jp2)
        assert not jp2.with_suffix(".tif").exists()

    def test_generic_exception_no_tmp_left(self, tmp_path):
        """Generic exceptions must clean up the .tmp.tif scratch file."""
        jp2 = tmp_path / "PSP_001430_1780_RED.JP2"
        jp2.touch()
        with mock.patch("rasterio.open", side_effect=RuntimeError("unexpected")):
            jp2_to_cog(jp2)
        assert not jp2.with_suffix(".tmp.tif").exists()


# ---------------------------------------------------------------------------
# convert_all — additional path tests
# ---------------------------------------------------------------------------


class TestConvertAllExtra:
    def test_skipped_count_when_cog_older_than_jp2(self, tmp_path):
        """jp2_to_cog returns an existing COG whose mtime < JP2 mtime → skipped."""
        # Create COG first so it has an older timestamp.
        cog = tmp_path / "PSP_001430_1780_RED.tif"
        cog.write_bytes(b"placeholder")
        time.sleep(0.06)
        # JP2 written after → newer mtime; jp2_to_cog will return cog immediately.
        jp2 = tmp_path / "PSP_001430_1780_RED.JP2"
        jp2.write_bytes(b"\x00" * 1024)  # non-zero size avoids ZeroDivisionError in _safe_worker_count
        counts = convert_all(tmp_path, workers=1)
        assert counts["skipped"] == 1
        assert counts["failed"] == 0
        assert counts["converted"] == 0

    def test_overwrite_flag_reconverts_existing_cog(self, valid_jp2):
        """overwrite=True must re-convert even when a fresh COG already exists."""
        # First pass: create COG.
        counts1 = convert_all(valid_jp2.parent, workers=1, overwrite=False)
        assert counts1["converted"] == 1
        # Second pass with overwrite: should convert again, not skip.
        counts2 = convert_all(valid_jp2.parent, workers=1, overwrite=True)
        assert counts2["converted"] == 1
        assert counts2["skipped"] == 0

    def test_worker_exception_counted_as_failed(self, tmp_path):
        """Exceptions raised by worker futures are counted as failed."""
        jp2 = tmp_path / "PSP_001430_1780_RED.JP2"
        jp2.write_bytes(b"\x00" * 1024)

        # Build a real Future object with an exception pre-set.
        failing_future = _cf.Future()
        failing_future.set_exception(RuntimeError("worker crash"))

        mock_pool = mock.MagicMock()
        mock_pool.submit.return_value = failing_future
        mock_pool.shutdown = mock.Mock()

        with mock.patch("concurrent.futures.ProcessPoolExecutor", return_value=mock_pool):
            counts = convert_all(tmp_path, workers=1)

        assert counts["failed"] == 1

    def test_keyboard_interrupt_raises_system_exit_130(self, tmp_path):
        """KeyboardInterrupt during iteration triggers a SystemExit(130)."""
        jp2 = tmp_path / "PSP_001430_1780_RED.JP2"
        jp2.write_bytes(b"\x00" * 1024)

        mock_pool = mock.MagicMock()
        mock_pool.submit.return_value = mock.MagicMock()
        mock_pool.shutdown = mock.Mock()

        with mock.patch("concurrent.futures.ProcessPoolExecutor", return_value=mock_pool):
            with mock.patch(
                    "concurrent.futures.as_completed", side_effect=KeyboardInterrupt
            ):
                with pytest.raises(SystemExit) as exc_info:
                    convert_all(tmp_path, workers=1)

        assert exc_info.value.code == 130

    def test_keyboard_interrupt_calls_shutdown_with_cancel(self, tmp_path):
        """On KeyboardInterrupt the pool is shut down with cancel_futures=True."""
        jp2 = tmp_path / "PSP_001430_1780_RED.JP2"
        jp2.write_bytes(b"\x00" * 1024)

        mock_pool = mock.MagicMock()
        mock_pool.submit.return_value = mock.MagicMock()

        with mock.patch("concurrent.futures.ProcessPoolExecutor", return_value=mock_pool):
            with mock.patch(
                    "concurrent.futures.as_completed", side_effect=KeyboardInterrupt
            ):
                with pytest.raises(SystemExit):
                    convert_all(tmp_path, workers=1)

        mock_pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# CLI __main__ block
# ---------------------------------------------------------------------------

_PREPROCESSING_SCRIPT = pathlib.Path(__file__).resolve().parents[3] / "src" / "dataset" / "preprocessing" / "cog_conversion.py"


class TestCLI:
    """Exercise the ``if __name__ == "__main__"`` block via runpy in-process.

    runpy.run_path(..., run_name="__main__") executes the guarded block in the
    current process so coverage is captured without subprocess overhead.
    """

    def _run_main(self, argv: list[str]) -> None:
        import runpy

        with mock.patch.object(sys, "argv", ["dataset/preprocessing.py"] + argv):
            runpy.run_path(str(_PREPROCESSING_SCRIPT), run_name="__main__")

    def test_runs_on_empty_root(self, tmp_path):
        """CLI completes successfully on a directory with no JP2 files."""
        self._run_main(["--root", str(tmp_path), "--workers", "1"])

    def test_overwrite_flag_accepted(self, tmp_path):
        """--overwrite flag is accepted without errors."""
        self._run_main(["--root", str(tmp_path), "--workers", "1", "--overwrite"])

    def test_falls_back_to_basicconfig_when_no_logger_json(self, tmp_path, monkeypatch):
        """CLI uses basicConfig when logger_config.json is absent."""
        # Redirect the config path lookup to a nonexistent file so the else
        # branch (basicConfig) is taken.
        import runpy

        nonexistent = tmp_path / "no_config.json"
        with mock.patch("pathlib.Path.exists", return_value=False):
            with mock.patch("logging.basicConfig") as mock_basic:
                with mock.patch.object(sys, "argv", ["dataset/preprocessing.py", "--root", str(tmp_path)]):
                    runpy.run_path(str(_PREPROCESSING_SCRIPT), run_name="__main__")
        mock_basic.assert_called()
