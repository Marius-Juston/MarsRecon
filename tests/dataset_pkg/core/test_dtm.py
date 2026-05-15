"""Comprehensive unit tests for :mod:`dataset.core.dtm`.

Covers the heavier surface area not exercised by ``test_dtm_helpers.py``:
``__getitem__``, ``_load_dtm_tile``, ``_load_ortho_tile``, ``_get_ortho_overlap``
success path, ``_build_spatial_index``, ``_build_download_tasks``, ``plot``,
and ``plot3d``. All synthetic — no real PDS data required.
"""

from __future__ import annotations

import json
import pathlib
import sys
import textwrap
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.crs import CRS as RasterioCRS
from shapely.geometry import box

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from dataset.core.dtm import MarsHiRISEDTM, _DTM_NODATA

_MARS_RCRS = RasterioCRS.from_proj4(
    "+proj=longlat +a=3396190 +b=3376200 +no_defs"
)

_T0 = pd.Timestamp("2010-01-01T00:00:00", tz="UTC")
_T1 = pd.Timestamp("2010-01-01T00:01:00", tz="UTC")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_dtm_tif(path: pathlib.Path, *, fill: float = 1500.0,
                   bounds=(-131.0, 18.0, -130.0, 19.0), size: int = 32) -> None:
    transform = rasterio.transform.from_bounds(*bounds, size, size)
    data = np.full((1, size, size), fill, dtype=np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", count=1, dtype="float32",
        width=size, height=size, crs=_MARS_RCRS, transform=transform,
        nodata=_DTM_NODATA,
    ) as dst:
        dst.write(data)


def _write_ortho_tif(path: pathlib.Path, *, bands: int = 1, fill: int = 800,
                     bounds=(-131.0, 18.0, -130.0, 19.0), size: int = 32) -> None:
    transform = rasterio.transform.from_bounds(*bounds, size, size)
    data = np.full((bands, size, size), fill, dtype=np.uint16)
    with rasterio.open(
        path, "w", driver="GTiff", count=bands, dtype="uint16",
        width=size, height=size, crs=_MARS_RCRS, transform=transform,
    ) as dst:
        dst.write(data)


def _write_lbl(path: pathlib.Path, *, scaling: float = 1e-4, offset: float = 0.05) -> None:
    path.write_text(textwrap.dedent(f"""\
        PDS_VERSION_ID = PDS3
        SCALING_FACTOR = {scaling}
        OFFSET = {offset}
        SAMPLE_BITS = 16
        BANDS = 3
        END
    """))


@pytest.fixture
def dtm_dataset(tmp_path):
    """MarsHiRISEDTM with _verify suppressed; index populated by individual tests."""
    with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
        ds = MarsHiRISEDTM(root=str(tmp_path), include_ortho=True, ortho_type="RED")
    return ds


@pytest.fixture
def dtm_files(tmp_path):
    """Create a dtm.tif + left/right RED .tif + .LBL files. Return paths dict."""
    paths = {
        "dtm": tmp_path / "DTEEC_test.tif",
        "left_red": tmp_path / "PSP_001_RED_A_01_ORTHO.tif",
        "right_red": tmp_path / "PSP_002_RED_A_01_ORTHO.tif",
        "left_red_lbl": tmp_path / "PSP_001_RED_A_01_ORTHO.LBL",
        "right_red_lbl": tmp_path / "PSP_002_RED_A_01_ORTHO.LBL",
    }
    _write_dtm_tif(paths["dtm"])
    _write_ortho_tif(paths["left_red"], bands=1)
    _write_ortho_tif(paths["right_red"], bands=1)
    _write_lbl(paths["left_red_lbl"])
    _write_lbl(paths["right_red_lbl"])
    return paths


def _make_index_row(dtm_path, left_red, right_red, mars_crs,
                    bounds=(-131.0, 18.0, -130.0, 19.0)) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "pair_key": ["PSP_001__PSP_002"],
            "left_obs_id": ["PSP_001"],
            "right_obs_id": ["PSP_002"],
            "dtm_path": [str(dtm_path)],
            "dtm_product_id": ["DTEEC_test"],
            "left_red_path": [str(left_red)],
            "right_red_path": [str(right_red)],
            "left_irb_path": [None],
            "right_irb_path": [None],
        },
        geometry=[box(*bounds)],
        crs=mars_crs,
        index=pd.IntervalIndex.from_tuples([(_T0, _T1)], closed="both", name="datetime"),
    )


# ---------------------------------------------------------------------------
# _load_dtm_tile
# ---------------------------------------------------------------------------


class TestLoadDtmTile:
    def test_returns_none_when_path_missing(self, dtm_dataset, tmp_path):
        ghost = tmp_path / "ghost.IMG"
        out = dtm_dataset._load_dtm_tile(ghost, slice(-131.0, -130.0), slice(18.0, 19.0))
        assert out is None

    def test_returns_tile_for_valid_geotiff(self, dtm_dataset, tmp_path):
        p = tmp_path / "dtm.tif"
        _write_dtm_tif(p, fill=2000.0)
        out = dtm_dataset._load_dtm_tile(p, slice(-131.0, -130.0), slice(18.0, 19.0))
        assert out is not None
        assert out.dtype == torch.float32
        assert out.shape[0] == 1
        # All-valid fill value is preserved
        assert torch.all(out == 2000.0)

    def test_zero_window_returns_none(self, dtm_dataset, tmp_path):
        p = tmp_path / "dtm.tif"
        _write_dtm_tif(p)
        # Slice entirely outside the file → window has h or w <= 0
        out = dtm_dataset._load_dtm_tile(p, slice(0.0, 0.000001), slice(0.0, 0.000001))
        assert out is None

    def test_nodata_pixels_become_nan(self, dtm_dataset, tmp_path):
        p = tmp_path / "dtm.tif"
        # Half-and-half: nodata sentinel + valid
        transform = rasterio.transform.from_bounds(-131.0, 18.0, -130.0, 19.0, 8, 8)
        data = np.full((1, 8, 8), 1000.0, dtype=np.float32)
        data[0, :4, :] = _DTM_NODATA
        with rasterio.open(
            p, "w", driver="GTiff", count=1, dtype="float32",
            width=8, height=8, crs=_MARS_RCRS, transform=transform,
            nodata=_DTM_NODATA,
        ) as dst:
            dst.write(data)
        out = dtm_dataset._load_dtm_tile(p, slice(-131.0, -130.0), slice(18.0, 19.0))
        assert out is not None
        # Half NaN, half valid
        assert torch.isnan(out).any()
        assert (~torch.isnan(out)).any()

    def test_rasterio_error_returns_none(self, dtm_dataset, tmp_path, caplog):
        bad = tmp_path / "broken.IMG"
        bad.write_bytes(b"not a raster")
        import logging
        with caplog.at_level(logging.WARNING, logger="dataset.core.dtm"):
            out = dtm_dataset._load_dtm_tile(bad, slice(-131.0, -130.0), slice(18.0, 19.0))
        assert out is None


# ---------------------------------------------------------------------------
# _load_ortho_tile
# ---------------------------------------------------------------------------


class TestLoadOrthoTile:
    def test_returns_none_when_path_missing(self, dtm_dataset, tmp_path):
        out = dtm_dataset._load_ortho_tile(
            tmp_path / "ghost.JP2", "RED",
            slice(-131.0, -130.0), slice(18.0, 19.0),
        )
        assert out is None

    def test_red_single_band(self, dtm_dataset, tmp_path):
        p = tmp_path / "ortho.tif"
        _write_ortho_tif(p, bands=1, fill=1000)
        _write_lbl(p.with_suffix(".LBL"))
        out = dtm_dataset._load_ortho_tile(
            p, "RED", slice(-131.0, -130.0), slice(18.0, 19.0)
        )
        assert out is not None
        assert out.shape[0] == 1
        # Calibrated values are clipped to [0, 1]
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_irb_three_bands(self, dtm_dataset, tmp_path):
        p = tmp_path / "ortho_irb.tif"
        _write_ortho_tif(p, bands=3, fill=500)
        _write_lbl(p.with_suffix(".LBL"))
        out = dtm_dataset._load_ortho_tile(
            p, "IRB", slice(-131.0, -130.0), slice(18.0, 19.0)
        )
        assert out is not None
        assert out.shape[0] == 3

    def test_irb_missing_bands_filled_with_zeros(self, dtm_dataset, tmp_path):
        p = tmp_path / "ortho_1band.tif"
        # Source has only 1 band but we request IRB (3 bands)
        _write_ortho_tif(p, bands=1, fill=500)
        _write_lbl(p.with_suffix(".LBL"))
        out = dtm_dataset._load_ortho_tile(
            p, "IRB", slice(-131.0, -130.0), slice(18.0, 19.0)
        )
        assert out is not None
        assert out.shape[0] == 3
        # Bands 2 and 3 are zero (band_idx > src.count → zeros)
        assert torch.all(out[1:] == 0.0)

    def test_zero_window_returns_none(self, dtm_dataset, tmp_path):
        p = tmp_path / "ortho.tif"
        _write_ortho_tif(p, bands=1)
        _write_lbl(p.with_suffix(".LBL"))
        out = dtm_dataset._load_ortho_tile(
            p, "RED", slice(0.0, 1e-9), slice(0.0, 1e-9)
        )
        assert out is None

    def test_zero_pixels_remain_nodata(self, dtm_dataset, tmp_path):
        p = tmp_path / "ortho.tif"
        _write_ortho_tif(p, bands=1, fill=0)  # all-zero → nodata mask
        _write_lbl(p.with_suffix(".LBL"))
        out = dtm_dataset._load_ortho_tile(
            p, "RED", slice(-131.0, -130.0), slice(18.0, 19.0)
        )
        assert out is not None
        assert torch.all(out == 0.0)

    def test_rasterio_error_returns_none(self, dtm_dataset, tmp_path):
        bad = tmp_path / "bad.JP2"
        bad.write_bytes(b"not a jp2")
        out = dtm_dataset._load_ortho_tile(
            bad, "RED", slice(-131.0, -130.0), slice(18.0, 19.0)
        )
        assert out is None


# ---------------------------------------------------------------------------
# _get_ortho_overlap (success path beyond test_dtm_helpers)
# ---------------------------------------------------------------------------


class TestGetOrthoOverlapSuccess:
    def test_full_overlap_returns_one(self, dtm_dataset, tmp_path):
        p = tmp_path / "ortho.tif"
        _write_ortho_tif(p, bands=1, bounds=(-131.0, 18.0, -130.0, 19.0))
        dtm_geom = box(-131.0, 18.0, -130.0, 19.0)
        ratio = dtm_dataset._get_ortho_overlap(dtm_geom, str(p))
        assert ratio == pytest.approx(1.0, rel=1e-3)

    def test_partial_overlap(self, dtm_dataset, tmp_path):
        p = tmp_path / "ortho.tif"
        # ortho footprint covers only east half of dtm geom
        _write_ortho_tif(p, bands=1, bounds=(-130.5, 18.0, -130.0, 19.0))
        dtm_geom = box(-131.0, 18.0, -130.0, 19.0)
        ratio = dtm_dataset._get_ortho_overlap(dtm_geom, str(p))
        assert 0.4 < ratio < 0.6

    def test_zero_area_dtm_with_real_file_returns_zero(self, dtm_dataset, tmp_path):
        """Real ortho file but zero-area DTM geom → 0.0 (dtm.py:546-547)."""
        from shapely.geometry import Point

        p = tmp_path / "ortho.tif"
        _write_ortho_tif(p, bands=1, bounds=(-131.0, 18.0, -130.0, 19.0))
        ratio = dtm_dataset._get_ortho_overlap(Point(-130.5, 18.5), str(p))
        assert ratio == 0.0

    def test_antimeridian_ortho_split_into_two_boxes(self, dtm_dataset, tmp_path):
        """Ortho straddling the antimeridian → unary_union branch (dtm.py:537-542)."""
        p = tmp_path / "ortho_am.tif"
        # Footprint spanning 179°E → -179°E (i.e. fl_norm > fr_norm after wrap).
        _write_ortho_tif(p, bands=1, bounds=(179.0, -1.0, 181.0, 1.0))
        dtm_geom = box(179.5, -1.0, 180.0, 1.0)
        ratio = dtm_dataset._get_ortho_overlap(dtm_geom, str(p))
        assert ratio > 0.0

    def test_no_crs_returns_one(self, dtm_dataset, tmp_path):
        p = tmp_path / "nocrs.tif"
        transform = rasterio.transform.from_bounds(-131.0, 18.0, -130.0, 19.0, 8, 8)
        with rasterio.open(
            p, "w", driver="GTiff", count=1, dtype="uint16",
            width=8, height=8, transform=transform,
        ) as dst:
            dst.write(np.ones((1, 8, 8), dtype=np.uint16) * 100)
        ratio = dtm_dataset._get_ortho_overlap(box(-131, 18, -130, 19), str(p))
        # No CRS → assumes overlap = 1.0
        assert ratio == 1.0


# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------


class TestGetItem:
    def test_no_candidates_raises(self, dtm_dataset, dtm_files, mars_crs):
        dtm_dataset.index = _make_index_row(
            dtm_files["dtm"], dtm_files["left_red"], dtm_files["right_red"], mars_crs
        )
        # Query window far outside the geometry
        with pytest.raises(IndexError, match="No MarsHiRISEDTM stereo pairs"):
            dtm_dataset[slice(50.0, 51.0), slice(0.0, 1.0), slice(None)]

    def test_happy_path_returns_elevation_and_orthos(
        self, dtm_dataset, dtm_files, mars_crs
    ):
        dtm_dataset.index = _make_index_row(
            dtm_files["dtm"], dtm_files["left_red"], dtm_files["right_red"], mars_crs
        )
        sample = dtm_dataset[
            slice(-130.6, -130.4), slice(18.4, 18.6), slice(None)
        ]
        assert "elevation" in sample
        assert "left_red" in sample
        assert "right_red" in sample
        assert "bounds" in sample
        assert "crs" in sample
        assert sample["elevation"].dtype == torch.float32

    def test_no_elevation_tiles_raises(self, dtm_dataset, dtm_files, mars_crs):
        # Index points dtm_path at a non-existent file → no elevation tiles loaded
        gdf = _make_index_row(
            dtm_files["dtm"], dtm_files["left_red"], dtm_files["right_red"], mars_crs
        )
        gdf["dtm_path"] = [str(dtm_files["dtm"].parent / "ghost.IMG")]
        dtm_dataset.index = gdf
        with pytest.raises(IndexError, match="no elevation data could be loaded"):
            dtm_dataset[slice(-130.6, -130.4), slice(18.4, 18.6), slice(None)]

    def test_normalize_elevation_applied(self, tmp_path, dtm_files, mars_crs):
        stats = tmp_path / "stats.json"
        stats.write_text(json.dumps({"mean": 1500.0, "std": 100.0}))
        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(
                root=str(tmp_path),
                include_ortho=False,
                normalize_elevation=True,
                elevation_stats_path=str(stats),
            )
        ds.index = _make_index_row(
            dtm_files["dtm"], dtm_files["left_red"], dtm_files["right_red"], mars_crs
        )
        sample = ds[slice(-130.6, -130.4), slice(18.4, 18.6), slice(None)]
        # fill=1500 → (1500 - 1500)/100 = 0
        assert torch.allclose(sample["elevation"], torch.zeros_like(sample["elevation"]))

    def test_transforms_applied(self, dtm_dataset, dtm_files, mars_crs):
        marker = {"called": False}

        def transform(s):
            marker["called"] = True
            s["marker"] = True
            return s

        dtm_dataset.transforms = transform
        dtm_dataset.index = _make_index_row(
            dtm_files["dtm"], dtm_files["left_red"], dtm_files["right_red"], mars_crs
        )
        sample = dtm_dataset[slice(-130.6, -130.4), slice(18.4, 18.6), slice(None)]
        assert marker["called"]
        assert sample.get("marker") is True

    def test_return_meta_attaches_metadata(self, tmp_path, dtm_files, mars_crs):
        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(
                root=str(tmp_path), include_ortho=True, ortho_type="RED",
                return_meta=True,
            )
        ds.index = _make_index_row(
            dtm_files["dtm"], dtm_files["left_red"], dtm_files["right_red"], mars_crs
        )
        sample = ds[slice(-130.6, -130.4), slice(18.4, 18.6), slice(None)]
        assert "meta" in sample
        assert isinstance(sample["meta"], list)
        assert sample["meta"][0]["dtm_product_id"] == "DTEEC_test"
        assert "left_red_meta" in sample["meta"][0]

    def test_none_ortho_path_column_is_skipped(self, dtm_dataset, dtm_files, mars_crs):
        """An ortho path column that is None (not str) → continue (dtm.py:285-286)."""
        gdf = _make_index_row(
            dtm_files["dtm"], dtm_files["left_red"], dtm_files["right_red"], mars_crs
        )
        gdf["left_red_path"] = [None]  # not a str → skipped
        dtm_dataset.index = gdf
        sample = dtm_dataset[slice(-130.6, -130.4), slice(18.4, 18.6), slice(None)]
        assert "elevation" in sample
        # left_red was skipped; right_red still loads.
        assert "right_red" in sample
        assert "left_red" not in sample

    def test_no_ortho_returns_only_elevation(self, tmp_path, dtm_files, mars_crs):
        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(
                root=str(tmp_path), include_ortho=False,
            )
        ds.index = _make_index_row(
            dtm_files["dtm"], dtm_files["left_red"], dtm_files["right_red"], mars_crs
        )
        sample = ds[slice(-130.6, -130.4), slice(18.4, 18.6), slice(None)]
        assert "elevation" in sample
        assert "left_red" not in sample
        assert "right_red" not in sample


# ---------------------------------------------------------------------------
# plot / plot3d
# ---------------------------------------------------------------------------


class TestPlot:
    def _sample(self, with_ortho: bool = True, all_nan: bool = False):
        elev = torch.full((1, 16, 16), 1500.0)
        if all_nan:
            elev[:] = float("nan")
        s = {"elevation": elev}
        if with_ortho:
            s["left_red"] = torch.rand(1, 16, 16)
            s["right_irb"] = torch.rand(3, 16, 16)
        return s

    def test_basic_plot(self, dtm_dataset):
        fig = dtm_dataset.plot(self._sample())
        assert fig.axes
        plt.close(fig)

    def test_plot_with_suptitle(self, dtm_dataset):
        fig = dtm_dataset.plot(self._sample(with_ortho=False), suptitle="Test")
        assert fig._suptitle.get_text() == "Test"
        plt.close(fig)

    def test_plot_no_titles(self, dtm_dataset):
        fig = dtm_dataset.plot(self._sample(), show_titles=False)
        for ax in fig.axes:
            assert ax.get_title() == ""
        plt.close(fig)

    def test_plot_handles_4d_elevation(self, dtm_dataset):
        s = self._sample()
        s["elevation"] = s["elevation"].unsqueeze(0)  # (1, 1, 16, 16)
        fig = dtm_dataset.plot(s)
        plt.close(fig)

    def test_plot_handles_4d_ortho(self, dtm_dataset):
        """4D ortho tensor → squeezed batch dim (dtm.py:392-393)."""
        s = self._sample()
        s["left_red"] = s["left_red"].unsqueeze(0)    # (1, 1, 16, 16)
        s["right_irb"] = s["right_irb"].unsqueeze(0)  # (1, 3, 16, 16)
        fig = dtm_dataset.plot(s)
        plt.close(fig)

    def test_plot_all_nan_elevation(self, dtm_dataset):
        fig = dtm_dataset.plot(self._sample(all_nan=True))
        plt.close(fig)

    def test_plot_empty_sample_returns_figure(self, dtm_dataset):
        fig = dtm_dataset.plot({})
        plt.close(fig)

    def test_plot3d_basic(self, dtm_dataset):
        fig = dtm_dataset.plot3d(self._sample(with_ortho=False))
        plt.close(fig)

    def test_plot3d_with_suptitle(self, dtm_dataset):
        fig = dtm_dataset.plot3d(self._sample(with_ortho=False), suptitle="3D")
        assert fig._suptitle.get_text() == "3D"
        plt.close(fig)

    def test_plot3d_handles_4d_and_all_nan(self, dtm_dataset):
        s = {"elevation": torch.full((1, 1, 16, 16), float("nan"))}
        fig = dtm_dataset.plot3d(s)
        plt.close(fig)


# ---------------------------------------------------------------------------
# _build_download_tasks
# ---------------------------------------------------------------------------


class TestBuildDownloadTasks:
    def test_emits_tasks_for_missing_files(self, dtm_dataset, tmp_path):
        dtm_dataset._raw_index = pd.DataFrame({
            "FILE_NAME_SPECIFICATION": [
                "MROHR_0001/DATA/DTM/test.IMG",
                "MROHR_0001/DATA/ORTHO/test.JP2",
            ],
        })
        tasks = dtm_dataset._build_download_tasks()
        # IMG → 1 task; JP2 → 2 tasks (JP2 + companion .LBL)
        assert len(tasks) == 3
        assert any(str(t[1]).endswith(".LBL") for t in tasks)

    def test_skips_existing_files(self, dtm_dataset, tmp_path):
        # Pre-create the JP2 to skip its task; LBL is missing so still one task
        existing = tmp_path / "images" / "test.JP2"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_bytes(b"x")
        dtm_dataset._raw_index = pd.DataFrame({
            "FILE_NAME_SPECIFICATION": ["MROHR_0001/DATA/ORTHO/test.JP2"],
        })
        tasks = dtm_dataset._build_download_tasks()
        # Only the .LBL companion download remains
        assert len(tasks) == 1
        assert str(tasks[0][1]).endswith(".LBL")


# ---------------------------------------------------------------------------
# _build_spatial_index — end-to-end with synthetic raw index
# ---------------------------------------------------------------------------


def _raw_index_for_pair(images_dir: pathlib.Path) -> pd.DataFrame:
    """Build a synthetic _raw_index covering one stereo pair (DTM + L/R RED)."""
    return pd.DataFrame({
        "PRODUCT_ID": [
            "DTEEC_001234_1780_005678_1780_V01",
            "PSP_001234_1780_RED_A_01_ORTHO",
            "PSP_005678_1780_RED_A_01_ORTHO",
        ],
        "DATA_TYPE": ["DTM", "ORTHOIMAGE", "ORTHOIMAGE"],
        "LEFT_OBSERVATION_ID": ["PSP_001234_1780"] * 3,
        "RIGHT_OBSERVATION_ID": ["PSP_005678_1780"] * 3,
        "FILE_NAME_SPECIFICATION": [
            "MRO/DTM/DTEEC_001.IMG",
            "MRO/ORTHO/PSP_001234_1780_RED_A_01_ORTHO.JP2",
            "MRO/ORTHO/PSP_005678_1780_RED_A_01_ORTHO.JP2",
        ],
        "MAP_SCALE": [1.0, 0.25, 0.25],
        "RATIONALE_DESC": ["Test target", "", ""],
        "MINIMUM_LONGITUDE": [229.0, 229.0, 229.0],
        "MAXIMUM_LONGITUDE": [230.0, 230.0, 230.0],
        "MINIMUM_LATITUDE": [18.0, 18.0, 18.0],
        "MAXIMUM_LATITUDE": [19.0, 19.0, 19.0],
        "START_TIME": ["2010-01-01T00:00:00"] * 3,
        "STOP_TIME": ["2010-01-01T00:01:00"] * 3,
        "CORNER1_LATITUDE": [18.0] * 3,
        "CORNER1_LONGITUDE": [229.0] * 3,
        "CORNER2_LATITUDE": [18.0] * 3,
        "CORNER2_LONGITUDE": [230.0] * 3,
        "CORNER3_LATITUDE": [19.0] * 3,
        "CORNER3_LONGITUDE": [230.0] * 3,
        "CORNER4_LATITUDE": [19.0] * 3,
        "CORNER4_LONGITUDE": [229.0] * 3,
    })


class TestBuildSpatialIndex:
    def test_no_dtm_rows_raises(self, dtm_dataset):
        df = _raw_index_for_pair(pathlib.Path("/tmp"))
        df = df[df["DATA_TYPE"] != "DTM"]
        dtm_dataset._raw_index = df.reset_index(drop=True)
        from torchgeo.datasets.errors import DatasetNotFoundError
        with pytest.raises(DatasetNotFoundError):
            dtm_dataset._build_spatial_index(force_rebuild=True)

    def test_happy_path_creates_index(self, tmp_path):
        # Materialise the IMG + JP2 + LBL files so footprint extraction has files
        images = tmp_path / "images"
        images.mkdir()
        _write_dtm_tif(images / "DTEEC_001.IMG")
        _write_ortho_tif(images / "PSP_001234_1780_RED_A_01_ORTHO.JP2", bands=1)
        _write_ortho_tif(images / "PSP_005678_1780_RED_A_01_ORTHO.JP2", bands=1)
        _write_lbl(images / "PSP_001234_1780_RED_A_01_ORTHO.LBL")
        _write_lbl(images / "PSP_005678_1780_RED_A_01_ORTHO.LBL")

        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(
                root=str(tmp_path), include_ortho=True, ortho_type="RED",
                reuse_cache=False,
            )
        ds._raw_index = _raw_index_for_pair(images)
        ds._build_spatial_index(force_rebuild=True)
        assert ds.index is not None
        assert len(ds.index) == 1
        assert ds.index.iloc[0]["pair_key"] == "PSP_001234_1780__PSP_005678_1780"

    def test_missing_required_ortho_drops_pair(self, tmp_path):
        # Only DTM file, no orthos materialised → no ortho overlap satisfies
        images = tmp_path / "images"
        images.mkdir()
        _write_dtm_tif(images / "DTEEC_001.IMG")

        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(
                root=str(tmp_path), include_ortho=True, ortho_type="RED",
                reuse_cache=False,
            )
        # Strip the right RED row → pair is incomplete → must be dropped
        df = _raw_index_for_pair(images)
        df = df[df["PRODUCT_ID"] != "PSP_005678_1780_RED_A_01_ORTHO"].reset_index(
            drop=True
        )
        ds._raw_index = df
        from torchgeo.datasets.errors import DatasetNotFoundError
        with pytest.raises(DatasetNotFoundError):
            ds._build_spatial_index(force_rebuild=True)

    def test_ortho_overlap_below_threshold_drops_pair(self, tmp_path, caplog):
        """Ortho footprint barely overlapping DTM (<75%) → pair dropped.

        Covers dtm.py:767-776 (Misalignment-detected drop).
        """
        images = tmp_path / "images"
        images.mkdir()
        _write_dtm_tif(images / "DTEEC_001.IMG",
                       bounds=(-131.0, 18.0, -130.0, 19.0))
        # Orthos shifted far east so intersection with the DTM is tiny.
        _write_ortho_tif(images / "PSP_001234_1780_RED_A_01_ORTHO.JP2",
                         bands=1, bounds=(-130.05, 18.0, -129.0, 19.0))
        _write_ortho_tif(images / "PSP_005678_1780_RED_A_01_ORTHO.JP2",
                         bands=1, bounds=(-130.05, 18.0, -129.0, 19.0))
        _write_lbl(images / "PSP_001234_1780_RED_A_01_ORTHO.LBL")
        _write_lbl(images / "PSP_005678_1780_RED_A_01_ORTHO.LBL")

        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(
                root=str(tmp_path), include_ortho=True, ortho_type="RED",
                reuse_cache=False,
            )
        ds._raw_index = _raw_index_for_pair(images)
        from torchgeo.datasets.errors import DatasetNotFoundError
        import logging
        with caplog.at_level(logging.ERROR, logger="dataset.core.dtm"):
            with pytest.raises(DatasetNotFoundError):
                ds._build_spatial_index(force_rebuild=True)
        assert "Misalignment detected" in caplog.text

    def test_invalid_scaling_factor_drops_pair(self, tmp_path, caplog):
        """Ortho overlaps DTM but LBL scaling=1/offset=0 → pair dropped.

        Covers dtm.py:781-791 (Incorrect scaling_factor drop).
        """
        images = tmp_path / "images"
        images.mkdir()
        _write_dtm_tif(images / "DTEEC_001.IMG")
        _write_ortho_tif(images / "PSP_001234_1780_RED_A_01_ORTHO.JP2", bands=1)
        _write_ortho_tif(images / "PSP_005678_1780_RED_A_01_ORTHO.JP2", bands=1)
        # Invalid radiometric calibration: scaling=1, offset=0.
        _write_lbl(images / "PSP_001234_1780_RED_A_01_ORTHO.LBL",
                   scaling=1.0, offset=0.0)
        _write_lbl(images / "PSP_005678_1780_RED_A_01_ORTHO.LBL",
                   scaling=1.0, offset=0.0)

        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(
                root=str(tmp_path), include_ortho=True, ortho_type="RED",
                reuse_cache=False,
            )
        ds._raw_index = _raw_index_for_pair(images)
        from torchgeo.datasets.errors import DatasetNotFoundError
        import logging
        with caplog.at_level(logging.ERROR, logger="dataset.core.dtm"):
            with pytest.raises(DatasetNotFoundError):
                ds._build_spatial_index(force_rebuild=True)
        assert "Incorrect scaling_factor" in caplog.text


# ---------------------------------------------------------------------------
# Cache reuse short-circuit
# ---------------------------------------------------------------------------


class TestBuildSpatialIndexBranches:
    """Cover the scattered _build_spatial_index branches.

    * non-numeric MAP_SCALE → except pass (dtm.py:678-680)
    * DTM file absent → empty footprint path list (dtm.py:705-709)
    * geometry unresolvable (antimeridian, no corners) → drop (dtm.py:752-754)
    """

    def test_bad_map_scale_and_missing_dtm_file(self, tmp_path):
        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(
                root=str(tmp_path), include_ortho=False, reuse_cache=False,
            )
        df = _raw_index_for_pair(tmp_path / "images")
        # DTM .IMG is never materialised → dtm_path won't exist (709 branch).
        df["MAP_SCALE"] = df["MAP_SCALE"].astype(object)
        df.loc[df["DATA_TYPE"] == "DTM", "MAP_SCALE"] = "not-a-number"
        ds._raw_index = df
        ds._build_spatial_index(force_rebuild=True)
        # Pair still resolves geometry from CORNER columns.
        assert ds.index is not None
        assert len(ds.index) == 1
        # map_scale stayed at its default (float() raised → except pass).
        assert "map_scale" in ds.index.columns

    def test_unresolvable_geometry_drops_pair(self, tmp_path, caplog):
        import logging
        from torchgeo.datasets.errors import DatasetNotFoundError

        with patch.object(MarsHiRISEDTM, "_verify", return_value=None):
            ds = MarsHiRISEDTM(
                root=str(tmp_path), include_ortho=False, reuse_cache=False,
            )
        df = _raw_index_for_pair(tmp_path / "images")
        # Remove CORNER columns and make min/max longitude straddle the
        # antimeridian so corners_to_polygon and the min/max bbox both fail
        # → _geometry_from_footprint_result returns None.
        corner_cols = [c for c in df.columns if c.startswith("CORNER")]
        df = df.drop(columns=corner_cols)
        # 170 → 170, 190 → -170 ⇒ lon_min (170) > lon_max (-170) ⇒ antimeridian
        df["MINIMUM_LONGITUDE"] = 170.0
        df["MAXIMUM_LONGITUDE"] = 190.0
        ds._raw_index = df
        with caplog.at_level(logging.WARNING, logger="dataset.core.dtm"):
            with pytest.raises(DatasetNotFoundError):
                ds._build_spatial_index(force_rebuild=True)
        assert "Could not generate valid geometry" in caplog.text


class TestBuildSpatialIndexCacheShortCircuit:
    def test_returns_when_cache_loadable(self, dtm_dataset):
        with patch.object(dtm_dataset, "_try_load_cache", return_value=True):
            # Should return early — _raw_index is None and would otherwise raise
            dtm_dataset._raw_index = None
            dtm_dataset._build_spatial_index(force_rebuild=False)
