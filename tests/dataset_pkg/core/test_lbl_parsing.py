"""Tests for ProductMeta LBL metadata parsing.

Covers:
- Correct extraction of SCALING_FACTOR, OFFSET, SAMPLE_BITS, SAMPLE_BIT_MASK,
  BANDS, and FILTER_NAME from a synthetic PDS3 label.
- Graceful fallback to defaults when the file is missing or empty.
"""

import pathlib
from unittest.mock import patch

import pytest

# src/ is on sys.path via conftest.py
from dataset.core.base import ProductMeta

# ---------------------------------------------------------------------------
# Default constants (replicate the values from temp.py for comparison)
# ---------------------------------------------------------------------------

_DEFAULT_SCALING_FACTOR = 2.37936949017414e-04
_DEFAULT_OFFSET = 0.037954361744101


# ---------------------------------------------------------------------------
# Parsing a valid LBL
# ---------------------------------------------------------------------------


class TestProductMetaFromValidLbl:
    def test_scaling_factor(self, synthetic_lbl: pathlib.Path):
        meta = ProductMeta.from_lbl(synthetic_lbl)
        assert meta.scaling_factor == pytest.approx(2.5e-4, rel=1e-6)

    def test_offset(self, synthetic_lbl: pathlib.Path):
        meta = ProductMeta.from_lbl(synthetic_lbl)
        assert meta.offset == pytest.approx(0.04, rel=1e-6)

    def test_sample_bits(self, synthetic_lbl: pathlib.Path):
        meta = ProductMeta.from_lbl(synthetic_lbl)
        assert meta.sample_bits == 16

    def test_effective_max_dn_from_bit_mask(self, synthetic_lbl: pathlib.Path):
        # 2#0000001111111111# = 10 bits set → 2^10 - 1 = 1023
        meta = ProductMeta.from_lbl(synthetic_lbl)
        assert meta.effective_max_dn == 1023

    def test_bands(self, synthetic_lbl: pathlib.Path):
        meta = ProductMeta.from_lbl(synthetic_lbl)
        assert meta.bands == 3

    def test_filter_names(self, synthetic_lbl: pathlib.Path):
        meta = ProductMeta.from_lbl(synthetic_lbl)
        assert meta.filter_names == ["NEAR-INFRARED", "RED", "BLUE-GREEN"]


# ---------------------------------------------------------------------------
# Fallback to defaults
# ---------------------------------------------------------------------------


class TestProductMetaDefaults:
    def test_missing_file_returns_defaults(self, tmp_path: pathlib.Path):
        missing = tmp_path / "nonexistent.LBL"
        meta = ProductMeta.from_lbl(missing)
        assert meta.scaling_factor == pytest.approx(_DEFAULT_SCALING_FACTOR, rel=1e-6)
        assert meta.offset == pytest.approx(_DEFAULT_OFFSET, rel=1e-6)

    def test_empty_file_returns_defaults(self, tmp_path: pathlib.Path):
        empty = tmp_path / "empty.LBL"
        empty.write_text("")
        meta = ProductMeta.from_lbl(empty)
        assert meta.scaling_factor == pytest.approx(_DEFAULT_SCALING_FACTOR, rel=1e-6)
        assert meta.offset == pytest.approx(_DEFAULT_OFFSET, rel=1e-6)

    def test_missing_file_does_not_raise(self, tmp_path: pathlib.Path):
        missing = tmp_path / "ghost.LBL"
        # Must not raise any exception.
        ProductMeta.from_lbl(missing)

    def test_partial_lbl_uses_defaults_for_missing_keys(self, tmp_path: pathlib.Path):
        partial = tmp_path / "partial.LBL"
        partial.write_text("PDS_VERSION_ID = PDS3\nSCALING_FACTOR = 1.0e-03\nEND\n")
        meta = ProductMeta.from_lbl(partial)
        # SCALING_FACTOR present
        assert meta.scaling_factor == pytest.approx(1.0e-3, rel=1e-6)
        # OFFSET absent → default
        assert meta.offset == pytest.approx(_DEFAULT_OFFSET, rel=1e-6)


# ---------------------------------------------------------------------------
# Radiometric calibration formula
# ---------------------------------------------------------------------------


class TestRadiometricCalibration:
    def test_dn_to_if_formula(self, synthetic_lbl: pathlib.Path):
        """I/F = DN * scaling_factor + offset, clipped to [0, 1]."""
        import numpy as np

        meta = ProductMeta.from_lbl(synthetic_lbl)
        dn = np.array([0, 512, 1023], dtype=np.float32)
        result = np.clip(dn * meta.scaling_factor + meta.offset, 0.0, 1.0)

        assert result[0] == pytest.approx(meta.offset, rel=1e-5)
        assert result[-1] == pytest.approx(
            min(1.0, 1023 * meta.scaling_factor + meta.offset), rel=1e-5
        )
        assert all(0.0 <= v <= 1.0 for v in result)


# ---------------------------------------------------------------------------
# OSError when reading LBL file (lines 131-133)
# ---------------------------------------------------------------------------


class TestProductMetaRichLbl:
    """Exercise every optional-field parsing branch (base.py:184-249)."""

    @pytest.fixture
    def rich_lbl(self, tmp_path: pathlib.Path) -> pathlib.Path:
        content = (
            'PDS_VERSION_ID = PDS3\n'
            'INCIDENCE_ANGLE = 45.5\n'
            'SOLAR_AZIMUTH = 120.25\n'
            'EMISSION_ANGLE = 3.2\n'
            'PHASE_ANGLE = 48.7\n'
            'LOCAL_TIME = 15.5\n'
            'SOLAR_LONGITUDE = 251.3\n'
            'SUB_SOLAR_AZIMUTH = 99.9\n'
            'NORTH_AZIMUTH = 270.0\n'
            'OBSERVATION_ID = "PSP_001430_1780"\n'
            'PRODUCT_ID = "PSP_001430_1780_COLOR"\n'
            'PRODUCT_VERSION_ID = "1"\n'
            'TARGET_NAME = "MARS"\n'
            'MISSION_PHASE_NAME = "PRIMARY SCIENCE PHASE"\n'
            'RATIONALE_DESC = "Layered deposits in crater"\n'
            'START_TIME = 2007-01-01T00:00:00.000\n'
            'STOP_TIME = 2007-01-01T00:01:00.000\n'
            'PRODUCT_CREATION_TIME = 2007-06-01T12:00:00.000\n'
            'ORBIT_NUMBER = 1430\n'
            'MAP_SCALE = 0.25\n'
            'MAP_RESOLUTION = 100000.0\n'
            'MAP_PROJECTION_TYPE = "EQUIRECTANGULAR"\n'
            'CENTER_LATITUDE = -5.0\n'
            'CENTER_LONGITUDE = 224.0\n'
            'CENTER_FILTER_WAVELENGTH = 700.0\n'
            'SCALING_FACTOR = 2.5e-04\n'
            'OFFSET = 0.04\n'
            'SAMPLE_BITS = 16\n'
            'BANDS = 3\n'
            'END\n'
        )
        p = tmp_path / "RICH.LBL"
        p.write_text(content)
        return p

    def test_all_optional_fields_parsed(self, rich_lbl):
        m = ProductMeta.from_lbl(rich_lbl)
        assert m.incidence_angle == pytest.approx(45.5)
        assert m.solar_azimuth == pytest.approx(120.25)
        assert m.emission_angle == pytest.approx(3.2)
        assert m.phase_angle == pytest.approx(48.7)
        assert m.local_time == pytest.approx(15.5)
        assert m.solar_longitude == pytest.approx(251.3)
        assert m.sub_solar_azimuth == pytest.approx(99.9)
        assert m.north_azimuth == pytest.approx(270.0)
        assert m.observation_id == "PSP_001430_1780"
        assert m.product_id == "PSP_001430_1780_COLOR"
        assert m.product_version_id == "1"
        assert m.target_name == "MARS"
        assert m.mission_phase_name == "PRIMARY SCIENCE PHASE"
        assert m.rationale_desc == "Layered deposits in crater"
        assert m.start_time.startswith("2007-01-01T00:00:00")
        assert m.stop_time.startswith("2007-01-01T00:01:00")
        assert m.product_creation_time.startswith("2007-06-01T12:00:00")
        assert m.orbit_number == 1430
        assert m.map_scale == pytest.approx(0.25)
        assert m.map_resolution == pytest.approx(100000.0)
        assert m.map_projection_type == "EQUIRECTANGULAR"
        assert m.center_latitude == pytest.approx(-5.0)
        assert m.center_longitude == pytest.approx(224.0)
        assert m.center_filter_wavelength == pytest.approx(700.0)

    def test_solar_azimuth_not_confused_with_sub_solar(self, rich_lbl):
        """SOLAR_AZIMUTH must win over SUB_SOLAR_AZIMUTH when both present."""
        m = ProductMeta.from_lbl(rich_lbl)
        assert m.solar_azimuth == pytest.approx(120.25)
        assert m.sub_solar_azimuth == pytest.approx(99.9)

    def test_sub_solar_azimuth_only_falls_back(self, tmp_path):
        """SUB_SOLAR_AZIMUTH-only label → solar_azimuth falls back (base.py:199-200)."""
        p = tmp_path / "SUBONLY.LBL"
        p.write_text(
            "PDS_VERSION_ID = PDS3\nSUB_SOLAR_AZIMUTH = 88.8\nEND\n"
        )
        m = ProductMeta.from_lbl(p)
        assert m.sub_solar_azimuth == pytest.approx(88.8)
        assert m.solar_azimuth == pytest.approx(88.8)

    def test_single_filter_name_branch(self, tmp_path):
        """Non-tuple FILTER_NAME → single-element list (base.py:248-249)."""
        p = tmp_path / "ONEFILTER.LBL"
        p.write_text('PDS_VERSION_ID = PDS3\nFILTER_NAME = "RED"\nEND\n')
        m = ProductMeta.from_lbl(p)
        assert m.filter_names == ["RED"]


class TestProductMetaOsError:
    def test_oserror_during_read_text_returns_defaults(self, tmp_path: pathlib.Path):
        """OSError raised by read_text → warning logged, returns defaults (lines 131-133)."""
        lbl = tmp_path / "unreadable.LBL"
        lbl.write_text("PDS_VERSION_ID = PDS3\nEND\n")

        with patch.object(pathlib.Path, "read_text", side_effect=OSError("permission denied")):
            meta = ProductMeta.from_lbl(lbl)

        assert meta.scaling_factor == pytest.approx(_DEFAULT_SCALING_FACTOR, rel=1e-6)
        assert meta.offset == pytest.approx(_DEFAULT_OFFSET, rel=1e-6)
