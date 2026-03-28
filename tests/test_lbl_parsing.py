"""Tests for _ProductMeta LBL metadata parsing.

Covers:
- Correct extraction of SCALING_FACTOR, OFFSET, SAMPLE_BITS, SAMPLE_BIT_MASK,
  BANDS, and FILTER_NAME from a synthetic PDS3 label.
- Graceful fallback to defaults when the file is missing or empty.
"""

import pathlib
import textwrap

import pytest

# src/ is on sys.path via conftest.py
from temp import _ProductMeta


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
        meta = _ProductMeta.from_lbl(synthetic_lbl)
        assert meta.scaling_factor == pytest.approx(2.5e-4, rel=1e-6)

    def test_offset(self, synthetic_lbl: pathlib.Path):
        meta = _ProductMeta.from_lbl(synthetic_lbl)
        assert meta.offset == pytest.approx(0.04, rel=1e-6)

    def test_sample_bits(self, synthetic_lbl: pathlib.Path):
        meta = _ProductMeta.from_lbl(synthetic_lbl)
        assert meta.sample_bits == 16

    def test_effective_max_dn_from_bit_mask(self, synthetic_lbl: pathlib.Path):
        # 2#0000001111111111# = 10 bits set → 2^10 - 1 = 1023
        meta = _ProductMeta.from_lbl(synthetic_lbl)
        assert meta.effective_max_dn == 1023

    def test_bands(self, synthetic_lbl: pathlib.Path):
        meta = _ProductMeta.from_lbl(synthetic_lbl)
        assert meta.bands == 3

    def test_filter_names(self, synthetic_lbl: pathlib.Path):
        meta = _ProductMeta.from_lbl(synthetic_lbl)
        assert meta.filter_names == ["NEAR-INFRARED", "RED", "BLUE-GREEN"]


# ---------------------------------------------------------------------------
# Fallback to defaults
# ---------------------------------------------------------------------------


class TestProductMetaDefaults:
    def test_missing_file_returns_defaults(self, tmp_path: pathlib.Path):
        missing = tmp_path / "nonexistent.LBL"
        meta = _ProductMeta.from_lbl(missing)
        assert meta.scaling_factor == pytest.approx(_DEFAULT_SCALING_FACTOR, rel=1e-6)
        assert meta.offset == pytest.approx(_DEFAULT_OFFSET, rel=1e-6)

    def test_empty_file_returns_defaults(self, tmp_path: pathlib.Path):
        empty = tmp_path / "empty.LBL"
        empty.write_text("")
        meta = _ProductMeta.from_lbl(empty)
        assert meta.scaling_factor == pytest.approx(_DEFAULT_SCALING_FACTOR, rel=1e-6)
        assert meta.offset == pytest.approx(_DEFAULT_OFFSET, rel=1e-6)

    def test_missing_file_does_not_raise(self, tmp_path: pathlib.Path):
        missing = tmp_path / "ghost.LBL"
        # Must not raise any exception.
        _ProductMeta.from_lbl(missing)

    def test_partial_lbl_uses_defaults_for_missing_keys(self, tmp_path: pathlib.Path):
        partial = tmp_path / "partial.LBL"
        partial.write_text("PDS_VERSION_ID = PDS3\nSCALING_FACTOR = 1.0e-03\nEND\n")
        meta = _ProductMeta.from_lbl(partial)
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

        meta = _ProductMeta.from_lbl(synthetic_lbl)
        dn = np.array([0, 512, 1023], dtype=np.float32)
        result = np.clip(dn * meta.scaling_factor + meta.offset, 0.0, 1.0)

        assert result[0] == pytest.approx(meta.offset, rel=1e-5)
        assert result[-1] == pytest.approx(
            min(1.0, 1023 * meta.scaling_factor + meta.offset), rel=1e-5
        )
        assert all(0.0 <= v <= 1.0 for v in result)
