"""Targeted coverage tests for HiRISEGeoSampler edge / optimal-mode paths.

Closes the remaining gaps in dataset.sampling.sampler:
  * _load_cached_split corrupt-file warning (302-304)
  * split=None → "all" (445)
  * center_mode / patch_overlap validation (462-469)
  * _user_bbox cache-key branch (587)
  * optimal-mode: valid-region exception fallback (815-820),
    packing exception fallback (842-854), per-strip simple fallback
    (_fallback_simple_strip 904-924), too-small optimal strip (792-794)
  * split_summary / per_strip_stats properties (976, 997)
"""

from __future__ import annotations

import logging
import types

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from helpers import make_mock_dataset
from dataset.sampling.sampler import HiRISEGeoSampler, _load_cached_split
from torchgeo.samplers import Units


@pytest.fixture
def strips_dataset(mars_crs):
    strips = [box(-140.0 + i * 5.0, 10.0, -140.0 + i * 5.0 + 0.5, 11.0)
              for i in range(6)]
    return make_mock_dataset(strips, mars_crs)


def _make(ds, **kw):
    kw.setdefault("size", 0.05)
    kw.setdefault("units", Units.CRS)
    return HiRISEGeoSampler(ds, **kw)


class TestLoadCachedSplitCorrupt:
    def test_corrupt_json_returns_none_and_warns(self, tmp_path, caplog):
        bad = tmp_path / "split_bad.json"
        bad.write_text("{ not valid json")
        with caplog.at_level(logging.WARNING, logger="dataset.sampling.sampler"):
            result = _load_cached_split(bad)
        assert result is None
        assert "Corrupt split cache" in caplog.text

    def test_missing_assignments_key_returns_none(self, tmp_path):
        bad = tmp_path / "split_noassign.json"
        bad.write_text('{"metadata": {}}')
        assert _load_cached_split(bad) is None

    def test_nonexistent_returns_none(self, tmp_path):
        assert _load_cached_split(tmp_path / "ghost.json") is None


class TestConstructorBranches:
    def test_split_none_defaults_to_all(self, strips_dataset):
        s = _make(strips_dataset, split=None)
        # "all" → train with fractions (1,0,0): every pair is train.
        assert all(v == "train" for v in s._assignments.values())

    def test_invalid_center_mode_raises(self, strips_dataset):
        with pytest.raises(ValueError, match="Invalid center_mode"):
            _make(strips_dataset, split="train", center_mode="turbo")

    def test_patch_overlap_out_of_range_raises(self, strips_dataset):
        with pytest.raises(ValueError, match="patch_overlap must be in"):
            _make(strips_dataset, split="train",
                  center_mode="optimal", patch_overlap=1.5)

    def test_user_bbox_used_in_cache_key(self, mars_crs):
        """Dataset exposing _user_bbox hits the _user_bbox cache branch (587)."""
        ds = make_mock_dataset([box(-140.0, 10.0, -139.0, 11.0)], mars_crs)
        ds._user_bbox = (-141.0, 9.0, -138.0, 12.0)
        s = _make(ds, split="all")
        assert s.cache_hash  # built without error via the _user_bbox branch


class TestOptimalMode:
    def test_optimal_mode_produces_centers(self, strips_dataset):
        s = _make(strips_dataset, split="all", center_mode="optimal")
        assert len(s._centers) > 0
        assert s.per_strip_stats  # property (997)
        assert any(d.get("mode") == "optimal" for d in s.per_strip_stats)

    def test_optimal_too_small_strip_skipped(self, mars_crs):
        tiny = box(-136.0, 18.0, -135.999, 18.001)
        ds = make_mock_dataset([tiny], mars_crs)
        s = _make(ds, split="all", center_mode="optimal", size=0.05)
        assert s._centers == []
        assert any(d.get("reason") == "too_small" for d in s.per_strip_stats)

    def test_valid_region_exception_falls_back_to_simple(self, strips_dataset):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "dataset.sampling.sampler.generate_valid_center_region",
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
            )
            s = _make(strips_dataset, split="all", center_mode="optimal")
        assert any(d.get("reason") == "fallback_simple"
                   for d in s.per_strip_stats)

    def test_packing_exception_falls_back_to_simple(self, strips_dataset):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "dataset.sampling.sampler.pack_patches_independent_strips",
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
            )
            s = _make(strips_dataset, split="all", center_mode="optimal")
        assert any(d.get("reason") == "fallback_simple"
                   for d in s.per_strip_stats)

    def test_valid_region_none_falls_back(self, strips_dataset):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "dataset.sampling.sampler.generate_valid_center_region",
                lambda *a, **k: None,
            )
            s = _make(strips_dataset, split="all", center_mode="optimal")
        assert any(d.get("reason") == "fallback_simple"
                   for d in s.per_strip_stats)


class TestSimpleModeEdges:
    """_build_valid_centers_simple buffer-except / too-small (722-733)."""

    def test_simple_buffer_exception_falls_back(self, mars_crs):
        from shapely.geometry import Polygon as P

        strip = P([(-136.0, 18.0), (-135.0, 18.0),
                   (-135.0, 19.0), (-136.0, 19.0)])
        ds = make_mock_dataset([strip], mars_crs)
        original = strip.buffer

        def bad_buffer(dist, *a, **k):
            if dist < 0:
                raise RuntimeError("simulated buffer failure")
            return original(dist, *a, **k)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(strip.__class__, "buffer", bad_buffer)
            s = _make(ds, split="all", center_mode="simple", size=0.05)
        # Falls back to the original footprint; centres still found.
        assert len(s._centers) > 0

    def test_simple_too_small_strip_recorded(self, mars_crs):
        tiny = box(-136.0, 18.0, -135.999, 18.001)
        ds = make_mock_dataset([tiny], mars_crs)
        s = _make(ds, split="all", center_mode="simple", size=0.05)
        assert s._centers == []
        assert any(d.get("reason") == "too_small" for d in s.per_strip_stats)


class TestOptimalSkipSplit:
    """Optimal mode skips pairs not assigned to this split (792-793)."""

    def test_optimal_skips_other_split_pairs(self, mars_crs):
        strips = [box(-170.0 + i * 20.0, 10.0, -170.0 + i * 20.0 + 0.5, 11.0)
                  for i in range(8)]
        ds = make_mock_dataset(strips, mars_crs)
        s = _make(
            ds, split="test", center_mode="optimal",
            split_method="random", split_fractions=(0.6, 0.2, 0.2),
        )
        # At least one pair is assigned to a non-test split → skip branch hit.
        assert any(v != "test" for v in s._assignments.values())


class TestFallbackSimpleStripAndDispatcher:
    """_fallback_simple_strip edges (904-909) + dispatcher (931-934)."""

    def _sampler(self, mars_crs, **kw):
        ds = make_mock_dataset(
            [box(-140.0, 10.0, -139.5, 11.0)], mars_crs
        )
        return _make(ds, split="all", **kw)

    def test_fallback_too_small_returns_zero(self, mars_crs):
        s = self._sampler(mars_crs, center_mode="optimal", size=0.05)
        tiny = box(0.0, 0.0, 0.001, 0.001)
        interval = s.index.index[0]
        added = s._fallback_simple_strip(0, tiny, interval)
        assert added == 0

    def test_fallback_buffer_exception_uses_original(self, mars_crs):
        s = self._sampler(mars_crs, center_mode="optimal", size=0.05)
        strip = box(-140.0, 10.0, -139.0, 11.0)
        interval = s.index.index[0]
        original = strip.buffer

        def bad_buffer(dist, *a, **k):
            if dist < 0:
                raise RuntimeError("buffer failed")
            return original(dist, *a, **k)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(strip.__class__, "buffer", bad_buffer)
            added = s._fallback_simple_strip(0, strip, interval)
        assert added > 0

    def test_build_valid_centers_dispatcher_simple(self, mars_crs):
        s = self._sampler(mars_crs, center_mode="simple", size=0.05)
        s._centers = []
        s._per_strip_stats = []
        s._build_valid_centers()  # dispatcher → simple
        assert len(s._centers) > 0

    def test_build_valid_centers_dispatcher_optimal(self, mars_crs):
        s = self._sampler(mars_crs, center_mode="optimal", size=0.05)
        s._centers = []
        s._per_strip_stats = []
        s._build_valid_centers()  # dispatcher → optimal
        assert len(s._centers) > 0


class TestSplitSummary:
    def test_split_summary_reports_counts(self, strips_dataset):
        s = _make(strips_dataset, split="all", center_mode="simple")
        summary = s.split_summary
        assert summary["split"] == "train"
        assert summary["total_pairs"] == 6
        assert summary["pairs_in_split"] == 6
        assert summary["centres_in_split"] == len(s._centers)
        assert set(summary["pairs_per_split"]) >= {"train", "val", "test"}
