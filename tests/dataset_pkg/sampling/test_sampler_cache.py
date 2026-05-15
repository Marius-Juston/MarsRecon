"""Tests for HiRISEGeoSampler split-assignment caching and split modes.

Exercises the disk split-cache round-trip, the cache/​dataset-size mismatch
recovery path, geographic vs random splitting, and K-fold cross-validation
through the real sampler (not just the ``_compute_split_assignments`` helper).
"""

from __future__ import annotations

import json
import logging

import pytest
from shapely.geometry import box

from helpers import make_mock_dataset
from dataset.sampling.sampler import HiRISEGeoSampler
from torchgeo.samplers import Units


@pytest.fixture
def many_strip_dataset(tmp_path, mars_crs):
    """20 narrow strips spread across longitude, with a real ``root`` on disk."""
    strips = [
        box(-170.0 + i * 15.0, 10.0, -170.0 + i * 15.0 + 0.1, 14.0)
        for i in range(20)
    ]
    ds = make_mock_dataset(strips, mars_crs)
    ds.root = str(tmp_path)
    ds.target = None
    return ds


def _make(ds, **kw):
    kw.setdefault("size", 0.02)
    kw.setdefault("units", Units.CRS)
    kw.setdefault("center_mode", "simple")
    return HiRISEGeoSampler(ds, **kw)


class TestSplitCacheRoundTrip:
    @staticmethod
    def _find_split_cache(tmp_path):
        # rglob("split_*.json") also matches the split_<hash>_MANIFEST.json
        # sidecar; filter it out so we always operate on the assignments file.
        return [
            p for p in tmp_path.rglob("split_*.json")
            if not p.name.endswith("_MANIFEST.json")
        ]

    def test_cache_file_written_then_reused(self, many_strip_dataset, tmp_path, caplog):
        s1 = _make(many_strip_dataset, split="train", reuse_cache=True)
        # A split_*.json cache file should now exist somewhere under root
        caches = self._find_split_cache(tmp_path)
        assert caches, "expected a split cache file to be written"
        payload = json.loads(caches[0].read_text())
        assert isinstance(payload, dict)

        # Second sampler with identical config loads the cache
        with caplog.at_level(logging.INFO, logger="dataset.sampling.sampler"):
            s2 = _make(many_strip_dataset, split="train", reuse_cache=True)
        assert "Loaded cached split" in caplog.text
        assert s1._assignments == s2._assignments

    def test_cache_size_mismatch_recomputes(self, many_strip_dataset, tmp_path, caplog):
        _make(many_strip_dataset, split="train", reuse_cache=True)
        cache = self._find_split_cache(tmp_path)[0]
        # Corrupt the cache so it has the wrong number of pairs
        data = json.loads(cache.read_text())
        # Drop assignment entries → length mismatch vs the 20-pair dataset
        data["assignments"] = dict(list(data["assignments"].items())[:3])
        cache.write_text(json.dumps(data))

        with caplog.at_level(logging.WARNING, logger="dataset.sampling.sampler"):
            s = _make(many_strip_dataset, split="train", reuse_cache=True)
        assert "recomputing" in caplog.text
        assert len(s._assignments) == 20

    def test_reuse_cache_false_ignores_existing(self, many_strip_dataset, caplog):
        _make(many_strip_dataset, split="train", reuse_cache=True)
        with caplog.at_level(logging.INFO, logger="dataset.sampling.sampler"):
            _make(many_strip_dataset, split="train", reuse_cache=False)
        assert "Loaded cached split" not in caplog.text


class TestSplitModes:
    def test_geographic_longitude_split(self, many_strip_dataset):
        train = _make(
            many_strip_dataset, split="train",
            split_method="geographic", split_axis="longitude",
        )
        test = _make(
            many_strip_dataset, split="test",
            split_method="geographic", split_axis="longitude",
        )
        assert set(train._assignments.values()) <= {"train", "val", "test"}
        # train and test select different pairs
        train_pairs = {k for k, v in train._assignments.items() if v == "train"}
        test_pairs = {k for k, v in test._assignments.items() if v == "test"}
        assert train_pairs.isdisjoint(test_pairs)

    def test_geographic_latitude_axis(self, many_strip_dataset):
        s = _make(
            many_strip_dataset, split="train",
            split_method="geographic", split_axis="latitude",
        )
        assert len(s._assignments) == 20

    def test_random_split(self, many_strip_dataset):
        s = _make(many_strip_dataset, split="train", split_method="random")
        assert len(s._assignments) == 20

    def test_invalid_split_axis_raises(self, many_strip_dataset):
        with pytest.raises(ValueError, match="Unknown split_axis"):
            _make(
                many_strip_dataset, split="train",
                split_method="geographic", split_axis="depth",
            )


class TestKFold:
    def test_kfold_disjoint_test_sets(self, many_strip_dataset):
        s0 = _make(many_strip_dataset, split="test", n_folds=5, fold_idx=0)
        s1 = _make(many_strip_dataset, split="test", n_folds=5, fold_idx=1)
        t0 = {k for k, v in s0._assignments.items() if v == "test"}
        t1 = {k for k, v in s1._assignments.items() if v == "test"}
        assert t0 and t1
        assert t0.isdisjoint(t1)

    def test_kfold_out_of_range_raises(self, many_strip_dataset):
        with pytest.raises(ValueError, match="out of range"):
            _make(many_strip_dataset, split="test", n_folds=5, fold_idx=9)

    def test_kfold_iteration_yields_patches(self, many_strip_dataset):
        s = _make(
            many_strip_dataset, split="train", n_folds=5, fold_idx=0,
            length=10,
        )
        patches = list(s)
        assert len(patches) == 10


class TestSplitValidation:
    def test_invalid_split_name_raises(self, many_strip_dataset):
        with pytest.raises(ValueError, match="Invalid split"):
            _make(many_strip_dataset, split="holdout")

    def test_invalid_method_raises(self, many_strip_dataset):
        with pytest.raises(ValueError, match="Invalid split_method"):
            _make(many_strip_dataset, split="train", split_method="kmeans")

    def test_fractions_must_sum_to_one(self, many_strip_dataset):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            _make(
                many_strip_dataset, split="train",
                split_fractions=(0.5, 0.2, 0.1),
            )

    def test_split_all_uses_whole_dataset(self, many_strip_dataset):
        s = _make(many_strip_dataset, split="all")
        assert all(v == "train" for v in s._assignments.values())
