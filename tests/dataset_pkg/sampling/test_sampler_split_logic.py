"""Unit tests for split-assignment helpers in :mod:`dataset.sampling.sampler`.

``_compute_split_assignments`` and ``_kfold_split`` are pure Python functions
that operate on positional indices and (optionally) per-pair spatial
coordinates.  They are deterministic given a seed and do not touch the
filesystem or GDAL, so they're tested directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataset.sampling.sampler import (
    _compute_split_assignments,
    _kfold_split,
)


# ---------------------------------------------------------------------------
# _compute_split_assignments — non-k-fold path
# ---------------------------------------------------------------------------


class TestComputeSplitAssignments:
    @pytest.fixture
    def kwargs(self):
        return dict(
            method="random",
            train_fraction=0.8,
            val_fraction=0.1,
            test_fraction=0.1,
            seed=42,
            n_folds=None,
            fold_idx=0,
        )

    def test_returns_dict_keyed_by_pair_index(self, kwargs):
        a = _compute_split_assignments(10, None, **kwargs)
        assert set(a.keys()) == set(range(10))
        assert set(a.values()) <= {"train", "val", "test"}

    def test_split_counts_match_fractions(self, kwargs):
        n = 100
        a = _compute_split_assignments(n, None, **kwargs)
        from collections import Counter
        counts = Counter(a.values())
        # 0.8 / 0.1 / 0.1 of 100 → 80 / 10 / 10
        assert counts["train"] == 80
        assert counts["val"] == 10
        assert counts["test"] == 10

    def test_deterministic_for_same_seed(self, kwargs):
        a = _compute_split_assignments(50, None, **kwargs)
        b = _compute_split_assignments(50, None, **kwargs)
        assert a == b

    def test_different_seeds_produce_different_splits(self, kwargs):
        a = _compute_split_assignments(50, None, **{**kwargs, "seed": 1})
        b = _compute_split_assignments(50, None, **{**kwargs, "seed": 2})
        assert a != b

    def test_geographic_split_uses_coordinate_order(self, kwargs):
        # Geographic split sorts by coord → low-coord pairs go to train first.
        kwargs["method"] = "geographic"
        kwargs["train_fraction"] = 0.6
        kwargs["val_fraction"] = 0.2
        kwargs["test_fraction"] = 0.2
        coords = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        a = _compute_split_assignments(5, coords, **kwargs)
        # n_test=1, n_val=1, n_train=3
        # order = argsort(coords) = [0,1,2,3,4]
        # first 3 → train, next 1 → val, last 1 → test
        assert a[0] == "train"
        assert a[1] == "train"
        assert a[2] == "train"
        assert a[3] == "val"
        assert a[4] == "test"

    def test_geographic_falls_back_to_random_when_coords_none(self, kwargs):
        # Even with method='geographic', if pair_coords is None we use rng.
        kwargs["method"] = "geographic"
        a = _compute_split_assignments(20, None, **kwargs)
        assert len(a) == 20

    def test_zero_train_fraction_raises(self, kwargs):
        kwargs["train_fraction"] = 0.0
        kwargs["val_fraction"] = 0.5
        kwargs["test_fraction"] = 0.5
        with pytest.raises(ValueError, match="no training data"):
            _compute_split_assignments(2, None, **kwargs)

    def test_kfold_dispatch_when_n_folds_set(self, kwargs):
        # n_folds > 1 → dispatches to _kfold_split
        kwargs["n_folds"] = 4
        a = _compute_split_assignments(20, None, **kwargs)
        # Test set should be ~ n_pairs / n_folds = 5
        from collections import Counter
        c = Counter(a.values())
        assert c["test"] == 5


# ---------------------------------------------------------------------------
# _kfold_split — fold disjointness, range validation
# ---------------------------------------------------------------------------


class TestKfoldSplit:
    def _kw(self, **over):
        defaults = dict(
            method="random",
            n_folds=5,
            fold_idx=0,
            val_fraction=0.1,
            rng=np.random.default_rng(0),
        )
        defaults.update(over)
        return defaults

    def test_invalid_fold_idx_raises(self):
        with pytest.raises(ValueError, match="fold_idx"):
            _kfold_split(20, None, **self._kw(fold_idx=10))

    def test_negative_fold_idx_raises(self):
        with pytest.raises(ValueError, match="fold_idx"):
            _kfold_split(20, None, **self._kw(fold_idx=-1))

    def test_all_pairs_assigned(self):
        a = _kfold_split(20, None, **self._kw())
        assert set(a.keys()) == set(range(20))
        assert set(a.values()) <= {"train", "val", "test"}

    def test_test_fold_size_matches_target(self):
        # With 20 pairs and 5 folds → each fold has 4 pairs in test set.
        a = _kfold_split(20, None, **self._kw(fold_idx=2))
        from collections import Counter
        c = Counter(a.values())
        assert c["test"] == 4

    def test_remainder_distributed_to_first_folds(self):
        """11 pairs / 5 folds → fold 0/1 each get 3, folds 2-4 each get 2."""
        a0 = _kfold_split(11, None, **self._kw(fold_idx=0))
        a3 = _kfold_split(11, None, **self._kw(fold_idx=3))
        from collections import Counter
        assert Counter(a0.values())["test"] == 3
        assert Counter(a3.values())["test"] == 2

    def test_different_folds_select_disjoint_test_sets(self):
        a0 = _kfold_split(20, None, **self._kw(fold_idx=0))
        a1 = _kfold_split(20, None, **self._kw(fold_idx=1))
        test0 = {k for k, v in a0.items() if v == "test"}
        test1 = {k for k, v in a1.items() if v == "test"}
        assert test0.isdisjoint(test1)

    def test_geographic_method_uses_coords(self):
        """Verify geographic k-fold is deterministic and partitions all pairs."""
        coords = np.linspace(0.0, 10.0, 10)
        a = _kfold_split(10, coords, **self._kw(n_folds=2, fold_idx=0))
        # Test set must be exactly n_pairs / n_folds in size (10 / 2 = 5).
        test_idx = {k for k, v in a.items() if v == "test"}
        assert len(test_idx) == 5
        # And deterministic
        b = _kfold_split(10, coords, **self._kw(n_folds=2, fold_idx=0))
        assert a == b

    def test_val_fraction_zero_produces_no_val_or_one(self):
        # n_val = max(1, round(n_non_test * 0)) = max(1, 0) = 1
        a = _kfold_split(10, None, **self._kw(n_folds=5, val_fraction=0.0))
        from collections import Counter
        c = Counter(a.values())
        # Always at least one val pair guaranteed by max(1, …)
        assert c["val"] == 1
