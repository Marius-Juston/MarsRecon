"""Tests for MarsCLIP split manifests."""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest
import torch

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from marsclip_splits import (
    build_patch_split_manifest,
    build_dataset_subsets,
    compute_split_counts,
    load_patch_split_manifest,
    resolve_manifest_indices,
    save_patch_split_manifest,
    summarize_patch_split_manifest,
)


class _DummyPatchDataset(torch.utils.data.Dataset):
    def __init__(self, size: int) -> None:
        self.items = list(range(size))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> int:
        return self.items[index]


def _patch_records(count: int = 20) -> pd.DataFrame:
    return pd.DataFrame({"patch_id": [f"patch_{idx}" for idx in range(count)]})


def test_compute_split_counts_matches_requested_fractions():
    train_count, val_count, test_count = compute_split_counts(
        100,
        train_fraction=0.70,
        val_fraction=0.15,
        test_fraction=0.15,
    )

    assert (train_count, val_count, test_count) == (70, 15, 15)


def test_build_patch_split_manifest_creates_holdout_and_folds():
    manifest = build_patch_split_manifest(_patch_records(100), split_seed=3, num_folds=5)

    assert len(manifest) == 100
    assert int((manifest["holdout_split"] == "train").sum()) == 70
    assert int((manifest["holdout_split"] == "val").sum()) == 15
    assert int((manifest["holdout_split"] == "test").sum()) == 15
    assert set(manifest.loc[manifest["is_test"], "fold"].dropna().tolist()) == set()
    assert set(int(value) for value in manifest.loc[~manifest["is_test"], "fold"].dropna().unique()) == {0, 1, 2, 3, 4}


def test_summarize_patch_split_manifest_reports_sizes():
    manifest = build_patch_split_manifest(_patch_records(20), split_seed=3, num_folds=4)

    summary = summarize_patch_split_manifest(manifest, num_folds=4)

    assert summary["num_samples"] == 20
    assert summary["train_count"] == 14
    assert summary["val_count"] == 3
    assert summary["test_count"] == 3
    assert summary["num_folds"] == 4


def test_manifest_round_trip_csv(tmp_path):
    manifest = build_patch_split_manifest(_patch_records(12), split_seed=1, num_folds=3)
    out_path = save_patch_split_manifest(manifest, tmp_path / "splits.csv")
    loaded = load_patch_split_manifest(out_path)

    assert loaded["patch_id"].tolist() == manifest["patch_id"].tolist()
    assert loaded["holdout_split"].tolist() == manifest["holdout_split"].tolist()


def test_resolve_manifest_indices_supports_holdout_and_kfold():
    patch_records = _patch_records(40)
    manifest = build_patch_split_manifest(patch_records, split_seed=2, num_folds=5)

    holdout_train = resolve_manifest_indices(patch_records, manifest, role="train", mode="holdout")
    holdout_val = resolve_manifest_indices(patch_records, manifest, role="val", mode="holdout")
    holdout_test = resolve_manifest_indices(patch_records, manifest, role="test", mode="holdout")
    kfold_val = resolve_manifest_indices(patch_records, manifest, role="val", mode="kfold", fold_index=0)

    assert len(set(holdout_train) & set(holdout_val)) == 0
    assert len(set(holdout_train) & set(holdout_test)) == 0
    assert len(set(kfold_val) & set(holdout_test)) == 0


def test_build_dataset_subsets_uses_manifest():
    dataset = _DummyPatchDataset(30)
    patch_records = _patch_records(30)
    manifest = build_patch_split_manifest(patch_records, split_seed=7, num_folds=5)

    train_subset, val_subset, test_subset = build_dataset_subsets(
        dataset,
        patch_records,
        manifest,
        mode="holdout",
    )

    assert len(train_subset) == 21
    assert len(val_subset) == 4
    assert len(test_subset) == 5


def test_resolve_manifest_indices_raises_when_manifest_is_misaligned():
    patch_records = _patch_records(10)
    manifest = build_patch_split_manifest(patch_records, split_seed=0, num_folds=2)
    manifest.loc[0, "patch_id"] = "missing_patch"

    with pytest.raises(ValueError, match="not present in current dataset"):
        resolve_manifest_indices(patch_records, manifest, role="train", mode="holdout")
