"""Patch split manifests for MarsCLIP Stage A and multimodal alignment workflows."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


def compute_split_counts(
    num_samples: int,
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[int, int, int]:
    """Convert split fractions into deterministic integer counts."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    total = float(train_fraction) + float(val_fraction) + float(test_fraction)
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0.")
    if min(train_fraction, val_fraction, test_fraction) < 0.0:
        raise ValueError("train/val/test fractions must be non-negative.")

    train_count = int(round(num_samples * float(train_fraction)))
    val_count = int(round(num_samples * float(val_fraction)))
    test_count = num_samples - train_count - val_count
    if test_count < 0:
        raise ValueError("Split fractions produced a negative test count.")

    counts = [train_count, val_count, test_count]
    targets = [float(train_fraction), float(val_fraction), float(test_fraction)]
    while sum(counts) != num_samples:
        diff = num_samples - sum(counts)
        if diff > 0:
            idx = max(range(3), key=lambda i: targets[i] - counts[i] / max(num_samples, 1))
            counts[idx] += 1
        else:
            candidates = [idx for idx in range(3) if counts[idx] > 0]
            idx = max(candidates, key=lambda i: counts[i] / max(num_samples, 1) - targets[i])
            counts[idx] -= 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def _balanced_fold_sizes(num_items: int, num_folds: int) -> list[int]:
    sizes = [num_items // num_folds] * num_folds
    for idx in range(num_items % num_folds):
        sizes[idx] += 1
    return sizes


def build_patch_split_manifest(
    patch_records: pd.DataFrame,
    *,
    split_seed: int = 0,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    num_folds: int = 5,
) -> pd.DataFrame:
    """Build a manifest with holdout and K-fold assignments for patch ids."""
    if "patch_id" not in patch_records.columns:
        raise ValueError("patch_records must include a 'patch_id' column.")
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")

    patch_ids = patch_records["patch_id"].astype(str).tolist()
    num_samples = len(patch_ids)
    train_count, val_count, test_count = compute_split_counts(
        num_samples,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    generator = torch.Generator().manual_seed(int(split_seed))
    order = torch.randperm(num_samples, generator=generator).tolist()
    test_set = set(order[:test_count])
    remaining = order[test_count:]

    holdout_val_set = set(remaining[:val_count])
    holdout_train_set = set(remaining[val_count : val_count + train_count])

    fold_sizes = _balanced_fold_sizes(len(remaining), num_folds)
    fold_assignment: dict[int, int] = {}
    cursor = 0
    for fold_idx, size in enumerate(fold_sizes):
        for patch_index in remaining[cursor : cursor + size]:
            fold_assignment[patch_index] = fold_idx
        cursor += size

    records: list[dict[str, Any]] = []
    for patch_index, patch_id in enumerate(patch_ids):
        if patch_index in test_set:
            holdout_split = "test"
            fold = pd.NA
        elif patch_index in holdout_val_set:
            holdout_split = "val"
            fold = int(fold_assignment[patch_index])
        elif patch_index in holdout_train_set:
            holdout_split = "train"
            fold = int(fold_assignment[patch_index])
        else:
            raise RuntimeError("Patch index was not assigned to a split.")

        records.append(
            {
                "patch_id": patch_id,
                "patch_index": int(patch_index),
                "holdout_split": holdout_split,
                "fold": fold,
                "is_test": bool(holdout_split == "test"),
            }
        )

    manifest = pd.DataFrame.from_records(records)
    manifest["fold"] = manifest["fold"].astype("Int64")
    return manifest


def summarize_patch_split_manifest(
    manifest: pd.DataFrame,
    *,
    num_folds: int | None = None,
) -> dict[str, Any]:
    """Summarize a patch split manifest for logging/reporting."""
    if manifest.empty:
        raise ValueError("manifest must not be empty.")
    num_samples = len(manifest)
    train_count = int((manifest["holdout_split"] == "train").sum())
    val_count = int((manifest["holdout_split"] == "val").sum())
    test_count = int((manifest["holdout_split"] == "test").sum())
    summary: dict[str, Any] = {
        "num_samples": num_samples,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "train_fraction_realized": train_count / float(num_samples),
        "val_fraction_realized": val_count / float(num_samples),
        "test_fraction_realized": test_count / float(num_samples),
        "num_folds": int(num_folds if num_folds is not None else manifest["fold"].dropna().nunique()),
    }
    fold_sizes = manifest.loc[manifest["fold"].notna(), "fold"].value_counts().sort_index()
    summary["fold_sizes"] = {int(idx): int(value) for idx, value in fold_sizes.items()}
    return summary


def save_patch_split_manifest(manifest: pd.DataFrame, path: pathlib.Path | str) -> pathlib.Path:
    """Persist a split manifest as CSV or Parquet based on suffix."""
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        manifest.to_parquet(out, index=False)
    else:
        manifest.to_csv(out, index=False)
    return out


def load_patch_split_manifest(path: pathlib.Path | str) -> pd.DataFrame:
    """Load a persisted split manifest."""
    source = pathlib.Path(path)
    if source.suffix.lower() == ".parquet":
        manifest = pd.read_parquet(source)
    else:
        manifest = pd.read_csv(source)
    manifest["patch_id"] = manifest["patch_id"].astype(str)
    if "fold" in manifest.columns:
        manifest["fold"] = manifest["fold"].astype("Int64")
    return manifest


def save_split_summary(summary: dict[str, Any], path: pathlib.Path | str) -> pathlib.Path:
    """Persist split summary metadata as JSON."""
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    return out


def select_patch_ids_for_role(
    manifest: pd.DataFrame,
    *,
    role: str,
    mode: str = "holdout",
    fold_index: int = 0,
) -> list[str]:
    """Select patch ids for a train/val/test role from a manifest."""
    normalized_role = role.lower().strip()
    if normalized_role not in {"train", "val", "test"}:
        raise ValueError("role must be one of: train, val, test.")
    normalized_mode = mode.lower().strip()
    if normalized_mode not in {"holdout", "kfold"}:
        raise ValueError("mode must be either 'holdout' or 'kfold'.")

    if normalized_mode == "holdout":
        selected = manifest.loc[manifest["holdout_split"] == normalized_role, "patch_id"]
        return selected.astype(str).tolist()

    if normalized_role == "test":
        selected = manifest.loc[manifest["holdout_split"] == "test", "patch_id"]
        return selected.astype(str).tolist()

    available_folds = sorted(int(value) for value in manifest["fold"].dropna().unique())
    if fold_index not in available_folds:
        raise ValueError(f"fold_index {fold_index} is not present in the manifest.")
    non_test = manifest.loc[manifest["holdout_split"] != "test"].copy()
    if normalized_role == "val":
        selected = non_test.loc[non_test["fold"] == fold_index, "patch_id"]
    else:
        selected = non_test.loc[non_test["fold"] != fold_index, "patch_id"]
    return selected.astype(str).tolist()


def resolve_manifest_indices(
    patch_records: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    role: str,
    mode: str = "holdout",
    fold_index: int = 0,
) -> list[int]:
    """Map manifest patch ids back to dataset row indices."""
    if "patch_id" not in patch_records.columns:
        raise ValueError("patch_records must include a 'patch_id' column.")
    selected_ids = select_patch_ids_for_role(manifest, role=role, mode=mode, fold_index=fold_index)
    id_to_index = {str(patch_id): int(idx) for idx, patch_id in enumerate(patch_records["patch_id"].astype(str))}
    missing = [patch_id for patch_id in selected_ids if patch_id not in id_to_index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Split manifest contains patch ids not present in current dataset: {preview}")
    return sorted(id_to_index[patch_id] for patch_id in selected_ids)


def build_dataset_subsets(
    dataset: Dataset,
    patch_records: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    mode: str = "holdout",
    fold_index: int = 0,
) -> tuple[Subset, Subset, Subset]:
    """Create train/val/test subsets from a manifest and patch table."""
    train_indices = resolve_manifest_indices(patch_records, manifest, role="train", mode=mode, fold_index=fold_index)
    val_indices = resolve_manifest_indices(patch_records, manifest, role="val", mode=mode, fold_index=fold_index)
    test_indices = resolve_manifest_indices(patch_records, manifest, role="test", mode=mode, fold_index=fold_index)
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)
