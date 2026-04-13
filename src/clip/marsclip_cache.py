"""Cache-backed MarsCLIP patch datasets for faster Stage A training."""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from clip.marsclip_splits import build_dataset_subsets, load_patch_split_manifest

CACHE_IMAGE_FILENAME = "images.npy"
CACHE_VALID_MASK_FILENAME = "valid_masks.npy"
CACHE_METADATA_FILENAME = "metadata.csv"
CACHE_INFO_FILENAME = "cache_info.json"
CACHE_PROGRESS_FILENAME = "cache_progress.json"
CACHE_SUCCESS_FILENAME = "_SUCCESS"
_PARTIAL_IMAGE_FILENAME = "_partial_images.npy"
_PARTIAL_VALID_MASK_FILENAME = "_partial_valid_masks.npy"

_METADATA_FIELDS = (
    "patch_id",
    "obs_id",
    "dominant_obs_id",
    "product_id",
    "overall_valid_fraction",
    "is_patch_valid",
    "min_valid_fraction",
)

def _sample_metadata_row(sample: dict[str, Any], index: int) -> dict[str, Any]:
    metadata = dict(sample.get("metadata", {}))
    row: dict[str, Any] = {"cache_index": int(index)}
    for field in _METADATA_FIELDS:
        value = metadata.get(field)
        if field == "patch_id" and value is None:
            value = f"item_{index}"
        row[field] = value
    row["patch_id"] = str(row["patch_id"])
    row["obs_id"] = None if row["obs_id"] is None else str(row["obs_id"])
    row["dominant_obs_id"] = (
        None if row["dominant_obs_id"] is None else str(row["dominant_obs_id"])
    )
    row["product_id"] = None if row["product_id"] is None else str(row["product_id"])
    row["overall_valid_fraction"] = float(row["overall_valid_fraction"] or 0.0)
    row["is_patch_valid"] = bool(row["is_patch_valid"])
    row["min_valid_fraction"] = float(row["min_valid_fraction"] or 0.0)
    return row


def _extract_sample(dataset: Dataset, index: int) -> dict[str, Any]:
    sample = dataset[int(index)]
    return {
        "image": sample["image"].detach().cpu().numpy().astype(np.float16, copy=False),
        "valid_mask": sample["valid_mask"].detach().cpu().numpy().astype(np.bool_, copy=False),
        "metadata": _sample_metadata_row(sample, int(index)),
    }


def _write_progress(
    out_path: pathlib.Path,
    *,
    split_name: str,
    num_samples: int,
    completed_samples: int,
    status: str,
    kept_samples: int | None = None,
    dropped_samples: int | None = None,
) -> None:
    payload = {
        "split_name": split_name,
        "num_samples": int(num_samples),
        "completed_samples": int(completed_samples),
        "status": status,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "percent_complete": float(completed_samples) / float(num_samples) if num_samples else 0.0,
    }
    if kept_samples is not None:
        payload["kept_samples"] = int(kept_samples)
    if dropped_samples is not None:
        payload["dropped_samples"] = int(dropped_samples)
    (out_path / CACHE_PROGRESS_FILENAME).write_text(json.dumps(payload, indent=2))


def _copy_partial_cache(
    *,
    source_path: pathlib.Path,
    target_path: pathlib.Path,
    kept_count: int,
    sample_shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> None:
    source = np.load(source_path, mmap_mode="r")
    target = np.lib.format.open_memmap(
        target_path,
        mode="w+",
        dtype=dtype,
        shape=(kept_count, *sample_shape),
    )
    chunk_size = 1024
    for start in range(0, kept_count, chunk_size):
        stop = min(start + chunk_size, kept_count)
        target[start:stop] = source[start:stop]
    target.flush()


def build_cached_split(
    dataset: Dataset,
    *,
    out_dir: pathlib.Path | str,
    split_name: str,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    progress_interval: int = 250,
    require_patch_valid: bool = True,
    source_info: dict[str, Any] | None = None,
) -> pathlib.Path:
    """Write a cache-backed split to memmap `.npy` files plus lightweight metadata."""
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    success_marker = out_path / CACHE_SUCCESS_FILENAME
    if success_marker.exists():
        return out_path

    dataset_len = len(dataset)
    if dataset_len <= 0:
        raise ValueError(f"Cannot cache empty split: {split_name}")

    if int(num_workers) > 0:
        print(
            f"[cache:{split_name}] Requested num_workers={num_workers}, but cache extraction is "
            "running in a single process for reliability in this environment."
        )
    if int(prefetch_factor) > 0 and int(num_workers) > 0:
        print(
            f"[cache:{split_name}] Ignoring prefetch_factor={prefetch_factor} during single-process "
            "cache extraction."
        )

    _write_progress(
        out_path,
        split_name=split_name,
        num_samples=dataset_len,
        completed_samples=0,
        status="running",
    )

    first_sample = _extract_sample(dataset, 0)
    first_image = np.asarray(first_sample["image"])
    first_mask = np.asarray(first_sample["valid_mask"])
    partial_image_path = out_path / _PARTIAL_IMAGE_FILENAME
    partial_valid_mask_path = out_path / _PARTIAL_VALID_MASK_FILENAME

    partial_images = np.lib.format.open_memmap(
        partial_image_path,
        mode="w+",
        dtype=np.float16,
        shape=(dataset_len, *first_image.shape),
    )
    partial_valid_masks = np.lib.format.open_memmap(
        partial_valid_mask_path,
        mode="w+",
        dtype=np.bool_,
        shape=(dataset_len, *first_mask.shape),
    )

    metadata_rows: list[dict[str, Any]] = []
    kept_count = 0
    dropped_count = 0
    progress_every = max(1, int(progress_interval))

    for sample_index in range(dataset_len):
        sample = first_sample if sample_index == 0 else _extract_sample(dataset, sample_index)
        keep_sample = bool(sample["metadata"].get("is_patch_valid", True)) or not require_patch_valid
        if keep_sample:
            partial_images[kept_count] = np.asarray(sample["image"])
            partial_valid_masks[kept_count] = np.asarray(sample["valid_mask"])
            metadata_rows.append(dict(sample["metadata"]))
            kept_count += 1
        else:
            dropped_count += 1
        completed_samples = sample_index + 1
        if completed_samples % progress_every == 0 or completed_samples == dataset_len:
            _write_progress(
                out_path,
                split_name=split_name,
                num_samples=dataset_len,
                completed_samples=completed_samples,
                status="running",
                kept_samples=kept_count,
                dropped_samples=dropped_count,
            )

    partial_images.flush()
    partial_valid_masks.flush()

    if kept_count <= 0:
        raise ValueError(f"Cache split {split_name} contains no valid samples after filtering.")

    _copy_partial_cache(
        source_path=partial_image_path,
        target_path=out_path / CACHE_IMAGE_FILENAME,
        kept_count=kept_count,
        sample_shape=tuple(first_image.shape),
        dtype=np.dtype(np.float16),
    )
    _copy_partial_cache(
        source_path=partial_valid_mask_path,
        target_path=out_path / CACHE_VALID_MASK_FILENAME,
        kept_count=kept_count,
        sample_shape=tuple(first_mask.shape),
        dtype=np.dtype(np.bool_),
    )
    partial_image_path.unlink(missing_ok=True)
    partial_valid_mask_path.unlink(missing_ok=True)

    metadata_frame = pd.DataFrame.from_records(metadata_rows)
    metadata_frame.to_csv(out_path / CACHE_METADATA_FILENAME, index=False)

    cache_info = {
        "split_name": split_name,
        "num_source_samples": int(dataset_len),
        "num_samples": int(kept_count),
        "num_dropped_samples": int(dropped_count),
        "image_shape": list(first_image.shape),
        "image_dtype": "float16",
        "valid_mask_dtype": "bool",
        "require_patch_valid": bool(require_patch_valid),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_info": dict(source_info or {}),
    }
    (out_path / CACHE_INFO_FILENAME).write_text(json.dumps(cache_info, indent=2))
    _write_progress(
        out_path,
        split_name=split_name,
        num_samples=dataset_len,
        completed_samples=dataset_len,
        status="completed",
        kept_samples=kept_count,
        dropped_samples=dropped_count,
    )
    success_marker.write_text("")
    return out_path


def build_cached_splits_from_manifest(
    dataset: Dataset,
    patch_records: pd.DataFrame,
    *,
    manifest_path: pathlib.Path | str,
    out_root: pathlib.Path | str,
    mode: str = "holdout",
    fold_index: int = 0,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    progress_interval: int = 250,
    require_patch_valid: bool = True,
    required_splits: tuple[str, ...] = ("train", "val", "test"),
    source_info: dict[str, Any] | None = None,
) -> pathlib.Path:
    """Build train/val/test caches using an existing split manifest."""
    out_path = pathlib.Path(out_root)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest = load_patch_split_manifest(manifest_path)
    train_dataset, val_dataset, test_dataset = build_dataset_subsets(
        dataset,
        patch_records,
        manifest,
        mode=mode,
        fold_index=fold_index,
    )

    split_lookup = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    for split_name in required_splits:
        split_dataset = split_lookup[split_name]
        build_cached_split(
            split_dataset,
            out_dir=out_path / split_name,
            split_name=split_name,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            progress_interval=progress_interval,
            require_patch_valid=require_patch_valid,
            source_info=source_info,
        )

    summary = {
        "manifest_path": str(manifest_path),
        "mode": mode,
        "fold_index": int(fold_index),
        "requested_splits": list(required_splits),
        "train_source_count": int(len(train_dataset)),
        "val_source_count": int(len(val_dataset)),
        "test_source_count": int(len(test_dataset)),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_info": dict(source_info or {}),
    }
    for split_name in required_splits:
        split_info = json.loads((out_path / split_name / CACHE_INFO_FILENAME).read_text())
        summary[f"{split_name}_count"] = int(split_info["num_samples"])
        summary[f"{split_name}_dropped_count"] = int(split_info["num_dropped_samples"])
    (out_path / "cache_summary.json").write_text(json.dumps(summary, indent=2))
    return out_path


class CachedMarsCLIPPatchDataset(Dataset):
    """Memory-mapped cache-backed Mars patch dataset."""

    def __init__(self, root: pathlib.Path | str) -> None:
        self.root = pathlib.Path(root)
        success_marker = self.root / CACHE_SUCCESS_FILENAME
        if not success_marker.exists():
            raise FileNotFoundError(f"Cache split is incomplete: {self.root}")

        self.images = np.load(self.root / CACHE_IMAGE_FILENAME, mmap_mode="r")
        self.valid_masks = np.load(self.root / CACHE_VALID_MASK_FILENAME, mmap_mode="r")
        self.metadata = pd.read_csv(self.root / CACHE_METADATA_FILENAME)
        self.metadata["patch_id"] = self.metadata["patch_id"].astype(str)
        self.metadata["overall_valid_fraction"] = self.metadata["overall_valid_fraction"].astype(float)
        self.metadata["min_valid_fraction"] = self.metadata["min_valid_fraction"].astype(float)
        self.metadata["is_patch_valid"] = self.metadata["is_patch_valid"].map(
            lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        )
        self.info = json.loads((self.root / CACHE_INFO_FILENAME).read_text())

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = torch.from_numpy(np.array(self.images[index], copy=True)).to(torch.float32)
        valid_mask = torch.from_numpy(np.array(self.valid_masks[index], copy=True)).to(torch.bool)
        metadata_row = self.metadata.iloc[int(index)].to_dict()
        metadata_row["patch_id"] = str(metadata_row["patch_id"])
        metadata_row["overall_valid_fraction"] = float(metadata_row["overall_valid_fraction"])
        metadata_row["min_valid_fraction"] = float(metadata_row["min_valid_fraction"])
        metadata_row["is_patch_valid"] = bool(metadata_row["is_patch_valid"])
        return {
            "image": image,
            "valid_mask": valid_mask,
            "metadata": metadata_row,
        }


def cache_root_complete(
    root: pathlib.Path | str,
    *,
    required_splits: tuple[str, ...] = ("train", "val", "test"),
) -> bool:
    """Return True when train/val/test cache markers are present."""
    base = pathlib.Path(root)
    return all((base / split / CACHE_SUCCESS_FILENAME).exists() for split in required_splits)
