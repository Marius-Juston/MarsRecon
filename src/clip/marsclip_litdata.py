"""LitData helpers for MarsCLIP Stage A SatMAE training."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from clip.fb_mae_train_utils import collate_patch_samples_for_fb_mae

logger = logging.getLogger(__name__)

_METADATA_KEYS = (
    "patch_id",
    "obs_id",
    "dominant_obs_id",
    "product_id",
    "overall_valid_fraction",
    "is_patch_valid",
    "min_valid_fraction",
    "loaded_from_dominant_obs_only",
)


def get_marsclip_litdata_cache_key(
    *,
    root: str,
    bbox: tuple[float, float, float, float],
    patch_size_deg: float,
    image_size: int,
    split_manifest: str,
    patch_records_path: str | None,
    color_only: bool,
    dataset_normalize: bool,
    dataset_normalization_path: str | None,
    min_valid_fraction: float,
    dominant_obs_only: bool,
    filter_invalid_patches: bool,
    required_splits: tuple[str, ...],
) -> str:
    """Return a deterministic cache key for Mars image streaming inputs."""
    payload = {
        "root": root,
        "bbox": list(bbox),
        "patch_size_deg": float(patch_size_deg),
        "image_size": int(image_size),
        "split_manifest": split_manifest,
        "patch_records_path": patch_records_path,
        "color_only": bool(color_only),
        "dataset_normalize": bool(dataset_normalize),
        "dataset_normalization_path": dataset_normalization_path,
        "min_valid_fraction": float(min_valid_fraction),
        "dominant_obs_only": bool(dominant_obs_only),
        "filter_invalid_patches": bool(filter_invalid_patches),
        "required_splits": list(required_splits),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_litdata_symbols():
    try:
        from litdata import StreamingDataLoader, StreamingDataset, optimize
    except ImportError as exc:  # pragma: no cover - exercised by runtime env
        raise ImportError(
            "litdata is not installed. Install the `streaming` extra or run "
            "`.venv/bin/pip install litdata>=0.2.61`."
        ) from exc
    return StreamingDataset, StreamingDataLoader, optimize


class MarsStreamingPatchDataset(Dataset):
    """StreamingDataset wrapper that emits Mars SatMAE-ready patch samples."""

    def __init__(
        self,
        input_dir: str | Path,
        *,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        StreamingDataset, _, _ = _load_litdata_symbols()
        self._dataset = StreamingDataset(
            input_dir=str(input_dir),
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = self._dataset[index]
        metadata_json = raw["metadata_json"]
        if isinstance(metadata_json, bytes):
            metadata_json = metadata_json.decode("utf-8")
        metadata = json.loads(str(metadata_json))
        for key in _METADATA_KEYS:
            metadata.setdefault(key, None)
        metadata["patch_id"] = str(metadata["patch_id"])
        metadata["obs_id"] = None if metadata["obs_id"] is None else str(metadata["obs_id"])
        metadata["dominant_obs_id"] = (
            None if metadata["dominant_obs_id"] is None else str(metadata["dominant_obs_id"])
        )
        metadata["product_id"] = (
            None if metadata["product_id"] is None else str(metadata["product_id"])
        )
        metadata["overall_valid_fraction"] = float(metadata["overall_valid_fraction"] or 0.0)
        metadata["min_valid_fraction"] = float(metadata["min_valid_fraction"] or 0.0)
        metadata["is_patch_valid"] = bool(metadata["is_patch_valid"])
        metadata["loaded_from_dominant_obs_only"] = bool(
            metadata["loaded_from_dominant_obs_only"]
        )
        return {
            "image": torch.from_numpy(np.array(raw["image"], dtype=np.float32, copy=True)),
            "valid_mask": torch.from_numpy(np.array(raw["valid_mask"], dtype=np.bool_, copy=True)).to(
                torch.bool
            ),
            "metadata": metadata,
        }


def build_marsclip_litdata_dataloader(
    input_dir: str | Path,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    seed: int = 42,
) -> DataLoader:
    """Build a regular DataLoader over a LitData-backed StreamingDataset."""
    dataset = MarsStreamingPatchDataset(
        input_dir=input_dir,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=bool(num_workers > 0),
        collate_fn=collate_patch_samples_for_fb_mae,
    )
