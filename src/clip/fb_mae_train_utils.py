"""Shared training utilities for the Facebook-style Mars MAE path."""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


def collate_patch_samples_for_fb_mae(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate Mars patch samples into the minimal batch needed by fb_mae."""
    if not samples:
        raise ValueError("samples must not be empty.")

    return {
        "image": torch.stack([sample["image"] for sample in samples], dim=0),
        "valid_mask": torch.stack([sample["valid_mask"] for sample in samples], dim=0),
        "metadata": [dict(sample.get("metadata", {})) for sample in samples],
    }


def build_fb_mae_dataloader(
    dataset: Dataset | list[dict[str, Any]],
    *,
    batch_size: int = 4,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Build a DataLoader that emits Facebook-MAE-ready patch batches."""
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "generator": generator,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_patch_samples_for_fb_mae,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts for a model."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable, total


def save_training_history(
    history: list[dict[str, float]],
    out_path: pathlib.Path | str,
) -> pathlib.Path:
    """Persist training history as JSON."""
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(history, indent=2))
    return out


def _progress_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def save_training_progress(
    progress: dict[str, Any],
    out_path: pathlib.Path | str,
) -> pathlib.Path:
    """Persist an incremental training progress snapshot as JSON."""
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(progress)
    payload["updated_at"] = _progress_timestamp()
    out.write_text(json.dumps(payload, indent=2))
    return out
