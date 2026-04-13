"""Tests for cache-backed MarsCLIP patch datasets."""

from __future__ import annotations

import pathlib
import sys
import json

import torch

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from clip.marsclip_cache import CachedMarsCLIPPatchDataset, build_cached_split, cache_root_complete


def _make_samples(num_samples: int = 4, image_size: int = 8) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for idx in range(num_samples):
        samples.append(
            {
                "image": torch.full((3, image_size, image_size), float(idx + 1), dtype=torch.float32),
                "valid_mask": torch.ones(image_size, image_size, dtype=torch.bool),
                "metadata": {
                    "patch_id": f"patch_{idx}",
                    "obs_id": f"obs_{idx}",
                    "dominant_obs_id": f"obs_{idx}",
                    "product_id": f"prod_{idx}",
                    "overall_valid_fraction": 1.0,
                    "is_patch_valid": True,
                    "min_valid_fraction": 0.8,
                },
            }
        )
    return samples


def test_build_cached_split_round_trips_samples(tmp_path):
    out_dir = tmp_path / "train"
    build_cached_split(
        _make_samples(),
        out_dir=out_dir,
        split_name="train",
        num_workers=0,
        prefetch_factor=2,
        progress_interval=2,
        source_info={"kind": "unit-test"},
    )

    dataset = CachedMarsCLIPPatchDataset(out_dir)
    sample = dataset[2]
    progress = json.loads((out_dir / "cache_progress.json").read_text())

    assert len(dataset) == 4
    assert sample["image"].shape == (3, 8, 8)
    assert sample["image"].dtype == torch.float32
    assert sample["valid_mask"].dtype == torch.bool
    assert sample["metadata"]["patch_id"] == "patch_2"
    assert sample["metadata"]["is_patch_valid"] is True
    assert progress["status"] == "completed"
    assert progress["completed_samples"] == 4
    assert progress["kept_samples"] == 4
    assert progress["dropped_samples"] == 0


def test_build_cached_split_filters_invalid_samples(tmp_path):
    out_dir = tmp_path / "train"
    samples = _make_samples()
    samples[1]["metadata"]["is_patch_valid"] = False
    build_cached_split(
        samples,
        out_dir=out_dir,
        split_name="train",
        num_workers=0,
        require_patch_valid=True,
        source_info={"kind": "unit-test"},
    )

    dataset = CachedMarsCLIPPatchDataset(out_dir)
    progress = json.loads((out_dir / "cache_progress.json").read_text())

    assert len(dataset) == 3
    assert progress["kept_samples"] == 3
    assert progress["dropped_samples"] == 1


def test_cache_root_complete_requires_all_splits(tmp_path):
    for split in ("train", "val", "test"):
        split_dir = tmp_path / split
        split_dir.mkdir(parents=True)
        (split_dir / "_SUCCESS").write_text("")

    assert cache_root_complete(tmp_path) is True


def test_cache_root_complete_supports_custom_required_splits(tmp_path):
    for split in ("train", "val"):
        split_dir = tmp_path / split
        split_dir.mkdir(parents=True)
        (split_dir / "_SUCCESS").write_text("")

    assert cache_root_complete(tmp_path, required_splits=("train", "val")) is True
    assert cache_root_complete(tmp_path, required_splits=("train", "val", "test")) is False
