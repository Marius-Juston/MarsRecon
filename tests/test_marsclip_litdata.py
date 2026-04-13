"""Tests for MarsCLIP LitData helpers."""

from __future__ import annotations

import pathlib
import sys
import types

import numpy as np
import pandas as pd
import torch

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from clip.marsclip_litdata import (
    MarsStreamingPatchDataset,
    build_marsclip_litdata_dataloader,
    get_marsclip_litdata_cache_key,
)
from clip.build_marsclip_litdata import _json_safe


class _FakeStreamingDataset:
    def __init__(self, *_, **__):
        self.items = [
            {
                "image": torch.ones(3, 4, 4, dtype=torch.float16).numpy(),
                "valid_mask": torch.ones(4, 4, dtype=torch.bool).numpy(),
                "metadata_json": (
                    '{"patch_id":"patch_0","obs_id":"obs_0","dominant_obs_id":"obs_0",'
                    '"product_id":"prod_0","overall_valid_fraction":1.0,"is_patch_valid":true,'
                    '"min_valid_fraction":0.8,"loaded_from_dominant_obs_only":true}'
                ),
            }
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def _fake_litdata_module():
    return types.SimpleNamespace(
        StreamingDataset=_FakeStreamingDataset,
        StreamingDataLoader=object,
        optimize=lambda **_: None,
    )


def test_litdata_cache_key_is_deterministic():
    kwargs = dict(
        root="/scratch/mars_hirise",
        bbox=(-136.0, 12.0, -124.0, 24.0),
        patch_size_deg=0.005,
        image_size=64,
        split_manifest="/tmp/manifest.csv",
        patch_records_path="/tmp/records.pkl",
        color_only=True,
        dataset_normalize=True,
        dataset_normalization_path="dataset_stats/image/dataset_stats.json",
        min_valid_fraction=0.8,
        dominant_obs_only=True,
        filter_invalid_patches=True,
        required_splits=("train", "val"),
    )
    assert get_marsclip_litdata_cache_key(**kwargs) == get_marsclip_litdata_cache_key(**kwargs)


def test_streaming_patch_dataset_decodes_fake_litdata(monkeypatch, tmp_path):
    fake_module = _fake_litdata_module()
    monkeypatch.setitem(sys.modules, "litdata", fake_module)

    dataset = MarsStreamingPatchDataset(tmp_path / "train", shuffle=False, drop_last=False, seed=0)
    sample = dataset[0]

    assert sample["image"].dtype == torch.float32
    assert sample["valid_mask"].dtype == torch.bool
    assert sample["metadata"]["patch_id"] == "patch_0"
    assert sample["metadata"]["loaded_from_dominant_obs_only"] is True


def test_litdata_dataloader_batches_fake_streaming_samples(monkeypatch, tmp_path):
    fake_module = _fake_litdata_module()
    monkeypatch.setitem(sys.modules, "litdata", fake_module)

    loader = build_marsclip_litdata_dataloader(
        tmp_path / "train",
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        seed=0,
    )
    batch = next(iter(loader))

    assert batch["image"].shape == (1, 3, 4, 4)
    assert batch["valid_mask"].shape == (1, 4, 4)
    assert batch["metadata"][0]["patch_id"] == "patch_0"


def test_json_safe_converts_tensor_and_timestamp_metadata():
    payload = {
        "scalar_tensor": torch.tensor(3.0),
        "vector_tensor": torch.tensor([1.0, 2.0]),
        "array": np.array([1, 2, 3]),
        "timestamp": pd.Timestamp("2026-04-13T04:00:00Z"),
    }

    converted = _json_safe(payload)

    assert converted["scalar_tensor"] == 3.0
    assert converted["vector_tensor"] == [1.0, 2.0]
    assert converted["array"] == [1, 2, 3]
    assert converted["timestamp"] == "2026-04-13T04:00:00+00:00"
