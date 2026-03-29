"""Tests for the first MarsCLIP tri-modal model slice."""

from __future__ import annotations

import pathlib
import sys

import torch

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from marsclip_model import (
    MarsCLIPModel,
    compute_patch_valid_mask,
    sample_patch_keep_mask,
)


def _batch() -> dict[str, torch.Tensor]:
    valid_mask = torch.stack(
        [
            torch.tensor(
                [
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=torch.bool,
            ),
            torch.ones(8, 8, dtype=torch.bool),
        ]
    )
    return {
        "image": torch.rand(2, 3, 8, 8),
        "valid_mask": valid_mask,
        "input_ids": torch.tensor(
            [
                [2, 4, 5, 3, 0, 0],
                [2, 6, 7, 8, 3, 0],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0],
            ],
            dtype=torch.bool,
        ),
        "geo_features": torch.rand(2, 8),
        "scale_features": torch.rand(2, 8),
        "viewing_features": torch.rand(2, 13),
        "quality_features": torch.rand(2, 7),
    }


def test_compute_patch_valid_mask():
    valid_mask = _batch()["valid_mask"]
    patch_valid = compute_patch_valid_mask(valid_mask, patch_size=4)
    assert patch_valid.shape == (2, 4)
    assert patch_valid[0].tolist() == [True, False, False, False]
    assert patch_valid[1].tolist() == [True, True, True, True]


def test_sample_patch_keep_mask_never_selects_invalid_patches():
    patch_valid = torch.tensor([[True, False, True, False]], dtype=torch.bool)
    keep = sample_patch_keep_mask(patch_valid, mask_ratio=0.5, generator=torch.Generator().manual_seed(0))
    assert keep.shape == patch_valid.shape
    assert not bool((keep & ~patch_valid).any())
    assert keep.sum().item() >= 1


def test_model_forward_returns_embeddings_and_loss():
    batch = _batch()
    model = MarsCLIPModel(
        vocab_size=32,
        image_size=8,
        patch_size=4,
        hidden_dim=32,
        embed_dim=16,
        text_max_length=6,
        mask_ratio=0.5,
    )
    out = model(batch, generator=torch.Generator().manual_seed(0))

    assert out.image_embedding.shape == (2, 16)
    assert out.text_embedding.shape == (2, 16)
    assert out.geo_embedding.shape == (2, 16)
    assert out.patch_valid_mask.shape == (2, 4)
    assert out.patch_keep_mask.shape == (2, 4)
    assert torch.isfinite(out.loss)
    assert set(out.losses) == {"image_text", "image_geo", "text_geo"}
    assert not bool((out.patch_keep_mask & ~out.patch_valid_mask).any())
