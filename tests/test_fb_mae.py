"""Tests for the Facebook-style Mars MAE implementation."""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from clip.fb_mae import (
    FacebookMaskedAutoencoderViT,
    compute_valid_patch_mask,
    normalize_patch_targets,
    patchify,
    patchify_valid_mask,
)


def test_patchify_returns_flattened_patch_vectors():
    image = torch.arange(1 * 3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    patches = patchify(image, patch_size=4)
    assert patches.shape == (1, 4, 48)


def test_compute_valid_patch_mask_uses_fraction_threshold():
    valid_mask = torch.tensor(
        [
            [
                [1, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ]
        ],
        dtype=torch.bool,
    )

    patch_valid = compute_valid_patch_mask(valid_mask, patch_size=2, min_valid_fraction=0.5)

    assert patch_valid.tolist() == [[True, False, False, True]]


def test_normalize_patch_targets_is_per_patch_and_preserves_invalid_zeros():
    target_patches = torch.tensor(
        [
            [
                [1.0, 3.0, 100.0, 200.0],
                [10.0, 14.0, 18.0, 22.0],
            ]
        ]
    )
    valid_pixel_mask = torch.tensor(
        [
            [
                [True, True, False, False],
                [True, True, True, True],
            ]
        ]
    )

    normalized = normalize_patch_targets(target_patches, valid_pixel_mask)

    assert normalized[0, 0, 2:].eq(0).all()
    first_patch_valid = normalized[0, 0, :2]
    second_patch_valid = normalized[0, 1]
    assert float(first_patch_valid.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(first_patch_valid.std(unbiased=False)) == pytest.approx(1.0, rel=1e-5)
    assert float(second_patch_valid.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(second_patch_valid.std(unbiased=False)) == pytest.approx(1.0, rel=1e-5)


def test_fb_mae_forward_returns_shapes_and_finite_loss():
    model = FacebookMaskedAutoencoderViT(
        image_size=16,
        patch_size=8,
        in_chans=3,
        embed_dim=32,
        depth=2,
        num_heads=4,
        decoder_embed_dim=16,
        decoder_depth=1,
        decoder_num_heads=4,
        norm_pix_loss=True,
    )
    image = torch.rand(2, 3, 16, 16)
    valid_mask = torch.ones(2, 16, 16, dtype=torch.bool)
    valid_mask[1, 8:, :8] = False

    output = model(
        image,
        valid_mask,
        mask_ratio=0.5,
        generator=torch.Generator().manual_seed(0),
    )

    assert output.pred.shape == (2, 4, 3 * 8 * 8)
    assert output.mask.shape == (2, 4)
    assert output.patch_valid_mask.shape == (2, 4)
    assert output.valid_pixel_mask.shape == (2, 4, 3 * 8 * 8)
    assert output.patch_valid_mask[1].tolist() == [True, True, False, True]
    assert torch.isfinite(output.loss)


def test_patchify_valid_mask_matches_flat_patch_size():
    valid_mask = torch.tensor(
        [[[True, False], [True, True]]],
        dtype=torch.bool,
    )

    patch_mask = patchify_valid_mask(valid_mask, patch_size=2)

    assert patch_mask.shape == (1, 1, 4)
    assert patch_mask[0, 0].tolist() == [True, False, True, True]
