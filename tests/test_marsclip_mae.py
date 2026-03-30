"""Tests for the Stage A MarsCLIP masked autoencoder."""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from marsclip_mae import (
    MarsMaskedAutoencoder,
    build_masked_input_image,
    build_reconstruction_composite,
    collate_patch_samples_for_mae,
    compute_patch_valid_fraction,
    compute_valid_patch_mask,
    expand_patch_mask,
    extract_scale_values,
    patchify,
    sample_visible_patch_mask,
    scale_sinusoidal_encoding,
    unpatchify,
)


def test_patchify_returns_flattened_patch_vectors():
    image = torch.arange(1 * 3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    patches = patchify(image, patch_size=4)
    assert patches.shape == (1, 4, 48)


def test_unpatchify_round_trip_matches_original_image():
    image = torch.arange(1 * 3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    patches = patchify(image, patch_size=4)
    rebuilt = unpatchify(patches, patch_size=4, channels=3, image_size=8)

    assert rebuilt.shape == image.shape
    assert torch.allclose(rebuilt, image)


def test_expand_patch_mask_returns_image_shaped_mask():
    patch_mask = torch.tensor([[True, False, False, True]], dtype=torch.bool)

    expanded = expand_patch_mask(patch_mask, patch_size=2, image_size=4)

    assert expanded.shape == (1, 4, 4)
    assert expanded[0, :2, :2].all()
    assert not expanded[0, :2, 2:].any()
    assert expanded[0, 2:, 2:].all()


def test_compute_patch_valid_fraction_and_mask_use_threshold():
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
    fractions = compute_patch_valid_fraction(valid_mask, patch_size=2)
    assert torch.allclose(fractions, torch.tensor([[0.75, 0.0, 0.0, 1.0]]))

    patch_valid = compute_valid_patch_mask(valid_mask, patch_size=2, min_valid_fraction=0.5)
    assert patch_valid.tolist() == [[True, False, False, True]]


def test_extract_scale_values_uses_first_scale_feature():
    scale_features = torch.tensor(
        [
            [0.25, 1000.0, 0.005],
            [0.50, 1000.0, 0.005],
        ],
        dtype=torch.float32,
    )

    scale_values = extract_scale_values(scale_features)

    assert scale_values.tolist() == pytest.approx([0.25, 0.50])


def test_collate_patch_samples_for_mae_stacks_stage_a_contract():
    samples = [
        {
            "image": torch.ones(3, 8, 8, dtype=torch.float32),
            "valid_mask": torch.ones(8, 8, dtype=torch.bool),
            "scale_features": torch.tensor([0.25, 1000.0, 0.005], dtype=torch.float32),
            "rationale_raw": "Patch A",
            "rationale_expanded": None,
            "metadata": {"patch_id": "patch_a"},
        },
        {
            "image": torch.zeros(3, 8, 8, dtype=torch.float32),
            "valid_mask": torch.zeros(8, 8, dtype=torch.bool),
            "scale_features": torch.tensor([0.50, 1000.0, 0.005], dtype=torch.float32),
            "rationale_raw": "Patch B",
            "rationale_expanded": "Expanded B",
            "metadata": {"patch_id": "patch_b"},
        },
    ]

    batch = collate_patch_samples_for_mae(samples)

    assert batch["image"].shape == (2, 3, 8, 8)
    assert batch["valid_mask"].shape == (2, 8, 8)
    assert batch["scale_features"].shape == (2, 3)
    assert batch["scale_values"].tolist() == pytest.approx([0.25, 0.50])
    assert batch["rationale_raw"] == ["Patch A", "Patch B"]
    assert batch["metadata"][1]["patch_id"] == "patch_b"


def test_build_masked_input_image_zeroes_masked_and_invalid_patches():
    image = torch.ones(1, 3, 4, 4, dtype=torch.float32)
    patch_valid_mask = torch.tensor([[True, True, False, True]], dtype=torch.bool)
    visible_mask = torch.tensor([[True, False, False, False]], dtype=torch.bool)

    masked = build_masked_input_image(
        image,
        patch_valid_mask,
        visible_mask,
        patch_size=2,
    )

    assert masked.shape == image.shape
    assert masked[:, :, :2, :2].gt(0).all()
    assert masked[:, :, :2, 2:].eq(0).all()
    assert masked[:, :, 2:, :2].eq(0).all()


def test_build_reconstruction_composite_uses_predictions_for_masked_valid_patches():
    image = torch.zeros(1, 3, 4, 4, dtype=torch.float32)
    reconstruction_patches = torch.ones(1, 4, 12, dtype=torch.float32)
    patch_valid_mask = torch.tensor([[True, True, False, True]], dtype=torch.bool)
    visible_mask = torch.tensor([[True, False, False, False]], dtype=torch.bool)

    reconstructed, composite = build_reconstruction_composite(
        image,
        reconstruction_patches,
        patch_valid_mask,
        visible_mask,
        patch_size=2,
    )

    assert reconstructed.shape == image.shape
    assert composite.shape == image.shape
    assert composite[:, :, :2, :2].eq(0).all()
    assert composite[:, :, :2, 2:].eq(1).all()
    assert composite[:, :, 2:, :2].eq(0).all()


def test_sample_visible_patch_mask_never_selects_invalid_patches():
    patch_valid_mask = torch.tensor([[True, False, True, True]], dtype=torch.bool)
    visible = sample_visible_patch_mask(
        patch_valid_mask,
        mask_ratio=0.5,
        generator=torch.Generator().manual_seed(0),
    )
    assert visible.shape == patch_valid_mask.shape
    assert not bool((visible & ~patch_valid_mask).any())
    assert visible.sum().item() >= 1


def test_scale_sinusoidal_encoding_shape_and_repeatability():
    scales = torch.tensor([0.25, 0.5], dtype=torch.float32)
    enc_a = scale_sinusoidal_encoding(scales, dim=8)
    enc_b = scale_sinusoidal_encoding(scales, dim=8)

    assert enc_a.shape == (2, 8)
    assert torch.allclose(enc_a, enc_b)


def test_mae_forward_returns_shapes_and_finite_loss():
    image = torch.rand(2, 3, 8, 8)
    valid_mask = torch.tensor(
        [
            [
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
            ],
            torch.ones(8, 8, dtype=torch.bool),
        ],
        dtype=torch.bool,
    )
    model = MarsMaskedAutoencoder(
        image_size=8,
        patch_size=4,
        encoder_dim=32,
        encoder_depth=2,
        encoder_heads=4,
        decoder_dim=16,
        decoder_depth=1,
        decoder_heads=4,
        min_valid_fraction=0.5,
    )

    out = model(
        image,
        valid_mask,
        scale_values=torch.tensor([0.5, 0.5]),
        mask_ratio=0.5,
        generator=torch.Generator().manual_seed(0),
    )

    assert out.encoded_tokens.shape == (2, 4, 32)
    assert out.pooled_embedding.shape == (2, 32)
    assert out.patch_valid_fraction.shape == (2, 4)
    assert out.patch_valid_mask.shape == (2, 4)
    assert out.visible_mask.shape == (2, 4)
    assert out.masked_valid_mask.shape == (2, 4)
    assert out.valid_pixel_mask.shape == (2, 4, 48)
    assert out.loss_mask.shape == (2, 4, 48)
    assert out.reconstruction.shape == (2, 4, 48)
    assert torch.isfinite(out.loss)
    assert not bool((out.visible_mask & ~out.patch_valid_mask).any())
    assert not bool((out.loss_mask & ~out.valid_pixel_mask).any())


def test_mae_forward_patch_batch_uses_stage_a1_inputs():
    samples = [
        {
            "image": torch.rand(3, 8, 8),
            "valid_mask": torch.ones(8, 8, dtype=torch.bool),
            "scale_features": torch.tensor([0.25, 1000.0, 0.005], dtype=torch.float32),
            "rationale_raw": "Patch A",
            "rationale_expanded": None,
            "metadata": {"patch_id": "patch_a"},
        },
        {
            "image": torch.rand(3, 8, 8),
            "valid_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                ],
                dtype=torch.bool,
            ),
            "scale_features": torch.tensor([0.50, 1000.0, 0.005], dtype=torch.float32),
            "rationale_raw": "Patch B",
            "rationale_expanded": None,
            "metadata": {"patch_id": "patch_b"},
        },
    ]
    batch = collate_patch_samples_for_mae(samples)
    model = MarsMaskedAutoencoder(
        image_size=8,
        patch_size=4,
        encoder_dim=32,
        encoder_depth=2,
        encoder_heads=4,
        decoder_dim=16,
        decoder_depth=1,
        decoder_heads=4,
        min_valid_fraction=0.5,
    )

    out = model.forward_patch_batch(
        batch,
        mask_ratio=0.5,
        generator=torch.Generator().manual_seed(0),
    )

    assert out.encoded_tokens.shape == (2, 4, 32)
    assert out.patch_valid_mask.shape == (2, 4)
    assert torch.isfinite(out.loss)


def test_mae_backward_smoke():
    image = torch.rand(2, 3, 8, 8)
    valid_mask = torch.ones(2, 8, 8, dtype=torch.bool)
    model = MarsMaskedAutoencoder(
        image_size=8,
        patch_size=4,
        encoder_dim=32,
        encoder_depth=2,
        encoder_heads=4,
        decoder_dim=16,
        decoder_depth=1,
        decoder_heads=4,
    )

    out = model(
        image,
        valid_mask,
        scale_values=torch.tensor([0.5, 0.5]),
        mask_ratio=0.75,
        generator=torch.Generator().manual_seed(0),
    )
    out.loss.backward()

    total_grad = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_grad += float(param.grad.abs().sum())
    assert total_grad > 0.0


def test_scale_encoding_requires_even_dimension():
    with pytest.raises(ValueError, match="dim must be even"):
        _ = scale_sinusoidal_encoding(torch.tensor([0.5]), dim=7)


@pytest.mark.integration
def test_mae_real_patch_batch_forward_backward_smoke():
    from mars_hirise import MarsHiRISE
    from marsclip_patches import (
        MarsCLIPPatchDataset,
        build_patch_observation_metadata,
        build_patch_records,
    )

    geo = MarsHiRISE(
        bbox=(-136.0, 12.0, -124.0, 24.0),
        channels=["NEAR-INFRARED", "RED", "BLUE-GREEN"],
        download=False,
    )
    observation_metadata = build_patch_observation_metadata(geo)

    first_geom = geo.index.geometry.iloc[0]
    minx, miny, maxx, maxy = first_geom.bounds
    center_x = (minx + maxx) / 2.0
    center_y = (miny + maxy) / 2.0
    interval = geo.index.index[0]

    patch_records = build_patch_records(
        geo,
        size=0.005,
        centers=[(center_x, center_y, interval)],
        observation_metadata=observation_metadata,
    )
    ds = MarsCLIPPatchDataset(
        geo_dataset=geo,
        patch_records=patch_records,
        observation_metadata=observation_metadata,
        image_size=64,
    )
    batch = collate_patch_samples_for_mae([ds[0]])

    model = MarsMaskedAutoencoder(
        image_size=64,
        patch_size=16,
        encoder_dim=64,
        encoder_depth=2,
        encoder_heads=4,
        decoder_dim=32,
        decoder_depth=1,
        decoder_heads=4,
    )

    out = model.forward_patch_batch(
        batch,
        mask_ratio=0.75,
        generator=torch.Generator().manual_seed(0),
    )
    out.loss.backward()

    total_grad = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_grad += float(param.grad.abs().sum())

    assert torch.isfinite(out.loss)
    assert total_grad > 0.0
