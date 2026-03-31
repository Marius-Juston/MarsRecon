"""Tests for the Stage A MarsCLIP masked autoencoder."""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from clip.marsclip_mae import (
    MarsMaskedAutoencoder,
    _coerce_channel_stats,
    _gather_visible_tokens,
    build_masked_input_image,
    build_reconstruction_composite,
    collate_patch_samples_for_mae,
    compute_patch_valid_fraction,
    compute_valid_patch_mask,
    denormalize_patch_tokens,
    expand_patch_mask,
    extract_scale_values,
    normalize_valid_image,
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


def test_normalize_valid_image_preserves_invalid_pixels_as_zero():
    image = torch.tensor(
        [[[[2.0, 4.0], [6.0, 8.0]], [[10.0, 12.0], [14.0, 16.0]]]],
        dtype=torch.float32,
    )
    valid_mask = torch.tensor([[[True, False], [True, False]]], dtype=torch.bool)

    normalized = normalize_valid_image(
        image,
        valid_mask,
        channel_mean=[4.0, 12.0],
        channel_std=[2.0, 2.0],
    )

    assert normalized[0, 0, 0, 0] == pytest.approx(-1.0)
    assert normalized[0, 1, 1, 0] == pytest.approx(1.0)
    assert normalized[0, :, 0, 1].eq(0).all()
    assert normalized[0, :, 1, 1].eq(0).all()


def test_denormalize_patch_tokens_restores_original_scale():
    normalized_patches = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]], dtype=torch.float32)

    denormalized = denormalize_patch_tokens(
        normalized_patches,
        patch_size=1,
        channels=4,
        channel_mean=[1.0, 2.0, 3.0, 4.0],
        channel_std=[2.0, 2.0, 2.0, 2.0],
    )

    expected = torch.tensor([[[1.0, 4.0, 7.0, 10.0]]], dtype=torch.float32)
    assert torch.allclose(denormalized, expected)


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


def test_mae_forward_supports_optional_input_and_target_normalization():
    image = torch.tensor(
        [
            [
                [[0.1, 0.2, 0.0, 0.0], [0.3, 0.4, 0.0, 0.0], [0.0, 0.0, 0.5, 0.6], [0.0, 0.0, 0.7, 0.8]],
                [[0.2, 0.3, 0.0, 0.0], [0.4, 0.5, 0.0, 0.0], [0.0, 0.0, 0.6, 0.7], [0.0, 0.0, 0.8, 0.9]],
                [[0.3, 0.4, 0.0, 0.0], [0.5, 0.6, 0.0, 0.0], [0.0, 0.0, 0.7, 0.8], [0.0, 0.0, 0.9, 1.0]],
            ]
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.tensor(
        [[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]],
        dtype=torch.bool,
    )
    model = MarsMaskedAutoencoder(
        image_size=4,
        patch_size=2,
        encoder_dim=16,
        encoder_depth=1,
        encoder_heads=4,
        decoder_dim=8,
        decoder_depth=1,
        decoder_heads=4,
        normalize_inputs=True,
        normalize_targets=True,
        input_mean=[0.45, 0.55, 0.65],
        input_std=[0.2, 0.2, 0.2],
    )

    out = model(
        image,
        valid_mask,
        scale_values=torch.tensor([0.5], dtype=torch.float32),
        mask_ratio=0.5,
        generator=torch.Generator().manual_seed(0),
    )

    assert torch.isfinite(out.loss)
    assert out.reconstruction.shape == (1, 4, 12)


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
    from dataset.mars_hirise import MarsHiRISE
    from clip.marsclip_patches import (
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


class TestValidationErrors:
    def test_patchify_requires_4d_tensor(self):
        with pytest.raises(ValueError, match="B, C, H, W"):
            patchify(torch.randn(3, 8, 8), patch_size=4)

    def test_patchify_requires_divisible_dimensions(self):
        with pytest.raises(ValueError, match="divisible by patch_size"):
            patchify(torch.randn(1, 3, 9, 8), patch_size=4)

    def test_unpatchify_requires_3d_patches(self):
        with pytest.raises(ValueError, match="B, N, D"):
            unpatchify(torch.randn(4, 48), patch_size=4, channels=3)

    def test_unpatchify_requires_matching_patch_dim(self):
        with pytest.raises(ValueError, match="does not match channels"):
            unpatchify(torch.randn(1, 4, 10), patch_size=4, channels=3)

    def test_unpatchify_requires_square_grid_without_image_size(self):
        with pytest.raises(ValueError, match="square grid"):
            unpatchify(torch.randn(1, 5, 48), patch_size=4, channels=3)

    def test_unpatchify_requires_divisible_image_size(self):
        with pytest.raises(ValueError, match="divisible by patch_size"):
            unpatchify(torch.randn(1, 4, 48), patch_size=4, channels=3, image_size=9)

    def test_unpatchify_requires_matching_image_size_and_num_patches(self):
        with pytest.raises(ValueError, match="does not match the number of patches"):
            unpatchify(torch.randn(1, 4, 48), patch_size=4, channels=3, image_size=12)

    def test_expand_patch_mask_requires_2d_mask(self):
        with pytest.raises(ValueError, match="B, N"):
            expand_patch_mask(torch.ones(2, 4, 4, dtype=torch.bool), patch_size=4)

    def test_expand_patch_mask_requires_square_grid_without_image_size(self):
        with pytest.raises(ValueError, match="square grid"):
            expand_patch_mask(torch.ones(1, 5, dtype=torch.bool), patch_size=4)

    def test_expand_patch_mask_requires_divisible_image_size(self):
        with pytest.raises(ValueError, match="divisible by patch_size"):
            expand_patch_mask(torch.ones(1, 4, dtype=torch.bool), patch_size=4, image_size=9)

    def test_expand_patch_mask_requires_matching_image_size_and_patches(self):
        with pytest.raises(ValueError, match="does not match the number of patches"):
            expand_patch_mask(torch.ones(1, 4, dtype=torch.bool), patch_size=4, image_size=12)

    def test_compute_patch_valid_fraction_requires_3d_mask(self):
        with pytest.raises(ValueError, match="B, H, W"):
            compute_patch_valid_fraction(torch.ones(8, 8, dtype=torch.bool), patch_size=4)

    def test_compute_patch_valid_fraction_requires_divisible_dimensions(self):
        with pytest.raises(ValueError, match="divisible by patch_size"):
            compute_patch_valid_fraction(torch.ones(1, 9, 8, dtype=torch.bool), patch_size=4)

    def test_compute_valid_patch_mask_requires_valid_fraction_range(self):
        with pytest.raises(ValueError, match="0 <= value <= 1"):
            compute_valid_patch_mask(torch.ones(1, 8, 8, dtype=torch.bool), patch_size=4, min_valid_fraction=-0.1)

    def test_extract_scale_values_requires_2d_tensor(self):
        with pytest.raises(ValueError, match="B, F"):
            extract_scale_values(torch.randn(2, 4, 4))

    def test_extract_scale_values_requires_at_least_one_column(self):
        with pytest.raises(ValueError, match="at least one column"):
            extract_scale_values(torch.randn(2, 0))

    def test_coerce_channel_stats_raises_on_wrong_count(self):
        with pytest.raises(ValueError, match="Expected 3 channel stats"):
            _coerce_channel_stats([1.0, 2.0], channels=3, default=0.0)

    def test_normalize_valid_image_requires_4d_image(self):
        with pytest.raises(ValueError, match="B, C, H, W"):
            normalize_valid_image(
                torch.randn(3, 8, 8),
                torch.ones(1, 8, 8, dtype=torch.bool),
                channel_mean=[0.0, 0.0, 0.0],
                channel_std=[1.0, 1.0, 1.0],
            )

    def test_normalize_valid_image_requires_3d_mask(self):
        with pytest.raises(ValueError, match="B, H, W"):
            normalize_valid_image(
                torch.randn(1, 3, 8, 8),
                torch.ones(8, 8, dtype=torch.bool),
                channel_mean=[0.0, 0.0, 0.0],
                channel_std=[1.0, 1.0, 1.0],
            )

    def test_normalize_valid_image_requires_matching_batch_and_spatial(self):
        with pytest.raises(ValueError, match="batch/spatial dimensions"):
            normalize_valid_image(
                torch.randn(2, 3, 8, 8),
                torch.ones(1, 8, 8, dtype=torch.bool),
                channel_mean=[0.0, 0.0, 0.0],
                channel_std=[1.0, 1.0, 1.0],
            )

    def test_denormalize_patch_tokens_requires_3d_patches(self):
        with pytest.raises(ValueError, match="B, N, D"):
            denormalize_patch_tokens(
                torch.randn(4, 48),
                patch_size=4,
                channels=3,
                channel_mean=[0.0, 0.0, 0.0],
                channel_std=[1.0, 1.0, 1.0],
            )

    def test_collate_patch_samples_raises_on_empty_list(self):
        with pytest.raises(ValueError, match="must not be empty"):
            collate_patch_samples_for_mae([])

    def test_build_masked_input_image_requires_4d_image(self):
        patch_mask = torch.ones(1, 4, dtype=torch.bool)
        with pytest.raises(ValueError, match="B, C, H, W"):
            build_masked_input_image(torch.randn(3, 8, 8), patch_mask, patch_mask, patch_size=4)

    def test_build_masked_input_image_requires_matching_mask_shapes(self):
        image = torch.randn(1, 3, 8, 8)
        with pytest.raises(ValueError, match="matching shape"):
            build_masked_input_image(
                image,
                torch.ones(1, 4, dtype=torch.bool),
                torch.ones(1, 5, dtype=torch.bool),
                patch_size=4,
            )

    def test_build_reconstruction_composite_requires_4d_image(self):
        patches = torch.randn(1, 4, 48)
        patch_mask = torch.ones(1, 4, dtype=torch.bool)
        with pytest.raises(ValueError, match="B, C, H, W"):
            build_reconstruction_composite(torch.randn(3, 8, 8), patches, patch_mask, patch_mask, patch_size=4)

    def test_sample_visible_patch_mask_requires_valid_mask_ratio(self):
        with pytest.raises(ValueError, match="0 <= value < 1"):
            sample_visible_patch_mask(torch.ones(1, 4, dtype=torch.bool), mask_ratio=-0.1)

    def test_sample_visible_patch_mask_handles_all_invalid_patches(self):
        valid_mask = torch.zeros(2, 4, dtype=torch.bool)
        result = sample_visible_patch_mask(valid_mask, mask_ratio=0.75)
        assert result.shape == (2, 4)
        assert not result.any()

    def test_gather_visible_tokens_handles_all_invisible(self):
        tokens = torch.randn(2, 4, 32)
        visible_mask = torch.zeros(2, 4, dtype=torch.bool)
        packed, padding_mask, visible_indices = _gather_visible_tokens(tokens, visible_mask)
        assert packed.shape[1] == 1
        assert padding_mask.all()
        assert all(idx.numel() == 0 for idx in visible_indices)

    def test_mae_init_requires_divisible_image_size(self):
        with pytest.raises(ValueError, match="divisible by patch_size"):
            MarsMaskedAutoencoder(image_size=225, patch_size=16)

    def test_mae_init_requires_valid_min_valid_fraction(self):
        with pytest.raises(ValueError, match="0 <= value <= 1"):
            MarsMaskedAutoencoder(image_size=16, patch_size=4, min_valid_fraction=2.0)

    def test_mae_forward_patch_batch_extracts_scale_values_when_missing(self):
        model = MarsMaskedAutoencoder(
            image_size=8, patch_size=4, encoder_dim=16, encoder_depth=1,
            encoder_heads=2, decoder_dim=8, decoder_depth=1, decoder_heads=2,
        )
        batch = {
            "image": torch.rand(1, 3, 8, 8),
            "valid_mask": torch.ones(1, 8, 8, dtype=torch.bool),
            "scale_features": torch.rand(1, 4),
        }
        out = model.forward_patch_batch(batch, mask_ratio=0.5)
        assert torch.isfinite(out.loss)

    def test_mae_forward_requires_4d_image(self):
        model = MarsMaskedAutoencoder(
            image_size=8, patch_size=4, encoder_dim=16, encoder_depth=1,
            encoder_heads=2, decoder_dim=8, decoder_depth=1, decoder_heads=2,
        )
        with pytest.raises(ValueError, match="B, C, H, W"):
            model.forward(torch.randn(3, 8, 8), torch.ones(1, 8, 8, dtype=torch.bool), torch.ones(1))

    def test_mae_forward_requires_3d_valid_mask(self):
        model = MarsMaskedAutoencoder(
            image_size=8, patch_size=4, encoder_dim=16, encoder_depth=1,
            encoder_heads=2, decoder_dim=8, decoder_depth=1, decoder_heads=2,
        )
        with pytest.raises(ValueError, match="B, H, W"):
            model.forward(torch.randn(1, 3, 8, 8), torch.ones(8, 8, dtype=torch.bool), torch.ones(1))

    def test_mae_forward_requires_matching_image_size(self):
        model = MarsMaskedAutoencoder(
            image_size=8, patch_size=4, encoder_dim=16, encoder_depth=1,
            encoder_heads=2, decoder_dim=8, decoder_depth=1, decoder_heads=2,
        )
        with pytest.raises(ValueError, match="does not match model image_size"):
            model.forward(torch.randn(1, 3, 16, 16), torch.ones(1, 16, 16, dtype=torch.bool), torch.ones(1))

    def test_mae_forward_all_invalid_patches_continue_path(self):
        model = MarsMaskedAutoencoder(
            image_size=8, patch_size=4, encoder_dim=16, encoder_depth=1,
            encoder_heads=2, decoder_dim=8, decoder_depth=1, decoder_heads=2,
        )
        image = torch.rand(2, 3, 8, 8)
        valid_mask = torch.zeros(2, 8, 8, dtype=torch.bool)
        scale_values = torch.ones(2)
        out = model.forward(image, valid_mask, scale_values, mask_ratio=0.75)
        assert out.loss.item() == pytest.approx(0.0)
