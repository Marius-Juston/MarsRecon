"""Tests for the paired multiscale multimodal alignment model."""

from __future__ import annotations

import pathlib
import sys

import torch

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from clip.marsclip_alignment import PairedMultiscaleAlignmentModel
from clip.marsclip_mae import MarsMaskedAutoencoder


def _stage_a_backbone() -> MarsMaskedAutoencoder:
    return MarsMaskedAutoencoder(
        image_size=8,
        patch_size=4,
        in_channels=3,
        encoder_dim=32,
        encoder_depth=1,
        encoder_heads=4,
        decoder_dim=16,
        decoder_depth=1,
        decoder_heads=4,
    )


def _alignment_batch() -> dict[str, torch.Tensor]:
    local_valid_mask = torch.stack(
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
    global_valid_mask = torch.ones(2, 8, 8, dtype=torch.bool)
    return {
        "local_image": torch.rand(2, 3, 8, 8),
        "local_valid_mask": local_valid_mask,
        "global_image": torch.rand(2, 3, 8, 8),
        "global_valid_mask": global_valid_mask,
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
        "location": torch.tensor([[10.0, 20.0], [15.0, 25.0]], dtype=torch.float32),
        "viewing_features": torch.rand(2, 13),
        "local_scale_features": torch.rand(2, 8),
        "global_scale_features": torch.rand(2, 8),
    }


def test_stage_a_backbone_encode_image_matches_unmasked_forward():
    model = _stage_a_backbone()
    model.eval()
    batch = _alignment_batch()

    encoded = model.encode_image(
        batch["local_image"],
        batch["local_valid_mask"],
        batch["local_scale_features"][:, 0],
    )
    out = model(
        batch["local_image"],
        batch["local_valid_mask"],
        batch["local_scale_features"][:, 0],
        mask_ratio=0.0,
        generator=torch.Generator().manual_seed(0),
    )

    assert encoded.encoded_tokens.shape == (2, 4, 32)
    assert encoded.pooled_embedding.shape == (2, 32)
    assert encoded.patch_valid_mask.shape == (2, 4)
    assert torch.allclose(encoded.pooled_embedding, out.pooled_embedding)
    assert torch.equal(encoded.patch_valid_mask, out.patch_valid_mask)


def test_alignment_model_forward_returns_multimodal_embeddings_and_losses():
    batch = _alignment_batch()
    model = PairedMultiscaleAlignmentModel(
        visual_backbone=_stage_a_backbone(),
        vocab_size=32,
        embed_dim=16,
        text_max_length=6,
        text_hidden_dim=32,
        text_depth=1,
        text_heads=4,
        location_hidden_dim=32,
        location_frequencies=3,
        geometry_hidden_dim=32,
        geometry_fourier_dim=8,
        fusion_heads=4,
    )
    out = model(batch)

    assert out.local_image_embedding.shape == (2, 16)
    assert out.global_image_embedding.shape == (2, 16)
    assert out.text_embedding.shape == (2, 16)
    assert out.location_embedding.shape == (2, 16)
    assert out.geometry_embedding.shape == (2, 16)
    assert out.context_embedding.shape == (2, 16)
    assert out.local_patch_valid_mask.shape == (2, 4)
    assert out.global_patch_valid_mask.shape == (2, 4)
    assert set(out.losses) == {"local_context", "global_context", "cross_scale"}
    assert torch.isfinite(out.loss)
    assert not bool((out.local_patch_valid_mask[0] & ~out.global_patch_valid_mask[0]).any())
