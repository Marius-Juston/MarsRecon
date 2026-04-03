"""Tests for the SatMAE bridge integration."""

from __future__ import annotations

import pathlib
import sys

import torch

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from clip.satmae_bridge import build_satmae_model, load_satmae_models_mae, satmae_root


def test_satmae_submodule_root_exists():
    assert satmae_root().exists()


def test_load_satmae_models_mae_imports_submodule():
    module = load_satmae_models_mae()

    assert hasattr(module, "MaskedAutoencoderViT")
    assert hasattr(module, "mae_vit_base_patch16")


def test_build_satmae_model_supports_tiny_direct_constructor():
    model = build_satmae_model(
        None,
        img_size=16,
        patch_size=8,
        in_chans=3,
        embed_dim=32,
        depth=1,
        num_heads=4,
        decoder_embed_dim=16,
        decoder_depth=1,
        decoder_num_heads=4,
        norm_pix_loss=True,
    )

    image = torch.rand(2, 3, 16, 16)
    loss, pred, mask = model(image, mask_ratio=0.5)

    assert pred.shape == (2, 4, 3 * 8 * 8)
    assert mask.shape == (2, 4)
    assert torch.isfinite(loss)
