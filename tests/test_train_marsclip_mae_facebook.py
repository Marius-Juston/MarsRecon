"""Tests for the Facebook-style Mars MAE training path."""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch
from torch.optim import AdamW

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from clip.fb_mae import FacebookMaskedAutoencoderViT
from clip.fb_mae_train_utils import build_fb_mae_dataloader
from clip.train_marsclip_mae_facebook import resolve_effective_lr, run_training


def _make_samples(num_samples: int = 4, image_size: int = 16) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for idx in range(num_samples):
        valid_mask = torch.ones(image_size, image_size, dtype=torch.bool)
        if idx % 2 == 1:
            valid_mask[image_size // 2 :, : image_size // 2] = False
        samples.append(
            {
                "image": torch.rand(3, image_size, image_size),
                "valid_mask": valid_mask,
                "metadata": {"patch_id": f"patch_{idx}"},
            }
        )
    return samples


def test_build_fb_mae_dataloader_returns_collated_batch():
    dataloader = build_fb_mae_dataloader(_make_samples(2), batch_size=2, shuffle=False)

    batch = next(iter(dataloader))

    assert batch["image"].shape == (2, 3, 16, 16)
    assert batch["valid_mask"].shape == (2, 16, 16)
    assert batch["metadata"][1]["patch_id"] == "patch_1"


def test_resolve_effective_lr_uses_upstream_blr_scaling():
    lr = resolve_effective_lr(
        batch_size=64,
        accum_iter=2,
        base_lr=1.5e-4,
        explicit_lr=None,
    )

    assert lr == pytest.approx(7.5e-5)


def test_run_training_writes_epoch_artifacts(tmp_path):
    train_loader = build_fb_mae_dataloader(_make_samples(4), batch_size=2, shuffle=False)
    val_loader = build_fb_mae_dataloader(_make_samples(2), batch_size=2, shuffle=False)
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
    optimizer = AdamW(model.parameters(), lr=1e-3)

    summary = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scaler=None,
        device=torch.device("cpu"),
        out_dir=tmp_path / "fb_mae_run",
        epochs=1,
        mask_ratio=0.5,
        accum_iter=1,
        lr=1e-3,
        min_lr=1e-4,
        warmup_epochs=0,
        amp_enabled=False,
        checkpoint_every=1,
        val_max_batches=1,
        config={"model": "tiny"},
    )

    out_dir = tmp_path / "fb_mae_run"
    assert summary["epochs"] == 1
    assert summary["best_metric_name"] == "val_loss"
    assert (out_dir / "checkpoint.pt").exists()
    assert (out_dir / "best_checkpoint.pt").exists()
    assert (out_dir / "history.json").exists()
    assert (out_dir / "val_history.json").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "progress.json").exists()
