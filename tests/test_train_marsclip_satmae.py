"""Tests for the Mars SatMAE training path."""

from __future__ import annotations

import importlib
import pathlib
import sys

import pytest
import torch
from torch.optim import AdamW

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from clip.fb_mae_train_utils import build_fb_mae_dataloader
from clip.satmae_bridge import build_satmae_model
from clip.train_marsclip_satmae import (
    _apply_spectral_dropout,
    _compute_token_validity_weights,
    _filter_batch_by_validity,
    _prepare_preview_display,
    _refine_valid_mask_from_image,
    _satmae_forward_with_valid_mask,
    _stretch_preview_rgb,
    init_wandb_logger,
    resolve_effective_lr,
    resolve_run_output_dir,
    run_training,
)


class _FakeWandbImage:
    def __init__(self, path: str, caption: str | None = None) -> None:
        self.path = path
        self.caption = caption


class _FakeWandbRun:
    def __init__(
        self,
        *,
        project: str,
        entity: str | None,
        name: str | None,
        mode: str,
        directory: str,
        config: dict[str, object],
    ) -> None:
        self.project = project
        self.entity = entity
        self.name = name
        self.mode = mode
        self.dir = str(pathlib.Path(directory) / "offline-run")
        self.id = "satmae-run-test-123"
        self.config = config
        self.logged: list[tuple[int | None, dict[str, object]]] = []
        self.summary: dict[str, object] = {}
        self.finished = False

    def log(self, data: dict[str, object], step: int | None = None) -> None:
        self.logged.append((step, dict(data)))

    def finish(self) -> None:
        self.finished = True


class _FakeWandbModule:
    Image = _FakeWandbImage

    def __init__(self) -> None:
        self.runs: list[_FakeWandbRun] = []

    def init(
        self,
        *,
        entity: str | None = None,
        project: str,
        name: str | None,
        mode: str,
        dir: str,
        config: dict[str, object],
        reinit: str,
    ) -> _FakeWandbRun:
        assert reinit == "finish_previous"
        run = _FakeWandbRun(
            project=project,
            entity=entity,
            name=name,
            mode=mode,
            directory=dir,
            config=config,
        )
        self.runs.append(run)
        return run


def _make_samples(num_samples: int = 4, image_size: int = 16) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for idx in range(num_samples):
        valid_mask = torch.ones(image_size, image_size, dtype=torch.bool)
        is_valid = idx % 2 == 0
        if not is_valid:
            valid_mask[image_size // 2 :, : image_size // 2] = False
        samples.append(
            {
                "image": torch.rand(3, image_size, image_size),
                "valid_mask": valid_mask,
                "metadata": {
                    "patch_id": f"patch_{idx}",
                    "is_patch_valid": is_valid,
                    "overall_valid_fraction": 1.0 if is_valid else 0.75,
                },
            }
        )
    return samples


def test_filter_batch_by_validity_keeps_only_valid_samples():
    batch = next(iter(build_fb_mae_dataloader(_make_samples(4), batch_size=4, shuffle=False)))

    filtered, stats = _filter_batch_by_validity(
        batch,
        require_patch_valid=True,
        min_valid_fraction=0.8,
    )

    assert filtered is not None
    assert filtered["image"].shape[0] == 2
    assert stats["kept_samples"] == pytest.approx(2.0)
    assert stats["dropped_samples"] == pytest.approx(2.0)


def test_filter_batch_by_validity_recomputes_from_refined_mask():
    batch = {
        "image": torch.tensor(
            [
                [
                    [[0.0, 0.4], [0.0, 0.1]],
                    [[0.9, 0.3], [0.0, 0.2]],
                    [[0.0, 0.5], [0.0, 0.0]],
                ],
                [
                    [[0.2, 0.2], [0.2, 0.2]],
                    [[0.2, 0.2], [0.2, 0.2]],
                    [[0.2, 0.2], [0.2, 0.2]],
                ],
            ],
            dtype=torch.float32,
        ),
        "valid_mask": torch.ones(2, 2, 2, dtype=torch.bool),
        "metadata": [
            {"patch_id": "patch_bad", "is_patch_valid": True, "min_valid_fraction": 0.8},
            {"patch_id": "patch_good", "is_patch_valid": True, "min_valid_fraction": 0.8},
        ],
    }

    filtered, stats = _filter_batch_by_validity(
        batch,
        require_patch_valid=True,
        min_valid_fraction=0.8,
    )

    assert filtered is not None
    assert filtered["image"].shape[0] == 1
    assert filtered["metadata"][0]["patch_id"] == "patch_good"
    assert stats["kept_samples"] == pytest.approx(1.0)
    assert stats["dropped_samples"] == pytest.approx(1.0)


def test_resolve_effective_lr_matches_satmae_blr_scaling():
    lr = resolve_effective_lr(batch_size=64, accum_iter=2, base_lr=1e-3, explicit_lr=None)

    assert lr == pytest.approx(5e-4)


def test_satmae_forward_with_valid_mask_masks_invalid_patch_from_loss():
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
    images = torch.rand(1, 3, 16, 16)
    valid_mask = torch.ones(1, 16, 16, dtype=torch.bool)
    valid_mask[:, :8, :8] = False

    loss, pred, mask, patch_valid_mask = _satmae_forward_with_valid_mask(
        model,
        images,
        valid_mask,
        mask_ratio=0.5,
        min_valid_fraction=0.8,
    )

    assert pred.shape == (1, 4, 3 * 8 * 8)
    assert patch_valid_mask.shape == (1, 4)
    assert patch_valid_mask[0].tolist() == [False, True, True, True]
    assert bool(mask[0, 0].item()) is True
    assert torch.isfinite(loss)


def test_compute_token_validity_weights_reproduces_binary_behavior_when_disabled():
    patch_valid_fraction = torch.tensor([[1.0, 0.8, 0.79]], dtype=torch.float32)

    weights = _compute_token_validity_weights(
        patch_valid_fraction,
        min_valid_fraction=0.8,
        enabled=False,
        exponent=2.0,
    )

    assert torch.allclose(weights, torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32))


def test_compute_token_validity_weights_softens_partially_valid_tokens_when_enabled():
    patch_valid_fraction = torch.tensor([[1.0, 0.8, 0.79]], dtype=torch.float32)

    weights = _compute_token_validity_weights(
        patch_valid_fraction,
        min_valid_fraction=0.8,
        enabled=True,
        exponent=2.0,
    )

    assert torch.allclose(weights, torch.tensor([[1.0, 0.64, 0.0]], dtype=torch.float32))


def test_stretch_preview_rgb_can_reuse_reference_image_stats():
    valid_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    reference = torch.linspace(0.0, 1.0, steps=16, dtype=torch.float32).view(1, 1, 4, 4)
    target = (reference * 0.5).clone()

    stretched_with_self = _stretch_preview_rgb(target, valid_mask)
    stretched_with_reference = _stretch_preview_rgb(target, valid_mask, reference_image=reference)

    assert stretched_with_self.max().item() > 0.9
    assert stretched_with_reference.max().item() < 0.6


def test_prepare_preview_display_can_render_red_channel_as_grayscale():
    image = torch.tensor(
        [
            [
                [[0.9, 0.8], [0.7, 0.6]],
                [[0.5, 0.4], [0.3, 0.2]],
                [[0.1, 0.2], [0.3, 0.4]],
            ]
        ],
        dtype=torch.float32,
    )

    display = _prepare_preview_display(image, mode="red_grayscale")

    expected = image[:, 1:2].repeat(1, 3, 1, 1)
    assert torch.allclose(display, expected)


def test_prepare_preview_display_supports_approx_natural_mapping():
    image = torch.tensor(
        [
            [
                [[0.9, 0.8], [0.7, 0.6]],
                [[0.5, 0.4], [0.3, 0.2]],
                [[0.1, 0.2], [0.3, 0.4]],
            ]
        ],
        dtype=torch.float32,
    )

    display = _prepare_preview_display(image, mode="approx_natural")

    expected = torch.cat([image[:, 1:2], image[:, 2:3], image[:, 2:3]], dim=1)
    assert torch.allclose(display, expected)


def test_apply_spectral_dropout_only_changes_valid_pixels():
    torch.manual_seed(0)
    images = torch.arange(3 * 4 * 4, dtype=torch.float32).view(1, 3, 4, 4)
    valid_mask = torch.ones(1, 4, 4, dtype=torch.bool)
    valid_mask[:, 0, 0] = False

    dropped = _apply_spectral_dropout(
        images,
        valid_mask,
        dropout_prob=1.0,
        max_channels=1,
    )

    assert dropped.shape == images.shape
    assert torch.equal(dropped[:, :, 0, 0], images[:, :, 0, 0])
    assert not torch.equal(dropped, images)


def test_refine_valid_mask_from_image_filters_single_band_pixels():
    images = torch.tensor(
        [
            [
                [[0.0, 0.4], [0.0, -0.1]],
                [[0.9, 0.3], [0.0, 0.2]],
                [[0.0, 0.5], [0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.ones(1, 2, 2, dtype=torch.bool)

    refined = _refine_valid_mask_from_image(images, valid_mask)

    assert refined.shape == valid_mask.shape
    assert not bool(refined[0, 0, 0])  # single-band support only
    assert bool(refined[0, 0, 1])      # three-band support
    assert not bool(refined[0, 1, 0])  # all-zero nodata
    assert bool(refined[0, 1, 1])      # two-band support remains valid


def test_resolve_run_output_dir_refuses_non_empty_directory(tmp_path):
    out_dir = tmp_path / "existing_run"
    out_dir.mkdir()
    (out_dir / "summary.json").write_text("{}")

    with pytest.raises(FileExistsError):
        resolve_run_output_dir(
            out_dir=out_dir,
            out_root=tmp_path / "ignored",
            run_name=None,
            model_name="mae_vit_base_patch16",
        )


def test_init_wandb_logger_disabled_returns_none():
    logger = init_wandb_logger(mode="disabled", out_dir=pathlib.Path("/tmp/satmae-test"))

    assert logger is None


def test_init_wandb_logger_requires_package_for_offline_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(importlib, "import_module", lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)))

    with pytest.raises(RuntimeError, match="wandb is not installed"):
        init_wandb_logger(mode="offline", out_dir=tmp_path / "run")


def test_init_wandb_logger_offline_uses_fake_module(monkeypatch, tmp_path):
    fake_wandb = _FakeWandbModule()
    original_import_module = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_wandb if name == "wandb" else original_import_module(name),
    )

    logger = init_wandb_logger(
        mode="offline",
        project="marsclip-satmae-tests",
        run_name="unit-test",
        out_dir=tmp_path / "run",
        config={"image_size": 16},
    )

    assert logger is not None
    assert logger.mode == "offline"
    assert logger.project == "marsclip-satmae-tests"
    assert logger.entity == "akshayn3-auvsl"
    assert logger.run_name == "unit-test"
    assert logger.log_dir == str(tmp_path / "run" / "wandb")
    assert fake_wandb.runs[0].config["image_size"] == 16


def test_run_training_writes_satmae_epoch_artifacts(tmp_path):
    train_loader = build_fb_mae_dataloader(_make_samples(4), batch_size=2, shuffle=False, drop_last=True)
    val_loader = build_fb_mae_dataloader(_make_samples(2), batch_size=2, shuffle=False, drop_last=False)
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
    optimizer = AdamW(model.parameters(), lr=1e-3)

    summary = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scaler=None,
        device=torch.device("cpu"),
        out_dir=tmp_path / "satmae_run",
        epochs=1,
        mask_ratio=0.5,
        accum_iter=1,
        lr=1e-3,
        min_lr=1e-4,
        warmup_epochs=0,
        amp_enabled=False,
        checkpoint_every=1,
        val_max_batches=1,
        require_patch_valid=True,
        config={"model": "tiny"},
        preview_loader=None,
        reconstruction_dir=tmp_path / "satmae_run" / "reconstructions",
        dataset_normalization_stats=None,
        reconstruction_max_items=2,
    )

    out_dir = tmp_path / "satmae_run"
    assert summary["epochs"] == 1
    assert summary["best_metric_name"] == "val_loss"
    assert (out_dir / "checkpoints" / "checkpoint.pt").exists()
    assert (out_dir / "checkpoints" / "best_checkpoint.pt").exists()
    assert (out_dir / "history.json").exists()
    assert (out_dir / "val_history.json").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "progress.json").exists()


def test_run_training_logs_to_wandb_when_enabled(monkeypatch, tmp_path):
    train_loader = build_fb_mae_dataloader(_make_samples(4), batch_size=2, shuffle=False, drop_last=True)
    val_loader = build_fb_mae_dataloader(_make_samples(2), batch_size=2, shuffle=False, drop_last=False)
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
    optimizer = AdamW(model.parameters(), lr=1e-3)
    fake_wandb = _FakeWandbModule()
    original_import_module = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_wandb if name == "wandb" else original_import_module(name),
    )
    wandb_logger = init_wandb_logger(
        mode="offline",
        project="marsclip-satmae-tests",
        run_name="satmae-offline-stagea",
        out_dir=tmp_path / "run",
        config={"model": "tiny"},
    )

    summary = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scaler=None,
        device=torch.device("cpu"),
        out_dir=tmp_path / "run",
        epochs=1,
        mask_ratio=0.5,
        accum_iter=1,
        lr=1e-3,
        min_lr=1e-4,
        warmup_epochs=0,
        amp_enabled=False,
        checkpoint_every=1,
        val_max_batches=1,
        require_patch_valid=True,
        config={"model": "tiny"},
        preview_loader=None,
        reconstruction_dir=tmp_path / "run" / "reconstructions",
        dataset_normalization_stats=None,
        reconstruction_max_items=2,
        progress_log_interval=1,
        wandb_logger=wandb_logger,
    )

    run = fake_wandb.runs[0]
    assert run.finished is True
    assert summary["wandb_mode"] == "offline"
    assert summary["wandb_project"] == "marsclip-satmae-tests"
    assert summary["wandb_run_name"] == "satmae-offline-stagea"
    assert summary["wandb_run_id"] == "satmae-run-test-123"
    assert summary["wandb_log_dir"] == str(tmp_path / "run" / "wandb")
    assert summary["wandb_run_dir"] == str((tmp_path / "run" / "wandb") / "offline-run")
    logged_keys = {key for _, payload in run.logged for key in payload.keys()}
    assert "train/loss" in logged_keys
    assert "epoch/train_loss" in logged_keys
    assert "epoch/val_loss" in logged_keys
