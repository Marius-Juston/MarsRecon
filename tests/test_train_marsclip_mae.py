"""Tests for Stage A MAE training utilities."""

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

from marsclip_mae import MarsMaskedAutoencoder
from train_marsclip_mae import (
    _resolve_device,
    build_mae_model_from_config,
    build_mae_dataloader,
    build_optimizer,
    build_scheduler,
    compute_valid_pixel_channel_stats,
    count_trainable_parameters,
    current_learning_rate,
    evaluate_mae_dataloader,
    generate_mae_preview,
    init_wandb_logger,
    load_mae_checkpoint,
    load_normalization_stats,
    resolve_map_location,
    resolve_mae_model_config,
    run_mae_training,
    save_normalization_stats,
    save_loss_curve,
    save_mae_checkpoint,
    save_training_history,
    select_best_history_entry,
    train_mae_steps,
)


class _FakeWandbImage:
    def __init__(self, path: str, caption: str | None = None) -> None:
        self.path = path
        self.caption = caption


class _FakeWandbRun:
    def __init__(self, *, project: str, name: str | None, mode: str, directory: str, config: dict[str, object]) -> None:
        self.project = project
        self.name = name
        self.mode = mode
        self.dir = str(pathlib.Path(directory) / "offline-run")
        self.id = "run-test-123"
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
        project: str,
        name: str | None,
        mode: str,
        dir: str,
        config: dict[str, object],
        reinit: str,
    ) -> _FakeWandbRun:
        assert reinit == "finish_previous"
        run = _FakeWandbRun(project=project, name=name, mode=mode, directory=dir, config=config)
        self.runs.append(run)
        return run


def _make_samples(num_samples: int = 2, image_size: int = 8) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for idx in range(num_samples):
        image = torch.rand(3, image_size, image_size)
        valid_mask = torch.ones(image_size, image_size, dtype=torch.bool)
        if idx % 2 == 1:
            valid_mask[:, image_size // 2 :] = False
        samples.append(
            {
                "image": image,
                "valid_mask": valid_mask,
                "scale_features": torch.tensor([0.25 + 0.25 * idx, 1000.0, 0.005]),
                "rationale_raw": f"Patch {idx}",
                "rationale_expanded": None,
                "metadata": {
                    "patch_id": f"patch_{idx}",
                    "obs_id": f"OBS_{idx}",
                    "overall_valid_fraction": float(valid_mask.float().mean()),
                },
            }
        )
    return samples


def test_build_mae_dataloader_returns_collated_batch():
    dataloader = build_mae_dataloader(
        _make_samples(2),
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    assert batch["image"].shape == (2, 3, 8, 8)
    assert batch["valid_mask"].shape == (2, 8, 8)
    assert batch["scale_values"].shape == (2,)


def test_build_mae_dataloader_accepts_worker_tuning_flags():
    dataloader = build_mae_dataloader(
        _make_samples(2),
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True,
    )

    assert dataloader.num_workers == 2
    assert dataloader.pin_memory is True
    assert dataloader.prefetch_factor == 3
    assert dataloader.persistent_workers is True


def test_build_optimizer_supports_adam_and_adamw():
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

    adamw = build_optimizer(model, optimizer_name="adamw", learning_rate=1e-3, weight_decay=1e-2)
    adam = build_optimizer(model, optimizer_name="adam", learning_rate=1e-3, weight_decay=1e-2)

    assert adamw.__class__.__name__ == "AdamW"
    assert adam.__class__.__name__ == "Adam"


def test_compute_valid_pixel_channel_stats_uses_only_valid_pixels():
    mean, std = compute_valid_pixel_channel_stats(
        [
            {
                "image": torch.tensor(
                    [
                        [[1.0, 9.0], [3.0, 9.0]],
                        [[2.0, 9.0], [4.0, 9.0]],
                    ],
                    dtype=torch.float32,
                ),
                "valid_mask": torch.tensor([[True, False], [True, False]], dtype=torch.bool),
            }
        ]
    )

    assert mean == pytest.approx([2.0, 3.0])
    assert std == pytest.approx([1.0, 1.0])


def test_normalization_stats_round_trip(tmp_path):
    path = tmp_path / "normalization_stats.json"

    saved = save_normalization_stats(
        path,
        mean=[0.1, 0.2, 0.3],
        std=[0.4, 0.5, 0.6],
        metadata={"split": "train", "color_only": True},
    )
    mean, std, payload = load_normalization_stats(saved)

    assert mean == pytest.approx([0.1, 0.2, 0.3])
    assert std == pytest.approx([0.4, 0.5, 0.6])
    assert payload["split"] == "train"
    assert payload["color_only"] is True


def test_build_scheduler_none_returns_none():
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
    optimizer = build_optimizer(model, optimizer_name="adamw", learning_rate=1e-3, weight_decay=1e-2)

    scheduler = build_scheduler(optimizer, scheduler_name="none", total_steps=4)

    assert scheduler is None


def test_cosine_scheduler_changes_learning_rate():
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
    optimizer = build_optimizer(model, optimizer_name="adamw", learning_rate=1e-3, weight_decay=1e-2)
    scheduler = build_scheduler(
        optimizer,
        scheduler_name="cosine",
        total_steps=4,
        warmup_steps=1,
        min_lr_ratio=0.1,
    )

    initial_lr = current_learning_rate(optimizer)
    optimizer.step()
    scheduler.step()
    optimizer.step()
    scheduler.step()
    updated_lr = current_learning_rate(optimizer)

    assert scheduler is not None
    assert updated_lr != pytest.approx(initial_lr)


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
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
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    device = _resolve_device("auto", model)

    assert device.type == "cpu"


def test_resolve_map_location_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    location = resolve_map_location("auto")

    assert location == "cpu"


def test_init_wandb_logger_disabled_returns_none():
    logger = init_wandb_logger(mode="disabled")

    assert logger is None


def test_init_wandb_logger_requires_package_for_offline_mode(monkeypatch):
    monkeypatch.setattr(importlib, "import_module", lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)))

    with pytest.raises(RuntimeError, match="wandb is not installed"):
        init_wandb_logger(mode="offline")


def test_init_wandb_logger_offline_uses_fake_module(monkeypatch, tmp_path):
    fake_wandb = _FakeWandbModule()
    monkeypatch.setattr(importlib, "import_module", lambda name: fake_wandb)

    logger = init_wandb_logger(
        mode="offline",
        project="marsclip-tests",
        run_name="unit-test",
        out_dir=tmp_path / "run",
        config={"image_size": 8},
    )

    assert logger is not None
    assert logger.mode == "offline"
    assert logger.project == "marsclip-tests"
    assert logger.run_name == "unit-test"
    assert logger.log_dir == str(tmp_path / "run" / "wandb")
    assert fake_wandb.runs[0].config["image_size"] == 8


def test_train_mae_steps_returns_history_and_updates_model():
    dataloader = build_mae_dataloader(
        _make_samples(2),
        batch_size=2,
        shuffle=False,
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
    )
    optimizer = AdamW(model.parameters(), lr=1e-3)

    history = train_mae_steps(
        model,
        dataloader,
        optimizer,
        num_steps=2,
        mask_ratio=0.5,
        device="cpu",
    )

    assert len(history) == 2
    assert history[0]["step"] == pytest.approx(1.0)
    assert torch.isfinite(torch.tensor(history[-1]["loss"]))
    assert "visible_patch_fraction" in history[-1]
    assert "masked_valid_patch_fraction" in history[-1]
    assert "loss_pixel_fraction" in history[-1]


def test_evaluate_mae_dataloader_returns_aggregate_metrics():
    dataloader = build_mae_dataloader(
        _make_samples(4),
        batch_size=2,
        shuffle=False,
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
    )

    metrics = evaluate_mae_dataloader(
        model,
        dataloader,
        mask_ratio=0.5,
        device="cpu",
    )

    assert metrics["num_batches"] == pytest.approx(2.0)
    assert metrics["loss"] >= 0.0


def test_evaluate_mae_dataloader_reports_batch_progress():
    dataloader = build_mae_dataloader(
        _make_samples(4),
        batch_size=2,
        shuffle=False,
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
    )
    seen: list[tuple[int, int | None]] = []

    metrics = evaluate_mae_dataloader(
        model,
        dataloader,
        mask_ratio=0.5,
        device="cpu",
        batch_callback=lambda batch_idx, total: seen.append((batch_idx, total)),
    )

    assert metrics["num_batches"] == pytest.approx(2.0)
    assert seen == [(1, 2), (2, 2)]


def test_checkpoint_round_trip_restores_step_and_history(tmp_path):
    dataloader = build_mae_dataloader(
        _make_samples(2),
        batch_size=2,
        shuffle=False,
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
    )
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(
        optimizer,
        scheduler_name="cosine",
        total_steps=4,
        warmup_steps=1,
        min_lr_ratio=0.1,
    )
    history = train_mae_steps(
        model,
        dataloader,
        optimizer,
        num_steps=1,
        mask_ratio=0.5,
        device="cpu",
    )
    if scheduler is not None:
        scheduler.step()

    checkpoint_path = save_mae_checkpoint(
        tmp_path / "checkpoint.pt",
        model,
        optimizer,
        scheduler,
        step=1,
        history=history,
        config={"image_size": 8},
    )

    new_model = MarsMaskedAutoencoder(
        image_size=8,
        patch_size=4,
        encoder_dim=32,
        encoder_depth=2,
        encoder_heads=4,
        decoder_dim=16,
        decoder_depth=1,
        decoder_heads=4,
    )
    new_optimizer = AdamW(new_model.parameters(), lr=1e-3)
    new_scheduler = build_scheduler(
        new_optimizer,
        scheduler_name="cosine",
        total_steps=4,
        warmup_steps=1,
        min_lr_ratio=0.1,
    )
    state = load_mae_checkpoint(checkpoint_path, new_model, new_optimizer, new_scheduler, map_location="auto")

    assert state["step"] == 1
    assert state["history"] == history
    assert state["config"]["image_size"] == 8
    assert new_scheduler is not None
    assert new_scheduler.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]


def test_history_and_loss_curve_writers_create_files(tmp_path):
    history = [{"step": 1.0, "loss": 0.5}, {"step": 2.0, "loss": 0.25}]

    history_path = save_training_history(history, tmp_path / "history.json")
    curve_path = save_loss_curve(history, tmp_path / "loss_curve.png")

    assert history_path.exists()
    assert curve_path.exists()
    assert curve_path.stat().st_size > 0


def test_save_loss_curve_uses_log_scale(monkeypatch, tmp_path):
    history = [{"step": 1.0, "loss": 0.5}, {"step": 2.0, "loss": 0.25}]
    captured: dict[str, object] = {}
    plot_module = save_loss_curve.__globals__["plt"]
    original_subplots = plot_module.subplots

    def _wrapped_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plot_module, "subplots", _wrapped_subplots)

    save_loss_curve(history, tmp_path / "loss_curve.png")

    ax = captured["ax"]
    assert ax.get_yscale() == "log"


def test_resolve_mae_model_config_uses_defaults_and_saved_keys():
    resolved = resolve_mae_model_config(
        {
            "image_size": 64,
            "patch_size_px": 8,
            "encoder_dim": 32,
            "encoder_depth": 2,
            "encoder_heads": 4,
            "decoder_dim": 16,
            "decoder_depth": 1,
            "decoder_heads": 4,
        }
    )

    assert resolved["image_size"] == 64
    assert resolved["patch_size"] == 8
    assert resolved["encoder_dim"] == 32
    assert resolved["decoder_dim"] == 16
    assert resolved["min_valid_fraction"] == pytest.approx(0.5)
    assert resolved["normalize_inputs"] is False
    assert resolved["normalize_targets"] is False


def test_resolve_mae_model_config_supports_vit_base_preset():
    resolved = resolve_mae_model_config({}, preset="vit_base")

    assert resolved["encoder_dim"] == 768
    assert resolved["encoder_depth"] == 12
    assert resolved["encoder_heads"] == 12
    assert resolved["decoder_dim"] == 512
    assert resolved["decoder_depth"] == 8
    assert resolved["decoder_heads"] == 16


def test_resolve_mae_model_config_treats_none_as_missing_for_preset_values():
    resolved = resolve_mae_model_config(
        {
            "model_preset": "vit_base",
            "encoder_dim": None,
            "encoder_depth": None,
            "encoder_heads": None,
            "decoder_dim": None,
            "decoder_depth": None,
            "decoder_heads": None,
        },
        preset="vit_base",
    )

    assert resolved["encoder_dim"] == 768
    assert resolved["decoder_dim"] == 512


def test_resolve_mae_model_config_raises_on_unknown_preset():
    with pytest.raises(ValueError, match="Unknown MAE model preset"):
        resolve_mae_model_config({}, preset="not-a-real-preset")


def test_build_mae_model_from_config_returns_compatible_model():
    model = build_mae_model_from_config(
        {
            "image_size": 8,
            "patch_size_px": 4,
            "encoder_dim": 32,
            "encoder_depth": 2,
            "encoder_heads": 4,
            "decoder_dim": 16,
            "decoder_depth": 1,
            "decoder_heads": 4,
        }
    )

    assert isinstance(model, MarsMaskedAutoencoder)
    assert model.image_size == 8
    assert model.patch_size == 4


def test_build_mae_model_from_config_preserves_normalization_settings():
    model = build_mae_model_from_config(
        {
            "image_size": 8,
            "patch_size_px": 4,
            "encoder_dim": 32,
            "encoder_depth": 2,
            "encoder_heads": 4,
            "decoder_dim": 16,
            "decoder_depth": 1,
            "decoder_heads": 4,
            "normalize_inputs": True,
            "normalize_targets": True,
            "input_mean": [0.1, 0.2, 0.3],
            "input_std": [0.5, 0.6, 0.7],
        }
    )

    assert model.normalize_inputs is True
    assert model.normalize_targets is True
    assert model.input_mean.flatten().tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert model.input_std.flatten().tolist() == pytest.approx([0.5, 0.6, 0.7])


def test_count_trainable_parameters_reports_expected_counts():
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

    trainable, total = count_trainable_parameters(model)

    assert trainable == total
    assert trainable > 0


def test_select_best_history_entry_picks_lowest_loss():
    history = [
        {"step": 1.0, "loss": 0.5},
        {"step": 2.0, "loss": 0.25},
        {"step": 3.0, "loss": 0.4},
    ]

    best = select_best_history_entry(history)

    assert best["step"] == pytest.approx(2.0)
    assert best["loss"] == pytest.approx(0.25)


def test_generate_mae_preview_writes_png(tmp_path):
    samples = _make_samples(2)
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

    out_path = generate_mae_preview(
        model,
        samples,
        tmp_path / "preview.png",
        mask_ratio=0.5,
        device="cpu",
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_run_mae_training_creates_periodic_artifacts(tmp_path):
    dataloader = build_mae_dataloader(
        _make_samples(2),
        batch_size=2,
        shuffle=False,
    )
    preview_samples = _make_samples(2)
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
    optimizer = AdamW(model.parameters(), lr=1e-3)

    summary = run_mae_training(
        model,
        dataloader,
        optimizer,
        out_dir=tmp_path / "run",
        num_steps=2,
        mask_ratio=0.5,
        preview_samples=preview_samples,
        checkpoint_every=1,
        preview_every=1,
        device="cpu",
        config={
            "image_size": 8,
            "batch_size": 8,
            "color_only": True,
            "num_workers": 2,
            "pin_memory": True,
            "prefetch_factor": 3,
            "persistent_workers": True,
            "normalization_mode": "input",
            "normalize_inputs": True,
            "normalize_targets": False,
            "normalization_stats_path": str(tmp_path / "norm_stats.json"),
            "patch_records_path": str(tmp_path / "patch_records.pkl"),
            "input_mean": [0.1, 0.2, 0.3],
            "input_std": [0.4, 0.5, 0.6],
        },
    )

    assert pathlib.Path(summary["checkpoint"]).exists()
    assert pathlib.Path(summary["best_checkpoint"]).exists()
    assert pathlib.Path(summary["history_path"]).exists()
    assert pathlib.Path(summary["loss_curve"]).exists()
    assert pathlib.Path(summary["progress_path"]).exists()
    assert pathlib.Path(summary["summary_path"]).exists()
    assert summary["model_preset"] is None
    assert summary["trainable_parameters"] is None
    assert summary["total_parameters"] is None
    assert summary["batch_size"] == 8
    assert summary["color_only"] is True
    assert summary["num_workers"] == 2
    assert summary["pin_memory"] is True
    assert summary["prefetch_factor"] == 3
    assert summary["persistent_workers"] is True
    assert summary["normalization_mode"] == "input"
    assert summary["normalization_stats_path"] == str(tmp_path / "norm_stats.json")
    assert summary["patch_records_path"] == str(tmp_path / "patch_records.pkl")
    assert summary["normalize_inputs"] is True
    assert summary["normalize_targets"] is False
    assert summary["input_mean"] == pytest.approx([0.1, 0.2, 0.3])
    assert summary["input_std"] == pytest.approx([0.4, 0.5, 0.6])
    assert summary["best_step"] is not None
    assert summary["best_checkpoint_rule"] == "minimum observed training loss"
    assert len(summary["checkpoint_paths"]) == 2
    assert len(summary["preview_paths"]) >= 2
    assert summary["wandb_mode"] == "disabled"


def test_run_mae_training_uses_validation_loss_for_best_checkpoint(tmp_path):
    train_loader = build_mae_dataloader(
        _make_samples(4),
        batch_size=2,
        shuffle=False,
    )
    val_loader = build_mae_dataloader(
        _make_samples(4),
        batch_size=2,
        shuffle=False,
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
    )
    optimizer = AdamW(model.parameters(), lr=1e-3)

    summary = run_mae_training(
        model,
        train_loader,
        optimizer,
        out_dir=tmp_path / "run_val",
        num_steps=2,
        mask_ratio=0.5,
        preview_samples=[],
        checkpoint_every=1,
        device="cpu",
        val_dataloader=val_loader,
        val_every=1,
        config={
            "model_preset": "mars_small",
            "train_count": 4,
            "val_count": 4,
            "test_count": 2,
            "split_mode": "holdout",
            "split_manifest": str(tmp_path / "splits.csv"),
            "split_summary_path": str(tmp_path / "split_summary.json"),
        },
    )

    assert summary["best_metric_name"] == "val_loss"
    assert summary["val_initial_loss"] is not None
    assert summary["val_final_loss"] is not None
    assert pathlib.Path(summary["val_history_path"]).exists()
    assert summary["best_checkpoint_rule"] == "minimum observed val loss"


def test_run_mae_training_preserves_model_metadata_in_summary(tmp_path):
    dataloader = build_mae_dataloader(
        _make_samples(2),
        batch_size=2,
        shuffle=False,
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
    )
    optimizer = AdamW(model.parameters(), lr=1e-3)
    trainable, total = count_trainable_parameters(model)

    summary = run_mae_training(
        model,
        dataloader,
        optimizer,
        out_dir=tmp_path / "run_meta",
        num_steps=1,
        mask_ratio=0.5,
        preview_samples=[],
        device="cpu",
        config={
            "model_preset": "vit_base",
            "trainable_parameters": trainable,
            "total_parameters": total,
        },
    )

    assert summary["model_preset"] == "vit_base"
    assert summary["trainable_parameters"] == trainable
    assert summary["total_parameters"] == total


def test_run_mae_training_resume_extends_history(tmp_path):
    dataloader = build_mae_dataloader(
        _make_samples(2),
        batch_size=2,
        shuffle=False,
    )
    preview_samples = _make_samples(2)
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
    optimizer = AdamW(model.parameters(), lr=1e-3)
    first = run_mae_training(
        model,
        dataloader,
        optimizer,
        out_dir=tmp_path / "run",
        num_steps=1,
        mask_ratio=0.5,
        preview_samples=preview_samples,
        device="cpu",
    )

    resumed_model = MarsMaskedAutoencoder(
        image_size=8,
        patch_size=4,
        encoder_dim=32,
        encoder_depth=2,
        encoder_heads=4,
        decoder_dim=16,
        decoder_depth=1,
        decoder_heads=4,
    )
    resumed_optimizer = AdamW(resumed_model.parameters(), lr=1e-3)
    resume_state = load_mae_checkpoint(first["checkpoint"], resumed_model, resumed_optimizer)
    resume_state["checkpoint_path"] = first["checkpoint"]

    second = run_mae_training(
        resumed_model,
        dataloader,
        resumed_optimizer,
        out_dir=tmp_path / "run_resume",
        num_steps=1,
        mask_ratio=0.5,
        preview_samples=preview_samples,
        device="cpu",
        resume_state=resume_state,
    )

    assert second["resume_step"] == 1
    assert second["num_steps_total"] == 2
    assert second["best_step"] is not None


def test_run_mae_training_logs_to_wandb_when_enabled(monkeypatch, tmp_path):
    fake_wandb = _FakeWandbModule()
    monkeypatch.setattr(importlib, "import_module", lambda name: fake_wandb)

    dataloader = build_mae_dataloader(
        _make_samples(2),
        batch_size=2,
        shuffle=False,
    )
    preview_samples = _make_samples(2)
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
    optimizer = AdamW(model.parameters(), lr=1e-3)
    wandb_logger = init_wandb_logger(
        mode="offline",
        project="marsclip-tests",
        run_name="offline-stagea",
        out_dir=tmp_path / "run",
        config={"image_size": 8},
    )

    summary = run_mae_training(
        model,
        dataloader,
        optimizer,
        out_dir=tmp_path / "run",
        num_steps=2,
        mask_ratio=0.5,
        preview_samples=preview_samples,
        checkpoint_every=1,
        preview_every=1,
        device="cpu",
        config={"image_size": 8},
        wandb_logger=wandb_logger,
    )

    run = fake_wandb.runs[0]
    logged_keys = {key for _, payload in run.logged for key in payload}

    assert summary["wandb_mode"] == "offline"
    assert summary["wandb_project"] == "marsclip-tests"
    assert summary["wandb_run_name"] == "offline-stagea"
    assert summary["wandb_run_id"] == "run-test-123"
    assert summary["wandb_log_dir"] == str(tmp_path / "run" / "wandb")
    assert summary["wandb_run_dir"] == str((tmp_path / "run" / "wandb") / "offline-run")
    assert "train/loss" in logged_keys
    assert "train/lr" in logged_keys
    assert "train/preview_start" in logged_keys
    assert "train/preview_final" in logged_keys
    assert "train/loss_curve" in logged_keys
    assert run.summary["best_checkpoint_rule"] == "minimum observed training loss"
    assert run.finished is True


@pytest.mark.integration
def test_train_mae_steps_real_patch_smoke():
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
    dataset = MarsCLIPPatchDataset(
        geo_dataset=geo,
        patch_records=patch_records,
        observation_metadata=observation_metadata,
        image_size=64,
    )
    dataloader = build_mae_dataloader(dataset, batch_size=1, shuffle=False)
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
    optimizer = AdamW(model.parameters(), lr=1e-4)

    history = train_mae_steps(
        model,
        dataloader,
        optimizer,
        num_steps=1,
        mask_ratio=0.75,
        device="cpu",
    )

    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history[0]["loss"]))
