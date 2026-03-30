"""Tests for Stage A MAE training utilities."""

from __future__ import annotations

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
    generate_mae_preview,
    load_mae_checkpoint,
    resolve_map_location,
    resolve_mae_model_config,
    run_mae_training,
    save_loss_curve,
    save_mae_checkpoint,
    save_training_history,
    select_best_history_entry,
    train_mae_steps,
)


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
    history = train_mae_steps(
        model,
        dataloader,
        optimizer,
        num_steps=1,
        mask_ratio=0.5,
        device="cpu",
    )

    checkpoint_path = save_mae_checkpoint(
        tmp_path / "checkpoint.pt",
        model,
        optimizer,
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
    state = load_mae_checkpoint(checkpoint_path, new_model, new_optimizer, map_location="auto")

    assert state["step"] == 1
    assert state["history"] == history
    assert state["config"]["image_size"] == 8


def test_history_and_loss_curve_writers_create_files(tmp_path):
    history = [{"step": 1.0, "loss": 0.5}, {"step": 2.0, "loss": 0.25}]

    history_path = save_training_history(history, tmp_path / "history.json")
    curve_path = save_loss_curve(history, tmp_path / "loss_curve.png")

    assert history_path.exists()
    assert curve_path.exists()
    assert curve_path.stat().st_size > 0


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
        config={"image_size": 8},
    )

    assert pathlib.Path(summary["checkpoint"]).exists()
    assert pathlib.Path(summary["best_checkpoint"]).exists()
    assert pathlib.Path(summary["history_path"]).exists()
    assert pathlib.Path(summary["loss_curve"]).exists()
    assert pathlib.Path(summary["summary_path"]).exists()
    assert summary["best_step"] is not None
    assert summary["best_checkpoint_rule"] == "minimum observed training loss"
    assert len(summary["checkpoint_paths"]) == 2
    assert len(summary["preview_paths"]) >= 2


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
