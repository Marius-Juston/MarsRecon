"""Tests for paired multiscale alignment training utilities."""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from clip.marsclip_alignment import PairedMultiscaleAlignmentModel
from clip.marsclip_mae import MarsMaskedAutoencoder
from clip.train_marsclip_alignment import (
    build_alignment_dataloader,
    build_alignment_optimizer,
    build_alignment_subsets,
    build_alignment_tokenizer,
    evaluate_alignment_dataloader,
    load_alignment_checkpoint,
    load_alignment_model_from_stage_a_checkpoint,
    run_alignment_training,
    save_alignment_checkpoint,
    train_alignment_epoch,
)
from clip.train_marsclip_mae import build_scheduler, save_mae_checkpoint


class _ToyPairedDataset(Dataset):
    def __init__(self) -> None:
        self.observation_metadata = pd.DataFrame(
            [
                {
                    "obs_id": "OBS_A",
                    "rationale_desc": "Olympus Mons channels",
                    "rationale_expanded": "Expanded ash flow context",
                },
                {
                    "obs_id": "OBS_B",
                    "rationale_desc": "Crater floor deposits",
                    "rationale_expanded": None,
                },
                {
                    "obs_id": "OBS_C",
                    "rationale_desc": "Dune field textures",
                    "rationale_expanded": "Expanded dune migration context",
                },
            ]
        ).set_index("obs_id", drop=False)
        self.paired_records = pd.DataFrame(
            [
                {"patch_id": "pair_0", "dominant_obs_id": "OBS_A"},
                {"patch_id": "pair_1", "dominant_obs_id": "OBS_B"},
                {"patch_id": "pair_2", "dominant_obs_id": "OBS_C"},
            ]
        )
        self.samples = [self._make_sample(i) for i in range(len(self.paired_records))]

    def _make_sample(self, index: int) -> dict[str, object]:
        obs_id = str(self.paired_records.iloc[index]["dominant_obs_id"])
        obs_row = self.observation_metadata.loc[obs_id]
        local_valid_mask = torch.ones(8, 8, dtype=torch.bool)
        if index == 0:
            local_valid_mask[:, 6:] = False
        return {
            "local_image": torch.rand(3, 8, 8),
            "local_valid_mask": local_valid_mask,
            "global_image": torch.rand(3, 8, 8),
            "global_valid_mask": torch.ones(8, 8, dtype=torch.bool),
            "rationale_raw": str(obs_row["rationale_desc"]),
            "rationale_expanded": obs_row["rationale_expanded"],
            "location": torch.tensor([10.0 + index, 20.0 + index], dtype=torch.float32),
            "geo_features": torch.linspace(0.0, 1.0, 9, dtype=torch.float32),
            "local_scale_features": torch.linspace(0.1, 0.8, 8, dtype=torch.float32),
            "global_scale_features": torch.linspace(0.2, 1.6, 8, dtype=torch.float32),
            "metadata": {
                "viewing_features": torch.linspace(0.0, 1.2, 13, dtype=torch.float32),
                "local_band_presence_mask": torch.tensor([1, 1, 1], dtype=torch.bool),
                "global_band_presence_mask": torch.tensor([1, 1, 1], dtype=torch.bool),
                "local_band_valid_fraction": torch.tensor([0.8, 0.7, 0.6], dtype=torch.float32),
                "global_band_valid_fraction": torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
                "local_overall_valid_fraction": 0.75,
                "global_overall_valid_fraction": 1.0,
            },
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.samples[index]


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


def _alignment_model() -> PairedMultiscaleAlignmentModel:
    return PairedMultiscaleAlignmentModel(
        visual_backbone=_stage_a_backbone(),
        vocab_size=64,
        embed_dim=16,
        text_max_length=8,
        text_hidden_dim=32,
        text_depth=1,
        text_heads=4,
        location_hidden_dim=32,
        location_frequencies=2,
        geometry_hidden_dim=32,
        geometry_fourier_dim=8,
        fusion_heads=4,
    )


def test_build_alignment_tokenizer_uses_subset_text_metadata():
    dataset = _ToyPairedDataset()
    subset = Subset(dataset, [0, 2])

    tokenizer = build_alignment_tokenizer(subset, text_mode="raw_plus_expanded")

    assert "expanded" in tokenizer.vocab
    assert "dune" in tokenizer.vocab


def test_build_alignment_subsets_and_dataloader():
    dataset = _ToyPairedDataset()
    manifest = pd.DataFrame(
        [
            {"patch_id": "pair_0", "holdout_split": "train", "fold": 0},
            {"patch_id": "pair_1", "holdout_split": "val", "fold": 0},
            {"patch_id": "pair_2", "holdout_split": "test", "fold": 0},
        ]
    )
    train_subset, val_subset, test_subset = build_alignment_subsets(
        dataset,
        split_manifest=manifest,
    )

    assert len(train_subset) == 1
    assert len(val_subset) == 1
    assert len(test_subset) == 1

    tokenizer = build_alignment_tokenizer(train_subset)
    dataloader = build_alignment_dataloader(
        dataset,
        tokenizer=tokenizer,
        batch_size=2,
        shuffle=False,
        max_length=12,
    )
    batch = next(iter(dataloader))

    assert batch["local_image"].shape == (2, 3, 8, 8)
    assert batch["global_image"].shape == (2, 3, 8, 8)
    assert batch["input_ids"].shape == (2, 12)
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["viewing_features"].shape == (2, 13)


def test_build_alignment_subsets_ignores_manifest_ids_missing_from_paired_dataset():
    dataset = _ToyPairedDataset()
    manifest = pd.DataFrame(
        [
            {"patch_id": "pair_0", "holdout_split": "train", "fold": 0},
            {"patch_id": "pair_1", "holdout_split": "val", "fold": 0},
            {"patch_id": "missing_pair", "holdout_split": "test", "fold": 0},
            {"patch_id": "pair_2", "holdout_split": "test", "fold": 0},
        ]
    )

    train_subset, val_subset, test_subset = build_alignment_subsets(
        dataset,
        split_manifest=manifest,
    )

    assert len(train_subset) == 1
    assert len(val_subset) == 1
    assert len(test_subset) == 1


def test_load_alignment_model_from_stage_a_checkpoint_can_freeze_backbone(tmp_path):
    backbone = _stage_a_backbone()
    checkpoint_path = tmp_path / "stage_a.pt"
    save_mae_checkpoint(
        checkpoint_path,
        backbone,
        optimizer=None,
        step=7,
        history=[{"loss": 1.0}],
        config={
            "image_size": 8,
            "patch_size_px": 4,
            "in_channels": 3,
            "encoder_dim": 32,
            "encoder_depth": 1,
            "encoder_heads": 4,
            "decoder_dim": 16,
            "decoder_depth": 1,
            "decoder_heads": 4,
        },
    )

    model, state = load_alignment_model_from_stage_a_checkpoint(
        checkpoint_path,
        vocab_size=64,
        freeze_visual_backbone=True,
        alignment_config={
            "embed_dim": 16,
            "text_max_length": 8,
            "text_hidden_dim": 32,
            "text_depth": 1,
            "text_heads": 4,
            "location_hidden_dim": 32,
            "location_frequencies": 2,
            "geometry_hidden_dim": 32,
            "geometry_fourier_dim": 8,
            "fusion_heads": 4,
        },
    )

    assert state["stage_a_step"] == 7
    assert state["freeze_visual_backbone"] is True
    assert all(
        not parameter.requires_grad
        for parameter in model.visual_backbone.stage_a_backbone.parameters()
    )
    assert any(parameter.requires_grad for parameter in model.text_tower.parameters())


def test_train_eval_and_checkpoint_roundtrip(tmp_path):
    dataset = _ToyPairedDataset()
    tokenizer = build_alignment_tokenizer(dataset)
    dataloader = build_alignment_dataloader(
        dataset,
        tokenizer=tokenizer,
        batch_size=2,
        shuffle=False,
        max_length=8,
    )

    model = _alignment_model()
    optimizer = build_alignment_optimizer(
        model,
        optimizer_name="adamw",
        learning_rate=1e-3,
        weight_decay=1e-2,
    )
    train_metrics = train_alignment_epoch(
        model,
        dataloader,
        optimizer,
        device="cpu",
        max_batches=1,
        max_grad_norm=1.0,
    )
    eval_metrics = evaluate_alignment_dataloader(
        model,
        dataloader,
        device="cpu",
        max_batches=1,
    )

    assert torch.isfinite(torch.tensor(train_metrics["loss"]))
    assert torch.isfinite(torch.tensor(eval_metrics["loss"]))
    assert train_metrics["num_batches"] == 1.0
    assert eval_metrics["num_batches"] == 1.0
    assert 0.0 <= train_metrics["cross_scale_top1_mean"] <= 1.0
    assert 0.0 <= eval_metrics["local_context_top1_mean"] <= 1.0
    assert train_metrics["cross_scale_mean_rank_mean"] >= 1.0
    assert eval_metrics["loss_to_random_ratio"] > 0.0

    checkpoint_path = tmp_path / "alignment.pt"
    save_alignment_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        build_scheduler(optimizer, scheduler_name="cosine", total_steps=4, warmup_steps=1),
        step=3,
        history=[train_metrics],
        config={"text_mode": "raw_plus_expanded"},
    )

    reloaded_model = _alignment_model()
    reloaded_optimizer = build_alignment_optimizer(reloaded_model)
    reloaded_scheduler = build_scheduler(
        reloaded_optimizer,
        scheduler_name="cosine",
        total_steps=4,
        warmup_steps=1,
    )
    state = load_alignment_checkpoint(
        checkpoint_path,
        reloaded_model,
        reloaded_optimizer,
        reloaded_scheduler,
    )

    assert state["step"] == 3
    assert state["config"]["text_mode"] == "raw_plus_expanded"
    assert len(state["history"]) == 1


def test_run_alignment_training_writes_artifacts(tmp_path):
    dataset = _ToyPairedDataset()
    tokenizer = build_alignment_tokenizer(dataset)
    train_loader = build_alignment_dataloader(
        dataset,
        tokenizer=tokenizer,
        batch_size=2,
        shuffle=False,
        max_length=8,
    )
    val_loader = build_alignment_dataloader(
        dataset,
        tokenizer=tokenizer,
        batch_size=2,
        shuffle=False,
        max_length=8,
    )

    model = _alignment_model()
    optimizer = build_alignment_optimizer(
        model,
        optimizer_name="adamw",
        learning_rate=1e-3,
        weight_decay=1e-2,
    )
    scheduler = build_scheduler(
        optimizer,
        scheduler_name="cosine",
        total_steps=2,
        warmup_steps=0,
        min_lr_ratio=0.1,
    )
    out_dir = tmp_path / "alignment_run"
    summary = run_alignment_training(
        model,
        train_loader,
        optimizer,
        scheduler,
        out_dir=out_dir,
        num_steps=2,
        device="cpu",
        checkpoint_every=1,
        val_dataloader=val_loader,
        val_every=1,
        val_max_batches=1,
        max_grad_norm=1.0,
        config={"text_mode": "raw_plus_expanded"},
    )

    assert pathlib.Path(summary["checkpoint"]).exists()
    assert pathlib.Path(summary["best_checkpoint"]).exists()
    assert pathlib.Path(summary["history_path"]).exists()
    assert pathlib.Path(summary["val_history_path"]).exists()
    assert pathlib.Path(summary["loss_curve"]).exists()
    assert pathlib.Path(summary["summary_path"]).exists()
    assert 0.0 <= summary["final_cross_scale_top1"] <= 1.0
    assert 0.0 <= summary["val_final_local_context_top1"] <= 1.0
    assert summary["final_loss_to_random_ratio"] >= 0.0
