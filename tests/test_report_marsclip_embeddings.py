"""Tests for Stage A embedding sanity report utilities."""

from __future__ import annotations

import pathlib
import sys

import torch
from torch.optim import AdamW

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from clip.marsclip_mae import MarsMaskedAutoencoder
from clip.report_marsclip_embeddings import (
    _resolve_device,
    collect_mae_embeddings,
    compute_topk_neighbors,
    load_trained_mae_from_checkpoint,
    project_embeddings_pca,
    save_embedding_report,
)
from clip.train_marsclip_mae import save_mae_checkpoint


def _make_samples(num_samples: int = 4, image_size: int = 8) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for idx in range(num_samples):
        image = torch.zeros(3, image_size, image_size, dtype=torch.float32)
        image[0, :, :] = 0.2 + 0.1 * idx
        image[1, : image_size // 2, :] = 0.3 + 0.05 * idx
        image[2, :, : image_size // 2] = 0.4
        valid_mask = torch.ones(image_size, image_size, dtype=torch.bool)
        if idx == num_samples - 1:
            valid_mask[:, image_size // 2 :] = False
        samples.append(
            {
                "image": image,
                "valid_mask": valid_mask,
                "location": torch.tensor([float(idx), float(idx) / 2.0], dtype=torch.float32),
                "scale_features": torch.tensor([0.25 + 0.1 * idx, 1000.0, 0.005], dtype=torch.float32),
                "rationale_raw": f"Olympus Mons patch {idx}",
                "rationale_expanded": None,
                "metadata": {
                    "patch_id": f"patch_{idx}",
                    "obs_id": f"OBS_{idx}",
                    "product_id": f"OBS_{idx}_COLOR",
                    "overall_valid_fraction": float(valid_mask.float().mean()),
                    "is_patch_valid": bool(valid_mask.float().mean() >= 0.5),
                    "source_obs_count": 1,
                    "dominant_overlap_fraction": 1.0,
                    "patch_bounds": [float(idx), 0.0, float(idx + 1), 1.0],
                    "has_rationale_expanded": False,
                },
            }
        )
    return samples


def _make_model() -> MarsMaskedAutoencoder:
    return MarsMaskedAutoencoder(
        image_size=8,
        patch_size=4,
        encoder_dim=32,
        encoder_depth=2,
        encoder_heads=4,
        decoder_dim=16,
        decoder_depth=1,
        decoder_heads=4,
    )


def test_collect_mae_embeddings_matches_sample_count():
    model = _make_model()
    embeddings, records, samples = collect_mae_embeddings(
        model,
        _make_samples(4),
        batch_size=2,
        device="cpu",
        mask_ratio=0.0,
    )

    assert embeddings.shape[0] == 4
    assert embeddings.shape[1] == 32
    assert len(records) == 4
    assert len(samples) == 4
    assert records[0]["patch_id"] == "patch_0"


def test_report_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    model = _make_model()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    device = _resolve_device("auto", model)

    assert device.type == "cpu"


def test_compute_topk_neighbors_excludes_self():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    indices, scores = compute_topk_neighbors(embeddings, top_k=2)

    assert indices.shape == (3, 2)
    assert scores.shape == (3, 2)
    assert 0 not in indices[0, :1].tolist()
    assert int(indices[0, 0]) == 1


def test_project_embeddings_pca_returns_2d_projection():
    projection = project_embeddings_pca(torch.rand(4, 8))

    assert projection.shape == (4, 2)
    assert torch.isfinite(projection).all()


def test_load_trained_mae_from_checkpoint_recovers_model_and_state(tmp_path):
    model = _make_model()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = save_mae_checkpoint(
        tmp_path / "checkpoint.pt",
        model,
        optimizer,
        step=3,
        history=[{"step": 1.0, "loss": 0.5}],
        config={},
    )

    loaded_model, state = load_trained_mae_from_checkpoint(checkpoint_path)

    assert isinstance(loaded_model, MarsMaskedAutoencoder)
    assert loaded_model.image_size == 8
    assert loaded_model.patch_size == 4
    assert loaded_model.patch_embed.out_channels == 32
    assert state["step"] == 3


def test_save_embedding_report_writes_artifacts(tmp_path):
    model = _make_model()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = save_mae_checkpoint(
        tmp_path / "checkpoint.pt",
        model,
        optimizer,
        step=1,
        history=[{"step": 1.0, "loss": 0.5}],
        config={
            "image_size": 8,
            "patch_size_px": 4,
            "encoder_dim": 32,
            "encoder_depth": 2,
            "encoder_heads": 4,
            "decoder_dim": 16,
            "decoder_depth": 1,
            "decoder_heads": 4,
            "patch_size_deg": 0.005,
            "min_valid_fraction": 0.5,
        },
    )

    summary = save_embedding_report(
        model,
        _make_samples(4),
        tmp_path / "report",
        checkpoint_path=checkpoint_path,
        config={"image_size": 8},
        batch_size=2,
        top_k=2,
        num_queries=2,
        device="cpu",
        mask_ratio=0.0,
    )

    assert pathlib.Path(summary["embeddings_path"]).exists()
    assert pathlib.Path(summary["metadata_path"]).exists()
    assert pathlib.Path(summary["gallery_path"]).exists()
    assert pathlib.Path(summary["scatter_path"]).exists()
    assert pathlib.Path(summary["summary_path"]).exists()
    assert summary["num_embeddings"] == 4
    assert summary["top_k"] == 2
    assert summary["mean_first_neighbor_similarity"] is not None
