"""Generate Stage A embedding sanity artifacts from a trained MarsCLIP MAE."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import tempfile
from collections.abc import Sequence
from typing import Any

_MPL_CACHE = pathlib.Path(tempfile.gettempdir()) / "marsrecon-mpl"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib as mpl
import numpy as np
import torch

mpl.use("Agg")
from matplotlib import pyplot as plt

from clip.marsclip_patches import DEFAULT_PATCH_VALID_FRACTION, MarsCLIPPatchDataset
from clip.train_marsclip_mae import (
    MAE_MODEL_CONFIG_DEFAULTS,
    build_mae_dataloader,
    build_mae_model_from_config,
    load_mae_checkpoint,
    resolve_map_location,
)
from clip.visualize_marsclip import _to_display_rgb


def _resolve_device(device: str | torch.device | None, model: torch.nn.Module) -> torch.device:
    if isinstance(device, str) and device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device is None:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
    return torch.device(device)


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def load_trained_mae_from_checkpoint(
    checkpoint_path: pathlib.Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a trained Stage A MAE and the associated checkpoint state."""
    checkpoint = torch.load(checkpoint_path, map_location=resolve_map_location(map_location))
    config = dict(checkpoint.get("config", {}))
    config.update(_infer_mae_config_from_state_dict(checkpoint.get("model_state", {}), config))
    model = build_mae_model_from_config(config)
    state = load_mae_checkpoint(checkpoint_path, model, map_location=map_location)
    state["config"] = config
    return model, state


def _count_transformer_layers(state_dict: dict[str, Any], prefix: str) -> int | None:
    indices: set[int] = set()
    for key in state_dict:
        if key.startswith(prefix):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                indices.add(int(parts[2]))
    if not indices:
        return None
    return max(indices) + 1


def _infer_mae_config_from_state_dict(
    state_dict: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Infer missing MAE architecture values from checkpoint tensor shapes."""
    cfg = dict(config or {})
    inferred: dict[str, Any] = {}
    patch_weight = state_dict.get("patch_embed.weight")
    encoder_pos = state_dict.get("encoder_pos_embed")
    decoder_pos = state_dict.get("decoder_pos_embed")

    if torch.is_tensor(patch_weight):
        inferred.setdefault("patch_size_px", int(patch_weight.shape[-1]))
        inferred.setdefault("in_channels", int(patch_weight.shape[1]))
        inferred.setdefault("encoder_dim", int(patch_weight.shape[0]))
    if torch.is_tensor(encoder_pos):
        inferred.setdefault("encoder_dim", int(encoder_pos.shape[-1]))
        num_patches = int(encoder_pos.shape[1])
        grid = int(round(num_patches ** 0.5))
        patch_size_px = int(cfg.get("patch_size_px", inferred.get("patch_size_px", MAE_MODEL_CONFIG_DEFAULTS["patch_size_px"])))
        inferred.setdefault("image_size", grid * patch_size_px)
    if torch.is_tensor(decoder_pos):
        inferred.setdefault("decoder_dim", int(decoder_pos.shape[-1]))

    encoder_depth = _count_transformer_layers(state_dict, "encoder.layers.")
    if encoder_depth is not None:
        inferred.setdefault("encoder_depth", encoder_depth)
    decoder_depth = _count_transformer_layers(state_dict, "decoder.layers.")
    if decoder_depth is not None:
        inferred.setdefault("decoder_depth", decoder_depth)

    # Attention head counts are not recoverable from tensor shapes alone.
    inferred.setdefault("encoder_heads", int(cfg.get("encoder_heads", MAE_MODEL_CONFIG_DEFAULTS["encoder_heads"])))
    inferred.setdefault("decoder_heads", int(cfg.get("decoder_heads", MAE_MODEL_CONFIG_DEFAULTS["decoder_heads"])))
    inferred.setdefault(
        "min_valid_fraction",
        float(cfg.get("min_valid_fraction", MAE_MODEL_CONFIG_DEFAULTS["min_valid_fraction"])),
    )
    return {key: value for key, value in inferred.items() if key not in cfg}


def _serialize_embedding_record(
    sample: dict[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    metadata = dict(sample.get("metadata", {}))
    location = sample.get("location")
    if torch.is_tensor(location):
        location_values = [float(x) for x in location.detach().cpu().tolist()]
    elif location is None:
        location_values = [0.0, 0.0]
    else:
        location_values = [float(location[0]), float(location[1])]

    patch_bounds = metadata.get("patch_bounds")
    if patch_bounds is None:
        patch_bounds = metadata.get("bounds")
    if torch.is_tensor(patch_bounds):
        patch_bounds = [float(x) for x in patch_bounds.detach().cpu().tolist()]
    elif patch_bounds is not None:
        patch_bounds = [float(x) for x in patch_bounds]

    return {
        "dataset_index": int(index),
        "patch_id": metadata.get("patch_id"),
        "obs_id": metadata.get("obs_id"),
        "product_id": metadata.get("product_id"),
        "rationale_raw": sample.get("rationale_raw"),
        "has_rationale_expanded": bool(metadata.get("has_rationale_expanded", False)),
        "centroid_lon": location_values[0],
        "centroid_lat": location_values[1],
        "overall_valid_fraction": float(metadata.get("overall_valid_fraction", 0.0)),
        "is_patch_valid": bool(metadata.get("is_patch_valid", True)),
        "source_obs_count": int(metadata.get("source_obs_count", 1)),
        "dominant_overlap_fraction": float(metadata.get("dominant_overlap_fraction", 1.0)),
        "patch_bounds": patch_bounds,
    }


def collect_mae_embeddings(
    model: torch.nn.Module,
    dataset: Sequence[dict[str, Any]],
    *,
    batch_size: int = 4,
    device: str | torch.device | None = None,
    mask_ratio: float = 0.0,
    max_items: int | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect pooled Stage A embeddings plus serializable metadata records."""
    count = len(dataset) if max_items is None else min(len(dataset), int(max_items))
    if count <= 0:
        raise ValueError("Dataset must contain at least one sample.")

    samples = [dataset[i] for i in range(count)]
    dataloader = build_mae_dataloader(samples, batch_size=batch_size, shuffle=False)
    resolved_device = _resolve_device(device, model)
    model.to(resolved_device)
    model.eval()

    embeddings: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_size_actual = len(batch["metadata"])
            batch_on_device = _move_batch_to_device(batch, resolved_device)
            output = model.forward_patch_batch(batch_on_device, mask_ratio=mask_ratio)
            embeddings.append(output.pooled_embedding.detach().cpu())
            for idx in range(batch_size_actual):
                records.append(
                    _serialize_embedding_record(
                        samples[offset + idx],
                        index=offset + idx,
                    )
                )
            offset += batch_size_actual

    return torch.cat(embeddings, dim=0), records, samples


def compute_topk_neighbors(
    embeddings: torch.Tensor,
    *,
    top_k: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cosine-nearest neighbor indices and scores for each embedding."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (N, D)")
    num_items = embeddings.shape[0]
    if num_items == 0:
        raise ValueError("embeddings must not be empty.")
    if num_items == 1:
        empty = torch.empty((1, 0), dtype=torch.long)
        return empty, empty.to(dtype=torch.float32)

    k = min(int(top_k), num_items - 1)
    normed = torch.nn.functional.normalize(embeddings.to(torch.float32), dim=1)
    similarity = normed @ normed.T
    similarity.fill_diagonal_(-float("inf"))
    scores, indices = torch.topk(similarity, k=k, dim=1)
    return indices.cpu(), scores.cpu()


def project_embeddings_pca(embeddings: torch.Tensor) -> torch.Tensor:
    """Project embeddings to 2D with a lightweight PCA for qualitative inspection."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (N, D)")
    num_items, dim = embeddings.shape
    if num_items == 0:
        raise ValueError("embeddings must not be empty.")
    if num_items == 1:
        return torch.zeros((1, 2), dtype=torch.float32)

    centered = embeddings.to(torch.float32) - embeddings.to(torch.float32).mean(dim=0, keepdim=True)
    rank = min(2, num_items, dim)
    _, _, right = torch.pca_lowrank(centered, q=rank, center=False)
    projected = centered @ right[:, :rank]
    if rank == 1:
        projected = torch.cat([projected, torch.zeros_like(projected)], dim=1)
    return projected[:, :2].cpu()


def save_embedding_neighbor_gallery(
    samples: Sequence[dict[str, Any]],
    neighbor_indices: torch.Tensor,
    neighbor_scores: torch.Tensor,
    out_path: pathlib.Path | str,
    *,
    num_queries: int = 4,
) -> pathlib.Path:
    """Save a query-plus-neighbors gallery from embedding similarity results."""
    if not samples:
        raise ValueError("samples must not be empty.")
    rows = min(int(num_queries), len(samples))
    top_k = int(neighbor_indices.shape[1]) if neighbor_indices.ndim == 2 else 0
    cols = 1 + max(top_k, 0)

    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.8 * rows))
    if rows == 1:
        axes = np.array([axes])
    if cols == 1:
        axes = axes.reshape(rows, 1)

    for row_idx in range(rows):
        query = samples[row_idx]
        q_md = query.get("metadata", {})
        ax = axes[row_idx, 0]
        ax.imshow(_to_display_rgb(query["image"]), interpolation="nearest")
        ax.axis("off")
        ax.set_title(
            f"query\n{q_md.get('obs_id', 'unknown')}  valid={q_md.get('overall_valid_fraction', 0.0):.0%}",
            fontsize=9,
        )
        for col_idx in range(top_k):
            neighbor_idx = int(neighbor_indices[row_idx, col_idx])
            neighbor = samples[neighbor_idx]
            n_md = neighbor.get("metadata", {})
            ax = axes[row_idx, col_idx + 1]
            ax.imshow(_to_display_rgb(neighbor["image"]), interpolation="nearest")
            ax.axis("off")
            ax.set_title(
                (
                    f"nn{col_idx + 1}  sim={float(neighbor_scores[row_idx, col_idx]):.3f}\n"
                    f"{n_md.get('obs_id', 'unknown')}"
                ),
                fontsize=9,
            )

    fig.suptitle("MarsCLIP Stage A nearest-neighbor gallery", fontsize=12)
    fig.tight_layout()
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def save_embedding_scatter(
    projection: torch.Tensor,
    records: Sequence[dict[str, Any]],
    out_path: pathlib.Path | str,
) -> pathlib.Path:
    """Save a simple 2D embedding scatter colored by valid-pixel fraction."""
    if projection.ndim != 2 or projection.shape[1] != 2:
        raise ValueError("projection must have shape (N, 2)")
    if len(records) != projection.shape[0]:
        raise ValueError("records length must match projection rows.")

    colors = [float(record.get("overall_valid_fraction", 0.0)) for record in records]
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    scatter = ax.scatter(
        projection[:, 0].numpy(),
        projection[:, 1].numpy(),
        c=colors,
        cmap="viridis",
        s=45,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.2,
    )
    for idx in range(min(8, len(records))):
        record = records[idx]
        label = record.get("patch_id") or record.get("obs_id") or f"item_{idx}"
        ax.text(
            float(projection[idx, 0]),
            float(projection[idx, 1]),
            str(label),
            fontsize=7,
            alpha=0.7,
        )
    ax.set_title("MarsCLIP Stage A embedding PCA")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="valid pixel fraction")
    fig.tight_layout()
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def save_embedding_report(
    model: torch.nn.Module,
    dataset: Sequence[dict[str, Any]],
    out_dir: pathlib.Path | str,
    *,
    checkpoint_path: pathlib.Path | str | None = None,
    config: dict[str, Any] | None = None,
    batch_size: int = 4,
    top_k: int = 3,
    num_queries: int = 4,
    device: str | torch.device | None = None,
    mask_ratio: float = 0.0,
    max_items: int | None = None,
) -> dict[str, Any]:
    """Save Stage A embedding tensors, metadata, gallery, scatter, and summary."""
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    embeddings, records, samples = collect_mae_embeddings(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
        mask_ratio=mask_ratio,
        max_items=max_items,
    )
    neighbor_indices, neighbor_scores = compute_topk_neighbors(embeddings, top_k=top_k)
    projection = project_embeddings_pca(embeddings)

    embeddings_path = out_path / "embeddings.pt"
    torch.save({"embeddings": embeddings, "records": records}, embeddings_path)
    metadata_path = out_path / "embedding_metadata.json"
    metadata_path.write_text(json.dumps(records, indent=2))

    gallery_path = save_embedding_neighbor_gallery(
        samples,
        neighbor_indices,
        neighbor_scores,
        out_path / "nearest_neighbors.png",
        num_queries=num_queries,
    )
    scatter_path = save_embedding_scatter(projection, records, out_path / "embedding_scatter.png")

    first_neighbor_mean = None
    if neighbor_scores.numel() > 0:
        first_neighbor_mean = float(neighbor_scores[:, 0].mean().item())
    summary = {
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "num_embeddings": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "batch_size": int(batch_size),
        "mask_ratio": float(mask_ratio),
        "top_k": int(min(top_k, max(0, embeddings.shape[0] - 1))),
        "num_queries": int(min(num_queries, embeddings.shape[0])),
        "mean_embedding_norm": float(embeddings.norm(dim=1).mean().item()),
        "mean_first_neighbor_similarity": first_neighbor_mean,
        "embeddings_path": str(embeddings_path),
        "metadata_path": str(metadata_path),
        "gallery_path": str(gallery_path),
        "scatter_path": str(scatter_path),
        "config": dict(config or {}),
    }
    summary_path = out_path / "embedding_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Create a Stage A embedding sanity report.")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("/scratch/mars_hirise"))
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("marsclip_embedding_report"),
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=(-136.0, 12.0, -124.0, 24.0),
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--patch-size-deg", type=float, default=None)
    parser.add_argument("--max-patches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--num-queries", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mask-ratio", type=float, default=0.0)
    parser.add_argument("--min-valid-fraction", type=float, default=None)
    args = parser.parse_args()

    model, state = load_trained_mae_from_checkpoint(args.checkpoint, map_location=args.device)
    config = dict(state.get("config", {}))
    image_size = int(args.image_size or config.get("image_size", 224))
    patch_size_deg = float(args.patch_size_deg or config.get("patch_size_deg", 0.005))
    min_valid_fraction = float(
        args.min_valid_fraction
        if args.min_valid_fraction is not None
        else config.get("min_valid_fraction", DEFAULT_PATCH_VALID_FRACTION)
    )

    dataset = MarsCLIPPatchDataset(
        root=args.root,
        bbox=tuple(args.bbox),
        patch_size=patch_size_deg,
        image_size=image_size,
        max_patches=args.max_patches,
        min_valid_fraction=min_valid_fraction,
    )
    outputs = save_embedding_report(
        model,
        dataset,
        args.out_dir,
        checkpoint_path=args.checkpoint,
        config=config,
        batch_size=args.batch_size,
        top_k=args.top_k,
        num_queries=args.num_queries,
        device=args.device,
        mask_ratio=args.mask_ratio,
    )
    print(f"Saved embeddings to {outputs['embeddings_path']}")
    print(f"Saved neighbor gallery to {outputs['gallery_path']}")
    print(f"Saved embedding scatter to {outputs['scatter_path']}")
    print(f"Saved embedding summary to {outputs['summary_path']}")


if __name__ == "__main__":
    main()
