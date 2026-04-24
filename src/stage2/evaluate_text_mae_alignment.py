"""Evaluate Stage-2 text/MAE alignment checkpoints.

Computes retrieval-style alignment metrics:
- image -> text: Recall@K, MRR, median rank
- text  -> image: Recall@K, MRR, median rank

Optionally exports aligned embeddings for downstream tasks.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from clip.marsclip_patches import MarsCLIPPatchDataset, load_patch_records
from stage2.T5_encoder import T5Encoder
from stage2.align_text_mae_embeddings import (
    AlignmentModel,
    collate_patch_text,
    encode_image_with_satmae_encoder,
    load_satmae_encoder,
)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _ranks_from_similarity(similarity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 1-indexed target ranks for rows and columns."""
    n = similarity.shape[0]
    target = torch.arange(n, device=similarity.device)

    row_order = torch.argsort(similarity, dim=1, descending=True)
    row_rank = (row_order == target.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1

    col_order = torch.argsort(similarity, dim=0, descending=True)
    col_rank = (col_order == target.unsqueeze(0)).nonzero(as_tuple=False)[:, 0] + 1

    return row_rank, col_rank


def _recall_at_k(ranks: torch.Tensor, k: int) -> float:
    return float((ranks <= int(k)).float().mean().item())


def _mean_reciprocal_rank(ranks: torch.Tensor) -> float:
    return float((1.0 / ranks.to(torch.float32)).mean().item())


@torch.no_grad()
def build_embeddings(
    *,
    dataloader: DataLoader,
    mae_encoder: torch.nn.Module,
    text_encoder: T5Encoder,
    aligner: AlignmentModel,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], list[str]]:
    image_embeddings: list[torch.Tensor] = []
    text_embeddings: list[torch.Tensor] = []
    metadata_rows: list[dict[str, Any]] = []
    text_rows: list[str] = []

    for batch in dataloader:
        images = batch["image"].to(device)
        texts = batch["text"]

        image_features = encode_image_with_satmae_encoder(mae_encoder, images)
        text_features, _ = text_encoder(texts, device=device)
        image_emb, text_emb = aligner(image_features, text_features)

        image_embeddings.append(image_emb.detach().cpu())
        text_embeddings.append(text_emb.detach().cpu())
        metadata_rows.extend(batch["metadata"])
        text_rows.extend(texts)

    return (
        torch.cat(image_embeddings, dim=0),
        torch.cat(text_embeddings, dim=0),
        metadata_rows,
        text_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage-2 text/MAE alignment.")
    parser.add_argument("--alignment-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("/scratch/mars_hirise"))
    parser.add_argument("--bbox", type=float, nargs=4, default=(-136.0, 12.0, -124.0, 24.0))
    parser.add_argument("--patch-size-deg", type=float, default=0.005)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size-px", type=int, default=16)
    parser.add_argument("--max-patches", type=int, default=1024)
    parser.add_argument("--patch-records-path", type=pathlib.Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=pathlib.Path, default=None)
    parser.add_argument("--export-embeddings", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--embeddings-out", type=pathlib.Path, default=pathlib.Path("stage2_eval_embeddings.pt"))
    args = parser.parse_args()

    checkpoint = torch.load(str(args.alignment_checkpoint), map_location="cpu")
    config = dict(checkpoint.get("config", {}))

    mae_checkpoint = pathlib.Path(checkpoint.get("mae_checkpoint", config.get("mae_checkpoint", "")))
    mae_model = str(checkpoint.get("mae_model_name", config.get("mae_model", "mae_vit_base_patch16")))
    text_model = str(checkpoint.get("text_model_name", config.get("text_model", "google-t5/t5-base")))
    text_max_length = int(config.get("text_max_length", 128))

    embed_dim = int(checkpoint["embed_dim"])
    image_dim = int(checkpoint["image_dim"])
    text_dim = int(checkpoint["text_dim"])

    device = _resolve_device(args.device)

    patch_records = None
    if args.patch_records_path is not None and args.patch_records_path.exists():
        patch_records = load_patch_records(args.patch_records_path)

    dataset = MarsCLIPPatchDataset(
        root=args.root,
        bbox=tuple(args.bbox),
        patch_size=args.patch_size_deg,
        image_size=args.image_size,
        max_patches=args.max_patches,
        color_only=True,
        patch_records=patch_records,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_patch_text,
    )

    mae_encoder = load_satmae_encoder(
        model_name=mae_model,
        checkpoint_path=mae_checkpoint,
        image_size=args.image_size,
        patch_size_px=args.patch_size_px,
        freeze=True,
    ).to(device)
    mae_encoder.eval()

    text_encoder = T5Encoder(
        model_name=text_model,
        max_length=text_max_length,
        trainable=False,
    ).to(device)
    text_encoder.eval()

    if checkpoint.get("text_encoder_state") is not None:
        text_encoder.load_state_dict(checkpoint["text_encoder_state"], strict=False)

    aligner = AlignmentModel(image_dim=image_dim, text_dim=text_dim, embed_dim=embed_dim).to(device)
    aligner.load_state_dict(checkpoint["aligner_state"], strict=True)
    aligner.eval()

    image_emb, text_emb, metadata_rows, text_rows = build_embeddings(
        dataloader=dataloader,
        mae_encoder=mae_encoder,
        text_encoder=text_encoder,
        aligner=aligner,
        device=device,
    )

    similarity = image_emb @ text_emb.T
    img_to_txt_rank, txt_to_img_rank = _ranks_from_similarity(similarity)

    metrics = {
        "num_samples": int(similarity.shape[0]),
        "image_to_text_r1": _recall_at_k(img_to_txt_rank, 1),
        "image_to_text_r5": _recall_at_k(img_to_txt_rank, 5),
        "image_to_text_r10": _recall_at_k(img_to_txt_rank, 10),
        "image_to_text_mrr": _mean_reciprocal_rank(img_to_txt_rank),
        "image_to_text_median_rank": float(torch.median(img_to_txt_rank.to(torch.float32)).item()),
        "text_to_image_r1": _recall_at_k(txt_to_img_rank, 1),
        "text_to_image_r5": _recall_at_k(txt_to_img_rank, 5),
        "text_to_image_r10": _recall_at_k(txt_to_img_rank, 10),
        "text_to_image_mrr": _mean_reciprocal_rank(txt_to_img_rank),
        "text_to_image_median_rank": float(torch.median(txt_to_img_rank.to(torch.float32)).item()),
    }

    print(json.dumps(metrics, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2))

    if args.export_embeddings:
        args.embeddings_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "image_embeddings": image_emb,
                "text_embeddings": text_emb,
                "texts": text_rows,
                "metadata": metadata_rows,
                "metrics": metrics,
                "alignment_checkpoint": str(args.alignment_checkpoint),
            },
            args.embeddings_out,
        )
        print(f"[eval] saved embeddings: {args.embeddings_out}")


if __name__ == "__main__":
    main()
