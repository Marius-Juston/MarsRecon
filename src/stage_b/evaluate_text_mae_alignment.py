"""Evaluate Stage-B text/MAE alignment checkpoints.

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

import torch

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from clip.marsclip_patches import MarsCLIPPatchDataset, load_patch_records
from stage_b.T5_encoder import T5Encoder
from stage_b.align_text_mae_embeddings import (
    AlignmentModel,
    attach_split_manifest,
    build_alignment_dataloader,
    build_alignment_embeddings,
    compute_retrieval_metrics,
    load_satmae_encoder,
    filter_patch_records_by_split,
    load_split_manifest,
    select_balanced_patch_records,
    _resolve_device,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage-B text/MAE alignment.")
    parser.add_argument("--alignment-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("/scratch/mars_hirise"))
    parser.add_argument("--bbox", type=float, nargs=4, default=(-136.0, 12.0, -124.0, 24.0))
    parser.add_argument("--patch-size-deg", type=float, default=0.005)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patch-size-px", type=int, default=8)
    parser.add_argument("--max-patches", type=int, default=1024)
    parser.add_argument("--patch-records-path", type=pathlib.Path, default=None)
    parser.add_argument("--split-manifest", type=pathlib.Path, default=None)
    parser.add_argument("--holdout-split", type=str, default=None, choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=pathlib.Path, default=None)
    parser.add_argument("--export-embeddings", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--embeddings-out", type=pathlib.Path, default=pathlib.Path("stage_b_eval_embeddings.pt"))
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
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
    aligner_config = dict(checkpoint.get("aligner_config") or {})
    projector_type = str(aligner_config.get("projector_type", checkpoint.get("projector_type", config.get("projector_type", "linear"))))
    projector_hidden_dim = int(aligner_config.get("projector_hidden_dim", config.get("projector_hidden_dim", 768)))
    projector_depth = int(aligner_config.get("projector_depth", config.get("projector_depth", 2)))
    projector_dropout = float(aligner_config.get("projector_dropout", config.get("projector_dropout", 0.0)))
    image_pool = str(checkpoint.get("image_pool", config.get("image_pool", "cls")))
    use_ema_weights = bool(args.use_ema) and checkpoint.get("ema_state") is not None

    device = _resolve_device(args.device)

    patch_records = None
    if args.patch_records_path is not None and args.patch_records_path.exists():
        patch_records = load_patch_records(args.patch_records_path)
    if args.split_manifest is not None:
        if patch_records is None:
            raise ValueError("--split-manifest requires --patch-records-path so patch ids can be aligned.")
        patch_records = attach_split_manifest(patch_records, load_split_manifest(args.split_manifest))
    if patch_records is not None and args.holdout_split is not None:
        patch_records = filter_patch_records_by_split(patch_records, split_name=args.holdout_split)
        patch_records = select_balanced_patch_records(
            patch_records,
            max_patches=args.max_patches,
            seed=0,
        )
        max_patches = None
    else:
        max_patches = args.max_patches

    dataset = MarsCLIPPatchDataset(
        root=args.root,
        bbox=tuple(args.bbox),
        patch_size=args.patch_size_deg,
        image_size=args.image_size,
        max_patches=max_patches,
        color_only=True,
        patch_records=patch_records,
    )
    dataloader = build_alignment_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        drop_last=False,
    )

    mae_encoder = load_satmae_encoder(
        model_name=mae_model,
        checkpoint_path=mae_checkpoint,
        image_size=args.image_size,
        patch_size_px=args.patch_size_px,
        freeze=True,
    ).to(device)
    if checkpoint.get("mae_encoder_state") is not None:
        mae_encoder.load_state_dict(checkpoint["mae_encoder_state"], strict=False)
    mae_encoder.eval()

    text_encoder = T5Encoder(
        model_name=text_model,
        max_length=text_max_length,
        trainable=False,
    ).to(device)
    text_encoder.eval()

    if checkpoint.get("text_encoder_state") is not None:
        text_encoder.load_state_dict(checkpoint["text_encoder_state"], strict=False)

    aligner = AlignmentModel(
        image_dim=image_dim,
        text_dim=text_dim,
        embed_dim=embed_dim,
        projector_type=projector_type,
        projector_hidden_dim=projector_hidden_dim,
        projector_depth=projector_depth,
        projector_dropout=projector_dropout,
    ).to(device)
    aligner.load_state_dict(checkpoint["aligner_state"], strict=True)
    if use_ema_weights:
        ema_state = {key: tensor.to(device) for key, tensor in checkpoint["ema_state"].items()}
        aligner.load_state_dict(ema_state, strict=True)
    aligner.eval()

    image_emb, text_emb, metadata_rows, text_rows = build_alignment_embeddings(
        dataloader=dataloader,
        mae_encoder=mae_encoder,
        text_encoder=text_encoder,
        aligner=aligner,
        device=device,
        image_pool=image_pool,
    )

    metrics = {key: float(value) for key, value in compute_retrieval_metrics(image_emb, text_emb, text_rows).items()}
    if args.holdout_split is not None:
        metrics["holdout_split"] = args.holdout_split

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
