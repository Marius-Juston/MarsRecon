"""Evaluate MarsCLIP B1a-geo alignment checkpoints.

Loads a checkpoint produced by ``stage_b.align_marsclip`` and computes:

- image -> text:  Recall@K, MRR, median rank
- text  -> image: Recall@K, MRR, median rank
- image -> geo:   Recall@K, MRR, median rank (diagonal-positive)
- geo   -> image: Recall@K, MRR, median rank (diagonal-positive)

Optionally exports the aligned image / text / geo embeddings for downstream
analysis. The aligner is reconstructed from ``aligner_config`` saved in the
checkpoint so MLP / pooling / EMA / geo-context choices are honored without
extra CLI flags.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from functools import partial

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from clip.marsclip_patches import MarsCLIPPatchDataset, load_patch_records
from stage_b.T5_encoder import T5Encoder
from stage_b.align_marsclip import (
    GEO_CONTEXT_CHOICES,
    MarsCLIPAlignmentModel,
    build_marsclip_embeddings,
    collate_geo_warmup,
    compute_pairwise_retrieval_metrics,
    geo_input_dim,
)
from stage_b.align_text_mae_embeddings import (
    _resolve_device,
    attach_split_manifest,
    compute_retrieval_metrics,
    filter_patch_records_by_split,
    load_satmae_encoder,
    load_split_manifest,
    select_balanced_patch_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MarsCLIP B1a-geo alignment checkpoint.")
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
    parser.add_argument(
        "--embeddings-out",
        type=pathlib.Path,
        default=pathlib.Path("marsclip_eval_embeddings.pt"),
    )
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--geo-context-override",
        type=str,
        default=None,
        choices=GEO_CONTEXT_CHOICES,
        help="Override geo_context (defaults to whatever the checkpoint stored).",
    )
    args = parser.parse_args()

    checkpoint = torch.load(str(args.alignment_checkpoint), map_location="cpu")
    config = dict(checkpoint.get("config", {}))

    mae_checkpoint = pathlib.Path(checkpoint.get("mae_checkpoint", config.get("mae_checkpoint", "")))
    mae_model = str(checkpoint.get("mae_model_name", config.get("mae_model", "mae_vit_base_patch16")))
    text_model = str(checkpoint.get("text_model_name", config.get("text_model", "google-t5/t5-base")))
    text_max_length = int(config.get("text_max_length", 128))

    aligner_config = dict(checkpoint.get("aligner_config") or {})
    if not aligner_config:
        raise ValueError(
            "Checkpoint is missing aligner_config; this evaluator only supports "
            "B1a-geo checkpoints from stage_b.align_marsclip."
        )

    image_dim = int(aligner_config.get("image_dim", checkpoint.get("image_dim", 0)))
    text_dim = int(aligner_config.get("text_dim", checkpoint.get("text_dim", 0)))
    geo_dim = int(aligner_config.get("geo_dim", checkpoint.get("geo_dim", 0)))
    embed_dim = int(aligner_config.get("embed_dim", checkpoint.get("embed_dim", 0)))
    projector_type = str(aligner_config.get("projector_type", checkpoint.get("projector_type", "mlp")))
    projector_hidden_dim = int(aligner_config.get("projector_hidden_dim", 768))
    projector_depth = int(aligner_config.get("projector_depth", 2))
    projector_dropout = float(aligner_config.get("projector_dropout", 0.0))
    geo_encoder_type = str(aligner_config.get("geo_encoder_type", "mlp"))
    geo_coords_dim_value = int(aligner_config.get("geo_coords_dim", 0))
    geo_hidden_dim = int(aligner_config.get("geo_hidden_dim", 256))
    geo_depth = int(aligner_config.get("geo_depth", 2))
    geo_dropout = float(aligner_config.get("geo_dropout", 0.0))
    geo_rff_sigmas = tuple(
        float(s) for s in aligner_config.get("geo_rff_sigmas", (1.0, 4.0, 16.0, 64.0))
    )
    geo_rff_encoded_size = int(aligner_config.get("geo_rff_encoded_size", 128))
    geo_siren_w0 = float(aligner_config.get("geo_siren_w0", 1.0))
    geo_siren_w0_initial = float(aligner_config.get("geo_siren_w0_initial", 30.0))
    geo_sh_legendre_polys = int(aligner_config.get("geo_sh_legendre_polys", 10))
    text_temperature_init = float(aligner_config.get("text_temperature_init", 0.07))
    geo_temperature_init = float(aligner_config.get("geo_temperature_init", 0.07))
    image_pool = str(checkpoint.get("image_pool", config.get("image_pool", "cls")))
    geo_context = str(
        args.geo_context_override
        or checkpoint.get("geo_context", config.get("geo_context", "latlon_view_scale"))
    )
    expected_geo_dim = geo_input_dim(geo_context)
    if geo_context != "none" and expected_geo_dim != geo_dim:
        raise ValueError(
            f"Checkpoint geo_dim={geo_dim} disagrees with --geo-context-override "
            f"'{geo_context}' (expected {expected_geo_dim})."
        )

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

    collate = partial(collate_geo_warmup, geo_context=geo_context)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": bool(args.pin_memory),
        "collate_fn": collate,
        "drop_last": False,
    }
    if int(args.num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        if args.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    dataloader = DataLoader(**loader_kwargs)

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

    aligner = MarsCLIPAlignmentModel(
        image_dim=image_dim,
        text_dim=text_dim,
        geo_dim=int(geo_dim) if int(geo_dim) > 0 else 1,
        embed_dim=embed_dim,
        projector_type=projector_type,
        projector_hidden_dim=projector_hidden_dim,
        projector_depth=projector_depth,
        projector_dropout=projector_dropout,
        geo_encoder_type=geo_encoder_type,
        geo_coords_dim=geo_coords_dim_value,
        geo_hidden_dim=geo_hidden_dim,
        geo_depth=geo_depth,
        geo_dropout=geo_dropout,
        geo_rff_sigmas=geo_rff_sigmas,
        geo_rff_encoded_size=geo_rff_encoded_size,
        geo_siren_w0=geo_siren_w0,
        geo_siren_w0_initial=geo_siren_w0_initial,
        geo_sh_legendre_polys=geo_sh_legendre_polys,
        text_temperature_init=text_temperature_init,
        geo_temperature_init=geo_temperature_init,
    ).to(device)
    aligner.load_state_dict(checkpoint["aligner_state"], strict=True)
    if use_ema_weights:
        ema_state = {key: tensor.to(device) for key, tensor in checkpoint["ema_state"].items()}
        aligner.load_state_dict(ema_state, strict=True)
    aligner.eval()

    image_emb, text_emb, geo_emb, text_rows = build_marsclip_embeddings(
        dataloader=dataloader,
        mae_encoder=mae_encoder,
        text_encoder=text_encoder,
        aligner=aligner,
        device=device,
        image_pool=image_pool,
        geo_context=geo_context,
    )

    metrics: dict[str, float] = {
        key: float(value)
        for key, value in compute_retrieval_metrics(image_emb, text_emb, text_rows).items()
    }
    if geo_emb is not None:
        geo_metrics = compute_pairwise_retrieval_metrics(
            image_emb,
            geo_emb,
            a_to_b_prefix="image_to_geo",
            b_to_a_prefix="geo_to_image",
        )
        for key, value in geo_metrics.items():
            if key == "num_samples":
                continue
            metrics[key] = float(value)
    if args.holdout_split is not None:
        metrics["holdout_split"] = args.holdout_split
    metrics["geo_context"] = geo_context
    metrics["used_ema_weights"] = bool(use_ema_weights)

    print(json.dumps(metrics, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2))

    if args.export_embeddings:
        args.embeddings_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "image_embeddings": image_emb,
            "text_embeddings": text_emb,
            "texts": text_rows,
            "metrics": metrics,
            "alignment_checkpoint": str(args.alignment_checkpoint),
            "geo_context": geo_context,
        }
        if geo_emb is not None:
            payload["geo_embeddings"] = geo_emb
        torch.save(payload, args.embeddings_out)
        print(f"[eval-marsclip] saved embeddings: {args.embeddings_out}")


if __name__ == "__main__":
    main()
