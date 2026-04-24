"""Train a Stage-2 aligner between Mars text and MAE image embeddings.

This script:
1) loads Mars patch samples from ``clip.marsclip_patches``,
2) loads a SatMAE encoder using ``clip.satmae_bridge``,
3) encodes rationale text with ``stage2.T5_encoder``,
4) learns projection heads with CLIP-style contrastive loss.

The trained aligner can then be reused for downstream retrieval, clustering,
and lightweight classifier heads.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from clip.marsclip_patches import MarsCLIPPatchDataset, load_patch_records
from clip.satmae_bridge import build_satmae_model
from stage2.T5_encoder import T5Encoder


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _extract_model_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model_state", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint  # type: ignore[return-value]
    raise ValueError("Checkpoint does not contain a recognized model state dict.")


def load_satmae_encoder(
    *,
    model_name: str,
    checkpoint_path: pathlib.Path,
    image_size: int,
    patch_size_px: int,
    in_chans: int = 3,
    freeze: bool = True,
) -> nn.Module:
    """Load a SatMAE model and return it ready for feature extraction."""
    model = build_satmae_model(
        model_name,
        img_size=image_size,
        patch_size=patch_size_px,
        in_chans=in_chans,
        norm_pix_loss=True,
    )
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = _extract_model_state(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[align] warning: missing MAE keys: {len(missing)}")
    if unexpected:
        print(f"[align] warning: unexpected MAE keys: {len(unexpected)}")

    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.eval()
    return model


@torch.no_grad()
def encode_image_with_satmae_encoder(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Encode images with SatMAE encoder (CLS token output)."""
    tokens = model.patch_embed(images)
    tokens = tokens + model.pos_embed[:, 1:, :]
    cls = model.cls_token + model.pos_embed[:, :1, :]
    cls = cls.expand(images.shape[0], -1, -1)
    hidden = torch.cat((cls, tokens), dim=1)
    for block in model.blocks:
        hidden = block(hidden)
    hidden = model.norm(hidden)
    return hidden[:, 0]  # CLS embedding


def collate_patch_text(samples: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([sample["image"] for sample in samples], dim=0)
    texts = [str(sample.get("rationale_raw", "")) for sample in samples]
    metadata = [dict(sample.get("metadata", {})) for sample in samples]
    return {"image": images, "text": texts, "metadata": metadata}


def symmetric_contrastive_loss(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    logits = torch.matmul(image_embeddings, text_embeddings.T) * logit_scale.exp()
    target = torch.arange(image_embeddings.shape[0], device=image_embeddings.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


@dataclass
class AlignmentModel(nn.Module):
    """Projection heads that map image/text into a shared embedding space."""

    image_projector: nn.Module
    text_projector: nn.Module
    logit_scale: nn.Parameter

    def __init__(self, image_dim: int, text_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.image_projector = nn.Linear(image_dim, embed_dim)
        self.text_projector = nn.Linear(text_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(1 / 0.07)))))

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_embeddings = F.normalize(self.image_projector(image_features), dim=1)
        text_embeddings = F.normalize(self.text_projector(text_features), dim=1)
        return image_embeddings, text_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Align Mars text with MAE image embeddings.")
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("/scratch/mars_hirise"))
    parser.add_argument("--bbox", type=float, nargs=4, default=(-136.0, 12.0, -124.0, 24.0))
    parser.add_argument("--patch-size-deg", type=float, default=0.005)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size-px", type=int, default=16)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--patch-records-path", type=pathlib.Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--mae-model", type=str, default="mae_vit_base_patch16")
    parser.add_argument("--mae-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--freeze-mae", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text-model", type=str, default="google-t5/t5-base")
    parser.add_argument("--text-max-length", type=int, default=128)
    parser.add_argument("--train-text-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("stage2_align_runs"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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
        shuffle=True,
        num_workers=0,
        collate_fn=collate_patch_text,
    )

    mae_encoder = load_satmae_encoder(
        model_name=args.mae_model,
        checkpoint_path=args.mae_checkpoint,
        image_size=args.image_size,
        patch_size_px=args.patch_size_px,
        freeze=args.freeze_mae,
    ).to(device)

    text_encoder = T5Encoder(
        model_name=args.text_model,
        max_length=args.text_max_length,
        trainable=args.train_text_encoder,
    ).to(device)

    with torch.no_grad():
        warmup_batch = next(iter(dataloader))
        warmup_images = warmup_batch["image"].to(device)
        image_dim = int(encode_image_with_satmae_encoder(mae_encoder, warmup_images).shape[-1])
        text_dim = int(text_encoder(warmup_batch["text"], device=device)[0].shape[-1])

    aligner = AlignmentModel(image_dim=image_dim, text_dim=text_dim, embed_dim=args.embed_dim).to(device)

    parameters: list[nn.Parameter] = list(aligner.parameters())
    if args.train_text_encoder:
        parameters += [p for p in text_encoder.parameters() if p.requires_grad]
    if not args.freeze_mae:
        parameters += [p for p in mae_encoder.parameters() if p.requires_grad]

    optimizer = AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    global_step = 0
    for epoch in range(args.epochs):
        aligner.train()
        if args.train_text_encoder:
            text_encoder.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in dataloader:
            images = batch["image"].to(device)
            texts = batch["text"]

            if args.freeze_mae:
                with torch.no_grad():
                    image_features = encode_image_with_satmae_encoder(mae_encoder, images)
            else:
                image_features = encode_image_with_satmae_encoder(mae_encoder, images)

            if args.train_text_encoder:
                text_features, _ = text_encoder(texts, device=device)
            else:
                with torch.no_grad():
                    text_features, _ = text_encoder(texts, device=device)

            image_embeddings, text_embeddings = aligner(image_features, text_features)
            loss = symmetric_contrastive_loss(image_embeddings, text_embeddings, aligner.logit_scale)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_steps += 1
            epoch_loss += float(loss.detach().cpu())

        avg_loss = epoch_loss / max(epoch_steps, 1)
        history.append({"epoch": float(epoch + 1), "loss": avg_loss})
        print(f"[align] epoch={epoch + 1}/{args.epochs} loss={avg_loss:.6f}")

    ckpt_path = out_dir / "text_mae_alignment.pt"
    torch.save(
        {
            "config": vars(args),
            "history": history,
            "aligner_state": aligner.state_dict(),
            "text_encoder_state": text_encoder.state_dict() if args.train_text_encoder else None,
            "mae_encoder_state": mae_encoder.state_dict() if not args.freeze_mae else None,
            "text_model_name": args.text_model,
            "mae_model_name": args.mae_model,
            "mae_checkpoint": str(args.mae_checkpoint),
            "image_dim": image_dim,
            "text_dim": text_dim,
            "embed_dim": int(args.embed_dim),
        },
        ckpt_path,
    )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[align] saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
