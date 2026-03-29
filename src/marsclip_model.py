"""First tri-modal MarsCLIP model slice with valid-token masking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _masked_mean(tokens: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    """Average token embeddings over tokens marked True in keep_mask."""
    weights = keep_mask.unsqueeze(-1).to(tokens.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denom


def compute_patch_valid_mask(valid_mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Aggregate a pixel valid mask into one boolean per image patch."""
    if valid_mask.ndim != 3:
        raise ValueError("valid_mask must have shape (B, H, W)")
    b, h, w = valid_mask.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError("Image size must be divisible by patch_size.")

    patches = valid_mask.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    return patches.any(dim=-1).any(dim=-1).reshape(b, -1)


def sample_patch_keep_mask(
    patch_valid_mask: torch.Tensor,
    mask_ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Randomly keep a subset of valid image patches, never selecting invalid ones."""
    if not (0.0 <= mask_ratio < 1.0):
        raise ValueError("mask_ratio must satisfy 0 <= mask_ratio < 1.")

    keep_mask = torch.zeros_like(patch_valid_mask, dtype=torch.bool)
    for i in range(patch_valid_mask.shape[0]):
        valid_idx = torch.nonzero(patch_valid_mask[i], as_tuple=False).flatten()
        if len(valid_idx) == 0:
            continue
        n_keep = max(1, int(math.ceil(len(valid_idx) * (1.0 - mask_ratio))))
        perm = torch.randperm(len(valid_idx), generator=generator, device=valid_idx.device)
        selected = valid_idx[perm[:n_keep]]
        keep_mask[i, selected] = True
    return keep_mask


class ImageViTTower(nn.Module):
    """A lightweight ViT-style image encoder with valid-patch masking."""

    def __init__(
        self,
        *,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=hidden_dim * 4,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(
        self,
        image: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        mask_ratio: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        tokens = self.patch_embed(image).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed

        patch_valid_mask = compute_patch_valid_mask(valid_mask, self.patch_size)
        patch_keep_mask = sample_patch_keep_mask(
            patch_valid_mask,
            mask_ratio=mask_ratio,
            generator=generator,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=~patch_keep_mask)
        pooled = _masked_mean(encoded, patch_keep_mask)
        embedding = self.proj(pooled)
        return {
            "embedding": embedding,
            "patch_valid_mask": patch_valid_mask,
            "patch_keep_mask": patch_keep_mask,
        }


class TextTransformerTower(nn.Module):
    """A lightweight text encoder for raw/expanded rationale text."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_length: int = 64,
        hidden_dim: int = 256,
        depth: int = 2,
        num_heads: int = 8,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_length, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=hidden_dim * 4,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embed(input_ids) + self.pos_embed[:, : input_ids.shape[1]]
        encoded = self.encoder(tokens, src_key_padding_mask=~attention_mask)
        pooled = _masked_mean(encoded, attention_mask)
        return self.proj(pooled)


class GeoContextTower(nn.Module):
    """Encode geo, scale, viewing, and quality metadata into one embedding."""

    def __init__(
        self,
        *,
        geo_dim: int,
        scale_dim: int,
        viewing_dim: int,
        quality_dim: int,
        hidden_dim: int = 256,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        total_dim = geo_dim + scale_dim + viewing_dim + quality_dim
        self.net = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        geo_features: torch.Tensor,
        scale_features: torch.Tensor,
        viewing_features: torch.Tensor,
        quality_features: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [geo_features, scale_features, viewing_features, quality_features],
            dim=1,
        )
        return self.net(x)


def symmetric_contrastive_loss(
    a: torch.Tensor,
    b: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """CLIP-style symmetric contrastive loss for one embedding pair."""
    logits = torch.matmul(a, b.T) * logit_scale.exp()
    targets = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)
    )


@dataclass
class MarsCLIPOutput:
    image_embedding: torch.Tensor
    text_embedding: torch.Tensor
    geo_embedding: torch.Tensor
    patch_valid_mask: torch.Tensor
    patch_keep_mask: torch.Tensor
    losses: dict[str, torch.Tensor]
    loss: torch.Tensor


class MarsCLIPModel(nn.Module):
    """First-pass tri-modal contrastive model for MarsCLIP."""

    def __init__(
        self,
        *,
        vocab_size: int,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_dim: int = 256,
        embed_dim: int = 256,
        text_max_length: int = 64,
        geo_dim: int = 8,
        scale_dim: int = 8,
        viewing_dim: int = 13,
        quality_dim: int = 7,
        mask_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.image_tower = ImageViTTower(
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
        )
        self.text_tower = TextTransformerTower(
            vocab_size=vocab_size,
            max_length=text_max_length,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
        )
        self.geo_tower = GeoContextTower(
            geo_dim=geo_dim,
            scale_dim=scale_dim,
            viewing_dim=viewing_dim,
            quality_dim=quality_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), dtype=torch.float32))

    def forward(
        self,
        batch: dict[str, Any],
        *,
        mask_ratio: float | None = None,
        generator: torch.Generator | None = None,
    ) -> MarsCLIPOutput:
        image_out = self.image_tower(
            batch["image"],
            batch["valid_mask"],
            mask_ratio=self.mask_ratio if mask_ratio is None else mask_ratio,
            generator=generator,
        )
        text_embedding = self.text_tower(batch["input_ids"], batch["attention_mask"])
        geo_embedding = self.geo_tower(
            batch["geo_features"],
            batch["scale_features"],
            batch["viewing_features"],
            batch["quality_features"],
        )

        image_embedding = F.normalize(image_out["embedding"], dim=1)
        text_embedding = F.normalize(text_embedding, dim=1)
        geo_embedding = F.normalize(geo_embedding, dim=1)

        losses = {
            "image_text": symmetric_contrastive_loss(
                image_embedding, text_embedding, self.logit_scale
            ),
            "image_geo": symmetric_contrastive_loss(
                image_embedding, geo_embedding, self.logit_scale
            ),
            "text_geo": symmetric_contrastive_loss(
                text_embedding, geo_embedding, self.logit_scale
            ),
        }
        total_loss = losses["image_text"] + losses["image_geo"] + losses["text_geo"]

        return MarsCLIPOutput(
            image_embedding=image_embedding,
            text_embedding=text_embedding,
            geo_embedding=geo_embedding,
            patch_valid_mask=image_out["patch_valid_mask"],
            patch_keep_mask=image_out["patch_keep_mask"],
            losses=losses,
            loss=total_loss,
        )
