"""Paired multiscale multimodal alignment model components for MarsCLIP."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from clip.marsclip_mae import MarsMaskedAutoencoder, extract_scale_values
from clip.marsclip_model import TextTransformerTower


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


class ProjectionHead(nn.Module):
    """Small projection head used after each modality-specific backbone."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianFourierFeatures(nn.Module):
    """Fixed Gaussian Fourier features for low-dimensional continuous inputs."""

    def __init__(self, in_dim: int, mapping_size: int, sigma: float = 1.0) -> None:
        super().__init__()
        if mapping_size <= 0:
            raise ValueError("mapping_size must be positive.")
        weight = torch.randn(in_dim, mapping_size) * float(sigma)
        self.register_buffer("weight", weight, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = 2.0 * math.pi * x @ self.weight
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class GeometryEncoder(nn.Module):
    """Encode acquisition geometry with Fourier features and periodic metadata."""

    def __init__(
        self,
        *,
        viewing_dim: int = 13,
        angle_dim: int = 3,
        fourier_dim: int = 32,
        hidden_dim: int = 256,
        embed_dim: int = 256,
        sigma: float = 4.0,
    ) -> None:
        super().__init__()
        if viewing_dim < angle_dim:
            raise ValueError("viewing_dim must be >= angle_dim.")
        self.angle_dim = angle_dim
        self.fourier = GaussianFourierFeatures(angle_dim, fourier_dim, sigma=sigma)
        base_dim = (2 * fourier_dim) + (viewing_dim - angle_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(base_dim),
            nn.Linear(base_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, viewing_features: torch.Tensor) -> torch.Tensor:
        angle_features = viewing_features[:, : self.angle_dim]
        residual_features = viewing_features[:, self.angle_dim :]
        x = torch.cat([self.fourier(angle_features), residual_features], dim=-1)
        return self.net(x)


class SineLayer(nn.Module):
    """SIREN-style sine activation block."""

    def __init__(self, in_dim: int, out_dim: int, *, w0: float = 1.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.w0 = float(w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


class SphericalLocationEncoder(nn.Module):
    """Encode lon/lat using spherical coordinates and a sine-network head."""

    def __init__(
        self,
        *,
        num_frequencies: int = 4,
        hidden_dim: int = 256,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        base_dim = 5 + (10 * self.num_frequencies)
        self.net = nn.Sequential(
            SineLayer(base_dim, hidden_dim, w0=8.0),
            SineLayer(hidden_dim, hidden_dim, w0=1.0),
            nn.Linear(hidden_dim, embed_dim),
        )

    @staticmethod
    def _to_sphere(location: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lon = torch.deg2rad(location[:, 0])
        lat = torch.deg2rad(location[:, 1])
        x = torch.cos(lat) * torch.cos(lon)
        y = torch.cos(lat) * torch.sin(lon)
        z = torch.sin(lat)
        return lon, lat, torch.stack([x, y, z], dim=-1)

    def forward(self, location: torch.Tensor) -> torch.Tensor:
        lon, lat, xyz = self._to_sphere(location)
        features = [xyz, lon.unsqueeze(-1), lat.unsqueeze(-1)]
        for k in range(self.num_frequencies):
            frequency = 2.0 ** k
            features.extend(
                [
                    torch.sin(frequency * lon).unsqueeze(-1),
                    torch.cos(frequency * lon).unsqueeze(-1),
                    torch.sin(frequency * lat).unsqueeze(-1),
                    torch.cos(frequency * lat).unsqueeze(-1),
                    torch.sin(frequency * xyz),
                    torch.cos(frequency * xyz),
                ]
            )
        return self.net(torch.cat(features, dim=-1))


class TextLocationFusion(nn.Module):
    """Fuse text and location embeddings with a gated attention-style update."""

    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, text_embedding: torch.Tensor, location_embedding: torch.Tensor) -> torch.Tensor:
        query = text_embedding.unsqueeze(1)
        key_value = torch.stack([text_embedding, location_embedding], dim=1)
        cross, _ = self.cross_attn(query, key_value, key_value, need_weights=False)
        gated = torch.sigmoid(self.gate) * cross.squeeze(1)
        return self.norm(text_embedding + gated)


class GeometryFiLM(nn.Module):
    """Apply geometry-conditioned affine modulation to visual embeddings."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(embed_dim, embed_dim)
        self.beta = nn.Linear(embed_dim, embed_dim)

    def forward(self, visual_embedding: torch.Tensor, geometry_embedding: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(geometry_embedding)
        beta = self.beta(geometry_embedding)
        return (1.0 + gamma) * visual_embedding + beta


class MAEVisualBackbone(nn.Module):
    """Reuse the Stage A MAE encoder as a visual backbone for alignment."""

    def __init__(self, stage_a_backbone: MarsMaskedAutoencoder) -> None:
        super().__init__()
        self.stage_a_backbone = stage_a_backbone
        self.output_dim = int(stage_a_backbone.patch_embed.out_channels)

    def forward(
        self,
        image: torch.Tensor,
        valid_mask: torch.Tensor,
        scale_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded = self.stage_a_backbone.encode_image(
            image,
            valid_mask,
            extract_scale_values(scale_features),
        )
        return {
            "embedding": encoded.pooled_embedding,
            "patch_valid_mask": encoded.patch_valid_mask,
            "visible_mask": encoded.visible_mask,
        }


@dataclass
class PairedMultiscaleAlignmentOutput:
    local_image_embedding: torch.Tensor
    global_image_embedding: torch.Tensor
    text_embedding: torch.Tensor
    location_embedding: torch.Tensor
    geometry_embedding: torch.Tensor
    context_embedding: torch.Tensor
    local_patch_valid_mask: torch.Tensor
    global_patch_valid_mask: torch.Tensor
    losses: dict[str, torch.Tensor]
    loss: torch.Tensor


class PairedMultiscaleAlignmentModel(nn.Module):
    """Literature-informed paired multiscale alignment model for MarsCLIP."""

    def __init__(
        self,
        *,
        visual_backbone: MarsMaskedAutoencoder,
        vocab_size: int,
        embed_dim: int = 256,
        text_max_length: int = 64,
        text_hidden_dim: int = 256,
        text_depth: int = 2,
        text_heads: int = 8,
        location_hidden_dim: int = 256,
        location_frequencies: int = 4,
        geometry_hidden_dim: int = 256,
        geometry_fourier_dim: int = 32,
        fusion_heads: int = 4,
        loss_weight_local_context: float = 1.0,
        loss_weight_global_context: float = 1.0,
        loss_weight_cross_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.visual_backbone = MAEVisualBackbone(visual_backbone)
        visual_dim = self.visual_backbone.output_dim

        self.local_projector = ProjectionHead(visual_dim, embed_dim)
        self.global_projector = ProjectionHead(visual_dim, embed_dim)
        self.geometry_encoder = GeometryEncoder(
            viewing_dim=13,
            fourier_dim=geometry_fourier_dim,
            hidden_dim=geometry_hidden_dim,
            embed_dim=embed_dim,
        )
        self.location_encoder = SphericalLocationEncoder(
            num_frequencies=location_frequencies,
            hidden_dim=location_hidden_dim,
            embed_dim=embed_dim,
        )
        self.text_tower = TextTransformerTower(
            vocab_size=vocab_size,
            max_length=text_max_length,
            hidden_dim=text_hidden_dim,
            depth=text_depth,
            num_heads=text_heads,
            embed_dim=embed_dim,
        )
        self.text_projector = ProjectionHead(embed_dim, embed_dim)
        self.location_projector = ProjectionHead(embed_dim, embed_dim)
        self.geometry_projector = ProjectionHead(embed_dim, embed_dim)
        self.context_fusion = TextLocationFusion(embed_dim, num_heads=fusion_heads)
        self.context_projector = ProjectionHead(embed_dim, embed_dim)
        self.geometry_film = GeometryFiLM(embed_dim)

        self.local_context_logit_scale = nn.Parameter(
            torch.tensor(math.log(1 / 0.07), dtype=torch.float32)
        )
        self.global_context_logit_scale = nn.Parameter(
            torch.tensor(math.log(1 / 0.07), dtype=torch.float32)
        )
        self.cross_scale_logit_scale = nn.Parameter(
            torch.tensor(math.log(1 / 0.07), dtype=torch.float32)
        )

        self.loss_weight_local_context = float(loss_weight_local_context)
        self.loss_weight_global_context = float(loss_weight_global_context)
        self.loss_weight_cross_scale = float(loss_weight_cross_scale)

    def forward(self, batch: dict[str, Any]) -> PairedMultiscaleAlignmentOutput:
        local_visual = self.visual_backbone(
            batch["local_image"],
            batch["local_valid_mask"],
            batch["local_scale_features"],
        )
        global_visual = self.visual_backbone(
            batch["global_image"],
            batch["global_valid_mask"],
            batch["global_scale_features"],
        )

        geometry_embedding = self.geometry_projector(
            self.geometry_encoder(batch["viewing_features"])
        )
        text_embedding = self.text_projector(
            self.text_tower(batch["input_ids"], batch["attention_mask"])
        )
        location_embedding = self.location_projector(
            self.location_encoder(batch["location"])
        )
        context_embedding = self.context_projector(
            self.context_fusion(text_embedding, location_embedding)
        )

        local_embedding = self.local_projector(local_visual["embedding"])
        global_embedding = self.global_projector(global_visual["embedding"])
        local_embedding = self.geometry_film(local_embedding, geometry_embedding)
        global_embedding = self.geometry_film(global_embedding, geometry_embedding)

        local_embedding = F.normalize(local_embedding, dim=-1)
        global_embedding = F.normalize(global_embedding, dim=-1)
        text_embedding = F.normalize(text_embedding, dim=-1)
        location_embedding = F.normalize(location_embedding, dim=-1)
        geometry_embedding = F.normalize(geometry_embedding, dim=-1)
        context_embedding = F.normalize(context_embedding, dim=-1)

        losses = {
            "local_context": symmetric_contrastive_loss(
                local_embedding,
                context_embedding,
                self.local_context_logit_scale,
            ),
            "global_context": symmetric_contrastive_loss(
                global_embedding,
                context_embedding,
                self.global_context_logit_scale,
            ),
            "cross_scale": symmetric_contrastive_loss(
                local_embedding,
                global_embedding,
                self.cross_scale_logit_scale,
            ),
        }
        loss = (
            self.loss_weight_local_context * losses["local_context"]
            + self.loss_weight_global_context * losses["global_context"]
            + self.loss_weight_cross_scale * losses["cross_scale"]
        )

        return PairedMultiscaleAlignmentOutput(
            local_image_embedding=local_embedding,
            global_image_embedding=global_embedding,
            text_embedding=text_embedding,
            location_embedding=location_embedding,
            geometry_embedding=geometry_embedding,
            context_embedding=context_embedding,
            local_patch_valid_mask=local_visual["patch_valid_mask"],
            global_patch_valid_mask=global_visual["patch_valid_mask"],
            losses=losses,
            loss=loss,
        )
