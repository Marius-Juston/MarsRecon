"""Facebook MAE-style masked autoencoder adapted to MarsCLIP patches.

This module follows the structure of the upstream facebookresearch/mae code:

- encoder processes only visible patches
- decoder reconstructs all patches after ids_restore unshuffling
- optional ``norm_pix_loss`` normalizes each target patch independently

The Mars adaptation is that invalid HiRISE regions are never used as training
targets, and random masking only samples from valid patches.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from clip.marsclip_patches import DEFAULT_PATCH_VALID_FRACTION


def _get_1d_sincos_pos_embed(embed_dim: int, positions: torch.Tensor) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for sin/cos positional embedding.")
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=positions.device)
    omega = 1.0 / (10000 ** (omega / float(embed_dim // 2)))
    out = positions.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat([out.sin(), out.cos()], dim=1)


def build_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    *,
    cls_token: bool,
) -> torch.Tensor:
    """Create a 2D sine/cosine positional embedding like the upstream MAE repo."""
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even.")
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape(2, 1, grid_size, grid_size)

    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0].reshape(-1))
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1].reshape(-1))
    pos_embed = torch.cat([emb_h, emb_w], dim=1)
    if cls_token:
        cls = torch.zeros(1, embed_dim, dtype=pos_embed.dtype)
        pos_embed = torch.cat([cls, pos_embed], dim=0)
    return pos_embed.unsqueeze(0)


def patchify(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert an image batch ``(B, C, H, W)`` into flattened patch vectors."""
    if image.ndim != 4:
        raise ValueError("image must have shape (B, C, H, W)")
    batch, channels, height, width = image.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("image height and width must be divisible by patch_size.")

    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.reshape(
        batch,
        (height // patch_size) * (width // patch_size),
        channels * patch_size * patch_size,
    )


def compute_patch_valid_fraction(valid_mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Compute the valid-pixel fraction for each patch in a mask batch."""
    if valid_mask.ndim != 3:
        raise ValueError("valid_mask must have shape (B, H, W)")
    batch, height, width = valid_mask.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("mask height and width must be divisible by patch_size.")

    patches = valid_mask.to(torch.float32).unfold(1, patch_size, patch_size).unfold(
        2, patch_size, patch_size
    )
    return patches.mean(dim=(-1, -2)).reshape(batch, -1)


def compute_valid_patch_mask(
    valid_mask: torch.Tensor,
    patch_size: int,
    *,
    min_valid_fraction: float = DEFAULT_PATCH_VALID_FRACTION,
) -> torch.Tensor:
    """Mark patches as valid only when their valid-pixel fraction clears a threshold."""
    if not (0.0 <= min_valid_fraction <= 1.0):
        raise ValueError("min_valid_fraction must satisfy 0 <= value <= 1.")
    return compute_patch_valid_fraction(valid_mask, patch_size) >= min_valid_fraction


def patchify_valid_mask(
    valid_mask: torch.Tensor,
    patch_size: int,
    *,
    channels: int = 1,
) -> torch.Tensor:
    """Expand a pixel-valid mask into one boolean per flattened patch value."""
    expanded = valid_mask.unsqueeze(1).expand(-1, int(channels), -1, -1).to(torch.float32)
    patch_values = patchify(expanded, patch_size)
    return patch_values > 0.5


def normalize_patch_targets(
    target_patches: torch.Tensor,
    valid_pixel_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize each target patch independently using valid pixels only."""
    if target_patches.shape != valid_pixel_mask.shape:
        raise ValueError("target_patches and valid_pixel_mask must have matching shape.")
    valid = valid_pixel_mask.to(target_patches.dtype)
    counts = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (target_patches * valid).sum(dim=-1, keepdim=True) / counts
    variance = (((target_patches - mean) * valid).pow(2)).sum(dim=-1, keepdim=True) / counts
    std = (variance + float(eps)).sqrt()
    normalized = (target_patches - mean) / std
    return torch.where(valid_pixel_mask, normalized, torch.zeros_like(normalized))


def _random_masking_with_valid_mask(
    tokens: torch.Tensor,
    patch_valid_mask: torch.Tensor,
    mask_ratio: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """MAE-style random masking, restricted to valid patches only."""
    if not (0.0 <= mask_ratio < 1.0):
        raise ValueError("mask_ratio must satisfy 0 <= value < 1.")

    batch, num_patches, dim = tokens.shape
    masked_batches: list[torch.Tensor] = []
    keep_indices_list: list[torch.Tensor] = []
    ids_restore = torch.zeros(batch, num_patches, dtype=torch.long, device=tokens.device)
    mask = torch.ones(batch, num_patches, dtype=torch.float32, device=tokens.device)
    max_keep = 0

    for batch_idx in range(batch):
        valid_idx = torch.nonzero(patch_valid_mask[batch_idx], as_tuple=False).flatten()
        invalid_idx = torch.nonzero(~patch_valid_mask[batch_idx], as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            keep_idx = valid_idx
            removed_idx = valid_idx
        else:
            len_keep = max(1, int(valid_idx.numel() * (1.0 - mask_ratio)))
            noise = torch.rand(valid_idx.numel(), generator=generator, device=tokens.device)
            order = torch.argsort(noise)
            keep_idx = valid_idx[order[:len_keep]]
            removed_idx = valid_idx[order[len_keep:]]
            mask[batch_idx, keep_idx] = 0.0

        shuffle = torch.cat([keep_idx, removed_idx, invalid_idx], dim=0)
        restore = torch.empty(num_patches, dtype=torch.long, device=tokens.device)
        restore[shuffle] = torch.arange(num_patches, device=tokens.device)
        ids_restore[batch_idx] = restore

        keep_indices_list.append(keep_idx)
        masked_batches.append(tokens[batch_idx, keep_idx])
        max_keep = max(max_keep, keep_idx.numel())

    masked_tokens = tokens.new_zeros((batch, max_keep, dim))
    visible_padding_mask = torch.ones(batch, max_keep, dtype=torch.bool, device=tokens.device)
    for batch_idx, sample_tokens in enumerate(masked_batches):
        if sample_tokens.numel() == 0:
            continue
        length = sample_tokens.shape[0]
        masked_tokens[batch_idx, :length] = sample_tokens
        visible_padding_mask[batch_idx, :length] = False

    return masked_tokens, mask, ids_restore, keep_indices_list, visible_padding_mask


@dataclass
class FacebookMAEOutput:
    latent: torch.Tensor
    pred: torch.Tensor
    mask: torch.Tensor
    ids_restore: torch.Tensor
    patch_valid_mask: torch.Tensor
    valid_pixel_mask: torch.Tensor
    loss: torch.Tensor


class FacebookMaskedAutoencoderViT(nn.Module):
    """Facebook MAE-style ViT adapted to MarsCLIP patch batches."""

    def __init__(
        self,
        *,
        image_size: int = 64,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_pix_loss: bool = False,
        min_valid_fraction: float = DEFAULT_PATCH_VALID_FRACTION,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.in_chans = int(in_chans)
        self.embed_dim = int(embed_dim)
        self.decoder_embed_dim = int(decoder_embed_dim)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.norm_pix_loss = bool(norm_pix_loss)
        self.min_valid_fraction = float(min_valid_fraction)

        self.patch_embed = nn.Conv2d(
            in_channels=self.in_chans,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            build_2d_sincos_pos_embed(
                self.embed_dim,
                self.image_size // self.patch_size,
                cls_token=True,
            ),
            requires_grad=False,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(num_heads),
            dim_feedforward=int(self.embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            layer_norm_eps=1e-6,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(depth),
            norm=nn.LayerNorm(self.embed_dim, eps=1e-6),
        )

        self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            build_2d_sincos_pos_embed(
                self.decoder_embed_dim,
                self.image_size // self.patch_size,
                cls_token=True,
            ),
            requires_grad=False,
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.decoder_embed_dim,
            nhead=int(decoder_num_heads),
            dim_feedforward=int(self.decoder_embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            layer_norm_eps=1e-6,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=int(decoder_depth),
            norm=nn.LayerNorm(self.decoder_embed_dim, eps=1e-6),
        )
        self.decoder_pred = nn.Linear(
            self.decoder_embed_dim,
            self.patch_size * self.patch_size * self.in_chans,
            bias=True,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight.view(module.weight.shape[0], -1))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward_encoder(
        self,
        imgs: torch.Tensor,
        valid_mask: torch.Tensor,
        mask_ratio: float,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, 1:, :]

        patch_valid_mask = compute_valid_patch_mask(
            valid_mask,
            self.patch_size,
            min_valid_fraction=self.min_valid_fraction,
        )
        x_masked, mask, ids_restore, _, visible_padding_mask = _random_masking_with_valid_mask(
            x,
            patch_valid_mask,
            mask_ratio,
            generator=generator,
        )

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(imgs.shape[0], -1, -1)
        x_masked = torch.cat([cls_tokens, x_masked], dim=1)

        encoder_padding_mask = torch.cat(
            [
                torch.zeros(imgs.shape[0], 1, dtype=torch.bool, device=imgs.device),
                visible_padding_mask,
            ],
            dim=1,
        )
        latent = self.encoder(x_masked, src_key_padding_mask=encoder_padding_mask)
        return latent, mask, ids_restore, patch_valid_mask, encoder_padding_mask

    def forward_decoder(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(latent)
        batch = x.shape[0]

        mask_tokens = self.mask_token.repeat(
            batch,
            ids_restore.shape[1] + 1 - x.shape[1],
            1,
        )
        x_visible = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_visible = torch.gather(
            x_visible,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]),
        )
        x = torch.cat([x[:, :1, :], x_visible], dim=1)
        x = x + self.decoder_pos_embed
        x = self.decoder(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :]

    def forward_loss(
        self,
        imgs: torch.Tensor,
        valid_mask: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        patch_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = patchify(imgs, self.patch_size)
        valid_pixel_mask = patchify_valid_mask(valid_mask, self.patch_size, channels=self.in_chans)
        if self.norm_pix_loss:
            target = normalize_patch_targets(target, valid_pixel_mask)

        valid = valid_pixel_mask.to(pred.dtype)
        valid_counts = valid.sum(dim=-1).clamp_min(1.0)
        per_patch_loss = ((pred - target).pow(2) * valid).sum(dim=-1) / valid_counts
        loss_mask = mask.bool() & patch_valid_mask
        if loss_mask.any():
            loss = (per_patch_loss * loss_mask.to(per_patch_loss.dtype)).sum() / loss_mask.sum()
        else:
            loss = per_patch_loss.sum() * 0.0
        return loss, valid_pixel_mask

    def forward(
        self,
        imgs: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        mask_ratio: float = 0.75,
        generator: torch.Generator | None = None,
    ) -> FacebookMAEOutput:
        latent, mask, ids_restore, patch_valid_mask, _ = self.forward_encoder(
            imgs,
            valid_mask,
            mask_ratio,
            generator=generator,
        )
        pred = self.forward_decoder(latent, ids_restore)
        loss, valid_pixel_mask = self.forward_loss(imgs, valid_mask, pred, mask, patch_valid_mask)
        return FacebookMAEOutput(
            latent=latent,
            pred=pred,
            mask=mask,
            ids_restore=ids_restore,
            patch_valid_mask=patch_valid_mask,
            valid_pixel_mask=valid_pixel_mask,
            loss=loss,
        )


FB_MAE_MODEL_PRESETS: dict[str, dict[str, int]] = {
    "mae_vit_small_patch16": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "decoder_embed_dim": 256,
        "decoder_depth": 4,
        "decoder_num_heads": 8,
    },
    "mae_vit_base_patch16": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "decoder_embed_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
    },
    "mae_vit_large_patch16": {
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "decoder_embed_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
    },
}


def build_fb_mae_model(model_name: str, **kwargs: object) -> FacebookMaskedAutoencoderViT:
    """Build a Facebook-style MAE preset adapted to Mars patches."""
    if model_name not in FB_MAE_MODEL_PRESETS:
        available = ", ".join(sorted(FB_MAE_MODEL_PRESETS))
        raise ValueError(f"Unknown MAE model '{model_name}'. Available presets: {available}")
    return FacebookMaskedAutoencoderViT(**FB_MAE_MODEL_PRESETS[model_name], **kwargs)
