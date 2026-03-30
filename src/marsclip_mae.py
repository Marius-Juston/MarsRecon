"""Stage A masked autoencoder components for MarsCLIP."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from marsclip_patches import DEFAULT_PATCH_VALID_FRACTION


def patchify(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert an image batch ``(B, C, H, W)`` into flattened patch vectors."""
    if image.ndim != 4:
        raise ValueError("image must have shape (B, C, H, W)")
    batch, channels, height, width = image.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("Image height and width must be divisible by patch_size.")

    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.reshape(
        batch,
        (height // patch_size) * (width // patch_size),
        channels * patch_size * patch_size,
    )


def unpatchify(
    patches: torch.Tensor,
    patch_size: int,
    *,
    channels: int,
    image_size: int | tuple[int, int] | None = None,
) -> torch.Tensor:
    """Rebuild an image batch ``(B, C, H, W)`` from flattened patch vectors."""
    if patches.ndim != 3:
        raise ValueError("patches must have shape (B, N, D)")
    batch, num_patches, patch_dim = patches.shape
    expected_dim = channels * patch_size * patch_size
    if patch_dim != expected_dim:
        raise ValueError(
            f"patch dimension {patch_dim} does not match channels*patch_size^2={expected_dim}"
        )

    if image_size is None:
        grid_h = int(math.isqrt(num_patches))
        grid_w = grid_h
        if grid_h * grid_w != num_patches:
            raise ValueError("num_patches must form a square grid when image_size is omitted.")
    else:
        if isinstance(image_size, int):
            image_h = image_w = int(image_size)
        else:
            image_h, image_w = int(image_size[0]), int(image_size[1])
        if image_h % patch_size != 0 or image_w % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        grid_h = image_h // patch_size
        grid_w = image_w // patch_size
        if grid_h * grid_w != num_patches:
            raise ValueError("image_size does not match the number of patches.")

    image = patches.reshape(batch, grid_h, grid_w, channels, patch_size, patch_size)
    image = image.permute(0, 3, 1, 4, 2, 5).contiguous()
    return image.reshape(batch, channels, grid_h * patch_size, grid_w * patch_size)


def expand_patch_mask(
    patch_mask: torch.Tensor,
    patch_size: int,
    *,
    image_size: int | tuple[int, int] | None = None,
) -> torch.Tensor:
    """Expand a patch mask ``(B, N)`` back to an image mask ``(B, H, W)``."""
    if patch_mask.ndim != 2:
        raise ValueError("patch_mask must have shape (B, N)")
    batch, num_patches = patch_mask.shape

    if image_size is None:
        grid_h = int(math.isqrt(num_patches))
        grid_w = grid_h
        if grid_h * grid_w != num_patches:
            raise ValueError("num_patches must form a square grid when image_size is omitted.")
    else:
        if isinstance(image_size, int):
            image_h = image_w = int(image_size)
        else:
            image_h, image_w = int(image_size[0]), int(image_size[1])
        if image_h % patch_size != 0 or image_w % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        grid_h = image_h // patch_size
        grid_w = image_w // patch_size
        if grid_h * grid_w != num_patches:
            raise ValueError("image_size does not match the number of patches.")

    grid = patch_mask.reshape(batch, grid_h, grid_w)
    return grid.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)


def compute_patch_valid_fraction(valid_mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Compute the valid-pixel fraction for each patch in a mask batch."""
    if valid_mask.ndim != 3:
        raise ValueError("valid_mask must have shape (B, H, W)")
    batch, height, width = valid_mask.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("Mask height and width must be divisible by patch_size.")

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


def extract_scale_values(scale_features: torch.Tensor) -> torch.Tensor:
    """Extract the Stage A scalar scale input from patch ``scale_features``."""
    if scale_features.ndim != 2:
        raise ValueError("scale_features must have shape (B, F)")
    if scale_features.shape[1] < 1:
        raise ValueError("scale_features must include at least one column.")
    return scale_features[:, 0].to(torch.float32)


def _coerce_channel_stats(
    values: Sequence[float] | torch.Tensor | None,
    *,
    channels: int,
    default: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalize per-channel statistics to a ``(1, C, 1, 1)`` tensor."""
    if values is None:
        tensor = torch.full((channels,), float(default), dtype=dtype)
    else:
        tensor = torch.as_tensor(values, dtype=dtype).reshape(-1)
        if tensor.numel() != channels:
            raise ValueError(f"Expected {channels} channel stats, got {tensor.numel()}.")
    return tensor.view(1, channels, 1, 1)


def normalize_valid_image(
    image: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    channel_mean: Sequence[float] | torch.Tensor,
    channel_std: Sequence[float] | torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize valid pixels channel-wise while keeping invalid pixels at zero."""
    if image.ndim != 4:
        raise ValueError("image must have shape (B, C, H, W)")
    if valid_mask.ndim != 3:
        raise ValueError("valid_mask must have shape (B, H, W)")
    if image.shape[0] != valid_mask.shape[0] or image.shape[-2:] != valid_mask.shape[-2:]:
        raise ValueError("image and valid_mask must share batch/spatial dimensions.")

    mean = _coerce_channel_stats(channel_mean, channels=image.shape[1], default=0.0, dtype=image.dtype).to(
        image.device
    )
    std = _coerce_channel_stats(channel_std, channels=image.shape[1], default=1.0, dtype=image.dtype).to(
        image.device
    )
    std = std.clamp_min(float(eps))
    normalized = (image - mean) / std
    expanded_valid = valid_mask.unsqueeze(1)
    return torch.where(expanded_valid, normalized, torch.zeros_like(normalized))


def denormalize_patch_tokens(
    patches: torch.Tensor,
    *,
    patch_size: int,
    channels: int,
    channel_mean: Sequence[float] | torch.Tensor,
    channel_std: Sequence[float] | torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Map normalized patch tokens back into the original channel scale."""
    if patches.ndim != 3:
        raise ValueError("patches must have shape (B, N, D)")
    mean = _coerce_channel_stats(channel_mean, channels=channels, default=0.0, dtype=patches.dtype).to(
        patches.device
    )
    std = _coerce_channel_stats(channel_std, channels=channels, default=1.0, dtype=patches.dtype).to(
        patches.device
    )
    std = std.clamp_min(float(eps))
    mean_tokens = mean.expand(1, channels, patch_size, patch_size).reshape(1, 1, -1)
    std_tokens = std.expand(1, channels, patch_size, patch_size).reshape(1, 1, -1)
    return patches * std_tokens + mean_tokens


def collate_patch_samples_for_mae(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate Stage A1 patch samples into a MAE-ready batch dictionary."""
    if not samples:
        raise ValueError("samples must not be empty.")

    images = torch.stack([sample["image"] for sample in samples], dim=0)
    valid_masks = torch.stack([sample["valid_mask"] for sample in samples], dim=0)
    scale_features = torch.stack([sample["scale_features"] for sample in samples], dim=0)
    scale_values = extract_scale_values(scale_features)
    metadata = [dict(sample.get("metadata", {})) for sample in samples]

    return {
        "image": images,
        "valid_mask": valid_masks,
        "scale_features": scale_features,
        "scale_values": scale_values,
        "metadata": metadata,
        "rationale_raw": [sample.get("rationale_raw") for sample in samples],
        "rationale_expanded": [sample.get("rationale_expanded") for sample in samples],
    }


def build_masked_input_image(
    image: torch.Tensor,
    patch_valid_mask: torch.Tensor,
    visible_mask: torch.Tensor,
    *,
    patch_size: int,
) -> torch.Tensor:
    """Render the visible-token input image used by the MAE encoder."""
    if image.ndim != 4:
        raise ValueError("image must have shape (B, C, H, W)")
    if patch_valid_mask.shape != visible_mask.shape:
        raise ValueError("patch_valid_mask and visible_mask must have matching shape.")

    image_size = (image.shape[-2], image.shape[-1])
    visible_pixels = expand_patch_mask(visible_mask, patch_size, image_size=image_size)
    valid_pixels = expand_patch_mask(patch_valid_mask, patch_size, image_size=image_size)
    visible_pixels = visible_pixels.unsqueeze(1)
    valid_pixels = valid_pixels.unsqueeze(1)

    masked = image.clone()
    masked = torch.where(valid_pixels & ~visible_pixels, torch.zeros_like(masked), masked)
    masked = torch.where(~valid_pixels, torch.zeros_like(masked), masked)
    return masked


def build_reconstruction_composite(
    image: torch.Tensor,
    reconstruction: torch.Tensor,
    patch_valid_mask: torch.Tensor,
    visible_mask: torch.Tensor,
    *,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build full reconstructed images and visible+reconstructed composites."""
    if image.ndim != 4:
        raise ValueError("image must have shape (B, C, H, W)")

    reconstructed = unpatchify(
        reconstruction,
        patch_size,
        channels=image.shape[1],
        image_size=(image.shape[-2], image.shape[-1]),
    )
    image_size = (image.shape[-2], image.shape[-1])
    masked_valid_pixels = expand_patch_mask(
        patch_valid_mask & ~visible_mask,
        patch_size,
        image_size=image_size,
    ).unsqueeze(1)
    invalid_pixels = expand_patch_mask(
        ~patch_valid_mask,
        patch_size,
        image_size=image_size,
    ).unsqueeze(1)

    composite = image.clone()
    composite = torch.where(masked_valid_pixels, reconstructed, composite)
    composite = torch.where(invalid_pixels, torch.zeros_like(composite), composite)
    return reconstructed, composite


def sample_visible_patch_mask(
    patch_valid_mask: torch.Tensor,
    mask_ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Select visible patches from the valid patch set for MAE encoding."""
    if not (0.0 <= mask_ratio < 1.0):
        raise ValueError("mask_ratio must satisfy 0 <= value < 1.")

    visible_mask = torch.zeros_like(patch_valid_mask, dtype=torch.bool)
    for i in range(patch_valid_mask.shape[0]):
        valid_idx = torch.nonzero(patch_valid_mask[i], as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            continue
        n_visible = max(1, int(math.ceil(valid_idx.numel() * (1.0 - mask_ratio))))
        perm = torch.randperm(valid_idx.numel(), generator=generator)
        if perm.device != valid_idx.device:
            perm = perm.to(valid_idx.device)
        selected = valid_idx[perm[:n_visible]]
        visible_mask[i, selected] = True
    return visible_mask


def scale_sinusoidal_encoding(scale_values: torch.Tensor, dim: int) -> torch.Tensor:
    """Encode physical scale values using ScaleMAE-style sinusoidal features."""
    if dim % 2 != 0:
        raise ValueError("dim must be even for sinusoidal scale encoding.")
    scale_values = scale_values.to(torch.float32).reshape(-1, 1)
    half_dim = dim // 2
    frequencies = torch.exp(
        -torch.arange(half_dim, device=scale_values.device, dtype=torch.float32)
        * (math.log(10000.0) / max(half_dim - 1, 1))
    )
    scaled = scale_values * frequencies.unsqueeze(0)
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)


def _masked_mean(tokens: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    """Average token embeddings over positions marked true in ``keep_mask``."""
    weights = keep_mask.unsqueeze(-1).to(tokens.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denom


def _gather_visible_tokens(
    tokens: torch.Tensor,
    visible_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Pack visible tokens into a padded batch for encoder-only MAE processing."""
    batch, _, dim = tokens.shape
    visible_indices = [
        torch.nonzero(visible_mask[i], as_tuple=False).flatten() for i in range(batch)
    ]
    max_len = max((idx.numel() for idx in visible_indices), default=0)
    if max_len == 0:
        max_len = 1

    packed = tokens.new_zeros((batch, max_len, dim))
    padding_mask = torch.ones((batch, max_len), dtype=torch.bool, device=tokens.device)
    for i, idx in enumerate(visible_indices):
        if idx.numel() == 0:
            continue
        packed[i, : idx.numel()] = tokens[i, idx]
        padding_mask[i, : idx.numel()] = False
    return packed, padding_mask, visible_indices


@dataclass
class MarsMAEOutput:
    encoded_tokens: torch.Tensor
    pooled_embedding: torch.Tensor
    patch_valid_fraction: torch.Tensor
    patch_valid_mask: torch.Tensor
    visible_mask: torch.Tensor
    masked_valid_mask: torch.Tensor
    valid_pixel_mask: torch.Tensor
    loss_mask: torch.Tensor
    reconstruction: torch.Tensor
    loss: torch.Tensor


class MarsMaskedAutoencoder(nn.Module):
    """Workflow-aligned Stage A masked autoencoder for MarsCLIP."""

    def __init__(
        self,
        *,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        encoder_dim: int = 256,
        encoder_depth: int = 4,
        encoder_heads: int = 8,
        decoder_dim: int = 128,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        min_valid_fraction: float = DEFAULT_PATCH_VALID_FRACTION,
        normalize_inputs: bool = False,
        normalize_targets: bool = False,
        input_mean: Sequence[float] | torch.Tensor | None = None,
        input_std: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        if not (0.0 <= min_valid_fraction <= 1.0):
            raise ValueError("min_valid_fraction must satisfy 0 <= value <= 1.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (image_size // patch_size) ** 2
        self.min_valid_fraction = min_valid_fraction
        self.normalize_inputs = bool(normalize_inputs)
        self.normalize_targets = bool(normalize_targets)
        self.register_buffer(
            "input_mean",
            _coerce_channel_stats(input_mean, channels=in_channels, default=0.0),
            persistent=True,
        )
        self.register_buffer(
            "input_std",
            _coerce_channel_stats(input_std, channels=in_channels, default=1.0),
            persistent=True,
        )

        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=encoder_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, encoder_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=encoder_heads,
            batch_first=True,
            dim_feedforward=encoder_dim * 4,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)

        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            batch_first=True,
            dim_feedforward=decoder_dim * 4,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.reconstruction_head = nn.Linear(
            decoder_dim,
            in_channels * patch_size * patch_size,
        )

    def forward_patch_batch(
        self,
        batch: dict[str, Any],
        *,
        mask_ratio: float = 0.75,
        generator: torch.Generator | None = None,
    ) -> MarsMAEOutput:
        """Run Stage A directly on a collated MarsCLIP patch batch."""
        scale_values = batch.get("scale_values")
        if scale_values is None:
            scale_values = extract_scale_values(batch["scale_features"])
        return self.forward(
            batch["image"],
            batch["valid_mask"],
            scale_values=scale_values,
            mask_ratio=mask_ratio,
            generator=generator,
        )

    def forward(
        self,
        image: torch.Tensor,
        valid_mask: torch.Tensor,
        scale_values: torch.Tensor,
        *,
        mask_ratio: float = 0.75,
        generator: torch.Generator | None = None,
    ) -> MarsMAEOutput:
        if image.ndim != 4:
            raise ValueError("image must have shape (B, C, H, W)")
        if valid_mask.ndim != 3:
            raise ValueError("valid_mask must have shape (B, H, W)")
        if image.shape[2] != self.image_size or image.shape[3] != self.image_size:
            raise ValueError("Input image size does not match model image_size.")

        normalized_image = None
        if self.normalize_inputs or self.normalize_targets:
            normalized_image = normalize_valid_image(
                image,
                valid_mask,
                channel_mean=self.input_mean.flatten(),
                channel_std=self.input_std.flatten(),
            )

        encoder_image = normalized_image if self.normalize_inputs and normalized_image is not None else image
        tokens = self.patch_embed(encoder_image).flatten(2).transpose(1, 2)
        encoder_scale = scale_sinusoidal_encoding(scale_values, tokens.shape[-1]).unsqueeze(1)
        tokens = tokens + self.encoder_pos_embed + encoder_scale

        patch_valid_fraction = compute_patch_valid_fraction(valid_mask, self.patch_size)
        patch_valid_mask = compute_valid_patch_mask(
            valid_mask,
            self.patch_size,
            min_valid_fraction=self.min_valid_fraction,
        )
        visible_mask = sample_visible_patch_mask(
            patch_valid_mask,
            mask_ratio=mask_ratio,
            generator=generator,
        )
        masked_valid_mask = patch_valid_mask & ~visible_mask

        visible_tokens, encoder_padding_mask, visible_indices = _gather_visible_tokens(
            tokens,
            visible_mask,
        )
        encoded_visible = self.encoder(
            visible_tokens,
            src_key_padding_mask=encoder_padding_mask,
        )

        batch, num_patches, encoder_dim = tokens.shape
        encoded_tokens = tokens.new_zeros((batch, num_patches, encoder_dim))
        for i, idx in enumerate(visible_indices):
            if idx.numel() == 0:
                continue
            encoded_tokens[i, idx] = encoded_visible[i, : idx.numel()]
        pooled_embedding = _masked_mean(encoded_tokens, visible_mask)

        decoder_tokens = self.encoder_to_decoder(encoded_tokens)
        decoder_scale = scale_sinusoidal_encoding(
            scale_values,
            decoder_tokens.shape[-1],
        ).unsqueeze(1)
        decoder_tokens = decoder_tokens + self.decoder_pos_embed + decoder_scale

        if masked_valid_mask.any():
            replacement = self.mask_token.expand(masked_valid_mask.sum().item(), -1, -1)
            decoder_tokens[masked_valid_mask] = replacement.squeeze(1)

        decoded = self.decoder(
            decoder_tokens,
            src_key_padding_mask=~patch_valid_mask,
        )
        reconstruction = self.reconstruction_head(decoded)

        target_image = normalized_image if self.normalize_targets and normalized_image is not None else image
        target_patches = patchify(target_image, self.patch_size)
        expanded_valid_mask = valid_mask.unsqueeze(1).expand(-1, image.shape[1], -1, -1)
        valid_pixel_mask = (
            patchify(expanded_valid_mask.to(image.dtype), self.patch_size) > 0.5
        )
        loss_mask = masked_valid_mask.unsqueeze(-1) & valid_pixel_mask

        if loss_mask.any():
            squared_error = (reconstruction - target_patches).pow(2)
            loss = squared_error[loss_mask].mean()
        else:
            loss = reconstruction.sum() * 0.0

        output_reconstruction = reconstruction
        if self.normalize_targets:
            output_reconstruction = denormalize_patch_tokens(
                reconstruction,
                patch_size=self.patch_size,
                channels=self.in_channels,
                channel_mean=self.input_mean.flatten(),
                channel_std=self.input_std.flatten(),
            )

        return MarsMAEOutput(
            encoded_tokens=encoded_tokens,
            pooled_embedding=pooled_embedding,
            patch_valid_fraction=patch_valid_fraction,
            patch_valid_mask=patch_valid_mask,
            visible_mask=visible_mask,
            masked_valid_mask=masked_valid_mask,
            valid_pixel_mask=valid_pixel_mask,
            loss_mask=loss_mask,
            reconstruction=output_reconstruction,
            loss=loss,
        )
