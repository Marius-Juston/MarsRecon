"""Binary-mask morphology used during void filling and artifact detection."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def erode_valid_mask(valid_mask: torch.Tensor, erode_radius: int = 1) -> torch.Tensor:
    """Erode a binary mask to trim noisy boundary pixels."""
    if erode_radius <= 0:
        return valid_mask

    is_3d = valid_mask.ndim == 3
    if is_3d:
        valid_mask = valid_mask.unsqueeze(0)

    kernel_size = 2 * erode_radius + 1
    padded_mask = F.pad(
        valid_mask,
        pad=(erode_radius, erode_radius, erode_radius, erode_radius),
        mode="constant",
        value=1.0,
    )
    eroded_mask = -F.max_pool2d(
        -padded_mask, kernel_size=kernel_size, stride=1, padding=0
    )
    eroded_mask = (eroded_mask > 0.5).float()

    if is_3d:
        return eroded_mask.squeeze(0)
    return eroded_mask
