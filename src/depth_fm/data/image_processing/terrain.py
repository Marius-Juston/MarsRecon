"""Terrain-level scalar metrics derived from elevation patches."""

from __future__ import annotations

import torch


def compute_topographic_residual(elevation: torch.Tensor, valid_mask: torch.Tensor) -> float:
    """Fit a 2D plane to elevation and return the RMS residual.

    Removes macroscopic slopes so the returned scalar isolates true topographic
    roughness (used as a manifest filter to reject overly-flat patches).
    """
    if elevation.ndim > 2:
        elevation = elevation.squeeze()
        valid_mask = valid_mask.squeeze()

    H, W = elevation.shape

    y = torch.linspace(-1, 1, H, dtype=elevation.dtype, device=elevation.device)
    x = torch.linspace(-1, 1, W, dtype=elevation.dtype, device=elevation.device)
    Y, X = torch.meshgrid(y, x, indexing="ij")

    valid_bool = valid_mask.bool()
    if not valid_bool.any():
        return 0.0

    X_v = X[valid_bool].unsqueeze(1)
    Y_v = Y[valid_bool].unsqueeze(1)
    Z_v = elevation[valid_bool].unsqueeze(1)

    A = torch.cat([X_v, Y_v, torch.ones_like(X_v)], dim=1)
    w = torch.linalg.lstsq(A, Z_v).solution

    Z_pred = A @ w
    return torch.sqrt(torch.mean((Z_v - Z_pred) ** 2)).item()
