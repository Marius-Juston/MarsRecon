"""Sun-vector estimation from elevation + orthoimage pairs.

* `estimate_sun_vector_ols`  — Ordinary least squares (deprecated: can produce
                                negative sz).
* `estimate_sun_vector_irls` — Robust IRLS with decoupled ambient estimate and
                                enforced positive sz. Use this.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing_extensions import deprecated


# FIXME: produces negative sun_z under some geometries.
@deprecated(
    "Use estimate_sun_vector_irls instead, this does not calculate the z sun vector correctly and can render it to have negative values."
)
def estimate_sun_vector_ols(
        dtm: torch.Tensor,
        ortho: torch.Tensor,
        valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimates the sun vector [sx, sy, sz] using Ordinary Least Squares.

    Handles shapes (C, H, W) or (1, C, H, W). Batch size must be 1.
    """
    device = dtm.device

    def get_defaults():
        default_sun = F.normalize(torch.tensor([0.5, -0.5, 1.0], device=device), p=2, dim=0)
        return default_sun, torch.tensor(1.0, device=device), torch.tensor(0.3, device=device)

    if dtm.ndim == 4:
        assert dtm.shape[0] == 1, (
            "The batch size should be 1. estimate_sun_vector_irls currently only works for a batch size of 1"
        )
        dtm = dtm[0]
    if ortho.ndim == 4:
        ortho = ortho[0]
    if valid_mask.ndim == 4:
        valid_mask = valid_mask[0]

    if ortho.shape[0] == 3:
        ortho = ortho.mean(dim=0, keepdim=True)

    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device) / 8.0
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device) / 8.0

    spatial_scale = max(dtm.shape[-2], dtm.shape[-1]) / 2.0
    padded_dtm = F.pad(dtm.unsqueeze(0), (1, 1, 1, 1), mode="replicate")

    n_x = -F.conv2d(padded_dtm, sobel_x.view(1, 1, 3, 3)) * spatial_scale
    n_y = -F.conv2d(padded_dtm, sobel_y.view(1, 1, 3, 3)) * spatial_scale
    n_z = torch.ones_like(n_x)

    normals = torch.cat([n_x, n_y, n_z], dim=1)
    normals = F.normalize(normals, p=2, dim=1).squeeze(0)

    mask = valid_mask.squeeze(0).bool()

    ortho_valid = ortho.squeeze(0)[mask]
    if len(ortho_valid) == 0:
        return get_defaults()

    intensity_threshold = torch.quantile(ortho_valid, 0.05)
    shadow_mask = ortho.squeeze(0) > intensity_threshold
    final_mask = mask & shadow_mask

    N_flat = normals[:, final_mask].t()
    Y_flat = ortho.squeeze(0)[final_mask].unsqueeze(1)

    if N_flat.shape[0] < 100:
        return get_defaults()

    ones = torch.ones((N_flat.shape[0], 1), device=device)
    A = torch.cat([N_flat, ones], dim=1)

    x = torch.linalg.lstsq(A, Y_flat).solution

    k = x[:3, 0]
    ambient = x[3, 0]
    intensity = torch.norm(k, p=2)
    sun_vec = F.normalize(k, p=2, dim=0)

    return sun_vec, intensity, ambient


@torch.no_grad()
def estimate_sun_vector_irls(
        dtm: torch.Tensor,
        ortho: torch.Tensor,
        valid_mask: torch.Tensor,
        max_iter: int = 15,
        tol: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decoupled IRLS sun-vector estimator.

    Separates ambient light estimation from the linear system to prevent the
    Nz / bias collinearity trap from inverting the sun vector. Returns
    (sun_vec, intensity, ambient).
    """
    device = dtm.device

    def get_defaults():
        default_sun = F.normalize(torch.tensor([0.5, -0.5, 1.0], device=device), p=2, dim=0)
        return default_sun, torch.tensor(1.0, device=device), torch.tensor(0.05, device=device)

    if dtm.ndim == 4:
        dtm = dtm[0]
    if ortho.ndim == 4:
        ortho = ortho[0]
    if valid_mask.ndim == 4:
        valid_mask = valid_mask[0]
    if ortho.shape[0] == 3:
        ortho = ortho.mean(dim=0, keepdim=True)

    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device) / 8.0
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device) / 8.0
    spatial_scale = max(dtm.shape[-2], dtm.shape[-1]) / 2.0
    padded_dtm = F.pad(dtm.unsqueeze(0), (1, 1, 1, 1), mode="replicate")

    n_x = -F.conv2d(padded_dtm, sobel_x.view(1, 1, 3, 3)) * spatial_scale
    n_y = -F.conv2d(padded_dtm, sobel_y.view(1, 1, 3, 3)) * spatial_scale
    n_z = torch.ones_like(n_x)

    normals = F.normalize(torch.cat([n_x, n_y, n_z], dim=1), p=2, dim=1).squeeze(0)

    mask = valid_mask.squeeze(0).bool()
    zero_mask = ortho.squeeze(0) > 1e-4
    final_mask = mask & zero_mask

    N_flat = normals[:, final_mask].t()
    Y_raw = ortho.squeeze(0)[final_mask].unsqueeze(1)

    if N_flat.shape[0] < 100:
        return get_defaults()

    # Decoupled ambient estimation: 1st percentile is a robust proxy for the
    # secondary scattering / ambient floor in deep shadows.
    ambient_est = torch.quantile(Y_raw, 0.01)
    Y_flat = torch.clamp(Y_raw - ambient_est, min=0.0)

    H = N_flat
    beta = torch.linalg.lstsq(H, Y_flat).solution
    c = 1.345

    for _ in range(max_iter):
        residuals = Y_flat - torch.mm(H, beta)
        median_res = torch.median(residuals)
        mad = torch.median(torch.abs(residuals - median_res))
        sigma = (mad / 0.67449) + 1e-6

        r_stand = torch.abs(residuals / sigma)
        weights = torch.clamp(c / (r_stand + 1e-8), max=1.0)

        w_sqrt = torch.sqrt(weights)
        H_w = H * w_sqrt
        Y_w = Y_flat * w_sqrt

        beta_new = torch.linalg.lstsq(H_w, Y_w).solution

        change = torch.norm(beta_new - beta, p=2)
        # Every successful solve becomes the next IRLS iterate, including
        # the final update when the iteration budget is exhausted.
        beta = beta_new
        if change < tol:
            break

    k = beta[:3, 0]
    # Failsafe: with the bias stripped, if anomalous geometry still pushes Nz
    # negative we reflect across the horizon line to maintain physical validity.
    if k[2] < 0:
        k[2] = -k[2]

    intensity = torch.norm(k, p=2).clamp(min=1e-4)
    sun_vec = F.normalize(k, p=2, dim=0)

    return sun_vec, intensity, ambient_est
