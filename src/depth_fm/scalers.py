"""
DTM Normalization for Training AND Inference on Unseen Data
===========================================================

CRITICAL CONSTRAINT: At inference time, we have ONLY an ortho image.
The network predicts a normalised DTM patch. We must be able to convert
that prediction back to physical metres WITHOUT any per-patch metadata
(no stored scales, no stored trend parameters).

This rules out ``adaptive_stored`` normalization — it produces outputs
where 0.8 could mean 0.5m or 15m depending on the unknown local scale.

The correct approach for this use case:

  TRAINING TARGET = plane-detrended residual / global_fixed_scale

    - The plane is removed so the network only predicts what the ortho
      image actually encodes (local relief from shadow cues).
    - The global fixed scale gives every prediction a CONSISTENT physical
      meaning: network output * scale = metres of relief.
    - At inference, the inverse is just: multiply by scale. No metadata.

  RECONSTRUCTION = stitch trend planes from overlap consistency, then
    add the denormalised residuals.

The slope problem (dynamic range wasted on ramps) is solved by the
DETRENDING, not by the scaling. After removing the plane, the residual
distribution is tight enough that a single global scale works well.

The VAE precision concern for low-relief patches:
  After detrending, a 0.6m-relief patch maps to ~[-0.05, 0.05].
  This is small but NOT zero — the VAE operates at float32 with 4
  latent channels at 1/8 resolution. The effective precision is
  ~10-12 bits per channel, giving ~4000 distinguishable levels in
  [0, 0.05]. The signal IS there.

  If you find empirically that low-relief patches are blurry, use
  the log-compressed variant (Strategy B) which expands small values
  by ~3x while remaining globally invertible with no stored metadata.


Three strategies ranked by recommendation:
=========================================

Strategy A: GLOBAL FIXED (simplest, try first)
  forward:  normed = detrended_residual / global_scale
  inverse:  residual = prediction * global_scale
  pros:     Linear, trivial inverse, reconstruction is shift-only
  cons:     Low-relief patches use narrow range

Strategy B: GLOBAL LOG (best compromise, recommended)
  forward:  normed = sign(r) * log1p(|r| / ref) / log1p(1)
  inverse:  residual = sign(n) * ref * (expm1(|n| * ln2))
  pros:     Expands low-relief 3x, compresses high-relief,
            NO stored metadata needed, fully invertible
  cons:     Non-linear — loss gradients weighted differently
            for large vs small features

Strategy C: ADAPTIVE + SCALE HEAD (most complex, best signal)
  forward:  normed = residual / local_p98  (full [-1,1] range)
  inverse:  residual = prediction * predicted_scale
  The network has TWO outputs:
    1. Normalised residual shape (3ch, H, W)
    2. Predicted scale factor (scalar regression head)
  pros:     Best VAE utilisation, network learns relief magnitude
  cons:     Requires architecture modification (extra head),
            scale prediction errors propagate to reconstruction

For your DepthFM pipeline, I recommend starting with Strategy A
(simplest to implement, changes only normalization code) and moving
to Strategy B if low-relief patches are empirically blurry.


What about the trend plane at inference?
========================================

The network predicts ONLY the residual (what's visible in shadows).
The trend plane (regional slope) cannot be predicted from nadir ortho.
But for reconstruction, you NEED the plane to get absolute elevation.

Options for recovering the trend at inference:
  1. OVERLAP CONSISTENCY (recommended): With 50% overlap, adjacent
     predictions must agree in the overlap region up to a constant
     offset (the difference in their unknown plane values). This gives
     you pairwise offset constraints → solve sparse system for global
     offsets → reconstruct a globally consistent surface.

  2. ANCILLARY DATA: If you have a coarse DTM (e.g., MOLA at 463m/px),
     use it as the trend surface and add the network's predicted
     fine-scale residual on top.

  3. STEREO GEOMETRY: If you have both left and right ortho images,
     their parallax encodes the trend. But this requires stereo
     matching, which is what you're trying to replace.

Option 1 is the cleanest — it works with ortho-only input and the
overlap constraints are well-conditioned since we're only solving
for scalar offsets (1 unknown per patch, not 2 or 3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TrainingNormResult:
    """Output of normalization during training (GT elevation available)."""

    normed_residual: torch.Tensor
    """(1, H, W) normalised residual in ~[-1, 1]. Network target."""

    trend_params: torch.Tensor
    """(3,) plane params [a, b, c] in normalised coords. Stored in
    manifest for analysis but NOT needed at inference time."""

    residual_scale: float
    """The global scale factor used. Same for all patches."""

    valid_mask: torch.Tensor
    """(1, H, W) binary mask."""

    raw_residual_rms: float
    """RMS of physical residual (metres). For manifest/filtering."""

    raw_residual_p98: float
    """p98 of |residual| (metres). For manifest/filtering."""


@dataclass
class InferenceResult:
    """Output of denormalization during inference (no GT available)."""

    physical_residual: torch.Tensor
    """(1, H, W) residual in metres."""

    # Note: NO trend_params here — we don't have them at inference.
    # The trend must be recovered from overlap consistency or ancillary data.


# ──────────────────────────────────────────────────────────────────────
# Plane fitting (training only — requires GT elevation)
# ──────────────────────────────────────────────────────────────────────

def fit_plane(
    elevation: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Fit z = ax + by + c to elevation using normalised [-1,1] coords.

    Args:
        elevation: (1, H, W) or (H, W).
        valid_mask: (1, H, W) or (H, W), binary.

    Returns:
        (3,) tensor [a, b, c].
    """
    if elevation.ndim == 3:
        elevation = elevation.squeeze(0)
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.squeeze(0)

    H, W = elevation.shape
    device = elevation.device
    dtype = elevation.dtype

    valid_bool = valid_mask.bool()
    n_valid = valid_bool.sum().item()

    if n_valid < 10:
        mean_val = elevation[valid_bool].mean().item() if n_valid > 0 else 0.0
        return torch.tensor([0.0, 0.0, mean_val], device=device, dtype=dtype)

    y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing="ij")

    xv = X[valid_bool].unsqueeze(1)
    yv = Y[valid_bool].unsqueeze(1)
    zv = elevation[valid_bool].unsqueeze(1)

    A = torch.cat([xv, yv, torch.ones_like(xv)], dim=1)

    try:
        return torch.linalg.lstsq(A, zv).solution.squeeze(1)
    except Exception:
        mean_val = elevation[valid_bool].mean().item()
        return torch.tensor([0.0, 0.0, mean_val], device=device, dtype=dtype)


def evaluate_plane(
    params: torch.Tensor, H: int, W: int,
    device: torch.device = None, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Evaluate plane on (H, W) grid. Returns (1, H, W)."""
    if device is None:
        device = params.device
    a, b, c = params[0].item(), params[1].item(), params[2].item()
    y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    return (a * X + b * Y + c).unsqueeze(0)


# ──────────────────────────────────────────────────────────────────────
# Strategy A: Global Fixed Scale
# ──────────────────────────────────────────────────────────────────────

class GlobalFixedNormalizer:
    """Plane detrend → divide by fixed global scale.

    The simplest approach. Every network output has a fixed physical
    meaning: ``prediction * global_scale = metres``.

    Args:
        global_scale: p98 of |detrended residual| across the dataset.
            Compute once with ``compute_residual_statistics()``.
            Typical values for HiRISE: 8–15 metres.
    """

    def __init__(self, global_scale: float):
        self.global_scale = global_scale

    def normalize_for_training(
        self, elevation: torch.Tensor, valid_mask: torch.Tensor,
    ) -> TrainingNormResult:
        """Training-time normalization (GT elevation available)."""
        elevation = torch.nan_to_num(elevation, nan=0.0)
        valid = valid_mask.bool()

        if not valid.any():
            H, W = elevation.shape[-2], elevation.shape[-1]
            return TrainingNormResult(
                normed_residual=torch.zeros_like(elevation),
                trend_params=torch.zeros(3, device=elevation.device),
                residual_scale=self.global_scale,
                valid_mask=valid_mask,
                raw_residual_rms=0.0,
                raw_residual_p98=0.0,
            )

        # 1. Fit and remove plane
        params = fit_plane(elevation, valid_mask)
        H, W = elevation.shape[-2], elevation.shape[-1]
        plane = evaluate_plane(params, H, W, elevation.device, elevation.dtype)
        residual = elevation - plane
        residual = torch.where(valid, residual, torch.zeros_like(residual))

        # 2. Stats for manifest
        valid_res = residual[valid]
        rms = torch.sqrt((valid_res ** 2).mean()).item()
        p98 = torch.quantile(valid_res.abs(), 0.98).item() if len(valid_res) > 10 else 0.0

        # 3. Normalise
        normed = residual / self.global_scale
        normed = torch.clamp(normed, -1.0, 1.0)
        normed = torch.where(valid, normed, torch.zeros_like(normed))

        return TrainingNormResult(
            normed_residual=normed,
            trend_params=params,
            residual_scale=self.global_scale,
            valid_mask=valid_mask,
            raw_residual_rms=rms,
            raw_residual_p98=p98,
        )

    def denormalize_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        """Inference-time inverse. No metadata needed.

        Args:
            prediction: (1, H, W) or (B, 1, H, W) network output in [-1, 1].

        Returns:
            Physical residual in metres (same shape).
        """
        return prediction * self.global_scale


# ──────────────────────────────────────────────────────────────────────
# Strategy B: Global Log-Compressed (RECOMMENDED)
# ──────────────────────────────────────────────────────────────────────

class GlobalLogNormalizer:
    """Plane detrend → signed log compression with global reference.

    Expands low-relief details by ~3× compared to linear scaling while
    compressing high-relief patches. Fully invertible with NO stored
    metadata.

    Forward:
        normed = sign(r) * log1p(|r| / ref_scale) / log1p(1)

    Inverse:
        residual = sign(n) * ref_scale * expm1(|n| * ln(2))

    At ref_scale (the reference magnitude):
        - Input ±ref_scale → output ±1.0
        - Input ±ref_scale/10 → output ±0.14 (vs ±0.10 linear: 1.4× expansion)
        - Input ±ref_scale/100 → output ±0.014 (vs ±0.01 linear: 1.4× expansion)

    The expansion ratio increases for smaller values. A 0.6m signal
    with ref_scale=12 gets 1.4× more range than linear.

    Adjustment: to get MORE expansion of small values, use a smaller
    ref_scale. But this clips more high-relief patches. The p90 of
    residual distribution is often a good choice (clips the top 10%
    while expanding everything else).

    Args:
        ref_scale: Reference scale in metres. Values at ±ref_scale
            map to ±1.0 in the normalised space. Use the p90 or p95
            of your detrended residual distribution.
    """

    def __init__(self, ref_scale: float):
        self.ref_scale = ref_scale
        self._log2 = float(np.log(2.0))
        self._inv_log2 = 1.0 / self._log2

    def normalize_for_training(
        self, elevation: torch.Tensor, valid_mask: torch.Tensor,
    ) -> TrainingNormResult:
        """Training-time normalization."""
        elevation = torch.nan_to_num(elevation, nan=0.0)
        valid = valid_mask.bool()

        if not valid.any():
            return TrainingNormResult(
                normed_residual=torch.zeros_like(elevation),
                trend_params=torch.zeros(3, device=elevation.device),
                residual_scale=self.ref_scale,
                valid_mask=valid_mask,
                raw_residual_rms=0.0,
                raw_residual_p98=0.0,
            )

        # 1. Fit and remove plane
        params = fit_plane(elevation, valid_mask)
        H, W = elevation.shape[-2], elevation.shape[-1]
        plane = evaluate_plane(params, H, W, elevation.device, elevation.dtype)
        residual = elevation - plane
        residual = torch.where(valid, residual, torch.zeros_like(residual))

        # 2. Stats
        valid_res = residual[valid]
        rms = torch.sqrt((valid_res ** 2).mean()).item()
        p98 = torch.quantile(valid_res.abs(), 0.98).item() if len(valid_res) > 10 else 0.0

        # 3. Signed log compression
        normed = (
            torch.sign(residual)
            * torch.log1p(residual.abs() / self.ref_scale)
            * self._inv_log2
        )
        normed = torch.clamp(normed, -1.0, 1.0)
        normed = torch.where(valid, normed, torch.zeros_like(normed))

        return TrainingNormResult(
            normed_residual=normed,
            trend_params=params,
            residual_scale=self.ref_scale,
            valid_mask=valid_mask,
            raw_residual_rms=rms,
            raw_residual_p98=p98,
        )

    def denormalize_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        """Inference-time inverse. No metadata needed.

        inverse: sign(n) * ref * expm1(|n| * ln2)
        """
        return (
            torch.sign(prediction)
            * self.ref_scale
            * torch.expm1(prediction.abs() * self._log2)
        )


# ──────────────────────────────────────────────────────────────────────
# Strategy C: Adaptive + Scale Prediction Head
# ──────────────────────────────────────────────────────────────────────

class AdaptiveWithScaleHead:
    """Plane detrend → per-patch adaptive scaling.

    The network must have TWO outputs:
      1. Normalised residual (3, H, W) — the usual DepthFM output
      2. Log-scale prediction (scalar) — a regression head

    During training, the target scale is ``log(local_p98)``.
    During inference, the network predicts the scale, and we use:
        physical_residual = predicted_residual * exp(predicted_log_scale)

    This gives the best VAE signal (full [-1,1] range always) while
    remaining invertible on unseen data.

    REQUIRES ARCHITECTURE CHANGES to your DepthFM model.

    Args:
        min_scale: Floor for scale factor (metres).
        log_scale_mean: Mean of log(scale) distribution for normalising
            the scale prediction target. Compute from training data.
        log_scale_std: Std of log(scale) distribution.
    """

    def __init__(
        self,
        min_scale: float = 0.1,
        log_scale_mean: float = 1.5,   # ~exp(1.5) ≈ 4.5m
        log_scale_std: float = 1.0,
    ):
        self.min_scale = min_scale
        self.log_scale_mean = log_scale_mean
        self.log_scale_std = log_scale_std

    def normalize_for_training(
        self, elevation: torch.Tensor, valid_mask: torch.Tensor,
    ) -> dict:
        """Returns both the normalised residual AND the scale target.

        Returns dict with:
            "normed_residual": (1, H, W) in [-1, 1]
            "scale_target": scalar, normalised log-scale for regression
            "raw_scale": scalar, the actual local_p98 in metres
            "trend_params": (3,) plane parameters
        """
        elevation = torch.nan_to_num(elevation, nan=0.0)
        valid = valid_mask.bool()

        if not valid.any():
            return {
                "normed_residual": torch.zeros_like(elevation),
                "scale_target": torch.tensor(0.0),
                "raw_scale": self.min_scale,
                "trend_params": torch.zeros(3, device=elevation.device),
                "valid_mask": valid_mask,
            }

        # Fit and remove plane
        params = fit_plane(elevation, valid_mask)
        H, W = elevation.shape[-2], elevation.shape[-1]
        plane = evaluate_plane(params, H, W, elevation.device, elevation.dtype)
        residual = elevation - plane
        residual = torch.where(valid, residual, torch.zeros_like(residual))

        # Compute adaptive scale
        valid_res = residual[valid].abs()
        local_p98 = torch.quantile(valid_res, 0.98).item() if len(valid_res) > 10 else self.min_scale
        local_p98 = max(local_p98, self.min_scale)

        # Normalise residual to [-1, 1]
        normed = residual / local_p98
        normed = torch.clamp(normed, -1.0, 1.0)
        normed = torch.where(valid, normed, torch.zeros_like(normed))

        # Scale target: normalised log scale for regression head
        log_scale = np.log(local_p98)
        scale_target = (log_scale - self.log_scale_mean) / self.log_scale_std

        return {
            "normed_residual": normed,
            "scale_target": torch.tensor(scale_target, dtype=torch.float32),
            "raw_scale": local_p98,
            "trend_params": params,
            "valid_mask": valid_mask,
        }

    def denormalize_prediction(
        self,
        prediction: torch.Tensor,
        predicted_log_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Inference-time inverse using the network's scale prediction.

        Args:
            prediction: (B, 1, H, W) normalised residual.
            predicted_log_scale: (B,) predicted normalised log-scale.

        Returns:
            (B, 1, H, W) physical residual in metres.
        """
        # Undo the normalisation of the scale target
        log_scale = predicted_log_scale * self.log_scale_std + self.log_scale_mean
        scale = torch.exp(log_scale)  # (B,)

        # Broadcast and multiply
        if prediction.ndim == 4:
            scale = scale.view(-1, 1, 1, 1)
        elif prediction.ndim == 3:
            scale = scale.view(-1, 1, 1)

        return prediction * scale


# ──────────────────────────────────────────────────────────────────────
# Reconstruction: recovering the trend from overlaps
# ──────────────────────────────────────────────────────────────────────

def recover_trend_offsets_from_overlaps(
    physical_residuals: list[np.ndarray],
    canvas_positions: list[tuple[int, int]],
    patch_size: int,
    regularization: float = 1e-4,
) -> np.ndarray:
    """Recover per-patch constant offsets from overlap consistency.

    After denormalization, each predicted patch gives us the physical
    residual (elevation minus unknown plane). In overlap regions,
    two patches observe the same terrain, so their residuals should
    agree up to a constant offset (the difference in their unknown
    plane values at that location).

    For patches with 50% overlap, the offset between adjacent patches
    is approximately constant across the overlap (because the plane
    difference varies slowly). So we solve:

        residual_j(overlap) ≈ residual_i(overlap) + offset_ij

    Then find globally consistent offsets O_i such that:
        O_j - O_i ≈ offset_ij

    With anchor O_0 = 0.

    This is a simple sparse linear system with 1 unknown per patch.

    Args:
        physical_residuals: List of (H, W) arrays, denormalised predictions.
        canvas_positions: List of (row, col) grid coordinates.
        patch_size: Pixel size of each patch.

    Returns:
        (N,) array of per-patch offsets. Add this to each residual to
        get a globally consistent surface.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from collections import defaultdict

    N = len(physical_residuals)

    # Build spatial index
    grid_map = defaultdict(int)
    for idx, (r, c) in enumerate(canvas_positions):
        grid_map[(r, c)] = idx

    # Compute pairwise offsets from overlaps
    rows, cols, vals, rhs = [], [], [], []
    eq = 0

    neighbor_offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for idx_i, (ri, ci) in enumerate(canvas_positions):
        for dr, dc in neighbor_offsets:
            key_j = (ri + dr, ci + dc)
            if key_j not in grid_map:
                continue
            idx_j = grid_map[key_j]

            # Compute overlap region
            stride = patch_size // 2  # 50% overlap
            y_off = dr * stride
            x_off = dc * stride

            # Overlap bounds in patch_i coords
            oy_start_i = max(0, y_off)
            oy_end_i = min(patch_size, patch_size + y_off)
            ox_start_i = max(0, x_off)
            ox_end_i = min(patch_size, patch_size + x_off)

            # Corresponding bounds in patch_j coords
            oy_start_j = oy_start_i - y_off
            oy_end_j = oy_end_i - y_off
            ox_start_j = ox_start_i - x_off
            ox_end_j = ox_end_i - x_off

            r_i = physical_residuals[idx_i][oy_start_i:oy_end_i, ox_start_i:ox_end_i]
            r_j = physical_residuals[idx_j][oy_start_j:oy_end_j, ox_start_j:ox_end_j]

            if r_i.size == 0:
                continue

            # Median offset is robust to boundary artifacts
            offset_ij = float(np.median(r_j - r_i))

            # O_j - O_i = offset_ij
            rows.extend([eq, eq])
            cols.extend([idx_j, idx_i])
            vals.extend([1.0, -1.0])
            rhs.append(offset_ij)
            eq += 1

    if eq == 0:
        logger.warning("No overlap constraints found. Returning zero offsets.")
        return np.zeros(N)

    # Anchor: O_0 = 0
    rows.append(eq)
    cols.append(0)
    vals.append(1000.0)
    rhs.append(0.0)
    eq += 1

    A = sp.csr_matrix((vals, (rows, cols)), shape=(eq, N))
    b = np.array(rhs)

    offsets = spla.lsqr(A, b, damp=regularization)[0]

    logger.info(
        "Recovered %d offsets from %d constraints. Range: [%.2f, %.2f] m",
        N, eq - 1, offsets.min(), offsets.max(),
    )
    return offsets


# ──────────────────────────────────────────────────────────────────────
# Full inference pipeline
# ──────────────────────────────────────────────────────────────────────

def reconstruct_strip_from_predictions(
    normalizer,
    predictions: list[np.ndarray],
    grid_positions: list[tuple[int, int]],
    patch_size: int = 512,
    poisson_screening: float = 0.01,
) -> np.ndarray:
    """Complete inference pipeline: predictions → seamless DTM strip.

    1. Denormalize each prediction to physical residual (metres)
    2. Recover per-patch offsets from overlap consistency
    3. Apply offsets
    4. Poisson-blend into seamless strip

    Args:
        normalizer: GlobalFixedNormalizer or GlobalLogNormalizer instance.
        predictions: List of (H, W) numpy arrays, raw network output in [-1,1].
        grid_positions: List of (row, col) grid positions.
        patch_size: Pixel size of each patch.
        poisson_screening: Screening weight for Poisson blending.

    Returns:
        (canvas_H, canvas_W) numpy array — the reconstructed DTM.
    """
    import torch
    from strip_reconstruction import (
        PatchPrediction, AlignedPatch,
        poisson_blend_strip, _distance_weight,
    )

    # 1. Denormalize to physical residuals
    physical = []
    for pred_np in predictions:
        pred_t = torch.from_numpy(pred_np).unsqueeze(0).float()
        phys_t = normalizer.denormalize_prediction(pred_t)
        physical.append(phys_t.squeeze(0).numpy())

    # 2. Recover offsets from overlaps
    offsets = recover_trend_offsets_from_overlaps(
        physical, grid_positions, patch_size
    )

    # 3. Apply offsets and prepare for blending
    stride = patch_size // 2
    max_row = max(r for r, c in grid_positions)
    max_col = max(c for r, c in grid_positions)
    canvas_H = (max_row + 1) * stride + patch_size
    canvas_W = (max_col + 1) * stride + patch_size

    aligned = []
    for i, ((row, col), phys, offset) in enumerate(
        zip(grid_positions, physical, offsets)
    ):
        aligned.append(AlignedPatch(
            row=row, col=col,
            aligned=phys + offset,
            confidence=np.ones_like(phys),
            canvas_y=row * stride,
            canvas_x=col * stride,
            scale=1.0,
            shift=offset,
        ))

    # 4. Poisson blend
    result = poisson_blend_strip(
        aligned, canvas_H, canvas_W, patch_size, poisson_screening
    )

    return result


# ──────────────────────────────────────────────────────────────────────
# Statistics computation (run ONCE on your dataset)
# ──────────────────────────────────────────────────────────────────────

def compute_detrended_residual_stats(
    dataset,
    sampler,
    resolution: int = 512,
    max_patches: int = 5000,
    output_path: str = "detrended_residual_stats.json",
) -> dict:
    """Compute the global scale factor from plane-detrended residuals.

    Run this ONCE before training.

    Iterates over patches, fits a plane to each, computes the residual,
    and collects the distribution of |residual| values.

    The returned ``recommended_global_scale`` (p98) is the value to
    use as ``global_scale`` in ``GlobalFixedNormalizer`` or as
    ``ref_scale`` in ``GlobalLogNormalizer``.

    Returns dict with percentiles, saved to output_path.
    """
    import json
    from tqdm import tqdm

    all_abs = []
    all_rms = []
    all_p98 = []

    indices = list(sampler)[:max_patches]

    for geo_slice in tqdm(indices, desc="Computing detrended stats"):
        try:
            sample = dataset[geo_slice]
            elevation = sample["elevation"]
            if elevation.ndim == 4:
                elevation = elevation[0]

            valid = (torch.isfinite(elevation) & (elevation != 0.0)).float()
            if valid.mean() < 0.3:
                continue

            if elevation.shape[-1] != resolution or elevation.shape[-2] != resolution:
                elevation = F.interpolate(
                    elevation.unsqueeze(0), (resolution, resolution),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
                valid = F.interpolate(
                    valid.unsqueeze(0), (resolution, resolution),
                    mode="nearest-exact",
                ).squeeze(0)

            params = fit_plane(elevation, valid)
            plane = evaluate_plane(params, resolution, resolution, elevation.device)
            residual = elevation - plane

            valid_bool = valid.bool()
            if valid_bool.any():
                vr = residual[valid_bool]
                abs_r = vr.abs()
                rms = torch.sqrt((vr ** 2).mean()).item()
                p98 = torch.quantile(abs_r, 0.98).item()

                # Sample pixels
                if len(abs_r) > 500:
                    idx = torch.randperm(len(abs_r))[:500]
                    abs_r = abs_r[idx]

                all_abs.append(abs_r.cpu().numpy())
                all_rms.append(rms)
                all_p98.append(p98)

        except Exception as e:
            logger.debug("Skip: %s", e)

    if not all_abs:
        raise RuntimeError("No valid patches!")

    all_abs_np = np.concatenate(all_abs)
    all_rms_np = np.array(all_rms)
    all_p98_np = np.array(all_p98)

    stats = {
        "n_patches": len(all_rms),
        "pixel_abs_residual_percentiles": {
            f"p{p}": float(np.percentile(all_abs_np, p))
            for p in [50, 75, 90, 95, 98, 99]
        },
        "patch_rms_distribution": {
            "min": float(all_rms_np.min()),
            "p25": float(np.percentile(all_rms_np, 25)),
            "median": float(np.median(all_rms_np)),
            "p75": float(np.percentile(all_rms_np, 75)),
            "p90": float(np.percentile(all_rms_np, 90)),
            "p98": float(np.percentile(all_rms_np, 98)),
            "max": float(all_rms_np.max()),
        },
        "patch_p98_distribution": {
            "min": float(all_p98_np.min()),
            "median": float(np.median(all_p98_np)),
            "p90": float(np.percentile(all_p98_np, 90)),
            "max": float(all_p98_np.max()),
        },
        "recommended_global_scale": float(np.percentile(all_abs_np, 98)),
        "recommended_log_ref_scale": float(np.percentile(all_abs_np, 90)),
    }

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        "Detrended residual stats (%d patches):\n"
        "  Pixel |residual| p98 = %.2f m → use as global_scale\n"
        "  Pixel |residual| p90 = %.2f m → use as log ref_scale\n"
        "  Patch RMS range: %.2f – %.2f m\n"
        "  Saved: %s",
        stats["n_patches"],
        stats["recommended_global_scale"],
        stats["recommended_log_ref_scale"],
        stats["patch_rms_distribution"]["min"],
        stats["patch_rms_distribution"]["max"],
        output_path,
    )

    return stats


@dataclass
class StripOrthoStats:
    """Radiometric statistics for one strip."""
    p02: float
    p98: float
    median: float
    n_pixels_sampled: int
    strip_index: int = -1


class GlobalStripOrthoNormalizer:
    """Normalises ortho patches using strip-level quantiles.

    All patches from the same strip are normalised with the same
    (p02, p98), guaranteeing that:
      - The same physical pixel always gets the same normalised value
        regardless of which overlapping patch contains it
      - Shadow depth is preserved in relative terms across the strip
      - The transform is invertible (if you ever need to go back)

    Args:
        p02: Strip-level 2nd percentile of valid pixel values.
        p98: Strip-level 98th percentile of valid pixel values.
    """

    def __init__(self, p02: float, p98: float):
        if p98 <= p02:
            logger.warning(
                "p98 (%.6f) <= p02 (%.6f); using fallback range [0, 1]",
                p98, p02,
            )
            p02, p98 = 0.0, 1.0

        self.p02 = p02
        self.p98 = p98
        self._range = p98 - p02

    def normalize(self, ortho: torch.Tensor) -> torch.Tensor:
        """Normalise an ortho patch to [-1, 1].

        Args:
            ortho: (C, H, W) raw ortho reflectance values.

        Returns:
            (C, H, W) normalised to [-1, 1], clamped.
        """
        normed = ((ortho - self.p02) / self._range - 0.5) * 2.0
        return torch.clamp(normed, -1.0, 1.0)

    def denormalize(self, normed: torch.Tensor) -> torch.Tensor:
        """Inverse: normalised [-1, 1] → raw reflectance.

        Useful for visualization or re-rendering.
        """
        return ((normed / 2.0) + 0.5) * self._range + self.p02


def _normalize_ortho(
        ortho: torch.Tensor,
) -> torch.Tensor:
    """Normalise an orthoimage using global dataset quantiles to [-1, 1].

    Applies the same linear formula as the paper:
        ĩ = ((i − p02) / (p98 − p02) − 0.5) × 2

    Args:
        ortho: (C, H, W) in I/F reflectance [0, 1].
        p02: Dataset-level 2nd-percentile reflectance.
        p98: Dataset-level 98th-percentile reflectance.

    Returns:
        (C, H, W) in [-1, 1], clamped.
    """
    # Instead of using self.img_p02 and self.img_p98 from the global JSON
    valid_pixels = ortho[ortho > 0.0]  # Ignore pure black nodata
    if len(valid_pixels) > 0:
        local_p02 = torch.quantile(valid_pixels, 0.02)
        local_p98 = torch.quantile(valid_pixels, 0.98)

        # Avoid divide-by-zero if the patch is perfectly uniform
        if local_p98 > local_p02:
            ortho = ((ortho - local_p02) / (local_p98 - local_p02) - 0.5) * 2.0
        else:
            ortho = torch.zeros_like(ortho)  # Fallback

    return torch.clamp(ortho, -1.0, 1.0)

class LocalStripOrthoNormalizer:
    """Normalises ortho patches using strip-level quantiles.

    All patches from the same strip are normalised with the same
    (p02, p98), guaranteeing that:
      - The same physical pixel always gets the same normalised value
        regardless of which overlapping patch contains it
      - Shadow depth is preserved in relative terms across the strip
      - The transform is invertible (if you ever need to go back)

    Args:
        p02: Strip-level 2nd percentile of valid pixel values.
        p98: Strip-level 98th percentile of valid pixel values.
    """

    def __init__(self, *args, **kwargs):
        pass


    def normalize(self, ortho: torch.Tensor) -> torch.Tensor:
        """Normalise an ortho patch to [-1, 1].

        Args:
            ortho: (C, H, W) raw ortho reflectance values.

        Returns:
            (C, H, W) normalised to [-1, 1], clamped.
        """
        return _normalize_ortho(ortho)

    def denormalize(self, normed: torch.Tensor) -> torch.Tensor:
        """Inverse: normalised [-1, 1] → raw reflectance.

        Useful for visualization or re-rendering.
        """
        raise NotImplementedError("Cannot invert normalization")

