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
        clip: Whether or not to clip the range to be withing [-1, 1] explicitly
    """

    def __init__(self, ref_scale: float, clip: bool = False):
        self.clip = clip
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

        if self.clip:
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
            log_scale_mean: float = 1.5,  # ~exp(1.5) ≈ 4.5m
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
