"""
DTM evaluation metrics for Mars DepthFM.

Implements all standard monocular depth estimation metrics used in the
literature (DepthFM, Marigold, MiDaS), plus Mars-DTM–specific metrics.

All metrics operate on **affine-aligned** predictions: since flow matching
produces affine-invariant depth, we first solve for optimal scale and shift
via least-squares before computing error metrics (following DepthFM §4).

Metrics:
    AbsRel     — mean |d - d*| / d*
    SqRel      — mean (d - d*)² / d*
    RMSE       — root mean squared error (metres)
    RMSElog    — RMSE on log-depth
    δ₁         — % pixels with max(d/d*, d*/d) < 1.25
    δ₂         — same with threshold 1.25²
    δ₃         — same with threshold 1.25³
    NormAng    — mean angular error between surface normals (degrees)
    Slope RMSE — RMSE of slope magnitude (degrees)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DTMMetrics:
    """Container for a single evaluation result."""
    abs_rel: float = 0.0
    sq_rel: float = 0.0
    rmse: float = 0.0
    rmse_log: float = 0.0
    delta_1: float = 0.0
    delta_2: float = 0.0
    delta_3: float = 0.0
    normal_angular_error: float = 0.0
    slope_rmse: float = 0.0
    scale: float = 1.0
    shift: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "abs_rel": self.abs_rel,
            "sq_rel": self.sq_rel,
            "rmse": self.rmse,
            "rmse_log": self.rmse_log,
            "delta_1": self.delta_1,
            "delta_2": self.delta_2,
            "delta_3": self.delta_3,
            "normal_angular_error": self.normal_angular_error,
            "slope_rmse": self.slope_rmse,
        }


def affine_align(
        pred: np.ndarray,
        gt: np.ndarray,
        valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """Solve for optimal scale s and shift t: aligned = s * pred + t.

    Uses least-squares to minimise || s * pred + t - gt ||² over valid pixels.
    This is the standard affine-invariant alignment used by DepthFM and Marigold.
    """
    if valid_mask is None:
        valid_mask = np.isfinite(pred) & np.isfinite(gt)

    p = pred[valid_mask].flatten()
    g = gt[valid_mask].flatten()

    if len(p) < 2:
        return pred.copy(), 1.0, 0.0

    # Least-squares: [s, t] = argmin || [p, 1] @ [s; t] - g ||²
    A = np.stack([p, np.ones_like(p)], axis=-1)
    result = np.linalg.lstsq(A, g, rcond=None)
    s, t = result[0]

    aligned = s * pred + t
    return aligned, float(s), float(t)


def compute_depth_metrics(
        pred: np.ndarray,
        gt: np.ndarray,
        valid_mask: np.ndarray | None = None,
        align: bool = True,
) -> DTMMetrics:
    """Compute all depth metrics between predicted and ground-truth DTMs.

    Args:
        pred: predicted depth (H, W), arbitrary scale
        gt: ground-truth depth (H, W), in metres
        valid_mask: boolean mask of valid pixels (H, W)
        align: if True, affine-align prediction to GT first

    Returns:
        DTMMetrics dataclass
    """
    if valid_mask is None:
        valid_mask = np.isfinite(pred) & np.isfinite(gt) & (gt > 0)

    if align:
        pred_aligned, scale, shift = affine_align(pred, gt, valid_mask)
    else:
        pred_aligned, scale, shift = pred.copy(), 1.0, 0.0

    p = pred_aligned[valid_mask]
    g = gt[valid_mask]

    if len(p) == 0:
        return DTMMetrics()

    # Clamp to positive for ratio metrics
    p_pos = np.clip(p, 1e-6, None)
    g_pos = np.clip(g, 1e-6, None)

    # Standard depth metrics
    abs_rel = np.mean(np.abs(p - g) / g_pos)
    sq_rel = np.mean((p - g) ** 2 / g_pos)
    rmse = np.sqrt(np.mean((p - g) ** 2))

    # Log-space RMSE (only where both are positive)
    log_mask = (p > 0) & (g > 0)
    if log_mask.any():
        rmse_log = np.sqrt(np.mean((np.log(p[log_mask]) - np.log(g[log_mask])) ** 2))
    else:
        rmse_log = 0.0

    # Threshold accuracy (δ metrics)
    ratio = np.maximum(p_pos / g_pos, g_pos / p_pos)
    delta_1 = np.mean(ratio < 1.25) * 100.0
    delta_2 = np.mean(ratio < 1.25 ** 2) * 100.0
    delta_3 = np.mean(ratio < 1.25 ** 3) * 100.0

    # Surface normals angular error
    norm_err = _surface_normal_error(pred_aligned, gt, valid_mask)

    # Slope RMSE
    slope_err = _slope_rmse(pred_aligned, gt, valid_mask)

    return DTMMetrics(
        abs_rel=float(abs_rel),
        sq_rel=float(sq_rel),
        rmse=float(rmse),
        rmse_log=float(rmse_log),
        delta_1=float(delta_1),
        delta_2=float(delta_2),
        delta_3=float(delta_3),
        normal_angular_error=float(norm_err),
        slope_rmse=float(slope_err),
        scale=scale,
        shift=shift,
    )


def _surface_normal_error(
        pred: np.ndarray,
        gt: np.ndarray,
        valid_mask: np.ndarray,
) -> float:
    """Mean angular error between surface normals (degrees)."""

    def _normals(z):
        dy, dx = np.gradient(z)
        n = np.stack([-dx, -dy, np.ones_like(dx)], axis=-1)
        norm = np.linalg.norm(n, axis=-1, keepdims=True)
        norm = np.clip(norm, 1e-8, None)
        return n / norm

    n_pred = _normals(pred)
    n_gt = _normals(gt)

    # Cosine similarity per pixel
    cos_sim = np.sum(n_pred * n_gt, axis=-1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    angles = np.degrees(np.arccos(cos_sim))

    # Interior valid mask (gradient is unreliable at edges)
    interior = valid_mask.copy()
    interior[0, :] = interior[-1, :] = interior[:, 0] = interior[:, -1] = False

    if interior.any():
        return float(np.mean(angles[interior]))
    return 0.0


def _slope_rmse(
        pred: np.ndarray,
        gt: np.ndarray,
        valid_mask: np.ndarray,
) -> float:
    """RMSE of slope magnitude (degrees)."""

    def _slope(z):
        dy, dx = np.gradient(z)
        return np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))

    s_pred = _slope(pred)
    s_gt = _slope(gt)

    interior = valid_mask.copy()
    interior[0, :] = interior[-1, :] = interior[:, 0] = interior[:, -1] = False

    if interior.any():
        return float(np.sqrt(np.mean((s_pred[interior] - s_gt[interior]) ** 2)))
    return 0.0


@dataclass
class MetricsAggregator:
    """Accumulates per-sample metrics and computes summary statistics."""
    records: list[dict] = field(default_factory=list)
    tile_ids: list[str] = field(default_factory=list)

    def add(self, metrics: DTMMetrics, tile_id: str = "") -> None:
        self.records.append(metrics.to_dict())
        self.tile_ids.append(tile_id)

    def summary(self) -> dict[str, dict[str, float]]:
        """Return mean ± std for each metric."""
        if not self.records:
            return {}

        import pandas as pd
        df = pd.DataFrame(self.records)
        result = {}
        for col in df.columns:
            result[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "median": float(df[col].median()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
        return result

    def per_sample_dataframe(self):
        import pandas as pd
        df = pd.DataFrame(self.records)
        df["tile_id"] = self.tile_ids
        return df

    def worst_k(self, metric: str = "rmse", k: int = 10) -> list[tuple[str, float]]:
        """Return the k worst-performing tile_ids by the given metric."""
        df = self.per_sample_dataframe()
        worst = df.nlargest(k, metric)
        return list(zip(worst["tile_id"], worst[metric]))

    def best_k(self, metric: str = "rmse", k: int = 10) -> list[tuple[str, float]]:
        df = self.per_sample_dataframe()
        best = df.nsmallest(k, metric)
        return list(zip(best["tile_id"], best[metric]))
