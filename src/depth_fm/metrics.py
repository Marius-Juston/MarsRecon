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
    """Container for a single evaluation result.

    Metrics follow the state-of-the-art depth estimation evaluation protocol:
      - SILog (Eigen et al., 2014; KITTI benchmark primary metric)
      - AbsRel, SqRel, RMSE, RMSElog (standard depth metrics)
      - δ₁, δ₂, δ₃ (threshold accuracy)
      - NormAng, SlopeRMSE (terrain-specific)
      - PhotoConsistency (SSIM of rendered shading vs orthoimage — resolution-
        independent quality signal for Mars DTM, not limited by GT resolution)
      - PSD ratio (power spectral density slope ratio — ensures predicted terrain
        has correct frequency content; Kirk et al. 2003)
    """
    abs_rel: float = 0.0
    sq_rel: float = 0.0
    rmse: float = 0.0
    rmse_log: float = 0.0
    si_log: float = 0.0          # Scale-Invariant Log Error (Eigen 2014)
    delta_1: float = 0.0
    delta_2: float = 0.0
    delta_3: float = 0.0
    normal_angular_error: float = 0.0
    slope_rmse: float = 0.0
    photo_consistency: float = 0.0  # SSIM(render(pred), ortho) — higher is better
    psd_ratio: float = 0.0         # log PSD slope ratio pred/gt — 1.0 is perfect
    scale: float = 1.0
    shift: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "abs_rel": self.abs_rel,
            "sq_rel": self.sq_rel,
            "rmse": self.rmse,
            "rmse_log": self.rmse_log,
            "si_log": self.si_log,
            "delta_1": self.delta_1,
            "delta_2": self.delta_2,
            "delta_3": self.delta_3,
            "normal_angular_error": self.normal_angular_error,
            "slope_rmse": self.slope_rmse,
            "photo_consistency": self.photo_consistency,
            "psd_ratio": self.psd_ratio,
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

    # Scale-Invariant Logarithmic Error (Eigen et al. 2014)
    # SILog = Var(log(pred) - log(gt)) — primary metric on KITTI benchmark
    # Using λ=1.0 for full scale invariance (Eigen recommends 0.5 for training,
    # but 1.0 for evaluation to be fully scale-invariant)
    if log_mask.any():
        d_log = np.log(p[log_mask]) - np.log(g[log_mask])
        si_log = float(np.sqrt(np.mean(d_log ** 2) - np.mean(d_log) ** 2)) * 100.0
    else:
        si_log = 0.0

    # Threshold accuracy (δ metrics)
    ratio = np.maximum(p_pos / g_pos, g_pos / p_pos)
    delta_1 = np.mean(ratio < 1.25) * 100.0
    delta_2 = np.mean(ratio < 1.25 ** 2) * 100.0
    delta_3 = np.mean(ratio < 1.25 ** 3) * 100.0

    # Surface normals angular error
    norm_err = _surface_normal_error(pred_aligned, gt, valid_mask)

    # Slope RMSE
    slope_err = _slope_rmse(pred_aligned, gt, valid_mask)

    # Power spectral density ratio (terrain frequency content)
    psd_r = _psd_slope_ratio(pred_aligned, gt, valid_mask)

    return DTMMetrics(
        abs_rel=float(abs_rel),
        sq_rel=float(sq_rel),
        rmse=float(rmse),
        rmse_log=float(rmse_log),
        si_log=float(si_log),
        delta_1=float(delta_1),
        delta_2=float(delta_2),
        delta_3=float(delta_3),
        normal_angular_error=float(norm_err),
        slope_rmse=float(slope_err),
        photo_consistency=0.0,  # Filled by caller when ortho+sun data available
        psd_ratio=float(psd_r),
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


def _psd_slope_ratio(
        pred: np.ndarray,
        gt: np.ndarray,
        valid_mask: np.ndarray,
) -> float:
    """Ratio of radially-averaged PSD slopes (pred vs GT).

    Mars terrain follows a power-law PSD: P(f) ∝ f^(-β).  A good depth
    prediction should reproduce the same spectral slope β.  The ratio
    β_pred / β_gt should be ≈ 1.0.  Values < 1 mean the prediction is
    too smooth (missing high-frequency detail); > 1 means too noisy.

    Reference: Kirk et al. (2003) "High-resolution topomapping of candidate
    MER landing sites with Mars Orbiter Camera narrow-angle images"
    """
    # Work on valid interior — fill NaN for FFT
    pred_filled = np.nan_to_num(pred, nan=float(np.nanmean(pred)))
    gt_filled = np.nan_to_num(gt, nan=float(np.nanmean(gt)))

    def _radial_psd(z):
        H, W = z.shape
        # Detrend (remove mean + linear trend for cleaner spectrum)
        z = z - np.mean(z)
        fft = np.fft.fft2(z)
        psd_2d = np.abs(fft) ** 2 / (H * W)
        psd_2d = np.fft.fftshift(psd_2d)

        cy, cx = H // 2, W // 2
        y, x = np.ogrid[-cy:H - cy, -cx:W - cx]
        r = np.sqrt(x ** 2 + y ** 2).astype(int)
        max_r = min(cy, cx)

        radial = np.zeros(max_r)
        counts = np.zeros(max_r)
        for ri in range(max_r):
            mask = r == ri
            radial[ri] = psd_2d[mask].sum()
            counts[ri] = mask.sum()
        valid = counts > 0
        radial[valid] /= counts[valid]
        return radial

    psd_pred = _radial_psd(pred_filled)
    psd_gt = _radial_psd(gt_filled)

    # Fit log-log slope over the middle frequency range (skip DC and Nyquist)
    n = len(psd_pred)
    lo, hi = max(2, n // 10), n // 2
    if hi <= lo + 3:
        return 1.0

    freqs = np.arange(lo, hi).astype(float)
    log_f = np.log(freqs)

    def _fit_slope(psd):
        vals = psd[lo:hi]
        valid = vals > 0
        if valid.sum() < 3:
            return 0.0
        log_p = np.log(vals[valid])
        lf = log_f[:valid.sum()] if valid.sum() < len(log_f) else log_f[valid]
        if len(lf) != len(log_p):
            lf = log_f[:len(log_p)]
        coeffs = np.polyfit(lf, log_p, 1)
        return coeffs[0]  # slope β

    beta_pred = _fit_slope(psd_pred)
    beta_gt = _fit_slope(psd_gt)

    if abs(beta_gt) < 1e-6:
        return 1.0
    return float(beta_pred / beta_gt)


def compute_photo_consistency(
        pred_elevation: np.ndarray,
        ortho_gray: np.ndarray,
        sun_vector: np.ndarray,
        intensity: float = 1.0,
        ambient: float = 0.0,
        lunar_lambert_weight: float = 0.5,
        valid_mask: np.ndarray | None = None,
) -> float:
    """SSIM between Lunar-Lambert render of predicted DTM and real orthoimage.

    This is the key resolution-independent quality metric for Mars DTM:
    even if the GT DTM is low-resolution, a good prediction should produce
    shading that matches the high-resolution orthoimage.

    Returns:
        SSIM value in [0, 1] (higher = better photometric consistency).
    """
    from skimage.metrics import structural_similarity as ssim

    H, W = pred_elevation.shape

    # Compute surface normals
    dy, dx = np.gradient(pred_elevation)
    spatial_scale = max(H, W) / 2.0
    n_x = -dx * spatial_scale
    n_y = -dy * spatial_scale
    n_z = np.ones_like(dx)
    norm = np.sqrt(n_x ** 2 + n_y ** 2 + n_z ** 2)
    norm = np.clip(norm, 1e-8, None)
    normals = np.stack([n_x / norm, n_y / norm, n_z / norm], axis=-1)

    # Lunar-Lambert render
    sv = sun_vector / (np.linalg.norm(sun_vector) + 1e-8)
    cos_i = np.sum(normals * sv, axis=-1)
    cos_i_clamped = np.clip(cos_i, 0.0, None)
    cos_e = normals[..., 2]

    lambert = cos_i_clamped
    lommel_seeliger = cos_i_clamped / (cos_i_clamped + cos_e + 1e-6)

    L = lunar_lambert_weight
    render = (L * lambert + (1.0 - L) * lommel_seeliger) * intensity + ambient
    render = np.clip(render, 0.0, 1.0).astype(np.float32)

    # Normalize ortho to [0, 1]
    ortho = ortho_gray.copy().astype(np.float32)
    o_min, o_max = np.nanpercentile(ortho, [1, 99])
    if o_max - o_min > 1e-6:
        ortho = np.clip((ortho - o_min) / (o_max - o_min), 0.0, 1.0)
    else:
        ortho = np.zeros_like(ortho)

    # Mask for SSIM
    if valid_mask is not None:
        # Fill invalid regions with mean to avoid SSIM edge artifacts
        render_m = np.where(valid_mask, render, np.nanmean(render))
        ortho_m = np.where(valid_mask, ortho, np.nanmean(ortho))
    else:
        render_m, ortho_m = render, ortho

    try:
        win_size = min(7, min(H, W))
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            return 0.0
        score = ssim(render_m, ortho_m, data_range=1.0, win_size=win_size)
        return float(score)
    except Exception:
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
        """Return mean ± std for each metric.

        Uses ddof=0 (population std) when n < 2 to avoid NaN.
        """
        if not self.records:
            return {}

        import pandas as pd
        df = pd.DataFrame(self.records)
        result = {}
        for col in df.columns:
            n = df[col].count()
            result[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std(ddof=0) if n < 2 else df[col].std()),
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