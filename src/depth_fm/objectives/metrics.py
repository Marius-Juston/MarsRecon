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
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize


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
    si_log: float = 0.0  # Scale-Invariant Log Error (Eigen 2014)
    delta_1: float = 0.0
    delta_2: float = 0.0
    delta_3: float = 0.0
    normal_angular_error: float = 0.0
    slope_rmse: float = 0.0
    dbf_score: float = 0.0  # Depth Boundary F-Score
    curvature_rmse: float = 0.0  # Laplacian RMSE
    ms_ssim_topo: float = 0.0  # MS-SSIM on actual elevation
    photo_consistency: float = 0.0  # SSIM(render(pred), ortho) — higher is better
    psd_ratio: float = 0.0  # log PSD slope ratio pred/gt — 1.0 is perfect
    scale: float = 1.0
    shift: float = 0.0
    patch_swd: float = 0.0

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
            "dbf_score": self.dbf_score,
            "curvature_rmse": self.curvature_rmse,
            "ms_ssim_topo": self.ms_ssim_topo,
            "photo_consistency": self.photo_consistency,
            "psd_ratio": self.psd_ratio,
            "patch_swd": self.patch_swd,
        }


def _depth_boundary_fscore(
        pred: np.ndarray,
        gt: np.ndarray,
        valid_mask: np.ndarray,
        threshold_pct: float = 85.0,
        dist_tolerance: float = 2.0
) -> float:
    """Computes the Depth Boundary F-Score (DBF)."""
    # 1. Compute gradients (Sobel approximation)
    dy_p, dx_p = np.gradient(pred)
    dy_g, dx_g = np.gradient(gt)

    grad_p = np.sqrt(dx_p ** 2 + dy_p ** 2)
    grad_g = np.sqrt(dx_g ** 2 + dy_g ** 2)

    grad_p[~valid_mask] = 0
    grad_g[~valid_mask] = 0

    # 2. Extract binary edges via percentile thresholding (top 15% gradients)
    thresh_g = np.percentile(grad_g[valid_mask], threshold_pct)
    if thresh_g <= 1e-6: return 0.0  # Flat terrain, no boundaries

    edges_g = (grad_g > thresh_g)
    edges_p = (grad_p > thresh_g)  # Use GT threshold for fair comparison

    # 3. Distance transforms to allow sub-pixel/minor spatial shifts
    dist_g = distance_transform_edt(~edges_g)
    dist_p = distance_transform_edt(~edges_p)

    # 4. Precision and Recall
    tp_p = np.sum(edges_p & (dist_g <= dist_tolerance))  # Predicted edges near GT edges
    tp_g = np.sum(edges_g & (dist_p <= dist_tolerance))  # GT edges near predicted edges

    n_pred = np.sum(edges_p)
    n_gt = np.sum(edges_g)

    if n_pred == 0 or n_gt == 0: return 0.0

    precision = tp_p / n_pred
    recall = tp_g / n_gt

    if precision + recall == 0: return 0.0
    return float(2 * (precision * recall) / (precision + recall))


def _curvature_rmse(
        pred: np.ndarray,
        gt: np.ndarray,
        valid_mask: np.ndarray,
) -> float:
    """RMSE of the topographic Laplacian (mean curvature)."""
    # Smooth slightly to mitigate high-frequency quantization noise
    pred_s = gaussian_filter(pred, sigma=1.0)
    gt_s = gaussian_filter(gt, sigma=1.0)

    dy_p, dx_p = np.gradient(pred_s)
    dy_g, dx_g = np.gradient(gt_s)

    d2y_p, _ = np.gradient(dy_p)
    _, d2x_p = np.gradient(dx_p)
    laplacian_p = d2x_p + d2y_p

    d2y_g, _ = np.gradient(dy_g)
    _, d2x_g = np.gradient(dx_g)
    laplacian_g = d2x_g + d2y_g

    interior = valid_mask.copy()
    interior[0:2, :] = interior[-2:, :] = interior[:, 0:2] = interior[:, -2:] = False

    if interior.any():
        return float(np.sqrt(np.mean((laplacian_p[interior] - laplacian_g[interior]) ** 2)))
    return 0.0


def _ms_ssim_topography(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> float:
    """Computes Structural Similarity directly on Topography."""

    # Normalize to [0,1] based on GT for stable SSIM
    g_min, g_max = np.nanmin(gt[valid_mask]), np.nanmax(gt[valid_mask])
    if g_max - g_min < 1e-6: return 1.0

    p_norm = np.clip((pred - g_min) / (g_max - g_min), 0, 1)
    g_norm = np.clip((gt - g_min) / (g_max - g_min), 0, 1)

    # Fill nan for downsampling stability
    p_norm[~valid_mask] = np.nanmean(p_norm)
    g_norm[~valid_mask] = np.nanmean(g_norm)

    weights = [0.0448, 0.2856, 0.3001]  # 3 scales is sufficient for 512px patches
    msssim = 0.0

    try:
        for i, w in enumerate(weights):
            if i > 0:
                h, w_dim = p_norm.shape
                p_norm = resize(p_norm, (h // 2, w_dim // 2), anti_aliasing=True)
                g_norm = resize(g_norm, (h // 2, w_dim // 2), anti_aliasing=True)

            win = min(7, min(p_norm.shape))
            if win % 2 == 0: win -= 1
            if win < 3: break

            s = ssim(p_norm, g_norm, data_range=1.0, win_size=win)
            msssim += w * s

        return float(msssim / sum(weights[:i + 1]))
    except Exception:
        return 0.0


def _patch_swd(
        pred: np.ndarray,
        gt: np.ndarray,
        valid_mask: np.ndarray,
        patch_size: int = 7,
        num_projections: int = 128,
) -> float:
    """Sliced Wasserstein Distance on local topographic patches."""
    from numpy.lib.stride_tricks import sliding_window_view

    # Impute invalid regions with mean to avoid NaN propagation during sorting
    p_fill = np.where(valid_mask, pred, np.nanmean(pred))
    g_fill = np.where(valid_mask, gt, np.nanmean(gt))

    # Extract sliding windows (H-p+1, W-p+1, p, p) and flatten to (N_patches, p^2)
    p_patches = sliding_window_view(p_fill, (patch_size, patch_size)).reshape(-1, patch_size ** 2)
    g_patches = sliding_window_view(g_fill, (patch_size, patch_size)).reshape(-1, patch_size ** 2)

    # Subsample to cap memory and compute time (Monte Carlo estimation)
    max_samples = 10000
    if len(p_patches) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(p_patches), max_samples, replace=False)
        p_patches, g_patches = p_patches[idx], g_patches[idx]

    d = patch_size ** 2

    # Sample uniform directions on the unit hypersphere
    theta = np.random.randn(d, num_projections)
    theta /= np.linalg.norm(theta, axis=0, keepdims=True)

    # Project patches onto 1D lines and sort (simulating inverse CDF matching)
    p_proj = np.sort(p_patches @ theta, axis=0)
    g_proj = np.sort(g_patches @ theta, axis=0)

    # Integrate squared differences across all slices
    return float(np.mean((p_proj - g_proj) ** 2))


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

    dbf_val = _depth_boundary_fscore(pred_aligned, gt, valid_mask)
    curv_val = _curvature_rmse(pred_aligned, gt, valid_mask)
    ms_ssim_val = _ms_ssim_topography(pred_aligned, gt, valid_mask)

    # Power spectral density ratio (terrain frequency content)
    psd_r = _psd_slope_ratio(pred_aligned, gt, valid_mask)

    swd_val = _patch_swd(pred_aligned, gt, valid_mask)

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
        dbf_score=float(dbf_val),
        curvature_rmse=float(curv_val),
        ms_ssim_topo=float(ms_ssim_val),
        photo_consistency=0.0,  # Filled by caller when ortho+sun data available
        psd_ratio=float(psd_r),
        scale=scale,
        shift=shift,
        patch_swd=float(swd_val),
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
