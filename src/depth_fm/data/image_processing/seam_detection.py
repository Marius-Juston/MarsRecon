"""Seam and TIN-artifact detection for HiRISE DTM patches.

Public entry points:

* `detect_seam_artifact` — main detector. Returns a `SeamResult`.
* `is_tin_artifact`      — scale-invariant TIN-flat-region detector.
* `SeamResult`           — dataclass with score, line geometry, and (optional)
                           diagnostic arrays.

Public helpers (also used by training-time diagnostics):

* `compute_piecewise_linearity`   — Hough-based linearity score.
* `compute_artifact_multipliers`  — span/sparsity structural multipliers.
* `compute_spatial_isolation`     — perpendicular cross-section isolation score.

Private gradient helpers (`_ensure_bchw`, `_sobel_mag`, `_sharp_grad_mag`,
`_build_oriented_kernels`) stay inside this module — they are only used by
`detect_seam_artifact`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def compute_piecewise_linearity(seam_heatmap: np.ndarray, threshold_ratio: float = 0.3) -> tuple[float, np.ndarray]:
    """Score how piecewise-linear the high-scoring pixels are (handles corners).

    Returns (linearity_score, line_mask) where linearity is in [0.0, 1.0]:
    1.0 = highly structured/linear, 0.0 = curved/messy.
    """
    heatmap_norm = cv2.normalize(seam_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary_map = cv2.threshold(heatmap_norm, int(255 * threshold_ratio), 255, cv2.THRESH_BINARY)

    # Bridge missing line segments before edge detection so fragmented seams
    # heal into a continuous solid line. 15×15 closing merges blobs that are
    # separated by up to ~15 pixels.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, close_kernel)

    edges = cv2.Canny(closed_map, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, rho=2, theta=np.pi / 180, threshold=50,
                            minLineLength=40, maxLineGap=30)

    line_mask = np.zeros_like(binary_map)

    if lines is None:
        return 0.0, line_mask

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=4)

    valid_signal_pixels = np.count_nonzero(binary_map)
    if valid_signal_pixels == 0:
        return 0.0, line_mask

    structured_pixels = np.count_nonzero(cv2.bitwise_and(binary_map, line_mask))

    # True seams (even with corners) score close to 1.0; natural curves fall
    # apart in the Hough transform and score low.
    return structured_pixels / valid_signal_pixels, line_mask


@dataclass
class SeamResult:
    seam_score: float
    ortho_score: float
    dtm_score: float
    cohens_d: float
    best_angle_rad: float
    best_y: int
    best_x: int
    line_length: int
    num_angles: int
    span: float  # checks length (kills craters)
    sparsity: float  # checks density (kills dunes)
    composite_score: float
    is_seam: bool
    # heavy arrays, only populated when return_diagnostics=True
    seam_heatmap: Optional[np.ndarray] = None
    cohens_d_heatmap: Optional[np.ndarray] = None
    per_angle_max: Optional[np.ndarray] = None
    diag_hot_mask: Optional[np.ndarray] = None
    diag_closed_components: Optional[np.ndarray] = None
    diag_hough_lines: Optional[np.ndarray] = None
    diag_isolation_profile: Optional[tuple] = None

    def to_dict(self, drop_arrays: bool = True) -> Dict:
        d = asdict(self)
        if drop_arrays:
            for k in ("seam_heatmap", "cohens_d_heatmap", "per_angle_max",
                      "diag_hot_mask", "diag_closed_components",
                      "diag_hough_lines", "diag_isolation_profile"):
                d.pop(k, None)
        return d

    @property
    def best_angle_deg(self) -> float:
        return math.degrees(self.best_angle_rad)

    def line_endpoints(self, clip_hw=None):
        """Return (y1, x1, y2, x2) pixel endpoints of the best-scoring line."""
        half = self.line_length // 2
        dx, dy = math.cos(self.best_angle_rad), math.sin(self.best_angle_rad)
        y1 = self.best_y - half * dy
        x1 = self.best_x - half * dx
        y2 = self.best_y + half * dy
        x2 = self.best_x + half * dx
        if clip_hw is not None:
            H, W = clip_hw
            y1 = float(np.clip(y1, 0, H - 1))
            y2 = float(np.clip(y2, 0, H - 1))
            x1 = float(np.clip(x1, 0, W - 1))
            x2 = float(np.clip(x2, 0, W - 1))
        return y1, x1, y2, x2


def compute_artifact_multipliers(
        seam_heatmap: np.ndarray,
        valid_mask: np.ndarray,
        threshold: float = 0.2,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Compute structural multipliers that distinguish seams from natural features.

    Returns (span_ratio, sparsity, hot_mask, labels):
    * span_ratio — defeats short craters; true seams cross the whole tile.
    * sparsity   — defeats dense dunes; true seams are a singular line.
    """
    max_val = np.max(seam_heatmap)
    if max_val <= 0:
        return 0.0, 1.0

    hot_mask = (seam_heatmap > max_val * threshold)

    # SPARSITY (defeats repeating textures like dunes)
    hot_area = np.count_nonzero(hot_mask & valid_mask)
    valid_area = max(np.count_nonzero(valid_mask), 1)
    density = hot_area / valid_area
    # True seams cover ~2-4% of the image; dune fields cover >15%.
    sparsity = math.exp(-density * 15.0)

    # STRUCTURAL SPAN (defeats short, isolated craters)
    binary = (hot_mask.astype(np.uint8)) * 255

    kernel = np.ones((9, 9), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    span_ratio = 0.0
    if num_labels > 1:
        H, W = seam_heatmap.shape
        max_span = 0.0
        for i in range(1, num_labels):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            span = math.hypot(w, h)
            if span > max_span:
                max_span = span
        max_possible_span = float(max(H, W))
        span_ratio = min(max_span / max_possible_span, 1.0)

    return float(span_ratio), float(sparsity), hot_mask, labels


def compute_spatial_isolation(
        score_map: torch.Tensor,
        valid_mask: torch.Tensor,
        best_x: int,
        best_y: int,
        angle_rad: float,
        profile_length: int = 50,
        exclusion_zone: int = 12,
) -> tuple[float, Optional[tuple]]:
    """Sample a perpendicular slice across the seam.

    Returns the ratio of the central peak to the surrounding parallel
    background, plus the raw profile data arrays for visualization.
    """
    H, W = score_map.shape
    device = score_map.device

    px = -math.sin(angle_rad)
    py = math.cos(angle_rad)

    t = torch.arange(-profile_length, profile_length + 1, device=device, dtype=torch.float32)
    grid_x = best_x + t * px
    grid_y = best_y + t * py

    in_bounds = (grid_x >= 0) & (grid_x < W) & (grid_y >= 0) & (grid_y < H)
    t = t[in_bounds]
    grid_x = grid_x[in_bounds]
    grid_y = grid_y[in_bounds]

    if len(t) == 0:
        return 0.0, None

    norm_x = (grid_x / (W - 1)) * 2 - 1
    norm_y = (grid_y / (H - 1)) * 2 - 1
    grid = torch.stack([norm_x, norm_y], dim=-1).view(1, 1, -1, 2)

    score_map_4d = score_map.unsqueeze(0).unsqueeze(0)
    valid_mask_4d = valid_mask.unsqueeze(0).unsqueeze(0)

    samples = F.grid_sample(score_map_4d, grid, mode="bilinear", align_corners=True).squeeze()
    v_samples = F.grid_sample(valid_mask_4d, grid, mode="nearest", align_corners=True).squeeze()

    center_mask = (t.abs() <= exclusion_zone) & (v_samples > 0)
    bg_mask = (t.abs() > exclusion_zone) & (v_samples > 0)

    profile_data = (
        t.cpu().numpy(),
        samples.cpu().numpy(),
        center_mask.cpu().numpy(),
        bg_mask.cpu().numpy(),
    )

    if not center_mask.any():
        return 0.0, profile_data
    if not bg_mask.any():
        # No valid background — treat as an isolated edge.
        return 5.0, profile_data

    center_max = samples[center_mask].amax().item()
    bg_mean = samples[bg_mask].mean().item()

    if bg_mean < 1e-6:
        return 20.0, profile_data

    return center_max / bg_mean, profile_data


# ---------------------------------------------------------------------------
# Private gradient helpers (used by detect_seam_artifact only)
# ---------------------------------------------------------------------------

def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.view(1, 1, *x.shape)
    if x.dim() == 3:
        return x.unsqueeze(0)
    if x.dim() == 4:
        return x
    raise ValueError(f"Unexpected tensor shape {tuple(x.shape)}")


def _sobel_mag(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    device, dtype = x.device, x.dtype
    safe = x.clone()
    safe[~valid_mask] = 0.0
    sx = torch.tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
        device=device, dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0
    sy = torch.tensor(
        [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
        device=device, dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0
    return torch.sqrt(
        F.conv2d(safe, sx, padding=1) ** 2
        + F.conv2d(safe, sy, padding=1) ** 2
        + 1e-8
    )


def _sharp_grad_mag(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Strict 1-D forward-difference gradient magnitude.

    Avoids the spatial blur of a 3×3 Sobel so 1–3 pixel step edges remain
    perfectly sharp.
    """
    device, dtype = x.device, x.dtype
    safe = x.clone()
    safe[~valid_mask] = 0.0

    kx = torch.tensor([-1., 1.], device=device, dtype=dtype).view(1, 1, 1, 2)
    ky = torch.tensor([[-1.], [1.]], device=device, dtype=dtype).view(1, 1, 2, 1)

    padded_x = F.pad(safe, (0, 1, 0, 0), mode="replicate")
    padded_y = F.pad(safe, (0, 0, 0, 1), mode="replicate")

    gx = F.conv2d(padded_x, kx)
    gy = F.conv2d(padded_y, ky)

    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


@lru_cache
def _build_oriented_kernels(line_length, num_angles, side_offset, device, dtype):
    ks = line_length + 2 * side_offset + 2
    if ks % 2 == 0:
        ks += 1
    kc = ks // 2
    half = line_length // 2
    shape = (num_angles, 1, ks, ks)
    line_ker = torch.zeros(shape, device=device, dtype=dtype)
    left_ker = torch.zeros(shape, device=device, dtype=dtype)
    right_ker = torch.zeros(shape, device=device, dtype=dtype)
    for i in range(num_angles):
        theta = math.pi * i / num_angles
        dx, dy = math.cos(theta), math.sin(theta)
        px, py = -dy, dx
        for r in range(line_length):
            t = r - half
            xc = int(round(kc + t * dx))
            yc = int(round(kc + t * dy))
            if 0 <= xc < ks and 0 <= yc < ks:
                line_ker[i, 0, yc, xc] = 1.0
            xl = int(round(kc + t * dx - side_offset * px))
            yl = int(round(kc + t * dy - side_offset * py))
            if 0 <= xl < ks and 0 <= yl < ks:
                left_ker[i, 0, yl, xl] = 1.0
            xr = int(round(kc + t * dx + side_offset * px))
            yr = int(round(kc + t * dy + side_offset * py))
            if 0 <= xr < ks and 0 <= yr < ks:
                right_ker[i, 0, yr, xr] = 1.0
    return line_ker, left_ker, right_ker, kc


@torch.no_grad()
def detect_seam_artifact(
        ortho: torch.Tensor,
        elevation: torch.Tensor,
        valid_mask: torch.Tensor,
        line_length: int = 41,
        num_angles: int = 12,
        side_offset: int = 2,
        min_valid_ratio: float = 0.6,
        ortho_weight: float = 1.0,
        dtm_weight: float = 0.3,
        erosion_kernel: int = 9,
        seam_threshold: float = 2.4,
        return_diagnostics: bool = False,
) -> SeamResult:
    """Detect seam artifacts (mosaicking discontinuities) in a HiRISE patch.

    Returns a `SeamResult`. With `return_diagnostics=True`, the result also
    carries `seam_heatmap`, `cohens_d_heatmap`, and `per_angle_max`, used by
    the visualization and refinement UI to show *where* and *along which
    angle* the detector fired.
    """
    ortho = _ensure_bchw(ortho).float()
    elevation = _ensure_bchw(elevation).float()
    valid_mask = _ensure_bchw(valid_mask).bool()
    device, dtype = ortho.device, ortho.dtype
    ortho_gray = ortho.mean(dim=1, keepdim=True)

    H, W = ortho_gray.shape[-2:]

    ortho_grad = _sharp_grad_mag(ortho_gray, valid_mask)
    dtm_grad = _sharp_grad_mag(elevation, valid_mask)

    pad_e = erosion_kernel // 2
    invalid_f = (~valid_mask).float()
    dilated = F.max_pool2d(invalid_f, kernel_size=erosion_kernel, stride=1, padding=pad_e)
    eroded_valid = (dilated == 0.0)
    ev_f = eroded_valid.float()

    empty_result = SeamResult(
        seam_score=0.0, ortho_score=0.0, dtm_score=0.0, cohens_d=0.0,
        best_angle_rad=0.0, best_y=H // 2, best_x=W // 2,
        line_length=line_length, num_angles=num_angles,
        composite_score=0, sparsity=0, span=0,
        is_seam=False,
    )

    if ev_f.sum() < 100:
        return empty_result

    def _bg(signal):
        num = (signal * ev_f).sum()
        den = ev_f.sum().clamp(min=1)
        return (num / den).clamp(min=1e-6)

    bg_ortho, bg_dtm = _bg(ortho_grad), _bg(dtm_grad)

    line_ker, left_ker, right_ker, pad = _build_oriented_kernels(
        line_length, num_angles, side_offset, device, dtype
    )

    def _line_avg(signal, ker):
        num = F.conv2d(signal * ev_f, ker, padding=pad)
        cnt = F.conv2d(ev_f, ker, padding=pad)
        return num / cnt.clamp(min=1.0), cnt

    ortho_line_avg, line_cnt = _line_avg(ortho_grad, line_ker)
    dtm_line_avg, _ = _line_avg(dtm_grad, line_ker)

    def _moments(signal, ker):
        n = F.conv2d(ev_f, ker, padding=pad).clamp(min=1.0)
        s = F.conv2d(signal * ev_f, ker, padding=pad)
        s2 = F.conv2d((signal ** 2) * ev_f, ker, padding=pad)
        mean = s / n
        var = (s2 / n - mean ** 2).clamp(min=0.0)
        return mean, var, n

    mu_l, var_l, n_l = _moments(ortho_gray, left_ker)
    mu_r, var_r, n_r = _moments(ortho_gray, right_ker)
    global_std = ortho_gray[valid_mask].std().clamp(min=1e-4)
    std_floor = 0.1 * global_std
    pooled_std = torch.sqrt(((var_l + var_r) / 2.0).clamp(min=0.0)) + std_floor
    cohens_d = (mu_l - mu_r).abs() / pooled_std

    mu_l_dtm, var_l_dtm, _ = _moments(elevation, left_ker)
    mu_r_dtm, var_r_dtm, _ = _moments(elevation, right_ker)

    global_dtm_std = elevation[valid_mask].std().clamp(min=1e-4)
    dtm_std_floor = 0.1 * global_dtm_std
    pooled_dtm_std = torch.sqrt(((var_l_dtm + var_r_dtm) / 2.0).clamp(min=0.0)) + dtm_std_floor

    dtm_cohens_d = (mu_l_dtm - mu_r_dtm).abs() / pooled_dtm_std

    min_line = min_valid_ratio * line_length
    min_side = min_valid_ratio * line_length * 0.5
    valid_q = (line_cnt >= min_line) & (n_l >= min_side) & (n_r >= min_side)
    if not valid_q.any():
        return empty_result

    ortho_norm = ortho_line_avg / bg_ortho
    dtm_norm = dtm_line_avg / bg_dtm
    gradient_term = ortho_weight * ortho_norm + dtm_weight * dtm_norm

    # Trigger if EITHER ortho OR dtm has a massive distribution shift.
    distribution_gate = torch.clamp(torch.maximum(cohens_d, dtm_cohens_d), min=0.1, max=3.0)

    combined = gradient_term * distribution_gate

    combined_masked = torch.where(valid_q, combined, torch.full_like(combined, -1e10))

    cm0 = combined_masked[0]
    flat = cm0.flatten().argmax()
    a_idx = int(flat // (H * W))
    rem = int(flat % (H * W))
    y_idx = rem // W
    x_idx = rem % W
    best_angle_rad = math.pi * a_idx / num_angles

    seam_score = float(cm0.amax().item())

    heatmap = cm0.amax(dim=0).cpu().numpy()
    heatmap[heatmap < 0] = 0.0

    valid_np = ev_f[0, 0].cpu().numpy().astype(bool)

    threshold = 0.25

    span, sparsity, diag_hot_mask, diag_labels = compute_artifact_multipliers(heatmap, valid_np, threshold)

    linearity, diag_hough_lines = compute_piecewise_linearity(heatmap, threshold_ratio=threshold)

    isolation_score, diag_iso_profile = compute_spatial_isolation(
        score_map=cm0.amax(dim=0),
        valid_mask=ev_f[0, 0],
        best_x=int(x_idx),
        best_y=int(y_idx),
        angle_rad=best_angle_rad,
    )

    isolation_mult = min(max((isolation_score - 1.5) / 1.5, 0.0), 1.0)
    composite_score = seam_score * (0.5 + 0.5 * span) * sparsity * linearity * isolation_mult

    result = SeamResult(
        seam_score=seam_score,
        ortho_score=float(torch.where(valid_q, ortho_norm, torch.full_like(ortho_norm, -1e10)).amax().item()),
        dtm_score=float(torch.where(valid_q, dtm_norm, torch.full_like(dtm_norm, -1e10)).amax().item()),
        cohens_d=float(torch.where(valid_q, cohens_d, torch.zeros_like(cohens_d)).amax().item()),
        best_angle_rad=best_angle_rad,
        best_y=int(y_idx),
        best_x=int(x_idx),
        line_length=line_length,
        span=span,
        sparsity=sparsity,
        num_angles=num_angles,
        composite_score=composite_score,
        is_seam=composite_score > seam_threshold,
    )

    if return_diagnostics:
        d0 = torch.where(valid_q, cohens_d, torch.full_like(cohens_d, float("nan")))[0]
        d_heatmap = d0.amax(dim=0).cpu().numpy()
        per_angle = cm0.amax(dim=(1, 2)).cpu().numpy()
        per_angle[per_angle < 0] = 0.0

        result.seam_heatmap = heatmap
        result.cohens_d_heatmap = d_heatmap
        result.per_angle_max = per_angle

        result.diag_hot_mask = diag_hot_mask
        result.diag_closed_components = diag_labels
        result.diag_hough_lines = diag_hough_lines
        result.diag_isolation_profile = diag_iso_profile

    return result


def is_tin_artifact(
        elevation: torch.Tensor,
        valid_mask: torch.Tensor,
        kernel_size: int = 32,
) -> float:
    """Scale-invariant TIN-artifact detection using localized maximum density.

    Returns the maximum local fraction of zero-curvature pixels within any
    `kernel_size × kernel_size` window. Patches with stretched-triangle TIN
    artifacts produce values near 1.0; natural terrain produces values <0.5.
    """
    if elevation.dim() == 2:
        elevation = elevation.view(1, 1, elevation.shape[0], elevation.shape[1])
        valid_mask = valid_mask.view(1, 1, valid_mask.shape[0], valid_mask.shape[1])

    safe_elev = elevation.clone()
    safe_elev[~valid_mask] = 0.0

    laplacian_kernel = torch.tensor([[[[0.0, 1.0, 0.0],
                                       [1.0, -4.0, 1.0],
                                       [0.0, 1.0, 0.0]]]], device=elevation.device)
    laplacian = F.conv2d(safe_elev, laplacian_kernel, padding=1)

    invalid_mask = (~valid_mask).float()
    dilated_invalid = F.max_pool2d(invalid_mask, kernel_size=3, stride=1, padding=1)
    eroded_valid = (dilated_invalid == 0.0).float()

    # True (1.0) if curvature ≈ 0 AND the pixel is deeply valid.
    zero_curvature_mask = ((laplacian.abs() < 1e-2) * eroded_valid.bool()).float()

    # Slide kernel_size×kernel_size window across every pixel via avg_pool2d.
    local_planar_sum = F.avg_pool2d(zero_curvature_mask, kernel_size=kernel_size, stride=1)
    local_valid_sum = F.avg_pool2d(eroded_valid, kernel_size=kernel_size, stride=1)

    safe_valid_sum = torch.clamp(local_valid_sum, min=1e-6)
    local_density = local_planar_sum / safe_valid_sum

    # Require ≥50% valid data in the window for the statistic to be sound.
    valid_window_mask = local_valid_sum >= 0.5

    if not valid_window_mask.any():
        return False

    return local_density[valid_window_mask].max().item()
