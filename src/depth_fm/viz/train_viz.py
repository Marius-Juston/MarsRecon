"""
Publication-quality visualization for Mars DepthFM.

All figures follow NeurIPS style guidelines:
- 300 DPI minimum
- Seaborn 'flare' palette as default warm colormap
- 'mako' for cool sequential (elevation)
- Readable font sizes (≥8pt for labels, ≥10pt for axes)
- Vector-friendly (PDF/SVG output)

Figure catalogue:
    1. Prediction triptych: input image | predicted DTM | GT DTM
    2. Cross-sectional profiles: 4 directional slices through the heightmap
    3. Error heatmap: per-pixel absolute error with statistics
    4. Flow evolution: intermediate predictions at t = 0, 0.25, 0.5, 0.75, 1.0
    5. Metric distributions: violin/box plots across test set
    6. Multi-run convergence: loss curves with error bands
    7. Worst/best patch gallery: identifying failure and success modes
    8. Normal map comparison: predicted vs GT surface normals
    9. Slope histogram: distribution of slope errors
   10. Scatter plot: predicted vs GT elevation for a representative patch
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from depth_fm.objectives.losses import PhotoclinometricLoss

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

_NEURIPS_RC = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
}


def set_neurips_style():
    """Apply NeurIPS-compatible matplotlib styling."""
    mpl.rcParams.update(_NEURIPS_RC)
    sns.set_theme(style="ticks", rc=_NEURIPS_RC)


# Colormaps
CMAP_ELEVATION = "mako"
CMAP_ERROR = "flare"
CMAP_IMAGE = "gray"
CMAP_NORMALS = "coolwarm"


def _valid_stats(arr: np.ndarray, mask: np.ndarray | None = None):
    """Return (vmin, vmax) at 2-98 percentile over valid pixels."""
    if mask is not None:
        data = arr[mask]
    else:
        data = arr[np.isfinite(arr)]
    if len(data) == 0:
        return 0, 1
    return np.percentile(data, 2), np.percentile(data, 98)


def _apply_mask(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Return a copy of `arr` with invalid pixels set to NaN.

    Accepts a float mask in [0, 1] or a bool mask; threshold at 0.5.
    If mask is None, just returns the input unchanged (but cast to
    float so downstream NaN assignment is safe).
    """
    if mask is None:
        return arr
    out = arr.astype(np.float32, copy=True)
    m = np.asarray(mask)
    if m.dtype != bool:
        m = m > 0.5
    # Broadcast mask to arr's spatial shape if needed
    if m.shape != out.shape[: m.ndim]:
        # Last-resort: if shapes don't align, skip masking rather than crash
        return out
    out[~m] = np.nan
    return out


# ---------------------------------------------------------------------------
# Core Physical Utilities
# ---------------------------------------------------------------------------

def compute_surface_normals(elevation: np.ndarray) -> np.ndarray:
    """Compute surface normals with proper spatial scaling."""
    H, W = elevation.shape
    spatial_scale = max(H, W) / 2.0

    # Calculate gradients
    dy, dx = np.gradient(elevation)

    # Scale gradients to match physical slope dimensions
    n_x = -dx * spatial_scale
    n_y = -dy * spatial_scale
    n_z = np.ones_like(dx)

    n = np.stack([n_x, n_y, n_z], axis=-1)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.clip(norm, 1e-8, None)


# ---------------------------------------------------------------------------
# 1. Prediction triptych
# ---------------------------------------------------------------------------

def plot_prediction_triptych(
        image: np.ndarray,
        pred_dtm: np.ndarray,
        gt_dtm: np.ndarray,
        title: str = "",
        save_path: str | Path | None = None,
        mask: np.ndarray | None = None,
) -> plt.Figure:
    """Three-panel figure: input orthoimage | predicted DTM | GT DTM.

    Args:
        image: (H, W) or (H, W, 3) input orthoimage
        pred_dtm: (H, W) predicted elevation
        gt_dtm: (H, W) ground-truth elevation
        mask: optional (H, W) valid-data mask; invalid pixels shown as NaN
    """
    set_neurips_style()

    pred_dtm = _apply_mask(pred_dtm, mask)
    gt_dtm = _apply_mask(gt_dtm, mask)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Input image
    ax = axes[0]
    if image.ndim == 3 and image.shape[-1] == 3:
        ax.imshow(np.clip(image, 0, 1), interpolation="nearest")
    else:
        ax.imshow(image, cmap=CMAP_IMAGE, interpolation="nearest")
    ax.set_title("Input orthoimage")
    ax.axis("off")

    # Shared elevation limits from GT
    valid = np.isfinite(gt_dtm)
    vmin, vmax = _valid_stats(gt_dtm, valid)

    # Predicted DTM
    ax = axes[1]
    im = ax.imshow(
        np.where(np.isfinite(pred_dtm), pred_dtm, np.nan),
        cmap=CMAP_ELEVATION, vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax.set_title("Predicted DTM")
    ax.axis("off")

    # GT DTM
    ax = axes[2]
    ax.imshow(
        np.where(valid, gt_dtm, np.nan),
        cmap=CMAP_ELEVATION, vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax.set_title("Ground truth DTM")
    ax.axis("off")

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, label="Elevation (m)")

    if title:
        fig.suptitle(title, fontweight="bold")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2. Cross-sectional profiles
# ---------------------------------------------------------------------------

def plot_cross_sections(
        pred_dtm: np.ndarray,
        gt_dtm: np.ndarray,
        title: str = "",
        save_path: str | Path | None = None,
        mask: np.ndarray | None = None,
) -> plt.Figure:
    """Four directional cross-sections through the heightmap centre.

    Directions: horizontal (→), vertical (↓), diagonal NW→SE (↘),
    diagonal NE→SW (↙).
    """
    set_neurips_style()
    pred_dtm = _apply_mask(pred_dtm, mask)
    gt_dtm = _apply_mask(gt_dtm, mask)
    H, W = pred_dtm.shape
    cy, cx = H // 2, W // 2

    palette = sns.color_palette("flare", 3)

    profiles = {
        "Horizontal (→)": (pred_dtm[cy, :], gt_dtm[cy, :]),
        "Vertical (↓)": (pred_dtm[:, cx], gt_dtm[:, cx]),
        "Diagonal NW→SE (↘)": _extract_diagonal(pred_dtm, gt_dtm, direction="nw_se"),
        "Diagonal NE→SW (↙)": _extract_diagonal(pred_dtm, gt_dtm, direction="ne_sw"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for ax, (name, (p_prof, g_prof)) in zip(axes.flat, profiles.items()):
        x = np.arange(len(p_prof))

        ax.plot(x, g_prof, color=palette[0], linewidth=1.5, label="Ground truth", alpha=0.9)
        ax.plot(x, p_prof, color=palette[2], linewidth=1.2, label="Predicted", linestyle="--")

        # Shade the error region
        valid = np.isfinite(p_prof) & np.isfinite(g_prof)
        ax.fill_between(
            x, p_prof, g_prof, where=valid,
            alpha=0.15, color=palette[1], label="Error",
        )

        ax.set_title(name)
        ax.set_xlabel("Pixel position")
        ax.set_ylabel("Elevation (m)")
        ax.legend(loc="best", frameon=True, framealpha=0.9)

    if title:
        fig.suptitle(title, fontweight="bold")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def _extract_diagonal(
        pred: np.ndarray, gt: np.ndarray, direction: str
) -> tuple[np.ndarray, np.ndarray]:
    """Extract diagonal profile."""
    H, W = pred.shape
    length = min(H, W)
    if direction == "nw_se":
        rows = np.linspace(0, H - 1, length).astype(int)
        cols = np.linspace(0, W - 1, length).astype(int)
    else:  # ne_sw
        rows = np.linspace(0, H - 1, length).astype(int)
        cols = np.linspace(W - 1, 0, length).astype(int)
    return pred[rows, cols], gt[rows, cols]


# ---------------------------------------------------------------------------
# 3. Error heatmap
# ---------------------------------------------------------------------------

def plot_error_heatmap(
        pred_dtm: np.ndarray,
        gt_dtm: np.ndarray,
        title: str = "",
        save_path: str | Path | None = None,
        mask: np.ndarray | None = None,
) -> plt.Figure:
    """Per-pixel absolute error map with marginal statistics."""
    set_neurips_style()

    pred_dtm = _apply_mask(pred_dtm, mask)
    gt_dtm = _apply_mask(gt_dtm, mask)

    valid = np.isfinite(pred_dtm) & np.isfinite(gt_dtm)
    error = np.abs(pred_dtm - gt_dtm)
    error[~valid] = np.nan

    fig = plt.figure(figsize=(8, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1], figure=fig)

    # Main heatmap
    ax_main = fig.add_subplot(gs[0, 0])
    vmax = np.nanpercentile(error, 98)
    im = ax_main.imshow(error, cmap=CMAP_ERROR, vmin=0, vmax=vmax, interpolation="nearest")
    ax_main.set_title(title or "Absolute error map")
    ax_main.axis("off")
    fig.colorbar(im, ax=ax_main, shrink=0.8, label="|pred − GT| (m)")

    # Right marginal: row-wise mean error
    ax_right = fig.add_subplot(gs[0, 1], sharey=ax_main)
    row_mean = np.nanmean(error, axis=1)
    ax_right.plot(row_mean, np.arange(len(row_mean)), color=sns.color_palette("flare")[3], linewidth=1)
    ax_right.set_xlabel("Mean |error| (m)")
    ax_right.tick_params(left=False, labelleft=False)

    # Bottom marginal: column-wise mean error
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_main)
    col_mean = np.nanmean(error, axis=0)
    ax_bottom.plot(np.arange(len(col_mean)), col_mean, color=sns.color_palette("flare")[3], linewidth=1)
    ax_bottom.set_ylabel("Mean |error| (m)")
    ax_bottom.tick_params(bottom=False, labelbottom=False)

    # Statistics box
    ax_stats = fig.add_subplot(gs[1, 1])
    ax_stats.axis("off")
    stats_text = (
        f"Mean: {np.nanmean(error):.3f} m\n"
        f"Median: {np.nanmedian(error):.3f} m\n"
        f"Std: {np.nanstd(error):.3f} m\n"
        f"P95: {np.nanpercentile(error, 95):.3f} m"
    )
    ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes,
                  fontsize=9, verticalalignment="center", family="monospace")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 4. Flow evolution
# ---------------------------------------------------------------------------

def plot_flow_evolution(
        intermediates: dict[float, np.ndarray],
        gt_dtm: np.ndarray,
        title: str = "Flow matching evolution",
        save_path: str | Path | None = None,
        mask: np.ndarray | None = None,
) -> plt.Figure:
    """Show predicted DTM at multiple ODE timesteps.

    Args:
        intermediates: dict mapping t → (H, W) predicted depth at that timestep.
            Example: {0.0: z_img, 0.25: ..., 0.5: ..., 0.75: ..., 1.0: z_pred}
        gt_dtm: (H, W) ground-truth
        mask: optional (H, W) valid-data mask
    """
    set_neurips_style()

    gt_dtm = _apply_mask(gt_dtm, mask)
    if mask is not None:
        intermediates = {t: _apply_mask(v, mask) for t, v in intermediates.items()}

    t_values = sorted(intermediates.keys())
    n = len(t_values) + 1  # +1 for GT

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    valid = np.isfinite(gt_dtm)
    vmin, vmax = _valid_stats(gt_dtm, valid)

    for i, t in enumerate(t_values):
        ax = axes[i]
        z = intermediates[t]
        ax.imshow(np.where(np.isfinite(z), z, np.nan),
                  cmap=CMAP_ELEVATION, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"t = {t:.2f}")
        ax.axis("off")

    # GT panel
    ax = axes[-1]
    im = ax.imshow(np.where(valid, gt_dtm, np.nan),
                   cmap=CMAP_ELEVATION, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title("Ground truth")
    ax.axis("off")

    fig.colorbar(im, ax=axes, shrink=0.6, label="Elevation (m)")
    fig.suptitle(title, fontweight="bold")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 5. Metric distributions (violin + box)
# ---------------------------------------------------------------------------

def plot_metric_distributions(
        metrics_df,  # pandas DataFrame from MetricsAggregator.per_sample_dataframe()
        metrics_to_plot: list[str] | None = None,
        title: str = "Test set metric distributions",
        save_path: str | Path | None = None,
) -> plt.Figure:
    """Violin + strip plots for each metric across the test set."""
    set_neurips_style()

    if metrics_to_plot is None:
        metrics_to_plot = ["rmse", "abs_rel", "delta_1", "normal_angular_error"]

    n = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
    if n == 1:
        axes = [axes]

    palette = sns.color_palette("flare", n)

    for ax, metric, color in zip(axes, metrics_to_plot, palette):
        data = metrics_df[metric].dropna()
        parts = ax.violinplot(data, positions=[0], showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.4)
        parts["cmedians"].set_color(color)

        # Overlay strip
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(data))
        ax.scatter(jitter, data, color=color, s=12, alpha=0.5, edgecolors="none")

        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xticks([])

        # Stats annotation
        ax.text(0.95, 0.95,
                f"μ={data.mean():.3f}\nσ={data.std():.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle(title, fontweight="bold")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 6. Multi-run convergence with error bands
# ---------------------------------------------------------------------------

def plot_convergence_curves(
        run_histories: list[dict[str, list[float]]],
        metric_key: str = "val/rmse",
        title: str = "Training convergence",
        save_path: str | Path | None = None,
) -> plt.Figure:
    """Loss/metric curves across multiple runs with mean ± std shading.

    Args:
        run_histories: list of dicts, each mapping metric_key → list of values
            at each validation step.
    """
    set_neurips_style()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = sns.color_palette("flare", 3)

    # Stack runs into (n_runs, n_steps)
    max_len = max(len(h[metric_key]) for h in run_histories if metric_key in h)
    matrix = np.full((len(run_histories), max_len), np.nan)
    for i, h in enumerate(run_histories):
        vals = h.get(metric_key, [])
        matrix[i, :len(vals)] = vals

    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    steps = np.arange(max_len)

    ax.plot(steps, mean, color=palette[0], linewidth=2, label=f"Mean ({len(run_histories)} runs)")
    ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=palette[0])

    # Individual runs as thin lines
    for i in range(len(run_histories)):
        ax.plot(steps, matrix[i], color=palette[2], linewidth=0.5, alpha=0.3)

    ax.set_xlabel("Validation step")
    ax.set_ylabel(metric_key.split("/")[-1].replace("_", " ").upper())
    ax.set_title(title)
    ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 7. Worst/best patch gallery
# ---------------------------------------------------------------------------

def plot_patch_gallery(
        patches: list[dict],
        title: str = "Patch gallery",
        save_path: str | Path | None = None,
) -> plt.Figure:
    """Grid of patches: each row = [image, pred, GT, error].

    Args:
        patches: list of dicts with keys "image", "pred", "gt", "tile_id",
            "rmse", and optionally "mask" (valid-data mask per patch).
    """
    set_neurips_style()
    n = len(patches)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input", "Predicted DTM", "Ground truth", "Absolute error"]
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=10)

    for i, patch in enumerate(patches):
        img = patch["image"]
        pred_raw = patch["pred"]
        gt_raw = patch["gt"]
        tile_id = patch.get("tile_id", "")
        rmse_val = patch.get("rmse", 0)

        # Apply per-patch mask (if present) so nodata regions don't
        # dominate the percentile stretch or the error map.
        pred = _apply_mask(pred_raw, patch.get("mask"))
        gt = _apply_mask(gt_raw, patch.get("mask"))

        valid = np.isfinite(gt) & np.isfinite(pred)
        vmin, vmax = _valid_stats(gt, valid)
        error = np.abs(pred - gt)
        error[~valid] = np.nan

        # Image
        if img.ndim == 3 and img.shape[-1] == 3:
            axes[i, 0].imshow(np.clip(img, 0, 1))
        else:
            axes[i, 0].imshow(img, cmap=CMAP_IMAGE)
        axes[i, 0].set_ylabel(f"{tile_id}\nPhoto Consistency={rmse_val:.2f}m", fontsize=8, rotation=0,
                              labelpad=60, va="center")

        # Predicted
        axes[i, 1].imshow(np.where(np.isfinite(pred), pred, np.nan),
                          cmap=CMAP_ELEVATION, vmin=vmin, vmax=vmax)

        # GT
        axes[i, 2].imshow(np.where(valid, gt, np.nan),
                          cmap=CMAP_ELEVATION, vmin=vmin, vmax=vmax)

        # Error
        emax = np.nanpercentile(error, 98) if np.any(np.isfinite(error)) else 1
        axes[i, 3].imshow(error, cmap=CMAP_ERROR, vmin=0, vmax=emax)

        for j in range(4):
            axes[i, j].axis("off")

    fig.suptitle(title, fontweight="bold")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 8. Normal map comparison
# ---------------------------------------------------------------------------

def plot_normal_maps(
        pred_dtm: np.ndarray,
        gt_dtm: np.ndarray,
        title: str = "",
        save_path: str | Path | None = None,
        mask: np.ndarray | None = None,
) -> plt.Figure:
    """Side-by-side surface normal maps (RGB-encoded) with spatial scaling."""
    set_neurips_style()

    # Fill invalid pixels with the local mean BEFORE computing gradients,
    # so we don't cause huge synthetic gradients at nodata boundaries,
    # then mask them out for display.
    if mask is not None:
        m = np.asarray(mask)
        if m.dtype != bool:
            m = m > 0.5
        p_fill = np.where(m, pred_dtm, np.nanmean(pred_dtm[m]) if m.any() else 0.0)
        g_fill = np.where(m, gt_dtm, np.nanmean(gt_dtm[m]) if m.any() else 0.0)
    else:
        m = None
        p_fill, g_fill = pred_dtm, gt_dtm

    def _normals_rgb(z):
        normals = compute_surface_normals(z)
        return (normals + 1.0) / 2.0  # map [-1,1] → [0,1] for RGB display

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    pred_rgb = np.clip(_normals_rgb(p_fill), 0, 1)
    gt_rgb = np.clip(_normals_rgb(g_fill), 0, 1)
    if m is not None:
        pred_rgb = np.where(m[..., None], pred_rgb, np.nan)
        gt_rgb = np.where(m[..., None], gt_rgb, np.nan)

    axes[0].imshow(pred_rgb)
    axes[0].set_title("Predicted normals")
    axes[0].axis("off")

    axes[1].imshow(gt_rgb)
    axes[1].set_title("GT normals")
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontweight="bold")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 9. Predicted vs GT scatter
# ---------------------------------------------------------------------------

def plot_elevation_scatter(
        pred_dtm: np.ndarray,
        gt_dtm: np.ndarray,
        title: str = "",
        save_path: str | Path | None = None,
        subsample: int = 5000,
        mask: np.ndarray | None = None,
) -> plt.Figure:
    """Scatter plot of predicted vs GT elevation with density coloring."""
    set_neurips_style()

    valid = np.isfinite(pred_dtm) & np.isfinite(gt_dtm)
    if mask is not None:
        m = np.asarray(mask)
        if m.dtype != bool:
            m = m > 0.5
        valid = valid & m
    p = pred_dtm[valid]
    g = gt_dtm[valid]

    # Subsample for performance
    if len(p) > subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(p), subsample, replace=False)
        p, g = p[idx], g[idx]

    fig, ax = plt.subplots(figsize=(5.5, 5))

    # 2D histogram for density
    ax.hexbin(g, p, gridsize=50, cmap="flare", mincnt=1)
    lims = [min(g.min(), p.min()), max(g.max(), p.max())]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Perfect prediction")
    ax.set_xlabel("GT elevation (m)")
    ax.set_ylabel("Predicted elevation (m)")
    ax.set_aspect("equal")
    ax.legend()

    # Correlation coefficient
    r = np.corrcoef(g, p)[0, 1]
    ax.text(0.05, 0.95, f"R² = {r ** 2:.4f}", transform=ax.transAxes,
            fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    if title:
        ax.set_title(title)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 10. Multi-run summary table (for paper)
# ---------------------------------------------------------------------------

def plot_multi_run_summary_table(
        run_summaries: list[dict[str, dict[str, float]]],
        metrics_to_show: list[str] | None = None,
        title: str = "Multi-run test set results",
        save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart with error bars showing mean ± std across runs."""
    set_neurips_style()

    if metrics_to_show is None:
        metrics_to_show = ["rmse", "abs_rel", "delta_1", "normal_angular_error"]

    import pandas as pd

    # Build per-run per-metric means
    rows = []
    for i, summary in enumerate(run_summaries):
        row = {"run": i}
        for m in metrics_to_show:
            if m in summary:
                row[m] = summary[m]["mean"]
        rows.append(row)

    df = pd.DataFrame(rows)

    n = len(metrics_to_show)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
    if n == 1:
        axes = [axes]

    palette = sns.color_palette("flare", n)

    for ax, metric, color in zip(axes, metrics_to_show, palette):
        vals = df[metric].dropna()
        mean = vals.mean()
        std = vals.std()

        ax.bar(0, mean, yerr=std, color=color, alpha=0.7, capsize=8, width=0.5)
        # Individual run dots
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(jitter, vals, color="black", s=20, zorder=5)

        label = metric.replace("_", " ").title()
        ax.set_ylabel(label)
        ax.set_xticks([])
        ax.set_title(f"{mean:.3f} ± {std:.3f}", fontsize=9)

    fig.suptitle(title, fontweight="bold")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 11. Hillshade comparison
# ---------------------------------------------------------------------------

def compute_hillshade(
        elevation: np.ndarray,
        azimuth_deg: float = 315.0,
        altitude_deg: float = 45.0,
        z_factor: float = 1.0,
) -> np.ndarray:
    """Compute an analytical hillshade from an elevation grid.

    Uses the standard ESRI/GDAL algorithm:
        shade = cos(zenith) * cos(slope) +
                sin(zenith) * sin(slope) * cos(azimuth - aspect)

    Args:
        elevation: (H, W) float32 elevation in metres.
        azimuth_deg: Solar azimuth (compass bearing of the sun), degrees.
            315° = NW illumination (standard planetary science convention).
        altitude_deg: Solar altitude above the horizon, degrees.
            45° is typical for planetary DTM display.
        z_factor: Vertical exaggeration factor.  Values > 1 emphasise relief.

    Returns:
        (H, W) float32 in [0, 1] where 1 = fully illuminated.
    """
    azimuth = np.radians(360.0 - azimuth_deg + 90.0)  # ESRI convention
    zenith = np.radians(90.0 - altitude_deg)

    dy, dx = np.gradient(elevation * z_factor)
    slope = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    aspect = np.arctan2(-dy, dx)

    shade = (
            np.cos(zenith) * np.cos(slope)
            + np.sin(zenith) * np.sin(slope) * np.cos(azimuth - aspect)
    )
    return np.clip(shade, 0.0, 1.0).astype(np.float32)


def plot_hillshade_comparison(
        pred_dtm: np.ndarray,
        gt_dtm: np.ndarray,
        azimuth: float = 315.0,
        altitude: float = 45.0,
        z_factor: float = 2.0,
        title: str = "",
        save_path: str | Path | None = None,
) -> plt.Figure:
    """Side-by-side hillshade rendering of predicted vs GT DTMs.

    This is the standard planetary science visualisation for DTMs.
    Synthetic solar illumination reveals fine-scale terrain features
    (crater rims, ridges, dune textures) that are invisible in
    elevation-coloured maps.

    A third panel shows the **hillshade difference**, which highlights
    exactly where the predicted surface normals diverge from ground truth:
    bright spots = predicted slope faces the sun more; dark = less.

    Args:
        pred_dtm: (H, W) predicted elevation (will be affine-aligned to GT).
        gt_dtm: (H, W) ground-truth elevation.
        azimuth: Solar azimuth in degrees (315 = NW, standard for Mars).
        altitude: Solar altitude in degrees.
        z_factor: Vertical exaggeration (2.0 emphasises subtle terrain).
    """
    set_neurips_style()

    from depth_fm.objectives.metrics import affine_align
    pred_aligned, _, _ = affine_align(pred_dtm, gt_dtm)

    # Fill NaN for gradient computation
    pred_filled = np.nan_to_num(pred_aligned, nan=np.nanmean(pred_aligned))
    gt_filled = np.nan_to_num(gt_dtm, nan=np.nanmean(gt_dtm))

    hs_pred = compute_hillshade(pred_filled, azimuth, altitude, z_factor)
    hs_gt = compute_hillshade(gt_filled, azimuth, altitude, z_factor)

    # Difference: >0.5 = predicted brighter, <0.5 = GT brighter
    hs_diff = (hs_pred - hs_gt + 1.0) / 2.0

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    axes[0].imshow(hs_pred, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("Predicted hillshade")
    axes[0].axis("off")

    axes[1].imshow(hs_gt, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("Ground truth hillshade")
    axes[1].axis("off")

    im = axes[2].imshow(hs_diff, cmap="RdBu_r", vmin=0.3, vmax=0.7, interpolation="nearest")
    axes[2].set_title("Hillshade difference")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], shrink=0.8, label="Pred brighter ← → GT brighter")

    fig.text(
        0.5, 0.01,
        f"Solar azimuth {azimuth}°, altitude {altitude}°, z-factor {z_factor}×",
        ha="center", fontsize=8, style="italic", color="gray",
    )

    if title:
        fig.suptitle(title, fontweight="bold")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


@torch.no_grad()
def plot_lunar_lambert_comparison(
        pred_dtm: Union[torch.Tensor, np.ndarray],
        gt_dtm: Union[torch.Tensor, np.ndarray],
        real_ortho: Union[torch.Tensor, np.ndarray],
        sun_vector: Union[torch.Tensor, np.ndarray],
        intensity: Union[torch.Tensor, np.ndarray, float],
        ambient: Union[torch.Tensor, np.ndarray, float],
        loss_fn: "PhotoclinometricLoss",
        valid_mask: Union[torch.Tensor, np.ndarray, None] = None,
        title: str = "",
        save_path: Union[str, Path, None] = None,
) -> plt.Figure:
    """
    Type-Agnostic PyTorch Lunar-Lambert diagnostic.
    Seamlessly handles both NumPy arrays and PyTorch tensors, automatically
    expanding shapes to (B, C, H, W) for the loss function's internal math.
    """

    # Extract the device the loss function is currently sitting on
    try:
        device = loss_fn.lunar_lambert_logit.device
    except AttributeError:
        device = torch.device("cpu")

    def _ensure_tensor(x, is_spatial=False) -> torch.Tensor | None:
        """Casts NumPy/Scalars to Tensors and enforces (B, C, H, W) dimensions."""
        if x is None:
            return None

        # 1. Cast to PyTorch FloatTensor
        if not isinstance(x, torch.Tensor):
            if isinstance(x, (float, int)):
                t = torch.tensor([x], dtype=torch.float32)
            else:
                t = torch.from_numpy(np.array(x)).float()
        else:
            t = x.float()  # Ensure float32 for safety

        # 2. Fix Spatial Dimensions -> strictly (B, C, H, W)
        if is_spatial:
            # Catch standard image arrays (H, W, C) where C is 1, 3, or 4
            if t.dim() == 3 and t.shape[-1] in [1, 3, 4]:
                t = t.permute(2, 0, 1)  # -> (C, H, W)

            # Add missing Batch or Channel dims
            if t.dim() == 2:  # (H, W) -> (1, 1, H, W)
                t = t.unsqueeze(0).unsqueeze(0)
            elif t.dim() == 3:  # (C, H, W) -> (1, C, H, W)
                t = t.unsqueeze(0)

            # Catch stray (B, H, W, C) if it somehow slipped through
            if t.dim() == 4 and t.shape[-1] in [1, 3, 4] and t.shape[1] not in [1, 3, 4]:
                t = t.permute(0, 3, 1, 2)

        # 3. Fix Parameter Dimensions
        else:
            if t.dim() == 1 and t.shape[0] == 3:  # Sun vector (3,) -> (1, 3)
                t = t.unsqueeze(0)
            elif t.dim() == 0:  # Scalar -> (1,)
                t = t.unsqueeze(0)

        return t.to(device)

    # --- Cast all inputs to standardized Tensors ---
    pred_dtm_t = _ensure_tensor(pred_dtm, is_spatial=True)
    gt_dtm_t = _ensure_tensor(gt_dtm, is_spatial=True)
    real_ortho_t = _ensure_tensor(real_ortho, is_spatial=True)
    valid_mask_t = _ensure_tensor(valid_mask, is_spatial=True)

    sun_vector_t = _ensure_tensor(sun_vector, is_spatial=False)
    intensity_t = _ensure_tensor(intensity, is_spatial=False)
    ambient_t = _ensure_tensor(ambient, is_spatial=False)

    # Fallback mask if none provided
    if valid_mask_t is None:
        valid_mask_t = torch.ones_like(gt_dtm_t)

    # Scrub NaN/Inf from the depth tensors before passing them into the
    # loss function. The internal _zscore does `x * valid_mask`, and
    # NaN * 0 = NaN in IEEE 754, so a single nodata pixel would poison
    # the whole render via the per-image mean. We replace NaN in
    # invalid regions with 0 — the mask will exclude them from stats.
    pred_dtm_t = torch.nan_to_num(pred_dtm_t, nan=0.0, posinf=0.0, neginf=0.0)
    gt_dtm_t = torch.nan_to_num(gt_dtm_t, nan=0.0, posinf=0.0, neginf=0.0)
    real_ortho_t = torch.nan_to_num(real_ortho_t, nan=0.0, posinf=0.0, neginf=0.0)

    mask_bool = (valid_mask_t > 0.5).squeeze()

    # 1. Convert real ortho to grayscale for SSIM comparison
    if real_ortho_t.shape[1] == 3:
        ortho_gray = real_ortho_t.mean(dim=1, keepdim=True)
    else:
        ortho_gray = real_ortho_t

    # 2. Fast GPU-Native Affine Alignment for the DTM
    p_valid = pred_dtm_t[valid_mask_t > 0.5]
    g_valid = gt_dtm_t[valid_mask_t > 0.5]
    if p_valid.numel() > 10:
        A = torch.stack([p_valid, torch.ones_like(p_valid)], dim=1)
        res = torch.linalg.lstsq(A, g_valid).solution
        pred_aligned = (pred_dtm_t * res[0]) + res[1]
    else:
        pred_aligned = pred_dtm_t

    # 3. Render using the exact loss function math
    render_pred, _ = loss_fn.render_from_depth(pred_aligned, sun_vector_t, intensity_t, ambient_t)
    render_gt, _ = loss_fn.render_from_depth(gt_dtm_t, sun_vector_t, intensity_t, ambient_t)

    # 4. Compute Z-Scores (What the SSIM actually minimizes)
    z_pred = loss_fn._zscore(render_pred, valid_mask_t)
    z_ortho = loss_fn._zscore(ortho_gray, valid_mask_t)

    # --- Move to CPU for Matplotlib ---
    def prep_for_display(
            x: torch.Tensor,
            mask: np.ndarray | None = None,
            pct_low: float = 2.0,
            pct_high: float = 98.0,
            normalize_vis: bool = False
    ) -> np.ndarray:
        """Percentile-stretch a (1,1,H,W) or (1,H,W) tensor to [0,1] for display."""
        arr = x.detach().cpu().numpy().squeeze()
        # The render is unclamped and can contain NaN/Inf from degenerate normals
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if mask is not None:
            valid = arr[mask]
            if valid.size > 0:
                lo, hi = np.percentile(valid, [pct_low, pct_high])
            else:
                lo, hi = float(arr.min()), float(arr.max())
        else:
            lo, hi = np.percentile(arr, [pct_low, pct_high])

        if hi - lo < 1e-6:
            hi = lo + 1e-6
        if normalize_vis:
            arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

        if mask is not None:
            arr[~mask] = np.nan
        return arr

    mask_bool_n = mask_bool.numpy()

    r_ortho_np = prep_for_display(ortho_gray, mask=mask_bool_n, normalize_vis=True)
    r_pred_np = prep_for_display(render_pred, mask=mask_bool_n, normalize_vis=True)
    r_gt_np = prep_for_display(render_gt, mask=mask_bool_n, normalize_vis=True)
    z_pred_np = prep_for_display(z_pred, mask=mask_bool_n, normalize_vis=False)
    z_ortho_np = prep_for_display(z_ortho, mask=mask_bool_n, normalize_vis=False)

    # 5. The TRUE Structural Error Map the network feels
    loss_map = np.abs(z_pred_np - z_ortho_np)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    L_weight = loss_fn.lunar_lambert_weight.item()

    # Panel 1: Real Ortho (The Target)
    axes[0].imshow(r_ortho_np, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("Real Ortho (Target)")

    # Panel 2: Predicted Render
    axes[1].imshow(r_pred_np, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(f"Predicted Render (L={L_weight:.2f})")

    # Panel 3: GT Render (Pure Topography Ideal)
    axes[2].imshow(r_gt_np, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title(f"GT Render (L={L_weight:.2f})")

    # Panel 4: True SSIM Loss Map (Pred vs Ortho)
    im4 = axes[3].imshow(loss_map, cmap="inferno", vmin=0, vmax=2.0, interpolation="nearest")
    axes[3].set_title("Actual Loss Penalty ($|Z_{pred} - Z_{ortho}|$)")
    fig.colorbar(im4, ax=axes[3], shrink=0.8, label="High SSIM Penalty")

    for ax in axes:
        ax.axis("off")

    sv = sun_vector_t.squeeze().cpu().numpy()
    footer_text = (
        f"Sun: [{sv[0]:.2f}, {sv[1]:.2f}, {sv[2]:.2f}] | "
        f"Intensity: {intensity_t.item():.2f} | Ambient: {ambient_t.item():.2f} | LL-Weight: {L_weight:.3f}"
    )
    fig.text(0.5, 0.01, footer_text, ha="center", fontsize=10, style="italic", color="gray")

    if title:
        fig.suptitle(title, fontweight="bold", fontsize=14)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150, facecolor="white")

    return fig


# ---------------------------------------------------------------------------
# 12. Timestep-conditioned error curve (ablation)
# ---------------------------------------------------------------------------

def plot_timestep_ablation(
        step_counts: list[int],
        metrics_per_step: dict[int, dict[str, dict[str, float]]],
        primary_metric: str = "rmse",
        secondary_metrics: list[str] | None = None,
        title: str = "Inference quality vs Euler steps",
        save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot RMSE (and other metrics) as a function of the number of Euler ODE steps.

    DepthFM's key claim is that 1-step inference is nearly as good as
    multi-step.  This figure tests whether that holds on Mars terrain,
    which has higher-frequency detail than the Earth scenes DepthFM was
    evaluated on.

    Args:
        step_counts: sorted list of step counts evaluated, e.g. [1, 2, 4, 8, 10, 20].
        metrics_per_step: dict mapping step_count → MetricsAggregator.summary().
            Each summary is a dict of metric_name → {"mean": ..., "std": ...}.
        primary_metric: metric plotted on the left y-axis (default "rmse").
        secondary_metrics: metrics plotted on the right y-axis.
    """
    set_neurips_style()

    if secondary_metrics is None:
        secondary_metrics = ["delta_1"]

    palette = sns.color_palette("flare", 2 + len(secondary_metrics))

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    # Primary metric (left axis)
    means = [metrics_per_step[s][primary_metric]["mean"] for s in step_counts]
    stds = [metrics_per_step[s][primary_metric]["std"] for s in step_counts]

    ax1.errorbar(
        step_counts, means, yerr=stds,
        color=palette[0], marker="o", markersize=6,
        linewidth=2, capsize=5, capthick=1.5,
        label=primary_metric.upper(),
    )
    ax1.set_xlabel("Number of Euler steps")
    ax1.set_ylabel(f"{primary_metric.upper()} (m)", color=palette[0])
    ax1.tick_params(axis="y", labelcolor=palette[0])
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(step_counts)
    ax1.set_xticklabels([str(s) for s in step_counts])

    # Annotate 1-step degradation vs best
    best_idx = int(np.argmin(means))
    one_step_val = means[0]
    best_val = means[best_idx]
    if one_step_val > 0 and best_val > 0:
        degradation_pct = ((one_step_val - best_val) / best_val) * 100
        ax1.annotate(
            f"1-step: +{degradation_pct:.1f}% vs best",
            xy=(step_counts[0], one_step_val),
            xytext=(step_counts[0] * 1.8, one_step_val * 1.05),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray"),
            color="gray",
        )

    # Secondary metrics (right axis)
    if secondary_metrics:
        ax2 = ax1.twinx()
        for j, sec_metric in enumerate(secondary_metrics):
            sec_means = [metrics_per_step[s][sec_metric]["mean"] for s in step_counts]
            sec_stds = [metrics_per_step[s][sec_metric]["std"] for s in step_counts]
            ax2.errorbar(
                step_counts, sec_means, yerr=sec_stds,
                color=palette[1 + j], marker="s", markersize=5,
                linewidth=1.5, linestyle="--", capsize=4,
                label=sec_metric.replace("_", " ").title(),
            )
        ax2.set_ylabel(
            ", ".join(m.replace("_", " ").title() for m in secondary_metrics),
            color=palette[1],
        )
        ax2.tick_params(axis="y", labelcolor=palette[1])

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    else:
        ax1.legend()

    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_uncertainty_map(
        pred_dtm: np.ndarray,
        var_dtm: np.ndarray,
        img: np.ndarray,
        title: str = "Epistemic Uncertainty",
        save_path: str | Path | None = None,
        mask: np.ndarray | None = None,
) -> plt.Figure:
    """Triptych: Input Image | Predicted DTM | Epistemic Variance (Uncertainty)."""
    set_neurips_style()

    pred_dtm = _apply_mask(pred_dtm, mask)
    var_dtm = _apply_mask(var_dtm, mask)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel 1: Input image
    if img.ndim == 3 and img.shape[-1] == 3:
        axes[0].imshow(np.clip(img, 0, 1), interpolation="nearest")
    else:
        axes[0].imshow(img, cmap=CMAP_IMAGE, interpolation="nearest")
    axes[0].set_title("Input Orthoimage")
    axes[0].axis("off")

    # Panel 2: Mean Prediction
    valid = np.isfinite(pred_dtm)
    vmin, vmax = _valid_stats(pred_dtm, valid)
    im_pred = axes[1].imshow(
        np.where(valid, pred_dtm, np.nan),
        cmap=CMAP_ELEVATION, vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    axes[1].set_title("Expected Topography")
    axes[1].axis("off")

    # Panel 3: Variance (Uncertainty)
    # Using 'inferno' to highlight highly uncertain regions (yellow/white)
    var_vmax = np.nanpercentile(var_dtm, 98)
    im_var = axes[2].imshow(
        var_dtm, cmap="inferno", vmin=0, vmax=var_vmax, interpolation="nearest"
    )
    axes[2].set_title(r"Epistemic Variance ($\sigma^2$)")
    axes[2].axis("off")

    # Colorbars
    cbar1 = fig.colorbar(im_pred, ax=axes[1], shrink=0.8, label="Elevation (m)")
    cbar2 = fig.colorbar(im_var, ax=axes[2], shrink=0.8, label="Variance (m$^2$)")

    if title:
        fig.suptitle(title, fontweight="bold")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_geomorphometric_analysis(
        pred_dtm: np.ndarray,
        gt_dtm: np.ndarray,
        title: str = "Geomorphometric Analysis: Edges & Curvature",
        save_path: str | Path | None = None,
        mask: np.ndarray | None = None,
) -> plt.Figure:
    """Triptych: Edge Boundaries | Predicted Laplacian | Error in Laplacian."""
    from scipy.ndimage import gaussian_filter
    set_neurips_style()

    # Fill invalid pixels before derivatives to avoid synthetic edges at
    # the nodata boundary, then mask for display.
    if mask is not None:
        m = np.asarray(mask)
        if m.dtype != bool:
            m = m > 0.5
        p_fill = np.where(m, pred_dtm, np.nanmean(pred_dtm[m]) if m.any() else 0.0)
        g_fill = np.where(m, gt_dtm, np.nanmean(gt_dtm[m]) if m.any() else 0.0)
    else:
        m = None
        p_fill, g_fill = pred_dtm, gt_dtm

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # 1. Edge Boundaries Plot (DBF Visualization)
    dy_p, dx_p = np.gradient(p_fill)
    dy_g, dx_g = np.gradient(g_fill)
    grad_p = np.sqrt(dx_p ** 2 + dy_p ** 2)
    grad_g = np.sqrt(dx_g ** 2 + dy_g ** 2)

    thresh = np.nanpercentile(grad_g, 85)
    # Map edges: GT = Blue, Pred = Red, Overlap = Purple
    edge_viz = np.ones((*pred_dtm.shape, 3))
    edge_viz[grad_g > thresh] = [0.2, 0.4, 0.8]  # GT edges (Blue)
    edge_viz[grad_p > thresh] = [0.8, 0.2, 0.2]  # Pred edges (Red)
    edge_viz[(grad_g > thresh) & (grad_p > thresh)] = [0.6, 0.1, 0.6]  # Overlap (Purple)
    if m is not None:
        edge_viz[~m] = np.nan

    axes[0].imshow(edge_viz, interpolation="nearest")
    axes[0].set_title("Structural Edges\n(Blue: GT, Red: Pred, Purple: Match)")
    axes[0].axis("off")

    # 2. Laplacian (Curvature)
    pred_s = gaussian_filter(p_fill, sigma=1.0)
    d2y, _ = np.gradient(np.gradient(pred_s)[0])
    _, d2x = np.gradient(np.gradient(pred_s)[1])
    laplacian_p = d2x + d2y
    if m is not None:
        laplacian_p = np.where(m, laplacian_p, np.nan)

    v_lim = np.nanpercentile(np.abs(laplacian_p), 98)
    im_lap = axes[1].imshow(
        laplacian_p, cmap="coolwarm", vmin=-v_lim, vmax=v_lim, interpolation="nearest"
    )
    axes[1].set_title("Predicted Curvature (Laplacian)")
    axes[1].axis("off")
    fig.colorbar(im_lap, ax=axes[1], shrink=0.8, label=r"$\nabla^2 z$")

    # 3. Laplacian Error (TIN Artifact / Terracing Identifier)
    gt_s = gaussian_filter(g_fill, sigma=1.0)
    d2y_g, _ = np.gradient(np.gradient(gt_s)[0])
    _, d2x_g = np.gradient(np.gradient(gt_s)[1])
    laplacian_g = d2x_g + d2y_g

    lap_error = np.abs(laplacian_p - laplacian_g)
    err_lim = np.nanpercentile(lap_error, 95)

    im_err = axes[2].imshow(
        lap_error, cmap="magma", vmin=0, vmax=err_lim, interpolation="nearest"
    )
    axes[2].set_title("Curvature Absolute Error")
    axes[2].axis("off")
    fig.colorbar(im_err, ax=axes[2], shrink=0.8, label=r"$|\Delta \nabla^2 z|$")

    if title:
        fig.suptitle(title, fontweight="bold")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_radial_psd_curves(
        pred_dtm: np.ndarray,
        gt_dtm: np.ndarray,
        valid_mask: np.ndarray | None = None,
        title: str = "Radial Power Spectral Density",
        save_path: str | Path | None = None,
        mask: np.ndarray | None = None,
) -> plt.Figure:
    """Log-log plot of radial PSD comparing synthetic vs true terrain frequencies."""
    set_neurips_style()

    # Accept either `mask` (new, consistent name) or `valid_mask` (legacy)
    if valid_mask is None and mask is not None:
        m = np.asarray(mask)
        if m.dtype != bool:
            m = m > 0.5
        valid_mask = m

    if valid_mask is None:
        valid_mask = np.isfinite(pred_dtm) & np.isfinite(gt_dtm)

    # Impute missing values with mean to avoid FFT artifacts
    p_fill = np.where(valid_mask, pred_dtm, np.nanmean(pred_dtm))
    g_fill = np.where(valid_mask, gt_dtm, np.nanmean(gt_dtm))

    def _get_radial_psd(z):
        H, W = z.shape
        z_detrend = z - np.mean(z)  # Remove DC offset
        fft = np.fft.fft2(z_detrend)
        psd_2d = np.abs(fft) ** 2 / (H * W)
        psd_2d = np.fft.fftshift(psd_2d)

        cy, cx = H // 2, W // 2
        y, x = np.ogrid[-cy:H - cy, -cx:W - cx]
        r = np.sqrt(x ** 2 + y ** 2).astype(int)

        radial = np.bincount(r.ravel(), psd_2d.ravel())
        counts = np.bincount(r.ravel())
        valid_bins = counts > 0
        radial[valid_bins] /= counts[valid_bins]
        return radial

    psd_pred = _get_radial_psd(p_fill)
    psd_gt = _get_radial_psd(g_fill)

    # Truncate to the Nyquist limit (min of half-dimensions)
    max_freq = min(pred_dtm.shape[0] // 2, pred_dtm.shape[1] // 2)
    freqs = np.arange(1, max_freq)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Log-Log plotting
    ax.loglog(freqs, psd_gt[1:max_freq], label="Ground Truth", color="black", linewidth=2)
    ax.loglog(freqs, psd_pred[1:max_freq], label="Predicted (Flow-Matched)",
              color=sns.color_palette("flare")[2], linewidth=1.5, linestyle="--")

    # Reference slope typical for Mars (-2.5 to -3.0 power law)
    ref_y = psd_gt[1] * (freqs / freqs[0]) ** -2.5
    ax.loglog(freqs, ref_y, label="Reference $f^{-2.5}$", color="gray", linestyle=":", alpha=0.7)

    ax.set_xlabel("Spatial Frequency (cycles/pixel)")
    ax.set_ylabel("Power Density")
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_pareto_frontier(
        metrics_per_step: dict[int, dict[str, dict[str, float]]],
        primary_metric: str = "rmse",
        title: str = "Compute vs. Accuracy Pareto Frontier",
        save_path: str | Path | None = None,
) -> plt.Figure:
    """Plots Error against Number of Function Evaluations (NFE)."""
    set_neurips_style()

    step_counts = sorted(metrics_per_step.keys())
    means = [metrics_per_step[s][primary_metric]["mean"] for s in step_counts]
    stds = [metrics_per_step[s][primary_metric]["std"] for s in step_counts]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    palette = sns.color_palette("flare", 3)

    # Plot the curve
    ax.plot(step_counts, means, marker="o", color=palette[1], linewidth=2, label="Mars DepthFM")
    ax.fill_between(step_counts, np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds), alpha=0.2, color=palette[1])

    # Annotate Pareto efficiency points
    ax.annotate("Optimal 1-Step\nZero-Shot",
                xy=(step_counts[0], means[0]),
                xytext=(step_counts[0] + 2, means[0] + (max(means) - min(means)) * 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    ax.set_xlabel("Compute Overhead (Number of Function Evaluations / NFE)")
    ax.set_ylabel(f"{primary_metric.upper()} Error (Lower is Better)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(step_counts)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_slope_error_map(
        pred_dtm: np.ndarray,
        gt_dtm: np.ndarray,
        img: np.ndarray,
        title: str = "Geomorphometric DoD (Slope Error)",
        save_path: str | Path | None = None,
        mask: np.ndarray | None = None,
) -> plt.Figure:
    """Triptych: Ortho | GT Slope | Absolute Slope Error."""
    set_neurips_style()

    # Fill invalid pixels before gradient, then mask display. This
    # prevents the nodata boundary from producing a spurious slope
    # spike that would dominate the 98th percentile.
    if mask is not None:
        m = np.asarray(mask)
        if m.dtype != bool:
            m = m > 0.5
        p_fill = np.where(m, pred_dtm, np.nanmean(pred_dtm[m]) if m.any() else 0.0)
        g_fill = np.where(m, gt_dtm, np.nanmean(gt_dtm[m]) if m.any() else 0.0)
    else:
        m = None
        p_fill, g_fill = pred_dtm, gt_dtm

    def _compute_slope(z):
        dy, dx = np.gradient(z)
        return np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))

    slope_pred = _compute_slope(p_fill)
    slope_gt = _compute_slope(g_fill)
    slope_error = np.abs(slope_pred - slope_gt)

    if m is not None:
        slope_gt = np.where(m, slope_gt, np.nan)
        slope_error = np.where(m, slope_error, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Input
    if img.ndim == 3 and img.shape[-1] == 3:
        axes[0].imshow(np.clip(img, 0, 1), interpolation="nearest")
    else:
        axes[0].imshow(img, cmap=CMAP_IMAGE, interpolation="nearest")
    axes[0].set_title("Input Orthoimage")
    axes[0].axis("off")

    # GT Slope
    vmax_s = np.nanpercentile(slope_gt, 98)
    im_gt = axes[1].imshow(slope_gt, cmap="viridis", vmin=0, vmax=vmax_s)
    axes[1].set_title(r"Ground Truth Slope ($\theta^\circ$)")
    axes[1].axis("off")
    fig.colorbar(im_gt, ax=axes[1], shrink=0.8, label="Degrees")

    # Slope Error
    vmax_e = np.nanpercentile(slope_error, 95)
    im_err = axes[2].imshow(slope_error, cmap="magma", vmin=0, vmax=vmax_e)
    axes[2].set_title(r"Absolute Slope Error ($|\Delta\theta|^\circ$)")
    axes[2].axis("off")
    fig.colorbar(im_err, ax=axes[2], shrink=0.8, label="Error Degrees")

    if title:
        fig.suptitle(title, fontweight="bold")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
