
import argparse
import concurrent.futures
import hashlib
import json
import logging
import math
import os
from copy import deepcopy
from typing import Callable, Optional, Any

import cuml
import matplotlib.patches as mpatches
import xgboost as xgb
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch

from depth_fm.litdata_datamodule import _build_litdata_loaders
from depth_fm.losses import PhotoclinometricLoss, AbsoluteDepthLoss, LaplacianLoss, \
    OrdinalRankingLoss

# ---------------------------------------------------------------------------
# GLOBAL GDAL/IO OPTIMIZATIONS (For 256-Core / NVMe setups)
# ---------------------------------------------------------------------------
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["VSI_CACHE"] = "TRUE"
os.environ["VSI_CACHE_SIZE"] = "500000000"
os.environ["GDAL_CACHEMAX"] = "10%"
os.environ["GDAL_MAX_DATASET_POOL_SIZE"] = "1024"

import lightning as L
import torch.fft
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from depth_fm.depthfm_adapter import (
    DepthFMHiRISEAdapterCached, fill_voids_gmrf, estimate_sun_vector_irls, SeamResult, detect_seam_artifact)
from depth_fm.lightning_module import DepthFMLightningModule, FasterEMAWeightAveraging
from src.depth_fm.viz.train_viz import (
    plot_convergence_curves,
    plot_metric_distributions,
    plot_multi_run_summary_table,
    set_neurips_style, plot_pareto_frontier,
)

import matplotlib.gridspec as gridspec
from scipy.spatial.transform import Rotation as R

from depth_fm.depthfm_adapter import compute_topographic_residual
from depth_fm.scalers import GlobalLogNormalizer, DEFAULT_ELEV_REF_SCALE
import torch.distributed as dist
import re
from src.depth_fm.viz.train_viz import plot_timestep_ablation
import pandas as pd
from tqdm import tqdm
from torch.func import vmap, grad, hessian
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import geoopt
from torch import nn


logger = logging.getLogger(__name__)


DPI = 300

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


# ---------------------------------------------------------------------------
# Visualization helpers (unchanged from original)
# ---------------------------------------------------------------------------

def save_fig(fig: Figure, path: Path, formats: tuple[str, ...] = (".png", ".pdf"), **kwargs):
    for f in formats:
        new_path = path.with_suffix(f)
        fig.savefig(new_path, **kwargs)
        logger.info(f"Saved {new_path}")


def _display_dtm(arr, mask=None) -> np.ndarray:
    """Percentile-stretch a DTM array to [0,1] for imshow.

    Handles clip=False where log-compressed values can exceed ±1.
    """
    if hasattr(arr, 'cpu'):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32).squeeze()
    src = arr[mask] if (mask is not None and mask.any()) else arr[np.isfinite(arr)]
    if src.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.nanpercentile(src, 2), np.nanpercentile(src, 98)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    if mask is not None:
        out[~mask] = np.nan
    return out


def _make_synthetic_pred(
        gt: torch.Tensor, noise_level: float = 0.08, seed: int = 42
) -> torch.Tensor:
    """Plausible fake prediction for standalone viz (no trained model needed).

    Generates low-frequency noise + a small global bias so the resulting
    'prediction' has realistic errors (not random high-frequency noise).
    If the caller passes a trained predictor via `predictor_fn`, this is
    never used.
    """
    B, C, H, W = gt.shape
    g = torch.Generator(device=gt.device).manual_seed(seed)
    noise = torch.randn(B, C, max(H // 8, 1), max(W // 8, 1),
                        generator=g, device=gt.device)
    noise = F.interpolate(noise, size=(H, W), mode="bicubic", align_corners=False)
    bias = torch.randn(B, C, 1, 1, generator=g, device=gt.device) * noise_level * 0.3
    lo, hi = gt.min(), gt.max()
    return (gt + noise_level * noise + bias).clamp(lo, hi)


def _gray_stretch(x: torch.Tensor, mask: np.ndarray | None = None) -> np.ndarray:
    """Percentile-stretch a 2D tensor to [0, 1] for display, NaN outside mask."""
    arr = x.detach().cpu().numpy().squeeze()
    if mask is not None and mask.any():
        valid = arr[mask]
        lo, hi = float(valid.min()), float(valid.max())
    else:
        lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    if mask is not None:
        out[~mask] = np.nan
    return out


def _signed_stretch(x: torch.Tensor, mask: np.ndarray | None = None,
                    vlim: float | None = None) -> tuple[np.ndarray, float]:
    """Symmetric [-vlim, vlim] -> [0, 1] stretch, NaN outside mask."""
    arr = x.detach().cpu().numpy().squeeze()
    if vlim is None:
        if mask is not None and mask.any():
            vlim = float(np.abs(arr[mask]).max()) + 1e-8
        else:
            vlim = float(np.abs(arr).max()) + 1e-8
    out = np.clip(arr / (2.0 * vlim) + 0.5, 0.0, 1.0)
    if mask is not None:
        out[~mask] = np.nan
    return out, vlim


def _extract_batch_sample(batch, i: int, device):
    """Pull sample i out of a batch dict, move to device, add batch dim."""
    img = batch["image"][i: i + 1].to(device).float()
    dtm = batch["dtm"][i: i + 1, :1].to(device).float()
    mask = batch["confidence"][i: i + 1].to(device).float()
    ortho = img.mean(dim=1, keepdim=True) if img.shape[1] == 3 else img
    mask_np = mask[0, 0].cpu().numpy().astype(bool)
    return img, dtm, mask, ortho, mask_np


def _get_pred(
        predictor_fn: Callable | None,
        batch: dict,
        i: int,
        gt_dtm: torch.Tensor,
        device,
) -> torch.Tensor:
    """Return a prediction tensor aligned to gt_dtm shape (1, 1, H, W)."""
    if predictor_fn is not None:
        pred = predictor_fn({k: v[i: i + 1].to(device) for k, v in batch.items()})
        if pred.shape[1] == 3:
            pred = pred[:, :1]
        return pred.float()
    return _make_synthetic_pred(gt_dtm, seed=42 + i)


# ============================================================================
# Visualizations
# ============================================================================


@torch.no_grad()
def visualize_huber_loss(
        dataloader,
        output_dir: Path,
        num_samples: int = 6,
        loss_fn: AbsoluteDepthLoss | None = None,
        predictor_fn: Callable | None = None,
):
    """Visualise Huber loss internals.

    Layout (6 columns):
        Ortho | GT DTM | Pred DTM | Signed Error (pred-gt) | Abs Err | Huber Map

    The last three columns together show the L1 vs L2 transition: pixels
    with |err| < delta have Huber ≈ 0.5·err² (quadratic darkening);
    pixels with |err| > delta have Huber ≈ delta·(|err| - 0.5·delta)
    (linear). Saturating regions in 'Abs Err' that stay mid-grey in
    'Huber Map' show the L1 clamping in action.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if loss_fn is None:
        loss_fn = AbsoluteDepthLoss()
    loss_fn = loss_fn.to(device).eval()
    delta = loss_fn.delta

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(num_samples, 6, figsize=(24, 4 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    count = 0
    with tqdm(total=num_samples, desc="Huber viz") as pbar:
        for batch in dataloader:
            if count >= num_samples:
                break
            for i in range(batch["image"].shape[0]):
                if count >= num_samples:
                    break

                img, dtm, mask, ortho, mask_np = _extract_batch_sample(batch, i, device)
                pred = _get_pred(predictor_fn, batch, i, dtm, device)

                # --- Use the actual loss method so the plot matches training ---
                huber_map = loss_fn.per_pixel_loss(pred, dtm)
                signed_err = (pred - dtm).float()
                abs_err = signed_err.abs()

                logger.info(f"Huber loss: {loss_fn(pred, dtm):.3f}")

                # --- Displays ---
                ortho_d = _gray_stretch(ortho, mask_np)
                dtm_d = _display_dtm(dtm[0, 0])
                pred_d = _display_dtm(pred[0, 0])
                signed_d, vlim_s = _signed_stretch(signed_err, mask_np)
                abs_d = abs_err[0, 0].cpu().numpy()
                if mask_np is not None:
                    abs_d_masked = np.where(mask_np, abs_d, np.nan)
                huber_d = huber_map[0, 0].cpu().numpy()
                if mask_np is not None:
                    huber_d_masked = np.where(mask_np, huber_d, np.nan)

                axes[count, 0].imshow(ortho_d, cmap="gray", vmin=0, vmax=1)
                axes[count, 1].imshow(dtm_d, cmap="terrain")
                axes[count, 2].imshow(pred_d, cmap="terrain")
                axes[count, 3].imshow(signed_d, cmap="RdBu_r", vmin=0, vmax=1)
                axes[count, 4].imshow(abs_d_masked, cmap="magma",
                                      vmin=0, vmax=max(2 * delta, 1e-3))
                axes[count, 5].imshow(huber_d_masked, cmap="magma",
                                      vmin=0, vmax=max(delta ** 2, 1e-4))

                for ax in axes[count]:
                    ax.axis("off")

                if count == 0:
                    titles = [
                        "Real Ortho",
                        "GT DTM",
                        "Pred DTM" if predictor_fn else "Pred DTM (synthetic)",
                        f"Signed Error (±{vlim_s:.2f})",
                        f"|Error|  (0..{2 * delta:.2f})",
                        f"Huber Loss (δ={delta:.2f})",
                    ]
                    for ax, t in zip(axes[0], titles):
                        ax.set_title(t, fontsize=11)

                count += 1
                pbar.update()

    save_path = output_dir / "huber_loss_inspection.png"
    save_fig(fig, save_path, bbox_inches="tight", dpi=DPI, facecolor="white")
    plt.close(fig)
    logger.info(f"Huber loss viz saved to: {save_path}")


@torch.no_grad()
def visualize_laplacian_loss(
        dataloader,
        output_dir: Path,
        num_samples: int = 6,
        loss_fn: LaplacianLoss | None = None,
        predictor_fn: Callable | None = None,
):
    """Visualise Laplacian (curvature) loss internals.

    Layout (6 columns):
        Ortho | GT DTM | GT ∇²d | Pred DTM | Pred ∇²d | |∇²d error|

    The Laplacian columns use a diverging colormap: blue = negative
    curvature (concave, crater floors), red = positive curvature (convex,
    central peaks and rims), white = flat. The error column uses the same
    ±vlim as the Laplacian columns so you can see at a glance whether the
    prediction has the RIGHT SHAPE (concave/convex structure matches)
    regardless of absolute-elevation mismatch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if loss_fn is None:
        loss_fn = LaplacianLoss()
    loss_fn = loss_fn.to(device).eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(num_samples, 6, figsize=(24, 4 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    count = 0
    with tqdm(total=num_samples, desc="Laplacian viz") as pbar:
        for batch in dataloader:
            if count >= num_samples:
                break
            for i in range(batch["image"].shape[0]):
                if count >= num_samples:
                    break

                img, dtm, mask, ortho, mask_np = _extract_batch_sample(batch, i, device)
                pred = _get_pred(predictor_fn, batch, i, dtm, device)

                gt_lap = loss_fn.laplacian(dtm)
                pr_lap = loss_fn.laplacian(pred)
                lap_err = (pr_lap - gt_lap).abs()

                logger.info(f"Laplacian loss: {loss_fn(pred, dtm):.3f}")

                # Shared diverging scale across GT and pred so colours are comparable
                combined = torch.cat([gt_lap, pr_lap], dim=0)
                _, shared_vlim = _signed_stretch(combined, None)

                ortho_d = _gray_stretch(ortho, mask_np)
                dtm_d = _display_dtm(dtm[0, 0])
                pred_d = _display_dtm(pred[0, 0])
                gt_lap_d, _ = _signed_stretch(gt_lap, mask_np, shared_vlim)
                pr_lap_d, _ = _signed_stretch(pr_lap, mask_np, shared_vlim)
                err_d = lap_err[0, 0].cpu().numpy()
                if mask_np is not None:
                    err_d = np.where(mask_np, err_d, np.nan)

                axes[count, 0].imshow(ortho_d, cmap="gray", vmin=0, vmax=1)
                axes[count, 1].imshow(dtm_d, cmap="terrain")
                axes[count, 2].imshow(gt_lap_d, cmap="RdBu_r")
                axes[count, 3].imshow(pred_d, cmap="terrain")
                axes[count, 4].imshow(pr_lap_d, cmap="RdBu_r")
                axes[count, 5].imshow(err_d, cmap="magma")

                for ax in axes[count]:
                    ax.axis("off")

                if count == 0:
                    titles = [
                        "Real Ortho",
                        "GT DTM",
                        f"GT ∇²d (±{shared_vlim:.2f})",
                        "Pred DTM" if predictor_fn else "Pred DTM (synthetic)",
                        "Pred ∇²d",
                        "|∇²d error|",
                    ]
                    for ax, t in zip(axes[0], titles):
                        ax.set_title(t, fontsize=11)

                count += 1
                pbar.update()

    save_path = output_dir / "laplacian_loss_inspection.png"
    save_fig(fig, save_path, bbox_inches="tight", dpi=DPI, facecolor="white")
    plt.close(fig)
    logger.info(f"Laplacian loss viz saved to: {save_path}")


@torch.no_grad()
def visualize_ordinal_ranking(
        dataloader,
        output_dir: Path,
        num_samples: int = 6,
        loss_fn: OrdinalRankingLoss | None = None,
        predictor_fn: Callable | None = None,
        pairs_to_draw: int = 250,
        seed: int = 0,
):
    """Visualise ordinal ranking internals.

    Layout (5 columns):
        Ortho | GT DTM | Pred DTM | Pair scatter (correct/violated/ambig) | Violation heatmap

    The pair-scatter column draws a random subset of sampled pairs as
    line segments between endpoints, coloured:
        green  = GT ordering unambiguous and prediction agrees
        red    = GT ordering unambiguous and prediction disagrees (violation)
        grey   = |gt_i - gt_j| <= margin (ambiguous, excluded from loss)

    The violation-heatmap column bins pair endpoints into a coarse grid
    and shows local violation rate, revealing WHERE the model tends to
    get ordering wrong (e.g. always at crater rims, or in shadowed
    regions).

    Per-image violation rate is printed in the title so you can track it
    as a scalar metric.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if loss_fn is None:
        loss_fn = OrdinalRankingLoss()
    loss_fn = loss_fn.to(device).eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(num_samples, 5, figsize=(22, 4 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]
    plt.subplots_adjust(wspace=0.08, hspace=0.15)

    count = 0
    with tqdm(total=num_samples, desc="Ordinal viz") as pbar:
        for batch in dataloader:
            if count >= num_samples:
                break
            for i in range(batch["image"].shape[0]):
                if count >= num_samples:
                    break

                img, dtm, mask, ortho, mask_np = _extract_batch_sample(batch, i, device)
                pred = _get_pred(predictor_fn, batch, i, dtm, device)
                H, W = pred.shape[-2:]
                N = H * W

                # Deterministic sampling for reproducible viz
                gen = torch.Generator(device=device).manual_seed(seed + count)
                idx_i, idx_j = loss_fn.sample_pairs(1, N, device, generator=gen, confidence=mask)
                stats = loss_fn.pair_stats(pred, dtm, idx_i, idx_j, mask)

                logger.info(f"Laplacian loss: {loss_fn(pred, dtm):.3f}")

                ordered = stats["ordered"][0].cpu().numpy()
                violations = stats["violations"][0].cpu().numpy()
                n_ord = int(ordered.sum())
                n_viol = int(violations.sum())
                viol_rate = (n_viol / n_ord) if n_ord > 0 else 0.0

                # ---- Pair scatter panel ----
                idx_i_np = idx_i[0].cpu().numpy()
                idx_j_np = idx_j[0].cpu().numpy()
                yi, xi = np.divmod(idx_i_np, W)
                yj, xj = np.divmod(idx_j_np, W)

                # Subsample for legibility
                draw = min(pairs_to_draw, len(idx_i_np))
                rng = np.random.default_rng(seed + count)
                subset = rng.choice(len(idx_i_np), size=draw, replace=False)

                ortho_d = _gray_stretch(ortho, mask_np)
                dtm_d = _display_dtm(dtm[0, 0])
                pred_d = _display_dtm(pred[0, 0])

                axes[count, 0].imshow(ortho_d, cmap="gray", vmin=0, vmax=1)
                axes[count, 1].imshow(dtm_d, cmap="terrain")
                axes[count, 2].imshow(pred_d, cmap="terrain")

                # Pair scatter on top of dimmed ortho
                ax_sc = axes[count, 3]
                ax_sc.imshow(ortho_d, cmap="gray", vmin=0, vmax=1, alpha=0.45)
                for k in subset:
                    if violations[k]:
                        col, lw, z = "#ff2a2a", 0.9, 3
                    elif ordered[k]:
                        col, lw, z = "#22cc55", 0.4, 2
                    else:
                        col, lw, z = "#888888", 0.2, 1
                    ax_sc.plot([xi[k], xj[k]], [yi[k], yj[k]],
                               "-", color=col, linewidth=lw, zorder=z, alpha=0.8)
                ax_sc.set_xlim(0, W)
                ax_sc.set_ylim(H, 0)

                # Violation heatmap over coarse grid
                grid = 32
                heat = np.zeros((grid, grid), dtype=np.float32)
                cnt = np.zeros((grid, grid), dtype=np.float32)
                for k in range(len(idx_i_np)):
                    if not ordered[k]:
                        continue
                    for (y, x) in [(yi[k], xi[k]), (yj[k], xj[k])]:
                        gy = min(int(y * grid / H), grid - 1)
                        gx = min(int(x * grid / W), grid - 1)
                        cnt[gy, gx] += 1
                        if violations[k]:
                            heat[gy, gx] += 1
                with np.errstate(invalid="ignore", divide="ignore"):
                    heat_rate = np.where(cnt > 0, heat / cnt, np.nan)

                axes[count, 4].imshow(heat_rate, cmap="magma",
                                      vmin=0, vmax=max(0.3, viol_rate * 1.5),
                                      interpolation="nearest")

                for ax in axes[count]:
                    ax.axis("off")

                # Per-sample stats in a caption above the pair scatter
                axes[count, 3].set_title(
                    f"pairs: {n_ord}/{len(ordered)} ordered | "
                    f"violations: {n_viol} ({100 * viol_rate:.1f} %)",
                    fontsize=9,
                )

                if count == 0:
                    titles = [
                        "Real Ortho",
                        "GT DTM",
                        "Pred DTM" if predictor_fn else "Pred DTM (synthetic)",
                        "Pair samples (red=violation)",
                        "Local violation rate",
                    ]
                    # Put the top-row title only where we don't already have a per-sample one
                    for j, t in enumerate(titles):
                        if j == 3:
                            continue  # already has per-sample title
                        axes[0, j].set_title(t, fontsize=11)

                count += 1
                pbar.update()

    # Add legend in the bottom margin
    handles = [
        mpatches.Patch(color="#22cc55", label="Correct ordering"),
        mpatches.Patch(color="#ff2a2a", label="Violation"),
        mpatches.Patch(color="#888888", label="Ambiguous (|Δgt|<margin)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    save_path = output_dir / "ordinal_ranking_inspection.png"
    save_fig(fig, save_path, bbox_inches="tight", dpi=DPI, facecolor="white")
    plt.close(fig)
    logger.info(f"Ordinal ranking viz saved to: {save_path}")


@torch.no_grad()
def visualize_random_flips_and_rotations(dataloader, output_dir: Path, num_samples: int = 4):
    """
    Simulates the 8 deterministic states of flips and rotations to visually
    validate that the sun vector stays physically locked to the terrain shading.
    Evaluates 'num_samples' separate patches and saves an image for each.
    """
    logger.info(f"Generating Sun Vector Augmentation Validation for {num_samples} samples...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect n samples safely across batches
    samples = []
    for batch in tqdm(dataloader, total=num_samples, desc="Collecting samples"):
        B = batch["image"].shape[0]
        for i in range(B):
            samples.append({
                "image": batch["image"][i],
                "dtm": batch["dtm"][i],
                "confidence": batch["confidence"][i],
                "sun_vector": batch["sun_vector"][i]
            })
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break

    # Define the core transformations to test (Name, H-Flip, V-Flip, k_rot)
    transformations = [
        ("Original", False, False, 0),
        ("H-Flip", True, False, 0),
        ("V-Flip", False, True, 0),
        ("Rot 90 (CCW)", False, False, 1),
        ("Rot 180", False, False, 2),
        ("Rot 270 (CW)", False, False, 3),
        ("H-Flip + Rot 90", True, False, 1),
        ("V-Flip + Rot 270", False, True, 3),
    ]

    # 2. Generate a grid for each sample
    for sample_idx, sample in tqdm(enumerate(samples), total=len(samples),
                                   desc="Generating Sun Vector Augmentation Validation"):
        fig, axes = plt.subplots(len(transformations), 3, figsize=(15, 5 * len(transformations)))
        plt.subplots_adjust(wspace=0.1, hspace=0.3)

        base_img = sample["image"]  # (C, H, W)
        base_dtm = sample["dtm"]  # (1, H, W)
        base_conf = sample["confidence"]  # (1, H, W)
        base_sun = sample["sun_vector"]  # (3,)

        for i, (name, h_flip, v_flip, k_rot) in enumerate(transformations):
            # Clone base tensors
            img = base_img.clone()
            dtm = base_dtm.clone()
            conf = base_conf.clone()
            sun = base_sun.clone()

            # Apply exact augmentation logic
            if h_flip:
                img = torch.flip(img, [-1])
                dtm = torch.flip(dtm, [-1])
                conf = torch.flip(conf, [-1])
                sun[0] = -sun[0]

            if v_flip:
                img = torch.flip(img, [-2])
                dtm = torch.flip(dtm, [-2])
                conf = torch.flip(conf, [-2])
                sun[1] = -sun[1]

            if k_rot > 0:
                img = torch.rot90(img, k=k_rot, dims=[-2, -1])
                dtm = torch.rot90(dtm, k=k_rot, dims=[-2, -1])
                conf = torch.rot90(conf, k=k_rot, dims=[-2, -1])
                sx, sy = sun[0].clone(), sun[1].clone()
                if k_rot == 1:
                    sun[0], sun[1] = sy, -sx
                elif k_rot == 2:
                    sun[0], sun[1] = -sx, -sy
                elif k_rot == 3:
                    sun[0], sun[1] = -sy, sx

            # Format for matplotlib
            img_np = np.clip((np.transpose(img.cpu().numpy(), (1, 2, 0)) + 1.0) / 2.0, 0.0, 1.0)
            dtm_np = _display_dtm(dtm[0])
            mask_np = conf[0].cpu().numpy()

            # Isolate spatial dimensions to place the arrow in the center
            H, W = dtm_np.shape
            cx, cy = W // 2, H // 2
            vx, vy = sun[0].item(), sun[1].item()

            # Scale arrow to be 30% of the image size for visibility
            arrow_scale = min(W, H) * 0.3

            # --- Plot Ortho ---
            axes[i, 0].imshow(img_np, cmap='gray' if img_np.shape[-1] == 1 else None, vmin=0, vmax=1)
            axes[i, 0].arrow(cx, cy, vx * arrow_scale, vy * arrow_scale, color='red', head_width=12, head_length=15,
                             linewidth=2)
            axes[i, 0].set_title(f"Sample {sample_idx} | {name} - Ortho\nSun XY: [{vx:.2f}, {vy:.2f}]")
            axes[i, 0].axis('off')

            # --- Plot DTM ---
            axes[i, 1].imshow(dtm_np, cmap='terrain', vmin=0, vmax=1)
            axes[i, 1].arrow(cx, cy, vx * arrow_scale, vy * arrow_scale, color='red', head_width=12, head_length=15,
                             linewidth=2)
            axes[i, 1].set_title(f"Sample {sample_idx} | {name} - DTM")
            axes[i, 1].axis('off')

            # --- Plot Mask ---
            axes[i, 2].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].arrow(cx, cy, vx * arrow_scale, vy * arrow_scale, color='red', head_width=12, head_length=15,
                             linewidth=2)
            axes[i, 2].set_title(f"Sample {sample_idx} | {name} - Mask")
            axes[i, 2].axis('off')

        save_path = output_dir / f"augmentation_sun_vector_validation_sample_{sample_idx:02d}.png"
        save_fig(fig, save_path, bbox_inches="tight", dpi=300, facecolor="white")
        plt.close(fig)

    logger.info(f"Saved {num_samples} augmentation validation grids to: {output_dir}")


@torch.no_grad()
def visualize_solar_distribution(dataloader, output_dir: Path, num_batches: int = -1):
    """
    Visualizes solar physics and saves individual plots for publication.
    """
    logger.info("Generating and saving individual solar distribution plots...")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_neurips_style()

    sun_vecs, intensities, ambients = [], [], []

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Extracting sun vectors"):
        if 0 < num_batches <= i: break
        sun_vecs.append(batch["sun_vector"].cpu().numpy())
        intensities.append(batch["intensity"].cpu().numpy())
        ambients.append(batch["ambient"].cpu().numpy())

    sv = np.concatenate(sun_vecs, axis=0)
    it = np.concatenate(intensities, axis=0).flatten()
    am = np.concatenate(ambients, axis=0).flatten()

    # CRITICAL: Fix for the RuntimeWarning (Negative sizes)
    # We clip ambient at a tiny positive value so the sqrt doesn't fail
    viz_ambient_sizes = np.clip(am, 1e-6, None) * 500

    azimuth = np.arctan2(sv[:, 1], sv[:, 0])
    elevation = np.degrees(np.arcsin(sv[:, 2]))

    # --- 1. Standalone 3D Solar Compass ---
    fig_3d = plt.figure(figsize=(8, 8))
    ax1 = fig_3d.add_subplot(111, projection='3d')
    # Wireframe hemisphere
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi / 2:15j]
    ax1.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                       color='gray', alpha=0.1, linewidth=0.5)

    p3d = ax1.scatter(sv[:, 0], sv[:, 1], sv[:, 2],
                      c=it, cmap='plasma', s=viz_ambient_sizes,
                      alpha=0.8, edgecolors='w', linewidth=0.2)
    ax1.set_title("3D Solar Vector Compass")
    fig_3d.colorbar(p3d, ax=ax1, shrink=0.6, label='Intensity')
    save_fig(fig_3d, output_dir / "solar_compass_3d.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_3d)

    # --- 2. Standalone Polar Sky-Map ---
    fig_polar = plt.figure(figsize=(8, 8))
    ax2 = fig_polar.add_subplot(111, projection='polar')
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    sc2 = ax2.scatter(azimuth, elevation, c=it, cmap='plasma', alpha=0.7)
    ax2.set_ylim(0, 90)
    ax2.set_title("Solar Sky-Map (Azimuth vs Elevation)")
    save_fig(fig_polar, output_dir / "solar_sky_map_polar.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_polar)

    # --- 3. Standalone Illumination Coupling ---
    fig_corr = plt.figure(figsize=(8, 8))
    ax3 = fig_corr.add_subplot(111)

    sns.regplot(x=it, y=am, ax=ax3, scatter_kws={'alpha': 0.4, 's': 20}, line_kws={'color': 'red'})
    ax3.set_title("Illumination Coupling (Intensity vs Ambient)")
    ax3.set_xlabel("Solar Intensity")
    ax3.set_ylabel("Ambient (Sky) Light")
    save_fig(fig_corr, output_dir / "solar_coupling_regression.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_corr)

    logger.info(f"Individual solar figures saved to {output_dir}")


def _prep_ortho_rgb(img_np: np.ndarray) -> np.ndarray:
    """(C,H,W) float in ~[-1,1] -> (H,W,3) float in [0,1]."""
    if img_np.ndim == 2:
        img_np = img_np[None]
    if img_np.shape[0] == 1:
        img_np = np.repeat(img_np, 3, axis=0)
    rgb = np.transpose(img_np[:3], (1, 2, 0))
    rgb = np.clip((rgb + 1.0) / 2.0, 0.0, 1.0)
    return rgb


def _apply_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = arr.copy().astype(np.float32)
    if out.ndim == 3:
        out[~mask] = np.nan
    else:
        out[~mask] = np.nan
    return out


def _draw_best_line(ax, result: SeamResult, H: int, W: int, color="#ff3b30"):
    if result.seam_score <= 0:
        return
    y1, x1, y2, x2 = result.line_endpoints(clip_hw=(H, W))
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=2.0, alpha=0.85)
    # small marker at centroid
    ax.plot([result.best_x], [result.best_y], marker="o",
            markersize=4, color=color, alpha=0.9)


def render_sample_diagnostics(
        image_chw: np.ndarray,
        dtm_hw: np.ndarray,
        mask_hw: np.ndarray,
        ortho_grad_hw: np.ndarray,
        result: SeamResult,
        *,
        title: str | None = None,
        figsize: tuple = (20, 4.2),
) -> plt.Figure:
    """Single-sample, 6-panel diagnostic figure."""
    H, W = mask_hw.shape
    mask_bool = mask_hw.astype(bool)

    rgb = _prep_ortho_rgb(image_chw)
    rgb_masked = rgb.copy()
    rgb_masked[~mask_bool] = np.nan

    dtm_masked = _apply_mask(np.clip((dtm_hw + 1) / 2.0, 0, 1), mask_bool)
    grad_masked = _apply_mask(ortho_grad_hw, mask_bool)
    p2, p98 = np.nanpercentile(grad_masked, [2, 98]) if np.any(~np.isnan(grad_masked)) else (0, 1)
    grad_disp = np.clip((grad_masked - p2) / (p98 - p2 + 1e-8), 0, 1)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.9], wspace=0.12)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(rgb_masked)
    _draw_best_line(ax0, result, H, W, color="#ff3b30")
    ax0.set_title("Ortho + seam line")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(dtm_masked, cmap="terrain", vmin=0, vmax=1)
    _draw_best_line(ax1, result, H, W, color="#ff3b30")
    ax1.set_title("GT DTM + seam line")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(grad_disp, cmap="magma")
    ax2.set_title("Ortho gradient (|∇I|)")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 3])
    heat = result.seam_heatmap
    if heat is not None:
        h_disp = heat.copy()
        h_disp[~mask_bool] = np.nan
        vmax = float(np.nanpercentile(h_disp, 99)) if np.any(~np.isnan(h_disp)) else 1.0
        ax3.imshow(h_disp, cmap="inferno", vmin=0, vmax=max(vmax, 1e-6))
        _draw_best_line(ax3, result, H, W, color="#00ffff")
    ax3.set_title("Seam response")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[0, 4])
    d_heat = result.cohens_d_heatmap
    if d_heat is not None:
        d_disp = d_heat.copy()
        d_disp[~mask_bool] = np.nan
        ax4.imshow(d_disp, cmap="viridis", vmin=0,
                   vmax=max(1.0, float(np.nanpercentile(d_disp, 99)) if np.any(~np.isnan(d_disp)) else 1.0))
    ax4.set_title("Cohen's d across line")
    ax4.axis("off")

    ax5 = fig.add_subplot(gs[0, 5], projection="polar")
    pa = result.per_angle_max
    if pa is not None and len(pa) > 0:
        # repeat so the polar plot closes, and double because directions are pi-periodic
        angles = np.linspace(0, np.pi, len(pa), endpoint=False)
        angles_full = np.concatenate([angles, angles + np.pi, [angles[0]]])
        vals_full = np.concatenate([pa, pa, [pa[0]]])
        ax5.plot(angles_full, vals_full, color="#ff3b30", linewidth=1.5)
        ax5.fill(angles_full, vals_full, color="#ff3b30", alpha=0.25)
        # highlight the winning angle
        ax5.plot([result.best_angle_rad, result.best_angle_rad + np.pi],
                 [max(pa), max(pa)],
                 color="#00ffff", linewidth=2.0, alpha=0.9)
    ax5.set_title(f"per-angle max\nbest={np.degrees(result.best_angle_rad):.0f}°",
                  fontsize=9)
    ax5.set_xticklabels([])
    ax5.set_yticklabels([])
    ax5.grid(alpha=0.3)

    score_text = (
        f"seam={result.seam_score:7.2f}   "
        f"ortho={result.ortho_score:5.2f}   "
        f"dtm={result.dtm_score:5.2f}   "
        f"d={result.cohens_d:4.2f}   "
        f"@({result.best_y},{result.best_x}) "
        f"θ={np.degrees(result.best_angle_rad):.0f}°"
    )
    if title is None:
        title = score_text
    else:
        title = f"{title}   |   {score_text}"
    fig.suptitle(title, fontsize=11, y=1.02)

    return fig


# ---------------------------------------------------------------------------
# batch mode (backward-compatible entry point)
# ---------------------------------------------------------------------------
@torch.no_grad()
def visualize_seam_artifacts(dataloader, output_dir: Path, num_samples: int = 16):
    """Enhanced drop-in replacement. Saves a top-N/bottom-N diagnostic grid."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluated = []
    scan_limit = len(dataloader.dataset)

    sobel_x = torch.tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
        device=device,
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor(
        [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
        device=device,
    ).view(1, 1, 3, 3) / 8.0

    with tqdm(total=scan_limit, desc="Scanning for seams") as pbar:
        for batch in dataloader:
            B = batch["image"].shape[0]
            for i in range(B):
                if len(evaluated) >= scan_limit:
                    break
                img = batch["image"][i:i + 1].to(device)
                dtm = batch["dtm"][i:i + 1, :1].to(device)
                mask = batch["confidence"][i:i + 1].to(device) > 0.5

                result = detect_seam_artifact(
                    img, dtm, mask, return_diagnostics=True
                )

                ortho_gray = img.float().mean(dim=1, keepdim=True)
                safe = ortho_gray.clone()
                safe[~mask] = 0.0
                gx = F.conv2d(safe, sobel_x, padding=1)
                gy = F.conv2d(safe, sobel_y, padding=1)
                ortho_grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

                evaluated.append({
                    "result": result,
                    "img": img[0].cpu().numpy(),
                    "dtm": dtm[0, 0].cpu().numpy(),
                    "mask": mask[0, 0].cpu().numpy(),
                    "ortho_grad": ortho_grad[0, 0].cpu().numpy(),
                })
                pbar.update(1)
            if len(evaluated) >= scan_limit:
                break

    evaluated.sort(key=lambda s: s["result"].composite_score, reverse=True)
    n = min(num_samples, len(evaluated))
    selected = evaluated[:n] + evaluated[-n:]

    # compose a tall figure: expanded to 8 columns for new structural diagnostics
    total_rows = len(selected)
    cols = 8
    scale = 4

    fig, axes = plt.subplots(total_rows, cols, figsize=(cols * scale, scale * total_rows),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1, 1, 0.9]})
    if total_rows == 1:
        axes = axes[None, :]

    for row, item in enumerate(selected):
        H, W = item["mask"].shape
        mask_bool = item["mask"].astype(bool)
        res = item["result"]
        rgb = _prep_ortho_rgb(item["img"])
        rgb[~mask_bool] = np.nan
        dtm_disp = _apply_mask(np.clip((item["dtm"] + 1) / 2, 0, 1), mask_bool)
        grad_masked = _apply_mask(item["ortho_grad"], mask_bool)
        p2, p98 = np.nanpercentile(grad_masked, [2, 98]) if np.any(~np.isnan(grad_masked)) else (0, 1)
        grad_disp = np.clip((grad_masked - p2) / (p98 - p2 + 1e-8), 0, 1)

        # Base Visualizations
        axes[row, 0].imshow(rgb)
        _draw_best_line(axes[row, 0], res, H, W)
        axes[row, 1].imshow(dtm_disp, cmap="terrain", vmin=0, vmax=1)
        _draw_best_line(axes[row, 1], res, H, W)
        axes[row, 2].imshow(grad_disp, cmap="magma")

        if res.seam_heatmap is not None:
            h = res.seam_heatmap.copy()
            h[~mask_bool] = np.nan
            vmax = float(np.nanpercentile(h, 99)) if np.any(~np.isnan(h)) else 1.0
            axes[row, 3].imshow(h, cmap="inferno", vmin=0, vmax=max(vmax, 1e-6))
            _draw_best_line(axes[row, 3], res, H, W, color="#00ffff")

        if res.cohens_d_heatmap is not None:
            d = res.cohens_d_heatmap.copy()
            d[~mask_bool] = np.nan
            axes[row, 4].imshow(d, cmap="viridis", vmin=0,
                                vmax=max(1.0, float(np.nanpercentile(d, 99)) if np.any(~np.isnan(d)) else 1.0))

        # New Diagnostic: Span / Connected Components
        if hasattr(res, 'diag_closed_components') and res.diag_closed_components is not None:
            cc = res.diag_closed_components.copy()
            cc_disp = np.ma.masked_where(cc == 0, cc)  # Hide background
            axes[row, 5].imshow(cc_disp, cmap="tab20", interpolation="nearest")
        else:
            axes[row, 5].text(0.5, 0.5, "No Span Data", ha='center', va='center')

        # New Diagnostic: Linearity (Hough lines superimposed on sparsity hot-mask)
        if hasattr(res, 'diag_hot_mask') and res.diag_hot_mask is not None:
            hm = res.diag_hot_mask.astype(float)
            axes[row, 6].imshow(hm, cmap="Reds", vmin=0, vmax=1)

            if hasattr(res, 'diag_hough_lines') and res.diag_hough_lines is not None:
                hl = res.diag_hough_lines.astype(float)
                hl_disp = np.ma.masked_where(hl == 0, hl)
                axes[row, 6].imshow(hl_disp, cmap="cool", vmin=0, vmax=1, alpha=0.8)
        else:
            axes[row, 6].text(0.5, 0.5, "No Linearity Data", ha='center', va='center')

        for ax in axes[row, :cols - 1]:
            ax.axis("off")

        # Polar Panel: replace cartesian with polar
        pa = res.per_angle_max
        pos = axes[row, cols - 1].get_position()
        axes[row, cols - 1].remove()
        pax = fig.add_subplot(total_rows, cols, row * cols + cols, projection="polar")
        pax.set_position(pos)
        if pa is not None and len(pa):
            angles = np.linspace(0, np.pi, len(pa), endpoint=False)
            angles_full = np.concatenate([angles, angles + np.pi, [angles[0]]])
            vals_full = np.concatenate([pa, pa, [pa[0]]])
            pax.plot(angles_full, vals_full, color="#ff3b30", linewidth=1.2)
            pax.fill(angles_full, vals_full, color="#ff3b30", alpha=0.25)
            pax.plot([res.best_angle_rad, res.best_angle_rad + np.pi],
                     [max(pa), max(pa)], color="#00ffff", linewidth=2.0)
        pax.set_xticklabels([])
        pax.set_yticklabels([])
        pax.grid(alpha=0.3)

        if row == 0:
            titles = ["Ortho+line", "DTM+line", "Ortho grad",
                      "Seam response", "Cohen's d", "Span (CCs)", "Linearity"]
            for ax, t in zip(axes[0, :cols - 1], titles):
                ax.set_title(t, fontweight='bold', pad=10)
            pax.set_title("Per-Angle", fontweight='bold', pad=10)

        threshold = res.is_seam

        # Dynamic label extracting structural multipliers
        cls = "HIGH" if threshold else "LOW"
        color = "red" if threshold else "green"
        label = (
            f"#{row}  {cls}\n"
            f"comp={res.composite_score:6.1f}\n"
            f"span={getattr(res, 'span', 0.0):4.2f}\n"
            f"spars={getattr(res, 'sparsity', 0.0):4.2f}\n"
            f"seam={res.seam_score:6.1f}\n"
            f"d={res.cohens_d:4.2f}\n"
            f"θ={np.degrees(res.best_angle_rad):.0f}°"
        )
        axes[row, 0].text(-0.12, 0.5, label,
                          transform=axes[row, 0].transAxes,
                          fontsize=10, fontweight="bold", family="monospace",
                          va="center", ha="right", color=color)

    save_path = output_dir / "seam_artifact_inspection.png"
    save_fig(fig, save_path, bbox_inches="tight", dpi=DPI, facecolor="white")
    return save_path


@torch.no_grad()
def visualize_tin_artifacts(dataloader, output_dir: Path, num_samples: int = 8, kernel_size: int = 32):
    """
    Visualizes TIN artifacts by finding patches with high local planar density.
    Layout: [Ortho] | [GT DTM] | [Laplacian Magnitude] | [TIN Density Map]
    """
    logger.info(f"Generating TIN Artifact visualization for {num_samples} samples...")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluated_samples = []
    scan_limit = len(dataloader.dataset)  # Scan a larger buffer to find actual TINs

    laplacian_kernel = torch.tensor([[[[0.0, 1.0, 0.0],
                                       [1.0, -4.0, 1.0],
                                       [0.0, 1.0, 0.0]]]], device=device)

    with tqdm(total=scan_limit, desc="Scanning for TIN artifacts") as pbar:
        for batch in dataloader:
            B = batch["image"].shape[0]
            for i in range(B):
                if len(evaluated_samples) >= scan_limit:
                    break

                img = batch["image"][i: i + 1].to(device)
                dtm = batch["dtm"][i: i + 1, :1].to(device)
                mask = batch["confidence"][i: i + 1].to(device)

                # --- Extract TIN Logic for Spatial Mapping ---
                safe_elev = dtm.clone()
                safe_elev[~mask.bool()] = 0.0

                laplacian = F.conv2d(safe_elev, laplacian_kernel, padding=1)

                invalid_mask = (~mask.bool()).float()
                dilated_invalid = F.max_pool2d(invalid_mask, kernel_size=3, stride=1, padding=1)
                eroded_valid = (dilated_invalid == 0.0).float()

                zero_curvature_mask = ((laplacian.abs() < 1e-2) * eroded_valid.bool()).float()

                local_planar_sum = F.avg_pool2d(zero_curvature_mask, kernel_size=kernel_size, stride=1)
                local_valid_sum = F.avg_pool2d(eroded_valid, kernel_size=kernel_size, stride=1)

                safe_valid_sum = torch.clamp(local_valid_sum, min=1e-6)
                local_density = local_planar_sum / safe_valid_sum

                valid_window_mask = local_valid_sum >= 0.5

                if valid_window_mask.any():
                    score = local_density[valid_window_mask].max().item()
                else:
                    score = 0.0

                # Interpolate density map back to original size for side-by-side visualization
                # (avg_pool2d with stride=1 shrinks size by kernel_size - 1)
                pad_top = kernel_size // 2
                pad_bottom = kernel_size - 1 - pad_top
                density_map = F.pad(local_density, (pad_top, pad_bottom, pad_top, pad_bottom), mode='constant',
                                    value=0.0)

                evaluated_samples.append({
                    "score": score,
                    "img": img[0].cpu().numpy(),
                    "dtm": dtm[0, 0].cpu().numpy(),
                    "mask": mask[0, 0].cpu().numpy(),
                    "laplacian": laplacian[0, 0].cpu().numpy(),
                    "density": density_map[0, 0].cpu().numpy()
                })
                pbar.update(1)

            if len(evaluated_samples) >= scan_limit:
                break

    # Sort by TIN score descending
    evaluated_samples.sort(key=lambda x: x["score"], reverse=True)

    # Select the Top N (Severe TINs) and Bottom N (Clean terrain)
    num_samples = min(num_samples, len(evaluated_samples))
    if num_samples == 0:
        logger.warning("No valid samples found for TIN visualization.")
        return

    half = num_samples // 2
    selected = evaluated_samples[:half] + evaluated_samples[-half:]

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    for count, item in enumerate(selected):
        img_disp = np.clip((np.transpose(item["img"], (1, 2, 0)) + 1.0) / 2.0, 0.0, 1.0)
        dtm_disp = _display_dtm(item["dtm"])
        mask_np = item["mask"].astype(bool)
        lap_np = np.abs(item["laplacian"])
        density_np = item["density"]

        # Mask invalid areas purely for visual clarity
        img_disp[~mask_np] = np.nan
        dtm_disp[~mask_np] = np.nan
        lap_np[~mask_np] = np.nan
        density_np[~mask_np] = np.nan

        axes[count, 0].imshow(img_disp, vmin=0, vmax=1)
        axes[count, 1].imshow(dtm_disp, cmap="terrain", vmin=0, vmax=1)

        # Stretch Laplacian for visibility (highlighting sharp edges)
        p98 = np.nanpercentile(lap_np, 98) if np.any(~np.isnan(lap_np)) else 1.0
        lap_disp = np.clip(lap_np / (p98 + 1e-8), 0, 1)
        axes[count, 2].imshow(lap_disp, cmap="magma")

        # Local Density Map (0 to 1 heatmap)
        axes[count, 3].imshow(density_np, cmap="jet", vmin=0, vmax=1)

        for ax in axes[count]:
            ax.axis("off")

        if count == 0:
            titles = ["Masked Ortho", "Masked GT DTM", "Laplacian Magnitude", "TIN Density Map"]
            for ax, t in zip(axes[0], titles):
                ax.set_title(t)

        # Add side-label indicating TIN Score
        label = "High Score\n(TIN Suspected)" if count < half else "Low Score\n(Clean Terrain)"
        axes[count, 0].text(-0.1, 0.5, f"Score: {item['score']:.2f}\n{label}",
                            transform=axes[count, 0].transAxes, fontsize=12, fontweight='bold',
                            va='center', ha='right', color='red' if count < half else 'green')

    save_path = output_dir / "tin_artifact_inspection.png"
    save_fig(fig, save_path, bbox_inches="tight", dpi=DPI, facecolor="white")
    plt.close(fig)
    logger.info(f"TIN artifact visualization saved to: {save_path}")


@torch.no_grad()
def visualize_loss_components(dataloader, output_dir, num_samples=4):
    """
    Visualizes the internal data representations used by the auxiliary loss functions.
    Layout: [Ortho] | [GT DTM] | [Grads 1x] | [Grads 4x] | [FFT Spectrum]
    """
    logger.info(f"Generating Loss Component visualizations for {num_samples} samples...")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    count = 0
    with tqdm(total=num_samples) as pbar:
        for batch in dataloader:
            if count >= num_samples:
                break

            B = batch["image"].shape[0]
            for i in range(B):
                if count >= num_samples:
                    break

                img = batch["image"][i: i + 1]
                dtm = batch["dtm"][i: i + 1, :1]
                mask = batch["confidence"][i: i + 1]

                def get_grad_mag(tensor, scale):
                    if scale > 1:
                        tensor = F.avg_pool2d(tensor, kernel_size=scale, stride=scale)
                    dy = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
                    dx = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
                    dy = F.pad(dy, (0, 0, 0, 1))
                    dx = F.pad(dx, (0, 1, 0, 0))
                    return torch.sqrt(dx ** 2 + dy ** 2)[0, 0].cpu().numpy()

                grad_1x = get_grad_mag(dtm, scale=1)
                grad_4x = get_grad_mag(dtm, scale=4)

                # ---------------------------------------------------------
                # 2. Focal Frequency (Simulating FocalFrequencyLoss)
                # ---------------------------------------------------------
                # GT Injection to prevent boundary ringing, just like your FFL code
                # (Using 0 as dummy prediction since we just want to see GT spectrum)
                dtm_masked = dtm * mask + 0.0 * (1.0 - mask)

                # 2D Fast Fourier Transform
                fft = torch.fft.fft2(dtm_masked[0, 0])
                fft_shift = torch.fft.fftshift(fft)  # Move low frequencies to center

                # Log magnitude for visualization (add 1 to avoid log(0))
                fft_mag = torch.log(torch.abs(fft_shift) + 1).cpu().numpy()
                # ---------------------------------------------------------
                # Prep for Plotting
                # ---------------------------------------------------------
                img_disp = np.clip((np.transpose(img[0].cpu().numpy(), (1, 2, 0)) + 1.0) / 2.0, 0.0, 1.0)
                dtm_disp = _display_dtm(dtm[0, 0])
                mask_np = mask[0, 0].cpu().numpy().astype(bool)
                img_disp[~mask_np] = np.nan
                dtm_disp[~mask_np] = np.nan

                p2, p98 = np.percentile(grad_1x, [2, 98])
                grad_1x_disp = np.clip((grad_1x - p2) / (p98 - p2 + 1e-8), 0, 1)
                p2, p98 = np.percentile(grad_4x, [2, 98])
                grad_4x_disp = np.clip((grad_4x - p2) / (p98 - p2 + 1e-8), 0, 1)

                axes[count, 0].imshow(img_disp)
                axes[count, 1].imshow(dtm_disp, cmap="terrain")
                axes[count, 2].imshow(grad_1x_disp, cmap="magma")
                axes[count, 3].imshow(grad_4x_disp, cmap="magma")
                axes[count, 4].imshow(fft_mag, cmap="plasma")
                for ax in axes[count]:
                    ax.axis("off")
                if count == 0:
                    for ax, t in zip(
                            axes[0],
                            ["Ortho Input", "GT DTM", "L_grad: 1x Scale Mag", "L_grad: 4x Scale Mag",
                             "L_FFL: 2D FFT Spectrum"],
                    ):
                        ax.set_title(t)

                count += 1
                pbar.update()

    save_path = output_dir / "loss_components_inspection.png"
    save_fig(fig, save_path, bbox_inches="tight", dpi=DPI, facecolor="white")
    plt.close(fig)
    logger.info(f"Loss components visualization saved to: {save_path}")


@torch.no_grad()
def visualize_invalid_fill(dataloader, output_dir: Path, num_samples: int = 4, iterations=64):
    """
    Visualizes the smooth diffusion infilling process for VAE optimization.
    Layout: [Mask] | [Ortho Masked] | [Ortho Smooth Fill] | [DTM Masked] | [DTM Smooth Fill]
    """
    logger.info(f"Generating Smooth Infilling visualization for {num_samples} samples...")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    count = 0
    with tqdm(total=num_samples) as pbar:
        for batch in dataloader:
            if count >= num_samples:
                break
            B = batch["image"].shape[0]

            for i in range(B):
                if count >= num_samples:
                    break

                img = batch["image"][i: i + 1].to(device)
                dtm = batch["dtm"][i: i + 1, :1].to(device)
                mask = batch["confidence"][i: i + 1].to(device)

                if not (mask == 0).any():
                    continue

                # 1. Force mask the inputs (in case dataset already filled them)
                img_masked = img * mask
                dtm_masked = dtm * mask

                # 2. Apply smooth diffusion
                img_filled, dtm_filled, mask = fill_voids_gmrf(img_masked, dtm_masked, mask)

                # 3. Prepare for plotting (Denormalize [-1, 1] to [0, 1])
                mask_np = mask[0, 0].cpu().numpy()

                img_masked_np = np.clip((np.transpose(img_masked[0].cpu().numpy(), (1, 2, 0)) + 1.0) / 2.0, 0.0, 1.0)
                img_filled_np = np.clip((np.transpose(img_filled[0].cpu().numpy(), (1, 2, 0)) + 1.0) / 2.0, 0.0, 1.0)

                dtm_masked_np = _display_dtm(dtm_masked[0, 0])
                dtm_filled_np = _display_dtm(dtm_filled[0, 0])

                # Set masked regions to NaN for the "Masked" plots so they show up clear white/blank
                bool_mask = mask_np.astype(bool)
                img_masked_np[~bool_mask] = np.nan
                dtm_masked_np[~bool_mask] = np.nan

                # Plot
                axes[count, 0].imshow(mask_np, cmap="gray")
                axes[count, 1].imshow(img_masked_np)
                axes[count, 2].imshow(img_filled_np)
                axes[count, 3].imshow(dtm_masked_np, cmap="terrain")
                axes[count, 4].imshow(dtm_filled_np, cmap="terrain")

                for ax in axes[count]:
                    ax.axis("off")

                if count == 0:
                    titles = ["Confidence Mask", "Masked Ortho", "Smooth Fill Ortho", "Masked DTM", "Smooth Fill DTM"]
                    for ax, t in zip(axes[0], titles):
                        ax.set_title(t)

                count += 1
                pbar.update()

    save_path = output_dir / "smooth_fill_inspection.png"
    save_fig(fig, save_path, bbox_inches="tight", dpi=DPI, facecolor="white")
    plt.close(fig)
    logger.info(f"Smooth filling visualization saved to: {save_path}")


@torch.no_grad()
def visualize_loss_physics(
        dataloader,
        output_dir: Path,
        num_samples: int = 6,
        loss_fn: "PhotoclinometricLoss | None" = None,
        lunar_lambert_weight_override: float | None = None,
        ref_scale: float = DEFAULT_ELEV_REF_SCALE,
):
    """Visualises the internal physics of the Photoclinometric Loss.

    Uses the actual `PhotoclinometricLoss` class methods (`surface_normals`,
    `render_from_depth`, `_zscore`) so the visualisation is guaranteed to
    match exactly what the training loss computes. If the render ever
    changes, this figure updates automatically.

    Layout (6 columns):
        [Real Ortho] [GT DTM] [Normals] [Render (pred params)]
        [Render (GT-fit params)] [z-SSIM comparison strip]

    The last column shows, side-by-side, the z-score normalised render
    and z-score normalised ortho — i.e. exactly what the SSIM term of
    the loss operates on. If those two look structurally similar, the
    loss is well-behaved regardless of any global luminance mismatch
    in the raw render columns.

    Args:
        dataloader: DataLoader yielding batches with keys
            {image, dtm, confidence, sun_vector, intensity, ambient}.
        output_dir: Where to save the figure.
        num_samples: How many samples to render.
        loss_fn: Optional trained `PhotoclinometricLoss` instance. If
            None, a fresh one is created (L initialised to 0.5).
        lunar_lambert_weight_override: If set, temporarily forces the
            loss's Lunar-Lambert blend weight to this value for the
            visualisation only. Useful for exploring what different
            L values look like (the trained weight is restored after).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get or create a loss instance (the viz uses its methods directly)
    if loss_fn is None:
        loss_fn = PhotoclinometricLoss()
    loss_fn = loss_fn.to(device).eval()

    # Optionally override the learned L for exploratory viz
    saved_logit = None
    if lunar_lambert_weight_override is not None:
        w = float(min(max(lunar_lambert_weight_override, 1e-4), 1.0 - 1e-4))
        saved_logit = loss_fn.lunar_lambert_logit.data.clone()
        loss_fn.lunar_lambert_logit.data = torch.tensor(
            math.log(w / (1.0 - w)), device=device, dtype=loss_fn.lunar_lambert_logit.dtype,
        )

    L_used = float(loss_fn.lunar_lambert_weight.item())
    logger.info(f"Generating Loss Physics visualization for {num_samples} samples (L={L_used:.2f})...")
    output_dir.mkdir(parents=True, exist_ok=True)

    def _to_gray_display(
            x: torch.Tensor,
            mask: np.ndarray | None = None,
            pct_low: float = 2.0,
            pct_high: float = 98.0,
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
        clipped = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

        if mask is not None:
            clipped[~mask] = np.nan
        return clipped

    n_cols = 6
    scale = 4

    width_ratios = [1, 1, 1, 1, 1, 3.1]

    fig_width = sum(width_ratios) * scale * 0.85  # 0.85 multiplier prevents the figure from getting too massive
    fig_height = scale * num_samples

    fig, axes = plt.subplots(num_samples, n_cols, figsize=(fig_width, fig_height),
                             gridspec_kw={'width_ratios': width_ratios})
    if num_samples == 1:
        axes = axes[None, :]
    plt.subplots_adjust(wspace=0.08, hspace=0.08)

    try:
        count = 0
        with tqdm(total=num_samples) as pbar:
            for batch in dataloader:
                if count >= num_samples:
                    break
                B = batch["image"].shape[0]
                for i in range(B):
                    if count >= num_samples:
                        break

                    # --- Extract inputs ---
                    img = batch["image"][i: i + 1].to(device).float()
                    dtm = batch["dtm"][i: i + 1, :1].to(device).float()
                    mask = batch["confidence"][i: i + 1].to(device).float()
                    sun_vec = batch["sun_vector"][i: i + 1].to(device).float()
                    intensity = batch["intensity"][i: i + 1].to(device).float()
                    ambient = batch["ambient"][i: i + 1].to(device).float()

                    # Denormalize DTM to physical metres for rendering
                    if "residual_scale" in batch:
                        _scale = batch["residual_scale"][i: i + 1].to(device)
                        dtm_physical = GlobalLogNormalizer.denormalize_batch(dtm, _scale)
                    else:
                        dtm_physical = GlobalLogNormalizer(ref_scale).denormalize_prediction(dtm)

                    ortho_gray = img.mean(dim=1, keepdim=True) if img.shape[1] == 3 else img

                    # --- Estimate GT exposure/sun via OLS from physical DTM ---
                    sun_vec_gt, intensity_gt, ambient_gt = estimate_sun_vector_irls(
                        dtm_physical, img, mask)
                    # OLS returns (3,), scalar, scalar — reshape for render_from_depth
                    sun_vec_gt = sun_vec_gt.view(1, 3)
                    intensity_gt = intensity_gt.view(1)
                    ambient_gt = ambient_gt.view(1)

                    # --- Use the ACTUAL loss class methods with physical DTM ---
                    render, normals = loss_fn.render_from_depth(
                        dtm_physical, sun_vec, intensity, ambient,
                    )
                    render_gt, _ = loss_fn.render_from_depth(
                        dtm_physical, sun_vec_gt, intensity_gt, ambient_gt,
                    )

                    # --- z-scored versions: what SSIM actually sees ---
                    render_z = loss_fn._zscore(render, mask)
                    ortho_z = loss_fn._zscore(ortho_gray, mask)

                    # --- Displays ---
                    mask_np = mask[0, 0].cpu().numpy().astype(bool)

                    img_disp = _to_gray_display(ortho_gray, mask_np)
                    dtm_disp = _display_dtm(dtm_physical[0, 0])
                    normals_disp = np.clip(
                        (normals[0].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0, 0.0, 1.0,
                    )
                    render_disp = _to_gray_display(render, mask_np)
                    render_disp_gt = _to_gray_display(render_gt, mask_np)

                    # Side-by-side z-scored render | z-scored ortho
                    # (clip to ±3 for display, then stretch to [0,1])
                    def _zdisp(z):
                        a = z[0, 0].cpu().numpy()
                        a = np.clip(a, -3.0, 3.0)
                        return (a + 3.0) / 6.0

                    z_render_img = _zdisp(render_z)
                    z_ortho_img = _zdisp(ortho_z)
                    # Stack horizontally with a thin separator
                    residual = np.abs(render_z[0, 0].cpu().numpy() - ortho_z[0, 0].cpu().numpy())
                    # Clip residual for display (values > 2.0 represent significant structural mismatch)
                    residual_disp = np.clip(residual / 2.0, 0.0, 1.0)

                    # Mask invalid regions to background color
                    z_render_img[~mask_np] = np.nan
                    z_ortho_img[~mask_np] = np.nan
                    residual_disp[~mask_np] = np.nan

                    # Stack horizontally: Render | Ortho | Residual
                    sep = np.ones((z_render_img.shape[0], 4)) * np.nan
                    z_combined = np.concatenate([z_render_img, sep, z_ortho_img, sep, residual_disp], axis=1)

                    # --- Plotting ---
                    axes[count, 0].imshow(img_disp, cmap="gray", vmin=0, vmax=1)
                    axes[count, 1].imshow(dtm_disp, cmap="terrain")
                    axes[count, 2].imshow(normals_disp)
                    axes[count, 3].imshow(render_disp, cmap="gray", vmin=0, vmax=1)
                    axes[count, 4].imshow(render_disp_gt, cmap="gray", vmin=0, vmax=1)
                    axes[count, 5].imshow(z_combined, cmap="inferno", vmin=0, vmax=1)  # Inferno highlights errors well

                    for ax in axes[count]:
                        ax.axis("off")

                    if count == 0:
                        titles = [
                            "Real Ortho (Gray)",
                            "GT DTM",
                            "Surface Normals",
                            f"LL Render (L={L_used:.2f}, pred params)",
                            "LL Render (OLS-fit params)",
                            "z-SSIM view: Render | Ortho | $|Z_r - Z_o|$",
                        ]
                        for ax, t in zip(axes[0], titles):
                            ax.set_title(t, fontsize=11)

                    count += 1
                    pbar.update()

        save_path = output_dir / "loss_physics_inspection.png"
        save_fig(fig, save_path, bbox_inches="tight", dpi=DPI, facecolor="white")
        plt.close(fig)
        logger.info(f"Loss physics visualization saved to: {save_path}")

    finally:
        # Restore the original Lunar-Lambert weight if we overrode it
        if saved_logit is not None:
            loss_fn.lunar_lambert_logit.data = saved_logit


@torch.no_grad()
def plot_radial_sun_sweep(dataloader, loss_fn, output_dir: Path):
    """
    Generates a radial visualization with the Surface Normals in the center,
    surrounded by 8 Lunar-Lambert renders generated with different sun vectors.
    Each render includes a mini 3D sphere indicating the lighting direction.
    """
    device = next(loss_fn.parameters()).device if hasattr(loss_fn, 'parameters') else torch.device("cuda")
    loss_fn.eval()

    # Extract a single sample
    batch = next(iter(dataloader))
    dtm = batch["dtm"][0:1].to(device)
    mask = batch["confidence"][0:1].to(device)
    intensity = batch["intensity"][0:1].to(device).view(1, 1, 1, 1)
    ambient = batch["ambient"][0:1].to(device).view(1, 1, 1, 1)

    dtm = dtm.mean(dim=1, keepdim=True) if dtm.shape[1] == 3 else dtm

    # Ground Truth Sun Vector
    gt_sun = batch["sun_vector"][0].cpu().numpy()

    # Calculate Central Normals
    normals = loss_fn.surface_normals(dtm)
    normals_disp = np.clip((normals[0].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0, 0.0, 1.0)
    mask_np = mask[0, 0].cpu().numpy().astype(bool)
    normals_disp[~mask_np] = np.nan

    fig = plt.figure(figsize=(18, 18))

    # 1. Center Axes for Normals
    ax_center = fig.add_axes([0.375, 0.375, 0.25, 0.25])
    ax_center.imshow(normals_disp)
    ax_center.set_title("Surface Normals", fontsize=14, weight='bold', pad=15)
    ax_center.axis('off')

    # 2. Define the 8 circular positions
    # Top-left in standard polar coordinates is 3*pi/4 (135 degrees)
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles = (angles + 3 * np.pi / 4) % (2 * np.pi)

    radius = 0.38
    size = 0.18

    for i, angle in enumerate(angles):
        # Top-left gets the GT vector, the rest get sweeping azimuthal shifts
        if i == 0:
            current_sun = gt_sun
            title = "Correct Vector (GT)"
            border_color = '#22cc55'  # Green
            line_width = 4
        else:
            # Rotate GT sun by the swept angle relative to GT
            rot_angle = i * (2 * np.pi / 8)
            c, s = np.cos(rot_angle), np.sin(rot_angle)
            rot_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            current_sun = rot_z @ gt_sun
            title = f"Azimuth Shift +{int(np.degrees(rot_angle))}°"
            border_color = '#ff2a2a'  # Red
            line_width = 2

        sun_t = torch.tensor(current_sun, device=device, dtype=torch.float32).unsqueeze(0)

        # Render the terrain under the current lighting configuration
        render, _ = loss_fn.render_from_depth(dtm, sun_t, intensity, ambient)

        # Normalize for display
        render_disp = render[0, 0].cpu().numpy()
        valid_pixels = render_disp[mask_np]
        if valid_pixels.size > 0:
            p2, p98 = np.percentile(valid_pixels, [2, 98])
            render_disp = np.clip((render_disp - p2) / (p98 - p2 + 1e-6), 0, 1)
        render_disp[~mask_np] = np.nan

        # Calculate figure-relative coordinates
        cx = 0.5 + radius * np.cos(angle)
        cy = 0.5 + radius * np.sin(angle)

        # Plot the Render
        ax_img = fig.add_axes([cx - size / 2, cy - size / 2, size, size])
        ax_img.imshow(render_disp, cmap="gray")
        ax_img.set_title(title, fontsize=12, weight='bold' if i == 0 else 'normal')
        ax_img.axis('off')

        # Add indicative border
        rect = plt.Rectangle((0, 0), 1, 1, transform=ax_img.transAxes,
                             color=border_color, fill=False, linewidth=line_width)
        ax_img.add_patch(rect)

        # Draw radial connection line from the center normals to the render
        con = ConnectionPatch(xyA=(0.5, 0.5), xyB=(cx, cy),
                              coordsA="figure fraction", coordsB="figure fraction",
                              axesA=ax_center, axesB=ax_img, color="black", alpha=0.15, linestyle="--")
        fig.add_artist(con)

        # 3. Mini 3D Sphere Indicator (inset at the bottom right of each render)
        mini_size = size * 0.4
        ax_mini = fig.add_axes([cx + size / 2 - mini_size * 0.8, cy - size / 2 - mini_size * 0.2, mini_size, mini_size],
                               projection='3d')

        # Wireframe sphere
        u, v = np.mgrid[0:2 * np.pi:15j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax_mini.plot_wireframe(x, y, z, color='gray', alpha=0.2, linewidth=0.5)

        # Sun Vector Arrow
        ax_mini.quiver(0, 0, 0, current_sun[0], current_sun[1], current_sun[2],
                       color='orange', length=1.5, normalize=True, arrow_length_ratio=0.3, linewidth=2.5)

        # Clean up 3D axis
        ax_mini.set_xlim([-1, 1])
        ax_mini.set_ylim([-1, 1])
        ax_mini.set_zlim([-1, 1])
        ax_mini.set_axis_off()
        ax_mini.view_init(elev=30, azim=-45)

    # Global Title
    fig.suptitle('Spatial Sensitivity of Lunar-Lambert Renders to Illumination Vectors',
                 fontsize=22, weight='bold', y=0.94)

    # Save Output
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / 'radial_sun_vector_sweep.pdf'
    save_fig(fig, save_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)
    print(f"Radial visualization saved to: {save_path}")


import torch
from pathlib import Path


@torch.no_grad()
def plot_statistically_significant_landscape(
        dataloader, loss_fn, output_dir: Path, resolution: int = 50, num_batches: int = 100
):
    device = next(loss_fn.parameters()).device if hasattr(loss_fn, 'parameters') else torch.device("cuda")
    loss_fn.eval()

    # 1. Generate Canonical Hemispherical Grid (Centered at Zenith)
    theta = np.linspace(0, np.pi / 2, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution * 2)
    T, P = np.meshgrid(theta, phi)

    S_x = np.sin(T) * np.cos(P)
    S_y = np.sin(T) * np.sin(P)
    S_z = np.cos(T)
    # canonical_grid shape: (N_points, 3)
    canonical_grid = torch.tensor(np.stack([S_x, S_y, S_z], axis=-1).reshape(-1, 3), device=device, dtype=torch.float32)

    total_samples = 0
    aggregate_losses = []

    for b_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Evaluating Dataset Topology")):
        if b_idx >= num_batches: break

        img = batch["image"].to(device).float()
        dtm = batch["dtm"][:, :1].to(device).float()
        mask = batch["confidence"].to(device).float()
        ambient = batch["ambient"].to(device).float()
        intensity = batch["intensity"].to(device).float()

        # Ground truth sun vector for each item in the batch: Shape (B, 3)
        s_gt = batch["sun_vector"].to(device).float()
        B = s_gt.shape[0]
        total_samples += B

        # 2. Batched SO(3) Rotation Mapping Z-axis to s_gt
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=device).expand(B, -1)
        v = torch.cross(z_axis, s_gt, dim=1)  # (B, 3)
        c = torch.sum(z_axis * s_gt, dim=1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)

        # Construct skew-symmetric matrices (B, 3, 3)
        v_x, v_y, v_z = v[:, 0], v[:, 1], v[:, 2]
        zeros = torch.zeros_like(v_x)
        V = torch.stack([
            torch.stack([zeros, -v_z, v_y], dim=1),
            torch.stack([v_z, zeros, -v_x], dim=1),
            torch.stack([-v_y, v_x, zeros], dim=1)
        ], dim=1)

        I = torch.eye(3, device=device).expand(B, -1, -1)
        # Rodrigues' formula for rotation matrix R: (B, 3, 3)
        R = I + V + torch.bmm(V, V) / (1 + c + 1e-8)

        # Apply rotation to canonical grid to get query vectors relative to each sample
        # canonical_grid: (N_points, 3) -> (B, N_points, 3)
        s_queries = torch.einsum('bij, nj -> bni', R, canonical_grid)

        # 3. Evaluate Batched Loss
        # Expanding images to match query grid size is memory intensive.
        # Evaluate iteratively over the grid points across the entire image batch.
        N_points = canonical_grid.shape[0]
        batch_losses = torch.zeros((B, N_points), device=device)

        eval_chunk_size = 64  # Chunking grid points to save memory
        for i in range(0, N_points, eval_chunk_size):
            end_idx = min(i + eval_chunk_size, N_points)
            current_chunk_size = end_idx - i

            # Shape (B, chunk_size, 3) -> Flatten to (B * chunk_size, 3)
            s_batch = s_queries[:, i:end_idx, :].reshape(-1, 3)

            # Tile images and DTMs to match the flattened query vectors
            img_exp = img.repeat_interleave(current_chunk_size, dim=0)
            dtm_exp = dtm.repeat_interleave(current_chunk_size, dim=0)
            mask_exp = mask.repeat_interleave(current_chunk_size, dim=0)
            amb_exp = ambient.repeat_interleave(current_chunk_size)
            int_exp = intensity.repeat_interleave(current_chunk_size)

            # Evaluate Photoclinometric Loss
            chunk_loss = loss_fn(dtm_exp, img_exp, mask_exp, s_batch, amb_exp, int_exp, reduction='none')

            # Reshape back and store: (B * chunk_size) -> (B, chunk_size)
            # Assuming loss_fn with reduction='none' returns a scalar per image/vector pair
            # You may need to take the mean over spatial dimensions here if loss_fn returns spatial maps:
            if chunk_loss.dim() > 1:
                chunk_loss = chunk_loss.view(B * current_chunk_size, -1).mean(dim=1)

            batch_losses[:, i:end_idx] = chunk_loss.view(B, current_chunk_size)

        aggregate_losses.append(batch_losses.cpu())

    # 4. Compute Statistical Topography
    all_losses = torch.cat(aggregate_losses, dim=0).numpy()  # (Total_Samples, N_points)
    L_mean = np.mean(all_losses, axis=0).reshape(T.shape)
    L_sem = (np.std(all_losses, axis=0) / np.sqrt(total_samples)).reshape(T.shape)

    # 5. Lambert Azimuthal Equal-Area Projection Plotting
    R_proj = 2 * np.sin(T / 2)
    X = R_proj * np.sin(P)
    Y = R_proj * np.cos(P)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Expected Loss (The Convex Basin)
    contour_mean = axes[0].contourf(X, Y, L_mean, levels=50, cmap='viridis')
    axes[0].contour(X, Y, L_mean, levels=20, colors='black', linewidths=0.3, alpha=0.5)
    axes[0].plot(0, 0, marker='*', color='white', markersize=15, markeredgecolor='black', label=r'GT $\mathbf{s}^*$')
    axes[0].set_title(r'Expected Empirical Risk $\mathbb{E}_{x \sim \mathcal{D}}[\mathcal{L}]$')
    axes[0].legend(loc='upper right')
    fig.colorbar(contour_mean, ax=axes[0], shrink=0.7)

    # Right: Standard Error (Statistical Significance)
    contour_sem = axes[1].contourf(X, Y, L_sem, levels=50, cmap='magma')
    axes[1].plot(0, 0, marker='*', color='white', markersize=15, markeredgecolor='black')
    axes[1].set_title(r'Uncertainty of the Mean ($SEM_{\mathcal{L}}$)')
    fig.colorbar(contour_sem, ax=axes[1], shrink=0.7)

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlabel(r"Relative $X$ (LAEA Projection)")
        ax.set_ylabel(r"Relative $Y$ (LAEA Projection)")
        ax.axhline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axvline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.6)
        limit = np.sqrt(2)
        ax.set_xlim(-limit * 1.05, limit * 1.05)
        ax.set_ylim(-limit * 1.05, limit * 1.05)

        # Add outer horizon circle
        horizon = plt.Circle((0, 0), limit, color='white', fill=False, linewidth=1.5, linestyle=':')
        ax.add_patch(horizon)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_dir / 'photoclinometric_loss_significance.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)


def compute_manifold_diagnostics(
        s_star: torch.Tensor,
        grad_E: torch.Tensor,
        H_R: torch.Tensor,
        U: torch.Tensor,
        valid_mask: torch.Tensor,
) -> dict[str, list[float]]:
    """Per-sample diagnostics for the eigenspectrum analysis.

    Returns
    -------
    grad_R_norm     : tangent gradient magnitude at s_star.
    align_azimuth   : |⟨u_max, â⟩|, alignment of the dominant Hessian
                      eigenvector with the local azimuth tangent.
    align_elevation : |⟨u_max, ê⟩|, alignment with the local elevation tangent.
    """
    B = s_star.shape[0]
    device = s_star.device

    grad_R = grad_E - (grad_E * s_star).sum(dim=1, keepdim=True) * s_star
    grad_R_norm = grad_R.norm(dim=1)

    # Dominant tangent eigenvector (in tangent coords), then lifted to ambient
    eigvals, eigvecs = torch.linalg.eigh(H_R)  # ascending
    v_max = eigvecs[:, :, -1]  # (B, 2)
    u_max = torch.einsum("bij,bj->bi", U, v_max)  # (B, 3)
    u_max = F.normalize(u_max, p=2, dim=1, eps=1e-12)

    # Local azimuth and elevation tangent directions at s_star.
    # Azimuth: â ∝ e_z × s, lies in the horizontal plane and is tangent to S^2.
    # Elevation: ê = s × â completes a right-handed frame in the tangent plane.
    e_z = torch.tensor([0.0, 0.0, 1.0], device=device).expand(B, 3)
    a_hat_raw = torch.linalg.cross(e_z, s_star, dim=1)
    a_norm = a_hat_raw.norm(dim=1, keepdim=True)
    pole = a_norm.squeeze(-1) < 1e-6
    a_hat_fallback = torch.tensor([1.0, 0.0, 0.0], device=device).expand(B, 3)
    a_hat = torch.where(
        pole.unsqueeze(-1),
        a_hat_fallback,
        a_hat_raw / a_norm.clamp(min=1e-12),
    )
    e_hat = torch.linalg.cross(s_star, a_hat, dim=1)
    e_hat = F.normalize(e_hat, p=2, dim=1, eps=1e-12)

    align_az = (u_max * a_hat).sum(dim=1).abs()
    align_el = (u_max * e_hat).sum(dim=1).abs()

    return {
        "grad_R_norm": grad_R_norm[valid_mask].cpu().tolist(),
        "align_azimuth": align_az[valid_mask].cpu().tolist(),
        "align_elevation": align_el[valid_mask].cpu().tolist(),
    }


@torch.no_grad()
def plot_spherical_loss_landscape(
        loss_fn: nn.Module,
        idx: int,
        s_star: torch.Tensor, U: torch.Tensor,
        dtm, img, mask, ambient, intensity,
        output_dir: Path, filename: str,
        title_suffix: str = "",
        grid_size: int = 51, radius: float = 0.5, chunk_size: int = 64,
        grad_R_norm: float | None = None,  # NEW
        stationarity_ratio: float | None = None,  # NEW
) -> None:
    """Loss landscape on T_{s*} S^2. Now also annotates stationarity."""
    was_training = loss_fn.training
    loss_fn.eval()
    try:
        device = s_star.device

        u_axis = torch.linspace(-radius, radius, grid_size, device=device)
        U1, U2 = torch.meshgrid(u_axis, u_axis, indexing="xy")
        xi = torch.stack([U1.flatten(), U2.flatten()], dim=1)
        G = xi.shape[0]

        s_base = s_star.expand(G, 3).contiguous()
        U_b = U.expand(G, 3, 2).contiguous()
        v = torch.einsum("bij,bj->bi", U_b, xi)
        s_grid = sphere_expmap(s_base, v)

        losses: list[torch.Tensor] = []
        for i in range(0, G, chunk_size):
            s_chunk = s_grid[i: i + chunk_size]
            n = s_chunk.shape[0]
            l = loss_fn(
                dtm.expand(n, *dtm.shape[1:]).contiguous(),
                img.expand(n, *img.shape[1:]).contiguous(),
                mask.expand(n, *mask.shape[1:]).contiguous(),
                s_chunk,
                ambient.expand(n, *ambient.shape[1:]).contiguous(),
                intensity.expand(n, *intensity.shape[1:]).contiguous(),
                reduction="none",
            )
            losses.append(l.detach().cpu())
        L_grid = torch.cat(losses).reshape(grid_size, grid_size).numpy()

        fig, ax = plt.subplots(figsize=(8, 7))
        u_np = u_axis.cpu().numpy()
        pcm = ax.pcolormesh(u_np, u_np, L_grid, cmap="viridis", shading="auto")
        cs = ax.contour(u_np, u_np, L_grid, levels=20, colors="white",
                        alpha=0.6, linewidths=0.8)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.4f")
        ax.scatter([0], [0], marker="*", s=220, color="crimson",
                   edgecolor="white", linewidth=1.5,
                   label=r"Empirical Min ($s^*$)", zorder=5)
        ax.set_xlabel(r"Tangent Basis $u_1$")
        ax.set_ylabel(r"Tangent Basis $u_2$")
        ax.set_title(rf"Spherical Loss Contour on Tangent Plane $T_{{s^*}}S^2$ {title_suffix}")
        fig.colorbar(pcm, ax=ax, label=r"Photoclinometric Loss $\mathcal{L}$")

        # Stationarity readout — flags unconverged inputs at a glance
        if grad_R_norm is not None or stationarity_ratio is not None:
            txt_lines = []
            if grad_R_norm is not None:
                txt_lines.append(rf"$\Vert\nabla_R\Vert = {grad_R_norm:.2e}$")
            if stationarity_ratio is not None:
                txt_lines.append(rf"$\Vert\nabla_R\Vert / \max|\lambda| = {stationarity_ratio:.2e}$")
            ax.text(0.02, 0.02, "\n".join(txt_lines),
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.5"),
                    verticalalignment="bottom")

        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

        output_dir.mkdir(parents=True, exist_ok=True)
        save_fig(fig, output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
    finally:
        if was_training:
            loss_fn.train()


def riemannian_grad_norm(s: torch.Tensor, grad_E: torch.Tensor) -> torch.Tensor:
    """Per-sample ‖proj_{T_s S^2}(∇_E)‖_2 → shape (B,).

    For x on the unit sphere, the tangent projection is
        ∇_R = ∇_E − ⟨∇_E, x⟩ x.
    """
    grad_R = grad_E - (grad_E * s).sum(dim=1, keepdim=True) * s
    return grad_R.norm(dim=1)


def sphere_expmap(s: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Exponential map on S^2:  exp_s(v) = cos(‖v‖) s + sin(‖v‖) v / ‖v‖."""
    theta = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)
    return torch.cos(theta) * s + torch.sin(theta) * (v / theta)


def tangent_basis(s: torch.Tensor) -> torch.Tensor:
    """Orthonormal tangent basis U ∈ R^{B×3×2} at each s ∈ S^2.

    Uses cross-products with a reference axis, switching the reference
    when s is close to colinear with [1,0,0] to avoid degeneracy.
    """
    B = s.shape[0]
    device = s.device

    v_ref = torch.tensor([1.0, 0.0, 0.0], device=device).expand(B, 3)
    collinear = (s * v_ref).sum(dim=1).abs() > 0.99
    v_ref_alt = torch.tensor([0.0, 1.0, 0.0], device=device).expand(B, 3)
    v_ref = torch.where(collinear.unsqueeze(1), v_ref_alt, v_ref)

    u1 = F.normalize(torch.linalg.cross(s, v_ref, dim=1), p=2, dim=1, eps=1e-12)
    u2 = F.normalize(torch.linalg.cross(s, u1, dim=1), p=2, dim=1, eps=1e-12)
    return torch.stack([u1, u2], dim=2)


# =============================================================================
# 3. Convergence-driven Riemannian optimizer
# =============================================================================
def optimize_s_on_sphere(
        loss_fn: nn.Module,
        dtm: torch.Tensor,
        img: torch.Tensor,
        mask: torch.Tensor,
        ambient: torch.Tensor,
        intensity: torch.Tensor,
        s_init: torch.Tensor,
        *,
        # Phase 1: coarse descent
        coarse_lr: float = 0.1,
        coarse_max_steps: int = 300,
        coarse_grad_tol: float = 1e-3,
        coarse_rel_grad_tol: float = 1e-2,
        coarse_plateau_tol: float = 1e-7,
        coarse_plateau_patience: int = 15,
        # Phase 2: fine refinement
        fine_lr: float = 1e-2,
        fine_max_steps: int = 2000,
        fine_grad_tol: float = 1e-6,
        fine_loss_tol: float = 1e-10,
        fine_patience: int = 25,
        fine_lr_decay: float = 0.5,
        fine_lr_decay_patience: int = 50,
        fine_lr_min: float = 1e-6,
        verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Two-phase Riemannian minimization on S^2.

    Phase 1: RiemannianAdam (lr=coarse_lr) — fast traversal to the basin.
    Phase 2: RiemannianSGD (lr=fine_lr, no momentum) — precise descent to
             a stationary point. Adam's momentum is intentionally dropped
             here because it undermines the precision needed for valid
             second-order analysis.

    Termination is driven by the worst per-sample tangent gradient norm
    across the batch, NOT by step count. The per-sample final ‖∇_R L‖
    is returned so downstream code can gate eigenvalue classification.

    Returns
    -------
    s_final : (B, 3) tensor on S^2 (detached).
    final_grad_R : (B,) per-sample tangent gradient norms at s_final.
    info : dict with `phase{1,2}_steps`, `phase{1,2}_exit`, and history.
    """
    s = geoopt.ManifoldParameter(
        F.normalize(s_init, dim=1).clone(),
        manifold=geoopt.Sphere(),
    )

    def closure_loss() -> torch.Tensor:
        return loss_fn(dtm, img, mask, s, ambient, intensity).mean()

    info: dict[str, Any] = {
        "phase1_steps": 0, "phase2_steps": 0,
        "phase1_exit": None, "phase2_exit": None,
        "loss_history": [], "grad_history": [],
    }

    # ---- Phase 1 ----------------------------------------------------------
    opt = geoopt.optim.RiemannianAdam([s], lr=coarse_lr)
    init_grad: Optional[float] = None
    plateau = 0
    prev_loss: Optional[float] = None
    p1_break = False

    for step in range(coarse_max_steps):
        opt.zero_grad()
        loss = closure_loss()
        loss.backward()
        gn = riemannian_grad_norm(s.detach(), s.grad).max().item()

        if init_grad is None:
            init_grad = max(gn, 1e-12)

        info["loss_history"].append(loss.item())
        info["grad_history"].append(gn)

        if gn < coarse_grad_tol:
            info["phase1_exit"] = "abs_grad"
            info["phase1_steps"] = step + 1
            p1_break = True
            break
        if gn < coarse_rel_grad_tol * init_grad:
            info["phase1_exit"] = "rel_grad"
            info["phase1_steps"] = step + 1
            p1_break = True
            break
        if prev_loss is not None and abs(prev_loss - loss.item()) < coarse_plateau_tol:
            plateau += 1
            if plateau >= coarse_plateau_patience:
                info["phase1_exit"] = "loss_plateau"
                info["phase1_steps"] = step + 1
                p1_break = True
                break
        else:
            plateau = 0
        prev_loss = loss.item()

        opt.step()

    if not p1_break:
        info["phase1_exit"] = "max_steps"
        info["phase1_steps"] = coarse_max_steps

    # ---- Phase 2 ----------------------------------------------------------
    lr = fine_lr
    opt = geoopt.optim.RiemannianSGD([s], lr=lr, momentum=0.0)
    plateau = 0
    lr_plateau = 0
    prev_loss = None
    best_loss = float("inf")
    p2_break = False

    for step in range(fine_max_steps):
        opt.zero_grad()
        loss = closure_loss()
        loss.backward()
        gn = riemannian_grad_norm(s.detach(), s.grad).max().item()

        info["loss_history"].append(loss.item())
        info["grad_history"].append(gn)

        if gn < fine_grad_tol:
            info["phase2_exit"] = "grad_converged"
            info["phase2_steps"] = step + 1
            p2_break = True
            break

        if prev_loss is not None:
            dloss = abs(prev_loss - loss.item())
            if dloss < fine_loss_tol:
                plateau += 1
                if plateau >= fine_patience:
                    info["phase2_exit"] = "loss_plateau"
                    info["phase2_steps"] = step + 1
                    p2_break = True
                    break
            else:
                plateau = 0

            # Adaptive step shrinking when descent stalls
            if loss.item() >= best_loss - fine_loss_tol:
                lr_plateau += 1
                if lr_plateau >= fine_lr_decay_patience and lr > fine_lr_min:
                    lr = max(lr * fine_lr_decay, fine_lr_min)
                    for g in opt.param_groups:
                        g["lr"] = lr
                    lr_plateau = 0
                    if verbose:
                        print(f"[refine] step={step} lr→{lr:.2e} gn={gn:.2e}")
            else:
                lr_plateau = 0

        prev_loss = loss.item()
        best_loss = min(best_loss, loss.item())

        opt.step()

    if not p2_break:
        info["phase2_exit"] = "max_steps"
        info["phase2_steps"] = fine_max_steps

    # Final per-sample tangent gradient norms (used for stationarity gate)
    opt.zero_grad()
    closure_loss().backward()
    final_grad_R = riemannian_grad_norm(s.detach(), s.grad).detach()

    return s.detach(), final_grad_R, info


def prove_and_visualize_local_convexity(
        dataloader,
        loss_fn: nn.Module,
        output_dir: Path,
        num_batches: int = 200,
        stationarity_tol: float = 1e-2,
        classification_tol_rel: float = 1e-4,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Estimate local topology of the photoclinometric loss at the empirical
    minimum on S^2 across a dataset.

    Pipeline per batch
    ------------------
    1. Optimize sun vector to a stationary point on S^2 (convergence-driven).
    2. Compute Euclidean gradient ∇_E and Hessian H_E via vmap+grad+hessian.
    3. Build orthonormal tangent basis U at s*.
    4. Riemannian Hessian: H_R = U^T H_E U − ⟨∇_E, s*⟩ I_2.
    5. Eigendecompose H_R.
    6. Stationarity gate:
            ‖∇_R L(s*)‖ / max|λ(H_R)| < stationarity_tol
       Samples failing the gate are reported as 'unconverged' and excluded
       from topological statistics.
    7. Tolerance-based topological classification (Index 0 / 1 / 2).

    Parameters
    ----------
    stationarity_tol :
        Maximum allowed ratio of ‖∇_R‖ to the dominant Hessian eigenvalue
        for a sample to be classifiable. Default 1e-2 means the linear
        term of the local Taylor expansion is at least 100× smaller than
        the quadratic term.
    classification_tol_rel :
        Eigenvalue tolerance, relative to max|λ|, for distinguishing
        positive/negative/zero curvature. Default 1e-4.
    """
    optimizer_kwargs = optimizer_kwargs or {}

    device = (
        next(loss_fn.parameters()).device
        if any(True for _ in loss_fn.parameters())
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    loss_fn.eval()

    # Vectorized first- and second-order operators over the batch
    def sample_loss_fn(s_vec, d_i, img_i, m_i, amb_i, int_i):
        return loss_fn(
            d_i.unsqueeze(0), img_i.unsqueeze(0), m_i.unsqueeze(0),
            s_vec.unsqueeze(0), amb_i.unsqueeze(0), int_i.unsqueeze(0),
        ).mean()

    compute_batch_grad = vmap(grad(sample_loss_fn), in_dims=(0, 0, 0, 0, 0, 0))
    compute_batch_hess = vmap(hessian(sample_loss_fn), in_dims=(0, 0, 0, 0, 0, 0))

    # Aggregators
    eigenvalues_min: list[float] = []
    eigenvalues_max: list[float] = []
    angular_shifts: list[float] = []
    stationarity_ratios: list[float] = []
    unconverged_grad_norms: list[float] = []

    all_grad_R_norms: list[float] = []
    all_align_azimuth: list[float] = []
    all_align_elevation: list[float] = []

    rep_convex: Optional[dict[str, torch.Tensor]] = None
    rep_saddle: Optional[dict[str, torch.Tensor]] = None

    n_total = 0
    n_unconverged = 0

    for b_idx, batch in enumerate(tqdm(dataloader, total=num_batches,
                                       desc="Computing Eigenspectra")):
        if b_idx >= num_batches:
            break

        # 1. Inputs
        img = batch["image"].to(device).float().detach()
        dtm = batch["dtm"][:, :1].to(device).float().detach()
        mask = batch["confidence"].to(device).float().detach()
        ambient = batch["ambient"].to(device).float().detach()
        intensity = batch["intensity"].to(device).float().detach()
        s_gt = batch["sun_vector"].to(device).float().detach()

        B = s_gt.shape[0]

        # 2. Initialization at the upper-hemisphere pole, then convergence-driven optimization
        s_init = torch.zeros_like(s_gt)
        s_init[:, -1] = 1.0
        s_init = F.normalize(s_init, p=2, dim=1)

        s_true, final_grad_R, opt_info = optimize_s_on_sphere(
            loss_fn, dtm, img, mask, ambient, intensity, s_init,
            **optimizer_kwargs,
        )

        # 3. Euclidean gradient and Hessian at s*
        with torch.enable_grad():
            grad_E = compute_batch_grad(s_true, dtm, img, mask, ambient, intensity)  # (B, 3)
            H_E = compute_batch_hess(s_true, dtm, img, mask, ambient, intensity)  # (B, 3, 3)

        # 4. Tangent basis and Riemannian Hessian
        U = tangent_basis(s_true)  # (B, 3, 2)
        U_T = U.transpose(1, 2)  # (B, 2, 3)
        H_E_proj = torch.bmm(torch.bmm(U_T, H_E), U)  # (B, 2, 2)
        radial_grad = (grad_E * s_true).sum(dim=1)  # (B,)
        I_2 = torch.eye(2, device=device).expand(B, 2, 2)
        H_R = H_E_proj - radial_grad.view(B, 1, 1) * I_2  # (B, 2, 2)

        # 5. Eigendecomposition
        eigvals = torch.linalg.eigvalsh(H_R)  # (B, 2) ascending

        # 6. Validity & stationarity gate
        finite_mask = (
                ~torch.isnan(eigvals[:, 0]) & ~torch.isinf(eigvals[:, 0])
                & ~torch.isnan(final_grad_R) & ~torch.isinf(final_grad_R)
        )
        hess_scale = eigvals.abs().max(dim=1).values.clamp(min=1e-12)
        stationarity = final_grad_R / hess_scale  # (B,)
        is_stationary = stationarity < stationarity_tol

        valid_mask_tensor = finite_mask & is_stationary
        unconv_mask = finite_mask & ~is_stationary

        n_total += int(finite_mask.sum().item())
        n_unconverged += int(unconv_mask.sum().item())

        # 7. Aggregate
        eigenvalues_min.extend(eigvals[valid_mask_tensor, 0].cpu().tolist())
        eigenvalues_max.extend(eigvals[valid_mask_tensor, 1].cpu().tolist())
        stationarity_ratios.extend(stationarity[finite_mask].cpu().tolist())
        unconverged_grad_norms.extend(final_grad_R[unconv_mask].cpu().tolist())

        cos_sim = torch.sum(s_true * s_gt, dim=1).clamp(-1.0, 1.0)
        shift_deg = torch.acos(cos_sim) * (180.0 / np.pi)
        angular_shifts.extend(shift_deg[valid_mask_tensor].cpu().tolist())

        diagnostics = compute_manifold_diagnostics(s_true, grad_E, H_R, U, valid_mask_tensor)
        all_grad_R_norms.extend(diagnostics["grad_R_norm"])
        all_align_azimuth.extend(diagnostics["align_azimuth"])
        all_align_elevation.extend(diagnostics["align_elevation"])

        # 8. Cache representative convex / saddle points (stationary only!)
        # Tolerance-based classification with per-sample scale
        scale = eigvals[:, 1].abs().clamp_min(1e-8)
        tol_per = classification_tol_rel * scale

        if rep_convex is None:
            convex_mask = valid_mask_tensor & (eigvals[:, 0] > tol_per)
            if convex_mask.any():
                idx_pick = torch.where(convex_mask)[0][0]
                rep_convex = {
                    "s_true": s_true[idx_pick].cpu(),
                    "U": U[idx_pick].cpu(),
                    "dtm": dtm[idx_pick: idx_pick + 1].cpu(),
                    "img": img[idx_pick: idx_pick + 1].cpu(),
                    "mask": mask[idx_pick: idx_pick + 1].cpu(),
                    "ambient": ambient[idx_pick: idx_pick + 1].cpu(),
                    "intensity": intensity[idx_pick: idx_pick + 1].cpu(),
                }

        if rep_saddle is None:
            saddle_mask = (
                    valid_mask_tensor
                    & (eigvals[:, 0] < -tol_per)
                    & (eigvals[:, 1] > tol_per)
            )
            if saddle_mask.any():
                idx_pick = torch.where(saddle_mask)[0][0]
                rep_saddle = {
                    "s_true": s_true[idx_pick].cpu(),
                    "U": U[idx_pick].cpu(),
                    "dtm": dtm[idx_pick: idx_pick + 1].cpu(),
                    "img": img[idx_pick: idx_pick + 1].cpu(),
                    "mask": mask[idx_pick: idx_pick + 1].cpu(),
                    "ambient": ambient[idx_pick: idx_pick + 1].cpu(),
                    "intensity": intensity[idx_pick: idx_pick + 1].cpu(),
                    "grad_R_norm": final_grad_R[idx_pick].item(),
                    "stationarity": stationarity[idx_pick].item(),
                }

    # ---------------------------------------------------------------------
    # Aggregation & validation
    # ---------------------------------------------------------------------
    L_min = np.array(eigenvalues_min)
    L_max = np.array(eigenvalues_max)
    shifts = np.array(angular_shifts)
    grad_norms = np.array(all_grad_R_norms)
    azimuth_aligns = np.array(all_align_azimuth)
    elevation_aligns = np.array(all_align_elevation)
    stat_ratios = np.array(stationarity_ratios)

    print(f"\n--- Stationarity Diagnostic ---")
    print(f"Total finite samples:                    {n_total}")
    print(f"Unconverged (excluded from topology):    {n_unconverged}"
          f"  ({100.0 * n_unconverged / max(n_total, 1):.2f}%)")
    if len(stat_ratios) > 0:
        print(f"Stationarity ratio  median:              {np.median(stat_ratios):.2e}")
        print(f"Stationarity ratio  p99:                 {np.quantile(stat_ratios, 0.99):.2e}")
    if n_unconverged > 0:
        u_arr = np.array(unconverged_grad_norms)
        print(f"Unconverged ‖∇_R‖   median:              {np.median(u_arr):.2e}")
        print(f"Unconverged ‖∇_R‖   max:                 {np.max(u_arr):.2e}")

    print(f"\n--- Topology Diagnostic (classifiable subset) ---")
    print(f"Classifiable samples:                    {len(L_min)}")
    negative_lambdas = L_min[L_min < 0]
    if len(negative_lambdas) > 0:
        print(f"Non-convex samples among classifiable:   {len(negative_lambdas)}")
        print(f"Mean negative magnitude:                 {np.mean(negative_lambdas):.2e}")
        print(f"Worst-case lambda_min:                   {np.min(negative_lambdas):.2e}")

    # Topological categorization
    scale_glob = np.maximum(np.abs(L_max), 1e-8)
    tol_glob = classification_tol_rel * scale_glob
    idx_convex = int(np.sum((L_min > tol_glob) & (L_max > tol_glob)))
    idx_saddle = int(np.sum((L_min < -tol_glob) & (L_max > tol_glob)))
    idx_concave = int(np.sum((L_min < -tol_glob) & (L_max < -tol_glob)))
    total_valid = len(L_min)

    # Bootstrap CI for convexity ratio
    convex_mean: float = float("nan")
    ci_lower: float = float("nan")
    ci_upper: float = float("nan")
    if total_valid > 0:
        rng = np.random.default_rng(42)
        bootstrapped_ratios = []
        for _ in range(1000):
            resample = rng.choice(L_min, size=total_valid, replace=True)
            bootstrapped_ratios.append(np.mean(resample > 0) * 100)
        convex_mean = float(np.mean(bootstrapped_ratios))
        ci_lower = float(np.percentile(bootstrapped_ratios, 2.5))
        ci_upper = float(np.percentile(bootstrapped_ratios, 97.5))

    # ---------------------------------------------------------------------
    # Contour maps from cached representatives
    # ---------------------------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if rep_convex is not None:
        rc = {k: v.to(device) for k, v in rep_convex.items()}
        plot_spherical_loss_landscape(
            loss_fn, 0,
            rc["s_true"].unsqueeze(0), rc["U"].unsqueeze(0),
            rc["dtm"], rc["img"], rc["mask"], rc["ambient"], rc["intensity"],
            output_dir, filename="spherical_contour_convex.pdf",
            title_suffix="(Index 0: Local Minimum)",
        )

    if rep_saddle is not None:
        rs = {k: v.to(device) for k, v in rep_saddle.items()}
        plot_spherical_loss_landscape(
            loss_fn, 0,
            rs["s_true"].unsqueeze(0), rs["U"].unsqueeze(0),
            rs["dtm"], rs["img"], rs["mask"], rs["ambient"], rs["intensity"],
            output_dir, filename="spherical_contour_saddle.pdf",
            title_suffix="(Index 1: Saddle Point)",
            grad_R_norm=rs["grad_R_norm"],
            stationarity_ratio=rs["stationarity"],
        )
    else:
        print("No saddle points detected among classifiable samples.")

    # ---------------------------------------------------------------------
    # Summary figure
    # ---------------------------------------------------------------------
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    # A. Eigenspectrum CDF
    if total_valid > 0:
        sns.ecdfplot(L_min, ax=axes[0, 0], color="blue",
                     label=r"$\lambda_{min}(H_R)$", linewidth=2)
        sns.ecdfplot(L_max, ax=axes[0, 0], color="red",
                     label=r"$\lambda_{max}(H_R)$", linewidth=2, linestyle="--")
        axes[0, 0].axvline(0, color="black", linestyle=":", linewidth=2)
        axes[0, 0].set_xlim([np.percentile(L_min, 1), np.percentile(L_max, 99)])
        axes[0, 0].legend()
    axes[0, 0].set_title(r"Empirical CDF of Riemannian Eigenspectrum")
    axes[0, 0].set_xlabel("Eigenvalue Magnitude")
    axes[0, 0].set_ylabel("Cumulative Probability")

    # B. Topology counts
    categories = [
        "Strictly Convex\n(Index 0)",
        "Saddle Point\n(Index 1)",
        "Strictly Concave\n(Index 2)",
    ]
    counts = [idx_convex, idx_saddle, idx_concave]
    ax_b = sns.barplot(
        x=categories, y=counts, hue=categories, ax=axes[0, 1],
        palette=["#2ecc71", "#f1c40f", "#e74c3c"], legend=False,
    )
    for container in ax_b.containers:
        ax_b.bar_label(container, fontsize=10, padding=3)
    axes[0, 1].set_title(r"Local Topology at Empirical Minimum $\mathbf{s}^*$")
    axes[0, 1].set_ylabel("Sample Count")
    if total_valid > 0:
        axes[0, 1].text(
            0, max(idx_convex, 1) * 0.5,
            rf"{convex_mean:.1f}%" "\n" rf"95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]",
            ha="center", va="center", color="black", fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

    # C. Convex-basin anisotropy
    convex_mask = L_min > 0
    if np.any(convex_mask):
        cond_numbers = L_max[convex_mask] / L_min[convex_mask]
        log_cond = np.log10(cond_numbers + 1e-12)
        sns.histplot(log_cond, ax=axes[0, 2], bins=40, color="purple", kde=True)
        axes[0, 2].set_title(r"Log-Anisotropy of Convex Basins ($\log_{10}\kappa$)")
        axes[0, 2].set_xlabel(r"$\log_{10}(\lambda_{max}/\lambda_{min})$")
        axes[0, 2].set_ylabel("Count")
    else:
        axes[0, 2].text(0.5, 0.5, "No strictly convex\nsamples detected.",
                        ha="center", va="center")

    # D. Tangent gradient norms
    if len(grad_norms) > 0:
        sns.histplot(grad_norms, ax=axes[1, 0], bins=40, color="crimson", kde=True)
        axes[1, 0].axvline(np.median(grad_norms), color="black", linestyle="--",
                           label=f"Median: {np.median(grad_norms):.1e}")
        axes[1, 0].legend()
    axes[1, 0].set_title(r"Riemannian Gradient Norms $\Vert\nabla_R\mathcal{L}\Vert_2$")
    axes[1, 0].set_xlabel(r"$\Vert U^T \nabla_E\mathcal{L}(s^*)\Vert_2$")
    axes[1, 0].set_ylabel("Sample Count")

    # E. Principal curvature alignment
    if len(azimuth_aligns) > 0:
        sns.scatterplot(x=azimuth_aligns, y=elevation_aligns,
                        ax=axes[1, 1], alpha=0.6, color="indigo")
        axes[1, 1].plot([0, 1], [1, 0], "k--", alpha=0.3)
    axes[1, 1].set_title(r"Alignment of $u_{max}$ with Physical Axes")
    axes[1, 1].set_xlabel(r"Azimuth Alignment $|\langle u_{max},\hat a\rangle|$")
    axes[1, 1].set_ylabel(r"Elevation Alignment $|\langle u_{max},\hat e\rangle|$")
    axes[1, 1].set_xlim(0, 1.05)
    axes[1, 1].set_ylim(0, 1.05)

    # F. Angular shift between s_gt and s*
    if len(shifts) > 0:
        sns.histplot(shifts, ax=axes[1, 2], bins=40, color="teal", kde=True)
    axes[1, 2].set_title(r"Deviation: $\mathbf{s}_{gt}$ vs Empirical $\mathbf{s}^*$")
    axes[1, 2].set_xlabel("Angular Shift (Degrees)")
    axes[1, 2].set_ylabel("Count")

    plt.tight_layout()
    save_fig(fig, output_dir / "eigenspectrum_convexity_proof.pdf",
             dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_total": n_total,
        "n_unconverged": n_unconverged,
        "n_classifiable": total_valid,
        "lambda_min_avg": float(np.mean(L_min)) if total_valid > 0 else float("nan"),
        "convex_ratio_mean": convex_mean,
        "convex_ratio_ci": (ci_lower, ci_upper),
        "mean_angular_shift": float(np.mean(shifts)) if len(shifts) > 0 else float("nan"),
        "median_grad_R_norm": float(np.median(grad_norms)) if len(grad_norms) > 0 else float("nan"),
        "median_stationarity": float(np.median(stat_ratios)) if len(stat_ratios) > 0 else float("nan"),
        "topology_counts": {
            "convex": idx_convex,
            "saddle": idx_saddle,
            "concave": idx_concave,
        },
    }


def compute_distance_correlation_gpu(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Computes the Distance Correlation (dCor) between two tensors on the GPU.
    A 50k x 50k matrix requires ~10GB VRAM, which fits easily on an A6000.
    """
    n = X.size(0)

    # Process X
    a = torch.cdist(X, X, p=2.0)
    a -= a.mean(dim=1, keepdim=True)
    a -= a.mean(dim=0, keepdim=True)
    dcov2_xx = (a * a).mean()

    # Process Y
    b = torch.cdist(Y, Y, p=2.0)
    b -= b.mean(dim=1, keepdim=True)
    b -= b.mean(dim=0, keepdim=True)
    dcov2_yy = (b * b).mean()

    # 3. Compute squared distance covariances
    dcov2_xy = (a * b).mean()

    del a, b
    torch.cuda.empty_cache()

    # 4. Compute distance correlation
    # Add a small epsilon to prevent division by zero in perfectly uniform edge cases
    dcor = torch.sqrt(dcov2_xy) / torch.sqrt(torch.sqrt(dcov2_xx * dcov2_yy) + 1e-8)

    return dcor.item()


def extract_content_agnostic_features(residual_2d, num_bins=30):
    """
    Collapses a 2D spatial residual into a translation-invariant statistical feature vector
    using distribution quantiles of the error and its spatial gradients.
    """
    q_points = np.linspace(0, 100, num_bins)

    # 1. Error Magnitude Distribution (Captures the overall shape of the error)
    res_quantiles = np.percentile(residual_2d, q_points)

    # 2. Error Gradient Distribution (Captures edge-error density regardless of spatial location)
    gy, gx = np.gradient(residual_2d)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_quantiles = np.percentile(grad_mag, q_points)

    # Concatenate into a 1D content-agnostic feature vector
    return np.concatenate([res_quantiles, grad_quantiles])


@torch.no_grad()
def plot_global_umap_invariance(dataloader, loss_fn, output_dir: Path, num_terrains: int = 100,
                                samples_per_terrain: int = 500):
    """
    Projects high-dimensional space via UMAP across MULTIPLE terrains by extracting
    content-agnostic spatial features, computing formal correlation statistics.
    """
    device = next(loss_fn.parameters()).device
    loss_fn.eval()

    all_features = []
    all_angular_errors = []

    # Track raw nuisance parameters separately for unbiased regression
    all_I_errors = []
    all_A_errors = []
    all_combined_nuisance = []

    data_iter = iter(dataloader)

    # Process multiple base topologies to prove global invariance
    for terrain_idx in tqdm(range(num_terrains), desc="Terrains"):
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        dtm = batch["dtm"][0:1].to(device)
        ortho = batch["image"][0:1].to(device)
        mask = batch["confidence"][0:1].to(device)

        gt_sun = batch["sun_vector"][0].cpu().numpy()
        gt_I = batch["intensity"][0].item()
        gt_A = batch["ambient"][0].item()

        np.random.seed(42 + terrain_idx)
        sun_perturb = np.random.randn(samples_per_terrain, 3)
        sun_perturb /= np.linalg.norm(sun_perturb, axis=1, keepdims=True)

        I_perturb = np.random.uniform(gt_I * 0.1, gt_I * 10.0, samples_per_terrain)
        A_perturb = np.random.uniform(gt_A - 0.5, gt_A + 0.5, samples_per_terrain)

        # Compute ground truth errors
        angular_errors = np.arccos(np.clip(np.sum(sun_perturb * gt_sun, axis=1), -1.0, 1.0))

        delta_I = np.abs(I_perturb - gt_I)
        delta_A = np.abs(A_perturb - gt_A)

        # Standardize before combining for the visual plot
        z_I = (delta_I - np.mean(delta_I)) / (np.std(delta_I) + 1e-8)
        z_A = (delta_A - np.mean(delta_A)) / (np.std(delta_A) + 1e-8)
        combined_nuisance = np.sqrt(z_I ** 2 + z_A ** 2)

        all_angular_errors.extend(angular_errors)
        all_I_errors.extend(delta_I)
        all_A_errors.extend(delta_A)
        all_combined_nuisance.extend(combined_nuisance)

        dtm = dtm.mean(dim=1, keepdim=True) if dtm.shape[1] == 3 else dtm
        ortho_gray = ortho.mean(dim=1, keepdim=True) if ortho.shape[1] == 3 else ortho
        ortho_z = loss_fn._zscore(ortho_gray, mask).cpu().numpy().squeeze()

        # Generate manifold points for this specific terrain
        for i in tqdm(range(samples_per_terrain), desc=f"Terrain {terrain_idx + 1}/{num_terrains}", leave=False):
            s_t = torch.tensor(sun_perturb[i:i + 1], device=device, dtype=torch.float32)
            I_t = torch.tensor([I_perturb[i]], device=device, dtype=torch.float32).view(1, 1, 1, 1)
            A_t = torch.tensor([A_perturb[i]], device=device, dtype=torch.float32).view(1, 1, 1, 1)

            render, _ = loss_fn.render_from_depth(dtm, s_t, I_t, A_t)
            render_z = loss_fn._zscore(render, mask)

            residual_2d = np.abs(render_z.cpu().numpy().squeeze() - ortho_z)
            agnostic_feats = extract_content_agnostic_features(residual_2d)
            all_features.append(agnostic_feats)

    # --- Convert to Tensors and Arrays ---
    features = np.stack(all_features)
    angular_errors_arr = np.array(all_angular_errors)
    I_errors_arr = np.array(all_I_errors)
    A_errors_arr = np.array(all_A_errors)
    combined_nuisance_arr = np.array(all_combined_nuisance)

    # --- 1. MATHEMATICAL PROOF: GPU Distance Correlation ---
    print("\nComputing GPU-Accelerated Distance Correlations (dCor)...")
    feats_tensor = torch.tensor(features, dtype=torch.float32, device=device)

    # We test features vs Angular Error (Expected: High dCor)
    ang_tensor = torch.tensor(angular_errors_arr, dtype=torch.float32, device=device).unsqueeze(1)
    dcor_ang = compute_distance_correlation_gpu(feats_tensor, ang_tensor)

    # We test features vs Nuisance (Expected: Near-Zero dCor)
    # Stacking Intensity and Ambient as a 2D target vector for simultaneous testing
    nuisance_tensor = torch.tensor(np.stack([I_errors_arr, A_errors_arr], axis=1),
                                   dtype=torch.float32, device=device)
    dcor_nuisance = compute_distance_correlation_gpu(feats_tensor, nuisance_tensor)

    # --- 2. PRACTICAL ML PROOF: Non-Linear Probing (R^2) ---
    print("Evaluating Disentanglement via Predictive Power (R^2)...")

    def compute_r2_gpu(X_tensor, y_tensor, cv=5):
        """
        Memory-optimized cross-validation loop natively computing R2 on the GPU.
        Uses QuantileDMatrix and inplace_predict to prevent VRAM spikes.
        """
        # Ensure memory is contiguous for C-backend compatibility
        X = X_tensor.contiguous()
        y = y_tensor.contiguous().squeeze()  # Ensure y is 1D

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        r2_scores = []
        indices = np.arange(X.size(0))

        params = {
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'device': 'cuda',
            'max_depth': 8,
            'random_state': 42,
            'verbosity': 0
        }

        for train_idx, test_idx in kf.split(indices):
            # 1. Slice directly on the GPU (No CPU transfers)
            X_train = X[torch.tensor(train_idx, device=X.device)]
            X_test = X[torch.tensor(test_idx, device=X.device)]
            y_train = y[torch.tensor(train_idx, device=y.device)]
            y_test = y[torch.tensor(test_idx, device=y.device)]

            # 2. Memory Efficient DMatrix: Compress directly from GPU memory
            dtrain = xgb.QuantileDMatrix(X_train, label=y_train)

            # 3. Train
            bst = xgb.train(params, dtrain, num_boost_round=100)

            # 4. Inplace predict keeps data on the GPU natively
            preds = bst.inplace_predict(X_test)

            if not isinstance(preds, torch.Tensor):
                preds = torch.as_tensor(preds, device=X.device)

            # 5. Compute R^2 natively on GPU (avoids sklearn CPU transfers)
            ss_res = torch.sum((y_test - preds) ** 2)
            ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)
            r2 = 1.0 - (ss_res / ss_tot)
            r2_scores.append(r2.item())

            # 6. Aggressive explicit memory cleanup
            del dtrain, bst, X_train, X_test, y_train, y_test, preds
            torch.cuda.empty_cache()

        return np.mean(r2_scores)

    # Initialize Nuisance Target Tensors (Angular is already a tensor from earlier)
    I_tensor = torch.tensor(I_errors_arr, dtype=torch.float32, device=device)
    A_tensor = torch.tensor(A_errors_arr, dtype=torch.float32, device=device)

    # Compute memory-safe CV
    r2_ang = compute_r2_gpu(feats_tensor, ang_tensor)
    r2_I = compute_r2_gpu(feats_tensor, I_tensor)
    r2_A = compute_r2_gpu(feats_tensor, A_tensor)

    # --- UMAP Visual Projection ---
    print("Fitting Global UMAP...")
    reducer = cuml.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42, verbose=True)
    embedding = reducer.fit_transform(features)

    # --- PLOTTING ---
    fig = plt.figure(figsize=(20, 9))

    # Plot 1: Angular Error (The Intended Signal)
    ax1 = fig.add_subplot(121)
    sns.kdeplot(x=embedding[:, 0], y=embedding[:, 1], fill=True, cmap="Reds", alpha=0.3, ax=ax1)
    sc1 = ax1.scatter(embedding[:, 0], embedding[:, 1], c=np.degrees(angular_errors_arr),
                      cmap='inferno', s=10, alpha=0.9, edgecolor='none')

    ang_title = (r"Global Angular Error ($\gamma$) Topology" + "\n" +
                 f"Predictive Power ($R^2$): {r2_ang:.3f} | Distance Corr: {dcor_ang:.3f}")
    ax1.set_title(ang_title, fontsize=14, pad=15)
    fig.colorbar(sc1, ax=ax1, label='Degrees')

    # Plot 2: Nuisance Parameter (The Entanglement Check)
    ax2 = fig.add_subplot(122)
    sns.kdeplot(x=embedding[:, 0], y=embedding[:, 1], fill=True, cmap="Blues", alpha=0.3, ax=ax2)
    sc2 = ax2.scatter(embedding[:, 0], embedding[:, 1], c=combined_nuisance_arr,
                      cmap='viridis', s=10, alpha=0.9, edgecolor='none')

    nuis_title = (r"Global Nuisance Parameter Invariance" + "\n" +
                  f"Predictive Power ($R^2$): I={r2_I:.3f}, A={r2_A:.3f} | Distance Corr: {dcor_nuisance:.3f}")
    ax2.set_title(nuis_title, fontsize=14, pad=15)
    fig.colorbar(sc2, ax=ax2, label='Standardized Combined Error (Z-Score)')

    plt.suptitle(f'Global Manifold Projection ({num_terrains} Terrains, {num_terrains * samples_per_terrain} Samples)',
                 fontsize=18, y=1.05)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_dir / 'umap_global_disentanglement.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"\n--- Disentanglement Report ---")
    print(f"Angular Error - R^2: {r2_ang:.3f}, dCor: {dcor_ang:.3f} (Ideal: High)")
    print(f"Nuisance (I)  - R^2: {r2_I:.3f} (Ideal: ~0.0)")
    print(f"Nuisance (A)  - R^2: {r2_A:.3f} (Ideal: ~0.0)")
    print(f"Combined Nuisance dCor: {dcor_nuisance:.3f} (Ideal: ~0.0)")


@torch.no_grad()
def plot_umap_invariance(dataloader, loss_fn, output_dir: Path, num_samples: int = 1500):
    """
    Projects high-dimensional space via UMAP and computes formal correlation
    statistics (dCor and R^2) to prove shift-invariance on a single terrain.
    """
    device = next(loss_fn.parameters()).device
    loss_fn.eval()

    batch = next(iter(dataloader))
    dtm = batch["dtm"][0:1].to(device)
    ortho = batch["image"][0:1].to(device)
    mask = batch["confidence"][0:1].to(device)

    gt_sun = batch["sun_vector"][0].cpu().numpy()
    gt_I = batch["intensity"][0].item()
    gt_A = batch["ambient"][0].item()

    np.random.seed(42)
    sun_perturb = np.random.randn(num_samples, 3)
    sun_perturb /= np.linalg.norm(sun_perturb, axis=1, keepdims=True)

    I_perturb = np.random.uniform(gt_I * 0.1, gt_I * 10.0, num_samples)
    A_perturb = np.random.uniform(gt_A - 0.5, gt_A + 0.5, num_samples)

    # Compute ground truth errors
    angular_errors = np.arccos(np.clip(np.sum(sun_perturb * gt_sun, axis=1), -1.0, 1.0))

    # Separate nuisance analysis
    delta_I = np.abs(I_perturb - gt_I)
    delta_A = np.abs(A_perturb - gt_A)

    # Standardize for combined visualization
    z_I = (delta_I - np.mean(delta_I)) / (np.std(delta_I) + 1e-8)
    z_A = (delta_A - np.mean(delta_A)) / (np.std(delta_A) + 1e-8)
    combined_nuisance = np.sqrt(z_I ** 2 + z_A ** 2)

    features = []
    dtm = dtm.mean(dim=1, keepdim=True) if dtm.shape[1] == 3 else dtm
    ortho_gray = ortho.mean(dim=1, keepdim=True) if ortho.shape[1] == 3 else ortho
    ortho_z = loss_fn._zscore(ortho_gray, mask).cpu().numpy().squeeze()

    for i in tqdm(range(num_samples), desc="Generating UMAP Manifold"):
        s_t = torch.tensor(sun_perturb[i:i + 1], device=device, dtype=torch.float32)
        I_t = torch.tensor([I_perturb[i]], device=device, dtype=torch.float32).view(1, 1, 1, 1)
        A_t = torch.tensor([A_perturb[i]], device=device, dtype=torch.float32).view(1, 1, 1, 1)

        render, _ = loss_fn.render_from_depth(dtm, s_t, I_t, A_t)
        render_z = loss_fn._zscore(render, mask)

        # Note: Depending on image size, this flattened array can be very large.
        # The 1TB RAM easily handles the Random Forest, and the A6000 handles the dCor pairwise distances.
        residual = np.abs(render_z.cpu().numpy().squeeze() - ortho_z)
        agnostic_feats = extract_content_agnostic_features(residual)
        features.append(agnostic_feats)

    features = np.stack(features)

    # --- 1. MATHEMATICAL PROOF: GPU Distance Correlation ---
    print("\nComputing GPU-Accelerated Distance Correlations (dCor)...")
    feats_tensor = torch.tensor(features, dtype=torch.float32, device=device)

    ang_tensor = torch.tensor(angular_errors, dtype=torch.float32, device=device).unsqueeze(1)
    dcor_ang = compute_distance_correlation_gpu(feats_tensor, ang_tensor)

    nuisance_tensor = torch.tensor(np.stack([delta_I, delta_A], axis=1),
                                   dtype=torch.float32, device=device)
    dcor_nuisance = compute_distance_correlation_gpu(feats_tensor, nuisance_tensor)

    # --- 2. PRACTICAL ML PROOF: Non-Linear Probing (R^2) ---
    print("Evaluating Disentanglement via Predictive Power (R^2)...")

    def compute_r2_gpu(X_tensor, y_tensor, cv=5):
        """
        Memory-optimized cross-validation loop natively computing R2 on the GPU.
        Uses QuantileDMatrix and inplace_predict to prevent VRAM spikes.
        """
        # Ensure memory is contiguous for C-backend compatibility
        X = X_tensor.contiguous()
        y = y_tensor.contiguous().squeeze()  # Ensure y is 1D

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        r2_scores = []
        indices = np.arange(X.size(0))

        params = {
            'objective': 'reg:squarederror',
            # 'tree_method': 'hist',
            'device': 'cuda',
            'max_depth': 8,
            'random_state': 42,
            'verbosity': 0
        }

        for train_idx, test_idx in kf.split(indices):
            # 1. Slice directly on the GPU (No CPU transfers)
            X_train = X[torch.tensor(train_idx, device=X.device)]
            X_test = X[torch.tensor(test_idx, device=X.device)]
            y_train = y[torch.tensor(train_idx, device=y.device)]
            y_test = y[torch.tensor(test_idx, device=y.device)]

            # 2. Memory Efficient DMatrix: Compress directly from GPU memory
            dtrain = xgb.QuantileDMatrix(X_train, label=y_train)

            # 3. Train
            bst = xgb.train(params, dtrain, num_boost_round=100)

            # 4. Inplace predict keeps data on the GPU natively
            preds = bst.inplace_predict(X_test)

            if not isinstance(preds, torch.Tensor):
                preds = torch.as_tensor(preds, device=X.device)

            # 5. Compute R^2 natively on GPU (avoids sklearn CPU transfers)
            ss_res = torch.sum((y_test - preds) ** 2)
            ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)
            r2 = 1.0 - (ss_res / ss_tot)
            r2_scores.append(r2.item())

            # 6. Aggressive explicit memory cleanup
            del dtrain, bst, X_train, X_test, y_train, y_test, preds
            torch.cuda.empty_cache()

        return np.mean(r2_scores)

    # Initialize Nuisance Target Tensors (Angular is already a tensor from earlier)
    I_tensor = torch.tensor(delta_I, dtype=torch.float32, device=device)
    A_tensor = torch.tensor(delta_A, dtype=torch.float32, device=device)

    # Compute memory-safe CV
    r2_ang = compute_r2_gpu(feats_tensor, ang_tensor)
    r2_I = compute_r2_gpu(feats_tensor, I_tensor)
    r2_A = compute_r2_gpu(feats_tensor, A_tensor)

    # --- UMAP Visual Projection ---
    print("Fitting Local UMAP...")
    reducer = cuml.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42, verbose=True)
    embedding = reducer.fit_transform(features)

    # --- PLOTTING ---
    fig = plt.figure(figsize=(20, 9))

    # Angular Error Plot
    ax1 = fig.add_subplot(121)
    sns.kdeplot(x=embedding[:, 0], y=embedding[:, 1], fill=True, cmap="Reds", alpha=0.3, ax=ax1)
    sc1 = ax1.scatter(embedding[:, 0], embedding[:, 1], c=np.degrees(angular_errors),
                      cmap='inferno', s=15, alpha=0.9, edgecolor='none')

    ang_title = (r"Angular Error ($\gamma$) Topology" + "\n" +
                 f"Predictive Power ($R^2$): {r2_ang:.3f} | Distance Corr: {dcor_ang:.3f}")
    ax1.set_title(ang_title, fontsize=14, pad=15)
    fig.colorbar(sc1, ax=ax1, label='Degrees')

    # Nuisance Error Plot
    ax2 = fig.add_subplot(122)
    sns.kdeplot(x=embedding[:, 0], y=embedding[:, 1], fill=True, cmap="Blues", alpha=0.3, ax=ax2)
    sc2 = ax2.scatter(embedding[:, 0], embedding[:, 1], c=combined_nuisance,
                      cmap='viridis', s=15, alpha=0.9, edgecolor='none')

    nuis_title = (r"Nuisance Parameter Invariance" + "\n" +
                  f"Predictive Power ($R^2$): I={r2_I:.3f}, A={r2_A:.3f} | Distance Corr: {dcor_nuisance:.3f}")
    ax2.set_title(nuis_title, fontsize=14, pad=15)
    fig.colorbar(sc2, ax=ax2, label='Standardized Combined Error (Z-Score)')

    plt.suptitle(f'Single Terrain Manifold Projection and Statistical Disentanglement ({num_samples} Samples)',
                 fontsize=18, y=1.05)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Replaced save_fig with standard plt/fig syntax
    save_fig(fig, output_dir / 'umap_disentanglement.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"\n--- Single Terrain Disentanglement Report ---")
    print(f"Angular Error - R^2: {r2_ang:.3f}, dCor: {dcor_ang:.3f}")
    print(f"Nuisance (I)  - R^2: {r2_I:.3f}")
    print(f"Nuisance (A)  - R^2: {r2_A:.3f}")
    print(f"Combined Nuisance dCor: {dcor_nuisance:.3f}")


@torch.no_grad()
def plot_component_ablation(dataloader, loss_fn, output_dir: Path, steps: int = 50, num_batches: int = 100):
    """
    Computes loss ablation dynamically over a statistically significant sample size,
    generating 95% Confidence Intervals via empirical bootstrapping.
    """
    device = next(loss_fn.parameters()).device

    results = []
    angles = np.linspace(0, 90, steps)

    for b_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Ablating Components")):
        if b_idx >= num_batches: break

        dtm = batch["dtm"].to(device)
        ortho = batch["image"].to(device)
        mask = batch["confidence"].to(device)
        gt_I = batch["intensity"].to(device).view(-1, 1, 1, 1)
        gt_A = batch["ambient"].to(device).view(-1, 1, 1, 1)
        gt_sun = batch["sun_vector"].to(device)

        ortho_gray = ortho.mean(dim=1, keepdim=True) if ortho.shape[1] == 3 else ortho
        dtm = dtm.mean(dim=1, keepdim=True) if dtm.shape[1] == 3 else dtm

        for idx in range(dtm.shape[0]):
            v = gt_sun[idx].cpu().numpy()
            random_vec = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            rot_axis = np.cross(v, random_vec)
            rot_axis /= (np.linalg.norm(rot_axis) + 1e-8)

            for angle in angles:
                r = R.from_rotvec(np.radians(angle) * rot_axis)
                perturbed_sun = torch.tensor(r.apply(v), device=device, dtype=torch.float32).unsqueeze(0)

                render, _ = loss_fn.render_from_depth(dtm[idx:idx + 1], perturbed_sun, gt_I[idx:idx + 1],
                                                      gt_A[idx:idx + 1])

                p_loss = loss_fn._masked_pearson(render, ortho_gray[idx:idx + 1], mask[idx:idx + 1], B=1).item()
                s_loss = loss_fn._masked_ssim(render, ortho_gray[idx:idx + 1], mask[idx:idx + 1]).item()
                total = (1.0 - loss_fn.ssim_weight) * p_loss + loss_fn.ssim_weight * s_loss

                results.append({'Angle': angle, 'Loss Value': p_loss, 'Metric': 'Pearson (1 - r)'})
                results.append({'Angle': angle, 'Loss Value': s_loss, 'Metric': 'SSIM (1 - s)'})
                results.append({'Angle': angle, 'Loss Value': total, 'Metric': 'Total Loss'})

    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(10, 6))

    # seaborn natively bootstraps 95% CIs and plots standard error bands
    sns.lineplot(data=df, x='Angle', y='Loss Value', hue='Metric',
                 errorbar=('ci', 95), linewidth=2.5, ax=ax,
                 palette=['#1f77b4', '#ff7f0e', '#2ca02c'])

    ax.set_xlabel(r'Angular Error $\gamma$ (Degrees)', fontweight='bold')
    ax.set_ylabel('Empirical Risk', fontweight='bold')
    ax.set_title('Loss Component Dynamics with 95% Confidence Intervals', pad=15)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_dir / 'component_ablation.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)


@torch.no_grad()
def plot_qualitative_physics_errors(dataloader, loss_fn, output_dir: Path, search_batches: int = 15):
    """
    Augments qualitative outputs with statistical residual distributions
    to rigorously prove error displacement dynamics. Actively searches for
    high-variance terrain (e.g., mountains/craters) for maximum visual clarity.
    """
    device = next(loss_fn.parameters()).device

    # --- 1. Find a sample with high feature distinction ---
    best_batch = None
    max_variance = -1

    print(f"Scanning up to {search_batches} batches for rugged terrain...")
    for b_idx, batch in enumerate(dataloader):
        if b_idx >= search_batches:
            break

        dtm_np = batch["dtm"][0, 0].cpu().numpy()
        mask_np = batch["confidence"][0, 0].cpu().numpy().astype(bool)

        valid_dtm = dtm_np[mask_np]
        if len(valid_dtm) > 0:
            variance = np.var(valid_dtm)
            if variance > max_variance:
                max_variance = variance
                best_batch = batch

    if best_batch is None:
        raise ValueError("Could not find any valid terrain in the dataloader.")

    print(f"Selected terrain with depth variance: {max_variance:.2f}")
    batch = best_batch

    # --- 2. Extract Data ---
    dtm = batch["dtm"][0:1].to(device)
    ortho = batch["image"][0:1].to(device)
    mask = batch["confidence"][0:1].to(device)
    I = batch["intensity"][0:1].to(device).view(1, 1, 1, 1)
    A = batch["ambient"][0:1].to(device).view(1, 1, 1, 1)
    gt_sun = batch["sun_vector"][0].cpu().numpy()

    # --- 3. Compute Illumination Geometries ---
    sun_gt = torch.tensor(gt_sun, device=device, dtype=torch.float32).unsqueeze(0)

    az_rot = R.from_euler('z', 45, degrees=True)
    sun_az = torch.tensor(az_rot.apply(gt_sun), device=device, dtype=torch.float32).unsqueeze(0)

    # Zenith Singularity Fix
    el_axis = np.cross(np.array([0, 0, 1]), gt_sun)
    axis_norm = np.linalg.norm(el_axis)
    if axis_norm < 1e-6:
        el_axis = np.array([1.0, 0.0, 0.0])  # Fallback if sun is perfectly at zenith
    else:
        el_axis /= axis_norm

    el_rot = R.from_rotvec(np.radians(45) * el_axis)
    sun_el = torch.tensor(el_rot.apply(gt_sun), device=device, dtype=torch.float32).unsqueeze(0)

    ortho_gray = ortho.mean(dim=1, keepdim=True) if ortho.shape[1] == 3 else ortho
    dtm = dtm.mean(dim=1, keepdim=True) if dtm.shape[1] == 3 else dtm
    ortho_z = loss_fn._zscore(ortho_gray, mask)

    conditions = [
        ("Ground Truth Base", sun_gt),
        (r"Azimuth Shift (+45$^\circ$)", sun_az),
        (r"Elevation Shift (+45$^\circ$)", sun_el)
    ]

    # --- Custom GridSpec Layout ---
    fig = plt.figure(figsize=(18, 12))

    # GridSpec 1: Images (Columns 0, 1, 2)
    # wspace=0.05 creates minimal horizontal gaps between the source, render, and residual maps.
    gs_img = gridspec.GridSpec(3, 3, figure=fig, wspace=0.05, hspace=0.35, left=0.02, right=0.42)

    # GridSpec 2: Statistical Plots (Columns 3, 4)
    # The gap between right=0.48 (img) and left=0.55 (plt) leaves room for the density y-labels.
    gs_plt = gridspec.GridSpec(3, 2, figure=fig, wspace=0.25, hspace=0.35, left=0.50, right=0.98,
                               width_ratios=[1.2, 1.0])

    mask_np = mask[0, 0].cpu().numpy().astype(bool)

    # Pre-calculate Ground Truth baseline distribution for visual anchoring
    render_gt, _ = loss_fn.render_from_depth(dtm, sun_gt, I, A)
    render_gt_z = loss_fn._zscore(render_gt, mask)
    gt_residual_raw = (render_gt_z - ortho_z)[0, 0].cpu().numpy()
    gt_valid_residuals = gt_residual_raw[mask_np].flatten()

    axes = np.empty((3, 5), dtype=object)

    for i in range(3):
        # Assign Image axes
        axes[i, 0] = fig.add_subplot(gs_img[i, 0])
        axes[i, 1] = fig.add_subplot(gs_img[i, 1])
        axes[i, 2] = fig.add_subplot(gs_img[i, 2])

        # Assign Plot axes
        axes[i, 3] = fig.add_subplot(gs_plt[i, 0])
        axes[i, 4] = fig.add_subplot(gs_plt[i, 1])

    for i, (title, sun_vec) in tqdm(enumerate(conditions), desc="Rendering Conditions", total=3):
        render, _ = loss_fn.render_from_depth(dtm, sun_vec, I, A)
        render_z = loss_fn._zscore(render, mask)

        residual_raw = (render_z - ortho_z)[0, 0].cpu().numpy()
        valid_residuals = residual_raw[mask_np].flatten()

        # Fix: Keep directional signs and clip symmetrically for the diverging colormap
        residual_img = residual_raw.copy()
        residual_img = np.clip(residual_img, -3.0, 3.0)
        residual_img[~mask_np] = np.nan

        ortho_disp = np.clip((ortho_gray[0, 0].cpu().numpy() + 1.0) / 2.0, 0, 1)
        render_disp = np.clip((render[0, 0].cpu().numpy() + 1.0) / 2.0, 0, 1)

        ortho_disp[~mask_np] = np.nan
        render_disp[~mask_np] = np.nan

        axes[i, 0].imshow(ortho_disp, cmap="gray")
        axes[i, 0].set_title(r"Source Image $\mathcal{I}_o$")

        axes[i, 1].imshow(render_disp, cmap="gray")
        axes[i, 1].set_title(rf"Render $\mathcal{{I}}_r$: {title}")

        # Fix: Enforce vmin/vmax so white is exactly 0.0
        im = axes[i, 2].imshow(residual_img, cmap="RdBu_r", vmin=-3.0, vmax=3.0)
        axes[i, 2].set_title("Structural Residual Map")

        for j in range(3):
            axes[i, j].axis('off')

        # Fix: Meaningful Error Metrics (RMSE and MAE)
        rmse = np.sqrt(np.mean(valid_residuals ** 2))
        mae = np.mean(np.abs(valid_residuals))

        # Fix: Plot Ground Truth KDE in the background of perturbed rows
        if i > 0:
            sns.kdeplot(gt_valid_residuals, ax=axes[i, 3], color='gray', fill=True, alpha=0.2)

        sns.histplot(valid_residuals, kde=True, ax=axes[i, 3], color='#8b0000' if i > 0 else '#2ca02c', bins=50,
                     stat='density')
        axes[i, 3].set_title(
            rf"Residual Distribution" + "\n" + rf"RMSE: {rmse:.2f} | MAE: {mae:.2f}")
        axes[i, 3].set_xlim(-4, 4)
        axes[i, 3].set_xlabel("$Z_r - Z_o$")
        axes[i, 3].set_ylabel("Pixel Density")

        sns.violinplot(x=valid_residuals, ax=axes[i, 4], color='#d3d3d3', inner="quartile")
        axes[i, 4].set_title("Quartile Shift")
        axes[i, 4].set_xlim(-4, 4)

    fig.suptitle('Qualitative and Statistical Displacement under Erroneous Illumination Profiles', fontsize=18,
                 fontweight='bold')

    # Optional: Add colorbar for the residual map to make the scale explicit
    cbar_ax = fig.add_axes([0.17, 0.06, 0.10, 0.02])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='Z-Scored Error')

    output_dir.mkdir(parents=True, exist_ok=True)
    # Using standard savefig directly on the figure object
    save_fig(fig, output_dir / 'qualitative_illumination_physics.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)


@torch.no_grad()
def compute_topography_statistics(dataloader, split_name="Dataset"):
    """Evaluates how many patches are essentially flat planes."""
    logger.info(f"Computing topography statistics for {split_name}...")

    residuals = []
    flat_stds = []

    for batch in tqdm(dataloader, desc=f"Evaluating {split_name} roughness"):
        dtms = batch["dtm"]
        masks = batch["confidence"]
        for i in range(dtms.shape[0]):
            elevation_m = dtms[i, 0]
            mask = masks[i, 0]
            residuals.append(compute_topographic_residual(elevation_m, mask))
            flat_stds.append(torch.std(elevation_m[mask.bool()]))

    for metric_name, vals in zip(["TOPOGRAPHY", "FLATNESS"], [residuals, flat_stds]):
        vals = np.array(vals)
        logger.info("-" * 50)
        logger.info(f"{metric_name} STATISTICS FOR: {split_name.upper()}")
        logger.info("-" * 50)
        logger.info(f"Total Patches Evaluated: {len(vals)}")
        logger.info(f"Mean Residual: {np.mean(vals):.2f} meters")
        logger.info(f"Median Residual: {np.median(vals):.2f} meters")
        logger.info(f"Max Residual: {np.max(vals):.2f} meters")
        logger.info("-" * 50)
        logger.info("Rejection Rates based on Thresholds:")
        for t in np.logspace(-3, 0, 10):
            rejected = np.sum(vals < t)
            pct = (rejected / len(vals)) * 100
            logger.info(f"  < {t:.3f}m (rejected): {rejected} patches ({pct:.1f}%)")
        logger.info("-" * 50)


@torch.no_grad()
def compute_mask_statistics(dataloader, split_name="Dataset"):
    """Computes statistics on the confidence masks."""
    logger.info(f"Computing mask statistics for {split_name}...")

    total_images = 0
    fully_valid_images = 0
    partially_masked_images = 0
    empty_images = 0
    total_valid_pixels = 0
    total_pixels = 0

    for batch in tqdm(dataloader, desc=f"Processing {split_name}"):
        mask = batch["confidence"]
        B = mask.shape[0]
        total_images += B
        per_image_mean = mask.view(B, -1).mean(dim=1)
        fully_valid_images += (per_image_mean == 1.0).sum().item()
        empty_images += (per_image_mean == 0.0).sum().item()
        partially_masked_images += ((per_image_mean > 0.0) & (per_image_mean < 1.0)).sum().item()
        total_valid_pixels += mask.sum().item()
        total_pixels += mask.numel()

    pct_fully_valid = (fully_valid_images / total_images) * 100 if total_images > 0 else 0
    pct_partial = (partially_masked_images / total_images) * 100 if total_images > 0 else 0
    pct_empty = (empty_images / total_images) * 100 if total_images > 0 else 0
    global_valid_pct = (total_valid_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    logger.info("-" * 50)
    logger.info(f"STATISTICS FOR: {split_name.upper()}")
    logger.info("-" * 50)
    logger.info(f"Total Images: {total_images}")
    logger.info(f"  - 100%% Valid Data (No Nodata):  {fully_valid_images} ({pct_fully_valid:.2f}%%)")
    logger.info(f"  - Partially Masked (Has Nodata): {partially_masked_images} ({pct_partial:.2f}%%)")
    logger.info(f"  - 100%% Nodata (Completely Empty): {empty_images} ({pct_empty:.2f}%%)")
    logger.info(f"Global Valid Pixel Percentage:    {global_valid_pct:.2f}%%")
    logger.info("-" * 50)
    return {
        "total": total_images,
        "fully_valid": fully_valid_images,
        "partial": partially_masked_images,
        "empty": empty_images,
        "global_valid_pct": global_valid_pct,
    }


@torch.no_grad()
def generate_thumbnail_grids(dataloader, output_dir: Path, num_samples: int = 3):
    """
    Extracts random samples and plots them with both macro and micro views.
    Layout: [Full Ortho] | [Full DTM] | [Full Detrended] | [Ortho Zoom] | [Slope Zoom] | [Detrended Zoom]
    """
    logger.info(f"Generating high-detail mask-aware thumbnail grid for {num_samples} samples...")
    output_dir.mkdir(parents=True, exist_ok=True)

    images, dtms, masks = [], [], []
    with tqdm(total=num_samples, desc="Collect samples") as pbar:
        for batch in dataloader:
            B = batch["image"].shape[0]
            for i in range(B):
                images.append(batch["image"][i].cpu().numpy())
                dtms.append(batch["dtm"][i].cpu().numpy())
                masks.append(batch["confidence"][i].cpu().numpy())
                pbar.update()
                if len(images) == num_samples:
                    break
            if len(images) == num_samples:
                break

    n_cols = 7
    scale = 4
    fig, axes = plt.subplots(num_samples, n_cols, figsize=(scale * n_cols, scale * num_samples))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    crop_size = 128

    def detrend_and_stretch(z_data, mask):
        h, w = z_data.shape
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        x_valid, y_valid, z_valid = x_grid[mask], y_grid[mask], z_data[mask]
        if len(z_valid) >= 3:
            A = np.c_[x_valid, y_valid, np.ones_like(x_valid)]
            C, _, _, _ = np.linalg.lstsq(A, z_valid, rcond=None)
            detrended = z_data - (C[0] * x_grid + C[1] * y_grid + C[2])
            dp2, dp98 = np.percentile(detrended[mask], [2, 98])
            normed = np.clip((detrended - dp2) / (dp98 - dp2), 0.0, 1.0) if dp98 > dp2 else np.zeros_like(detrended)
        else:
            normed = np.zeros_like(z_data)
        # normed[~mask] = np.nan
        return normed

    for idx in tqdm(range(num_samples), desc="Generating thumbnails"):
        img_np = np.clip((np.transpose(images[idx], (1, 2, 0)) + 1.0) / 2.0, 0.0, 1.0)
        dtm_np = _display_dtm(np.transpose(dtms[idx], (1, 2, 0)))
        mask_np = masks[idx][0].astype(bool)
        # img_np[~mask_np] = np.nan
        # dtm_np[~mask_np] = np.nan

        dtm_full_1ch = dtm_np[..., 0]
        detrended_full_norm = detrend_and_stretch(dtm_full_1ch, mask_np)

        H, W = img_np.shape[:2]
        cy, cx = H // 2, W // 2
        half_c = crop_size // 2

        img_crop = img_np[cy - half_c: cy + half_c, cx - half_c: cx + half_c]
        dtm_crop = dtm_full_1ch[cy - half_c: cy + half_c, cx - half_c: cx + half_c]
        mask_crop = mask_np[cy - half_c: cy + half_c, cx - half_c: cx + half_c]

        dy, dx = np.gradient(dtm_crop)
        slope_mag = np.sqrt(dx ** 2 + dy ** 2)
        valid_slopes = slope_mag[mask_crop]
        if len(valid_slopes) > 0:
            p2, p98 = np.percentile(valid_slopes, [2, 98])
            slope_norm = np.clip((slope_mag - p2) / (p98 - p2), 0.0, 1.0) if p98 > p2 else np.zeros_like(slope_mag)
        else:
            slope_norm = np.zeros_like(slope_mag)
        # slope_norm[~mask_crop] = np.nan
        detrended_crop_norm = detrend_and_stretch(dtm_crop, mask_crop)

        axes[idx, 0].imshow(img_np)
        axes[idx, 1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
        axes[idx, 2].imshow(dtm_full_1ch, cmap="terrain")
        axes[idx, 3].imshow(detrended_full_norm, cmap="terrain")
        axes[idx, 4].imshow(img_crop)
        axes[idx, 5].imshow(slope_norm, cmap="magma")
        axes[idx, 6].imshow(detrended_crop_norm, cmap="terrain")
        for ax in axes[idx]:
            ax.axis("off")
        if idx == 0:
            for ax, t in zip(
                    axes[0],
                    ["Ortho (Full)", "Mask (Full)", "DTM (Full)", "Detrended (Full)", f"Ortho Zoom ({crop_size}px)",
                     f"Masked Slope ({crop_size}px)", f"Masked Detrend ({crop_size}px)"],
            ):
                ax.set_title(t)

    save_path = output_dir / "dataset_thumbnails_detailed.png"
    save_fig(fig, save_path, bbox_inches="tight", dpi=DPI, transparent=False, facecolor="white")
    plt.close(fig)
    logger.info(f"Detailed thumbnails successfully saved to: {save_path}")