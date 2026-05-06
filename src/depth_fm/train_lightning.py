"""
Mars DepthFM Training — Lightning-based with multi-run evaluation.

Features:
- Uses MarsHiRISEDTM + HiRISEGeoSampler with train/val/test splits
- LitData StreamingDataset for hyper-optimized I/O (primary)
- DepthFMHiRISEAdapterCached fallback for first-time runs
- Multiple independent training runs for statistical significance
- Publication-quality figures with error bars
- Patch-level error analysis identifying failure modes
- All metrics logged to wandb with proper grouping

Hardware target: 2×128-core Ryzen, 4×A6000, 1 TB RAM, NVMe storage.

Usage:
    # Single run
    python src/train_lightning.py --config configs/train_hirise.yaml

    # Multi-run for error bars (3 seeds)
    python src/train_lightning.py --config configs/train_hirise.yaml --n_runs 3

    # K-fold cross-validation
    python src/train_lightning.py --config configs/train_hirise.yaml --n_folds 5
"""

import argparse
import concurrent.futures
import hashlib
import json
import logging
import math
import os
from copy import deepcopy
from typing import Callable

import matplotlib.patches as mpatches
from matplotlib.figure import Figure

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
import torch.nn.functional as F
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
from depth_fm.visualization import (
    plot_convergence_curves,
    plot_metric_distributions,
    plot_multi_run_summary_table,
    set_neurips_style, plot_pareto_frontier,
)

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------------------------
# Hardware-aware constants for 2×128-core Ryzen / 4×A6000 / 1 TB RAM
# ---------------------------------------------------------------------------
_TOTAL_CORES = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 256
_NUM_GPUS_DEFAULT = torch.cuda.device_count() if torch.cuda.is_available() else 1

# Workers per GPU: leave headroom for the main process and OS.
# 256 cores / 4 GPUs = 64 per GPU; cap at 48 to avoid memory pressure from
# too many rasterio/GDAL file handles.
_WORKERS_PER_GPU = min(24, max(4, (_TOTAL_CORES - 16) // max(_NUM_GPUS_DEFAULT, 1)))
_VAL_WORKERS = min(4, _WORKERS_PER_GPU)

DPI = 300


# ---------------------------------------------------------------------------
# Visualization helpers (unchanged from original)
# ---------------------------------------------------------------------------

def save_fig(fig: Figure, path: Path, formats: tuple[str, ...] = (".png", ".pdf"), **kwargs):
    for f in formats:
        new_path = path.with_suffix(f)
        fig.savefig(new_path, **kwargs)
        logger.info(f"Saved {new_path}")


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
    return (gt + noise_level * noise + bias).clamp(-1.0, 1.0)


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
                dtm_d = np.clip((dtm[0, 0].cpu().numpy() + 1.0) / 2.0, 0, 1)
                pred_d = np.clip((pred[0, 0].cpu().numpy() + 1.0) / 2.0, 0, 1)
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
                dtm_d = np.clip((dtm[0, 0].cpu().numpy() + 1.0) / 2.0, 0, 1)
                pred_d = np.clip((pred[0, 0].cpu().numpy() + 1.0) / 2.0, 0, 1)
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
                dtm_d = np.clip((dtm[0, 0].cpu().numpy() + 1.0) / 2.0, 0, 1)
                pred_d = np.clip((pred[0, 0].cpu().numpy() + 1.0) / 2.0, 0, 1)

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
            dtm_np = np.clip((dtm[0].cpu().numpy() + 1.0) / 2.0, 0.0, 1.0)
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
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import numpy as np

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
        dtm_disp = np.clip((item["dtm"] + 1.0) / 2.0, 0.0, 1.0)
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
                dtm_disp = np.clip((dtm[0, 0].cpu().numpy() + 1.0) / 2.0, 0.0, 1.0)
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

                dtm_masked_np = np.clip((dtm_masked[0, 0].cpu().numpy() + 1.0) / 2.0, 0.0, 1.0)
                dtm_filled_np = np.clip((dtm_filled[0, 0].cpu().numpy() + 1.0) / 2.0, 0.0, 1.0)

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

                    ortho_gray = img.mean(dim=1, keepdim=True) if img.shape[1] == 3 else img

                    # --- Estimate GT exposure/sun via OLS (for sanity check column) ---
                    sun_vec_gt, intensity_gt, ambient_gt = estimate_sun_vector_irls(dtm, img, mask)
                    # OLS returns (3,), scalar, scalar — reshape for render_from_depth
                    sun_vec_gt = sun_vec_gt.view(1, 3)
                    intensity_gt = intensity_gt.view(1)
                    ambient_gt = ambient_gt.view(1)

                    # --- Use the ACTUAL loss class methods ---
                    render, normals = loss_fn.render_from_depth(
                        dtm, sun_vec, intensity, ambient,
                    )
                    render_gt, _ = loss_fn.render_from_depth(
                        dtm, sun_vec_gt, intensity_gt, ambient_gt,
                    )

                    # --- z-scored versions: what SSIM actually sees ---
                    render_z = loss_fn._zscore(render, mask)
                    ortho_z = loss_fn._zscore(ortho_gray, mask)

                    # --- Displays ---
                    mask_np = mask[0, 0].cpu().numpy().astype(bool)

                    img_disp = _to_gray_display(ortho_gray, mask_np)
                    dtm_disp = np.clip((dtm[0, 0].cpu().numpy() + 1.0) / 2.0, 0.0, 1.0)
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


import torch
from matplotlib.patches import ConnectionPatch
from pathlib import Path


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
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


@torch.no_grad()
def plot_spherical_loss_landscape(
        dataloader,
        loss_fn,
        output_dir: Path,
        resolution: int = 50,
):
    """
    Computes and plots the loss landscape over the entire solar hemisphere 
    using a Lambert Azimuthal Equal-Area projection.
    """
    device = next(loss_fn.parameters()).device if hasattr(loss_fn, 'parameters') else torch.device("cuda")
    loss_fn.eval()

    # Grab a single high-quality validation patch
    batch = next(iter(dataloader))
    img = batch["image"][0:1].to(device).float()
    dtm = batch["dtm"][0:1, :1].to(device).float()
    mask = batch["confidence"][0:1].to(device).float()
    ambient = batch["ambient"][0:1].to(device).float()
    intensity = batch["intensity"][0:1].to(device).float()
    gt_sun = batch["sun_vector"][0].cpu().numpy()

    # Generate hemispherical grid (Elevation 0 to 90, Azimuth 0 to 360)
    theta = np.linspace(0, np.pi / 2, resolution)  # Zenith angle (90 - elevation)
    phi = np.linspace(0, 2 * np.pi, resolution * 2)  # Azimuth
    T, P = np.meshgrid(theta, phi)

    # Convert to Cartesian sun vectors
    S_x = np.sin(T) * np.cos(P)
    S_y = np.sin(T) * np.sin(P)
    S_z = np.cos(T)
    sun_grid = np.stack([S_x, S_y, S_z], axis=-1).reshape(-1, 3)

    losses = []
    batch_size = 256

    # Evaluate loss surface in batches
    for i in tqdm(range(0, len(sun_grid), batch_size), desc="Scanning Hemisphere"):
        s_batch = torch.tensor(sun_grid[i:i + batch_size], device=device, dtype=torch.float32)
        current_bs = s_batch.shape[0]

        # Expand inputs to match batch size
        img_b = img.expand(current_bs, -1, -1, -1)
        dtm_b = dtm.expand(current_bs, -1, -1, -1)
        mask_b = mask.expand(current_bs, -1, -1, -1)
        amb_b = ambient.expand(current_bs)
        int_b = intensity.expand(current_bs)

        loss_vals = [loss_fn(dtm_b[j:j + 1], img_b[j:j + 1], mask_b[j:j + 1], s_batch[j:j + 1], amb_b[j:j + 1],
                             int_b[j:j + 1]).item() for j in range(current_bs)]
        losses.extend(loss_vals)

    L = np.array(losses).reshape(T.shape)

    # Lambert Azimuthal Equal-Area Projection Mathematics
    # R = 2 * sin(theta / 2) preserves area
    R = 2 * np.sin(T / 2)
    X = R * np.sin(P)
    Y = R * np.cos(P)

    fig, ax = plt.subplots(figsize=(8, 8))
    contour = ax.contourf(X, Y, L, levels=50, cmap='viridis')
    ax.contour(X, Y, L, levels=20, colors='black', linewidths=0.3, alpha=0.5)

    # Plot GT Sun Vector
    gt_zenith = np.arccos(np.clip(gt_sun[2], -1.0, 1.0))
    gt_azimuth = np.arctan2(gt_sun[1], gt_sun[0])
    gt_R = 2 * np.sin(gt_zenith / 2)
    ax.scatter(gt_R * np.sin(gt_azimuth), gt_R * np.cos(gt_azimuth),
               color='red', marker='*', s=200, edgecolors='white', label='GT Sun Vector')

    ax.set_aspect('equal')
    ax.axis('off')
    fig.colorbar(contour, ax=ax, label=r'Photoclinometric Loss ($\mathcal{L}$)', shrink=0.7)
    ax.legend(loc='lower right')

    output_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_dir / 'spherical_loss_landscape.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)


import umap
import seaborn as sns


@torch.no_grad()
def plot_umap_invariance(dataloader, loss_fn, output_dir: Path, num_samples: int = 1500):
    """
    Projects the high-dimensional parameter space down to 2D via UMAP to demonstrate 
    that the loss topology is driven strictly by angular error, completely ignoring 
    luminance and ambient scaling mismatches.
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

    # Generate synthetic permutations of parameters
    np.random.seed(42)
    sun_perturb = np.random.randn(num_samples, 3)
    sun_perturb /= np.linalg.norm(sun_perturb, axis=1, keepdims=True)

    I_perturb = np.random.uniform(gt_I * 0.1, gt_I * 10.0, num_samples)
    A_perturb = np.random.uniform(gt_A - 0.5, gt_A + 0.5, num_samples)

    # Store errors
    angular_errors = np.arccos(np.clip(np.sum(sun_perturb * gt_sun, axis=1), -1.0, 1.0))
    nuisance_errors = np.sqrt((I_perturb - gt_I) ** 2 + (A_perturb - gt_A) ** 2)

    # Render and compute z-scored images to build the feature manifold
    features = []

    dtm = dtm.mean(dim=1, keepdim=True) if dtm.shape[1] == 3 else dtm
    ortho_gray = ortho.mean(dim=1, keepdim=True) if ortho.shape[1] == 3 else ortho
    ortho_z = loss_fn._zscore(ortho_gray, mask).cpu().numpy().flatten()

    for i in tqdm(range(num_samples), desc="Generating renders for UMAP"):
        s_t = torch.tensor(sun_perturb[i:i + 1], device=device, dtype=torch.float32)
        I_t = torch.tensor([I_perturb[i]], device=device, dtype=torch.float32).view(1, 1, 1, 1)
        A_t = torch.tensor([A_perturb[i]], device=device, dtype=torch.float32).view(1, 1, 1, 1)

        render, _ = loss_fn.render_from_depth(dtm, s_t, I_t, A_t)
        render_z = loss_fn._zscore(render, mask)

        # The residual structurally defines the error state
        residual = np.abs(render_z.cpu().numpy().flatten() - ortho_z)
        features.append(residual)

    features = np.stack(features)

    # UMAP Projection
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    embedding = reducer.fit_transform(features)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc1 = axes[0].scatter(embedding[:, 0], embedding[:, 1], c=np.degrees(angular_errors), cmap='inferno', s=10,
                          alpha=0.8)
    axes[0].set_title(r'Manifold colored by Angular Error ($\gamma$)')
    fig.colorbar(sc1, ax=axes[0], label='Angular Error (degrees)')
    axes[0].axis('off')

    sc2 = axes[1].scatter(embedding[:, 0], embedding[:, 1], c=nuisance_errors, cmap='viridis', s=10, alpha=0.8)
    axes[1].set_title(r'Manifold colored by Nuisance Error ($\Delta_{IA}$)')
    fig.colorbar(sc2, ax=axes[1], label='L2 distance from GT Intensity/Ambient')
    axes[1].axis('off')

    fig.suptitle('UMAP of Z-Scored Render Residuals: Proving Shift-Invariance', fontsize=14)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_dir / 'umap_disentanglement.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)


from scipy.spatial.transform import Rotation as R


@torch.no_grad()
def plot_component_ablation(dataloader, loss_fn, output_dir: Path, steps: int = 100):
    device = next(loss_fn.parameters()).device

    batch = next(iter(dataloader))
    dtm = batch["dtm"][0:1].to(device)
    ortho = batch["image"][0:1].to(device)
    mask = batch["confidence"][0:1].to(device)
    gt_I = batch["intensity"][0:1].to(device).view(1, 1, 1, 1)
    gt_A = batch["ambient"][0:1].to(device).view(1, 1, 1, 1)
    gt_sun = batch["sun_vector"][0:1].to(device)

    ortho_gray = ortho.mean(dim=1, keepdim=True) if ortho.shape[1] == 3 else ortho
    dtm = dtm.mean(dim=1, keepdim=True) if dtm.shape[1] == 3 else dtm

    # Define an arbitrary orthogonal axis to rotate the sun vector around
    v = gt_sun[0].cpu().numpy()
    random_vec = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    rot_axis = np.cross(v, random_vec)
    rot_axis /= np.linalg.norm(rot_axis)

    angles = np.linspace(0, 90, steps)

    pearson_vals, ssim_vals, total_vals = [], [], []

    for angle in angles:
        r = R.from_rotvec(np.radians(angle) * rot_axis)
        perturbed_sun = torch.tensor(r.apply(v), device=device, dtype=torch.float32).unsqueeze(0)

        render, _ = loss_fn.render_from_depth(dtm, perturbed_sun, gt_I, gt_A)

        # Call isolated loss components directly at scale 1x
        p_loss = loss_fn._masked_pearson(render, ortho_gray, mask, B=1).item()
        s_loss = loss_fn._masked_ssim(render, ortho_gray, mask).item()

        alpha = loss_fn.ssim_weight
        total = (1.0 - alpha) * p_loss + alpha * s_loss

        pearson_vals.append(p_loss)
        ssim_vals.append(s_loss)
        total_vals.append(total)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(angles, pearson_vals, label=r'$(1 - \mathrm{Pearson})$', color='blue', linewidth=2, linestyle='--')
    ax.plot(angles, ssim_vals, label=r'$(1 - \mathrm{SSIM}_z)$', color='orange', linewidth=2, linestyle='--')
    ax.plot(angles, total_vals, label=r'Total $\mathcal{L}_{photo}$', color='black', linewidth=3)

    ax.set_xlabel(r'Angular Error $\gamma$ (Degrees)')
    ax.set_ylabel('Loss Value')
    ax.set_title('Loss Components vs. Illumination Error')
    ax.grid(alpha=0.3)
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_dir / 'component_ablation.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)


@torch.no_grad()
def plot_qualitative_physics_errors(dataloader, loss_fn, output_dir: Path):
    device = next(loss_fn.parameters()).device

    batch = next(iter(dataloader))
    dtm = batch["dtm"][0:1].to(device)
    ortho = batch["image"][0:1].to(device)
    mask = batch["confidence"][0:1].to(device)
    I = batch["intensity"][0:1].to(device).view(1, 1, 1, 1)
    A = batch["ambient"][0:1].to(device).view(1, 1, 1, 1)

    gt_sun = batch["sun_vector"][0].cpu().numpy()

    # 1. Ground Truth 
    sun_gt = torch.tensor(gt_sun, device=device, dtype=torch.float32).unsqueeze(0)

    # 2. Azimuth Error (+45 deg)
    az_rot = R.from_euler('z', 45, degrees=True)
    sun_az = torch.tensor(az_rot.apply(gt_sun), device=device, dtype=torch.float32).unsqueeze(0)

    # 3. Elevation Error (+45 deg)
    el_axis = np.cross(np.array([0, 0, 1]), gt_sun)
    el_axis /= np.linalg.norm(el_axis)
    el_rot = R.from_rotvec(np.radians(45) * el_axis)
    sun_el = torch.tensor(el_rot.apply(gt_sun), device=device, dtype=torch.float32).unsqueeze(0)

    ortho_gray = ortho.mean(dim=1, keepdim=True) if ortho.shape[1] == 3 else ortho
    dtm = dtm.mean(dim=1, keepdim=True) if dtm.shape[1] == 3 else dtm
    ortho_z = loss_fn._zscore(ortho_gray, mask)

    conditions = [
        ("Ground Truth", sun_gt),
        ("Azimuth Error (+45°)", sun_az),
        ("Elevation Error (+45°)", sun_el)
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    mask_np = mask[0, 0].cpu().numpy().astype(bool)

    for i, (title, sun_vec) in enumerate(conditions):
        render, _ = loss_fn.render_from_depth(dtm, sun_vec, I, A)
        render_z = loss_fn._zscore(render, mask)
        residual = np.abs((render_z - ortho_z)[0, 0].cpu().numpy())
        residual = np.clip(residual / 2.0, 0.0, 1.0)
        residual[~mask_np] = np.nan

        ortho_disp = np.clip((ortho_gray[0, 0].cpu().numpy() + 1.0) / 2.0, 0, 1)
        render_disp = np.clip((render[0, 0].cpu().numpy() + 1.0) / 2.0, 0, 1)
        dtm_disp = np.clip((dtm[0, 0].cpu().numpy() + 1.0) / 2.0, 0, 1)

        ortho_disp[~mask_np] = np.nan
        render_disp[~mask_np] = np.nan
        dtm_disp[~mask_np] = np.nan

        axes[i, 0].imshow(dtm_disp, cmap="terrain")
        axes[i, 0].set_title("GT Depth")

        axes[i, 1].imshow(ortho_disp, cmap="gray")
        axes[i, 1].set_title("Real Ortho")

        axes[i, 2].imshow(render_disp, cmap="gray")
        axes[i, 2].set_title(f"Render: {title}")

        im = axes[i, 3].imshow(residual, cmap="RdBu_r")
        axes[i, 3].set_title("$|Z_r - Z_o|$ Residual")

        for ax in axes[i]:
            ax.axis('off')

    fig.suptitle('Structural Residuals under Incorrect Illumination Topologies', fontsize=16)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_dir / 'qualitative_illumination_physics.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)


@torch.no_grad()
def compute_topography_statistics(dataloader, split_name="Dataset"):
    """Evaluates how many patches are essentially flat planes."""
    logger.info(f"Computing topography statistics for {split_name}...")
    from depth_fm.depthfm_adapter import compute_topographic_residual

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
        dtm_np = np.clip((np.transpose(dtms[idx], (1, 2, 0)) + 1.0) / 2.0, 0.0, 1.0)
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def configure_worker_logger(worker_id):
    """Forces the spawned PyTorch worker to actually print INFO logs."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format=f"[Worker {worker_id}] %(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )


def get_litdata_cache_key(config) -> str:
    """Deterministic hash for LitData cache location (matches build_litdata.py)."""
    key_parts = {
        "hirise": OmegaConf.to_container(config.data.hirise, resolve=True),
        "sampler": OmegaConf.to_container(config.data.sampler, resolve=True),
        "resolution": config.data.get("resolution", 512),
        "dtm_normalization": config.data.get("dtm_normalization", "relative"),
    }

    clip = config.data.get("clip", False)

    if not clip:
        key_parts["clip"] = clip

    raw = json.dumps(key_parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _build_cached_loaders(config, split_seed: int = 42, parallel: bool = True) -> dict:
    """Fallback: build DataLoaders from DepthFMHiRISEAdapterCached (live GDAL reads)."""
    from dataset.hirise_sampler import HiRISEGeoSampler
    from dataset.mars_hirise_dtm import MarsHiRISEDTM
    from torchgeo.samplers import Units

    hc = config.data.hirise
    sc = config.data.sampler
    tc = config.training

    bbox_tuple = tuple(hc.bbox) if hc.get("bbox") else None
    ortho_type = hc.get("ortho_type", "RED")
    if isinstance(ortho_type, str):
        ortho_type = [ortho_type]

    base_dataset = MarsHiRISEDTM(
        root=hc.root,
        include_ortho=hc.get("include_ortho", True),
        ortho_type=ortho_type,
        ortho_scale=hc.get("ortho_scale"),
        download=hc.get("download", False),
        bbox=bbox_tuple,
        reuse_cache=hc.get("reuse_cache", True),
        target=hc.get("target"),
        return_meta=True,
    )
    base_dataset._raw_index = None

    split_fractions = tuple(config.data.get("split_fractions", [0.8, 0.1, 0.1]))
    split_method = config.data.get("split_method", "geographic")
    split_axis = config.data.get("split_axis", "longitude")
    n_folds = config.data.get("n_folds")
    fold_idx = config.data.get("fold_idx", 0)

    common_sampler_kwargs = dict(
        size=sc.get("size", 0.009),
        units=Units.CRS,
        split_fractions=split_fractions,
        split_method=split_method,
        split_axis=split_axis,
        n_folds=n_folds,
        fold_idx=fold_idx,
        seed=split_seed,
        reuse_cache=True,
        center_mode=sc.get("center_mode", "simple")
    )

    resolution = config.data.get("resolution", 512)
    clip = config.data.get("clip", False)
    dtm_norm = config.data.get("dtm_normalization", "relative")
    stats_path = config.data.get("stats_path")
    num_workers = tc.get("num_workers", _WORKERS_PER_GPU)

    def _build_split_loader(split: str):
        is_train = split == "train"
        sampler = HiRISEGeoSampler(
            base_dataset,
            split=split,
            length=sc.get("length") if is_train else None,
            replacement=is_train,
            **common_sampler_kwargs,
        )
        adapter = DepthFMHiRISEAdapterCached(
            base_dataset=base_dataset,
            sampler=sampler,
            resolution=resolution,
            dtm_normalization=dtm_norm,
            random_flip=is_train,
            brightness_jitter=config.data.get("brightness_jitter", 0.1) if is_train else 0.0,
            stats_path=stats_path,
            clip=clip
        )
        loader = DataLoader(
            adapter,
            batch_size=tc.per_gpu_batch_size,
            shuffle=is_train,
            num_workers=num_workers if is_train else _VAL_WORKERS,
            pin_memory=tc.pin_memory,
            prefetch_factor=tc.get("prefetch_factor", 4) if is_train else 2,
            drop_last=is_train,
            persistent_workers=is_train,
            worker_init_fn=configure_worker_logger,
        )
        logger.info(
            "Cached DataLoader [%s]: %d samples, batch_size=%d, workers=%d",
            split,
            len(adapter),
            tc.per_gpu_batch_size,
            num_workers if is_train else _VAL_WORKERS,
        )
        return split, loader

    loaders = {}
    splits = ("train", "val", "test")

    if parallel:
        logger.info("Initializing cached dataloaders in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(splits)) as executor:
            future_to_split = {executor.submit(_build_split_loader, s): s for s in splits}
            for future in concurrent.futures.as_completed(future_to_split):
                split_name = future_to_split[future]
                try:
                    _, loader = future.result()
                    loaders[split_name] = loader
                except Exception as exc:
                    logger.error(f"Failed to build DataLoader for '{split_name}': {exc}")
                    raise
    else:
        for split in splits:
            _, loader = _build_split_loader(split)
            loaders[split] = loader

    return loaders


def build_dataloaders(config, split_seed: int = 42, parallel: bool = True) -> dict:
    """Build train/val/test DataLoaders.

    Priority:
        1. LitData StreamingDataset (pre-optimized binary chunks, fastest)
        2. DepthFMHiRISEAdapterCached (live GDAL reads, slower but always works)
    """

    # ── Try LitData first ──
    try:
        loaders = _build_litdata_loaders(config, split_seed=split_seed)
        logger.info("Using LitData StreamingDataset for maximum I/O throughput.")
        return loaders
    except FileNotFoundError as e:
        logger.exception("LitData not available (%s). Falling back to cached GDAL loader.")
    except Exception as e:
        logger.exception("LitData failed unexpectedly (%s). Falling back to cached GDAL loader.")

    # ── Fallback to cached adapter ──
    return _build_cached_loaders(config, split_seed=split_seed, parallel=parallel)


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------
import torch.distributed as dist


def run_single_training(
        config,
        run_idx: int = 0,
        seed: int = 42,
        output_dir: str | Path = "outputs",
) -> dict:
    if config.training.get("num_gpus", 1) and "WORLD_SIZE" not in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        logger.warning(
            "You NEED to ensure that the DDP is already initialized for StreamingDataLoader to generate the correct "
            "dataloader length. Otherwise it will do it's dataset length based on purely the batch size rather than "
            "with the world size as well")

    if config.training.get("num_gpus", 1) and "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "You need to have DDP's WORLD_SIZE environemnt variable intialized for proper StreamingDataLoader to work run with `torchrun`")

    """Execute a single training run and return test metrics."""
    L.seed_everything(seed, workers=True)

    # Directory isolation for multi-run / K-fold
    n_folds = config.data.get("n_folds")
    fold_idx = config.data.get("fold_idx", 0)

    if n_folds is not None:
        output_dir = Path(output_dir) / f"fold_{fold_idx}" / f"run_{run_idx}"
    else:
        output_dir = Path(output_dir) / f"run_{run_idx}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip completed runs
    summary_path = output_dir / "test_summary.json"
    if summary_path.exists():
        logger.info(f"Run {run_idx} at {output_dir} is already complete. Skipping.")
        with open(summary_path, "r") as f:
            test_summary = json.load(f)

        # Load the saved dataframe so downstream plotting doesn't crash
        import pandas as pd
        test_df_path = output_dir / "test_results.csv"
        test_df = pd.read_csv(test_df_path) if test_df_path.exists() else None

        # Load the timestep ablation if it exists
        ablation_path = output_dir / "timestep_ablation.json"
        timestep_ablation = {}
        if ablation_path.exists():
            with open(ablation_path, "r") as f:
                timestep_ablation = json.load(f)

        return {
            "test_summary": test_summary,
            "test_df": test_df,
            "timestep_ablation": timestep_ablation,
            "output_dir": output_dir,
            "skipped": True,
            "val_history": [],  # Empty list to prevent KeyError in multi-run convergence plots
            "test_aggregator": None  # Object is not in memory; requires a guard in the plotting function
        }

    # Build data
    # FIXME
    # MAJOR CRITICAL CONCERN YOU NEED TO ENSURE THAT DISTRIBUTED HAS ALREADY STARTED WITH THE PROPER SETUP OTHERWISE StreamingDataLoader WILL NOT BE USING THE CORRECT SIZES AS IT WILL NOT BE CALCULATING THE LENGTHS CORRECTLY!!!!!
    loaders = build_dataloaders(config, split_seed=seed, parallel=config.data.get("parallel_load", False))

    # Build model
    module = DepthFMLightningModule(config)

    # torch.compile
    cache_path = Path(config.training.get("cache_dir", ".torch_compile_cache")) / "mega_cache.pt"
    if config.model.get("torch_compile", False):
        if cache_path.exists():
            logger.info("Loading torch.compile Mega-Cache artifacts from %s", cache_path)
            try:
                with open(cache_path, "rb") as f:
                    torch.compiler.load_cache_artifacts(f.read())
            except Exception as e:
                logger.warning("Failed to load Mega-Cache artifacts: %s", e)

        compile_mode = config.model.get("torch_compile_mode", "default")
        full_graph = config.model.get("full_graph", False)
        module.model.backbone = torch.compile(module.model.backbone, mode=compile_mode, fullgraph=full_graph)
        logger.info("torch.compile enabled on UNet backbone (mode=%s)", compile_mode)

    # Callbacks — dual checkpoint strategy
    # 1. Best by standard RMSE (traditional depth estimation metric)
    best_checkpoint = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="depthfm-best-rmse-{step}-{val/rmse_mean:.4f}",
        monitor="val/rmse_mean",
        mode="min",
        save_top_k=3,
        save_last=False,
    )
    # 2. Best by photometric consistency (resolution-independent quality)
    #    This may select a different model than RMSE when GT is low-resolution,
    #    since photo_consistency measures whether the predicted terrain
    #    correctly reproduces the observed shading in the high-res orthoimage.
    best_photo_checkpoint = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="depthfm-best-photo-{step}-{val/photo_consistency_mean:.4f}",
        monitor="val/photo_consistency_mean",
        mode="max",
        save_top_k=2,
        save_last=False,
    )
    recovery_checkpoint = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="last",
        every_n_train_steps=config.training.save_every_steps,
        save_top_k=1,
    )
    callbacks = [LearningRateMonitor(logging_interval="step"), best_checkpoint, best_photo_checkpoint,
                 recovery_checkpoint]

    if config.training.get("use_ema", True):
        logger.info("EMA is ENABLED.")
        callbacks.append(FasterEMAWeightAveraging(
            decay=config.training.get("ema_decay", False),
            device=config.training.get("ema_device", None)
        ))
    else:
        logger.info("EMA is DISABLED.")

    if config.training.get("early_stopping_patience"):
        callbacks.append(
            EarlyStopping(monitor="val/rmse_mean", patience=config.training.early_stopping_patience, mode="min")
        )

    # Logger
    wandb_logger = None
    if config.training.logger == "wandb":
        wandb_logger = WandbLogger(
            project=config.training.project_name,
            name=f"{config.training.run_name}_run{run_idx}",
            save_dir=str(output_dir),
            group=config.training.run_name,
            tags=["mars", "depthfm", "flow-matching"],
        )

    # Precision
    _prec = config.training.mixed_precision
    if _prec == "bf16":
        precision = "bf16-mixed"
    elif _prec in ("f16", "fp16"):
        precision = "16-mixed"
    else:
        precision = "32-true"

    using_litdata = loaders["train"].__class__.__name__ == "StreamingDataLoader"

    trainer = L.Trainer(
        max_steps=config.training.max_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices='auto',
        strategy="ddp" if config.training.num_gpus > 1 else "auto",
        precision=precision,
        callbacks=callbacks,
        logger=wandb_logger,
        val_check_interval=config.training.val_every_steps,
        log_every_n_steps=config.training.log_every_steps,
        gradient_clip_val=config.training.max_grad_norm,
        accumulate_grad_batches=config.training.gradient_accumulation_steps,
        enable_progress_bar=True,
        default_root_dir=str(output_dir),
        use_distributed_sampler=not using_litdata,
    )

    # Train (with resumption)
    last_ckpt_path = output_dir / "checkpoints" / "last.ckpt"

    if config.training.enable:
        if last_ckpt_path.exists():
            logger.info(f"*** Resuming run {run_idx} from {last_ckpt_path} ***")
            trainer.fit(module, loaders["train"], loaders["val"], ckpt_path=str(last_ckpt_path), weights_only=False)
        else:
            logger.info(f"*** Starting fresh training for run {run_idx} ***")
            trainer.fit(module, loaders["train"], loaders["val"])

    best_choice = config.training.get("test_choice", "rmse")

    # Test — use best RMSE checkpoint as primary
    best_path_rmse = best_checkpoint.best_model_path
    best_photo_path = best_photo_checkpoint.best_model_path

    # Log both checkpoint paths for comparison
    if trainer.is_global_zero:
        logger.info("Best RMSE checkpoint: %s", best_path_rmse or "N/A")
        logger.info("Best photo checkpoint: %s", best_photo_path or "N/A")

    if best_choice == "rmse":
        best_path = best_path_rmse
    else:
        best_path = best_photo_path

    # Fallback if training was skipped (enable=False) and best_path is empty in memory
    if not best_path:
        ckpt_dir = output_dir / "checkpoints"
        best_ckpts = list(ckpt_dir.glob(f"depthfm-best-{best_choice}-*.ckpt"))

        if best_ckpts:
            import re

            # The float metric is right before the .ckpt extension
            def extract_rmse(path):
                match = re.search(r"([0-9]+\.[0-9]+)\.ckpt$", path.name)
                return float(match.group(1)) if match else float('inf')

            # Grab the checkpoint with the lowest RMSE value
            best_path = str(min(best_ckpts, key=extract_rmse))
            logger.info(f"*** Found best {best_choice.upper()} checkpoint via glob: {best_path} ***")
        else:
            logger.warning("*** No best checkpoint found via glob, falling back to last.ckpt ***")
            best_path = str(last_ckpt_path)
    else:
        logger.info(f"*** Testing with best {best_choice.upper()} checkpoint from memory: {best_path} ***")

    # Test
    trainer.test(
        module,
        loaders["test"],
        ckpt_path=best_path,
        weights_only=False
    )

    module.to(trainer.strategy.root_device)

    # Timestep ablation
    logger.info("Running timestep ablation...")

    timestep_results = module.run_timestep_ablation(
        loaders["test"],
        step_counts=[1, 2, 4, 8, 10, 20],
        max_batches=config.training.get("ablation_max_batches"),
    )

    if trainer.is_global_zero:
        with open(output_dir / "timestep_ablation.json", "w") as f:
            json.dump({str(k): v for k, v in timestep_results.items()}, f, indent=2)

        from depth_fm.visualization import plot_timestep_ablation

        set_neurips_style()
        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig = plot_timestep_ablation(
            step_counts=sorted(timestep_results.keys()),
            metrics_per_step=timestep_results,
            primary_metric="rmse",
            secondary_metrics=["delta_1", "normal_angular_error"],
            title="Mars DTM: inference quality vs Euler steps",
            save_path=fig_dir / "timestep_ablation_rmse.pdf",
        )
        plt.close(fig)
        fig = plot_timestep_ablation(
            step_counts=sorted(timestep_results.keys()),
            metrics_per_step=timestep_results,
            primary_metric="photo_consistency",
            secondary_metrics=["delta_1", "normal_angular_error"],
            title="Mars DTM: inference quality vs Euler steps",
            save_path=fig_dir / "timestep_ablation_photo.pdf",
        )
        plt.close(fig)

        fig = plot_pareto_frontier(
            metrics_per_step=timestep_results,
            primary_metric="rmse",
            save_path=fig_dir / "pareto_frontier_rmse.pdf",
        )
        plt.close(fig)

    # Collect results
    test_summary = module._test_aggregator.summary()
    test_df = module._test_aggregator.per_sample_dataframe()

    if trainer.is_global_zero:
        if config.model.get("torch_compile", False):
            logger.info("Extracting torch.compile Mega-Cache artifacts...")
            artifacts = torch.compiler.save_cache_artifacts()
            if artifacts is not None:
                artifact_bytes, cache_info = artifacts
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    f.write(artifact_bytes)
                logger.info("Mega-Cache saved to %s. Info: %s", cache_path, cache_info)

        test_df.to_csv(output_dir / "test_results.csv", index=False)
        with open(output_dir / "test_summary.json", "w") as f:
            json.dump(test_summary, f, indent=2)

    return {
        "test_summary": test_summary,
        "val_history": deepcopy(module.val_history),
        "test_aggregator": module._test_aggregator,
        "test_df": test_df,
        "timestep_ablation": timestep_results,
        "output_dir": output_dir,
        "best_rmse_ckpt": best_path,
        "best_photo_ckpt": best_photo_path or "",
    }


# ---------------------------------------------------------------------------
# Multi-run with statistical evaluation
# ---------------------------------------------------------------------------


def run_multi_seed_experiment(config, n_runs: int = 3, base_seed: int = 42):
    """Run training multiple times with different seeds for error bars."""
    output_root = Path(config.training.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    fig_dir = output_root / "figures"
    fig_dir.mkdir(exist_ok=True)

    all_results = []
    seeds = [base_seed + i * 1000 for i in range(n_runs)]

    for run_idx, seed in enumerate(seeds):
        logger.info("=" * 60)
        logger.info("  RUN %d / %d  (seed=%d)", run_idx + 1, n_runs, seed)
        logger.info("=" * 60)
        result = run_single_training(config, run_idx=run_idx, seed=seed, output_dir=output_root)
        all_results.append(result)

    is_global_zero = int(os.environ.get("GLOBAL_RANK", os.environ.get("RANK", 0))) == 0

    if is_global_zero:
        set_neurips_style()
        logger.info("Generating publication figures...")

        fig = plot_convergence_curves(
            [r["val_history"] for r in all_results],
            metric_key="val/rmse",
            title="Validation RMSE convergence",
            save_path=fig_dir / "convergence_rmse.pdf",
        )
        plt.close(fig)

        fig = plot_convergence_curves(
            [r["val_history"] for r in all_results],
            metric_key="val/delta_1",
            title="Validation δ₁ convergence",
            save_path=fig_dir / "convergence_delta1.pdf",
        )
        plt.close(fig)

        fig = plot_multi_run_summary_table(
            [r["test_summary"] for r in all_results],
            metrics_to_show=["rmse", "abs_rel", "si_log", "delta_1", "normal_angular_error", "photo_consistency"],
            title=f"Test results ({n_runs} runs)",
            save_path=fig_dir / "multi_run_summary.pdf",
        )
        plt.close(fig)

        best_run = min(all_results, key=lambda r: r["test_summary"].get("rmse", {}).get("mean", 1e9))
        fig = plot_metric_distributions(
            best_run["test_df"],
            metrics_to_plot=["rmse", "abs_rel", "si_log", "delta_1", "normal_angular_error", "photo_consistency"],
            title="Test metric distributions (best run)",
            save_path=fig_dir / "metric_distributions.pdf",
        )
        plt.close(fig)

        _generate_patch_analysis(best_run, fig_dir, config)

        if all("timestep_ablation" in r for r in all_results):
            from depth_fm.visualization import plot_timestep_ablation

            first_ablation = all_results[0]["timestep_ablation"]
            step_counts = sorted(first_ablation.keys())
            averaged_ablation = {}
            for s in step_counts:
                merged = {}
                for metric_name in first_ablation[s]:
                    run_means = [r["timestep_ablation"][s][metric_name]["mean"] for r in all_results if
                                 s in r["timestep_ablation"]]
                    merged[metric_name] = {"mean": float(np.mean(run_means)), "std": float(np.std(run_means))}
                averaged_ablation[s] = merged

            fig = plot_timestep_ablation(
                step_counts=step_counts,
                metrics_per_step=averaged_ablation,
                primary_metric="rmse",
                secondary_metrics=["delta_1", "normal_angular_error"],
                title=f"Inference quality vs Euler steps ({n_runs}-run avg)",
                save_path=fig_dir / "timestep_ablation_averaged_rmse.pdf",
            )
            plt.close(fig)
            fig = plot_timestep_ablation(
                step_counts=step_counts,
                metrics_per_step=averaged_ablation,
                primary_metric="photo_consistency",
                secondary_metrics=["delta_1", "normal_angular_error"],
                title="Mars DTM: inference quality vs Euler steps",
                save_path=fig_dir / "timestep_ablation_averaged_photo.pdf",
            )
            plt.close(fig)

        _print_final_summary(all_results, output_root)
        logger.info("All figures saved to %s", fig_dir)


def _generate_patch_analysis(best_run: dict, fig_dir: Path, config):
    """Generate detailed per-patch analysis from the best run."""
    aggregator = best_run.get("test_aggregator")
    if aggregator is None:
        logger.warning("Test aggregator not found in memory (run was skipped). Skipping patch analysis plotting.")
        return

    worst = aggregator.worst_k("rmse", k=5)
    best = aggregator.best_k("rmse", k=5)
    logger.info("Worst 5 test patches: %s", worst)
    logger.info("Best 5 test patches: %s", best)

    analysis = {
        "worst_5_rmse": [{"tile_id": t, "rmse": float(v)} for t, v in worst],
        "best_5_rmse": [{"tile_id": t, "rmse": float(v)} for t, v in best],
        "summary": best_run["test_summary"],
    }
    with open(fig_dir / "patch_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    df = best_run["test_df"]
    import seaborn as sns

    set_neurips_style()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["rmse"], bins=30, color=sns.color_palette("flare")[2], kde=True, ax=ax, alpha=0.6)
    for tile_id, rmse_val in worst[:3]:
        ax.axvline(rmse_val, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.text(rmse_val, ax.get_ylim()[1] * 0.9, tile_id, rotation=45, fontsize=7, color="red")
    ax.set_xlabel("RMSE (m)")
    ax.set_ylabel("Count")
    ax.set_title("Test RMSE distribution with worst patches")
    save_fig(fig, fig_dir / "rmse_distribution.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    if "slope_rmse" in df.columns and "rmse" in df.columns:
        ax.scatter(df["slope_rmse"], df["rmse"], c=df["normal_angular_error"], cmap="flare", s=20, alpha=0.6)
        ax.set_xlabel("Slope RMSE (°)")
        ax.set_ylabel("Elevation RMSE (m)")
        ax.set_title("Error vs terrain complexity")
        fig.colorbar(ax.collections[0], label="Normal error (°)")
        save_fig(fig, fig_dir / "error_vs_complexity.pdf", bbox_inches="tight")
    plt.close(fig)


def _print_final_summary(all_results: list[dict], output_dir: Path):
    """Print and save the final multi-run summary."""
    import pandas as pd

    metrics_of_interest = [
        "rmse", "abs_rel", "si_log", "delta_1",
        "normal_angular_error", "slope_rmse",
        "photo_consistency", "psd_ratio",
    ]
    rows = []
    for i, r in enumerate(all_results):
        row = {"run": i}
        for m in metrics_of_interest:
            if m in r["test_summary"]:
                row[m] = r["test_summary"][m]["mean"]
        rows.append(row)

    df = pd.DataFrame(rows)
    summary_lines = ["=" * 70, "FINAL RESULTS (mean ± std across runs)", "=" * 70]
    for m in metrics_of_interest:
        if m in df.columns:
            summary_lines.append(f"  {m:<30s}  {df[m].mean():.4f} ± {df[m].std():.4f}")
    summary_lines.append("=" * 70)

    for line in summary_lines:
        logger.info(line)

    with open(output_dir / "final_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    df.to_csv(output_dir / "all_runs_metrics.csv", index=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def dataload_switch_test(config, args):
    loaders = build_dataloaders(config, split_seed=args.seed, parallel=config.data.get("parallel_load", False))

    for i in range(3):
        logger.info(f"Loading data for run {i}")
        for split, loader in loaders.items():
            logger.info(f"Starting dataloader for split {split}")
            for b in tqdm(loader):
                pass


def hash_config(config: OmegaConf):
    raw = json.dumps(OmegaConf.to_container(config, resolve=True), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:4]


def main():
    import warnings

    warnings.filterwarnings("ignore", message=r".*isinstance(treespec, LeafSpec).*")
    warnings.filterwarnings("ignore", message=r"Found \d+ module")
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    parser = argparse.ArgumentParser(description="Train Mars DepthFM")
    parser.add_argument("--config", type=str, default="configs/train_hirise.yaml")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of training runs for error bars")
    parser.add_argument("--n_folds", type=int, default=None, help="K-fold CV (overrides config)")
    parser.add_argument("--fold_idx", type=int, default=0, help="Which fold to use as test (0 to n_folds-1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analyze_topography", action="store_true")
    parser.add_argument("--analyze_masks", action="store_true")
    parser.add_argument("--view_thumbnails", action="store_true")
    parser.add_argument("--view_loss_physics", action="store_true")
    parser.add_argument("--view_loss_components", action="store_true")
    parser.add_argument("--view_invalid_fill", action="store_true")
    parser.add_argument("--view_seam_artifacts", action="store_true")
    parser.add_argument("--view_tin_artifacts", action="store_true")
    parser.add_argument("--view_solar_distribution", action="store_true")
    parser.add_argument("--view_augmentations", action="store_true")
    parser.add_argument("--view_extras", action="store_true")
    parser.add_argument("--view_lambert_ablation", action="store_true")
    parser.add_argument("--all_viz", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    if args.n_folds is not None:
        config.data.n_folds = args.n_folds
        config.data.fold_idx = args.fold_idx

    is_global_zero = int(os.environ.get("GLOBAL_RANK", os.environ.get("RANK", 0))) == 0
    config_hash = hash_config(config)

    config.training.output_dir = Path(config.training.output_dir) / config.model.get("model_type",
                                                                                     "depthfm") / config_hash

    logger.info("Saving the data to the output directory: %s", config.training.output_dir)

    # Log hardware info
    if is_global_zero:
        logger.info("Hardware: %d CPU cores detected, %d GPUs, workers/GPU=%d", _TOTAL_CORES, _NUM_GPUS_DEFAULT,
                    _WORKERS_PER_GPU)

    all_viz = args.all_viz
    # Inspection modes
    inspection = (all_viz or
                  args.analyze_masks or
                  args.view_thumbnails or
                  args.analyze_topography or
                  args.view_loss_physics or
                  args.view_loss_components or
                  args.view_invalid_fill or
                  args.view_tin_artifacts or
                  args.view_solar_distribution or
                  args.view_seam_artifacts or
                  args.view_augmentations or
                  args.view_extras or
                  args.view_lambert_ablation)
    if inspection:
        if is_global_zero:
            logger.info("Executing isolated data inspection routine...")
            L.seed_everything(args.seed, workers=True)
            # For the visualisation we do not want to use the distributed setup, just GPU 0, as such to trick the
            # creation for the dataloaders we want to have it look at the full dataset; however, it splits per
            # rank using the WORLD_SIZE environment variable
            if "WORLD_SIZE" in os.environ:
                del os.environ["WORLD_SIZE"]
                logger.warning(
                    "Because we are running in a distributed environment and we are planning to run visualisation, we disable the other GPUs")

            loaders = build_dataloaders(config, split_seed=args.seed, parallel=config.data.get("parallel_load", False))
            output_path = Path(config.training.output_dir) / "inspection"

            if args.analyze_topography:
                compute_topography_statistics(loaders["test"], split_name="Test Set")
            if args.analyze_masks:
                for split_name, loader in loaders.items():
                    compute_mask_statistics(loader, split_name=f"{split_name.capitalize()} Set")
            if all_viz or args.view_thumbnails:
                generate_thumbnail_grids(loaders["train"], output_dir=output_path, num_samples=8)
            if all_viz or args.view_loss_physics:
                visualize_loss_physics(loaders["train"], output_dir=output_path, num_samples=8)
            if all_viz or args.view_loss_components:
                visualize_loss_components(loaders["train"], output_dir=output_path, num_samples=8)
            if all_viz or args.view_invalid_fill:
                visualize_invalid_fill(loaders["train"], output_dir=output_path, num_samples=16)
            if all_viz or args.view_seam_artifacts:
                visualize_seam_artifacts(loaders["train"], output_dir=output_path, num_samples=200)
            if all_viz or args.view_tin_artifacts:
                visualize_tin_artifacts(loaders["train"], output_dir=output_path, num_samples=8)
            if all_viz or args.view_solar_distribution:
                visualize_solar_distribution(loaders["train"], output_dir=output_path)
            if all_viz or args.view_augmentations:
                visualize_random_flips_and_rotations(loaders["train"], output_dir=output_path, num_samples=2)
            if all_viz or args.view_extras:
                visualize_huber_loss(loaders["train"], output_dir=output_path, num_samples=6)
                visualize_laplacian_loss(loaders["train"], output_dir=output_path, num_samples=6)
                visualize_ordinal_ranking(loaders["train"], output_dir=output_path, num_samples=6)
            if all_viz or args.view_lambert_ablation:
                plot_spherical_loss_landscape(loaders["train"], loss_fn=PhotoclinometricLoss(), output_dir=output_path)
                plot_umap_invariance(loaders["train"], loss_fn=PhotoclinometricLoss(), output_dir=output_path)
                plot_component_ablation(loaders["train"], loss_fn=PhotoclinometricLoss(), output_dir=output_path)
                plot_qualitative_physics_errors(loaders["train"], loss_fn=PhotoclinometricLoss(), output_dir=output_path)
                plot_radial_sun_sweep(loaders["train"], loss_fn=PhotoclinometricLoss(), output_dir=output_path)

            logger.info("Data inspection complete. Exiting without training.")
        return

    # Training
    if args.n_runs > 1:
        run_multi_seed_experiment(config, n_runs=args.n_runs, base_seed=args.seed)
    else:
        result = run_single_training(config, run_idx=0, seed=args.seed, output_dir=config.training.output_dir)

        if is_global_zero:
            fig_dir = Path(config.training.output_dir) / "run_0" / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            set_neurips_style()
            fig = plot_metric_distributions(result["test_df"], save_path=fig_dir / "metrics.pdf")
            plt.close(fig)
            _generate_patch_analysis(result, fig_dir, config)
            _print_final_summary([result], Path(config.training.output_dir))


if __name__ == "__main__":
    main()
