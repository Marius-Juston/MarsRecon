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
import os
from copy import deepcopy
from pathlib import Path

from depth_fm.litdata_datamodule import _build_litdata_loaders

# ---------------------------------------------------------------------------
# GLOBAL GDAL/IO OPTIMIZATIONS (For 256-Core / NVMe setups)
# ---------------------------------------------------------------------------
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["VSI_CACHE"] = "TRUE"
os.environ["VSI_CACHE_SIZE"] = "500000000"
os.environ["GDAL_CACHEMAX"] = "10%"
os.environ["GDAL_MAX_DATASET_POOL_SIZE"] = "1024"

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
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
from tqdm import tqdm

from depth_fm.depthfm_adapter import (
    DepthFMHiRISEAdapterCached,
    estimate_sun_vector_ols, fill_dtm_smart_diffusion, )
from depth_fm.lightning_module import DepthFMLightningModule, EMACallback
from depth_fm.visualization import (
    plot_convergence_curves,
    plot_metric_distributions,
    plot_multi_run_summary_table,
    set_neurips_style,
)

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")

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


# ---------------------------------------------------------------------------
# Visualization helpers (unchanged from original)
# ---------------------------------------------------------------------------


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
    fig.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
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
                img_filled = fill_dtm_smart_diffusion(img_masked, mask, iterations=iterations)
                dtm_filled = fill_dtm_smart_diffusion(dtm_masked, mask, iterations=iterations)

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
    fig.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    logger.info(f"Smooth filling visualization saved to: {save_path}")


@torch.no_grad()
def visualize_loss_physics(
        dataloader,
        output_dir: Path,
        num_samples: int = 6,
        lunar_lambert_weight: float = 0.5
):
    """
    Visualizes the internal physics of the Photoclinometric Loss.
    Layout: [Real Ortho] | [GT DTM] | [Surface Normals] | [Lunar-Lambert Render] | [Lunar-Lambert GT Check]
    """
    logger.info(f"Generating Loss Physics visualization for {num_samples} samples (L={lunar_lambert_weight:.2f})...")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sobel filters for surface normals
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device) / 8.0
    sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=device) / 8.0
    kx = sobel_x.view(1, 1, 3, 3)
    ky = sobel_y.view(1, 1, 3, 3)

    n_cols = 5
    scale = 4
    fig, axes = plt.subplots(num_samples, n_cols, figsize=(n_cols * scale, scale * num_samples))
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

                # Extract inputs
                img = batch["image"][i: i + 1].to(device)
                dtm = batch["dtm"][i: i + 1, :1].to(device)
                mask = batch["confidence"][i: i + 1].to(device)
                sun_vec = batch["sun_vector"][i: i + 1].to(device)
                intensity = batch["intensity"][i: i + 1].to(device)
                ambient = batch["ambient"][i: i + 1].to(device)

                # Estimate GT parameters via OLS
                sun_vec_gt, intensity_gt, ambient_gt = estimate_sun_vector_ols(dtm, img, mask)

                ortho_gray = img.mean(dim=1, keepdim=True) if img.shape[1] == 3 else img
                _, _, H, W = dtm.shape
                spatial_scale = max(H, W) / 2.0

                # Compute Normals
                padded_dtm = F.pad(dtm, (1, 1, 1, 1), mode="replicate")
                n_x = -F.conv2d(padded_dtm, kx) * spatial_scale
                n_y = -F.conv2d(padded_dtm, ky) * spatial_scale
                n_z = torch.ones_like(n_x)
                normals = F.normalize(torch.cat([n_x, n_y, n_z], dim=1), p=2, dim=1)

                # Emission angle is simply the Z-normal for a top-down (Nadir) satellite view
                cos_e = normals[:, 2:3, :, :]

                # --- 1. LUNAR-LAMBERT RENDER (GT Parameters) ---
                cos_i_gt = torch.sum(normals * sun_vec_gt.view(1, 3, 1, 1), dim=1, keepdim=True)
                cos_i_clamped_gt = torch.clamp(cos_i_gt, min=0.0)

                lambert_comp_gt = cos_i_clamped_gt
                ls_comp_gt = cos_i_clamped_gt / (cos_i_clamped_gt + cos_e + 1e-6)

                render_blend_gt = (lunar_lambert_weight * lambert_comp_gt) + ((1.0 - lunar_lambert_weight) * ls_comp_gt)
                render_gt = (render_blend_gt * intensity_gt) + ambient_gt

                # --- 2. LUNAR-LAMBERT RENDER (Batch Parameters) ---
                cos_i = torch.sum(normals * sun_vec.view(1, 3, 1, 1), dim=1, keepdim=True)
                cos_i_clamped = torch.clamp(cos_i, min=0.0)

                lambert_comp = cos_i_clamped
                ls_comp = cos_i_clamped / (cos_i_clamped + cos_e + 1e-6)

                render_blend = (lunar_lambert_weight * lambert_comp) + ((1.0 - lunar_lambert_weight) * ls_comp)
                render = (render_blend * intensity) + ambient

                # --- 3. CLIPPING & MASKING ---
                img_disp = np.clip((ortho_gray[0, 0].cpu().numpy() + 1.0) / 2.0, 0.0, 1.0)
                dtm_disp = np.clip((dtm[0, 0].cpu().numpy() + 1.0) / 2.0, 0.0, 1.0)
                mask_np = mask[0, 0].cpu().numpy().astype(bool)
                normals_disp = np.clip((normals[0].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0, 0.0, 1.0)
                render_disp = np.clip(render[0, 0].cpu().numpy(), 0.0, 1.0)
                render_disp_gt = np.clip(render_gt[0, 0].cpu().numpy(), 0.0, 1.0)

                for arr in (img_disp, dtm_disp, render_disp, render_disp_gt):
                    arr[~mask_np] = np.nan
                normals_disp[~mask_np] = np.nan

                # --- 4. PLOTTING ---
                axes[count, 0].imshow(img_disp, cmap="gray", vmin=0, vmax=1)
                axes[count, 1].imshow(dtm_disp, cmap="terrain")
                axes[count, 2].imshow(normals_disp)
                axes[count, 3].imshow(render_disp, cmap="gray", vmin=0, vmax=1)
                axes[count, 4].imshow(render_disp_gt, cmap="gray", vmin=0, vmax=1)

                for ax in axes[count]:
                    ax.axis("off")

                if count == 0:
                    titles = [
                        "Real Ortho (Gray)",
                        "GT DTM",
                        "Surface Normals",
                        f"Lunar-Lambert Render (L={lunar_lambert_weight:.2f})",
                        f"Lunar-Lambert GT Check"
                    ]
                    for ax, t in zip(axes[0], titles):
                        ax.set_title(t)

                count += 1
                pbar.update()

    save_path = output_dir / "loss_physics_inspection.png"
    fig.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    logger.info(f"Loss physics visualization saved to: {save_path}")


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

    fig, axes = plt.subplots(num_samples, 6, figsize=(24, 4 * num_samples))
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
        normed[~mask] = np.nan
        return normed

    for idx in tqdm(range(num_samples), desc="Generating thumbnails"):
        img_np = np.clip((np.transpose(images[idx], (1, 2, 0)) + 1.0) / 2.0, 0.0, 1.0)
        dtm_np = np.clip((np.transpose(dtms[idx], (1, 2, 0)) + 1.0) / 2.0, 0.0, 1.0)
        mask_np = masks[idx][0].astype(bool)
        img_np[~mask_np] = np.nan
        dtm_np[~mask_np] = np.nan

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
        slope_norm[~mask_crop] = np.nan
        detrended_crop_norm = detrend_and_stretch(dtm_crop, mask_crop)

        axes[idx, 0].imshow(img_np)
        axes[idx, 1].imshow(dtm_full_1ch, cmap="terrain")
        axes[idx, 2].imshow(detrended_full_norm, cmap="terrain")
        axes[idx, 3].imshow(img_crop)
        axes[idx, 4].imshow(slope_norm, cmap="magma")
        axes[idx, 5].imshow(detrended_crop_norm, cmap="terrain")
        for ax in axes[idx]:
            ax.axis("off")
        if idx == 0:
            for ax, t in zip(
                    axes[0],
                    ["Ortho (Full)", "DTM (Full)", "Detrended (Full)", f"Ortho Zoom ({crop_size}px)",
                     f"Masked Slope ({crop_size}px)", f"Masked Detrend ({crop_size}px)"],
            ):
                ax.set_title(t)

    save_path = output_dir / "dataset_thumbnails_detailed.png"
    fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=False, facecolor="white")
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
    )

    resolution = config.data.get("resolution", 512)
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
        return {"test_summary": test_summary, "output_dir": output_dir, "skipped": True}

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
        callbacks.append(EMACallback())
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
    if config.training.get("use_ema", True) and module._ema_initialised:
        module.load_ema_weights()
    timestep_results = module.run_timestep_ablation(
        loaders["test"],
        step_counts=[1, 2, 4, 8, 10, 20],
        max_batches=config.training.get("ablation_max_batches"),
    )
    if config.training.get("use_ema", True) and module._ema_initialised:
        module.restore_training_weights()

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
            save_path=fig_dir / "timestep_ablation_rmse.pdf",
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
    aggregator = best_run["test_aggregator"]
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
    fig.savefig(fig_dir / "rmse_distribution.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    if "slope_rmse" in df.columns and "rmse" in df.columns:
        ax.scatter(df["slope_rmse"], df["rmse"], c=df["normal_angular_error"], cmap="flare", s=20, alpha=0.6)
        ax.set_xlabel("Slope RMSE (°)")
        ax.set_ylabel("Elevation RMSE (m)")
        ax.set_title("Error vs terrain complexity")
        fig.colorbar(ax.collections[0], label="Normal error (°)")
        fig.savefig(fig_dir / "error_vs_complexity.pdf", bbox_inches="tight")
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
    config.training.output_dir = Path(config.training.output_dir) / config.model.get("model_type", "depthfm")

    # Log hardware info
    if is_global_zero:
        logger.info("Hardware: %d CPU cores detected, %d GPUs, workers/GPU=%d", _TOTAL_CORES, _NUM_GPUS_DEFAULT,
                    _WORKERS_PER_GPU)

    # Inspection modes
    inspection = args.analyze_masks or args.view_thumbnails or args.analyze_topography or args.view_loss_physics or args.view_loss_components or args.view_invalid_fill
    if inspection:
        if is_global_zero:
            logger.info("Executing isolated data inspection routine...")
            L.seed_everything(args.seed, workers=True)
            loaders = build_dataloaders(config, split_seed=args.seed, parallel=config.data.get("parallel_load", False))
            output_path = Path(config.training.output_dir) / "inspection"

            if args.analyze_topography:
                compute_topography_statistics(loaders["test"], split_name="Test Set")
            if args.analyze_masks:
                for split_name, loader in loaders.items():
                    compute_mask_statistics(loader, split_name=f"{split_name.capitalize()} Set")
            if args.view_thumbnails:
                generate_thumbnail_grids(loaders["val"], output_dir=output_path, num_samples=8)
            if args.view_loss_physics:
                visualize_loss_physics(loaders["val"], output_dir=output_path, num_samples=8)
            if args.view_loss_components:
                visualize_loss_components(loaders["val"], output_dir=output_path, num_samples=8)
            if args.view_invalid_fill:
                visualize_invalid_fill(loaders["train"], output_dir=output_path, num_samples=16)

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
