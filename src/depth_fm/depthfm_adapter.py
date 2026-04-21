"""
DepthFM dataset adapter for MarsHiRISEDTM.

Wraps the existing MarsHiRISEDTM TorchGeo dataset to produce training pairs
compatible with DepthFM's flow matching pipeline:

    image:  (3, H, W) float32 in [-1, 1]  — RED orthoimage (grayscale → 3ch)
    dtm:    (3, H, W) float32 in [-1, 1]  — normalised elevation (1ch → 3ch)

Stereo augmentation: each __getitem__ randomly selects the left or right
orthoimage as the conditioning input.  Both are paired with the same DTM,
effectively doubling the training data.

Usage with the HiRISE sampler::

    from dataset.mars_hirise_dtm import MarsHiRISEDTM
    from dataset.hirise_sampler import HiRISEGeoSampler
    from depth_fm.depthfm_adapter import DepthFMHiRISEAdapter

    base = MarsHiRISEDTM(
        root="/scratch/mars_hirise_dtm",
        include_ortho=True,
        ortho_type="RED",
        download=True,
    )
    sampler = HiRISEGeoSampler(base, size=0.009, length=10000)

    adapter = DepthFMHiRISEAdapter(
        base_dataset=base,
        sampler=sampler,
        resolution=512,
        dtm_normalization="relative",
        stats_path="dataset_stats/dtm/dataset_stats.json",
    )

    # Standard PyTorch DataLoader
    loader = DataLoader(adapter, batch_size=2, num_workers=32, shuffle=True)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional
from typing import Literal

import cv2
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn.functional as F
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from scipy import ndimage
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from typing_extensions import deprecated

from dataset.hirise_sampler import HiRISEGeoSampler
from dataset.mars_hirise_dtm import MarsHiRISEDTM
from depth_fm.scalers import GlobalLogNormalizer, TrainingNormResult, \
    LocalStripOrthoNormalizer

logger = logging.getLogger(__name__)

# Hardcoded fallback quantiles derived from dataset_stats/dtm/dataset_stats.json
# (Olympus Mons region, 52 k patches)
_DEFAULT_ELEV_P02 = -4396.5664071121255
_DEFAULT_ELEV_P98 = 20757.65899590482
_DEFAULT_IMG_P02 = 0.06526107076433259  # average of left_red / right_red p02
_DEFAULT_IMG_P98 = 0.1828855234319496  # average of left_red / right_red p98
# Scale factor for relative-topography mode: 98th-percentile of patch-centred
# elevation distribution (metres).  98 % of patches stay within [-1, 1] before
# clamping while physical slope magnitudes remain consistent across the dataset.
_DEFAULT_ELEV_SCALE = 45.90752235993998  # centered_p98[elevation] from Olympus stats


def _load_quantiles(
        stats_path: str | None,
) -> tuple[float, float, float, float, float]:
    """Load elevation and image p02/p98, plus elevation scale, from dataset_stats JSON.

    Returns:
        (elev_p02, elev_p98, img_p02, img_p98, elev_scale)
    """
    if stats_path is None:
        return (
            _DEFAULT_ELEV_P02, _DEFAULT_ELEV_P98,
            _DEFAULT_IMG_P02, _DEFAULT_IMG_P98,
            _DEFAULT_ELEV_SCALE,
        )

    path = Path(stats_path)
    if not path.exists():
        logger.warning("Stats file not found: %s — using hardcoded defaults", stats_path)
        return (
            _DEFAULT_ELEV_P02, _DEFAULT_ELEV_P98,
            _DEFAULT_IMG_P02, _DEFAULT_IMG_P98,
            _DEFAULT_ELEV_SCALE,
        )

    with open(path) as f:
        stats = json.load(f)

    channels = stats["channels"]  # ["elevation", "left_red", "right_red"]
    p02 = stats["p02"]
    p98 = stats["p98"]
    centered_p98 = stats["centered_p98"]

    elev_idx = channels.index("elevation")
    left_idx = channels.index("left_red") if "left_red" in channels else None
    right_idx = channels.index("right_red") if "right_red" in channels else None

    elev_p02 = p02[elev_idx]
    elev_p98 = p98[elev_idx]
    # Symmetric scale: use centered_p98 so 98 % of patch relief lands in [-1, 1]
    elev_scale = centered_p98[elev_idx]

    if left_idx is not None and right_idx is not None:
        img_p02 = (p02[left_idx] + p02[right_idx]) / 2.0
        img_p98 = (p98[left_idx] + p98[right_idx]) / 2.0
    elif left_idx is not None:
        img_p02, img_p98 = p02[left_idx], p98[left_idx]
    elif right_idx is not None:
        img_p02, img_p98 = p02[right_idx], p98[right_idx]
    else:
        img_p02, img_p98 = _DEFAULT_IMG_P02, _DEFAULT_IMG_P98

    return elev_p02, elev_p98, img_p02, img_p98, elev_scale


def _normalize_dtm_per_patch(
        elevation: torch.Tensor,
        valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Dynamic per-patch normalization to stretch sub-meter craters to [-1, 1]."""
    valid = valid_mask.bool()
    if not valid.any():
        return torch.zeros_like(elevation)

    valid_pixels = elevation[valid]
    patch_min = valid_pixels.min()
    patch_max = valid_pixels.max()
    relief = patch_max - patch_min

    if relief < 1e-4:  # Prevent division by zero on flat patches
        return torch.zeros_like(elevation)

    normed = ((elevation - patch_min) / relief) * 2.0 - 1.0
    return torch.where(valid, normed, torch.zeros_like(normed))


def _safe_resize(tensor: torch.Tensor, size: int, is_mask: bool = False, has_nans: bool = False) -> torch.Tensor:
    """Resizes tensors safely using PyTorch, avoiding NaN poisoning."""
    if tensor.shape[-2] == size and tensor.shape[-1] == size:
        return tensor

    # Masks MUST use nearest neighbor so edges aren't blurred
    if is_mask:
        return F.interpolate(tensor.unsqueeze(0), size=(size, size), mode='nearest-exact').squeeze(0)

    # If it's a DTM with NaNs, temporarily swap NaNs for 0 to prevent bilinear poisoning
    if has_nans:
        valid = ~torch.isnan(tensor)
        safe_tensor = tensor.clone()
        safe_tensor[~valid] = 0.0

        resized_tensor = F.interpolate(safe_tensor.unsqueeze(0), size=(size, size), mode='bilinear',
                                       align_corners=False).squeeze(0)
        resized_valid = F.interpolate(valid.float().unsqueeze(0), size=(size, size), mode='nearest-exact').squeeze(
            0).bool()

        resized_tensor[~resized_valid] = float('nan')
        return resized_tensor

    # Standard orthoimage resizing
    return F.interpolate(tensor.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False).squeeze(0)


def fill_invalid_nearest_neighbor(
        tensor: torch.Tensor,
        valid_mask: torch.Tensor
) -> torch.Tensor:
    """
    Fill invalid regions using nearest-neighbor propagation via EDT.

    Supports:
        (H, W)
        (C, H, W)
        (B, C, H, W)
    """
    device = tensor.device
    dtype = tensor.dtype

    arr_np = tensor.cpu().numpy()
    mask_np = valid_mask.squeeze().cpu().numpy() > 0  # (H, W)

    # Degenerate cases
    if mask_np.all() or not mask_np.any():
        return tensor

    # Get nearest valid pixel indices
    indices = ndimage.distance_transform_edt(
        ~mask_np,
        return_distances=False,
        return_indices=True
    )
    # indices: (2, H, W)

    iy, ix = indices  # each (H, W)

    # --------------------------------------------------
    # CASE 1: (H, W)
    # --------------------------------------------------
    if arr_np.ndim == 2:
        filled_np = arr_np[iy, ix]

    # --------------------------------------------------
    # CASE 2: (C, H, W)
    # --------------------------------------------------
    elif arr_np.ndim == 3:
        C = arr_np.shape[0]

        c_idx = np.arange(C)[:, None, None]  # (C,1,1)
        iy_b = iy[None, :, :]  # (1,H,W)
        ix_b = ix[None, :, :]  # (1,H,W)

        filled_np = arr_np[c_idx, iy_b, ix_b]  # (C,H,W)

    # --------------------------------------------------
    # CASE 3: (B, C, H, W)
    # --------------------------------------------------
    elif arr_np.ndim == 4:
        B, C = arr_np.shape[:2]

        b_idx = np.arange(B)[:, None, None, None]  # (B,1,1,1)
        c_idx = np.arange(C)[None, :, None, None]  # (1,C,1,1)

        iy_b = iy[None, None, :, :]  # (1,1,H,W)
        ix_b = ix[None, None, :, :]  # (1,1,H,W)

        filled_np = arr_np[b_idx, c_idx, iy_b, ix_b]  # (B,C,H,W)

    else:
        raise ValueError(f"Unsupported tensor dimension: {arr_np.ndim}")

    return torch.from_numpy(filled_np).to(device=device, dtype=dtype)


def fill_voids_kriging(
        image: torch.Tensor,
        dtm: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        erode_radius: int = 2,
        max_training_points: int = 2500,
        variogram_model: str = "linear",
        seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fill voids in an orthoimage and DTM with a single kriging pass per channel.

    Optimised for 512x512 tiles: subsamples valid pixels for training,
    then predicts every void pixel at once. No iteration, no per-void
    labelling, no diffusion loop.

    Parameters
    ----------
    image : torch.Tensor
        (C, H, W) or (1, C, H, W), normalised to [-1, 1].
    dtm : torch.Tensor
        (1, H, W) or (1, 1, H, W), normalised to [-1, 1].
    valid_mask : torch.Tensor
        (1, H, W) or (1, 1, H, W), binary (1 = valid, 0 = void).
    erode_radius : int
        Pixels to erode from mask edges before filling.
    max_training_points : int
        Cap on valid pixels used for kriging (controls speed).
        2500 keeps fit under ~1s per channel on CPU.
    variogram_model : str
        PyKrige variogram model. 'linear' is best for slope continuation.
    seed : int
        RNG seed for reproducible subsampling.

    Returns
    -------
    (filled_image, filled_dtm) with the same shapes/dtypes as inputs.
    """
    rng = np.random.default_rng(seed)

    # --- shape bookkeeping ------------------------------------------------
    img_3d = image.ndim == 3
    dtm_3d = dtm.ndim == 3
    msk_3d = valid_mask.ndim == 3

    if img_3d:
        image = image.unsqueeze(0)
    if dtm_3d:
        dtm = dtm.unsqueeze(0)
    if msk_3d:
        valid_mask = valid_mask.unsqueeze(0)

    device = image.device
    img_dtype = image.dtype
    dtm_dtype = dtm.dtype

    # --- erode and convert to numpy ---------------------------------------
    eroded = erode_valid_mask(valid_mask, erode_radius)
    valid = eroded[0, 0].cpu().numpy() > 0.5  # (H, W) bool
    void = ~valid

    img_np = np.nan_to_num(image[0].cpu().numpy(), nan=0.0).astype(np.float64)
    dtm_np = np.nan_to_num(dtm[0].cpu().numpy(), nan=0.0).astype(np.float64)

    if not void.any():
        if img_3d:
            image = image.squeeze(0)
        if dtm_3d:
            dtm = dtm.squeeze(0)
        return image, dtm

    # --- coordinates -------------------------------------------------------
    y_valid, x_valid = np.where(valid)
    y_void, x_void = np.where(void)

    n_valid = len(y_valid)
    if n_valid < 6:
        logger.warning("Fewer than 6 valid pixels — returning inputs unchanged.")
        if img_3d:
            image = image.squeeze(0)
        if dtm_3d:
            dtm = dtm.squeeze(0)
        return image, dtm

    # subsample training set
    if n_valid > max_training_points:
        idx = rng.choice(n_valid, size=max_training_points, replace=False)
        y_train = y_valid[idx].astype(np.float64)
        x_train = x_valid[idx].astype(np.float64)
    else:
        y_train = y_valid.astype(np.float64)
        x_train = x_valid.astype(np.float64)

    y_pred = y_void.astype(np.float64)
    x_pred = x_void.astype(np.float64)

    filled_img = img_np.copy()
    filled_dtm = dtm_np.copy()

    # --- DTM: Universal Kriging (slope continuation) ----------------------
    for c in range(dtm_np.shape[0]):
        z_train = dtm_np[c, y_train.astype(int), x_train.astype(int)]
        try:
            uk = UniversalKriging(
                x_train, y_train, z_train,
                variogram_model=variogram_model,
                drift_terms=["regional_linear"],
                verbose=False,
                enable_plotting=False,
            )
            z_pred, _ = uk.execute("points", x_pred, y_pred)
            filled_dtm[c, y_void, x_void] = np.asarray(z_pred).ravel()
        except Exception as e:
            logger.warning("DTM kriging failed (ch %d): %s", c, e)

    # --- Image: Ordinary Kriging (smooth colour) --------------------------
    for c in range(img_np.shape[0]):
        z_train = img_np[c, y_train.astype(int), x_train.astype(int)]
        try:
            ok = OrdinaryKriging(
                x_train, y_train, z_train,
                variogram_model=variogram_model,
                verbose=False,
                enable_plotting=False,
            )
            z_pred, _ = ok.execute("points", x_pred, y_pred)
            filled_img[c, y_void, x_void] = np.asarray(z_pred).ravel()
        except Exception as e:
            logger.warning("Image kriging failed (ch %d): %s", c, e)

    # --- back to torch ----------------------------------------------------
    filled_img_t = (
        torch.from_numpy(filled_img).unsqueeze(0).to(device=device, dtype=img_dtype)
    )
    filled_dtm_t = (
        torch.from_numpy(filled_dtm).unsqueeze(0).to(device=device, dtype=dtm_dtype)
    )

    if img_3d:
        filled_img_t = filled_img_t.squeeze(0)
    if dtm_3d:
        filled_dtm_t = filled_dtm_t.squeeze(0)

    return filled_img_t, filled_dtm_t


def fill_dtm_smart_diffusion(
        tensor: torch.Tensor,
        valid_mask: torch.Tensor,
        iterations: int = 64,
        erode_radius: int = 2
) -> torch.Tensor:
    """
    Fills invalid regions using Laplacian diffusion (solving the heat equation).
    """
    is_3d = tensor.ndim == 3
    if is_3d:
        tensor = tensor.unsqueeze(0)
        valid_mask = valid_mask.unsqueeze(0)

    B, C, H, W = tensor.shape
    device = tensor.device
    dtype = tensor.dtype

    # 1. Sanitize the inputs
    tensor = torch.nan_to_num(tensor, nan=0.0)

    # 2. Mask Erosion (Using the corrected function)
    valid_bool = erode_valid_mask(valid_mask, erode_radius) > 0.5

    # 3. Calculate the global mean
    valid_sum = (tensor * valid_bool.to(dtype)).sum(dim=[-2, -1], keepdim=True)
    valid_count = valid_bool.sum(dim=[-2, -1], keepdim=True).clamp(min=1.0)
    global_mean = valid_sum / valid_count

    # 4. Initialize working tensor
    filled = torch.where(valid_bool, tensor, global_mean)

    # 5. Create averaging kernel
    kernel = torch.ones((C, 1, 3, 3), device=device, dtype=dtype) / 9.0

    # 6. Laplacian Diffusion
    for _ in range(iterations):
        # -------------------------------------------------------------------------
        # FIX: Pad the image using 'replicate' instead of F.conv2d's default zeros.
        # This stops the physical outside edges of the image tensor from dragging
        # the boundary values down to 0.0 during the blur phase.
        # -------------------------------------------------------------------------
        padded_filled = F.pad(filled, pad=(1, 1, 1, 1), mode='replicate')
        blurred = F.conv2d(padded_filled, kernel, padding=0, groups=C)

        # Re-clamp safely eroded regions back to ground-truth
        filled = torch.where(valid_mask, tensor, blurred)

    if is_3d:
        return filled.squeeze(0)

    return filled


# ---------------------------------------------------------------------------
# Mask erosion
# ---------------------------------------------------------------------------

def erode_valid_mask(valid_mask: torch.Tensor, erode_radius: int = 1) -> torch.Tensor:
    """Erode a binary mask to trim noisy boundary pixels."""
    if erode_radius <= 0:
        return valid_mask

    is_3d = valid_mask.ndim == 3
    if is_3d:
        valid_mask = valid_mask.unsqueeze(0)

    kernel_size = 2 * erode_radius + 1
    padded_mask = F.pad(
        valid_mask,
        pad=(erode_radius, erode_radius, erode_radius, erode_radius),
        mode="constant",
        value=1.0,
    )
    eroded_mask = -F.max_pool2d(
        -padded_mask, kernel_size=kernel_size, stride=1, padding=0
    )
    eroded_mask = (eroded_mask > 0.5).float()

    if is_3d:
        return eroded_mask.squeeze(0)
    return eroded_mask


# ---------------------------------------------------------------------------
# Vectorised graph Laplacian
# ---------------------------------------------------------------------------
@lru_cache
def _build_grid_laplacian(H: int, W: int, connectivity: int = 4) -> sp.csc_matrix:
    """
    Build graph Laplacian L = D - W for an H×W grid.
    Fully vectorised — no Python loops over pixels.
    """
    N = H * W
    idx = np.arange(N).reshape(H, W)

    # Collect all (i, j) neighbour pairs
    pairs = []

    # Horizontal: every pixel to its right neighbour
    left = idx[:, :-1].ravel()
    right = idx[:, 1:].ravel()
    pairs.append((left, right))

    # Vertical: every pixel to its lower neighbour
    top = idx[:-1, :].ravel()
    bot = idx[1:, :].ravel()
    pairs.append((top, bot))

    if connectivity == 8:
        # Diagonal ↘
        tl = idx[:-1, :-1].ravel()
        br = idx[1:, 1:].ravel()
        pairs.append((tl, br))
        # Diagonal ↙
        tr = idx[:-1, 1:].ravel()
        bl = idx[1:, :-1].ravel()
        pairs.append((tr, bl))

    # Stack all edges
    src = np.concatenate([p[0] for p in pairs] + [p[1] for p in pairs])
    dst = np.concatenate([p[1] for p in pairs] + [p[0] for p in pairs])

    # Off-diagonal: W_{ij} = -1 for neighbours
    n_edges = len(src)
    off_diag = sp.coo_matrix(
        (-np.ones(n_edges), (src, dst)), shape=(N, N)
    )

    # Diagonal: D_{ii} = degree of node i
    degree = np.zeros(N)
    np.add.at(degree, src, 1.0)
    diag = sp.diags(degree, format="coo")

    L = (diag + off_diag).tocsc()
    return L


# ---------------------------------------------------------------------------
# Conditional GMRF solve
# ---------------------------------------------------------------------------

def _gmrf_fill_channel(
        channel: np.ndarray,
        void_idx: np.ndarray,
        obs_idx: np.ndarray,
        Q: sp.csc_matrix,
        nugget: float,
) -> np.ndarray:
    """
    Fill void pixels via  u_void = -Q_vv⁻¹ Q_vo u_obs.
    """
    flat = channel.ravel().astype(np.float64)

    Q_vv = Q[np.ix_(void_idx, void_idx)] + nugget * sp.eye(len(void_idx), format="csc")
    Q_vo = Q[np.ix_(void_idx, obs_idx)]

    rhs = -Q_vo @ flat[obs_idx]

    try:
        u_void = spla.spsolve(Q_vv, rhs)
    except Exception as e:
        logger.warning("spsolve failed (%s), using LSQR.", e)
        u_void = spla.lsqr(Q_vv, rhs)[0]

    filled = flat.copy()
    filled[void_idx] = u_void
    return filled.reshape(channel.shape)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
# TODO instead of using GMRF, use a more approate distribution, especailly for the height maps since that does not follow a gaussian distribution but probably closer to a log normal distribution, or specifically from the
def fill_voids_gmrf(
        image: torch.Tensor,
        dtm: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        erode_radius: int = 2,
        connectivity: int = 4,
        tau: float = 1.0,
        nugget: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fill voids in an orthoimage and DTM using GMRF conditional distribution.

    One sparse linear solve per channel. No iteration. O(n^{3/2}) on 2D grids.

    Parameters
    ----------
    image : torch.Tensor
        (C, H, W) or (1, C, H, W).
    dtm : torch.Tensor
        (1, H, W) or (1, 1, H, W).
    valid_mask : torch.Tensor
        (1, H, W) or (1, 1, H, W), binary (1 = valid, 0 = void).
    erode_radius : int
        Pixels to erode from mask edges.
    connectivity : int
        4 (rook) or 8 (queen) pixel neighbours.
    tau : float
        Precision scale (higher = stronger spatial smoothing).
    nugget : float
        Small diagonal regularisation for numerical stability.

    Returns
    -------
    (filled_image, filled_dtm, eroded_mask)
        filled_image, filled_dtm: same shapes/dtypes as inputs, voids filled.
        eroded_mask: the post-erosion binary mask, same shape as valid_mask input.
            After filling, all pixels are valid, but this mask records which
            pixels were considered trustworthy (1) vs infilled (0).
    """
    # --- shape bookkeeping ------------------------------------------------
    img_3d = image.ndim == 3
    dtm_3d = dtm.ndim == 3
    msk_3d = valid_mask.ndim == 3

    if img_3d:
        image = image.unsqueeze(0)
    if dtm_3d:
        dtm = dtm.unsqueeze(0)
    if msk_3d:
        valid_mask = valid_mask.unsqueeze(0)

    device = image.device
    img_dtype = image.dtype
    dtm_dtype = dtm.dtype

    # --- erode and convert to numpy ---------------------------------------
    eroded = erode_valid_mask(valid_mask, erode_radius)
    valid = eroded[0, 0].cpu().numpy() > 0.5

    img_np = np.nan_to_num(image[0].cpu().numpy(), nan=0.0).astype(np.float64)
    dtm_np = np.nan_to_num(dtm[0].cpu().numpy(), nan=0.0).astype(np.float64)

    if not (~valid).any():
        eroded_out = eroded
        if img_3d:
            image = image.squeeze(0)
        if dtm_3d:
            dtm = dtm.squeeze(0)
        if msk_3d:
            eroded_out = eroded_out.squeeze(0)
        return image, dtm, eroded_out

    H, W = valid.shape

    # --- build Laplacian and index sets (shared across channels) ----------
    L = _build_grid_laplacian(H, W, connectivity)
    Q = tau * L

    obs_idx = np.where(valid.ravel())[0]
    void_idx = np.where(~valid.ravel())[0]

    logger.debug(
        "GMRF fill: %d void pixels (%.1f%%), %d-connected, H=%d W=%d",
        len(void_idx), 100.0 * len(void_idx) / (H * W), connectivity, H, W,
    )

    # --- fill each channel ------------------------------------------------
    filled_dtm = dtm_np.copy()
    filled_img = img_np.copy()

    for c in range(dtm_np.shape[0]):
        filled_dtm[c] = _gmrf_fill_channel(dtm_np[c], void_idx, obs_idx, Q, nugget)

    for c in range(img_np.shape[0]):
        filled_img[c] = _gmrf_fill_channel(img_np[c], void_idx, obs_idx, Q, nugget)

    # --- back to torch ----------------------------------------------------
    filled_img_t = torch.from_numpy(filled_img).unsqueeze(0).to(device=device, dtype=img_dtype)
    filled_dtm_t = torch.from_numpy(filled_dtm).unsqueeze(0).to(device=device, dtype=dtm_dtype)

    if img_3d:
        filled_img_t = filled_img_t.squeeze(0)
    if dtm_3d:
        filled_dtm_t = filled_dtm_t.squeeze(0)
    if msk_3d:
        eroded = eroded.squeeze(0)

    return filled_img_t, filled_dtm_t, eroded


def compute_piecewise_linearity(seam_heatmap: np.ndarray, threshold_ratio: float = 0.5) -> tuple[float, np.ndarray]:
    """
    Analyzes the seam heatmap to see if the high-scoring pixels form
    a network of distinct straight lines (handles corners).
    Returns a score from 0.0 (curved/messy) to 1.0 (highly structured/linear).
    """
    # 1. Normalize and threshold the heatmap to isolate the strongest signals
    heatmap_norm = cv2.normalize(seam_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary_map = cv2.threshold(heatmap_norm, int(255 * threshold_ratio), 255, cv2.THRESH_BINARY)

    # 2. Extract the skeleton/edges of the high-scoring regions
    edges = cv2.Canny(binary_map, 50, 150, apertureSize=3)

    # 3. Probabilistic Hough Transform to find line segments
    # Adjust minLineLength based on your expected tile size
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50,
                            minLineLength=40, maxLineGap=10)

    if lines is None:
        return 0.0  # No straight lines found at all

    # 4. Calculate what percentage of the high-scoring pixels belong to these straight segments
    # Create a blank mask to draw the found Hough lines
    line_mask = np.zeros_like(binary_map)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=2)

    # Count pixels that are BOTH in the original binary map AND covered by the Hough lines
    valid_signal_pixels = np.count_nonzero(binary_map)
    if valid_signal_pixels == 0:
        return 0.0

    structured_pixels = np.count_nonzero(cv2.bitwise_and(binary_map, line_mask))

    # 5. Return the ratio. True seams (even with corners) will be close to 1.0
    # Natural curves will fall apart in the Hough transform and score low.
    return structured_pixels / valid_signal_pixels, line_mask


# ---------------------------------------------------------------------------
# data container
# ---------------------------------------------------------------------------
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
    span: float  # <--- Checks length (kills craters)
    sparsity: float  # <--- Checks density (kills dunes)
    composite_score: float
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


def compute_artifact_multipliers(seam_heatmap: np.ndarray, valid_mask: np.ndarray) -> tuple[
    float, float, np.ndarray, np.ndarray]:
    """
    Calculates two structural multipliers (0.0 to 1.0):
    1. span_ratio: Defeats short craters. True seams cross the whole tile (high span).
    2. sparsity: Defeats dense dunes. True seams are a singular line.
    """
    max_val = np.max(seam_heatmap)
    if max_val <= 0: return 0.0, 1.0

    # Isolate the core 40% of the strongest signal
    hot_mask = (seam_heatmap > max_val * 0.4)

    # --- 1. SPARSITY (Defeats repeating textures like dunes) ---
    hot_area = np.count_nonzero(hot_mask & valid_mask)
    valid_area = max(np.count_nonzero(valid_mask), 1)
    density = hot_area / valid_area

    # A true seam covers ~2-4% of the image. A dune field covers > 15%.
    # Exponential decay penalizes dense areas heavily.
    sparsity = math.exp(-density * 15.0)

    # --- 2. STRUCTURAL SPAN (Defeats short, isolated craters) ---
    binary = (hot_mask.astype(np.uint8)) * 255

    # Morphological close to bridge any tiny gaps in a real seam
    kernel = np.ones((9, 9), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    span_ratio = 0.0
    if num_labels > 1:
        H, W = seam_heatmap.shape
        max_span = 0.0

        # Skip background (label 0), find the longest continuous blob
        for i in range(1, num_labels):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            span = math.hypot(w, h)
            if span > max_span:
                max_span = span

        # Compare against the image bounds.
        # A perfectly vertical seam spans H. We cap at 1.0 for diagonal seams.
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
        exclusion_zone: int = 12
) -> tuple[float, Optional[tuple]]:
    """
    Takes a perpendicular slice across the seam.
    Returns ratio of the central peak to the surrounding parallel background,
    plus the raw profile data arrays for visualization.
    """
    H, W = score_map.shape
    device = score_map.device

    # Perpendicular vector to the detected seam
    px = -math.sin(angle_rad)
    py = math.cos(angle_rad)

    # Sample points along the perpendicular cross-section
    t = torch.arange(-profile_length, profile_length + 1, device=device, dtype=torch.float32)
    grid_x = best_x + t * px
    grid_y = best_y + t * py

    # Filter out of bounds
    in_bounds = (grid_x >= 0) & (grid_x < W) & (grid_y >= 0) & (grid_y < H)
    t = t[in_bounds]
    grid_x = grid_x[in_bounds]
    grid_y = grid_y[in_bounds]

    if len(t) == 0: return 0.0, None

    # Normalize for grid_sample [-1, 1]
    norm_x = (grid_x / (W - 1)) * 2 - 1
    norm_y = (grid_y / (H - 1)) * 2 - 1
    grid = torch.stack([norm_x, norm_y], dim=-1).view(1, 1, -1, 2)

    score_map_4d = score_map.unsqueeze(0).unsqueeze(0)
    valid_mask_4d = valid_mask.unsqueeze(0).unsqueeze(0)

    # Extract heatmap values and validity
    samples = F.grid_sample(score_map_4d, grid, mode='bilinear', align_corners=True).squeeze()
    v_samples = F.grid_sample(valid_mask_4d, grid, mode='nearest', align_corners=True).squeeze()

    # Define the "Seam" region vs the "Parallel Neighbors" region
    center_mask = (t.abs() <= exclusion_zone) & (v_samples > 0)
    bg_mask = (t.abs() > exclusion_zone) & (v_samples > 0)

    # Package profile data for viz
    profile_data = (
        t.cpu().numpy(),
        samples.cpu().numpy(),
        center_mask.cpu().numpy(),
        bg_mask.cpu().numpy()
    )

    if not center_mask.any(): return 0.0, profile_data
    if not bg_mask.any(): return 5.0, profile_data  # If there is no valid background, assume it's an isolated edge

    # Compare the max of the suspected seam to the average of the surrounding terrain
    center_max = samples[center_mask].amax().item()
    bg_mean = samples[bg_mask].mean().item()

    if bg_mean < 1e-6:
        return 20.0, profile_data  # Extremely isolated (avoids division by zero)

    return center_max / bg_mean, profile_data


# ---------------------------------------------------------------------------
# helpers
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
    """
    Calculates gradient magnitude using strict 1D differences [-1, 1]
    to prevent the spatial blurring of a standard 3x3 Sobel.
    Ensures 1-3 pixel step edges remain perfectly sharp.
    """
    device, dtype = x.device, x.dtype
    safe = x.clone()
    safe[~valid_mask] = 0.0

    # Strict 1D Forward Difference Kernels
    kx = torch.tensor([-1., 1.], device=device, dtype=dtype).view(1, 1, 1, 2)
    ky = torch.tensor([[-1.], [1.]], device=device, dtype=dtype).view(1, 1, 2, 1)

    # Pad the right and bottom edges by 1 pixel before convolution
    # so the output shape exactly matches the input shape (H, W)
    padded_x = F.pad(safe, (0, 1, 0, 0), mode='replicate')
    padded_y = F.pad(safe, (0, 0, 0, 1), mode='replicate')

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
            xc = int(round(kc + t * dx));
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


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
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
        return_diagnostics: bool = False,
) -> SeamResult:
    """
    Returns a SeamResult. When return_diagnostics=True, the result also carries
    a seam_heatmap, cohens_d_heatmap and per_angle_max vector — used by the
    visualization and refinement UI to show *where* and *along which angle*
    the detector fired.
    """
    ortho = _ensure_bchw(ortho).float()
    elevation = _ensure_bchw(elevation).float()
    valid_mask = _ensure_bchw(valid_mask).bool()
    device, dtype = ortho.device, ortho.dtype
    ortho_gray = ortho.mean(dim=1, keepdim=True)

    H, W = ortho_gray.shape[-2:]

    # gradients
    # ortho_grad = _sobel_mag(ortho_gray, valid_mask)
    # dtm_grad = _sobel_mag(elevation, valid_mask)

    ortho_grad = _sharp_grad_mag(ortho_gray, valid_mask)
    dtm_grad = _sharp_grad_mag(elevation, valid_mask)

    # erode mask to avoid scoring physical edges
    pad_e = erosion_kernel // 2
    invalid_f = (~valid_mask).float()
    dilated = F.max_pool2d(invalid_f, kernel_size=erosion_kernel, stride=1, padding=pad_e)
    eroded_valid = (dilated == 0.0)
    ev_f = eroded_valid.float()

    empty_result = SeamResult(
        seam_score=0.0, ortho_score=0.0, dtm_score=0.0, cohens_d=0.0,
        best_angle_rad=0.0, best_y=H // 2, best_x=W // 2,
        line_length=line_length, num_angles=num_angles,
        composite_score=0, sparsity=0, span=0
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

    # Add DTM moments
    mu_l_dtm, var_l_dtm, _ = _moments(elevation, left_ker)
    mu_r_dtm, var_r_dtm, _ = _moments(elevation, right_ker)

    # Calculate pooled standard deviation for the DTM
    global_dtm_std = elevation[valid_mask].std().clamp(min=1e-4)
    dtm_std_floor = 0.1 * global_dtm_std
    pooled_dtm_std = torch.sqrt(((var_l_dtm + var_r_dtm) / 2.0).clamp(min=0.0)) + dtm_std_floor

    # Calculate DTM Cohen's d
    dtm_cohens_d = (mu_l_dtm - mu_r_dtm).abs() / pooled_dtm_std

    min_line = min_valid_ratio * line_length
    min_side = min_valid_ratio * line_length * 0.5
    valid_q = (line_cnt >= min_line) & (n_l >= min_side) & (n_r >= min_side)
    if not valid_q.any():
        return empty_result

    ortho_norm = ortho_line_avg / bg_ortho
    dtm_norm = dtm_line_avg / bg_dtm
    gradient_term = ortho_weight * ortho_norm + dtm_weight * dtm_norm

    # Update the distribution gate to trigger if EITHER the ortho OR the dtm has a massive shift
    # distribution_gate = 1.0 + torch.maximum(cohens_d, dtm_cohens_d).clamp(min=0.0)
    distribution_gate = torch.clamp(torch.maximum(cohens_d, dtm_cohens_d), min=0.1, max=3.0)

    combined = gradient_term * distribution_gate  # shape (B, A, H, W)

    # mask invalid -> very negative so argmax skips them
    combined_masked = torch.where(valid_q, combined, torch.full_like(combined, -1e10))

    # argmax in (A, H, W) for batch 0
    cm0 = combined_masked[0]
    flat = cm0.flatten().argmax()
    A = num_angles
    a_idx = int(flat // (H * W))
    rem = int(flat % (H * W))
    y_idx = rem // W
    x_idx = rem % W
    best_angle_rad = math.pi * a_idx / num_angles

    seam_score = float(cm0.amax().item())

    heatmap = cm0.amax(dim=0).cpu().numpy()
    heatmap[heatmap < 0] = 0.0

    valid_np = ev_f[0, 0].cpu().numpy().astype(bool)
    span, sparsity, diag_hot_mask, diag_labels = compute_artifact_multipliers(heatmap, valid_np)

    linearity, diag_hough_lines = compute_piecewise_linearity(heatmap, threshold_ratio=0.5)

    isolation_score, diag_iso_profile = compute_spatial_isolation(
        score_map=cm0.amax(dim=0),
        valid_mask=ev_f[0, 0],
        best_x=int(x_idx),
        best_y=int(y_idx),
        angle_rad=best_angle_rad
    )

    isolation_mult = min(max((isolation_score - 1.5) / 1.5, 0.0), 1.0)

    result = SeamResult(
        seam_score=seam_score,
        ortho_score=float(torch.where(valid_q, ortho_norm, torch.full_like(ortho_norm, -1e10)).amax().item()),
        dtm_score=float(torch.where(valid_q, dtm_norm, torch.full_like(dtm_norm, -1e10)).amax().item()),
        cohens_d=float(torch.where(valid_q, cohens_d, torch.zeros_like(cohens_d)).amax().item()),
        best_angle_rad=best_angle_rad,
        best_y=int(y_idx),
        best_x=int(x_idx),
        line_length=line_length,
        span=span,  # <--- Add to output
        sparsity=sparsity,  # <--- Add to output,
        num_angles=num_angles,
        composite_score=seam_score * (0.5 + 0.5 * span) * sparsity * linearity * isolation_mult
    )

    if return_diagnostics:
        # Cohen's d heatmap: max over angles at each pixel (keep invalid as nan)
        d0 = torch.where(valid_q, cohens_d, torch.full_like(cohens_d, float("nan")))[0]
        d_heatmap = d0.amax(dim=0).cpu().numpy()
        # per-angle scalar: max over spatial
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


def is_tin_artifact(elevation: torch.Tensor, valid_mask: torch.Tensor,
                    kernel_size: int = 32) -> float:
    """
    Scale-invariant TIN artifact detection using Localized Maximum Density.

    Args:
        elevation: (H, W) or (1, 1, H, W) tensor of elevation data.
        valid_mask: Boolean tensor of same shape indicating valid pixels.
        kernel_size: The fixed physical size of the sliding window (e.g., 32x32 pixels).
                     This should be sized to match the minimum physical area a TIN
                     artifact is expected to cover.
    """
    # 1. Format tensors
    if elevation.dim() == 2:
        elevation = elevation.view(1, 1, elevation.shape[0], elevation.shape[1])
        valid_mask = valid_mask.view(1, 1, valid_mask.shape[0], valid_mask.shape[1])

    safe_elev = elevation.clone()
    safe_elev[~valid_mask] = 0.0

    # 2. Compute Laplacian (2nd derivative)
    laplacian_kernel = torch.tensor([[[[0.0, 1.0, 0.0],
                                       [1.0, -4.0, 1.0],
                                       [0.0, 1.0, 0.0]]]], device=elevation.device)
    laplacian = F.conv2d(safe_elev, laplacian_kernel, padding=1)

    # 3. Erode valid mask to remove 0-padding corruption at boundaries
    invalid_mask = (~valid_mask).float()
    dilated_invalid = F.max_pool2d(invalid_mask, kernel_size=3, stride=1, padding=1)
    eroded_valid = (dilated_invalid == 0.0).float()

    # 4. Generate the Boolean Indicator Tensor I(x,y)
    # True (1.0) if curvature is approx 0 AND the pixel is deeply valid
    zero_curvature_mask = ((laplacian.abs() < 1e-2) * eroded_valid.bool()).float()

    # 5. Localized Maximum Density (The Scale-Invariant Step)
    # We use average pooling with stride=1 to slide the window across every pixel.
    # We must separately pool the valid mask to normalize edges properly.

    local_planar_sum = F.avg_pool2d(zero_curvature_mask, kernel_size=kernel_size, stride=1)
    local_valid_sum = F.avg_pool2d(eroded_valid, kernel_size=kernel_size, stride=1)

    # Avoid division by zero in areas with no valid data
    safe_valid_sum = torch.clamp(local_valid_sum, min=1e-6)

    # Calculate density: planar pixels / valid pixels in the local window
    local_density = local_planar_sum / safe_valid_sum

    # Ignore windows that don't have enough valid data to make a statistically sound judgment
    # (e.g., require the window to be at least 50% valid data)
    valid_window_mask = local_valid_sum >= 0.5

    if not valid_window_mask.any():
        return False

    # The scale-invariant statistic S
    max_local_density = local_density[valid_window_mask].max().item()

    return max_local_density


# FIXME the problem is that this had z sun vector be negative
@deprecated(
    "Use estimate_sun_vector_irls instead, this does not calculate the z sun vector correctly and can render it to have negative values.")
def estimate_sun_vector_ols(dtm: torch.Tensor, ortho: torch.Tensor, valid_mask: torch.Tensor) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Estimates the sun vector [sx, sy, sz] using Ordinary Least Squares.
    Handles shapes (C, H, W) or (1, C, H, W).
    """
    device = dtm.device

    # Helper for fallback returns
    def get_defaults():
        default_sun = F.normalize(torch.tensor([0.5, -0.5, 1.0], device=device), p=2, dim=0)
        return default_sun, torch.tensor(1.0, device=device), torch.tensor(0.3, device=device)

    # Ensure 3D (C, H, W)
    if dtm.ndim == 4:
        assert dtm.shape[
                   0] == 1, "The batch size should be 1. estimate_sun_vector_irls currently only works for a batch size of 1"
        dtm = dtm[0]
    if ortho.ndim == 4: ortho = ortho[0]
    if valid_mask.ndim == 4: valid_mask = valid_mask[0]

    # Convert ortho to grayscale if it's RGB
    if ortho.shape[0] == 3:
        ortho = ortho.mean(dim=0, keepdim=True)

    # Sobel kernels
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device) / 8.0
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device) / 8.0

    spatial_scale = max(dtm.shape[-2], dtm.shape[-1]) / 2.0
    padded_dtm = F.pad(dtm.unsqueeze(0), (1, 1, 1, 1), mode='replicate')

    n_x = -F.conv2d(padded_dtm, sobel_x.view(1, 1, 3, 3)) * spatial_scale
    n_y = -F.conv2d(padded_dtm, sobel_y.view(1, 1, 3, 3)) * spatial_scale
    n_z = torch.ones_like(n_x)

    normals = torch.cat([n_x, n_y, n_z], dim=1)
    normals = F.normalize(normals, p=2, dim=1).squeeze(0)  # Shape: (3, H, W)

    mask = valid_mask.squeeze(0).bool()

    # Filter extreme shadows
    ortho_valid = ortho.squeeze(0)[mask]
    if len(ortho_valid) == 0:
        return get_defaults()

    intensity_threshold = torch.quantile(ortho_valid, 0.05)
    shadow_mask = ortho.squeeze(0) > intensity_threshold
    final_mask = mask & shadow_mask

    N_flat = normals[:, final_mask].t()  # Shape: (M, 3)
    Y_flat = ortho.squeeze(0)[final_mask].unsqueeze(1)  # Shape: (M, 1)

    if N_flat.shape[0] < 100:
        return get_defaults()

    # Add column of 1s for bias/ambient light
    ones = torch.ones((N_flat.shape[0], 1), device=device)
    A = torch.cat([N_flat, ones], dim=1)  # Shape: (M, 4)

    # Solve Least Squares
    x = torch.linalg.lstsq(A, Y_flat).solution

    # Extract and normalize the sun vector
    k = x[:3, 0]
    ambient = x[3, 0]  # <--- The scene's ambient light bounce
    intensity = torch.norm(k, p=2)  # <--- The scene's overall brightness

    sun_vec = F.normalize(k, p=2, dim=0)

    return sun_vec, intensity, ambient


@torch.no_grad()
def estimate_sun_vector_irls(
        dtm: torch.Tensor,
        ortho: torch.Tensor,
        valid_mask: torch.Tensor,
        max_iter: int = 15,
        tol: float = 1e-4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decoupled IRLS: Separates ambient light estimation from the linear system
    to prevent the Nz / Bias collinearity trap from inverting the sun vector.
    """
    device = dtm.device

    def get_defaults():
        default_sun = F.normalize(torch.tensor([0.5, -0.5, 1.0], device=device), p=2, dim=0)
        return default_sun, torch.tensor(1.0, device=device), torch.tensor(0.05, device=device)

    if dtm.ndim == 4: dtm = dtm[0]
    if ortho.ndim == 4: ortho = ortho[0]
    if valid_mask.ndim == 4: valid_mask = valid_mask[0]
    if ortho.shape[0] == 3: ortho = ortho.mean(dim=0, keepdim=True)

    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device) / 8.0
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device) / 8.0
    spatial_scale = max(dtm.shape[-2], dtm.shape[-1]) / 2.0
    padded_dtm = F.pad(dtm.unsqueeze(0), (1, 1, 1, 1), mode='replicate')

    n_x = -F.conv2d(padded_dtm, sobel_x.view(1, 1, 3, 3)) * spatial_scale
    n_y = -F.conv2d(padded_dtm, sobel_y.view(1, 1, 3, 3)) * spatial_scale
    n_z = torch.ones_like(n_x)

    normals = F.normalize(torch.cat([n_x, n_y, n_z], dim=1), p=2, dim=1).squeeze(0)

    mask = valid_mask.squeeze(0).bool()
    zero_mask = ortho.squeeze(0) > 1e-4
    final_mask = mask & zero_mask

    N_flat = normals[:, final_mask].t()
    Y_raw = ortho.squeeze(0)[final_mask].unsqueeze(1)

    if N_flat.shape[0] < 100:
        return get_defaults()

    # --- 1. Decoupled Ambient Estimation ---
    # The 1st percentile of valid pixels is a highly robust proxy for the
    # secondary scattering/ambient floor in deep shadows.
    ambient_est = torch.quantile(Y_raw, 0.01)

    # Subtract ambient to isolate the pure directional irradiance
    Y_flat = torch.clamp(Y_raw - ambient_est, min=0.0)

    # --- 2. Design Matrix (No Bias Column) ---
    H = N_flat  # Shape: (M, 3)

    beta = torch.linalg.lstsq(H, Y_flat).solution
    c = 1.345

    for _ in range(max_iter):
        residuals = Y_flat - torch.mm(H, beta)
        median_res = torch.median(residuals)
        mad = torch.median(torch.abs(residuals - median_res))
        sigma = (mad / 0.67449) + 1e-6

        r_stand = torch.abs(residuals / sigma)
        weights = torch.clamp(c / (r_stand + 1e-8), max=1.0)

        w_sqrt = torch.sqrt(weights)
        H_w = H * w_sqrt
        Y_w = Y_flat * w_sqrt

        beta_new = torch.linalg.lstsq(H_w, Y_w).solution

        if torch.norm(beta_new - beta, p=2) < tol:
            beta = beta_new
            break

    # --- 3. Parameter Extraction and Enforced Positivity ---
    k = beta[:3, 0]

    # Absolute failsafe: Since we stripped the bias, if anomalous geometry
    # somehow still forces a negative Z, we mathematically reflect it
    # across the horizon line to maintain physical validity.
    if k[2] < 0:
        k[2] = -k[2]

    intensity = torch.norm(k, p=2).clamp(min=1e-4)
    sun_vec = F.normalize(k, p=2, dim=0)

    return sun_vec, intensity, ambient_est


def _normalize_dtm_relative(
        elevation: torch.Tensor,
        valid: torch.Tensor,
        scale_factor: float,
) -> torch.Tensor:
    """Normalise elevation via local centering + fixed global scale.

    This produces "relative topography": the absolute Martian altitude (datum)
    is subtracted per-patch (unlearnable from orthorectified overhead imagery),
    while a fixed physical scale maps consistent slope magnitudes to the same
    latent values everywhere in the dataset.

    Args:
        elevation: Raw elevation in metres; NaN = nodata.  Shape (1, H, W).
        scale_factor: Half the expected relief range in metres.  Values in
            ``[-scale_factor, +scale_factor]`` map to ``[-1, 1]``.  Use the
            dataset ``centered_p98`` for the elevation channel so that 98 % of
            real patches land within range before clamping.

    Returns:
        (1, H, W) tensor in [-1, 1]; nodata pixels filled with 0.
    """
    bool_valid = valid == 1
    if not bool_valid.any():
        return torch.zeros_like(elevation)

    # 1. Remove absolute altitude — the network cannot infer this from texture
    patch_mean = elevation[bool_valid].mean()
    centered = elevation - patch_mean

    # 2. Dynamic Local Scale
    # Find the maximum absolute deviation from the mean in THIS specific patch
    local_max = centered[bool_valid].abs().max()

    # Avoid divide-by-zero if the patch is somehow perfectly flat
    if local_max < 1e-4:
        local_scale = scale_factor
    else:
        local_scale = local_max

    # 2. Fixed physical scale: a 10 m ridge always produces the same latent delta
    normed = centered / local_scale

    # clipped = normed[bool_valid].abs() > 1.0
    #
    # if clipped.sum() > 10:
    #     vals = (normed[bool_valid].abs() < 1).sum().item()
    #     b = clipped.sum().item()
    #     raise RuntimeError(f"Values out of range, they had to be slipped {b} vs {vals} ratio {b / vals}")

    # 3. Clamp extreme outliers (craters, scarps) without distorting the core
    normed = torch.clamp(normed, -1.0, 1.0)
    normed = torch.where(bool_valid, normed, torch.zeros_like(normed))
    return normed


def _to_3ch(tensor: torch.Tensor) -> torch.Tensor:
    """Replicate a (1, H, W) tensor to (3, H, W) for VAE compatibility."""
    if tensor.shape[0] == 1:
        return tensor.expand(3, -1, -1).contiguous()
    return tensor[:3]


def _resize(tensor: torch.Tensor, size: int, mode: str = "bilinear") -> torch.Tensor:
    """Resize (C, H, W) to (C, size, size)."""
    if tensor.shape[-2] == size and tensor.shape[-1] == size:
        return tensor

    kwargs = {"mode": mode}
    if mode != "nearest-exact":
        kwargs["align_corners"] = False

    return F.interpolate(
        tensor.unsqueeze(0), size=(size, size), **kwargs
    ).squeeze(0)


def compute_topographic_residual(elevation: torch.Tensor, valid_mask: torch.Tensor) -> float:
    """
    Fits a 2D plane to the elevation data and returns the RMS residual.
    This removes macroscopic slopes and isolates true topographic roughness.
    """
    if elevation.ndim > 2:
        elevation = elevation.squeeze()
        valid_mask = valid_mask.squeeze()

    H, W = elevation.shape

    # Create normalized grid coordinates [-1, 1] for numerical stability
    y = torch.linspace(-1, 1, H, dtype=elevation.dtype, device=elevation.device)
    x = torch.linspace(-1, 1, W, dtype=elevation.dtype, device=elevation.device)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    valid_bool = valid_mask.bool()
    if not valid_bool.any():
        return 0.0

    X_v = X[valid_bool].unsqueeze(1)  # (N, 1)
    Y_v = Y[valid_bool].unsqueeze(1)  # (N, 1)
    Z_v = elevation[valid_bool].unsqueeze(1)  # (N, 1)

    # Design matrix A: [X, Y, 1]
    A = torch.cat([X_v, Y_v, torch.ones_like(X_v)], dim=1)  # (N, 3)

    # Solve Least Squares: A * w = Z
    w = torch.linalg.lstsq(A, Z_v).solution

    # Calculate RMS of the residual (distance from actual elevation to the fitted plane)
    Z_pred = A @ w
    residual_rms = torch.sqrt(torch.mean((Z_v - Z_pred) ** 2)).item()

    return residual_rms


class DepthFMHiRISEAdapterCached(Dataset):
    """Adapter that converts MarsHiRISEDTM samples into DepthFM training pairs.

    Each ``__getitem__`` call draws a geo-slice from the sampler, loads the
    corresponding elevation + orthoimage(s) via the base dataset, and returns
    a normalised ``{image, dtm}`` dict ready for the DepthFM training loop.

    Args:
        base_dataset: An initialised ``MarsHiRISEDTM`` instance.
        sampler: A TorchGeo geo-sampler that yields GeoSlice indices.
        resolution: Output spatial resolution in pixels (square crop).
        dtm_normalization: ``"relative"`` (default) — local centering + fixed
            physical scale derived from ``centered_p98`` in the stats file.
            ``"log"`` / ``"linear"`` — global quantile modes (kept for reference).
        random_flip: Apply random horizontal/vertical flips.
        brightness_jitter: Max relative brightness perturbation on image only.
        stats_path: Path to ``dataset_stats.json`` for normalization quantiles.
            Falls back to hardcoded Olympus-region defaults if ``None``.
    """

    def __init__(
            self,
            base_dataset: MarsHiRISEDTM,
            sampler: HiRISEGeoSampler,
            resolution: int = 512,
            dtm_normalization: Literal["relative", "log", "linear"] = "relative",
            random_flip: bool = True,
            random_jitter: bool = False,
            brightness_jitter: float = 0.1,
            stats_path: str | None = None,
            clip: bool = False,
            use_manifest: bool = True,
            manifest_workers: int = 16,  # Set this high to build the cache fast
            manifest_dir: str = ".cache/manifests",
            erode_radius: int = 2,
            multiprocessing_context='fork'
    ):
        super().__init__()
        self.multiprocessing_context = multiprocessing_context
        self.erode_radius = erode_radius
        self.base = base_dataset
        self.sampler = sampler
        self.resolution = resolution
        self.dtm_norm = dtm_normalization
        self.flip = random_flip
        self.random_jitter = random_jitter
        self.bright_jitter = brightness_jitter
        self.is_train = random_flip

        self.use_manifest = use_manifest
        self.manifest_workers = manifest_workers
        self.manifest_dir = self.base.root / Path(manifest_dir)
        self.clip = clip

        # Load global quantiles
        (
            self.elev_p02, self.elev_p98,
            self.img_p02, self.img_p98,
            self.elev_scale,
        ) = _load_quantiles(stats_path)

        # TODO instead of local patch ortho normalizer, it should be a per-strip normalization. Sadly infrastructure does not handle this well yet
        self.ortho_normalizer = LocalStripOrthoNormalizer(self.img_p02, self.img_p98)
        self.evel_normalizer = GlobalLogNormalizer(self.elev_scale, clip=clip)

        # Pre-materialise sampler indices
        self._raw_indices = list(sampler)

        # --- Cache Handling ---
        if self.use_manifest:
            self._init_manifest()
        else:
            self.clean_records = None
            logger.warning("Manifest disabled. Training will be slow due to on-the-fly validation.")

    def _get_manifest_hash(self) -> str:
        """Create a unique key so the cache rebuilds if dataset/sampler params change."""
        key_parts = {
            "root": str(getattr(self.base, "root", "unknown")),
            "resolution": self.resolution,
            "sampler_length": len(self._raw_indices),
            "split": getattr(self.sampler, "split", "unknown"),
            "seed": getattr(self.sampler, "seed", 0),
            "dataset_hash": str(self.base.spatial_index_cache),
            "sampler_hash": str(self.sampler.cache_hash)
        }

        if not self.clip:
            key_parts["clip"] = self.clip

        raw = json.dumps(key_parts, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _init_manifest(self):
        """Loads the parquet manifest or triggers a fast multiprocessing build."""
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.manifest_dir / f"hirise_manifest_{self._get_manifest_hash()}.parquet"

        if manifest_path.exists():
            logger.info(f"Loading rich manifest from {manifest_path}")
            df = pd.read_parquet(manifest_path)
        else:
            logger.info(f"Manifest not found. Building cache using {self.manifest_workers} workers...")
            df = self._build_manifest_parallel(manifest_path)

        # Filter the dataframe to only keep good patches
        # You can easily adjust these thresholds in the future without rebuilding the cache!
        clean_df = df[
            (df['is_valid_data'] == True) &
            (df['valid_ratio'] >= 0.5) &
            (df['residual'] >= 0.1) &
            (df[
                 'is_tin'] <= 0.95) &  # High TIN values means 2nd derivative (Laplacian) is nearly zero, which indicates perfectly flat planes.
            (df["num_merges"] == 1)  # Currently tile merges have issues for DTM estimation
            ]

        # Convert to a list of dicts for O(1) lookup during training
        self.clean_records = clean_df.to_dict('records')
        logger.info(
            f"Manifest ready: Filtered {len(df)} total patches down to {len(self.clean_records)} clean pairs. Saving manifest to {manifest_path}")

    def _build_manifest_parallel(self, save_path: Path) -> pd.DataFrame:
        """Uses a temporary PyTorch DataLoader to build the cache at maximum speed."""

        # 1. Define a lightweight inner dataset just for computing stats
        class _ManifestBuilderDS(Dataset):
            def __init__(self, adapter):
                self.adapter = adapter

            def __len__(self):
                return len(self.adapter._raw_indices)

            def __getitem__(self, idx):
                return self.adapter._evaluate_patch_for_manifest(idx)

        # 2. Use standard PyTorch DataLoader to bypass the GIL
        builder_loader = DataLoader(
            _ManifestBuilderDS(self),
            batch_size=1,  # Process one by one
            num_workers=self.manifest_workers,
            collate_fn=lambda x: x[0],  # Prevent PyTorch from batching dicts into tensors
            shuffle=False,
            multiprocessing_context=self.multiprocessing_context
        )

        records = []
        for record in tqdm(builder_loader, desc="Scanning HiRISE Data", unit="patch"):
            records.append(record)

        # 3. Save and return
        df = pd.DataFrame(records)
        df.to_parquet(save_path)
        return df

    def _evaluate_patch_for_manifest(self, idx: int) -> dict:
        """The heavy lifting: loads data, calculates stats, and returns a dictionary.
        This ONLY runs once during cache generation."""
        try:
            geo_slice = self._raw_indices[idx]
            sample = self.base[geo_slice]

            elevation = sample["elevation"]
            if elevation.ndim == 4: elevation = elevation[0]
            if elevation.shape[-1] == 0 or elevation.shape[-2] == 0:
                return {"idx": idx, "is_valid_data": False}

            # Select Orthoimage
            left_key, right_key = "left_red", "right_red"
            has_left = left_key in sample and sample[left_key] is not None
            has_right = right_key in sample and sample[right_key] is not None

            if has_left and has_right:
                key = left_key if torch.rand(1).item() > 0.5 else right_key
            elif has_left:
                key = left_key
            elif has_right:
                key = right_key
            else:
                for fk in ("left_irb", "right_irb"):
                    if fk in sample and sample[fk] is not None:
                        key = fk
                        break
                else:
                    return {"idx": idx, "is_valid_data": False}

            ortho = sample[key]
            if ortho.ndim == 4: ortho = ortho[0]

            # -------------------------------------------------------------
            # 1. Masking and Resizing
            # -------------------------------------------------------------
            dtm_resized = _safe_resize(elevation, self.resolution, has_nans=True)
            image_resized = _safe_resize(ortho, self.resolution)

            ortho_valid = (image_resized != 0.0).any(dim=0, keepdim=True)
            elev_valid = torch.isfinite(dtm_resized) & (dtm_resized != 0.0)
            valid_mask_resized = (ortho_valid & elev_valid).float()

            valid_ratio = valid_mask_resized.mean().item()

            # -------------------------------------------------------------
            # 2. Heavy Math Operations
            # -------------------------------------------------------------
            residual = compute_topographic_residual(dtm_resized, valid_mask_resized)

            elev_valid_native = torch.isfinite(elevation) & (elevation != 0.0)
            is_tin = is_tin_artifact(elevation, elev_valid_native)

            # -------------------------------------------------------------
            # 3. Sun Vector Math (Fixed)
            # -------------------------------------------------------------
            # We MUST normalize the data before computing the OLS sun vector because
            # the loss function renders shadows in the [-1, 1] normalized latent space.
            # Physical metadata is incompatible because local scaling distorts Z geometry.
            dtm_norm: TrainingNormResult = self.evel_normalizer.normalize_for_training(dtm_resized, valid_mask_resized)
            image_norm = self.ortho_normalizer.normalize(image_resized)

            sun_vec, intensity, ambient = estimate_sun_vector_irls(dtm_norm.normed_residual, image_norm,
                                                                   valid_mask_resized)

            sun_x = sun_vec[0].item()
            sun_y = sun_vec[1].item()
            sun_z = sun_vec[2].item()
            intensity_val = intensity.item()
            ambient_val = ambient.item()

            return {
                "idx": idx,
                "is_valid_data": True,
                "ortho_key": key,
                "valid_ratio": valid_ratio,
                "residual": residual,
                "is_tin": is_tin,
                "sun_x": sun_x,
                "sun_y": sun_y,
                "sun_z": sun_z,
                "intensity": intensity_val,
                "ambient": ambient_val,
                "num_merges": len(sample['meta'])
            }

        except Exception as e:
            logger.exception("Failed to generate manifest for patch")
            return {"idx": idx, "is_valid_data": False}

    def __len__(self) -> int:
        if self.use_manifest:
            return len(self.clean_records)
        return len(self._raw_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | float]:
        """Lightning fast __getitem__. No retries, no heavy math."""
        if not self.use_manifest:
            raise NotImplementedError(
                "Fallback on-the-fly __getitem__ removed for brevity. Please run with use_manifest=True.")

        # 1. Get pre-validated metadata in O(1) time
        record = self.clean_records[idx]
        geo_slice = self._raw_indices[record["idx"]]
        key = record["ortho_key"]

        # 2. Load the data (Guaranteed to be valid)
        sample = self.base[geo_slice]

        elevation = sample["elevation"]
        if elevation.ndim == 4: elevation = elevation[0]

        ortho = sample[key]
        if ortho.ndim == 4: ortho = ortho[0]

        sun_vector = torch.tensor([record["sun_x"], record["sun_y"], record["sun_z"]], dtype=torch.float32)
        intensity = torch.tensor(record["intensity"], dtype=torch.float32)
        ambient = torch.tensor(record["ambient"], dtype=torch.float32)

        # 3. Resize
        dtm_resized = _safe_resize(elevation, self.resolution, has_nans=True)
        image_resized = _safe_resize(ortho, self.resolution)

        # Masks
        ortho_valid = (image_resized != 0.0).any(dim=0, keepdim=True)
        elev_valid = torch.isfinite(dtm_resized) & (dtm_resized != 0.0)
        valid_mask_resized = (ortho_valid & elev_valid).float()

        # 4. Normalization
        dtm_res: TrainingNormResult = self.evel_normalizer.normalize_for_training(dtm_resized, valid_mask_resized)
        dtm = dtm_res.normed_residual
        image = self.ortho_normalizer.normalize(image_resized)

        # --- PRE-VAE NEAREST NEIGHBOR FILL ---
        # Fills regions outside the valid mask so the VAE doesn't encode sharp black boundaries
        # 2. Mask Erosion (Using the corrected function)
        image, dtm, valid_mask_resized = fill_voids_gmrf(image, dtm, valid_mask_resized)

        dtm = _to_3ch(dtm)
        image = _to_3ch(image)

        # 5. Synchronised Augmentation
        if getattr(self, 'is_train', False) and getattr(self, 'flip', False):
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, [-1])
                dtm = torch.flip(dtm, [-1])
                valid_mask_resized = torch.flip(valid_mask_resized, [-1])
                sun_vector[0] = -sun_vector[0]

            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, [-2])
                dtm = torch.flip(dtm, [-2])
                valid_mask_resized = torch.flip(valid_mask_resized, [-2])
                sun_vector[1] = -sun_vector[1]

            k_rot = torch.randint(0, 4, (1,)).item()
            if k_rot > 0:
                image = torch.rot90(image, k=k_rot, dims=[-2, -1])
                dtm = torch.rot90(dtm, k=k_rot, dims=[-2, -1])
                valid_mask_resized = torch.rot90(valid_mask_resized, k=k_rot, dims=[-2, -1])

                sx, sy = sun_vector[0].clone(), sun_vector[1].clone()
                if k_rot == 1:
                    sun_vector[0], sun_vector[1] = sy, -sx
                elif k_rot == 2:
                    sun_vector[0], sun_vector[1] = -sx, -sy
                elif k_rot == 3:
                    sun_vector[0], sun_vector[1] = -sy, sx

        # Brightness jitter
        if getattr(self, 'is_train', False) and getattr(self, 'bright_jitter', 0) > 0:
            factor = 1.0 + random.uniform(-self.bright_jitter, self.bright_jitter)
            image = (image * factor)

            if self.clip:
                image = torch.clip(image, -1.0, 1.0)

            # Scale the physical lighting parameters so the loss physics still match the augmented image
            intensity *= factor
            ambient *= factor

        res = {
            "image": image,
            "dtm": dtm,
            "confidence": valid_mask_resized,
            "sun_vector": sun_vector,
            "intensity": intensity,
            "ambient": ambient,
            "original_dtm": dtm_resized,
            "original_image": image_resized,
            "trend_params": dtm_res.trend_params,
            "residual_scale": dtm_res.residual_scale,
            "raw_residual_p98": dtm_res.raw_residual_p98,
            "key": key,
        }

        if 'meta' in sample:
            res['meta'] = sample['meta']

        return res
