"""Void/nodata filling for HiRISE DTM and orthoimage patches.

Four strategies, in order of cost:

* `fill_invalid_nearest_neighbor` — EDT-based nearest valid pixel propagation.
* `fill_dtm_smart_diffusion`      — iterative Laplacian heat-equation diffusion.
* `fill_voids_kriging`            — Universal/Ordinary kriging (subsampled).
* `fill_voids_gmrf`               — single GMRF conditional solve per channel.

The GMRF path is the default used by the training adapter.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn.functional as F
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from scipy import ndimage

from depth_fm.data.image_processing.mask_ops import erode_valid_mask

logger = logging.getLogger(__name__)


def fill_invalid_nearest_neighbor(
        tensor: torch.Tensor,
        valid_mask: torch.Tensor
) -> torch.Tensor:
    """Fill invalid regions using nearest-neighbor propagation via EDT.

    Supports (H, W), (C, H, W), and (B, C, H, W) tensors.
    """
    device = tensor.device
    dtype = tensor.dtype

    arr_np = tensor.cpu().numpy()
    mask_np = valid_mask.squeeze().cpu().numpy() > 0

    if mask_np.all() or not mask_np.any():
        return tensor

    indices = ndimage.distance_transform_edt(
        ~mask_np,
        return_distances=False,
        return_indices=True,
    )
    iy, ix = indices

    if arr_np.ndim == 2:
        filled_np = arr_np[iy, ix]

    elif arr_np.ndim == 3:
        C = arr_np.shape[0]
        c_idx = np.arange(C)[:, None, None]
        iy_b = iy[None, :, :]
        ix_b = ix[None, :, :]
        filled_np = arr_np[c_idx, iy_b, ix_b]

    elif arr_np.ndim == 4:
        B, C = arr_np.shape[:2]
        b_idx = np.arange(B)[:, None, None, None]
        c_idx = np.arange(C)[None, :, None, None]
        iy_b = iy[None, None, :, :]
        ix_b = ix[None, None, :, :]
        filled_np = arr_np[b_idx, c_idx, iy_b, ix_b]

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
    """Fill voids in an orthoimage and DTM with a single kriging pass per channel.

    Optimised for 512x512 tiles: subsamples valid pixels for training, then
    predicts every void pixel at once. No iteration, no per-void labelling,
    no diffusion loop.
    """
    rng = np.random.default_rng(seed)

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

    eroded = erode_valid_mask(valid_mask, erode_radius)
    valid = eroded[0, 0].cpu().numpy() > 0.5
    void = ~valid

    img_np = np.nan_to_num(image[0].cpu().numpy(), nan=0.0).astype(np.float64)
    dtm_np = np.nan_to_num(dtm[0].cpu().numpy(), nan=0.0).astype(np.float64)

    if not void.any():
        if img_3d:
            image = image.squeeze(0)
        if dtm_3d:
            dtm = dtm.squeeze(0)
        return image, dtm

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

    filled_img_t = torch.from_numpy(filled_img).unsqueeze(0).to(device=device, dtype=img_dtype)
    filled_dtm_t = torch.from_numpy(filled_dtm).unsqueeze(0).to(device=device, dtype=dtm_dtype)

    if img_3d:
        filled_img_t = filled_img_t.squeeze(0)
    if dtm_3d:
        filled_dtm_t = filled_dtm_t.squeeze(0)

    return filled_img_t, filled_dtm_t


def fill_dtm_smart_diffusion(
        tensor: torch.Tensor,
        valid_mask: torch.Tensor,
        iterations: int = 64,
        erode_radius: int = 2,
) -> torch.Tensor:
    """Fill invalid regions using Laplacian diffusion (heat equation)."""
    is_3d = tensor.ndim == 3
    if is_3d:
        tensor = tensor.unsqueeze(0)
        valid_mask = valid_mask.unsqueeze(0)

    B, C, H, W = tensor.shape
    device = tensor.device
    dtype = tensor.dtype

    tensor = torch.nan_to_num(tensor, nan=0.0)

    valid_bool = erode_valid_mask(valid_mask, erode_radius) > 0.5

    valid_sum = (tensor * valid_bool.to(dtype)).sum(dim=[-2, -1], keepdim=True)
    valid_count = valid_bool.sum(dim=[-2, -1], keepdim=True).clamp(min=1.0)
    global_mean = valid_sum / valid_count

    filled = torch.where(valid_bool, tensor, global_mean)

    kernel = torch.ones((C, 1, 3, 3), device=device, dtype=dtype) / 9.0

    for _ in range(iterations):
        # Replicate padding stops physical image edges from dragging boundary
        # values down to 0.0 during the blur phase.
        padded_filled = F.pad(filled, pad=(1, 1, 1, 1), mode="replicate")
        blurred = F.conv2d(padded_filled, kernel, padding=0, groups=C)
        filled = torch.where(valid_mask, tensor, blurred)

    if is_3d:
        return filled.squeeze(0)
    return filled


@lru_cache
def _build_grid_laplacian(H: int, W: int, connectivity: int = 4) -> sp.csc_matrix:
    """Build graph Laplacian L = D - W for an H×W grid. Fully vectorised."""
    N = H * W
    idx = np.arange(N).reshape(H, W)

    pairs = []
    left = idx[:, :-1].ravel()
    right = idx[:, 1:].ravel()
    pairs.append((left, right))

    top = idx[:-1, :].ravel()
    bot = idx[1:, :].ravel()
    pairs.append((top, bot))

    if connectivity == 8:
        tl = idx[:-1, :-1].ravel()
        br = idx[1:, 1:].ravel()
        pairs.append((tl, br))
        tr = idx[:-1, 1:].ravel()
        bl = idx[1:, :-1].ravel()
        pairs.append((tr, bl))

    src = np.concatenate([p[0] for p in pairs] + [p[1] for p in pairs])
    dst = np.concatenate([p[1] for p in pairs] + [p[0] for p in pairs])

    n_edges = len(src)
    off_diag = sp.coo_matrix(
        (-np.ones(n_edges), (src, dst)), shape=(N, N)
    )

    degree = np.zeros(N)
    np.add.at(degree, src, 1.0)
    diag = sp.diags(degree, format="coo")

    return (diag + off_diag).tocsc()


def _gmrf_fill_channel(
        channel: np.ndarray,
        void_idx: np.ndarray,
        obs_idx: np.ndarray,
        Q: sp.csc_matrix,
        nugget: float,
) -> np.ndarray:
    """Fill void pixels via u_void = -Q_vv⁻¹ Q_vo u_obs."""
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


# TODO: GMRF assumes Gaussian residuals — DTM elevation distributions are
# closer to log-normal. A targeted heightmap prior would be more appropriate.
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
    """Fill voids in an orthoimage and DTM using GMRF conditional distribution.

    One sparse linear solve per channel. No iteration. O(n^{3/2}) on 2D grids.

    Returns (filled_image, filled_dtm, eroded_mask). The eroded mask records
    which pixels were considered trustworthy (1) vs infilled (0) — all pixels
    are valid in the filled outputs.
    """
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

    L = _build_grid_laplacian(H, W, connectivity)
    Q = tau * L

    obs_idx = np.where(valid.ravel())[0]
    void_idx = np.where(~valid.ravel())[0]

    logger.debug(
        "GMRF fill: %d void pixels (%.1f%%), %d-connected, H=%d W=%d",
        len(void_idx), 100.0 * len(void_idx) / (H * W), connectivity, H, W,
    )

    filled_dtm = dtm_np.copy()
    filled_img = img_np.copy()

    for c in range(dtm_np.shape[0]):
        filled_dtm[c] = _gmrf_fill_channel(dtm_np[c], void_idx, obs_idx, Q, nugget)

    for c in range(img_np.shape[0]):
        filled_img[c] = _gmrf_fill_channel(img_np[c], void_idx, obs_idx, Q, nugget)

    filled_img_t = torch.from_numpy(filled_img).unsqueeze(0).to(device=device, dtype=img_dtype)
    filled_dtm_t = torch.from_numpy(filled_dtm).unsqueeze(0).to(device=device, dtype=dtm_dtype)

    if img_3d:
        filled_img_t = filled_img_t.squeeze(0)
    if dtm_3d:
        filled_dtm_t = filled_dtm_t.squeeze(0)
    if msk_3d:
        eroded = eroded.squeeze(0)

    return filled_img_t, filled_dtm_t, eroded
