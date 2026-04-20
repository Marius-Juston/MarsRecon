"""
Mars DepthFM — Graphviz Architecture Diagrams (with real image nodes).

Generates three PDF + SVG diagrams in the style of marsrecon_architecture.py,
but with **image nodes** that show real outputs from
``depth_fm.depthfm_adapter`` inline in the graph.

Pipeline:

    1.  Run the real adapter algorithms on synthetic Mars-like terrain to
        produce numpy panels (GMRF fill, is_tin, seam, sun OLS, flow evolution,
        normals, FFT, etc).
    2.  Render each panel to a standalone PNG on disk with matplotlib.
    3.  Build a graphviz ``Digraph`` that references those PNGs inside HTML
        table labels  —  the node renders as "image + caption + metric box".
    4.  Invoke ``dot`` to render PDF + SVG.

Three figures:

    Figure 1 — ``pipeline_overview``:
        Full system pipeline (Phase 0 infrastructure → Phase 1 manifest filter
        → Phase 2 preprocessing → Phase 3 flow training → Phase 4 losses →
        Phase 5 optimiser). Image nodes inline at each data stage.

    Figure 2 — ``flow_and_losses``:
        Zoom on the flow-matching step and the loss branches, with image nodes
        for flow evolution, velocity maps, normals, gradient pyramid, FFT,
        and Lunar-Lambert render.

    Figure 3 — ``manifest_filter``:
        The four reject criteria side-by-side with real image patches and
        algorithm-computed metric values overlaid.

Usage::

    PYTHONPATH=src uv run python scripts/generate_architecture_diagrams.py

Outputs::

    outputs/figures/pipeline_overview.{pdf,svg}
    outputs/figures/flow_and_losses.{pdf,svg}
    outputs/figures/manifest_filter.{pdf,svg}
    outputs/figures/panels/*.png          (individual panels)
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from pathlib import Path

import graphviz
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy import ndimage

# Register seaborn's 'mako' / 'flare' / 'rocket' palettes as matplotlib colormaps
for _name in ("mako", "flare", "rocket", "crest", "vlag"):
    try:
        _cmap = sns.color_palette(_name, as_cmap=True)
        try:
            mpl.colormaps.register(cmap=_cmap, name=_name)
        except (ValueError, AttributeError):
            try:
                mpl.cm.register_cmap(name=_name, cmap=_cmap)
            except (ValueError, AttributeError):
                pass
    except Exception:
        pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adapter imports — use real algorithms when available, otherwise fall
# through to the bundled bit-identical implementations.
# ---------------------------------------------------------------------------
try:
    from depth_fm.depthfm_adapter import (
        erode_valid_mask as _real_erode,
        fill_voids_gmrf as _real_gmrf,
        is_tin_artifact as _real_tin,
        detect_dtm_seam_artifact as _real_seam,
        estimate_sun_vector_ols as _real_sun,
        compute_topographic_residual as _real_residual,
    )

    _USING_REAL_ADAPTER = True
except Exception:  # pragma: no cover
    _USING_REAL_ADAPTER = False


# ===========================================================================
# PALETTE — mako/flare-inspired, matched to depth_fm/visualization.py
# ===========================================================================

C = {
    # Stage/cluster fills (pale, for cluster backgrounds)
    "bg_infra":     "#EEF5FB",  # cool pale
    "bg_filter":    "#FDEEED",  # pale red
    "bg_preproc":   "#E7F7EF",  # pale green
    "bg_flow":      "#EFEAF7",  # pale purple
    "bg_loss":      "#FFF2E3",  # pale orange
    "bg_optim":     "#EAF7EE",  # pale emerald
    "bg_artifact":  "#F4ECF7",  # pale lavender
    # Node fills (saturated accents)
    "node_infra":    "#1a6b7c",
    "node_filter":   "#c0392b",
    "node_accept":   "#27ae60",
    "node_preproc":  "#2d9b78",
    "node_latent":   "#4a90d9",
    "node_unet":     "#6b5ea8",
    "node_loss_vel": "#e74c3c",
    "node_loss_pix": "#e67e22",
    "node_optim":    "#27ae60",
    "node_artifact": "#8e44ad",
    "node_data":     "#0b3d54",
    # Neutrals
    "edge":          "#495057",
    "border":        "#343A40",
    "light":         "#FFFFFF",
    "text":          "#2c3e50",
    "cluster_line":  "#DEE2E6",
}


# ===========================================================================
# ADAPTER ALGORITHMS  (mirrored from depth_fm.depthfm_adapter)
# ===========================================================================

def _fallback_erode(valid_mask: torch.Tensor, erode_radius: int = 1) -> torch.Tensor:
    if erode_radius <= 0:
        return valid_mask
    is_3d = valid_mask.ndim == 3
    if is_3d:
        valid_mask = valid_mask.unsqueeze(0)
    ks = 2 * erode_radius + 1
    padded = F.pad(valid_mask, (erode_radius,) * 4, mode="constant", value=1.0)
    eroded = -F.max_pool2d(-padded, kernel_size=ks, stride=1, padding=0)
    eroded = (eroded > 0.5).float()
    if is_3d:
        eroded = eroded.squeeze(0)
    return eroded


@lru_cache(maxsize=4)
def _fallback_laplacian(H: int, W: int, connectivity: int = 4) -> sp.csc_matrix:
    N = H * W
    idx = np.arange(N).reshape(H, W)
    pairs = [(idx[:, :-1].ravel(), idx[:, 1:].ravel()),
             (idx[:-1, :].ravel(), idx[1:, :].ravel())]
    if connectivity == 8:
        pairs.append((idx[:-1, :-1].ravel(), idx[1:, 1:].ravel()))
        pairs.append((idx[:-1, 1:].ravel(), idx[1:, :-1].ravel()))
    src = np.concatenate([p[0] for p in pairs] + [p[1] for p in pairs])
    dst = np.concatenate([p[1] for p in pairs] + [p[0] for p in pairs])
    off = sp.coo_matrix((-np.ones(len(src)), (src, dst)), shape=(N, N))
    deg = np.zeros(N)
    np.add.at(deg, src, 1.0)
    return (sp.diags(deg, format="coo") + off).tocsc()


def _fallback_gmrf(image, dtm, valid_mask, *,
                   erode_radius=2, connectivity=4, tau=1.0, nugget=1e-6):
    img_3d = image.ndim == 3
    if img_3d: image = image.unsqueeze(0)
    if dtm.ndim == 3: dtm = dtm.unsqueeze(0)
    if valid_mask.ndim == 3: valid_mask = valid_mask.unsqueeze(0)

    eroded = _fallback_erode(valid_mask, erode_radius)
    valid = eroded[0, 0].cpu().numpy() > 0.5
    img_np = np.nan_to_num(image[0].cpu().numpy(), nan=0.0).astype(np.float64)
    dtm_np = np.nan_to_num(dtm[0].cpu().numpy(), nan=0.0).astype(np.float64)

    if not (~valid).any():
        return image.squeeze(0), dtm.squeeze(0), eroded.squeeze(0)

    H, W = valid.shape
    Q = tau * _fallback_laplacian(H, W, connectivity)
    obs_idx = np.where(valid.ravel())[0]
    void_idx = np.where(~valid.ravel())[0]
    Q_vv = Q[np.ix_(void_idx, void_idx)] + nugget * sp.eye(len(void_idx), format="csc")
    Q_vo = Q[np.ix_(void_idx, obs_idx)]

    filled_img = img_np.copy(); filled_dtm = dtm_np.copy()
    for c in range(dtm_np.shape[0]):
        flat = dtm_np[c].ravel()
        u = spla.spsolve(Q_vv, -Q_vo @ flat[obs_idx])
        flat[void_idx] = u
        filled_dtm[c] = flat.reshape(H, W)
    for c in range(img_np.shape[0]):
        flat = img_np[c].ravel()
        u = spla.spsolve(Q_vv, -Q_vo @ flat[obs_idx])
        flat[void_idx] = u
        filled_img[c] = flat.reshape(H, W)

    img_t = torch.from_numpy(filled_img).unsqueeze(0).float()
    dtm_t = torch.from_numpy(filled_dtm).unsqueeze(0).float()
    return img_t.squeeze(0), dtm_t.squeeze(0), eroded.squeeze(0)


def _fallback_tin(elevation, valid_mask, kernel_size=32):
    if elevation.dim() == 2:
        elevation = elevation.view(1, 1, *elevation.shape)
        valid_mask = valid_mask.view(1, 1, *valid_mask.shape)
    safe = elevation.clone()
    safe[~valid_mask] = 0.0
    lap_k = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]],
                         device=elevation.device, dtype=elevation.dtype)
    lap = F.conv2d(safe, lap_k, padding=1)
    inv = (~valid_mask).float()
    dil = F.max_pool2d(inv, kernel_size=3, stride=1, padding=1)
    eroded = (dil == 0.0).float()
    zero_curv = ((lap.abs() < 1e-2) * eroded.bool()).float()
    local_pl = F.avg_pool2d(zero_curv, kernel_size=kernel_size, stride=1)
    local_vd = F.avg_pool2d(eroded, kernel_size=kernel_size, stride=1)
    density = local_pl / torch.clamp(local_vd, min=1e-6)
    valid_win = local_vd >= 0.5
    if not valid_win.any():
        return 0.0
    return density[valid_win].max().item()


def _fallback_seam(elevation, valid_mask,
                   line_length=35, num_angles=8, min_valid_ratio=0.5):
    if elevation.dim() == 2:
        elevation = elevation.view(1, 1, *elevation.shape)
        valid_mask = valid_mask.view(1, 1, *valid_mask.shape)
    dev, dt = elevation.device, elevation.dtype
    safe = elevation.clone()
    safe[~valid_mask] = 0.0
    sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=dev, dtype=dt).view(1, 1, 3, 3) / 8.0
    sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                      device=dev, dtype=dt).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(safe, sx, padding=1)
    gy = F.conv2d(safe, sy, padding=1)
    g = torch.sqrt(gx**2 + gy**2 + 1e-8)
    inv = (~valid_mask).float()
    dil = F.max_pool2d(inv, kernel_size=7, stride=1, padding=3)
    eroded = (dil == 0.0).float()
    g = g * eroded
    bg = (g.sum() / eroded.sum().clamp(min=1)).clamp(min=1e-5)
    kc = line_length // 2
    ker = torch.zeros((num_angles, 1, line_length, line_length),
                      device=dev, dtype=dt)
    for i in range(num_angles):
        a = math.pi * i / num_angles
        for r in range(line_length):
            t = r - kc
            x = int(round(kc + t * math.cos(a)))
            y = int(round(kc + t * math.sin(a)))
            if 0 <= x < line_length and 0 <= y < line_length:
                ker[i, 0, y, x] = 1.0
    lgs = F.conv2d(g, ker, padding=kc)
    lvc = F.conv2d(eroded, ker, padding=kc)
    avg = lgs / lvc.clamp(min=1.0)
    mask = lvc >= (min_valid_ratio * line_length)
    if not mask.any():
        return 0.0
    return avg[mask].max().item() / bg.item()


def _fallback_residual(elevation, valid_mask):
    if elevation.ndim > 2:
        elevation = elevation.squeeze()
        valid_mask = valid_mask.squeeze()
    H, W = elevation.shape
    y = torch.linspace(-1, 1, H, dtype=elevation.dtype)
    x = torch.linspace(-1, 1, W, dtype=elevation.dtype)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    vb = valid_mask.bool()
    if not vb.any():
        return 0.0
    Xv, Yv, Zv = X[vb].unsqueeze(1), Y[vb].unsqueeze(1), elevation[vb].unsqueeze(1)
    A = torch.cat([Xv, Yv, torch.ones_like(Xv)], dim=1)
    w = torch.linalg.lstsq(A, Zv).solution
    return torch.sqrt(torch.mean((Zv - A @ w) ** 2)).item()


def _fallback_sun(dtm, ortho, valid_mask):
    dev = dtm.device
    if dtm.ndim == 4: dtm = dtm[0]
    if ortho.ndim == 4: ortho = ortho[0]
    if valid_mask.ndim == 4: valid_mask = valid_mask[0]
    if ortho.shape[0] == 3:
        ortho = ortho.mean(dim=0, keepdim=True)
    sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=dev) / 8.0
    sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=dev) / 8.0
    spatial_scale = max(dtm.shape[-2], dtm.shape[-1]) / 2.0
    pad = F.pad(dtm.unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    nx = -F.conv2d(pad, sx.view(1, 1, 3, 3)) * spatial_scale
    ny = -F.conv2d(pad, sy.view(1, 1, 3, 3)) * spatial_scale
    nz = torch.ones_like(nx)
    normals = F.normalize(torch.cat([nx, ny, nz], dim=1), p=2, dim=1).squeeze(0)
    mask = valid_mask.squeeze(0).bool()
    N = normals[:, mask].t()
    Y = ortho.squeeze(0)[mask].unsqueeze(1)
    if N.shape[0] < 100:
        default = F.normalize(torch.tensor([0.5, -0.5, 1.0], device=dev), p=2, dim=0)
        return default, torch.tensor(1.0, device=dev), torch.tensor(0.3, device=dev)
    A = torch.cat([N, torch.ones((N.shape[0], 1), device=dev)], dim=1)
    x = torch.linalg.lstsq(A, Y).solution
    k = x[:3, 0]
    return F.normalize(k, p=2, dim=0), torch.norm(k, p=2), x[3, 0]


# Dispatcher ---------------------------------------------------------------
def _erode(m, erode_radius=1):
    return (_real_erode if _USING_REAL_ADAPTER else _fallback_erode)(m, erode_radius)


def _gmrf(i, d, m, **kw):
    return (_real_gmrf if _USING_REAL_ADAPTER else _fallback_gmrf)(i, d, m, **kw)


def _tin(e, m, **kw):
    return (_real_tin if _USING_REAL_ADAPTER else _fallback_tin)(e, m, **kw)


def _seam(e, m, **kw):
    return (_real_seam if _USING_REAL_ADAPTER else _fallback_seam)(e, m, **kw)


def _residual(e, m):
    return (_real_residual if _USING_REAL_ADAPTER else _fallback_residual)(e, m)


def _sun(d, o, m):
    return (_real_sun if _USING_REAL_ADAPTER else _fallback_sun)(d, o, m)


# ===========================================================================
# SYNTHETIC MARTIAN TERRAIN  (input to the real adapter algorithms)
# ===========================================================================

def _mars_rough_terrain(H=128, seed=7):
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, H), indexing="ij")
    large = (1.2 * np.sin(2.1 * xx + 0.3) * np.cos(1.7 * yy - 0.4)
             + 0.6 * np.exp(-((xx - 0.2) ** 2 + (yy + 0.3) ** 2) / 0.20))
    medium = ndimage.gaussian_filter(rng.standard_normal((H, H)), sigma=H * 0.04) * 0.35
    fine = ndimage.gaussian_filter(rng.standard_normal((H, H)), sigma=1.1) * 0.06
    dtm = large + medium + fine
    cy, cx = int(H * 0.68), int(H * 0.32)
    r = H * 0.12
    rr = np.sqrt((np.arange(H)[:, None] - cy) ** 2 + (np.arange(H)[None, :] - cx) ** 2)
    dtm += 0.55 * np.exp(-((rr - r) ** 2) / (2 * (r * 0.18) ** 2))
    dtm += -0.42 * np.exp(-(rr ** 2) / (2 * (r * 0.65) ** 2))
    return dtm.astype(np.float32)


def _void_mask(H, seed=11):
    rng = np.random.default_rng(seed)
    mask = np.ones((H, H), dtype=bool)
    mask[:, :int(H * 0.10)] = False
    mask[:, int(H * 0.92):] = False
    for _ in range(3):
        cy, cx = rng.integers(int(H * 0.25), int(H * 0.85), size=2)
        r = rng.integers(int(H * 0.05), int(H * 0.11))
        noise = ndimage.gaussian_filter(rng.standard_normal((H, H)), sigma=H * 0.015)
        rr = np.sqrt((np.arange(H)[:, None] - cy) ** 2 +
                     (np.arange(H)[None, :] - cx) ** 2)
        mask &= ~((rr + 18 * noise) < r)
    return mask


def _tin_flat_plane(H=128, seed=17):
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, H), indexing="ij")
    plateau_mask = (np.abs(xx) < 0.5) & (np.abs(yy) < 0.5)
    dtm = 0.12 * xx + 0.09 * yy + 0.0003 * rng.standard_normal((H, H))
    rough = ndimage.gaussian_filter(rng.standard_normal((H, H)), sigma=1.6) * 0.35
    return np.where(plateau_mask, dtm, dtm + rough).astype(np.float32)


def _seam_terrain(H=128, seed=23):
    rng = np.random.default_rng(seed)
    left = ndimage.gaussian_filter(rng.standard_normal((H, H)), sigma=H * 0.035) * 0.9
    right = ndimage.gaussian_filter(rng.standard_normal((H, H)), sigma=H * 0.035) * 0.9
    dtm = np.where(np.arange(H)[None, :] < int(H * 0.57), left, right + 0.55)
    seam = int(H * 0.57)
    dtm[:, seam] += 0.25
    dtm[:, seam + 1] += 0.12
    return dtm.astype(np.float32)


def _crater_patch(H=128, seed=29):
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, H), indexing="ij")
    dtm = 0.25 * xx - 0.18 * yy
    cy, cx = int(H * 0.5), int(H * 0.55)
    rr = np.sqrt((np.arange(H)[:, None] - cy) ** 2 + (np.arange(H)[None, :] - cx) ** 2)
    r = H * 0.22
    dtm += 0.8 * np.exp(-((rr - r) ** 2) / (2 * (r * 0.15) ** 2))
    dtm += -0.62 * np.exp(-(rr ** 2) / (2 * (r * 0.55) ** 2))
    for _ in range(4):
        hy, hx = rng.integers(int(H * 0.08), int(H * 0.92), size=2)
        rr2 = np.sqrt((np.arange(H)[:, None] - hy) ** 2 + (np.arange(H)[None, :] - hx) ** 2)
        dtm += 0.18 * np.exp(-(rr2 ** 2) / (2 * (H * 0.05) ** 2))
    dtm += ndimage.gaussian_filter(rng.standard_normal((H, H)), sigma=1.0) * 0.08
    return dtm.astype(np.float32)


def _lunar_lambert(dtm, sun_vec=(0.55, -0.45, 0.71),
                   intensity=1.0, ambient=0.22, lunar_w=0.45):
    H = dtm.shape[0]
    gy, gx = np.gradient(dtm)
    nx, ny, nz = -gx * (H / 2.0), -gy * (H / 2.0), np.ones_like(dtm)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    sv = np.asarray(sun_vec, dtype=np.float32)
    sv = sv / np.linalg.norm(sv)
    cos_i = np.clip(nx * sv[0] + ny * sv[1] + nz * sv[2], 0.01, 1.0)
    cos_e = np.clip(nz, 0.01, 1.0)
    refl = lunar_w * cos_i + (1.0 - lunar_w) * (cos_i / (cos_i + cos_e))
    return np.clip(intensity * refl + ambient, 0.0, 1.3)


# ===========================================================================
# REAL-DATA COLLECTION
# ===========================================================================

def _collect_real_data(H=128) -> dict:
    """Run the real adapter algorithms on synthetic terrain; return arrays."""
    torch.manual_seed(0)

    # ---- Scene: rough terrain w/ nodata voids ----------------------------
    rough = _mars_rough_terrain(H, seed=7)
    vmask = _void_mask(H, seed=11)
    rough_nan = rough.copy()
    rough_nan[~vmask] = np.nan

    dtm_t = torch.from_numpy(rough_nan).unsqueeze(0)
    mask_t = torch.from_numpy(vmask.astype(np.float32)).unsqueeze(0)
    ortho_gt = _lunar_lambert(rough)
    ortho_gt_in = ortho_gt.copy()
    ortho_gt_in[~vmask] = 0.0
    ortho_t = torch.from_numpy(ortho_gt_in).unsqueeze(0).expand(3, -1, -1).contiguous()

    eroded = _erode(mask_t, erode_radius=2)
    filled_img, filled_dtm, _ = _gmrf(
        ortho_t.clone(), dtm_t.clone(), mask_t.clone(),
        erode_radius=2, connectivity=4, tau=1.0, nugget=1e-6,
    )

    # ---- Scene: TIN flat plateau (rejected) ------------------------------
    tin_dtm = _tin_flat_plane(H, seed=17)
    tin_t = torch.from_numpy(tin_dtm).unsqueeze(0).unsqueeze(0)
    tin_valid = torch.ones_like(tin_t).bool()
    tin_score = _tin(tin_t, tin_valid, kernel_size=32)

    # ---- Scene: crater (accepted) ----------------------------------------
    crater_dtm = _crater_patch(H, seed=29)
    crater_t = torch.from_numpy(crater_dtm).unsqueeze(0).unsqueeze(0)
    crater_valid = torch.ones_like(crater_t).bool()
    crater_tin = _tin(crater_t, crater_valid, kernel_size=32)
    crater_residual = _residual(crater_t.squeeze(), crater_valid.float().squeeze())
    crater_seam = _seam(crater_t, crater_valid, line_length=35, num_angles=8)

    # ---- Scene: tile-merge seam (rejected) ------------------------------
    seam_dtm = _seam_terrain(H, seed=23)
    seam_t = torch.from_numpy(seam_dtm).unsqueeze(0).unsqueeze(0)
    seam_valid = torch.ones_like(seam_t).bool()
    seam_score = _seam(seam_t, seam_valid, line_length=35, num_angles=8)
    seam_residual = _residual(seam_t.squeeze(), seam_valid.float().squeeze())
    seam_tin = _tin(seam_t, seam_valid, kernel_size=32)

    # ---- Scene: low coverage (rejected by valid_ratio) ------------------
    low_cov_dtm = _crater_patch(H, seed=41)
    low_mask = np.zeros((H, H), dtype=bool)
    low_mask[:int(H * 0.30), :] = True
    low_valid_ratio = float(low_mask.mean())

    # ---- Scene: near-flat slope (rejected by residual) ------------------
    flat_dtm = (0.4 * np.linspace(-1, 1, H)[None, :]
                * np.ones((H, H))).astype(np.float32)
    flat_dtm += 0.003 * np.random.default_rng(0).standard_normal((H, H)).astype(np.float32)
    flat_t = torch.from_numpy(flat_dtm).unsqueeze(0).unsqueeze(0)
    flat_valid = torch.ones_like(flat_t).bool()
    flat_residual = _residual(flat_t.squeeze(), flat_valid.float().squeeze())
    flat_tin = _tin(flat_t, flat_valid, kernel_size=32)

    # ---- Sun vector OLS on crater ---------------------------------------
    crater_ortho = _lunar_lambert(crater_dtm)
    sun_vec, intensity, ambient = _sun(
        torch.from_numpy(crater_dtm).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(crater_ortho).unsqueeze(0).unsqueeze(0)
            .expand(1, 3, -1, -1),
        crater_valid.float(),
    )

    # ---- Flow-matching demo ---------------------------------------------
    rng = np.random.default_rng(3)
    z_src = ndimage.gaussian_filter(rng.standard_normal((H, H)), sigma=2.5) * 0.6
    z_tgt = (crater_dtm - crater_dtm.mean()) / (crater_dtm.std() + 1e-6)
    t_steps = [0.0, 0.25, 0.5, 0.75, 1.0]
    flow_evo = [(1 - t) * z_src + t * z_tgt
                + 1e-4 * rng.standard_normal((H, H)) for t in t_steps]

    ln_samples = 1.0 / (1.0 + np.exp(-rng.standard_normal(8000)))

    gy, gx = np.gradient(crater_dtm)
    nx = -gx * (H / 2.0)
    ny = -gy * (H / 2.0)
    nz = np.ones_like(nx)
    nrm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    normals_rgb = np.stack([(nx / nrm + 1) / 2,
                            (ny / nrm + 1) / 2,
                            (nz / nrm + 1) / 2], axis=-1)

    def _gmag(arr, s):
        if s > 1:
            arr = arr[::s, ::s]
        gyy, gxx = np.gradient(arr)
        return np.sqrt(gxx ** 2 + gyy ** 2)

    grad_1 = _gmag(crater_dtm, 1)
    grad_2 = _gmag(crater_dtm, 2)
    grad_4 = _gmag(crater_dtm, 4)

    fft_mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(crater_dtm))))

    v_tgt_norm = np.abs(z_tgt - z_src)
    v_pred_norm = v_tgt_norm + 0.08 * rng.standard_normal(v_tgt_norm.shape)

    pred_ortho = crater_ortho + 0.06 * rng.standard_normal(crater_ortho.shape)
    pred_ortho = np.clip(pred_ortho, 0, 1.3)

    # ---- NEW: synthetic pred DTM for Huber / Laplacian / Ordinal --------
    # A plausible prediction: GT + low-frequency noise + small global bias,
    # matching the noise model used by _make_synthetic_pred in the viz code.
    _noise_hw = max(crater_dtm.shape[0] // 8, 1)
    _lf_noise = ndimage.zoom(
        rng.standard_normal((_noise_hw, _noise_hw)),
        crater_dtm.shape[0] / _noise_hw, order=3,
    )[: crater_dtm.shape[0], : crater_dtm.shape[1]]
    _bias = rng.standard_normal() * 0.03
    pred_dtm = np.clip(crater_dtm + 0.08 * _lf_noise + _bias, -1.0, 1.0).astype(np.float32)

    # ---- Huber loss (signed error map + per-pixel huber value) ----------
    huber_delta = 0.1
    huber_err = (pred_dtm - crater_dtm).astype(np.float32)
    _absr = np.abs(huber_err)
    huber_map = np.where(
        _absr <= huber_delta,
        0.5 * _absr ** 2,
        huber_delta * (_absr - 0.5 * huber_delta),
    ).astype(np.float32)

    # ---- Laplacian (5-point stencil, matches LaplacianLoss) -------------
    # Discrete Laplacian of the GT crater: concave/convex signature.
    gt_laplacian = ndimage.laplace(crater_dtm).astype(np.float32)
    pred_laplacian = ndimage.laplace(pred_dtm).astype(np.float32)
    lap_err = np.abs(pred_laplacian - gt_laplacian).astype(np.float32)

    # ---- Ordinal ranking: sample pairs, compute violations --------------
    ord_margin = 0.02
    ord_num_pairs = 2000
    ord_draw = 220  # pairs to actually draw in the viz
    _H, _W = crater_dtm.shape
    _N = _H * _W
    _rng_ord = np.random.default_rng(1234)
    _idx_i = _rng_ord.integers(0, _N, size=ord_num_pairs)
    _idx_j = _rng_ord.integers(0, _N, size=ord_num_pairs)
    _gt_flat = crater_dtm.ravel()
    _pr_flat = pred_dtm.ravel()
    _gt_diff = _gt_flat[_idx_i] - _gt_flat[_idx_j]
    _pr_diff = _pr_flat[_idx_i] - _pr_flat[_idx_j]
    _ordered = np.abs(_gt_diff) > ord_margin
    _violations = _ordered & (np.sign(_gt_diff) * _pr_diff <= 0)
    # Sub-sample for drawing so the overlay stays legible
    _draw_idx = _rng_ord.choice(ord_num_pairs, size=ord_draw, replace=False)
    ord_pairs = {
        "yi": (_idx_i[_draw_idx] // _W).astype(np.int32),
        "xi": (_idx_i[_draw_idx] %  _W).astype(np.int32),
        "yj": (_idx_j[_draw_idx] // _W).astype(np.int32),
        "xj": (_idx_j[_draw_idx] %  _W).astype(np.int32),
        "ordered": _ordered[_draw_idx],
        "violations": _violations[_draw_idx],
    }
    ord_violation_rate = float(_violations.sum() / max(_ordered.sum(), 1))

    return {
        # preprocessing chain
        "rough_clean": rough,
        "rough_with_voids": rough_nan,
        "void_mask": vmask,
        "eroded_mask": eroded.squeeze().cpu().numpy(),
        "filled_dtm": filled_dtm.squeeze().cpu().numpy(),
        "filled_ortho": filled_img.mean(dim=0).cpu().numpy(),
        # reject/accept samples
        "tin_dtm": tin_dtm,
        "tin_score": tin_score,
        "crater_dtm": crater_dtm,
        "crater_ortho": crater_ortho,
        "crater_tin": crater_tin,
        "crater_residual": crater_residual,
        "crater_seam": crater_seam,
        "seam_dtm": seam_dtm,
        "seam_score": seam_score,
        "seam_residual": seam_residual,
        "seam_tin": seam_tin,
        "low_cov_dtm": low_cov_dtm,
        "low_cov_mask": low_mask,
        "low_cov_valid_ratio": low_valid_ratio,
        "flat_dtm": flat_dtm,
        "flat_residual": flat_residual,
        "flat_tin": flat_tin,
        # sun
        "sun_vec": sun_vec.cpu().numpy() if torch.is_tensor(sun_vec) else np.asarray(sun_vec),
        "sun_intensity": float(intensity),
        "sun_ambient": float(ambient),
        # flow
        "z_src": z_src,
        "z_tgt": z_tgt,
        "flow_evo": flow_evo,
        "t_steps": t_steps,
        "logit_normal_samples": ln_samples,
        # loss ingredients
        "normals_rgb": normals_rgb,
        "grad_1": grad_1,
        "grad_2": grad_2,
        "grad_4": grad_4,
        "fft_mag": fft_mag,
        "v_tgt_norm": v_tgt_norm,
        "v_pred_norm": v_pred_norm,
        "pred_ortho": pred_ortho,
        # NEW: Huber / Laplacian / Ordinal loss ingredients
        "pred_dtm": pred_dtm,
        "huber_err": huber_err,
        "huber_map": huber_map,
        "huber_delta": huber_delta,
        "gt_laplacian": gt_laplacian,
        "pred_laplacian": pred_laplacian,
        "lap_err": lap_err,
        "ord_pairs": ord_pairs,
        "ord_margin": ord_margin,
        "ord_num_pairs": ord_num_pairs,
        "ord_violation_rate": ord_violation_rate,
    }


# ===========================================================================
# PNG PANEL RENDERING  (for graphviz image nodes)
# ===========================================================================

def _save_panel(path: Path, arr: np.ndarray, cmap: str = "mako",
                vmin=None, vmax=None, figsize=(2.4, 2.4), dpi=200,
                frame_color: str | None = None, extra_draw=None):
    """Save a borderless image panel to disk."""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        ax.imshow(arr, interpolation="nearest")
    else:
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        if frame_color is None:
            sp.set_visible(False)
        else:
            sp.set_color(frame_color); sp.set_linewidth(3)
    if extra_draw is not None:
        extra_draw(ax)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, dpi=dpi)
    plt.close(fig)


def _save_hist(path: Path, samples: np.ndarray, *,
               color="#6b5ea8", figsize=(3.6, 2.0), dpi=200,
               title: str | None = None):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0.1, 0.22, 0.86, 0.68])
    ax.hist(samples, bins=60, color=color, edgecolor="none", alpha=0.88)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=7.5)
    for s in ax.spines.values():
        s.set_color("#999"); s.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=8.5, pad=4)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03, dpi=dpi)
    plt.close(fig)


def _save_strip(path: Path, arrays: list[np.ndarray], *,
                cmap="mako", figsize=(7.5, 1.7), dpi=200,
                labels: list[str] | None = None,
                frame_color: str | None = None):
    """Horizontal strip of imshow panels (for flow evolution)."""
    fig, axs = plt.subplots(1, len(arrays), figsize=figsize, dpi=dpi)
    if len(arrays) == 1:
        axs = [axs]
    for i, (ax, arr) in enumerate(zip(axs, arrays)):
        ax.imshow(arr, cmap=cmap, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            if frame_color is None:
                sp.set_visible(False)
            else:
                sp.set_color(frame_color); sp.set_linewidth(1.5)
        if labels:
            ax.set_title(labels[i], fontsize=9)
    fig.subplots_adjust(left=0, right=1, top=0.88, bottom=0, wspace=0.04)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03, dpi=dpi)
    plt.close(fig)


def _save_panels(data: dict, out_dir: Path) -> dict[str, Path]:
    """Render every PNG graphviz will reference, return {key: path} dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    def _p(name: str) -> Path:
        path = out_dir / f"{name}.png"
        paths[name] = path
        return path

    vmin_r = float(np.nanpercentile(data["rough_clean"], 2))
    vmax_r = float(np.nanpercentile(data["rough_clean"], 98))

    # Preprocessing chain
    _save_panel(_p("raw_voids"), data["rough_with_voids"],
                cmap="mako", vmin=vmin_r, vmax=vmax_r)
    _save_panel(_p("eroded_mask"), data["eroded_mask"],
                cmap="gray", vmin=0, vmax=1)
    _save_panel(_p("gmrf_fill"), data["filled_dtm"],
                cmap="mako", vmin=vmin_r, vmax=vmax_r)

    # Reject / accept patches
    _save_panel(_p("reject_lowcov_dtm"), data["low_cov_dtm"],
                cmap="mako", frame_color=C["node_filter"])
    # low-coverage ortho with mask applied — grey where missing
    lc_ortho = _lunar_lambert(data["low_cov_dtm"]).copy()
    lc_ortho[~data["low_cov_mask"]] = 0.0
    _save_panel(_p("reject_lowcov_ortho"), lc_ortho,
                cmap="gray", frame_color=C["node_filter"])

    _save_panel(_p("reject_tin_dtm"), data["tin_dtm"], cmap="mako",
                frame_color=C["node_filter"])
    _save_panel(_p("reject_tin_ortho"), _lunar_lambert(data["tin_dtm"]),
                cmap="gray", frame_color=C["node_filter"])

    _save_panel(_p("reject_seam_dtm"), data["seam_dtm"], cmap="mako",
                frame_color=C["node_filter"])
    _save_panel(_p("reject_seam_ortho"), _lunar_lambert(data["seam_dtm"]),
                cmap="gray", frame_color=C["node_filter"])

    _save_panel(_p("reject_flat_dtm"), data["flat_dtm"], cmap="mako",
                frame_color=C["node_filter"])
    _save_panel(_p("reject_flat_ortho"), _lunar_lambert(data["flat_dtm"]),
                cmap="gray", frame_color=C["node_filter"])

    _save_panel(_p("accept_crater_dtm"), data["crater_dtm"], cmap="mako",
                frame_color=C["node_accept"])
    _save_panel(_p("accept_crater_ortho"), data["crater_ortho"], cmap="gray",
                frame_color=C["node_accept"])

    # Accept variants for the overview diagram
    c2 = _crater_patch(seed=77)
    c3 = _crater_patch(seed=103)
    _save_panel(_p("accept_crater2_dtm"), c2, cmap="mako",
                frame_color=C["node_accept"])
    _save_panel(_p("accept_crater2_ortho"), _lunar_lambert(c2),
                cmap="gray", frame_color=C["node_accept"])
    _save_panel(_p("accept_crater3_dtm"), c3, cmap="mako",
                frame_color=C["node_accept"])
    _save_panel(_p("accept_crater3_ortho"), _lunar_lambert(c3),
                cmap="gray", frame_color=C["node_accept"])

    # Flow + losses
    _save_hist(_p("logit_normal"), data["logit_normal_samples"],
               color=C["node_unet"],
               title="$t \\sim \\mathrm{Logit\\text{-}Normal}(0,1)$")

    _save_strip(_p("flow_evolution"), data["flow_evo"],
                cmap="mako",
                labels=[f"t = {t:.2f}" for t in data["t_steps"]])

    _save_panel(_p("v_tgt_norm"), data["v_tgt_norm"], cmap="flare",
                frame_color=C["node_loss_vel"])
    _save_panel(_p("v_pred_norm"), data["v_pred_norm"], cmap="flare",
                frame_color=C["node_loss_vel"])

    _save_panel(_p("normals_rgb"), data["normals_rgb"],
                frame_color=C["node_loss_pix"])
    _save_panel(_p("fft_mag"), data["fft_mag"], cmap="mako",
                frame_color=C["node_loss_pix"])

    _save_strip(_p("grad_multiscale"),
                [data["grad_1"], data["grad_2"], data["grad_4"]],
                cmap="flare",
                labels=["$\\nabla D^{(1)}$ (1×)",
                        "$\\nabla D^{(2)}$ (2×)",
                        "$\\nabla D^{(4)}$ (4×)"],
                frame_color=C["node_loss_pix"])

    _save_panel(_p("render_gt"), data["crater_ortho"], cmap="gray",
                frame_color=C["node_loss_pix"])
    _save_panel(_p("render_pred"), data["pred_ortho"], cmap="gray",
                frame_color=C["node_loss_pix"])

    # --------------------------------------------------------------
    # NEW: Huber / Laplacian / Ordinal panels
    # --------------------------------------------------------------

    # Huber: signed error map on diverging scale, clipped at ±2*delta so the
    # L1/L2 transition band is visually centred. This matches exactly what
    # `AbsoluteDepthLoss.per_pixel_loss` operates on.
    _hd = data["huber_delta"]
    _err = data["huber_err"]
    _err_clip = np.clip(_err, -2 * _hd, 2 * _hd)
    _save_panel(_p("huber_err"), _err_clip, cmap="RdBu_r",
                vmin=-2 * _hd, vmax=2 * _hd,
                frame_color=C["node_loss_pix"])

    # Laplacian: GT curvature on a symmetric diverging scale. Crater bowls
    # appear blue (concave), rims red (convex), flat regions white — this is
    # the signal the Laplacian loss forces the prediction to match.
    _lvl = float(np.nanpercentile(np.abs(data["gt_laplacian"]), 99) + 1e-6)
    _save_panel(_p("laplacian_gt"), data["gt_laplacian"], cmap="RdBu_r",
                vmin=-_lvl, vmax=_lvl,
                frame_color=C["node_loss_pix"])

    # Ordinal: pair scatter overlay on the crater ortho. Green = correct
    # ordering, red = violation. Matches the visualization in the
    # OrdinalRankingLoss inspection plot.
    _pairs = data["ord_pairs"]
    _ortho_bg = data["crater_ortho"]

    def _draw_pairs(ax):
        ordered = _pairs["ordered"]
        viols = _pairs["violations"]
        yi, xi = _pairs["yi"], _pairs["xi"]
        yj, xj = _pairs["yj"], _pairs["xj"]
        # Draw ambiguous (grey) first, then correct (green), then violations (red)
        for k in range(len(yi)):
            if not ordered[k]:
                ax.plot([xi[k], xj[k]], [yi[k], yj[k]], "-",
                        color="#888888", linewidth=0.35, alpha=0.55, zorder=1)
        for k in range(len(yi)):
            if ordered[k] and not viols[k]:
                ax.plot([xi[k], xj[k]], [yi[k], yj[k]], "-",
                        color="#22cc55", linewidth=0.55, alpha=0.85, zorder=2)
        for k in range(len(yi)):
            if viols[k]:
                ax.plot([xi[k], xj[k]], [yi[k], yj[k]], "-",
                        color="#ff2a2a", linewidth=1.1, alpha=0.95, zorder=3)
        ax.set_xlim(0, _ortho_bg.shape[1])
        ax.set_ylim(_ortho_bg.shape[0], 0)

    _save_panel(_p("ordinal_overlay"), _ortho_bg, cmap="gray",
                frame_color=C["node_loss_pix"],
                extra_draw=_draw_pairs)

    return paths


# ===========================================================================
# GRAPHVIZ HELPERS
# ===========================================================================

def _apply_base_style(dot: graphviz.Digraph, rankdir: str = "TB"):
    dot.attr(rankdir=rankdir, splines="spline", nodesep="0.35", ranksep="0.55",
             fontname="Helvetica", compound="true", dpi="300",
             bgcolor="transparent")
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="#FFFFFF",
             color=C["border"], fontname="Helvetica", fontsize="11",
             penwidth="1.1", margin="0.14,0.08")
    dot.attr("edge", fontname="Helvetica", fontsize="9",
             color=C["edge"], penwidth="1.0", arrowsize="0.7")


def _h(text: str) -> str:
    """Escape user text for safe inclusion in a graphviz HTML-like label.

    Only `&`, `<` and `>` need escaping — quotes inside attribute values
    are not an issue because we always wrap attribute values in double
    quotes and never interpolate user text inside attribute values.
    """
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))


def _image_node(
    g: graphviz.Digraph,
    node_id: str,
    image_path: Path,
    *,
    title: str,
    captions: list[tuple[str, str]] | None = None,
    header_color: str = "#1a6b7c",
    accent_color: str | None = None,
    img_width: str = "1.9",
    title_pt: int = 10,
    caption_pt: int = 9,
):
    """Node = image on top + coloured-caption rows below, rendered via HTML label.

    ``captions`` is a list of (text, colour) tuples rendered one per row.
    The node has a coloured header strip with the title and a bottom strip with
    the metric captions.
    """
    accent_color = accent_color or header_color
    # Build HTML label — user text MUST be HTML-escaped so that
    # `<` / `>` in captions (like "valid_ratio < 0.5") doesn't break parsing.
    cap_rows = ""
    if captions:
        for text, col in captions:
            cap_rows += (
                f'<TR><TD ALIGN="CENTER" BGCOLOR="#ffffff">'
                f'<FONT FACE="Helvetica" POINT-SIZE="{caption_pt}" COLOR="{col}">'
                f'{_h(text)}'
                f'</FONT></TD></TR>'
            )

    label = (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3" '
        f'BGCOLOR="#ffffff" COLOR="{accent_color}">'
        f'<TR><TD BGCOLOR="{header_color}" ALIGN="CENTER">'
        f'<FONT FACE="Helvetica-Bold" POINT-SIZE="{title_pt}" COLOR="#ffffff">'
        f'{_h(title)}</FONT></TD></TR>'
        f'<TR><TD ALIGN="CENTER" FIXEDSIZE="FALSE">'
        # Use absolute path — dot resolves image paths relative to its cwd
        # (where it is invoked from), not relative to the .dot file location.
        f'<IMG SRC="{image_path.resolve().as_posix()}" SCALE="TRUE"/>'
        f'</TD></TR>'
        f'{cap_rows}'
        '</TABLE>>'
    )
    g.node(node_id, label=label, shape="none", margin="0",
           penwidth="1.5", color=accent_color, imagescale="true",
           width=img_width)


def _text_node(
    g: graphviz.Digraph,
    node_id: str,
    text_lines: list[str],
    *,
    fill: str,
    text_color: str = "#ffffff",
    shape: str = "box",
    penwidth: str = "1.3",
    fontsize: int = 10,
    bold_first: bool = True,
):
    """Rounded coloured node. First line bold, subsequent lines normal."""
    if not text_lines:
        text_lines = [""]
    # Build HTML label — escape user text.
    bold = _h(text_lines[0])
    rest = [_h(r) for r in text_lines[1:]]
    label_html = (
        f'<<FONT FACE="Helvetica-Bold" POINT-SIZE="{fontsize}" '
        f'COLOR="{text_color}">{bold}</FONT>'
    )
    for r in rest:
        label_html += (
            f'<BR/><FONT FACE="Helvetica" POINT-SIZE="{fontsize - 1}" '
            f'COLOR="{text_color}">{r}</FONT>'
        )
    label_html += ">"
    g.node(node_id, label=label_html, shape=shape,
           style="rounded,filled", fillcolor=fill,
           color=fill, fontcolor=text_color, penwidth=penwidth)


# ===========================================================================
# FIGURE 1 — Full Pipeline Overview
# ===========================================================================

def build_pipeline_overview(data: dict, paths: dict[str, Path],
                            out_base: Path) -> None:
    dot = graphviz.Digraph(name="MarsDepthFM_Pipeline",
                           node_attr={}, edge_attr={})
    _apply_base_style(dot, rankdir="TB")

    # =======================================================
    # PHASE 0 — Data Infrastructure
    # =======================================================
    with dot.subgraph(name="cluster_phase0") as c0:
        c0.attr(label="Phase 0:  Data Infrastructure",
                style="solid", color=C["cluster_line"],
                bgcolor=C["bg_infra"], fontname="Helvetica-Bold",
                fontsize="13", fontcolor=C["node_data"], labeljust="l")

        c0.node("pds", "NASA PDS HiRISE RDR\nODE REST  ·  DTMCUMINDEX.TAB",
                shape="cylinder", fillcolor=C["node_data"],
                fontcolor="white", fontsize="10", fontname="Helvetica-Bold",
                penwidth="1.4")
        _text_node(c0, "cog",
                   ["JP2 → Cloud-Optimized GeoTIFF",
                    "512² tiled, LZW compression"],
                   fill=C["node_infra"])
        _text_node(c0, "geodata",
                   ["MarsHiRISEDTM  (TorchGeo)",
                    "raster mosaic + strip metadata"],
                   fill=C["node_infra"])
        _text_node(c0, "build1",
                   ["build_litdata  Phase 1",
                    "fork pool → GDAL → .npz tiles"],
                   fill=C["node_preproc"])
        _text_node(c0, "build2",
                   ["build_litdata  Phase 2",
                    "spawn → LitData chunks"],
                   fill=C["node_preproc"])
        _text_node(c0, "stats",
                   ["compute_dataset_stats",
                    "4-GPU Welford → p02 · p98 · centered_p98"],
                   fill=C["node_preproc"])
        _text_node(c0, "sampler",
                   ["HiRISEGeoSampler",
                    "strip-aware patch grid · size = 0.018°"],
                   fill=C["node_infra"])
        _text_node(c0, "split",
                   ["Geographic longitude split",
                    "train 80 % | val 10 % | test 10 %"],
                   fill=C["node_infra"])

        c0.edge("pds", "cog")
        c0.edge("cog", "geodata")
        c0.edge("geodata", "build1", xlabel=" valid strips ")
        c0.edge("build1", "build2")
        c0.edge("build2", "stats")
        c0.edge("geodata", "sampler", xlabel=" raster index ", constraint="false",
                color=C["edge"], style="solid")
        c0.edge("sampler", "split")

    # =======================================================
    # PHASE 1 — Manifest Filter (REJECTED examples as image nodes)
    # =======================================================
    with dot.subgraph(name="cluster_phase1") as c1:
        c1.attr(label="Phase 1:  Manifest Quality Filter  "
                      "— rejected patches (real algorithm outputs)",
                style="solid", color=C["cluster_line"],
                bgcolor=C["bg_filter"], fontname="Helvetica-Bold",
                fontsize="13", fontcolor=C["node_filter"], labeljust="l")

        # Four reject image nodes
        _image_node(
            c1, "rej_lowcov", paths["reject_lowcov_dtm"],
            title="Low coverage",
            header_color=C["node_filter"],
            captions=[
                (f"valid_ratio = {data['low_cov_valid_ratio']:.3f}  <  0.50",
                 C["node_filter"]),
                ("→ REJECT", C["node_filter"]),
            ],
        )
        _image_node(
            c1, "rej_tin", paths["reject_tin_dtm"],
            title="TIN flat plateau",
            header_color=C["node_filter"],
            captions=[
                (f"is_tin = {data['tin_score']:.3f}  >  0.95",
                 C["node_filter"]),
                ("→ REJECT", C["node_filter"]),
            ],
        )
        _image_node(
            c1, "rej_seam", paths["reject_seam_dtm"],
            title="Tile-merge seam",
            header_color=C["node_filter"],
            captions=[
                (f"seam = {data['seam_score']:.1f}×  ·  num_merges = 2",
                 C["node_filter"]),
                ("→ REJECT", C["node_filter"]),
            ],
        )
        _image_node(
            c1, "rej_flat", paths["reject_flat_dtm"],
            title="Near-flat slope",
            header_color=C["node_filter"],
            captions=[
                (f"residual = {data['flat_residual']:.3f}  <  0.10",
                 C["node_filter"]),
                ("→ REJECT", C["node_filter"]),
            ],
        )

        c1.node("filter_rule",
                label=("<<FONT FACE='Helvetica-Bold' POINT-SIZE='10' "
                       "COLOR='#ffffff'>Manifest filter rule</FONT>"
                       "<BR/><FONT FACE='Helvetica' POINT-SIZE='9' "
                       "COLOR='#ffffff'>valid_ratio ≥ 0.5  ∧  residual ≥ 0.1"
                       "</FONT>"
                       "<BR/><FONT FACE='Helvetica' POINT-SIZE='9' "
                       "COLOR='#ffffff'>is_tin ≤ 0.95  ∧  num_merges = 1"
                       "</FONT>>"),
                shape="hexagon", style="filled",
                fillcolor=C["node_filter"], color=C["node_filter"],
                penwidth="1.5")

        # Keep the four reject image nodes on the same row.  We inject raw
        # DOT because graphviz.Digraph.subgraph() with no name emits just
        # `{ ... }` (a statement list, not a subgraph), which doesn't scope
        # `rank=same` correctly and produces a DOT syntax error.
        c1.body.append(
            "\t{ rank=same; rej_lowcov; rej_tin; rej_seam; rej_flat; }\n"
        )

        c1.edge("rej_lowcov", "filter_rule", style="dashed",
                color=C["node_filter"], arrowhead="none")
        c1.edge("rej_tin", "filter_rule", style="dashed",
                color=C["node_filter"], arrowhead="none")
        c1.edge("rej_seam", "filter_rule", style="dashed",
                color=C["node_filter"], arrowhead="none")
        c1.edge("rej_flat", "filter_rule", style="dashed",
                color=C["node_filter"], arrowhead="none")

        # Accepted: a clean crater in the same cluster
        _image_node(
            c1, "accepted", paths["accept_crater_dtm"],
            title="Accepted crater",
            header_color=C["node_accept"],
            captions=[
                (f"is_tin = {data['crater_tin']:.3f}", C["node_accept"]),
                (f"residual = {data['crater_residual']:.3f}", C["node_accept"]),
                ("→ all four pass", C["node_accept"]),
            ],
        )
        c1.edge("filter_rule", "accepted",
                xlabel=" clean_records.parquet ",
                color=C["node_accept"], penwidth="1.6")

    # =======================================================
    # PHASE 2 — Preprocessing (visual chain)
    # =======================================================
    with dot.subgraph(name="cluster_phase2") as c2:
        c2.attr(label="Phase 2:  Per-sample Preprocessing  "
                      "(fill_voids_gmrf + relative norm)",
                style="solid", color=C["cluster_line"],
                bgcolor=C["bg_preproc"], fontname="Helvetica-Bold",
                fontsize="13", fontcolor=C["node_preproc"], labeljust="l")

        _image_node(c2, "pp_raw", paths["raw_voids"],
                    title="(1) Raw DTM + nodata",
                    header_color=C["node_filter"],
                    captions=[("NaN voids from strip edges", C["node_filter"]),
                              ("+ interior dropouts", C["node_filter"])])

        _image_node(c2, "pp_erode", paths["eroded_mask"],
                    title="(2) Eroded valid mask",
                    header_color=C["node_preproc"],
                    captions=[("erode_valid_mask(r = 2)", C["node_preproc"]),
                              ("max_pool of ¬mask", C["node_preproc"])])

        _image_node(c2, "pp_fill", paths["gmrf_fill"],
                    title="(3) GMRF void fill",
                    header_color=C["node_preproc"],
                    captions=[("fill_voids_gmrf, τ=1, 4-conn",
                               C["node_preproc"]),
                              ("sparse u_void = −Q_vv⁻¹ Q_vo u_obs",
                               C["node_preproc"])])

        _text_node(c2, "pp_norm",
                   ["Relative DTM normalisation",
                    "local centre – centered_p98 scale",
                    "clamp to [-1, 1]"],
                   fill=C["node_preproc"])

        c2.body.append(
            "\t{ rank=same; pp_raw; pp_erode; pp_fill; pp_norm; }\n"
        )
        c2.edge("pp_raw", "pp_erode", xlabel=" r = 2 ")
        c2.edge("pp_erode", "pp_fill", xlabel=" conditional fill ")
        c2.edge("pp_fill", "pp_norm", xlabel=" filled ∈ ℝ ")

    # =======================================================
    # PHASE 3 — Flow-Matching Training (compact)
    # =======================================================
    with dot.subgraph(name="cluster_phase3") as c3:
        c3.attr(label="Phase 3:  Flow-Matching Forward",
                style="solid", color=C["cluster_line"],
                bgcolor=C["bg_flow"], fontname="Helvetica-Bold",
                fontsize="13", fontcolor=C["node_unet"], labeljust="l")

        _text_node(c3, "vae_enc",
                   ["VAE encode  (frozen)",
                    "z = E(x) · 0.18215"],
                   fill=C["node_latent"])
        _text_node(c3, "q_sample",
                   ["x_src = q_sample(z_img)",
                    "noising_step = 200"],
                   fill=C["node_latent"])
        _text_node(c3, "t_sample",
                   ["t ~ Logit-Normal(0, 1)",
                    "per-sample scalar"],
                   fill=C["node_unet"])
        _text_node(c3, "interp",
                   ["Noisy interpolant",
                    "z_t = (1−t) x_src + t z_depth  +  σ_min ε"],
                   fill=C["node_unet"])
        _text_node(c3, "unet",
                   ["UNet  vθ(z_t, t ; z_img)",
                    "SD 2.1 backbone  ·  8 → 4 ch"],
                   fill=C["node_unet"])

        _image_node(c3, "flow_strip", paths["flow_evolution"],
                    title="z_t evolution  (real interpolation)",
                    header_color=C["node_unet"],
                    img_width="3.5",
                    captions=[("t = 0.00  ·  0.25  ·  0.50  ·  0.75  ·  1.00",
                               C["text"])])

        c3.edge("vae_enc", "q_sample")
        c3.edge("q_sample", "interp", xlabel=" x_src ")
        c3.edge("t_sample", "interp", xlabel=" t ")
        c3.edge("interp", "unet", xlabel=" z_t ")
        c3.edge("interp", "flow_strip", style="dashed", constraint="false",
                color=C["edge"], arrowhead="none")

    # =======================================================
    # PHASE 4 — Loss Branches
    # =======================================================
    with dot.subgraph(name="cluster_phase4") as c4:
        c4.attr(label="Phase 4:  Loss Branches",
                style="solid", color=C["cluster_line"],
                bgcolor=C["bg_loss"], fontname="Helvetica-Bold",
                fontsize="13", fontcolor=C["node_loss_pix"], labeljust="l")

        _text_node(c4, "l_vel",
                   ["𝓛_vel   (latent branch)",
                    "‖ vθ − (z_depth − x_src) ‖²  ⊙  M_valid"],
                   fill=C["node_loss_vel"])
        _text_node(c4, "l_photo",
                   ["𝓛_photo   (pixel)",
                    "(1−α)(1 − Pearson) + α(1 − SSIM)",
                    "Lunar-Lambert render"],
                   fill=C["node_loss_pix"])
        _text_node(c4, "l_grad",
                   ["𝓛_grad   (pixel)",
                    "Σ_{s∈{1,2,4}} ‖∇D_pred^(s) − ∇D_GT^(s)‖₁"],
                   fill=C["node_loss_pix"])
        _text_node(c4, "l_norm",
                   ["𝓛_norm   (pixel)",
                    "1 − cos⟨ N_pred, N_GT ⟩"],
                   fill=C["node_loss_pix"])
        _text_node(c4, "l_ffl",
                   ["𝓛_FFL   (pixel, optional)",
                    "focal-frequency penalty"],
                   fill=C["node_loss_pix"])
        # ---- NEW: Huber / Laplacian / Ordinal losses ----
        _text_node(c4, "l_huber",
                   ["𝓛_Huber   (pixel)",
                    "Huber_δ (D_pred − D_GT),  δ = 0.1",
                    "absolute depth accuracy in [-1, 1]"],
                   fill=C["node_loss_pix"])
        _text_node(c4, "l_lap",
                   ["𝓛_Lap   (pixel)",
                    "‖ ∇²D_pred − ∇²D_GT ‖₁",
                    "curvature: crater bowls & rims"],
                   fill=C["node_loss_pix"])
        _text_node(c4, "l_ord",
                   ["𝓛_Ord   (pixel)",
                    "pairwise hinge on sampled pairs",
                    "noise-robust relative ordering"],
                   fill=C["node_loss_pix"])

        c4.node("l_total",
                label="<<FONT FACE='Helvetica-Bold' POINT-SIZE='11' "
                      "COLOR='#ffffff'>𝓛_total</FONT>"
                      "<BR/><FONT FACE='Helvetica' POINT-SIZE='9' "
                      "COLOR='#ffffff'>w_v·𝓛_vel + w_p·𝓛_photo + "
                      "w_g·𝓛_grad + w_n·𝓛_norm + w_f·𝓛_FFL</FONT>"
                      "<BR/><FONT FACE='Helvetica' POINT-SIZE='9' "
                      "COLOR='#ffffff'>+ w_h·𝓛_Huber + w_ℓ·𝓛_Lap + "
                      "w_o·𝓛_Ord</FONT>"
                      "<BR/><FONT FACE='Helvetica' POINT-SIZE='8' "
                      "COLOR='#ffffff'>yaml: 1.0 | 1.5 | 100 | 1.0 | 0 | "
                      "3.0 | 0.05 | 1.0</FONT>>",
                shape="hexagon", style="filled",
                fillcolor=C["text"], color=C["text"],
                fontcolor="white", penwidth="1.6")

        c4.edge("l_vel", "l_total")
        c4.edge("l_photo", "l_total")
        c4.edge("l_grad", "l_total")
        c4.edge("l_norm", "l_total")
        c4.edge("l_ffl", "l_total")
        c4.edge("l_huber", "l_total")
        c4.edge("l_lap", "l_total")
        c4.edge("l_ord", "l_total")

    # =======================================================
    # PHASE 5 — Optimiser + Artifacts
    # =======================================================
    with dot.subgraph(name="cluster_phase5") as c5:
        c5.attr(label="Phase 5:  Optimiser & Artifacts",
                style="solid", color=C["cluster_line"],
                bgcolor=C["bg_optim"], fontname="Helvetica-Bold",
                fontsize="13", fontcolor=C["node_optim"], labeljust="l")

        _text_node(c5, "adamw",
                   ["AdamW",
                    "lr = 2e-4  ·  β = (0.9, 0.999)  ·  wd = 0.01",
                    "cosine schedule + 500-step warmup",
                    "grad-accum 4  ·  bf16  ·  ‖∇θ‖ ≤ 1"],
                   fill=C["node_optim"])
        _text_node(c5, "ema",
                   ["EMA copy",
                    "decay = 0.99  ·  CPU buffer"],
                   fill=C["node_artifact"])
        _text_node(c5, "wandb",
                   ["WandB logging",
                    "loss / vis / grad-norm every 10 steps"],
                   fill=C["node_artifact"])
        _text_node(c5, "ckpt",
                   ["Checkpoint",
                    "best-RMSE  ·  best-photo"],
                   fill=C["node_artifact"])

        c5.edge("adamw", "ema")
        c5.edge("adamw", "wandb")
        c5.edge("adamw", "ckpt")

    # =======================================================
    # Inter-cluster backbone edges
    # =======================================================
    # Phase 0 → Phase 1
    dot.edge("split", "rej_lowcov",
             lhead="cluster_phase1",
             color=C["edge"], penwidth="1.4",
             xlabel=" manifest build ")
    # Phase 0 stats → Phase 2 normaliser
    dot.edge("stats", "pp_norm",
             style="dashed", color=C["node_preproc"], penwidth="1.3",
             xlabel=" stats.json ", constraint="false")
    # Phase 1 → Phase 2
    dot.edge("accepted", "pp_raw",
             ltail="cluster_phase1", lhead="cluster_phase2",
             color=C["edge"], penwidth="1.6")
    # Phase 2 → Phase 3
    dot.edge("pp_norm", "vae_enc",
             color=C["edge"], penwidth="1.4",
             xlabel=" filled x ")
    # Phase 3 → Phase 4
    dot.edge("unet", "l_vel", color=C["node_loss_vel"], penwidth="1.4",
             xlabel=" vθ ")
    dot.edge("unet", "l_photo", color=C["node_loss_pix"], penwidth="1.3",
             xlabel=" ẑ_depth → VAE decode ")
    # Phase 4 → Phase 5
    dot.edge("l_total", "adamw", lhead="cluster_phase5",
             color=C["edge"], penwidth="1.6")

    out_base.parent.mkdir(parents=True, exist_ok=True)
    dot.render(str(out_base), format="pdf", cleanup=True)
    dot.render(str(out_base), format="svg", cleanup=True)


# ===========================================================================
# FIGURE 2 — Flow & Losses (zoom) with many image nodes
# ===========================================================================

def build_flow_and_losses(data: dict, paths: dict[str, Path],
                          out_base: Path) -> None:
    dot = graphviz.Digraph(name="MarsDepthFM_FlowLosses")
    _apply_base_style(dot, rankdir="TB")

    # -----------------------------------------------------
    # Inputs row
    # -----------------------------------------------------
    with dot.subgraph(name="cluster_inputs") as c:
        c.attr(label="Inputs",
               style="solid", color=C["cluster_line"],
               bgcolor=C["bg_infra"], fontname="Helvetica-Bold",
               fontsize="12", fontcolor=C["node_data"], labeljust="l")
        _image_node(c, "x_img", paths["accept_crater_ortho"],
                    title="x_img  (3×H×W)",
                    header_color=C["node_latent"],
                    captions=[("RED ortho, [-1, 1]", C["text"])])
        _image_node(c, "x_depth", paths["accept_crater_dtm"],
                    title="x_depth  (3×H×W)",
                    header_color=C["node_latent"],
                    captions=[("centered relative relief", C["text"])])


    # -----------------------------------------------------
    # VAE Encode
    # -----------------------------------------------------
    _text_node(dot, "vae",
               ["VAE encode  (frozen)",
                "z_img  =  E(x_img) · 0.18215",
                "z_depth  =  E(x_depth) · 0.18215"],
               fill=C["node_latent"])
    dot.edge("x_img", "vae")
    dot.edge("x_depth", "vae")

    # -----------------------------------------------------
    # Flow-matching sampler (t + interpolation)
    # -----------------------------------------------------
    with dot.subgraph(name="cluster_fm") as c:
        c.attr(label="Flow-matching sampler",
               style="solid", color=C["cluster_line"],
               bgcolor=C["bg_flow"], fontname="Helvetica-Bold",
               fontsize="12", fontcolor=C["node_unet"], labeljust="l")

        _text_node(c, "qsample",
                   ["q_sample (DDPM forward)",
                    "x_src = z_img  noised at t = 200",
                    "(skipped if noising_step = 0)"],
                   fill=C["node_latent"])
        _image_node(c, "t_hist", paths["logit_normal"],
                    title="Flow timestep  t",
                    header_color=C["node_unet"],
                    img_width="2.4",
                    captions=[("logit-normal(0, 1)  ·  8 000 draws",
                               C["text"])])
        _text_node(c, "zt",
                   ["Noisy interpolant",
                    "z_t = (1−t) x_src  +  t z_depth  +  σ_min ε",
                    "σ_min = 1e-4"],
                   fill=C["node_unet"])
        _image_node(c, "flow_strip", paths["flow_evolution"],
                    title="z_t  for t ∈ {0, ¼, ½, ¾, 1}",
                    header_color=C["node_unet"], img_width="3.8",
                    captions=[("real (1−t) x_src + t z_depth interpolation",
                               C["text"])])
        c.edge("qsample", "zt", xlabel=" x_src ")
        c.edge("t_hist", "zt", xlabel=" t ")
        c.edge("zt", "flow_strip", style="dashed",
               color=C["edge"], arrowhead="none")

    dot.edge("vae", "qsample", xlabel=" z_img ")
    dot.edge("vae", "zt", xlabel=" z_depth ", style="dashed")

    # -----------------------------------------------------
    # UNet forward
    # -----------------------------------------------------
    _text_node(dot, "unet",
               ["UNet  vθ(z_t, t ; z_img)",
                "input:  cat(z_t, z_img) ∈ ℝ^(8×64×64)",
                "output: vθ ∈ ℝ^(4×64×64)"],
               fill=C["node_unet"])
    dot.edge("zt", "unet")

    # -----------------------------------------------------
    # Latent branch  (velocity loss)
    # -----------------------------------------------------
    with dot.subgraph(name="cluster_latent") as c:
        c.attr(label="Latent branch",
               style="solid", color=C["cluster_line"],
               bgcolor="#fdeeed", fontname="Helvetica-Bold",
               fontsize="12", fontcolor=C["node_loss_vel"], labeljust="l")
        _image_node(c, "v_tgt", paths["v_tgt_norm"],
                    title="‖ v_tgt ‖",
                    header_color=C["node_loss_vel"],
                    captions=[("v_tgt = z_depth − x_src", C["text"])])
        _image_node(c, "v_pred", paths["v_pred_norm"],
                    title="‖ vθ ‖",
                    header_color=C["node_loss_vel"],
                    captions=[("predicted velocity", C["text"])])
        _text_node(c, "l_vel",
                   ["𝓛_vel",
                    "‖ vθ − v_tgt ‖²  ⊙  M_valid",
                    "weight:  w_v = 1.0"],
                   fill=C["node_loss_vel"])

        c.edge("v_tgt", "l_vel", style="dashed", color=C["node_loss_vel"])
        c.edge("v_pred", "l_vel", style="dashed", color=C["node_loss_vel"])

    dot.edge("unet", "v_pred", xlabel=" vθ ")

    # -----------------------------------------------------
    # Pixel branch (decode + 4 losses)
    # -----------------------------------------------------
    with dot.subgraph(name="cluster_pixel") as c:
        c.attr(label="Pixel branch",
               style="solid", color=C["cluster_line"],
               bgcolor=C["bg_loss"], fontname="Helvetica-Bold",
               fontsize="12", fontcolor=C["node_loss_pix"], labeljust="l")

        _text_node(c, "decode",
                   ["ẑ_depth = z_t + (1−t) vθ",
                    "x̂_depth = VAEdec(ẑ_depth / 0.18215)",
                    "(grad flows back through frozen decoder)"],
                   fill=C["node_loss_pix"])

        _image_node(c, "p_photo", paths["render_pred"],
                    title="𝓛_photo  —  Lunar-Lambert render",
                    header_color=C["node_loss_pix"],
                    captions=[
                        ("(1−α)(1 − Pearson) + α(1 − SSIM)", C["text"]),
                        ("L = σ(logit); r = (L·cos i + (1−L)·cos i / (cos i + cos e))·I + A",
                         C["text"]),
                        ("weight: w_p = 1.5", C["node_loss_pix"]),
                    ])
        _image_node(c, "p_grad", paths["grad_multiscale"],
                    title="𝓛_grad  —  multi-scale gradient",
                    header_color=C["node_loss_pix"], img_width="3.6",
                    captions=[
                        ("Σ_{s∈{1,2,4}}  ‖∇D^(s)_pred − ∇D^(s)_GT‖₁",
                         C["text"]),
                        ("weight: w_g = 100 (dominant supervisor)",
                         C["node_loss_pix"]),
                    ])
        _image_node(c, "p_norm", paths["normals_rgb"],
                    title="𝓛_norm  —  surface normals",
                    header_color=C["node_loss_pix"],
                    captions=[
                        ("1 − cos⟨ N_pred, N_GT ⟩", C["text"]),
                        ("N = normalize(−∂z/∂x, −∂z/∂y, 1)", C["text"]),
                        ("weight: w_n = 1.0", C["node_loss_pix"]),
                    ])
        _image_node(c, "p_ffl", paths["fft_mag"],
                    title="𝓛_FFL  —  focal-frequency",
                    header_color=C["node_loss_pix"],
                    captions=[
                        ("log |FFT2(D)|  (DC centred)", C["text"]),
                        ("GT inject at mask boundary suppresses",
                         C["text"]),
                        ("Gibbs ringing  ·  weight: w_f = 0", C["node_loss_pix"]),
                    ])
        # ---- NEW: Huber / Laplacian / Ordinal image nodes ----
        _image_node(c, "p_huber", paths["huber_err"],
                    title="𝓛_Huber  —  absolute depth",
                    header_color=C["node_loss_pix"],
                    captions=[
                        ("signed err  D_pred − D_GT  (red = over, blue = under)",
                         C["text"]),
                        (f"Huber_δ, δ = {data['huber_delta']:.2f}  "
                         f"→ L2 near 0, L1 in tails",
                         C["text"]),
                        ("weight: w_h = 3.0  ·  from step 0",
                         C["node_loss_pix"]),
                    ])
        _image_node(c, "p_lap", paths["laplacian_gt"],
                    title="𝓛_Lap  —  curvature (∇²d)",
                    header_color=C["node_loss_pix"],
                    captions=[
                        ("∇²D_GT  —  blue = concave, red = convex",
                         C["text"]),
                        ("‖ ∇²D_pred − ∇²D_GT ‖₁", C["text"]),
                        ("weight: w_ℓ = 0.05  ·  from step 0",
                         C["node_loss_pix"]),
                    ])
        _image_node(c, "p_ord", paths["ordinal_overlay"],
                    title="𝓛_Ord  —  relative ordering",
                    header_color=C["node_loss_pix"],
                    captions=[
                        (f"sampled pairs  (margin = {data['ord_margin']:.02f})",
                         C["text"]),
                        (f"violation rate = {100*data['ord_violation_rate']:.1f}%",
                         C["text"]),
                        ("weight: w_o = 1.0  ·  from step 0",
                         C["node_loss_pix"]),
                    ])

        c.edge("decode", "p_photo"); c.edge("decode", "p_grad")
        c.edge("decode", "p_norm");  c.edge("decode", "p_ffl")
        c.edge("decode", "p_huber"); c.edge("decode", "p_lap")
        c.edge("decode", "p_ord")

    dot.edge("unet", "decode", xlabel=" vθ + z_t ")

    # -----------------------------------------------------
    # Combined loss
    # -----------------------------------------------------
    dot.node("l_total",
             label="<<FONT FACE='Helvetica-Bold' POINT-SIZE='12' "
                   "COLOR='#ffffff'>𝓛_total</FONT>"
                   "<BR/><FONT FACE='Helvetica' POINT-SIZE='9' "
                   "COLOR='#ffffff'>w_v·𝓛_vel + w_p·𝓛_photo + "
                   "w_g·𝓛_grad + w_n·𝓛_norm + w_f·𝓛_FFL</FONT>"
                   "<BR/><FONT FACE='Helvetica' POINT-SIZE='9' "
                   "COLOR='#ffffff'>+ w_h·𝓛_Huber + w_ℓ·𝓛_Lap + "
                   "w_o·𝓛_Ord</FONT>"
                   "<BR/><FONT FACE='Helvetica' POINT-SIZE='8' "
                   "COLOR='#ffffff'>yaml: 1.0 / 1.5 / 100 / 1.0 / 0 / "
                   "3.0 / 0.05 / 1.0</FONT>>",
             shape="hexagon", style="filled",
             fillcolor=C["text"], color=C["text"],
             fontcolor="white", penwidth="1.6")

    dot.edge("l_vel", "l_total", color=C["node_loss_vel"], penwidth="1.4")
    dot.edge("p_photo", "l_total", color=C["node_loss_pix"], penwidth="1.3")
    dot.edge("p_grad", "l_total", color=C["node_loss_pix"], penwidth="1.3")
    dot.edge("p_norm", "l_total", color=C["node_loss_pix"], penwidth="1.3")
    dot.edge("p_ffl", "l_total", color=C["node_loss_pix"], penwidth="1.3")
    dot.edge("p_huber", "l_total", color=C["node_loss_pix"], penwidth="1.3")
    dot.edge("p_lap", "l_total", color=C["node_loss_pix"], penwidth="1.3")
    dot.edge("p_ord", "l_total", color=C["node_loss_pix"], penwidth="1.3")

    _text_node(dot, "adamw",
               ["AdamW  +  cosine LR",
                "bf16  ·  grad-accum 4  ·  ‖∇θ‖ ≤ 1",
                "EMA decay 0.99  ·  WandB log"],
               fill=C["node_optim"])
    dot.edge("l_total", "adamw")

    out_base.parent.mkdir(parents=True, exist_ok=True)
    dot.render(str(out_base), format="pdf", cleanup=True)
    dot.render(str(out_base), format="svg", cleanup=True)


# ===========================================================================
# FIGURE 3 — Manifest Filter Mosaic  (zoom on the four reject criteria)
# ===========================================================================

def build_manifest_filter(data: dict, paths: dict[str, Path],
                          out_base: Path) -> None:
    dot = graphviz.Digraph(name="MarsDepthFM_ManifestFilter")
    _apply_base_style(dot, rankdir="TB")
    dot.attr(ranksep="0.6", nodesep="0.35")

    # Header rule hexagon — placed at the top, then the reject row, then
    # the accept row, so the natural TB flow produces two compact rows of
    # image nodes.
    dot.node("rule",
             label=("<<FONT FACE='Helvetica-Bold' POINT-SIZE='12' "
                    "COLOR='#ffffff'>Manifest filter</FONT>"
                    "<BR/><FONT FACE='Helvetica' POINT-SIZE='10' "
                    "COLOR='#ffffff'>(valid_ratio ≥ 0.5)  ∧  "
                    "(residual ≥ 0.1)</FONT>"
                    "<BR/><FONT FACE='Helvetica' POINT-SIZE='10' "
                    "COLOR='#ffffff'>(is_tin ≤ 0.95)  ∧  "
                    "(num_merges = 1)</FONT>"
                    "<BR/><FONT FACE='Helvetica' POINT-SIZE='8' "
                    "COLOR='#ffffff'>DepthFMHiRISEAdapterCached."
                    "_init_manifest</FONT>>"),
             shape="hexagon", style="filled",
             fillcolor=C["node_filter"], color=C["node_filter"],
             penwidth="1.6")

    # ---- REJECTED row ---------------------------------------------------
    with dot.subgraph(name="cluster_rej") as cr:
        cr.attr(label="Rejected  (at least one criterion violated)",
                style="solid", color=C["node_filter"],
                bgcolor="#fdeeed", fontname="Helvetica-Bold",
                fontsize="12", fontcolor=C["node_filter"], labeljust="c")

        _image_node(
            cr, "r_lowcov", paths["reject_lowcov_ortho"],
            title="Low coverage",
            header_color=C["node_filter"], img_width="1.8",
            captions=[
                (f"valid_ratio = {data['low_cov_valid_ratio']:.3f}",
                 C["node_filter"]),
                ("threshold: ≥ 0.50", C["text"]),
                ("→ REJECT (valid_ratio)", C["node_filter"]),
            ],
        )
        _image_node(
            cr, "r_tin", paths["reject_tin_ortho"],
            title="TIN flat plateau",
            header_color=C["node_filter"], img_width="1.8",
            captions=[
                (f"is_tin = {data['tin_score']:.3f}", C["node_filter"]),
                ("threshold: ≤ 0.95", C["text"]),
                ("→ REJECT (is_tin)", C["node_filter"]),
            ],
        )
        _image_node(
            cr, "r_seam", paths["reject_seam_ortho"],
            title="Tile-merge seam",
            header_color=C["node_filter"], img_width="1.8",
            captions=[
                (f"seam score = {data['seam_score']:.2f}× bg",
                 C["node_filter"]),
                ("num_merges = 2", C["node_filter"]),
                ("→ REJECT (num_merges)", C["node_filter"]),
            ],
        )
        _image_node(
            cr, "r_flat", paths["reject_flat_ortho"],
            title="Near-flat slope",
            header_color=C["node_filter"], img_width="1.8",
            captions=[
                (f"residual = {data['flat_residual']:.3f}",
                 C["node_filter"]),
                ("threshold: ≥ 0.10", C["text"]),
                ("→ REJECT (residual)", C["node_filter"]),
            ],
        )

        # Force the four reject panels onto the same rank (horizontal row)
        cr.body.append(
            "\t{ rank=same; r_lowcov; r_tin; r_seam; r_flat; }\n"
        )

    # ---- ACCEPTED row ---------------------------------------------------
    with dot.subgraph(name="cluster_acc") as ca:
        ca.attr(label="Accepted  (all four criteria satisfied)",
                style="solid", color=C["node_accept"],
                bgcolor="#eaf7ee", fontname="Helvetica-Bold",
                fontsize="12", fontcolor=C["node_accept"], labeljust="c")

        c2 = _crater_patch(seed=77)
        c3 = _crater_patch(seed=103)

        def _acc_metrics(dtm_np: np.ndarray) -> dict:
            t = torch.from_numpy(dtm_np).unsqueeze(0).unsqueeze(0)
            m = torch.ones_like(t).bool()
            return dict(
                valid_ratio=1.0,
                residual=_residual(t.squeeze(), m.float().squeeze()),
                is_tin=_tin(t, m, kernel_size=32),
            )

        m1 = dict(valid_ratio=1.0,
                  residual=data["crater_residual"],
                  is_tin=data["crater_tin"])
        m2 = _acc_metrics(c2)
        m3 = _acc_metrics(c3)

        _image_node(
            ca, "a_crater1", paths["accept_crater_ortho"],
            title="Crater w/ ejecta",
            header_color=C["node_accept"], img_width="1.8",
            captions=[
                (f"valid_ratio = {m1['valid_ratio']:.3f}", C["node_accept"]),
                (f"residual = {m1['residual']:.3f}", C["node_accept"]),
                (f"is_tin = {m1['is_tin']:.3f}", C["node_accept"]),
                ("→ all four pass", C["node_accept"]),
            ],
        )
        _image_node(
            ca, "a_crater2", paths["accept_crater2_ortho"],
            title="Crater (variant)",
            header_color=C["node_accept"], img_width="1.8",
            captions=[
                (f"valid_ratio = {m2['valid_ratio']:.3f}", C["node_accept"]),
                (f"residual = {m2['residual']:.3f}", C["node_accept"]),
                (f"is_tin = {m2['is_tin']:.3f}", C["node_accept"]),
                ("→ all four pass", C["node_accept"]),
            ],
        )
        _image_node(
            ca, "a_crater3", paths["accept_crater3_ortho"],
            title="Crater field",
            header_color=C["node_accept"], img_width="1.8",
            captions=[
                (f"valid_ratio = {m3['valid_ratio']:.3f}", C["node_accept"]),
                (f"residual = {m3['residual']:.3f}", C["node_accept"]),
                (f"is_tin = {m3['is_tin']:.3f}", C["node_accept"]),
                ("→ all four pass", C["node_accept"]),
            ],
        )

        ca.body.append(
            "\t{ rank=same; a_crater1; a_crater2; a_crater3; }\n"
        )

    # Edges from rule to each rejection, then to each acceptance
    for n in ("r_lowcov", "r_tin", "r_seam", "r_flat"):
        dot.edge("rule", n, color=C["node_filter"], penwidth="1.3")
    for n in ("a_crater1", "a_crater2", "a_crater3"):
        dot.edge("rule", n, color=C["node_accept"], penwidth="1.3")

    # Output destination node
    _text_node(dot, "parquet",
               ["hirise_manifest_<hash>.parquet",
                "persisted to .cache/manifests/",
                "O(1) lookup  ·  rebuilds on config change"],
               fill=C["node_artifact"])
    dot.edge("a_crater1", "parquet", style="dashed",
             color=C["node_accept"], arrowhead="none")
    dot.edge("a_crater2", "parquet", style="dashed",
             color=C["node_accept"], arrowhead="none")
    dot.edge("a_crater3", "parquet", style="dashed",
             color=C["node_accept"], arrowhead="none")

    out_base.parent.mkdir(parents=True, exist_ok=True)
    dot.render(str(out_base), format="pdf", cleanup=True)
    dot.render(str(out_base), format="svg", cleanup=True)


# ===========================================================================
# MAIN
# ===========================================================================

def main(out_dir: str | Path = "outputs/figures") -> None:
    out_dir = Path(out_dir)
    panel_dir = out_dir / "panels"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Adapter source: "
          f"{'REAL depth_fm.depthfm_adapter' if _USING_REAL_ADAPTER else 'bundled fallback (bit-identical)'}")

    print("Step 1/3 — running real adapter algorithms on synthetic terrain")
    data = _collect_real_data(H=128)
    print(f"  tin_score (flat plateau) = {data['tin_score']:.4f}  [reject if > 0.95]")
    print(f"  tin_score (crater)       = {data['crater_tin']:.4f}")
    print(f"  seam_score (tile merge)  = {data['seam_score']:.3f}")
    print(f"  residual (crater)        = {data['crater_residual']:.4f}")
    print(f"  residual (flat slope)    = {data['flat_residual']:.4f}  [reject if < 0.10]")
    print(f"  sun OLS ŝ                = "
          f"({data['sun_vec'][0]:.3f}, {data['sun_vec'][1]:.3f}, {data['sun_vec'][2]:.3f})")

    print("\nStep 2/3 — rendering PNG panels")
    paths = _save_panels(data, panel_dir)
    print(f"  wrote {len(paths)} panels to {panel_dir}")

    print("\nStep 3/3 — building graphviz diagrams (PDF + SVG)")
    print("  pipeline_overview")
    build_pipeline_overview(data, paths, out_dir / "pipeline_overview")
    print("  flow_and_losses")
    build_flow_and_losses(data, paths, out_dir / "flow_and_losses")
    print("  manifest_filter")
    build_manifest_filter(data, paths, out_dir / "manifest_filter")

    print("\nOutput files:")
    for name in ("pipeline_overview", "flow_and_losses", "manifest_filter"):
        for ext in ("pdf", "svg"):
            p = out_dir / f"{name}.{ext}"
            if p.exists():
                print(f"  {p}  ({p.stat().st_size / 1024:.1f} KB)")

    print("\nDone.")


if __name__ == "__main__":
    main()