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

import json
import logging
import math
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Hardcoded fallback quantiles derived from dataset_stats/dtm/dataset_stats.json
# (Olympus Mons region, 52 k patches)
_DEFAULT_ELEV_P02 = -4396.5664071121255
_DEFAULT_ELEV_P98 = 20757.65899590482
_DEFAULT_IMG_P02 = 0.049661101862306406  # average of left_red / right_red p02
_DEFAULT_IMG_P98 = 0.24805250879347127  # average of left_red / right_red p98
# Scale factor for relative-topography mode: 98th-percentile of patch-centred
# elevation distribution (metres).  98 % of patches stay within [-1, 1] before
# clamping while physical slope magnitudes remain consistent across the dataset.
_DEFAULT_ELEV_SCALE = 26.74  # centered_p98[elevation] from Olympus stats


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


def is_tin_artifact(elevation: torch.Tensor, valid_mask: torch.Tensor, threshold: float = 0.15) -> bool:
    """
    Detects artificial TIN (Triangular Irregular Network) interpolation in DTMs.
    TINs have perfectly planar facets, meaning their second derivative (Laplacian) is exactly 0.
    """
    # 1. Prevent NaN poisoning in the convolution
    safe_elev = elevation.clone()
    safe_elev[~valid_mask] = 0.0

    elev_4d = safe_elev.view(1, 1, safe_elev.shape[-2], safe_elev.shape[-1])

    # 3x3 Laplacian kernel
    kernel = torch.tensor([[[[0.0, 1.0, 0.0],
                             [1.0, -4.0, 1.0],
                             [0.0, 1.0, 0.0]]]], device=elevation.device)

    # Calculate 2nd derivative
    laplacian = torch.nn.functional.conv2d(elev_4d, kernel, padding=1)

    # 2. Extract only valid pixels
    # Note: Boundary pixels where NaNs were zeroed will have massive Laplacian values.
    # This is fine, as they will safely fail the < 1e-2 check and not inflate our TIN count.
    valid_laplacian = laplacian.view(-1)[valid_mask.view(-1).bool()]

    if len(valid_laplacian) == 0:
        return True

    # 3. Evaluate curvature
    # 1e-2 accounts for float32 stepping limits at high Martian altitudes (e.g. 20,000m)
    zero_curvature_ratio = (valid_laplacian.abs() < 1e-2).float().mean().item()

    return zero_curvature_ratio > threshold


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


def _normalize_ortho(
        ortho: torch.Tensor,
        p02: float,
        p98: float,
) -> torch.Tensor:
    """Normalise an orthoimage using global dataset quantiles to [-1, 1].

    Applies the same linear formula as the paper:
        ĩ = ((i − p02) / (p98 − p02) − 0.5) × 2

    Args:
        ortho: (C, H, W) in I/F reflectance [0, 1].
        p02: Dataset-level 2nd-percentile reflectance.
        p98: Dataset-level 98th-percentile reflectance.

    Returns:
        (C, H, W) in [-1, 1], clamped.
    """
    # range_ = p98 - p02
    # normed = ((ortho - p02) / range_ - 0.5) * 2.0
    # return torch.clamp(normed, -1.0, 1.0)

    # Instead of using self.img_p02 and self.img_p98 from the global JSON
    valid_pixels = ortho[ortho > 0.0]  # Ignore pure black nodata
    if len(valid_pixels) > 0:
        local_p02 = torch.quantile(valid_pixels, 0.02)
        local_p98 = torch.quantile(valid_pixels, 0.98)

        # Avoid divide-by-zero if the patch is perfectly uniform
        if local_p98 > local_p02:
            ortho = ((ortho - local_p02) / (local_p98 - local_p02) - 0.5) * 2.0
        else:
            ortho = torch.zeros_like(ortho)  # Fallback

    return torch.clamp(ortho, -1.0, 1.0)


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


class DepthFMHiRISEAdapter(Dataset):
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
            base_dataset,
            sampler,
            resolution: int = 512,
            dtm_normalization: Literal["relative", "log", "linear"] = "relative",
            random_flip: bool = True,
            random_jitter: bool = False,
            brightness_jitter: float = 0.1,
            stats_path: str | None = None,
            max_retries: int = 10
    ):
        super().__init__()
        self.max_retries = max_retries
        self.random_jitter = random_jitter
        self.base = base_dataset
        self.sampler = sampler
        self.resolution = resolution
        self.dtm_norm = dtm_normalization
        self.flip = random_flip
        self.bright_jitter = brightness_jitter

        self.is_train = random_flip

        # Load global quantiles for normalization
        (
            self.elev_p02, self.elev_p98,
            self.img_p02, self.img_p98,
            self.elev_scale,
        ) = _load_quantiles(stats_path)
        logger.info(
            "DepthFMHiRISEAdapter: elev scale=%.2f m, img p02=%.4f p98=%.4f (norm=%s)",
            self.elev_scale, self.img_p02, self.img_p98, dtm_normalization,
        )

        # Pre-materialise sampler indices for random access
        self._indices = list(sampler)
        logger.info(
            "DepthFMHiRISEAdapter: %d samples, resolution=%d, norm=%s",
            len(self._indices), resolution, dtm_normalization,
        )

    def __len__(self) -> int:
        return len(self._indices)

    def rand_idx(self) -> torch.int64:
        return torch.randint(0, len(self._indices), (1,)).item()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        for retries in range(getattr(self, 'max_retries', 10)):
            geo_slice = self._indices[idx]

            # Load from the base MarsHiRISEDTM dataset
            try:
                # Load from the base MarsHiRISEDTM dataset
                sample = self.base[geo_slice]
            except IndexError as e:
                idx = self.rand_idx()
                logger.exception("An error occurred while trying to retrieve a sample.")
                continue

            # ── 1. Load Elevation ──
            elevation = sample["elevation"]  # (1, H, W)
            if elevation.ndim == 4:
                elevation = elevation[0]

            if elevation.shape[-1] == 0 or elevation.shape[-2] == 0:
                idx = self.rand_idx()
                continue

            # ── 2. Select Orthoimage FIRST ──
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
                key = None

            if key is not None:
                ortho = sample[key]
            else:
                # Fallback: try IRB
                for fallback_key in ("left_irb", "right_irb"):
                    if fallback_key in sample and sample[fallback_key] is not None:
                        ortho = sample[fallback_key]
                        key = fallback_key
                        break
                else:
                    logger.warning(f"No orthoimage found for sample {idx}, using zeros")
                    ortho = torch.zeros(1, elevation.shape[-2], elevation.shape[-1])
                    key = "dummy"

            if ortho.ndim == 4:
                ortho = ortho[0]

            if ortho.shape[-1] == 0 or ortho.shape[-2] == 0:
                idx = self.rand_idx()
                continue

            # ── 3. Extract MATCHING Metadata ──
            meta_list = sample.get("meta", [])
            meta_key = f"{key}_meta"

            incidence, azimuth = 45.0, 270.0  # Safe defaults
            if meta_list and meta_key in meta_list[0]:
                incidence = meta_list[0][meta_key]["incidence_angle"]
                azimuth = meta_list[0][meta_key]["solar_azimuth"]

            # Convert spherical to cartesian vector
            inc_rad = math.radians(incidence)
            az_rad = math.radians(azimuth)
            sun_x = math.sin(inc_rad) * math.cos(az_rad)
            sun_y = math.sin(inc_rad) * math.sin(az_rad)
            sun_z = math.cos(inc_rad)

            sun_vector = torch.tensor([-sun_x, -sun_y, sun_z], dtype=torch.float32)

            # ── 4. Resize and Mask ──
            dtm_resized = _safe_resize(elevation, self.resolution, has_nans=True)
            image_resized = _safe_resize(ortho, self.resolution)

            ortho_valid = (image_resized != 0.0).any(dim=0, keepdim=True)
            elev_valid = torch.isfinite(dtm_resized) & (dtm_resized != 0.0)
            valid_mask_resized = (ortho_valid & elev_valid).float()

            # ── 5. Filtering Logic ──
            if True or getattr(self, 'is_train', False):
                valid_ratio = valid_mask_resized.mean().item()

                if valid_ratio >= 0.85:
                    try:
                        std = compute_topographic_residual(dtm_resized, valid_mask_resized)
                        if std < 0.1:
                            logger.info(f"Skipping flat patch, std: {std:.3f}")
                            idx = self.rand_idx()
                            continue

                        elev_valid_native = torch.isfinite(elevation) & (elevation != 0.0)

                        # from depth_fm.depthfm_adapter import is_tin_artifact
                        if is_tin_artifact(elevation, elev_valid_native, threshold=0.15):
                            print("Skipping")
                            logger.info("Skipping patch: Detected artificial TIN triangles.")
                            idx = self.rand_idx()
                            continue
                    except Exception:
                        logger.exception("An error occured while validating.")
                        idx = self.rand_idx()
                        continue

                    break  # Passed all checks
                else:
                    idx = self.rand_idx()
                    continue
            else:
                break  # Validation/Test always breaks immediately

        else:
            # ── Fallback if max_retries hit ──
            logger.warning(f"Hit max_retries in DataLoader. Returning dummy patch.")
            return {
                "image": torch.zeros(3, self.resolution, self.resolution),
                "dtm": torch.zeros(3, self.resolution, self.resolution),
                "confidence": torch.zeros(1, self.resolution, self.resolution),
                "sun_vector": torch.tensor([0.5, -0.5, 1.0], dtype=torch.float32)
            }

        # ── 6. Normalization ──
        dtm = _normalize_dtm_relative(dtm_resized, valid_mask_resized, scale_factor=self.elev_scale)
        dtm = _to_3ch(dtm)

        image = _normalize_ortho(image_resized, p02=self.img_p02, p98=self.img_p98)
        image = _to_3ch(image)

        # ── 7. Synchronised Augmentation (INCLUDING SUN VECTOR) ──
        if getattr(self, 'is_train', False) and getattr(self, 'flip', False):
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, [-1])
                dtm = torch.flip(dtm, [-1])
                valid_mask_resized = torch.flip(valid_mask_resized, [-1])
                sun_vector[0] = -sun_vector[0]  # Flip X axis

            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, [-2])
                dtm = torch.flip(dtm, [-2])
                valid_mask_resized = torch.flip(valid_mask_resized, [-2])
                sun_vector[1] = -sun_vector[1]  # Flip Y axis

            # 90-degree rotations
            k_rot = torch.randint(0, 4, (1,)).item()
            if k_rot > 0:
                image = torch.rot90(image, k=k_rot, dims=[-2, -1])
                dtm = torch.rot90(dtm, k=k_rot, dims=[-2, -1])
                valid_mask_resized = torch.rot90(valid_mask_resized, k=k_rot, dims=[-2, -1])

                # Rotate sun vector in X-Y plane
                sx, sy = sun_vector[0].clone(), sun_vector[1].clone()
                if k_rot == 1:
                    sun_vector[0], sun_vector[1] = -sy, sx
                elif k_rot == 2:
                    sun_vector[0], sun_vector[1] = -sx, -sy
                elif k_rot == 3:
                    sun_vector[0], sun_vector[1] = sy, -sx

        # Brightness jitter on image only
        if getattr(self, 'is_train', False) and getattr(self, 'bright_jitter', 0) > 0:
            import random
            factor = 1.0 + random.uniform(-self.bright_jitter, self.bright_jitter)
            image = (image * factor).clamp(-1.0, 1.0)

        return {
            "image": image,
            "dtm": dtm,
            "confidence": valid_mask_resized,
            "sun_vector": sun_vector
        }
