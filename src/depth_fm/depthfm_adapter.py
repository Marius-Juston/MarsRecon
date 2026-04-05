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
import random
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Hardcoded fallback quantiles derived from dataset_stats/dtm/dataset_stats.json
# (Olympus Mons region, 52 k patches)
_DEFAULT_ELEV_P02 = -4396.57
_DEFAULT_ELEV_P98 = 20757.66
_DEFAULT_IMG_P02 = 0.0502  # average of left_red / right_red p02
_DEFAULT_IMG_P98 = 0.2450  # average of left_red / right_red p98
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


def _normalize_dtm_relative(
        elevation: torch.Tensor,
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
    valid = torch.isfinite(elevation)
    if not valid.any():
        return torch.zeros_like(elevation)

    # 1. Remove absolute altitude — the network cannot infer this from texture
    patch_mean = elevation[valid].mean()
    centered = elevation - patch_mean

    # 2. Fixed physical scale: a 10 m ridge always produces the same latent delta
    normed = centered / scale_factor

    # 3. Clamp extreme outliers (craters, scarps) without distorting the core
    normed = torch.clamp(normed, -1.0, 1.0)
    normed = torch.where(valid, normed, torch.zeros_like(normed))
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
    range_ = p98 - p02
    normed = ((ortho - p02) / range_ - 0.5) * 2.0
    return torch.clamp(normed, -1.0, 1.0)


def _to_3ch(tensor: torch.Tensor) -> torch.Tensor:
    """Replicate a (1, H, W) tensor to (3, H, W) for VAE compatibility."""
    if tensor.shape[0] == 1:
        return tensor.expand(3, -1, -1).contiguous()
    return tensor[:3]


def _resize(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Resize (C, H, W) to (C, size, size) with bilinear interpolation."""
    if tensor.shape[-2] == size and tensor.shape[-1] == size:
        return tensor
    return F.interpolate(
        tensor.unsqueeze(0), size=(size, size),
        mode="bilinear", align_corners=False,
    ).squeeze(0)


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
            brightness_jitter: float = 0.1,
            stats_path: str | None = None,
    ):
        super().__init__()
        self.base = base_dataset
        self.sampler = sampler
        self.resolution = resolution
        self.dtm_norm = dtm_normalization
        self.flip = random_flip
        self.bright_jitter = brightness_jitter

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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        geo_slice = self._indices[idx]

        # Load from the base MarsHiRISEDTM dataset
        sample = self.base[geo_slice]

        # ── Elevation → normalised 3-channel DTM ──
        elevation = sample["elevation"]  # (1, H, W) float32, NaN = nodata
        if elevation.ndim == 4:
            elevation = elevation[0]  # remove batch dim from DataLoader

        valid_mask = torch.isfinite(elevation).float()


        dtm = _normalize_dtm_relative(elevation, scale_factor=self.elev_scale)
        dtm = _to_3ch(dtm)
        dtm = _resize(dtm, self.resolution)

        # ── Orthoimage → normalised 3-channel image ──
        # Stereo augmentation: randomly pick left or right
        left_key, right_key = "left_red", "right_red"
        has_left = left_key in sample and sample[left_key] is not None
        has_right = right_key in sample and sample[right_key] is not None

        if has_left and has_right:
            # Random stereo augmentation
            key = random.choice([left_key, right_key])
            ortho = sample[key]
        elif has_left:
            ortho = sample[left_key]
        elif has_right:
            ortho = sample[right_key]
        else:
            # Fallback: try IRB
            for fallback_key in ("left_irb", "right_irb"):
                if fallback_key in sample and sample[fallback_key] is not None:
                    ortho = sample[fallback_key]
                    break
            else:
                # No ortho available — create a dummy (training will skip)
                logger.warning(
                    "No orthoimage found for sample %d, using zeros", idx
                )
                ortho = torch.zeros(1, elevation.shape[-2], elevation.shape[-1])

        if ortho.ndim == 4:
            ortho = ortho[0]

        image = _normalize_ortho(ortho, p02=self.img_p02, p98=self.img_p98)
        image = _to_3ch(image)
        image = _resize(image, self.resolution)

        # ── Synchronised augmentation ──
        if self.flip:
            if random.random() > 0.5:
                image = torch.flip(image, [-1])  # horizontal
                dtm = torch.flip(dtm, [-1])
            if random.random() > 0.5:
                image = torch.flip(image, [-2])  # vertical
                dtm = torch.flip(dtm, [-2])

        # Brightness jitter on image only (simulates illumination variation)
        if self.bright_jitter > 0:
            factor = 1.0 + random.uniform(-self.bright_jitter, self.bright_jitter)
            image = (image * factor).clamp(-1.0, 1.0)

        return {
            "image": image,  # (3, H, W) in [-1, 1]
            "dtm": dtm,  # (3, H, W) in [-1, 1]
            "confidence": valid_mask,
        }
