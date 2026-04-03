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
        dtm_normalization="minmax",
    )

    # Standard PyTorch DataLoader
    loader = DataLoader(adapter, batch_size=2, num_workers=32, shuffle=True)
"""

from __future__ import annotations

import logging
import random
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _normalize_dtm(
        elevation: torch.Tensor,
        method: Literal["minmax", "log"] = "minmax",
) -> torch.Tensor:
    """Normalise a (1, H, W) elevation tensor to [-1, 1].

    Args:
        elevation: Raw elevation in metres; NaN = nodata.
        method: ``"minmax"`` (per-tile linear) or ``"log"`` (log-transform).

    Returns:
        (1, H, W) tensor in [-1, 1] with NaN filled to 0.
    """
    valid = torch.isfinite(elevation)
    if not valid.any():
        return torch.zeros_like(elevation)

    vals = elevation[valid]

    if method == "log":
        shift = vals.min()
        elev_shifted = torch.where(valid, elevation - shift + 1.0, torch.ones_like(elevation))
        elev_log = torch.log(elev_shifted)
        lmin = elev_log[valid].min()
        lmax = elev_log[valid].max()
        if lmax - lmin < 1e-8:
            return torch.zeros_like(elevation)
        normed = 2.0 * (elev_log - lmin) / (lmax - lmin) - 1.0
    else:  # minmax
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin < 1e-8:
            return torch.zeros_like(elevation)
        normed = 2.0 * (elevation - vmin) / (vmax - vmin) - 1.0

    normed = torch.where(valid, normed, torch.zeros_like(normed))
    return normed


def _normalize_ortho(ortho: torch.Tensor) -> torch.Tensor:
    """Normalise an orthoimage from I/F [0, 1] to [-1, 1].

    Args:
        ortho: (C, H, W) in [0, 1] (I/F reflectance).

    Returns:
        (C, H, W) in [-1, 1].
    """
    return ortho * 2.0 - 1.0


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
        dtm_normalization: ``"minmax"`` or ``"log"``.
        random_flip: Apply random horizontal/vertical flips.
        brightness_jitter: Max relative brightness perturbation on image only.
    """

    def __init__(
            self,
            base_dataset,
            sampler,
            resolution: int = 512,
            dtm_normalization: Literal["minmax", "log"] = "minmax",
            random_flip: bool = True,
            brightness_jitter: float = 0.1,
    ):
        super().__init__()
        self.base = base_dataset
        self.sampler = sampler
        self.resolution = resolution
        self.dtm_norm = dtm_normalization
        self.flip = random_flip
        self.bright_jitter = brightness_jitter

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

        dtm = _normalize_dtm(elevation, method=self.dtm_norm)
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

        image = _normalize_ortho(ortho)
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
        }
