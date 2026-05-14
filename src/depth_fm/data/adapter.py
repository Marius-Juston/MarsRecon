"""DepthFM dataset adapter for MarsHiRISEDTM.

Map-style wrapper around `MarsHiRISEDTM` + `HiRISEGeoSampler` that produces
DepthFM-ready training pairs::

    image:  (3, H, W) float32 in [-1, 1]  — orthoimage (1ch → 3ch replicated)
    dtm:    (3, H, W) float32 in [-1, 1]  — normalised elevation (1ch → 3ch)

Stereo augmentation: each `__getitem__` randomly selects the left or right
orthoimage as the conditioning input. Both pair with the same DTM, doubling
the effective training data.

The heavy image-processing primitives (void filling, seam detection, sun
vector estimation, TIN-artifact detection, topographic residual) used to live
inline in this file. They are now in
`depth_fm.data.image_processing.*` and re-exported below for the existing
callers (`build_litdata.py`, `train_lightning.py`, `debug_viz.py`,
`depthfm_pipeline_diagram.py`).

Usage::

    from dataset.core.dtm import MarsHiRISEDTM
    from dataset.sampling.sampler import HiRISEGeoSampler
    from depth_fm.data.adapter import DepthFMHiRISEAdapterCached

    base = MarsHiRISEDTM(root="/scratch/mars_hirise_dtm", include_ortho=True,
                         ortho_type="RED", download=True)
    sampler = HiRISEGeoSampler(base, size=0.009, length=10000)
    adapter = DepthFMHiRISEAdapterCached(
        base_dataset=base, sampler=sampler, resolution=512,
        dtm_normalization="relative",
        stats_path="dataset_stats/dtm/dataset_stats.json",
    )
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset.core.dtm import MarsHiRISEDTM
from dataset.sampling.sampler import HiRISEGeoSampler
from depth_fm.data.scalers import (
    DEFAULT_ELEV_REF_SCALE as _DEFAULT_ELEV_SCALE,
    GlobalLogNormalizer,
    LocalStripOrthoNormalizer,
    TrainingNormResult,
)

from depth_fm.data.image_processing.mask_ops import erode_valid_mask
from depth_fm.data.image_processing.void_filling import (
    fill_dtm_smart_diffusion,
    fill_invalid_nearest_neighbor,
    fill_voids_gmrf,
    fill_voids_kriging,
)
from depth_fm.data.image_processing.seam_detection import (
    SeamResult,
    compute_artifact_multipliers,
    compute_piecewise_linearity,
    compute_spatial_isolation,
    detect_seam_artifact,
    is_tin_artifact,
)
from depth_fm.data.image_processing.sun_vector import (
    estimate_sun_vector_irls,
    estimate_sun_vector_ols,
)
from depth_fm.data.image_processing.terrain import compute_topographic_residual

logger = logging.getLogger(__name__)


# Hardcoded fallback quantiles derived from dataset_stats/dtm/dataset_stats.json
# (Olympus Mons region, 52k patches).
_DEFAULT_ELEV_P02 = -4396.5664071121255
_DEFAULT_ELEV_P98 = 20757.65899590482
_DEFAULT_IMG_P02 = 0.06526107076433259  # average of left_red / right_red p02
_DEFAULT_IMG_P98 = 0.1828855234319496   # average of left_red / right_red p98


# ---------------------------------------------------------------------------
# Stats / normalization helpers used by the adapter class
# ---------------------------------------------------------------------------

def _load_quantiles(
        stats_path: str | None,
) -> tuple[float, float, float, float, float]:
    """Load elevation and image p02/p98 plus elevation scale from dataset_stats JSON.

    Returns (elev_p02, elev_p98, img_p02, img_p98, elev_scale). Falls back to
    hardcoded Olympus-region defaults if the path is missing.
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

    channels = stats["channels"]
    p02 = stats["p02"]
    p98 = stats["p98"]
    centered_p98 = stats["centered_p98"]

    elev_idx = channels.index("elevation")
    left_idx = channels.index("left_red") if "left_red" in channels else None
    right_idx = channels.index("right_red") if "right_red" in channels else None

    elev_p02 = p02[elev_idx]
    elev_p98 = p98[elev_idx]
    # Symmetric scale: use centered_p98 so 98% of patch relief lands in [-1, 1].
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

    if relief < 1e-4:
        return torch.zeros_like(elevation)

    normed = ((elevation - patch_min) / relief) * 2.0 - 1.0
    return torch.where(valid, normed, torch.zeros_like(normed))


def _normalize_dtm_relative(
        elevation: torch.Tensor,
        valid: torch.Tensor,
        scale_factor: float,
) -> torch.Tensor:
    """Normalise elevation via local centering + dynamic local scale.

    Produces "relative topography": the absolute Martian altitude (datum) is
    subtracted per-patch (unlearnable from orthorectified overhead imagery)
    while a physical scale maps consistent slope magnitudes to the same latent
    values across the dataset.

    Args:
        elevation: Raw elevation in metres; NaN = nodata.  Shape (1, H, W).
        scale_factor: Half the expected relief range in metres; used as the
            fallback scale on perfectly-flat patches.

    Returns:
        (1, H, W) tensor in [-1, 1]; nodata pixels filled with 0.
    """
    bool_valid = valid == 1
    if not bool_valid.any():
        return torch.zeros_like(elevation)

    patch_mean = elevation[bool_valid].mean()
    centered = elevation - patch_mean

    local_max = centered[bool_valid].abs().max()
    if local_max < 1e-4:
        local_scale = scale_factor
    else:
        local_scale = local_max

    normed = centered / local_scale
    normed = torch.clamp(normed, -1.0, 1.0)
    return torch.where(bool_valid, normed, torch.zeros_like(normed))


def _safe_resize(
        tensor: torch.Tensor,
        size: int,
        is_mask: bool = False,
        has_nans: bool = False,
) -> torch.Tensor:
    """Resize tensors safely using PyTorch, avoiding NaN poisoning."""
    if tensor.shape[-2] == size and tensor.shape[-1] == size:
        return tensor

    # Masks must use nearest neighbor so edges aren't blurred.
    if is_mask:
        return F.interpolate(tensor.unsqueeze(0), size=(size, size), mode="nearest-exact").squeeze(0)

    # DTMs with NaNs: swap NaNs for 0 to prevent bilinear poisoning, then restore.
    if has_nans:
        valid = ~torch.isnan(tensor)
        safe_tensor = tensor.clone()
        safe_tensor[~valid] = 0.0

        resized_tensor = F.interpolate(
            safe_tensor.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False,
        ).squeeze(0)
        resized_valid = F.interpolate(
            valid.float().unsqueeze(0), size=(size, size), mode="nearest-exact",
        ).squeeze(0).bool()

        resized_tensor[~resized_valid] = float("nan")
        return resized_tensor

    return F.interpolate(
        tensor.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False,
    ).squeeze(0)


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

    return F.interpolate(tensor.unsqueeze(0), size=(size, size), **kwargs).squeeze(0)


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------

class DepthFMHiRISEAdapterCached(Dataset):
    """Adapter that converts MarsHiRISEDTM samples into DepthFM training pairs.

    Each ``__getitem__`` draws a geo-slice from the sampler, loads elevation +
    orthoimage(s) via the base dataset, runs the manifest-cached preprocessing
    (sun vector, valid mask, void fill), and returns a normalised dict ready
    for the DepthFM training loop.

    Args:
        base_dataset: An initialised ``MarsHiRISEDTM`` instance.
        sampler: A TorchGeo geo-sampler that yields GeoSlice indices.
        resolution: Output spatial resolution in pixels (square crop).
        dtm_normalization: ``"relative"`` (default) — local centering + fixed
            physical scale derived from ``centered_p98`` in the stats file.
            ``"log"`` / ``"linear"`` — global quantile modes (kept for reference).
        random_flip: Apply random horizontal/vertical flips + 90° rotations.
        brightness_jitter: Max relative brightness perturbation on image only.
        stats_path: Path to ``dataset_stats.json`` for normalization quantiles.
            Falls back to hardcoded Olympus-region defaults if ``None``.
        use_manifest: If True, build / load a parquet manifest of clean patches
            so ``__getitem__`` is O(1) and never retries.
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
            manifest_workers: int = 16,
            manifest_dir: str | None = None,
            erode_radius: int = 2,
            multiprocessing_context="fork",
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
        from cache import manifest_cache_dir
        self.manifest_dir = (
            Path(manifest_dir) if manifest_dir is not None
            else manifest_cache_dir(self.base.root)
        )
        self.clip = clip

        # Load global quantiles
        (
            self.elev_p02, self.elev_p98,
            self.img_p02, self.img_p98,
            self.elev_scale,
        ) = _load_quantiles(stats_path)

        # TODO: per-strip ortho normalization would be more correct than
        # per-patch, but the infrastructure doesn't yet thread strip identity
        # through the sampler.
        self.ortho_normalizer = LocalStripOrthoNormalizer(self.img_p02, self.img_p98)
        self.evel_normalizer = GlobalLogNormalizer(self.elev_scale, clip=clip)

        self._raw_indices = list(sampler)

        if self.use_manifest:
            self._init_manifest()
        else:
            self.clean_records = None
            logger.warning("Manifest disabled. Training will be slow due to on-the-fly validation.")

    def _get_manifest_hash(self) -> str:
        """Unique cache key so the manifest rebuilds if dataset/sampler params change."""
        from cache import compute_hash
        key_parts = {
            "root": str(getattr(self.base, "root", "unknown")),
            "resolution": self.resolution,
            "sampler_length": len(self._raw_indices),
            "split": getattr(self.sampler, "split", "unknown"),
            "seed": getattr(self.sampler, "seed", 0),
            "dataset_hash": str(self.base.spatial_index_cache),
            "sampler_hash": str(self.sampler.cache_hash),
            "clip": self.clip,
        }
        return compute_hash(key_parts)

    def _init_manifest(self):
        """Load the parquet manifest or trigger a fast multiprocessing build."""
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.manifest_dir / f"hirise_manifest_{self._get_manifest_hash()}.parquet"

        if manifest_path.exists():
            logger.info(f"Loading rich manifest from {manifest_path}")
            df = pd.read_parquet(manifest_path)
        else:
            logger.info(f"Manifest not found. Building cache using {self.manifest_workers} workers...")
            df = self._build_manifest_parallel(manifest_path)

        # Thresholds can be tuned without rebuilding the cache.
        clean_df = df[
            (df["is_valid_data"] == True) &
            (df["valid_ratio"] >= 0.5) &
            (df["residual"] >= 0.1) &
            # High TIN values mean Laplacian ≈ 0 — perfectly-flat planes.
            (df["is_tin"] <= 0.95) &
            # Currently tile merges have issues for DTM estimation.
            (df["num_merges"] == 1)
        ]

        self.clean_records = clean_df.to_dict("records")
        logger.info(
            f"Manifest ready: Filtered {len(df)} total patches down to {len(self.clean_records)} clean pairs. "
            f"Saving manifest to {manifest_path}"
        )

    def _build_manifest_parallel(self, save_path: Path) -> pd.DataFrame:
        """Use a temporary PyTorch DataLoader to build the cache at max speed."""

        class _ManifestBuilderDS(Dataset):
            def __init__(self, adapter):
                self.adapter = adapter

            def __len__(self):
                return len(self.adapter._raw_indices)

            def __getitem__(self, idx):
                return self.adapter._evaluate_patch_for_manifest(idx)

        builder_loader = DataLoader(
            _ManifestBuilderDS(self),
            batch_size=1,
            num_workers=self.manifest_workers,
            collate_fn=lambda x: x[0],
            shuffle=False,
            multiprocessing_context=self.multiprocessing_context,
        )

        records = []
        for record in tqdm(builder_loader, desc="Scanning HiRISE Data", unit="patch"):
            records.append(record)

        df = pd.DataFrame(records)
        df.to_parquet(save_path)
        from cache import write_manifest
        write_manifest(save_path, cache_hash=self._get_manifest_hash(), config_snapshot={
            "root": str(getattr(self.base, "root", "unknown")),
            "resolution": self.resolution,
            "split": getattr(self.sampler, "split", "unknown"),
            "clip": self.clip,
        })
        return df

    def _evaluate_patch_for_manifest(self, idx: int) -> dict:
        """Heavy lifting: load data, calculate stats, return a manifest row.

        Only runs once during cache generation.
        """
        try:
            geo_slice = self._raw_indices[idx]
            sample = self.base[geo_slice]

            elevation = sample["elevation"]
            if elevation.ndim == 4:
                elevation = elevation[0]
            if elevation.shape[-1] == 0 or elevation.shape[-2] == 0:
                return {"idx": idx, "is_valid_data": False}

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
            if ortho.ndim == 4:
                ortho = ortho[0]

            # Masking + resizing
            dtm_resized = _safe_resize(elevation, self.resolution, has_nans=True)
            image_resized = _safe_resize(ortho, self.resolution)

            ortho_valid = (image_resized != 0.0).any(dim=0, keepdim=True)
            elev_valid = torch.isfinite(dtm_resized) & (dtm_resized != 0.0)
            valid_mask_resized = (ortho_valid & elev_valid).float()
            valid_ratio = valid_mask_resized.mean().item()

            # Heavy math
            residual = compute_topographic_residual(dtm_resized, valid_mask_resized)
            elev_valid_native = torch.isfinite(elevation) & (elevation != 0.0)
            is_tin = is_tin_artifact(elevation, elev_valid_native)

            # Sun vector
            dtm_norm: TrainingNormResult = self.evel_normalizer.normalize_for_training(dtm_resized, valid_mask_resized)
            image_norm = self.ortho_normalizer.normalize(image_resized)
            physical_residual = self.evel_normalizer.denormalize_prediction(dtm_norm.normed_residual)

            sun_vec, intensity, ambient = estimate_sun_vector_irls(
                physical_residual, image_norm, valid_mask_resized,
            )

            return {
                "idx": idx,
                "is_valid_data": True,
                "ortho_key": key,
                "valid_ratio": valid_ratio,
                "residual": residual,
                "is_tin": is_tin,
                "sun_x": sun_vec[0].item(),
                "sun_y": sun_vec[1].item(),
                "sun_z": sun_vec[2].item(),
                "intensity": intensity.item(),
                "ambient": ambient.item(),
                "num_merges": len(sample["meta"]),
            }

        except Exception:
            logger.exception("Failed to generate manifest for patch")
            return {"idx": idx, "is_valid_data": False}

    def __len__(self) -> int:
        if self.use_manifest:
            return len(self.clean_records)
        return len(self._raw_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | float]:
        """Lightning-fast __getitem__. No retries, no heavy math (manifest required)."""
        if not self.use_manifest:
            raise NotImplementedError(
                "Fallback on-the-fly __getitem__ removed for brevity. "
                "Please run with use_manifest=True."
            )

        record = self.clean_records[idx]
        geo_slice = self._raw_indices[record["idx"]]
        key = record["ortho_key"]

        sample = self.base[geo_slice]

        elevation = sample["elevation"]
        if elevation.ndim == 4:
            elevation = elevation[0]

        ortho = sample[key]
        if ortho.ndim == 4:
            ortho = ortho[0]

        sun_vector = torch.tensor([record["sun_x"], record["sun_y"], record["sun_z"]], dtype=torch.float32)
        intensity = torch.tensor(record["intensity"], dtype=torch.float32)
        ambient = torch.tensor(record["ambient"], dtype=torch.float32)

        dtm_resized = _safe_resize(elevation, self.resolution, has_nans=True)
        image_resized = _safe_resize(ortho, self.resolution)

        ortho_valid = (image_resized != 0.0).any(dim=0, keepdim=True)
        elev_valid = torch.isfinite(dtm_resized) & (dtm_resized != 0.0)
        valid_mask_resized = (ortho_valid & elev_valid).float()

        dtm_res: TrainingNormResult = self.evel_normalizer.normalize_for_training(dtm_resized, valid_mask_resized)
        dtm = dtm_res.normed_residual
        image = self.ortho_normalizer.normalize(image_resized)

        # Pre-VAE void fill so the VAE doesn't encode sharp black boundaries.
        image, dtm, valid_mask_resized = fill_voids_gmrf(image, dtm, valid_mask_resized)

        dtm = _to_3ch(dtm)
        image = _to_3ch(image)

        # Synchronised augmentation
        if getattr(self, "is_train", False) and getattr(self, "flip", False):
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

        if getattr(self, "is_train", False) and getattr(self, "bright_jitter", 0) > 0:
            factor = 1.0 + random.uniform(-self.bright_jitter, self.bright_jitter)
            image = image * factor

            if self.clip:
                image = torch.clip(image, -1.0, 1.0)

            # Scale lighting parameters so the loss physics still match the
            # augmented image.
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

        if "meta" in sample:
            res["meta"] = sample["meta"]

        return res


# ---------------------------------------------------------------------------
# Backwards-compat re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "DepthFMHiRISEAdapterCached",
    # image_processing re-exports — existing callers import these from here
    "erode_valid_mask",
    "fill_invalid_nearest_neighbor",
    "fill_voids_kriging",
    "fill_dtm_smart_diffusion",
    "fill_voids_gmrf",
    "SeamResult",
    "detect_seam_artifact",
    "is_tin_artifact",
    "compute_piecewise_linearity",
    "compute_artifact_multipliers",
    "compute_spatial_isolation",
    "estimate_sun_vector_ols",
    "estimate_sun_vector_irls",
    "compute_topographic_residual",
]
