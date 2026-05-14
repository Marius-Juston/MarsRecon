"""Legacy filesystem-based DTM dataset used by `precompute_latents.py`.

Reads pre-extracted image/DTM TIFFs from disk (the LitData streaming path in
`src/depth_fm/data/datamodule.py` is the current production path; this remains
for VAE latent precomputation only).

Provides:
* `load_geotiff(path)`        — small wrapper, also re-exported by callers.
* `normalize_dtm(dtm, method)` — minmax/log normalization helpers.
* `MarsDTMDataset`            — map-style dataset over a directory tree.
* `LatentDataset`             — alternative for pre-encoded latents.
"""

import random
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import tifffile

    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

from PIL import Image


def load_geotiff(path: str) -> np.ndarray:
    """Load a GeoTIFF/TIFF as a numpy array. Tries rasterio first, falls back to tifffile."""
    if HAS_RASTERIO:
        with rasterio.open(path) as src:
            data = src.read()  # (C, H, W)
            if data.shape[0] == 1:
                return data[0]  # (H, W) for single-band
            return data
    elif HAS_TIFFFILE:
        data = tifffile.imread(path)
        return data
    else:
        # Fallback: try PIL (won't work for float32 TIFFs)
        img = Image.open(path)
        return np.array(img)


def normalize_dtm(dtm: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize DTM values for VAE encoding.

    Args:
        dtm: raw elevation values (H, W), float32
        method: "minmax" or "log"

    Returns:
        Normalized DTM in [-1, 1] range, shape (H, W)
    """
    # Handle NaN/nodata values
    valid_mask = np.isfinite(dtm)
    if not valid_mask.any():
        return np.zeros_like(dtm)

    if method == "log":
        # Log-depth: compress large ranges, expand small differences
        # Shift so minimum is 1 (log(1) = 0), then log-transform
        dtm_min = dtm[valid_mask].min()
        dtm_shifted = np.where(valid_mask, dtm - dtm_min + 1.0, 1.0)
        dtm_log = np.log(dtm_shifted)
        # Then min-max to [-1, 1]
        log_min = dtm_log[valid_mask].min()
        log_max = dtm_log[valid_mask].max()
        if log_max - log_min < 1e-8:
            return np.zeros_like(dtm)
        dtm_norm = 2.0 * (dtm_log - log_min) / (log_max - log_min) - 1.0

    elif method == "minmax":
        # Simple min-max to [-1, 1]
        dtm_min = dtm[valid_mask].min()
        dtm_max = dtm[valid_mask].max()
        if dtm_max - dtm_min < 1e-8:
            return np.zeros_like(dtm)
        dtm_norm = 2.0 * (dtm - dtm_min) / (dtm_max - dtm_min) - 1.0
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Fill invalid pixels with 0 (neutral value in [-1,1])
    dtm_norm = np.where(valid_mask, dtm_norm, 0.0)
    return dtm_norm.astype(np.float32)


def dtm_to_3ch(dtm_normalized: np.ndarray) -> np.ndarray:
    """
    Replicate single-channel DTM to 3 channels for VAE encoding.
    Both DepthFM and Marigold do this to match the RGB-trained VAE.

    Args:
        dtm_normalized: (H, W) normalized DTM in [-1, 1]
    Returns:
        (3, H, W) tensor
    """
    return np.stack([dtm_normalized] * 3, axis=0)


class MarsDTMDataset(Dataset):
    """
    Dataset for Mars image-DTM pairs.

    Handles stereo augmentation: if files are named tile_0001_L.tif and
    tile_0001_R.tif, both are paired with tile_0001.tif from the DTM directory.
    """

    def __init__(
            self,
            image_dir: str,
            dtm_dir: str,
            confidence_dir: Optional[str] = None,
            resolution: int = 512,
            dtm_normalization: str = "minmax",
            use_stereo_augmentation: bool = True,
            random_horizontal_flip: bool = True,
            random_vertical_flip: bool = True,
            random_rotation_degrees: float = 0.0,
            brightness_jitter: float = 0.0,
            contrast_jitter: float = 0.0,
            is_validation: bool = False,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.dtm_dir = dtm_dir
        self.confidence_dir = confidence_dir
        self.resolution = resolution
        self.dtm_normalization = dtm_normalization
        self.use_stereo = use_stereo_augmentation
        self.is_validation = is_validation

        # Augmentation params
        self.h_flip = random_horizontal_flip and not is_validation
        self.v_flip = random_vertical_flip and not is_validation
        self.rot_degrees = random_rotation_degrees if not is_validation else 0.0
        self.brightness = brightness_jitter if not is_validation else 0.0
        self.contrast = contrast_jitter if not is_validation else 0.0

        # Build paired file list
        self.pairs = self._build_pairs()
        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No image-DTM pairs found. Check directories:\n"
                f"  images: {image_dir}\n  dtms: {dtm_dir}"
            )

    def _build_pairs(self):
        """
        Match image files to DTM files by name.
        Handles stereo naming: tile_0001_L.tif, tile_0001_R.tif -> tile_0001.tif
        """
        img_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        image_files = sorted([
            f for f in Path(self.image_dir).iterdir()
            if f.suffix.lower() in img_extensions
        ])

        pairs = []
        for img_path in image_files:
            stem = img_path.stem

            # Strip _L or _R suffix to find DTM match
            dtm_stem = re.sub(r'[_-][LR]$', '', stem, flags=re.IGNORECASE)

            # Find matching DTM
            dtm_path = None
            for ext in img_extensions:
                candidate = Path(self.dtm_dir) / f"{dtm_stem}{ext}"
                if candidate.exists():
                    dtm_path = candidate
                    break

            if dtm_path is None:
                continue  # skip images without matching DTM

            # If not using stereo augmentation, skip _R images
            if not self.use_stereo and re.search(r'[_-]R$', stem, re.IGNORECASE):
                continue

            # Confidence map (optional)
            conf_path = None
            if self.confidence_dir:
                for ext in img_extensions:
                    candidate = Path(self.confidence_dir) / f"{dtm_stem}{ext}"
                    if candidate.exists():
                        conf_path = candidate
                        break

            pairs.append({
                "image": str(img_path),
                "dtm": str(dtm_path),
                "confidence": str(conf_path) if conf_path else None,
                "tile_id": dtm_stem,
            })

        return pairs

    def __len__(self):
        return len(self.pairs)

    def _random_crop(self, *arrays, size):
        """Apply the same random crop to multiple arrays."""
        h, w = arrays[0].shape[-2:]
        if h < size or w < size:
            # Resize up if too small
            scale = max(size / h, size / w) * 1.05
            new_h, new_w = int(h * scale), int(w * scale)
            resized = []
            for arr in arrays:
                if arr is None:
                    resized.append(None)
                    continue
                t = torch.from_numpy(arr).unsqueeze(0) if arr.ndim == 2 else torch.from_numpy(arr).unsqueeze(0)
                t = torch.nn.functional.interpolate(
                    t.unsqueeze(0).float(), size=(new_h, new_w), mode="bilinear", align_corners=False
                ).squeeze(0).squeeze(0)
                resized.append(t.numpy())
            arrays = resized
            h, w = new_h, new_w

        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
        results = []
        for arr in arrays:
            if arr is None:
                results.append(None)
            elif arr.ndim == 2:
                results.append(arr[top:top + size, left:left + size])
            else:
                results.append(arr[..., top:top + size, left:left + size])
        return results

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Load image and DTM
        image = load_geotiff(pair["image"]).astype(np.float32)
        dtm = load_geotiff(pair["dtm"]).astype(np.float32)

        # Load confidence if available
        confidence = None
        if pair["confidence"]:
            confidence = load_geotiff(pair["confidence"]).astype(np.float32)

        # Ensure image is (H, W) or (C, H, W)
        if image.ndim == 2:
            # Grayscale Mars RED image -> replicate to 3 channels later
            pass
        elif image.ndim == 3 and image.shape[0] in (1, 3, 4):
            if image.shape[0] == 1:
                image = image[0]
            elif image.shape[0] == 4:
                image = image[:3]  # drop alpha

        # Ensure DTM is (H, W)
        if dtm.ndim == 3:
            dtm = dtm[0]

        # Random crop to resolution
        crop_items = [image, dtm, confidence]
        image, dtm, confidence = self._random_crop(*crop_items, size=self.resolution)

        # Normalize image to [-1, 1]
        if image.ndim == 2:
            # Grayscale: normalize then replicate
            img_min, img_max = image.min(), image.max()
            if img_max - img_min > 1e-8:
                image = 2.0 * (image - img_min) / (img_max - img_min) - 1.0
            else:
                image = np.zeros_like(image)
            image = np.stack([image] * 3, axis=0)  # (3, H, W)
        else:
            # Already multi-channel
            img_min, img_max = image.min(), image.max()
            if img_max - img_min > 1e-8:
                image = 2.0 * (image - img_min) / (img_max - img_min) - 1.0
            else:
                image = np.zeros_like(image)

        # Normalize DTM
        dtm_norm = normalize_dtm(dtm, method=self.dtm_normalization)
        dtm_3ch = dtm_to_3ch(dtm_norm)  # (3, H, W)

        # Convert to tensors
        image_t = torch.from_numpy(image).float()
        dtm_t = torch.from_numpy(dtm_3ch).float()
        conf_t = torch.from_numpy(confidence).float() if confidence is not None else torch.ones(1, self.resolution,
                                                                                                self.resolution)

        # Normalize confidence to [0, 1]
        if confidence is not None:
            conf_min, conf_max = conf_t.min(), conf_t.max()
            if conf_max - conf_min > 1e-8:
                conf_t = (conf_t - conf_min) / (conf_max - conf_min)

        # Apply synchronized augmentations (same transform to image and DTM)
        if self.h_flip and random.random() > 0.5:
            image_t = TF.hflip(image_t)
            dtm_t = TF.hflip(dtm_t)
            conf_t = TF.hflip(conf_t)

        if self.v_flip and random.random() > 0.5:
            image_t = TF.vflip(image_t)
            dtm_t = TF.vflip(dtm_t)
            conf_t = TF.vflip(conf_t)

        if self.rot_degrees > 0:
            angle = random.uniform(-self.rot_degrees, self.rot_degrees)
            image_t = TF.rotate(image_t, angle)
            dtm_t = TF.rotate(dtm_t, angle)
            conf_t = TF.rotate(conf_t, angle)

        # Brightness/contrast jitter on image ONLY (not DTM)
        if self.brightness > 0 or self.contrast > 0:
            # Shift image to [0, 1] for jitter, then back
            image_01 = (image_t + 1.0) / 2.0
            if self.brightness > 0:
                b_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
                image_01 = torch.clamp(image_01 * b_factor, 0, 1)
            if self.contrast > 0:
                c_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
                mean = image_01.mean()
                image_01 = torch.clamp((image_01 - mean) * c_factor + mean, 0, 1)
            image_t = image_01 * 2.0 - 1.0

        return {
            "image": image_t,  # (3, H, W) in [-1, 1]
            "dtm": dtm_t,  # (3, H, W) in [-1, 1]
            "confidence": conf_t,  # (1, H, W) in [0, 1]
            "tile_id": pair["tile_id"],
        }


class LatentDataset(Dataset):
    """
    Dataset that loads pre-computed VAE latents from disk.
    Much faster than encoding on-the-fly during training.

    Expected structure:
        latent_dir/
            tile_0001_L_image.pt    # image latent, shape (C, h, w)
            tile_0001_L_dtm.pt      # DTM latent, shape (C, h, w)
            tile_0001_L_conf.pt     # confidence map (optional)
    """

    def __init__(self, latent_dir: str, is_validation: bool = False):
        super().__init__()
        self.latent_dir = Path(latent_dir)

        # Find all image latent files
        image_latents = sorted(self.latent_dir.glob("*_image.pt"))
        self.samples = []
        for img_lat in image_latents:
            stem = img_lat.stem.replace("_image", "")
            dtm_lat = self.latent_dir / f"{stem}_dtm.pt"
            if dtm_lat.exists():
                conf_lat = self.latent_dir / f"{stem}_conf.pt"
                self.samples.append({
                    "image_latent": str(img_lat),
                    "dtm_latent": str(dtm_lat),
                    "conf_latent": str(conf_lat) if conf_lat.exists() else None,
                    "tile_id": stem,
                })

        if len(self.samples) == 0:
            raise RuntimeError(f"No latent pairs found in {latent_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_latent = torch.load(sample["image_latent"], map_location="cpu", weights_only=True)
        dtm_latent = torch.load(sample["dtm_latent"], map_location="cpu", weights_only=True)

        if sample["conf_latent"]:
            conf = torch.load(sample["conf_latent"], map_location="cpu", weights_only=True)
        else:
            conf = torch.ones(1, image_latent.shape[-2], image_latent.shape[-1])

        return {
            "image_latent": image_latent,
            "dtm_latent": dtm_latent,
            "confidence": conf,
            "tile_id": sample["tile_id"],
        }
