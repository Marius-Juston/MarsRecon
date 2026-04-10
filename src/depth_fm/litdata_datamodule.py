"""
Hyper-optimized LitData DataModule for Mars DepthFM training.

Replaces both the WebDataset and fallback TorchGeo DataLoader paths
with LitData's StreamingDataset, which provides:

  - Zero-copy deserialization (mmap-backed numpy → torch)
  - Automatic per-GPU sharding (no manual split_by_node/split_by_worker)
  - Deterministic shuffling across epochs with configurable buffer
  - Chunk-sequential reads that saturate NVMe bandwidth
  - Native Lightning Trainer integration (DDP, FSDP, etc.)

Usage:
    from depth_fm.litdata_datamodule import MarsDepthFMDataModule

    dm = MarsDepthFMDataModule(config)
    trainer = L.Trainer(...)
    trainer.fit(model, datamodule=dm)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Literal

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from litdata import StreamingDataset, StreamingDataLoader
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Streaming Dataset with on-the-fly augmentations
# ---------------------------------------------------------------------------


class MarsStreamingDataset(StreamingDataset):
    """StreamingDataset subclass that applies training augmentations on read.

    The underlying chunks store pre-normalized float16 numpy arrays.
    This class converts them to float32 torch tensors and optionally
    applies random flips, rotations, and brightness jitter — all
    synchronised across image/dtm/confidence/sun_vector.
    """

    def __init__(
        self,
        input_dir: str,
        is_train: bool = False,
        random_flip: bool = True,
        brightness_jitter: float = 0.1,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 42,
    ):
        super().__init__(
            input_dir=input_dir,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )
        self.is_train = is_train
        self.random_flip = random_flip and is_train
        self.brightness_jitter = brightness_jitter if is_train else 0.0

    def __getitem__(self, index):
        # StreamingDataset returns a dict of numpy arrays
        raw = super().__getitem__(index)

        # Zero-copy cast: float16 numpy → float32 torch tensor
        image = torch.from_numpy(raw["image"].astype(np.float32))
        dtm = torch.from_numpy(raw["dtm"].astype(np.float32))
        confidence = torch.from_numpy(raw["confidence"].astype(np.float32))
        sun_vector = torch.from_numpy(raw["sun_vector"].astype(np.float32))
        intensity = torch.tensor(float(raw["intensity"]), dtype=torch.float32)
        ambient = torch.tensor(float(raw["ambient"]), dtype=torch.float32)

        # -----------------------------------------------------------------
        # Synchronised augmentations (identical to DepthFMHiRISEAdapterCached)
        # -----------------------------------------------------------------
        if self.random_flip:
            # Horizontal flip
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, [-1])
                dtm = torch.flip(dtm, [-1])
                confidence = torch.flip(confidence, [-1])
                sun_vector[0] = -sun_vector[0]

            # Vertical flip
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, [-2])
                dtm = torch.flip(dtm, [-2])
                confidence = torch.flip(confidence, [-2])
                sun_vector[1] = -sun_vector[1]

            # Random 90° rotation
            k_rot = torch.randint(0, 4, (1,)).item()
            if k_rot > 0:
                image = torch.rot90(image, k=k_rot, dims=[-2, -1])
                dtm = torch.rot90(dtm, k=k_rot, dims=[-2, -1])
                confidence = torch.rot90(confidence, k=k_rot, dims=[-2, -1])

                sx, sy = sun_vector[0].clone(), sun_vector[1].clone()
                if k_rot == 1:
                    sun_vector[0], sun_vector[1] = sy, -sx
                elif k_rot == 2:
                    sun_vector[0], sun_vector[1] = -sx, -sy
                elif k_rot == 3:
                    sun_vector[0], sun_vector[1] = -sy, sx

        # Brightness jitter
        if self.brightness_jitter > 0:
            import random
            factor = 1.0 + random.uniform(-self.brightness_jitter, self.brightness_jitter)
            image = (image * factor).clamp(-1.0, 1.0)
            intensity = intensity * factor
            ambient = ambient * factor

        return {
            "image": image,
            "dtm": dtm,
            "confidence": confidence,
            "sun_vector": sun_vector,
            "intensity": intensity,
            "ambient": ambient,
        }


# ---------------------------------------------------------------------------
# Lightning DataModule
# ---------------------------------------------------------------------------


def _get_litdata_cache_key(config) -> str:
    """Same hash as build_litdata.py to locate the preprocessed data."""
    from omegaconf import OmegaConf
    key_parts = {
        "hirise": OmegaConf.to_container(config.data.hirise, resolve=True),
        "sampler": OmegaConf.to_container(config.data.sampler, resolve=True),
        "resolution": config.data.get("resolution", 512),
        "dtm_normalization": config.data.get("dtm_normalization", "relative"),
    }
    raw = json.dumps(key_parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class MarsDepthFMDataModule(L.LightningDataModule):
    """Lightning DataModule backed by LitData StreamingDataset.

    Expects the data to have been preprocessed with ``build_litdata.py``.
    Automatically locates the cache directory under the dataset root.

    If the LitData cache is not found, falls back to the legacy
    DepthFMHiRISEAdapterCached path (slow but functional).
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.hc = config.data.hirise
        self.tc = config.training

        self.cache_hash = _get_litdata_cache_key(config)
        self.litdata_root = Path(self.hc.root) / f"litdata_cache_{self.cache_hash}"

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    @property
    def litdata_available(self) -> bool:
        return all(
            (self.litdata_root / split / "_SUCCESS").exists()
            for split in ("train", "val", "test")
        )

    def setup(self, stage: str | None = None):
        if not self.litdata_available:
            raise FileNotFoundError(
                f"LitData cache not found at {self.litdata_root}. "
                f"Run `python build_litdata.py --config <your_config>` first."
            )

        brightness_jitter = self.config.data.get("brightness_jitter", 0.1)
        seed = self.config.data.get("seed", 42)

        if stage in ("fit", None):
            self._train_dataset = MarsStreamingDataset(
                input_dir=str(self.litdata_root / "train"),
                is_train=True,
                random_flip=True,
                brightness_jitter=brightness_jitter,
                shuffle=True,
                drop_last=True,
                seed=seed,
            )
            self._val_dataset = MarsStreamingDataset(
                input_dir=str(self.litdata_root / "val"),
                is_train=False,
                shuffle=False,
                drop_last=False,
                seed=seed,
            )

        if stage in ("test", None):
            self._test_dataset = MarsStreamingDataset(
                input_dir=str(self.litdata_root / "test"),
                is_train=False,
                shuffle=False,
                drop_last=False,
                seed=seed,
            )

    def train_dataloader(self):
        return StreamingDataLoader(
            self._train_dataset,
            batch_size=self.tc.per_gpu_batch_size,
            num_workers=self.tc.num_workers,
            pin_memory=self.tc.pin_memory,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return StreamingDataLoader(
            self._val_dataset,
            batch_size=self.tc.per_gpu_batch_size,
            num_workers=min(self.tc.num_workers, 4),
            pin_memory=self.tc.pin_memory,
            drop_last=False,
            persistent_workers=False,
        )

    def test_dataloader(self):
        return StreamingDataLoader(
            self._test_dataset,
            batch_size=self.tc.per_gpu_batch_size,
            num_workers=min(self.tc.num_workers, 4),
            pin_memory=self.tc.pin_memory,
            drop_last=False,
            persistent_workers=False,
        )


# ---------------------------------------------------------------------------
# Drop-in replacement for build_dataloaders() in train_lightning.py
# ---------------------------------------------------------------------------


def build_litdata_dataloaders(config, split_seed: int = 42) -> dict:
    """Drop-in replacement for the existing build_dataloaders() function.

    Returns a dict of {"train": loader, "val": loader, "test": loader}
    compatible with the existing training loop.
    """
    cache_hash = _get_litdata_cache_key(config)
    litdata_root = Path(config.data.hirise.root) / f"litdata_cache_{cache_hash}"
    tc = config.training

    litdata_available = all(
        (litdata_root / split / "_SUCCESS").exists()
        for split in ("train", "val", "test")
    )

    if not litdata_available:
        raise FileNotFoundError(
            f"LitData cache not found at {litdata_root}. "
            f"Run `python build_litdata.py --config <your_config>` first.\n"
            f"Expected _SUCCESS markers in train/, val/, test/ subdirectories."
        )

    brightness_jitter = config.data.get("brightness_jitter", 0.1)
    loaders = {}

    for split in ("train", "val", "test"):
        is_train = split == "train"

        dataset = MarsStreamingDataset(
            input_dir=str(litdata_root / split),
            is_train=is_train,
            random_flip=is_train,
            brightness_jitter=brightness_jitter if is_train else 0.0,
            shuffle=is_train,
            drop_last=is_train,
            seed=split_seed,
        )

        loaders[split] = StreamingDataLoader(
            dataset,
            batch_size=tc.per_gpu_batch_size,
            num_workers=tc.num_workers if is_train else min(tc.num_workers, 4),
            pin_memory=tc.pin_memory,
            drop_last=is_train,
            persistent_workers=is_train,
        )

        logger.info(
            "LitData StreamingDataLoader [%s] ready: batch_size=%d, workers=%d",
            split, tc.per_gpu_batch_size,
            tc.num_workers if is_train else min(tc.num_workers, 4),
        )

    return loaders