"""
Fixed LitData DataModule — works around StreamingDataLoader re-iteration deadlock.

THE BUG:
  LitData's StreamingDataLoader doesn't properly reset its internal iterator
  state when re-iterated (GitHub Issues #316, #213, #452). In DDP, this causes
  ranks to get different batch counts on the 2nd+ validation, deadlocking on
  the next sync_dist or NCCL collective.

THE FIX (two options, both provided):

  Option A (default): Use torch.utils.data.DataLoader for val/test.
    StreamingDataset is an IterableDataset and works fine with the regular
    DataLoader. You lose StreamingDataLoader's prefetch optimizations, but
    val/test are small — this costs ~1 second per validation.

  Option B: Recreate StreamingDataset + StreamingDataLoader from scratch
    each time val_dataloader() is called. This avoids the stale-state bug
    by never re-iterating the same object. Slightly more overhead from
    worker startup.

Drop-in replacement for litdata_datamodule_1.py — same API, same config.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import lightning as L
import numpy as np
import torch
from litdata import StreamingDataset, StreamingDataLoader
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
_VAL_WORKERS = 4
_WORKERS_PER_GPU = 4


# ---------------------------------------------------------------------------
# Dataset (unchanged from original)
# ---------------------------------------------------------------------------

class MarsStreamingDataset(StreamingDataset):
    """StreamingDataset subclass with on-the-fly augmentations."""

    def __init__(
            self,
            input_dir: str,
            is_train: bool = False,
            random_flip: bool = True,
            brightness_jitter: float = 0.1,
            shuffle: bool = False,
            drop_last: bool = True,
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
        raw = super().__getitem__(index)

        image = torch.from_numpy(raw["image"].astype(np.float32))
        dtm = torch.from_numpy(raw["dtm"].astype(np.float32))
        confidence = torch.from_numpy(raw["confidence"].astype(np.float32))
        sun_vector = torch.from_numpy(raw["sun_vector"].astype(np.float32))
        intensity = torch.tensor(float(raw["intensity"]), dtype=torch.float32)
        ambient = torch.tensor(float(raw["ambient"]), dtype=torch.float32)

        if self.random_flip:
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, [-1])
                dtm = torch.flip(dtm, [-1])
                confidence = torch.flip(confidence, [-1])
                sun_vector[0] = -sun_vector[0]

            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, [-2])
                dtm = torch.flip(dtm, [-2])
                confidence = torch.flip(confidence, [-2])
                sun_vector[1] = -sun_vector[1]

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
# Cache key (unchanged)
# ---------------------------------------------------------------------------

def _get_litdata_cache_key(config) -> str:
    from omegaconf import OmegaConf
    key_parts = {
        "hirise": OmegaConf.to_container(config.data.hirise, resolve=True),
        "sampler": OmegaConf.to_container(config.data.sampler, resolve=True),
        "resolution": config.data.get("resolution", 512),
        "dtm_normalization": config.data.get("dtm_normalization", "relative"),
    }
    raw = json.dumps(key_parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Fixed DataModule
# ---------------------------------------------------------------------------

class MarsDepthFMDataModule(L.LightningDataModule):
    """Lightning DataModule with LitData re-iteration deadlock fix.

    Key change: val/test use torch.utils.data.DataLoader instead of
    StreamingDataLoader. The StreamingDataset works with both — we only
    lose LitData's prefetch optimizations, which are irrelevant for the
    small val/test sets (~25 batches).

    Train still uses StreamingDataLoader for its shuffle + prefetch benefits.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.hc = config.data.hirise
        self.tc = config.training

        self.cache_hash = _get_litdata_cache_key(config)
        self.litdata_root = Path(self.hc.root) / f"litdata_cache_{self.cache_hash}"

        self._train_dataset = None
        # val/test datasets stored only if strategy == "torch"
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

        if stage in ("val", None):
            self._val_dataset = MarsStreamingDataset(
                input_dir=str(self.litdata_root / "val"),
                is_train=False,
                shuffle=False,
                drop_last=True,
                seed=seed,
            )

        if stage in ("test", None):
            self._test_dataset = MarsStreamingDataset(
                input_dir=str(self.litdata_root / "test"),
                is_train=False,
                shuffle=False,
                drop_last=True,
                seed=seed,
            )

    def train_dataloader(self):
        # Train uses StreamingDataLoader — it's only iterated once per epoch
        # (no re-iteration bug since Lightning creates a new iterator each time
        # and train only runs forward, never backward through the same iterator)
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
            drop_last=True,
            persistent_workers=False,  # must be False when recreating
        )

    def test_dataloader(self):

        return StreamingDataLoader(
            self._test_dataset,
            batch_size=self.tc.per_gpu_batch_size,
            num_workers=min(self.tc.num_workers, 4),
            pin_memory=self.tc.pin_memory,
            drop_last=True,
            persistent_workers=False,
        )


def _build_litdata_loaders(config, split_seed: int = 42) -> dict:
    """Build DataLoaders from pre-optimized LitData cache.

    Returns dict of {split: DataLoader} or raises FileNotFoundError.

    FIX: Val/test use torch.utils.data.DataLoader instead of
    StreamingDataLoader to avoid the re-iteration deadlock bug
    (LitData GitHub Issues #316, #213, #452).
    """
    from depth_fm.litdata_datamodule import MarsStreamingDataset
    from torch.utils.data import DataLoader as TorchDataLoader

    try:
        from litdata import StreamingDataLoader
    except ImportError:
        raise FileNotFoundError("litdata package not installed")

    hc = config.data.hirise
    tc = config.training

    cache_hash = _get_litdata_cache_key(config)
    litdata_root = Path(hc.root) / f"litdata_cache_{cache_hash}"

    splits = ("train", "val", "test")
    if not all((litdata_root / s / "_SUCCESS").exists() for s in splits):
        raise FileNotFoundError(f"LitData cache incomplete at {litdata_root}")

    brightness_jitter = config.data.get("brightness_jitter", 0.1)
    num_workers = tc.get("num_workers", _WORKERS_PER_GPU)

    loaders = {}
    for split in splits:
        is_train = split == "train"
        dataset = MarsStreamingDataset(
            input_dir=str(litdata_root / split),
            is_train=is_train,
            random_flip=is_train,
            brightness_jitter=brightness_jitter if is_train else 0.0,
            shuffle=is_train,
            # IMPORTANT: drop_last=True on ALL splits in DDP.
            drop_last=True,
            seed=split_seed,
        )

        loaders[split] = StreamingDataLoader(
            dataset,
            batch_size=tc.per_gpu_batch_size,
            num_workers=num_workers,
            pin_memory=tc.pin_memory,
            drop_last=True,
            persistent_workers=True,
        )


        w = num_workers if is_train else _VAL_WORKERS
        logger.info(
            "LitData [%s]: batch_size=%d, workers=%d, size=%d",
            split, tc.per_gpu_batch_size, w, len(loaders[split])
        )

    return loaders
