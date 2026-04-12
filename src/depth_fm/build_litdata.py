"""
Preprocess MarsHiRISEDTM into LitData optimized streaming format.

Uses a two-phase approach to avoid spawn/pickle incompatibility:
  Phase 1: Fork-based DataLoader extracts samples to temp .npz files
           (GDAL/rasterio require fork — spawn can't pickle them)
           *Also recomputes robust OLS sun vectors on-the-fly*
  Phase 2: litdata.optimize repacks .npz files into optimized chunks
           (litdata forces spawn — but .npz reading is trivially picklable)

Usage:
    python build_litdata.py --config configs/train_hirise.yaml --workers 96
"""
import argparse
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

# GDAL / threading optimizations for the extraction phase
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["VSI_CACHE"] = "TRUE"
os.environ["VSI_CACHE_SIZE"] = "200000000"
os.environ["GDAL_NUM_THREADS"] = "1"
os.environ["GDAL_MAX_DATASET_POOL_SIZE"] = "1024"
os.environ["OMP_NUM_THREADS"] = "2"

import rasterio
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from litdata import optimize

from dataset.mars_hirise_dtm import MarsHiRISEDTM
from dataset.hirise_sampler import HiRISEGeoSampler
from torchgeo.samplers import Units
from depth_fm.depthfm_adapter import DepthFMHiRISEAdapterCached, estimate_sun_vector_ols

import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)


def get_litdata_cache_key(config) -> str:
    """Deterministic hash for the current preprocessing configuration."""
    key_parts = {
        "hirise": OmegaConf.to_container(config.data.hirise, resolve=True),
        "sampler": OmegaConf.to_container(config.data.sampler, resolve=True),
        "resolution": config.data.get("resolution", 512),
        "dtm_normalization": config.data.get("dtm_normalization", "relative"),
    }
    raw = json.dumps(key_parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _configure_worker_logger(worker_id):
    """Configure logging in fork workers."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Worker {worker_id}] %(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE="TRUE").__enter__()


# ── Top-level function for litdata.optimize (must be picklable for spawn) ──

def _repack_npz(npz_path: str) -> dict:
    """Read a temp .npz and return dict for litdata chunking.

    Top-level function with no closures — trivially picklable.
    """
    import numpy as np
    data = np.load(npz_path)
    return {
        "image": data["image"],  # float16 (3, H, W)
        "dtm": data["dtm"],  # float16 (3, H, W)
        "confidence": data["confidence"],  # float16 (1, H, W)
        "sun_vector": data["sun_vector"],  # float32 (3,)
        "intensity": data["intensity"],  # float32 scalar
        "ambient": data["ambient"],  # float32 scalar
    }


def build_litdata_for_split(config, split: str, cache_hash: str, workers: int = 64):
    """Extract one split into LitData optimized format."""
    hc = config.data.hirise
    sc = config.data.sampler

    dataset_root = Path(hc.root)
    output_dir = str(dataset_root / f"litdata_cache_{cache_hash}" / split)

    # Check for completion marker
    success_marker = Path(output_dir) / "_SUCCESS"
    if success_marker.exists():
        logger.info(f"[{split}] Valid LitData cache found at {output_dir}. Skipping.")
        return

    bbox_tuple = tuple(hc.bbox) if hc.get("bbox") else None
    ortho_type = hc.get("ortho_type", "RED")
    if isinstance(ortho_type, str):
        ortho_type = [ortho_type]

    base_dataset = MarsHiRISEDTM(
        root=hc.root,
        include_ortho=hc.get("include_ortho", True),
        ortho_type=ortho_type,
        ortho_scale=hc.get("ortho_scale"),
        download=hc.get("download", False),
        bbox=bbox_tuple,
        reuse_cache=hc.get("reuse_cache", True),
        target=hc.get("target"),
        return_meta=True,
    )

    if hasattr(base_dataset, 'index'):
        _ = base_dataset.index

    split_fractions = tuple(config.data.get("split_fractions", [0.8, 0.1, 0.1]))
    split_method = config.data.get("split_method", "geographic")
    split_axis = config.data.get("split_axis", "longitude")
    n_folds = config.data.get("n_folds")
    fold_idx = config.data.get("fold_idx", 0)

    resolution = config.data.get("resolution", 512)
    dtm_norm = config.data.get("dtm_normalization", "relative")
    stats_path = config.data.get("stats_path")

    manifest_cache_dir = dataset_root / ".cache" / "manifests"
    manifest_cache_dir.mkdir(parents=True, exist_ok=True)

    sampler = HiRISEGeoSampler(
        base_dataset,
        split=split,
        size=sc.get("size", 0.018),
        units=Units.CRS,
        split_fractions=split_fractions,
        split_method=split_method,
        split_axis=split_axis,
        n_folds=n_folds,
        fold_idx=fold_idx,
        reuse_cache=True,
    )

    adapter = DepthFMHiRISEAdapterCached(
        base_dataset=base_dataset,
        sampler=sampler,
        resolution=resolution,
        dtm_normalization=dtm_norm,
        random_flip=False,
        brightness_jitter=0.0,
        stats_path=stats_path,
        use_manifest=True,
        manifest_workers=min(workers, 94),
        manifest_dir=str(manifest_cache_dir),
    )

    num_samples = len(adapter)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Extract with fork-based DataLoader → temp .npz files
    #          Includes on-the-fly sun vector estimation & patching
    # ═══════════════════════════════════════════════════════════════════

    tmp_dir = dataset_root / f"_litdata_tmp_{cache_hash}_{split}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    effective_workers = min(workers, max(1, num_samples // 4))

    loader = DataLoader(
        adapter,
        batch_size=None,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=False,
        prefetch_factor=2 if workers > 0 else None,
        drop_last=False,
        persistent_workers=True if workers > 0 else False,
        multiprocessing_context="fork" if workers > 0 else None,
        worker_init_fn=_configure_worker_logger if workers > 0 else None,
    )

    logger.info(f"[{split}] Phase 1/2: Extracting {num_samples} samples to temp .npz files...")
    npz_paths = []

    for i, sample in enumerate(tqdm(loader, total=num_samples, desc=f"Extract {split}")):
        # Ensure we have float32 tensors for the OLS math
        image_fp32 = sample["image"].float()  # (3, H, W)
        dtm_fp32 = sample["dtm"].float()  # (3, H, W)
        conf_fp32 = sample["confidence"].float()  # (1, H, W)

        # Recompute sun vectors using single-channel DTM and the image
        dtm_1ch = dtm_fp32[:1]  # (1, H, W)
        sun_vec, intensity, ambient = estimate_sun_vector_ols(dtm_1ch, image_fp32, conf_fp32)

        # Normalize to strict unit vector
        sun_vec = torch.nn.functional.normalize(sun_vec, p=2, dim=0)

        npz_path = str(tmp_dir / f"{i:08d}.npz")
        np.savez(
            npz_path,
            image=sample["image"].numpy().astype(np.float16),
            dtm=sample["dtm"].numpy().astype(np.float16),
            confidence=sample["confidence"].numpy().astype(np.float16),
            sun_vector=sun_vec.numpy(),
            intensity=intensity.numpy(),
            ambient=ambient.numpy(),
        )
        npz_paths.append(npz_path)

    del loader, adapter, sampler, base_dataset
    logger.info(f"[{split}] Phase 1 complete: {len(npz_paths)} .npz files.")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: Repack into LitData optimized chunks (spawn-safe)
    # ═══════════════════════════════════════════════════════════════════

    logger.info(f"[{split}] Phase 2/2: Packing into LitData chunks at {output_dir}...")

    optimize(
        fn=_repack_npz,
        inputs=npz_paths,
        output_dir=output_dir,
        num_workers=min(workers, 32),
        chunk_bytes="256MB",
    )

    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info(f"[{split}] Cleaned up temp directory.")

    success_marker = Path(output_dir) / "_SUCCESS"
    success_marker.touch()
    logger.info(f"[{split}] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_hirise.yaml")
    parser.add_argument("--workers", type=int, default=96 * 2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = OmegaConf.load(args.config)

    cache_hash = get_litdata_cache_key(config)
    logger.info(f"LitData Cache Hash: {cache_hash}")

    for split in ["test", "val", "train"]:
        build_litdata_for_split(config, split, cache_hash, workers=args.workers)
