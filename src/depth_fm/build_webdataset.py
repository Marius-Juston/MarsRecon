"""
Extracts raw tensors from MarsHiRISEDTM and packs them into WebDataset .tar archives.
Run this ONCE per configuration. Subsequence runs with the same config will skip extraction.
"""
import argparse
import logging
import os
from pathlib import Path

from tqdm import tqdm

# Force maximum GDAL speed for the extraction phase & prevent CPU choking
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["VSI_CACHE"] = "FALSE"  # Good for sequential extraction
os.environ["GDAL_NUM_THREADS"] = "1"
os.environ["GDAL_MAX_DATASET_POOL_SIZE"] = "1024"
os.environ["OMP_NUM_THREADS"] = "1"  # Prevent NumPy/SciPy thread explosion with many workers

import webdataset as wds
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from dataset.mars_hirise_dtm import MarsHiRISEDTM
from dataset.hirise_sampler import HiRISEGeoSampler
from torchgeo.samplers import Units
from depth_fm.depthfm_adapter import DepthFMHiRISEAdapterCached
from depth_fm.train_lightning import configure_worker_logger, get_wds_cache_key

# Prevent "Too many open files" errors when using many DataLoader workers
import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)


def build_wds_for_split(config, split: str, wds_hash: str, workers: int = 64):
    hc = config.data.hirise
    sc = config.data.sampler

    # Anchor the WebDataset output directly inside the dataset root
    dataset_root = Path(hc.root)
    wds_output_dir = dataset_root / f"wds_cache_{wds_hash}" / split
    wds_output_dir.mkdir(parents=True, exist_ok=True)

    # Check for completion marker
    success_marker = wds_output_dir / "_SUCCESS"
    if success_marker.exists():
        logger.info(f"[{split}] Valid WebDataset cache found at {wds_output_dir}. Skipping.")
        return

    bbox_tuple = tuple(hc.bbox) if hc.get("bbox") else None
    ortho_type = hc.get("ortho_type", "RED")
    if isinstance(ortho_type, str):
        ortho_type = [ortho_type]

    # 1. Base Dataset
    base_dataset = MarsHiRISEDTM(
        root=hc.root,
        include_ortho=hc.get("include_ortho", True),
        ortho_type=ortho_type,
        ortho_scale=hc.get("ortho_scale"),
        download=hc.get("download", False),
        bbox=bbox_tuple,
        reuse_cache=hc.get("reuse_cache", True),
        target=hc.get("target"),
        return_meta=True
    )

    # Pre-build/load the spatial index in the main thread.
    if hasattr(base_dataset, 'index'):
        _ = base_dataset.index

    split_fractions = tuple(config.data.get("split_fractions", [0.8, 0.1, 0.1]))
    split_method = config.data.get("split_method", "geographic")
    split_axis = config.data.get("split_axis", "longitude")
    n_folds = config.data.get("n_folds")
    fold_idx = config.data.get("fold_idx", 0)

    common_sampler_kwargs = dict(
        size=sc.get("size", 0.009),
        units=Units.CRS,
        split_fractions=split_fractions,
        split_method=split_method,
        split_axis=split_axis,
        n_folds=n_folds,
        fold_idx=fold_idx,
        reuse_cache=True,
    )

    resolution = config.data.get("resolution", 512)
    dtm_norm = config.data.get("dtm_normalization", "relative")
    stats_path = config.data.get("stats_path")

    # 2. Sampler
    is_train = (split == "train")

    sampler = HiRISEGeoSampler(
        base_dataset,
        split=split,
        **common_sampler_kwargs,
    )

    # Ensure manifest caches properly in the dataset root, not the execution dir
    manifest_cache_dir = dataset_root / ".cache" / "manifests"
    manifest_cache_dir.mkdir(parents=True, exist_ok=True)

    # FIXME it seems that since the current math is done on the CPU rather on the GPU some of the algorithms do not
    # return exactly what is expected, so there is a difference between the true sun_view and the expected

    # 3. Adapter (CRITICAL: Disable augmentations for static storage!)
    adapter = DepthFMHiRISEAdapterCached(
        base_dataset=base_dataset,
        sampler=sampler,
        resolution=resolution,
        dtm_normalization=dtm_norm,
        random_flip=False,
        brightness_jitter=0.0,
        stats_path=stats_path,
        use_manifest=True,
        manifest_workers=min(workers, 94),  # Cap manifest workers to avoid OOM
        manifest_dir=str(manifest_cache_dir)
    )

    # 4. DataLoader
    loader = DataLoader(
        adapter,
        batch_size=None,  # CRITICAL: Must be None to return raw uncollated dictionaries
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        prefetch_factor=32 if workers > 0 else None,
        drop_last=False,
        persistent_workers=True if workers > 0 else False,
        multiprocessing_context="fork" if workers > 0 else None,
        worker_init_fn=configure_worker_logger if workers > 0 else None
    )

    # 5. WebDataset ShardWriter
    pattern = str(wds_output_dir / f"{split}-%06d.tar")

    # Max 2GB per tar file for optimal NVMe streaming
    sink = wds.ShardWriter(pattern, maxsize=2e9)

    logger.info(f"[{split}] Extracting {len(adapter)} patches to {wds_output_dir}...")

    for i, sample in enumerate(tqdm(loader, total=len(adapter), desc=f"Extracting {split}")):
        sink.write({
            "__key__": f"{split}_{i:08d}",
            "image.pth": sample["image"].clone(),
            "dtm.pth": sample["dtm"].clone(),
            "confidence.pth": sample["confidence"].clone(),
            "sun_vector.pth": sample["sun_vector"].clone(),
            "intensity.pth": sample["intensity"].clone(),
            "ambient.pth": sample["ambient"].clone()
        })

    sink.close()

    # Write completion marker so future runs skip this config/split safely
    success_marker.touch()
    logger.info(f"[{split}] Finished and verified extraction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_hirise.yaml")
    parser.add_argument("--workers", type=int, default=96)
    # Removed --out argument
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = OmegaConf.load(args.config)

    wds_hash = get_wds_cache_key(config)
    logger.info(f"WebDataset Cache Hash for current config: {wds_hash}")

    for split in ["test", "val", "train"]:
        # Removed out_path from the function call
        build_wds_for_split(config, split, wds_hash, workers=args.workers)
