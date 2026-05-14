"""
Preprocess MarsHiRISEDTM into LitData optimized streaming format.

Uses a two-phase approach to avoid spawn/pickle incompatibility:
  Phase 1: Fork-based DataLoader extracts samples to temp .npz files
           (GDAL/rasterio require fork — spawn can't pickle them)
           *Also recomputes robust OLS sun vectors on-the-fly*
  Phase 2: litdata.optimize repacks .npz files into optimized chunks
           (litdata forces spawn — but .npz reading is trivially picklable)

Phase 3 (optional): Upload to HuggingFace Hub so users can stream via:
    from litdata import StreamingDataset
    ds = StreamingDataset(input_dir="hf://datasets/<repo_id>/train")

Emits the processed training format (image/dtm/confidence/sun_vector/intensity/
ambient/original_image/original_dtm/trend_params/residual_scale/...). For a raw
DTM stream without sun-vector estimation, use `build_litdata_raw.py`.

Usage:
    PYTHONPATH=src uv run python scripts/training/build_litdata.py --config configs/train_hirise.yaml --workers 96
    PYTHONPATH=src uv run python scripts/training/build_litdata.py --config configs/train_hirise.yaml --workers 96 \\
        --hf-repo your-org/mars-hirise-dtm --hf-private
"""
import argparse
import glob
import json
import logging
import os
import shutil
from pathlib import Path

from cache import (
    litdata_cache_key, litdata_cache_root, litdata_tmp_dir,
    write_manifest, )

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

from dataset.core.dtm import MarsHiRISEDTM
from dataset.sampling.sampler import HiRISEGeoSampler
from torchgeo.samplers import Units
from depth_fm.data.adapter import DepthFMHiRISEAdapterCached, estimate_sun_vector_irls
from depth_fm.data.scalers import GlobalLogNormalizer, DEFAULT_ELEV_REF_SCALE

import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)

# ── HuggingFace Hub helpers ────────────────────────────────────────────────

DATASET_CARD_TEMPLATE = """\
---
license: apache-2.0
task_categories:
  - depth-estimation
tags:
  - mars
  - hirise
  - dtm
  - litdata
  - streaming
pretty_name: Mars HiRISE DTM (LitData Streaming)
dataset_info:
  config_name: {cache_hash}
  splits:
    - name: train
    - name: val
    - name: test
---

# Mars HiRISE DTM — LitData Streaming Dataset

Pre-processed Mars HiRISE orthoimage + DTM patches in
[LitData](https://github.com/Lightning-AI/litdata) optimized streaming format.

## Quick Start

```python
from litdata import StreamingDataset

# Stream directly from HuggingFace — no full download needed
train_ds = StreamingDataset(input_dir="hf://datasets/{repo_id}/train")
val_ds   = StreamingDataset(input_dir="hf://datasets/{repo_id}/val")
test_ds  = StreamingDataset(input_dir="hf://datasets/{repo_id}/test")

sample = train_ds[0]

# Core Tensors
print(sample["image"].shape)        # (3, H, W) float16
print(sample["dtm"].shape)          # (3, H, W) float16
print(sample["confidence"].shape)   # (1, H, W) float16

# Physical / Lighting Parameters
print(sample["sun_vector"].shape)   # (3,)      float32
print(sample["intensity"])          # float32
print(sample["ambient"])            # float32

# Original / Unnormalized Data
print(sample["original_image"].shape) # (C, H, W) float16
print(sample["original_dtm"].shape)   # (1, H, W) float16
print(sample["trend_params"].shape)   # (3,)      float16

# Normalization / Processing Metadata
print(sample["residual_scale"])     # float32
print(sample["raw_residual_p98"])   # float32
print(sample["key"])                # str (e.g., "left_red")
```

## Fields

During extraction, model inputs are quantized to `float16` to optimize streaming bandwidth. Physical lighting parameters remain `float32`.

| Key                | Dtype   | Shape      | Description                                    |
|--------------------|---------|------------|------------------------------------------------|
| `image`            | float16 | (3, H, W)  | Normalized HiRISE orthoimage [-1, 1]           |
| `dtm`              | float16 | (3, H, W)  | Normalized DTM elevation                       |
| `confidence`       | float16 | (1, H, W)  | Binary valid data mask (eroded/cleaned)        |
| `sun_vector`       | float32 | (3,)       | Estimated sun direction (OLS, unit-normalized) |
| `intensity`        | float32 | scalar     | Estimated sun intensity                        |
| `ambient`          | float32 | scalar     | Estimated ambient light                        |
| `original_image`   | float16 | (C, H, W)  | Unnormalized resized orthoimage                |
| `original_dtm`     | float16 | (1, H, W)  | Unnormalized resized DTM elevation             |
| `trend_params`     | float16 | (3,)       | LSQR detrend parameters for the DTM plane      |
| `residual_scale`   | float32 | scalar     | Normalization scale applied to the DTM residual|
| `raw_residual_p98` | float32 | scalar     | 98th percentile of the raw topographic residual|
| `key`              | string  | scalar     | Orthoimage source used (e.g., left_red)        |


## Preprocessing Configuration

**Config Hash:** `{cache_hash}`

This dataset was generated with the following pipeline parameters:

```yaml
{config_yaml}
```
"""


def upload_split_to_hf(
        local_dir: str,
        repo_id: str,
        split: str,
        private: bool = False,
):
    """Upload a single split's LitData directory to HuggingFace Hub.

    Uploads to: <repo_id>/<split>/  (e.g. your-org/mars-hirise-dtm/train/)
    Uses the HF Hub API so large files go through LFS automatically.
    """
    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    # Create repo if it doesn't exist (idempotent)
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    logger.info(f"[{split}] Uploading LitData to hf://datasets/{repo_id}/{split} ...")

    api.upload_folder(
        folder_path=local_dir,
        path_in_repo=split,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Update {split} split (LitData optimized chunks)",
        # Don't upload the _SUCCESS marker — it's a local-only signal
        ignore_patterns=["_SUCCESS"],
    )

    logger.info(f"[{split}] Upload complete.")


def upload_dataset_card(repo_id: str, cache_hash: str, config_yaml: str, private: bool = False):
    """Create or update the dataset README card on HuggingFace."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    card_content = DATASET_CARD_TEMPLATE.format(
        repo_id=repo_id,
        cache_hash=cache_hash,
        config_yaml=config_yaml,
    )

    api.upload_file(
        path_or_fileobj=card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update dataset card with config details",
    )
    logger.info(f"Dataset card uploaded to hf://datasets/{repo_id}")


# ── Core preprocessing (unchanged) ────────────────────────────────────────

get_litdata_cache_key = litdata_cache_key  # backwards-compat alias


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

    data = np.load(npz_path, allow_pickle=True)
    return dict(data)


def build_litdata_for_split(
        config,
        split: str,
        cache_hash: str,
        workers: int = 64,
        hf_repo: str | None = None,
        hf_private: bool = False,
):
    """Extract one split into LitData optimized format, optionally upload to HF."""
    hc = config.data.hirise
    sc = config.data.sampler

    dataset_root = Path(hc.root)
    output_dir = str(litdata_cache_root(config) / split)

    # Check for completion marker
    success_marker = Path(output_dir) / "_SUCCESS"
    if success_marker.exists():
        logger.info(f"[{split}] Valid LitData cache found at {output_dir}. Skipping build.")
    else:
        _build_split(config, split, cache_hash, workers, output_dir, success_marker)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3 (optional): Upload to HuggingFace Hub
    # ═══════════════════════════════════════════════════════════════════
    if hf_repo:
        upload_split_to_hf(output_dir, hf_repo, split, private=hf_private)


def _build_split(config, split, cache_hash, workers, output_dir, success_marker):
    """Phases 1 & 2: extract samples and pack into LitData."""
    hc = config.data.hirise
    sc = config.data.sampler
    dataset_root = Path(hc.root)

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

    clip = config.data.get("clip", False)

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
        center_mode=sc.get("center_mode", "simple")
    )

    adapter = DepthFMHiRISEAdapterCached(
        base_dataset=base_dataset,
        sampler=sampler,
        resolution=resolution,
        dtm_normalization=dtm_norm,
        random_flip=False,
        brightness_jitter=0.0,
        stats_path=stats_path,
        clip=clip,
        use_manifest=True,
        manifest_workers=min(workers, 94),
        manifest_dir=str(manifest_cache_dir),
    )

    logger.info("Number dataset %d, number samples %d, num filtered %d", len(base_dataset), len(sampler), len(adapter))

    num_samples = len(adapter)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Extract with fork-based DataLoader → temp .npz files
    # ═══════════════════════════════════════════════════════════════════

    tmp_dir = litdata_tmp_dir(config, split)
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
        multiprocessing_context="spawn" if workers > 0 else None,
        worker_init_fn=_configure_worker_logger if workers > 0 else None,
    )

    logger.info(f"[{split}] Phase 1/2: Extracting {num_samples} samples to temp .npz files...")
    npz_paths = []

    path = str(tmp_dir / "*.npz")

    paths = glob.glob(path)

    if len(paths) > 0:
        logger.info(f"Able to see {len(paths)} already prebuilt paths")
    else:
        logger.info(f"No prebuilt npzs inside {path}")

    if len(paths) == num_samples:
        logger.info(f"There is the correct number of prebuilt paths, skipping npz generation")
        npz_paths = paths

    else:
        for i, sample in enumerate(tqdm(loader, total=num_samples, desc=f"Extract {split}")):
            image_fp32 = sample["image"].float()
            dtm_fp32 = sample["dtm"].float()
            conf_fp32 = sample["confidence"].float()

            ref_scale = float(sample.get("residual_scale", DEFAULT_ELEV_REF_SCALE))
            _norm = GlobalLogNormalizer(ref_scale)
            dtm_normalised = dtm_fp32[:1]
            physical_residual = _norm.denormalize_prediction(dtm_normalised)
            sun_vec, intensity, ambient = estimate_sun_vector_irls(physical_residual, image_fp32, conf_fp32)
            sun_vec = torch.nn.functional.normalize(sun_vec, p=2, dim=0)

            npz_path = str(tmp_dir / f"{i:08d}.npz")
            np.savez(
                npz_path,
                image=sample["image"].numpy().astype(np.float16),
                original_dtm=sample["original_dtm"].numpy().astype(np.float16),
                trend_params=sample["trend_params"].numpy().astype(np.float16),
                original_image=sample["original_image"].numpy().astype(np.float16),
                meta=json.dumps(sample["meta"]),
                dtm=sample["dtm"].numpy().astype(np.float16),
                confidence=sample["confidence"].numpy().astype(np.float16),
                sun_vector=sun_vec.numpy(),
                intensity=intensity.numpy(),
                ambient=ambient.numpy(),
                residual_scale=np.array(ref_scale, dtype=np.float32),
                raw_residual_p98=np.array(
                    float(sample.get("raw_residual_p98", 0.0)), dtype=np.float32),
            )
            npz_paths.append(npz_path)

    # Explicitly shut down persistent DataLoader workers before they linger into
    # Phase 2 (otherwise they pile up across splits and prevent process exit).
    try:
        if getattr(loader, "_iterator", None) is not None:
            loader._iterator._shutdown_workers()
    except Exception:
        pass
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

    success_marker.touch()

    from omegaconf import OmegaConf
    write_manifest(
        Path(output_dir),
        cache_hash=cache_hash,
        config_snapshot=OmegaConf.to_container(config, resolve=True),
    )
    logger.info(f"[{split}] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_hirise.yaml")
    parser.add_argument("--workers", type=int, default=96 * 2)
    parser.add_argument(
        "--hf-repo", type=str, default="SuperComputer/mars_hirise_dtm_processed",
        help="Base HuggingFace repo id to upload to (e.g. your-org/mars-hirise-dtm). "
             "The config hash will be automatically appended.",
    )
    parser.add_argument(
        "--hf-private", action="store_true",
        help="Create the HF dataset repo as private.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = OmegaConf.load(args.config)

    cache_hash = litdata_cache_key(config)
    logger.info(f"LitData Cache Hash: {cache_hash}")
    logger.info(f"LitData Cache Root: {litdata_cache_root(config)}")

    # Automatically append the hash to the repo name
    final_repo_id = f"{args.hf_repo}-{cache_hash}" if args.hf_repo else None

    for split in ["test", "val", "train"]:
        build_litdata_for_split(
            config, split, cache_hash,
            workers=args.workers,
            hf_repo=final_repo_id,
            hf_private=args.hf_private,
        )

    # Upload dataset card after all splits are done
    if final_repo_id:
        # Extract the exact config parts used for the hash and format as YAML
        relevant_config = {
            "data": OmegaConf.to_container(config.data, resolve=True),
        }
        config_yaml_str = OmegaConf.to_yaml(relevant_config)

        upload_dataset_card(
            repo_id=final_repo_id,
            cache_hash=cache_hash,
            config_yaml=config_yaml_str,
            private=args.hf_private
        )

        logger.info(
            f"\nDataset ready! Users can load it with:\n"
            f'  from litdata import StreamingDataset\n'
            f'  ds = StreamingDataset(input_dir="hf://datasets/{final_repo_id}/train")\n'
        )

    # Need to manually terminate the program to ensure that the system does not hang.
    # LitData's optimize() and the persistent spawn-based DataLoader workers leave
    # lingering child processes whose atexit handlers block a clean sys.exit. Force
    # immediate termination — all success markers and manifests are already written.
    import gc

    gc.collect()
    logging.shutdown()
    os._exit(0)
