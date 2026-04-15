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

Usage:
    python build_litdata.py --config configs/train_hirise.yaml --workers 96
    python build_litdata.py --config configs/train_hirise.yaml --workers 96 \\
        --hf-repo your-org/mars-hirise-dtm --hf-private
    PYTHONPATH=src uv run -m src.depth_fm.build_litdata_raw --config configs/train_hirise.yaml
"""
import argparse
import glob
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import torch

# GDAL / threading optimizations for the extraction phase
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["VSI_CACHE"] = "TRUE"
os.environ["VSI_CACHE_SIZE"] = "200000000"
os.environ["GDAL_NUM_THREADS"] = "1"
os.environ["GDAL_MAX_DATASET_POOL_SIZE"] = "1024"
os.environ["OMP_NUM_THREADS"] = "2"

import rasterio
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from litdata import optimize

from dataset.mars_hirise_dtm import MarsHiRISEDTM
from dataset.hirise_sampler import HiRISEGeoSampler
from torchgeo.samplers import Units

import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)

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
print(sample["elevation"].shape)    # (1, H, W) float32
print(sample["bounds"].shape)       # (4,)      float32
print(sample["crs"])                # str

# Orthoimages (presence depends on config)
if "left_red" in sample:
    print(sample["left_red"].shape) # (1, H, W) float32
if "left_irb" in sample:
    print(sample["left_irb"].shape) # (3, H, W) float32

# Flattened Metadata
print(sample["dtm_product_id"])     # str
print(sample["left_obs_id"])        # str
print(sample["incidence_angle"])    # float32
```

## Preprocessing Configuration

**Config Hash:** `{cache_hash}`

This dataset was generated with the following pipeline parameters:

```yaml
{config_yaml}
```

*(Note: Depending on the dataset configuration, some orthoimage keys may be omitted).*
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

def get_litdata_cache_key(config) -> str:
    """Deterministic hash for the current preprocessing configuration."""
    key_parts = {
        "hirise": OmegaConf.to_container(config.data.hirise, resolve=True),
        "sampler": OmegaConf.to_container(config.data.sampler, resolve=True),
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
    data = np.load(npz_path, allow_pickle=True)
    out_dict = dict(data)

    # Decode the JSON payload back into a list of dicts
    if "meta" in out_dict:
        meta_val = out_dict["meta"]
        # Extract string from 0-d numpy array if necessary
        if isinstance(meta_val, np.ndarray):
            meta_val = meta_val.item()

        # Reconstruct the native Python list of dicts for LitData
        out_dict["meta"] = json.loads(meta_val)

    return out_dict


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
    output_dir = str(dataset_root / f"litdata_cache_{cache_hash}" / split)

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
    mp.set_sharing_strategy('file_system')

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

    num_samples = len(sampler)

    tmp_dir = dataset_root / f"_litdata_tmp_{cache_hash}_{split}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    effective_workers = min(workers, max(1, num_samples // 4))

    loader = DataLoader(
        base_dataset,
        sampler=sampler,
        batch_size=None,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=False,
        prefetch_factor=4 if workers > 0 else None,
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
            npz_path = str(tmp_dir / f"{i:08d}.npz")

            data = {}

            def check(sample_v):
                for key, value in sample_v.items():
                    if isinstance(value, torch.Tensor):
                        data[key] = value.float().numpy()
                    elif isinstance(value, dict) or isinstance(value, list):
                        data[key] = json.dumps(value)
                    else:
                        data[key] = value

            check(sample)

            np.savez(
                npz_path,
                **data
            )
            npz_paths.append(npz_path)

    del loader, sampler, base_dataset
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
        compression="zstd"
    )

    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info(f"[{split}] Cleaned up temp directory.")

    success_marker.touch()
    logger.info(f"[{split}] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_hirise.yaml")
    parser.add_argument("--workers", type=int, default=96 * 2)
    parser.add_argument(
        "--hf-repo", type=str, default="SuperComputer/mars_hirise_dtm_raw",
        help="HuggingFace repo id to upload to (e.g. your-org/mars-hirise-dtm). "
             "Requires `huggingface-cli login` or HF_TOKEN env var.",
    )
    parser.add_argument(
        "--hf-private", action="store_true",
        help="Create the HF dataset repo as private.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = OmegaConf.load(args.config)

    cache_hash = get_litdata_cache_key(config)
    logger.info(f"LitData Cache Hash: {cache_hash}")

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
            "hirise": OmegaConf.to_container(config.data.hirise, resolve=True),
            "sampler": OmegaConf.to_container(config.data.sampler, resolve=True),
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
