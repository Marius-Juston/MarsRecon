"""Build LitData-backed MarsCLIP train/val/test datasets for faster SatMAE training."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import shutil
import sys
import time
from typing import Any

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("VSI_CACHE_SIZE", "200000000")
os.environ.setdefault("GDAL_NUM_THREADS", "1")
os.environ.setdefault("GDAL_MAX_DATASET_POOL_SIZE", "1024")

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from clip.marsclip_litdata import get_marsclip_litdata_cache_key
from clip.marsclip_patches import (
    DEFAULT_PATCH_VALID_FRACTION,
    MarsCLIPPatchDataset,
    load_patch_records,
)
from clip.marsclip_splits import (
    align_manifest_to_patch_records,
    build_dataset_subsets,
    load_patch_split_manifest,
)

logger = logging.getLogger(__name__)

_SUCCESS_FILENAME = "_SUCCESS"
_PROGRESS_FILENAME = "litdata_progress.json"
_INFO_FILENAME = "litdata_info.json"


def _load_litdata_optimize():
    try:
        from litdata import optimize
    except ImportError as exc:  # pragma: no cover - runtime environment path
        raise ImportError(
            "litdata is not installed. Install the `streaming` extra or run "
            "`.venv/bin/pip install litdata>=0.2.61`."
        ) from exc
    return optimize


def _configure_worker_logger(worker_id: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f"[LitData Worker {worker_id}] %(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )


def _repack_npz(npz_path: str) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=False)
    metadata_json = data["metadata_json"]
    if isinstance(metadata_json, np.ndarray):
        metadata_json = metadata_json.item()
    if isinstance(metadata_json, bytes):
        metadata_json = metadata_json.decode("utf-8")
    return {
        "image": data["image"],
        "valid_mask": data["valid_mask"],
        "metadata_json": str(metadata_json),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _json_safe(value.item())
        return [_json_safe(item) for item in value.detach().cpu().tolist()]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_progress(
    out_dir: pathlib.Path,
    *,
    split_name: str,
    status: str,
    source_count: int,
    extracted_count: int,
    kept_count: int,
    dropped_count: int,
) -> None:
    payload = {
        "split_name": split_name,
        "status": status,
        "source_count": int(source_count),
        "extracted_count": int(extracted_count),
        "kept_count": int(kept_count),
        "dropped_count": int(dropped_count),
        "percent_complete": float(extracted_count) / float(source_count) if source_count else 0.0,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / _PROGRESS_FILENAME).write_text(json.dumps(payload, indent=2))


def build_litdata_split(
    dataset,
    *,
    out_dir: pathlib.Path,
    split_name: str,
    workers: int,
    chunk_bytes: str,
    progress_interval: int,
    require_patch_valid: bool,
) -> pathlib.Path:
    success_marker = out_dir / _SUCCESS_FILENAME
    if success_marker.exists():
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir.parent / f"_{split_name}_npz_tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    source_count = len(dataset)
    _write_progress(
        out_dir,
        split_name=split_name,
        status="extracting",
        source_count=source_count,
        extracted_count=0,
        kept_count=0,
        dropped_count=0,
    )

    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=int(workers),
        pin_memory=False,
        prefetch_factor=2 if int(workers) > 0 else None,
        drop_last=False,
        persistent_workers=bool(int(workers) > 0),
        multiprocessing_context="fork" if int(workers) > 0 else None,
        worker_init_fn=_configure_worker_logger if int(workers) > 0 else None,
    )

    kept_count = 0
    dropped_count = 0
    npz_paths: list[str] = []
    progress_every = max(1, int(progress_interval))

    for sample_index, sample in enumerate(loader, start=1):
        metadata = _json_safe(dict(sample.get("metadata", {})))
        keep_sample = bool(metadata.get("is_patch_valid", True)) or not require_patch_valid
        if keep_sample:
            npz_path = tmp_dir / f"{kept_count:08d}.npz"
            np.savez(
                npz_path,
                image=sample["image"].detach().cpu().numpy().astype(np.float16, copy=False),
                valid_mask=sample["valid_mask"].detach().cpu().numpy().astype(np.bool_, copy=False),
                metadata_json=json.dumps(metadata),
            )
            npz_paths.append(str(npz_path))
            kept_count += 1
        else:
            dropped_count += 1

        if sample_index % progress_every == 0 or sample_index == source_count:
            _write_progress(
                out_dir,
                split_name=split_name,
                status="extracting",
                source_count=source_count,
                extracted_count=sample_index,
                kept_count=kept_count,
                dropped_count=dropped_count,
            )

    optimize = _load_litdata_optimize()
    optimizer_cache_dir = out_dir / "_optimizer_cache"
    optimizer_data_dir = out_dir / "_optimizer_data"
    optimizer_cache_dir.mkdir(parents=True, exist_ok=True)
    optimizer_data_dir.mkdir(parents=True, exist_ok=True)
    _write_progress(
        out_dir,
        split_name=split_name,
        status="packing",
        source_count=source_count,
        extracted_count=source_count,
        kept_count=kept_count,
        dropped_count=dropped_count,
    )
    previous_cache_dir = os.environ.get("DATA_OPTIMIZER_CACHE_FOLDER")
    previous_data_cache_dir = os.environ.get("DATA_OPTIMIZER_DATA_CACHE_FOLDER")
    os.environ["DATA_OPTIMIZER_CACHE_FOLDER"] = str(optimizer_cache_dir)
    os.environ["DATA_OPTIMIZER_DATA_CACHE_FOLDER"] = str(optimizer_data_dir)
    try:
        optimize(
            fn=_repack_npz,
            inputs=npz_paths,
            output_dir=str(out_dir),
            num_workers=max(1, min(int(workers), 32)),
            chunk_bytes=chunk_bytes,
        )
    finally:
        if previous_cache_dir is None:
            os.environ.pop("DATA_OPTIMIZER_CACHE_FOLDER", None)
        else:
            os.environ["DATA_OPTIMIZER_CACHE_FOLDER"] = previous_cache_dir
        if previous_data_cache_dir is None:
            os.environ.pop("DATA_OPTIMIZER_DATA_CACHE_FOLDER", None)
        else:
            os.environ["DATA_OPTIMIZER_DATA_CACHE_FOLDER"] = previous_data_cache_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    info = {
        "split_name": split_name,
        "source_count": int(source_count),
        "num_samples": int(kept_count),
        "num_dropped_samples": int(dropped_count),
        "require_patch_valid": bool(require_patch_valid),
        "chunk_bytes": chunk_bytes,
        "workers": int(workers),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / _INFO_FILENAME).write_text(json.dumps(info, indent=2))
    _write_progress(
        out_dir,
        split_name=split_name,
        status="completed",
        source_count=source_count,
        extracted_count=source_count,
        kept_count=kept_count,
        dropped_count=dropped_count,
    )
    success_marker.touch()
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build LitData-backed MarsCLIP split datasets from a reusable manifest."
    )
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("/scratch/mars_hirise"))
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=(-136.0, 12.0, -124.0, 24.0),
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
    )
    parser.add_argument("--patch-size-deg", type=float, default=0.005)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--split-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--patch-records-path", type=pathlib.Path, required=True)
    parser.add_argument("--out-root", type=pathlib.Path, required=True)
    parser.add_argument("--split-mode", type=str, default="holdout", choices=("holdout", "kfold"))
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--color-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-valid-fraction", type=float, default=DEFAULT_PATCH_VALID_FRACTION)
    parser.add_argument("--dataset-normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dataset-normalization-path", type=pathlib.Path, default=None)
    parser.add_argument("--filter-invalid-patches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dominant-obs-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--required-splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val"),
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--chunk-bytes", type=str, default="256MB")
    parser.add_argument("--progress-interval", type=int, default=250)
    parser.add_argument("--max-patches", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    patch_records = load_patch_records(args.patch_records_path)
    required_splits = tuple(args.required_splits)
    cache_key = get_marsclip_litdata_cache_key(
        root=str(args.root),
        bbox=tuple(args.bbox),
        patch_size_deg=float(args.patch_size_deg),
        image_size=int(args.image_size),
        split_manifest=str(args.split_manifest),
        patch_records_path=str(args.patch_records_path),
        color_only=bool(args.color_only),
        dataset_normalize=bool(args.dataset_normalize),
        dataset_normalization_path=(
            str(args.dataset_normalization_path) if args.dataset_normalization_path is not None else None
        ),
        min_valid_fraction=float(args.min_valid_fraction),
        dominant_obs_only=bool(args.dominant_obs_only),
        filter_invalid_patches=bool(args.filter_invalid_patches),
        required_splits=required_splits,
    )
    out_root = args.out_root.expanduser().resolve() / cache_key
    manifest = load_patch_split_manifest(args.split_manifest)

    dataset = MarsCLIPPatchDataset(
        root=args.root,
        bbox=tuple(args.bbox),
        patch_size=args.patch_size_deg,
        image_size=args.image_size,
        max_patches=args.max_patches,
        min_valid_fraction=args.min_valid_fraction,
        color_only=args.color_only,
        patch_records=patch_records,
        dataset_normalize=args.dataset_normalize,
        dataset_normalization_path=args.dataset_normalization_path,
        use_dominant_obs_only=args.dominant_obs_only,
    )
    if args.max_patches is not None:
        manifest = align_manifest_to_patch_records(manifest, dataset.patch_records)
    train_dataset, val_dataset, test_dataset = build_dataset_subsets(
        dataset,
        dataset.patch_records,
        manifest,
        mode=args.split_mode,
        fold_index=args.fold_index,
    )
    split_lookup = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }

    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "cache_key": cache_key,
        "root": str(args.root),
        "bbox": list(args.bbox),
        "patch_size_deg": float(args.patch_size_deg),
        "image_size": int(args.image_size),
        "split_manifest": str(args.split_manifest),
        "patch_records_path": str(args.patch_records_path),
        "color_only": bool(args.color_only),
        "dataset_normalize": bool(args.dataset_normalize),
        "dataset_normalization_path": (
            str(args.dataset_normalization_path) if args.dataset_normalization_path is not None else None
        ),
        "filter_invalid_patches": bool(args.filter_invalid_patches),
        "dominant_obs_only": bool(args.dominant_obs_only),
        "required_splits": list(required_splits),
        "workers": int(args.workers),
        "chunk_bytes": args.chunk_bytes,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for split_name in required_splits:
        logger.info("Building LitData split %s at %s", split_name, out_root / split_name)
        split_dataset = split_lookup[split_name]
        build_litdata_split(
            split_dataset,
            out_dir=out_root / split_name,
            split_name=split_name,
            workers=args.workers,
            chunk_bytes=args.chunk_bytes,
            progress_interval=args.progress_interval,
            require_patch_valid=args.filter_invalid_patches,
        )
        info = json.loads((out_root / split_name / _INFO_FILENAME).read_text())
        summary[f"{split_name}_count"] = int(info["num_samples"])
        summary[f"{split_name}_dropped_count"] = int(info["num_dropped_samples"])

    (out_root / "litdata_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved Mars LitData cache to {out_root}")


if __name__ == "__main__":
    main()
