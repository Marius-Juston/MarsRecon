"""Build cache-backed MarsCLIP train/val/test datasets for faster Stage A training."""

from __future__ import annotations

import argparse
import pathlib
import sys

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from clip.marsclip_cache import build_cached_splits_from_manifest, cache_root_complete
from clip.marsclip_patches import (
    DEFAULT_PATCH_VALID_FRACTION,
    MarsCLIPPatchDataset,
    load_patch_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cache-backed MarsCLIP split datasets from a reusable manifest."
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
    parser.add_argument("--out-root", type=pathlib.Path, required=True)
    parser.add_argument("--split-mode", type=str, default="holdout", choices=("holdout", "kfold"))
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--color-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-valid-fraction", type=float, default=DEFAULT_PATCH_VALID_FRACTION)
    parser.add_argument("--patch-records-path", type=pathlib.Path, default=None)
    parser.add_argument("--dataset-normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dataset-normalization-path", type=pathlib.Path, default=None)
    parser.add_argument("--filter-invalid-patches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dominant-obs-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--required-splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--progress-interval", type=int, default=250)
    parser.add_argument("--max-patches", type=int, default=None)
    args = parser.parse_args()

    required_splits = tuple(args.required_splits)
    if cache_root_complete(args.out_root, required_splits=required_splits):
        print(f"Cache already complete at {args.out_root}")
        return

    patch_records = None
    if args.patch_records_path is not None and args.patch_records_path.exists():
        patch_records = load_patch_records(args.patch_records_path)

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
    source_info = {
        "root": str(args.root),
        "bbox": list(args.bbox),
        "patch_size_deg": float(args.patch_size_deg),
        "image_size": int(args.image_size),
        "color_only": bool(args.color_only),
        "dataset_normalize": bool(args.dataset_normalize),
        "filter_invalid_patches": bool(args.filter_invalid_patches),
        "dominant_obs_only": bool(args.dominant_obs_only),
        "required_splits": list(required_splits),
        "dataset_normalization_path": str(args.dataset_normalization_path)
        if args.dataset_normalization_path is not None
        else None,
        "patch_records_path": str(args.patch_records_path) if args.patch_records_path is not None else None,
    }
    build_cached_splits_from_manifest(
        dataset,
        dataset.patch_records,
        manifest_path=args.split_manifest,
        out_root=args.out_root,
        mode=args.split_mode,
        fold_index=args.fold_index,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        progress_interval=args.progress_interval,
        require_patch_valid=args.filter_invalid_patches,
        required_splits=required_splits,
        source_info=source_info,
    )
    print(f"Saved cache-backed splits to {args.out_root}")


if __name__ == "__main__":
    main()
