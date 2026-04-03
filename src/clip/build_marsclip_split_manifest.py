"""Build a reusable MarsCLIP patch split manifest."""

from __future__ import annotations

import argparse
import pathlib
import sys

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from clip.marsclip_patches import (
    DEFAULT_PATCH_VALID_FRACTION,
    MarsCLIPPatchDataset,
    save_patch_records,
)
from clip.marsclip_splits import (
    build_patch_split_manifest,
    save_patch_split_manifest,
    save_split_summary,
    summarize_patch_split_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reusable MarsCLIP patch split manifest.")
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
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument(
        "--color-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only full three-band COLOR observations/patches in the manifest.",
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=DEFAULT_PATCH_VALID_FRACTION,
    )
    parser.add_argument(
        "--out-manifest",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--out-summary",
        type=pathlib.Path,
        default=None,
    )
    parser.add_argument(
        "--out-patch-records",
        type=pathlib.Path,
        default=None,
        help="Optional cache path for the full patch-record table used to build the manifest.",
    )
    args = parser.parse_args()

    dataset = MarsCLIPPatchDataset(
        root=args.root,
        bbox=tuple(args.bbox),
        patch_size=args.patch_size_deg,
        image_size=args.image_size,
        max_patches=args.max_patches,
        min_valid_fraction=args.min_valid_fraction,
        color_only=args.color_only,
    )
    manifest = build_patch_split_manifest(
        dataset.patch_records,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        num_folds=args.num_folds,
    )
    manifest_path = save_patch_split_manifest(manifest, args.out_manifest)
    patch_records_path = None
    if args.out_patch_records is not None:
        patch_records_path = save_patch_records(dataset.patch_records, args.out_patch_records)

    summary = summarize_patch_split_manifest(manifest, num_folds=args.num_folds)
    summary.update(
        {
            "bbox": list(args.bbox),
            "patch_size_deg": args.patch_size_deg,
            "image_size": args.image_size,
            "max_patches": args.max_patches,
            "color_only": bool(args.color_only),
            "split_seed": args.split_seed,
            "manifest_path": str(manifest_path),
            "patch_records_path": str(patch_records_path) if patch_records_path is not None else None,
        }
    )
    summary_path = args.out_summary
    if summary_path is None:
        summary_path = manifest_path.with_suffix(".summary.json")
    save_split_summary(summary, summary_path)

    print(f"Saved split manifest to {manifest_path}")
    if patch_records_path is not None:
        print(f"Saved patch records to {patch_records_path}")
    print(f"Saved split summary to {summary_path}")


if __name__ == "__main__":
    main()
