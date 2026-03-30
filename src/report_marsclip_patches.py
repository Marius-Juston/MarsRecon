"""Generate lightweight Stage A1 patch reports and preview artifacts."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import tempfile
from typing import Any

_MPL_CACHE = pathlib.Path(tempfile.gettempdir()) / "marsrecon-mpl"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

from marsclip_patches import (
    DEFAULT_PATCH_VALID_FRACTION,
    MarsCLIPPatchDataset,
    summarize_patch_records,
    summarize_patch_samples,
)
from visualize_marsclip import save_sample_preview


def save_patch_report(
    dataset: MarsCLIPPatchDataset,
    out_dir: pathlib.Path | str,
    *,
    preview_items: int = 4,
    report_items: int = 16,
) -> dict[str, Any]:
    """Save a JSON patch summary and PNG preview for a Stage A1 dataset."""
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    count = min(len(dataset), max(preview_items, report_items))
    if count <= 0:
        raise ValueError("Patch dataset is empty.")

    samples = [dataset[i] for i in range(count)]
    summary = summarize_patch_records(dataset.patch_records)
    summary.update(summarize_patch_samples(samples[: min(report_items, len(samples))]))
    summary["image_size"] = list(dataset.image_size)
    summary["patch_size_deg"] = list(dataset.patch_size)

    preview_path = save_sample_preview(
        samples,
        out_path / "patch_preview.png",
        max_items=min(preview_items, len(samples)),
    )
    summary_path = out_path / "patch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    return {
        "summary": summary,
        "summary_path": summary_path,
        "preview_path": preview_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Stage A1 patch summary report.")
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("/scratch/mars_hirise"))
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("marsclip_patch_report"),
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=(-136.0, 12.0, -124.0, 24.0),
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
    )
    parser.add_argument("--patch-size", type=float, default=0.005)
    parser.add_argument("--stride", type=float, default=None)
    parser.add_argument("--max-patches", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--preview-items", type=int, default=4)
    parser.add_argument("--report-items", type=int, default=16)
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=DEFAULT_PATCH_VALID_FRACTION,
        help="Patch-level valid-pixel threshold used to label a sample as usable.",
    )
    args = parser.parse_args()

    dataset = MarsCLIPPatchDataset(
        root=args.root,
        bbox=tuple(args.bbox),
        patch_size=args.patch_size,
        stride=args.stride,
        image_size=args.image_size,
        max_patches=args.max_patches,
        min_valid_fraction=args.min_valid_fraction,
    )
    outputs = save_patch_report(
        dataset,
        args.out_dir,
        preview_items=args.preview_items,
        report_items=args.report_items,
    )
    print(f"Saved Stage A1 patch preview to {outputs['preview_path']}")
    print(f"Saved Stage A1 patch summary to {outputs['summary_path']}")


if __name__ == "__main__":
    main()
