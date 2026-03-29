"""Preview helpers for MarsCLIP samples and batches."""

from __future__ import annotations

import argparse
import os
import pathlib
import tempfile
from typing import Any

_MPL_CACHE = pathlib.Path(tempfile.gettempdir()) / "marsrecon-mpl"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib as mpl
import numpy as np
import torch

mpl.use("Agg")
from matplotlib import pyplot as plt

from marsclip_dataset import MarsCLIPDataset
from marsclip_patches import MarsCLIPPatchDataset


def _to_display_rgb(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().numpy()
    if arr.ndim == 3:
        disp = arr.transpose(1, 2, 0)
    else:
        raise ValueError("Expected CHW image tensor.")
    return np.clip(disp, 0.0, 1.0)


def save_sample_preview(
    samples: list[dict[str, Any]],
    out_path: pathlib.Path | str,
    *,
    max_items: int = 4,
) -> pathlib.Path:
    """Save a quick-look preview of MarsCLIP samples."""
    items = samples[:max_items]
    if not items:
        raise ValueError("No samples provided.")

    rows = len(items)
    fig, axes = plt.subplots(rows, 2, figsize=(10, 3.8 * rows))
    if rows == 1:
        axes = np.array([axes])

    for row_idx, sample in enumerate(items):
        ax_img, ax_mask = axes[row_idx]
        img = _to_display_rgb(sample["image"])
        valid_mask = sample["valid_mask"].detach().cpu().numpy()

        ax_img.imshow(img, interpolation="nearest")
        ax_img.axis("off")

        md = sample["metadata"]
        title = (
            f"{md['obs_id']}  valid={md['overall_valid_fraction']:.0%}\n"
            f"{sample['rationale_raw'][:70]}"
        )
        if sample.get("rationale_expanded"):
            title += "\nexpanded"
        ax_img.set_title(title, fontsize=9)

        ax_mask.imshow(valid_mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax_mask.axis("off")
        ax_mask.set_title("valid_mask", fontsize=9)

    fig.suptitle("MarsCLIP sample preview", fontsize=12)
    fig.tight_layout()
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview MarsCLIP observation samples.")
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("/scratch/mars_hirise"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("marsclip_preview.png"))
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--dataset-kind",
        choices=("observation", "patch"),
        default="observation",
        help="Preview observation-level thumbnails or patch-level Stage A samples.",
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
    parser.add_argument("--min-valid-fraction", type=float, default=0.5)
    args = parser.parse_args()

    if args.dataset_kind == "patch":
        ds = MarsCLIPPatchDataset(
            root=args.root,
            bbox=tuple(args.bbox),
            image_size=args.image_size,
            patch_size=args.patch_size,
            stride=args.stride,
            max_patches=args.max_patches,
            min_valid_fraction=args.min_valid_fraction,
        )
    else:
        ds = MarsCLIPDataset(
            root=args.root,
            bbox=tuple(args.bbox),
            image_size=args.image_size,
        )
    n = min(args.num_samples, len(ds))
    samples = [ds[i] for i in range(n)]
    out = save_sample_preview(samples, args.out, max_items=n)
    print(f"Saved preview to {out}")


if __name__ == "__main__":
    main()
