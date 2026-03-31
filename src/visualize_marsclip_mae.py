"""Preview utilities for Stage A MAE masked/reconstructed MarsCLIP patches."""

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

from marsclip_mae import (
    MarsMAEOutput,
    MarsMaskedAutoencoder,
    build_masked_input_image,
    build_reconstruction_composite,
    collate_patch_samples_for_mae,
    expand_patch_mask,
)
from marsclip_patches import DEFAULT_PATCH_VALID_FRACTION, MarsCLIPPatchDataset
from visualize_marsclip import _to_display_rgb


def _build_patch_state_overlay(
    image: torch.Tensor,
    patch_valid_mask: torch.Tensor,
    visible_mask: torch.Tensor,
    *,
    patch_size: int,
) -> np.ndarray:
    """Render patch states: visible=green, masked-valid=orange, invalid=red."""
    rgb = _to_display_rgb(image)
    overlay = np.repeat(rgb.mean(axis=2, keepdims=True), 3, axis=2) * 0.65

    image_size = (image.shape[-2], image.shape[-1])
    visible = expand_patch_mask(
        visible_mask.unsqueeze(0),
        patch_size,
        image_size=image_size,
    )[0].detach().cpu().numpy()
    masked_valid = expand_patch_mask(
        (patch_valid_mask & ~visible_mask).unsqueeze(0),
        patch_size,
        image_size=image_size,
    )[0].detach().cpu().numpy()
    invalid = expand_patch_mask(
        (~patch_valid_mask).unsqueeze(0),
        patch_size,
        image_size=image_size,
    )[0].detach().cpu().numpy()

    overlay[visible.astype(bool)] = np.clip(
        0.45 * overlay[visible.astype(bool)] + 0.55 * np.array([0.20, 0.85, 0.25]),
        0.0,
        1.0,
    )
    overlay[masked_valid.astype(bool)] = np.array([0.98, 0.64, 0.16])
    overlay[invalid.astype(bool)] = np.array([0.90, 0.16, 0.16])
    return overlay


def save_mae_reconstruction_preview(
    samples: list[dict[str, Any]],
    output: MarsMAEOutput,
    out_path: pathlib.Path | str,
    *,
    patch_size: int,
    max_items: int = 4,
) -> pathlib.Path:
    """Save original/masked/reconstructed MAE panels for a small batch."""
    items = min(max_items, len(samples), output.reconstruction.shape[0])
    if items <= 0:
        raise ValueError("No samples provided.")

    images = torch.stack([sample["image"] for sample in samples[:items]], dim=0)
    masked_inputs = build_masked_input_image(
        images,
        output.patch_valid_mask[:items],
        output.visible_mask[:items],
        patch_size=patch_size,
    )
    _, composites = build_reconstruction_composite(
        images,
        output.reconstruction[:items],
        output.patch_valid_mask[:items],
        output.visible_mask[:items],
        patch_size=patch_size,
    )

    fig, axes = plt.subplots(items, 4, figsize=(16, 3.8 * items))
    if items == 1:
        axes = np.array([axes])

    for idx in range(items):
        sample = samples[idx]
        md = sample["metadata"]
        row_axes = axes[idx]
        row_axes[0].imshow(_to_display_rgb(images[idx]), interpolation="nearest")
        row_axes[0].set_title(f"input\n{md['obs_id']}", fontsize=9)
        row_axes[1].imshow(_to_display_rgb(masked_inputs[idx]), interpolation="nearest")
        row_axes[1].set_title("masked encoder input", fontsize=9)
        row_axes[2].imshow(_to_display_rgb(composites[idx]), interpolation="nearest")
        row_axes[2].set_title(
            f"reconstruction composite\nloss={float(output.loss):.4f}",
            fontsize=9,
        )
        overlay = _build_patch_state_overlay(
            images[idx],
            output.patch_valid_mask[idx],
            output.visible_mask[idx],
            patch_size=patch_size,
        )
        row_axes[3].imshow(overlay, interpolation="nearest")
        row_axes[3].set_title("patch states", fontsize=9)

        for ax in row_axes:
            ax.axis("off")

    fig.suptitle("MarsCLIP Stage A MAE preview", fontsize=12)
    fig.tight_layout()
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Preview Stage A MAE reconstructions.")
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("/scratch/mars_hirise"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("marsclip_mae_preview.png"))
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=(-136.0, 12.0, -124.0, 24.0),
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
    )
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size-deg", type=float, default=0.005)
    parser.add_argument("--patch-size-px", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--max-patches", type=int, default=8)
    parser.add_argument("--encoder-dim", type=int, default=64)
    parser.add_argument("--encoder-depth", type=int, default=2)
    parser.add_argument("--encoder-heads", type=int, default=4)
    parser.add_argument("--decoder-dim", type=int, default=32)
    parser.add_argument("--decoder-depth", type=int, default=1)
    parser.add_argument("--decoder-heads", type=int, default=4)
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=DEFAULT_PATCH_VALID_FRACTION,
    )
    args = parser.parse_args()

    ds = MarsCLIPPatchDataset(
        root=args.root,
        bbox=tuple(args.bbox),
        patch_size=args.patch_size_deg,
        image_size=args.image_size,
        max_patches=args.max_patches,
        min_valid_fraction=args.min_valid_fraction,
    )
    n = min(args.num_samples, len(ds))
    samples = [ds[i] for i in range(n)]
    batch = collate_patch_samples_for_mae(samples)

    model = MarsMaskedAutoencoder(
        image_size=args.image_size,
        patch_size=args.patch_size_px,
        encoder_dim=args.encoder_dim,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
        min_valid_fraction=args.min_valid_fraction,
    )
    model.eval()
    with torch.no_grad():
        output = model.forward_patch_batch(
            batch,
            mask_ratio=args.mask_ratio,
            generator=torch.Generator().manual_seed(0),
        )

    out = save_mae_reconstruction_preview(
        samples,
        output,
        args.out,
        patch_size=args.patch_size_px,
        max_items=n,
    )
    print(f"Saved MAE preview to {out}")


if __name__ == "__main__":
    main()
