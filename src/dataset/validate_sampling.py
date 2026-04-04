#!/usr/bin/env python
"""Validate MarsHiRISE strip geometry, GeoSampler coverage, and pixel calibration.

Generates diagnostic PNG plots saved to disk — designed for headless servers
accessed via SSH.  Every figure is self-contained: no interactive display
windows are opened.

Run from the project root::

    # Quick validation (default Olympus Mons bbox, 16 thumbnails)
    uv run python validate_sampling.py --root /scratch/mars_hirise

    # Full validation with custom region
    uv run python validate_sampling.py \\
        --root /scratch/mars_hirise \\
        --bbox -136 12 -124 24 \\
        --patch-size 0.005 \\
        --n-thumbnails 25 \\
        --n-hist-patches 30 \\
        --out validation/

Outputs (all saved to --out directory)::

    validate_overview.png      Strip polygons + bboxes + sampler centres
    validate_strip_detail.png  Single-strip deep dive: valid vs rejected grid
    validate_histogram.png     Pixel-value distributions + nodata diagnostic
    validate_thumbnails.png    Grid of actual sample patches with per-patch stats
    validate_calibration.png   Before/after nodata-mask calibration comparison
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # headless — must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from shapely.geometry import box as shapely_box

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OFFSET_DEFAULT = 0.037954361744101  # from ProductMeta defaults


def _stretch(img: np.ndarray, nodata_thresh: float = 1e-6) -> np.ndarray:
    """Per-channel percentile stretch, ignoring nodata pixels."""
    out = img.copy()
    if out.ndim == 3:
        for c in range(out.shape[2]):
            band = out[..., c]
            dp = band[band > nodata_thresh]
            if len(dp) > 10:
                p2, p98 = np.percentile(dp, [2, 98])
                if p98 > p2:
                    out[..., c] = np.clip((band - p2) / (p98 - p2), 0, 1)
                    out[..., c][band <= nodata_thresh] = 0
    else:
        dp = out[out > nodata_thresh]
        if len(dp) > 10:
            p2, p98 = np.percentile(dp, [2, 98])
            if p98 > p2:
                out = np.clip((out - p2) / (p98 - p2), 0, 1)
                out[img <= nodata_thresh] = 0
    return out


# ---------------------------------------------------------------------------
# Plot 1 — Overview: all strips + sampler centres
# ---------------------------------------------------------------------------


def plot_overview(dataset, sampler, out_dir: Path) -> None:
    """All strip polygons, their axis-aligned bboxes, and sampler centres."""
    fig, ax = plt.subplots(figsize=(14, 10))

    n_strips = len(dataset.index)
    cmap = mpl.colormaps.get_cmap("tab20")

    # Draw each strip polygon + bbox
    for i in range(n_strips):
        geom = dataset.index.geometry.iloc[i]
        colour = cmap(i % 20)
        b = geom.bounds

        # Axis-aligned bbox (dashed)
        bbox_rect = Rectangle(
            (b[0], b[1]),
            b[2] - b[0],
            b[3] - b[1],
            linestyle="--",
            edgecolor=colour,
            facecolor="none",
            linewidth=0.4,
            alpha=0.45,
        )
        ax.add_patch(bbox_rect)

        # Actual polygon
        try:
            xs, ys = geom.exterior.xy
            ax.fill(xs, ys, alpha=0.12, color=colour)
            ax.plot(xs, ys, color=colour, linewidth=0.7, alpha=0.8)
        except AttributeError:
            # MultiPolygon fallback
            for part in geom.geoms:
                xs, ys = part.exterior.xy
                ax.fill(xs, ys, alpha=0.12, color=colour)
                ax.plot(xs, ys, color=colour, linewidth=0.7, alpha=0.8)

    # Sampler valid centres
    if sampler._centers:
        xs = [c[0] for c in sampler._centers]
        ys = [c[1] for c in sampler._centers]
        ax.scatter(
            xs, ys, s=0.4, c="green", alpha=0.25, rasterized=True,
            label=f"Valid centres ({len(sampler._centers):,})",
        )

    # A random subset of sample patches
    half_w = sampler.size[1] / 2
    half_h = sampler.size[0] / 2
    n_show = min(150, len(sampler._centers))
    if n_show > 0:
        rng = np.random.default_rng(42)
        idxs = rng.choice(len(sampler._centers), n_show, replace=False)
        rects = []
        for idx in idxs:
            cx, cy, _ = sampler._centers[idx]
            rects.append(
                Rectangle((cx - half_w, cy - half_h), 2 * half_w, 2 * half_h)
            )
        pc = PatchCollection(
            rects,
            facecolor="orange",
            edgecolor="darkorange",
            alpha=0.12,
            linewidth=0.3,
        )
        ax.add_collection(pc)

    ax.set_xlabel("Longitude (\u00b0E)")
    ax.set_ylabel("Latitude (\u00b0N)")
    ax.set_title(
        f"MarsHiRISE — {n_strips} strip polygons  |  "
        f"{len(sampler._centers):,} valid sampler centres  |  "
        f"{n_show} sample patches shown"
    )
    ax.legend(loc="upper right", fontsize=8, markerscale=8)
    ax.set_aspect("equal")
    fig.tight_layout()

    path = out_dir / "validate_overview.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Plot 2 — Single-strip detail: valid vs rejected grid
# ---------------------------------------------------------------------------


def plot_strip_detail(dataset, sampler, out_dir: Path, strip_idx: int = 0) -> None:
    """One strip showing the full candidate grid with pass/fail markers."""
    if strip_idx >= len(dataset.index):
        logger.warning("strip_idx %d out of range (%d strips)", strip_idx, len(dataset.index))
        return

    geom = dataset.index.geometry.iloc[strip_idx]
    obs_id = dataset.index.iloc[strip_idx].get("obs_id", f"Strip {strip_idx}")
    b = geom.bounds

    half_h = sampler.size[0] / 2
    half_w = sampler.size[1] / 2
    stride_h, stride_w = sampler.stride
    inset = max(sampler.size[0], sampler.size[1]) * 0.05

    # Reproduce the sampler's effective polygon
    try:
        effective = geom.buffer(-inset)
        if effective.is_empty or not effective.is_valid:
            effective = geom
    except Exception:
        effective = geom

    eb = effective.bounds

    valid_coords = np.array([(c[0], c[1]) for c in sampler._centers])

    from scipy.spatial import cKDTree
    tree = cKDTree(valid_coords) if len(valid_coords) > 0 else None

    valid, rejected = [], []
    for cy in np.arange(eb[1] + half_h, eb[3] - half_h + stride_h * 1e-6, stride_h):
        for cx in np.arange(eb[0] + half_w, eb[2] - half_w + stride_w * 1e-6, stride_w):

            # 2. Check if this generated grid point exists in the sampler's list
            is_valid = False
            if tree is not None:
                # Query nearest neighbor distance
                dist, _ = tree.query([cx, cy])
                if dist < 1e-5:  # Tolerance for floating-point drift
                    is_valid = True

            # ----- No geometric math needed here! -----
            if is_valid:
                valid.append((cx, cy))
            else:
                rejected.append((cx, cy))

    total = len(valid) + len(rejected)
    waste_pct = 100 * len(rejected) / total if total else 0

    margin = max(b[2] - b[0], b[3] - b[1]) * 0.12

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    # --- Left panel: grid pass/fail ---
    ax = axes[0]
    bbox_rect = Rectangle(
        (b[0], b[1]), b[2] - b[0], b[3] - b[1],
        linestyle="--", edgecolor="gray", facecolor="#f0f0f0", alpha=0.25, linewidth=1,
    )
    ax.add_patch(bbox_rect)

    try:
        xs, ys = geom.exterior.xy
        ax.fill(xs, ys, alpha=0.18, color="steelblue")
        ax.plot(xs, ys, color="steelblue", linewidth=1.2)
    except AttributeError:
        for part in geom.geoms:
            xs, ys = part.exterior.xy
            ax.fill(xs, ys, alpha=0.18, color="steelblue")
            ax.plot(xs, ys, color="steelblue", linewidth=1.2)

    # Show effective (inset) polygon
    try:
        exs, eys = effective.exterior.xy
        ax.plot(exs, eys, color="teal", linewidth=0.6, linestyle=":", alpha=0.6)
    except Exception:
        pass

    if rejected:
        rx, ry = zip(*rejected)
        ax.scatter(rx, ry, s=14, c="red", alpha=0.45, marker="x", linewidths=0.6,
                   label=f"Rejected ({len(rejected)})", zorder=3)
    if valid:
        vx, vy = zip(*valid)
        ax.scatter(vx, vy, s=14, c="green", alpha=0.65, zorder=4,
                   label=f"Valid ({len(valid)})")

    ax.set_xlim(b[0] - margin, b[2] + margin)
    ax.set_ylim(b[1] - margin, b[3] + margin)
    ax.set_xlabel("Longitude (\u00b0E)")
    ax.set_ylabel("Latitude (\u00b0N)")
    ax.set_title(
        f"{obs_id}\n"
        f"Valid: {len(valid)}  |  Rejected: {len(rejected)}  |  "
        f"Waste prevented: {waste_pct:.0f}%"
    )
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    # --- Right panel: sampled patches ---
    ax = axes[1]
    bbox_rect2 = Rectangle(
        (b[0], b[1]), b[2] - b[0], b[3] - b[1],
        linestyle="--", edgecolor="gray", facecolor="#f0f0f0", alpha=0.15, linewidth=1,
    )
    ax.add_patch(bbox_rect2)

    try:
        xs, ys = geom.exterior.xy
        ax.fill(xs, ys, alpha=0.15, color="steelblue")
        ax.plot(xs, ys, color="steelblue", linewidth=1.2)
    except AttributeError:
        for part in geom.geoms:
            xs, ys = part.exterior.xy
            ax.fill(xs, ys, alpha=0.15, color="steelblue")
            ax.plot(xs, ys, color="steelblue", linewidth=1.2)

    n_show = min(25, len(valid))
    if valid:
        rng = np.random.default_rng(42)
        show_idxs = rng.choice(len(valid), n_show, replace=False)
        rects = []
        for idx in show_idxs:
            cx, cy = valid[idx]
            rects.append(
                Rectangle((cx - half_w, cy - half_h), 2 * half_w, 2 * half_h)
            )
            ax.plot(cx, cy, ".", color="darkorange", markersize=3, zorder=5)
        pc = PatchCollection(
            rects, facecolor="orange", edgecolor="darkorange",
            alpha=0.2, linewidth=0.8, zorder=3,
        )
        ax.add_collection(pc)

    ax.set_xlim(b[0] - margin, b[2] + margin)
    ax.set_ylim(b[1] - margin, b[3] + margin)
    ax.set_xlabel("Longitude (\u00b0E)")
    ax.set_ylabel("Latitude (\u00b0N)")
    ax.set_title(f"{obs_id}\n{n_show} sampled patches")
    ax.set_aspect("equal")

    fig.suptitle("Single-strip detail — grid validation", fontsize=14, y=1.01)
    fig.tight_layout()
    path = out_dir / "validate_strip_detail.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    path = out_dir / "validate_strip_detail.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Plot 3 — Pixel-value histograms + nodata diagnostic
# ---------------------------------------------------------------------------


def _load_patch(dataset, sampler, idx: int):
    """Load a single patch by valid-centre index.  Returns image tensor or None."""
    cx, cy, interval = sampler._centers[idx]
    half_h = sampler.size[0] / 2
    half_w = sampler.size[1] / 2
    query = (
        slice(cx - half_w, cx + half_w),
        slice(cy - half_h, cy + half_h),
        slice(interval.left, interval.right),
    )
    sample = dataset[query]
    img = sample["image"]
    if img.ndim == 4:
        img = img[0]
    return img


def plot_histogram(dataset, sampler, out_dir: Path, n_patches: int = 20) -> None:
    """Pixel-value distributions with nodata contamination detection."""
    rng = np.random.default_rng(42)
    idxs = rng.choice(
        len(sampler._centers), min(n_patches, len(sampler._centers)), replace=False
    )

    all_vals: list[np.ndarray] = []
    nodata_fracs: list[float] = []

    for idx in idxs:
        try:
            img = _load_patch(dataset, sampler, idx)
            arr = img.numpy()
            for c in range(arr.shape[0]):
                band = arr[c].ravel()
                all_vals.append(band)
                nodata_fracs.append(float(np.mean(np.abs(band) < 1e-6)))
        except Exception as exc:
            cx, cy, _ = sampler._centers[idx]
            logger.warning("Patch (%.4f, %.4f) failed: %s", cx, cy, exc)

    if not all_vals:
        logger.error("No patches loaded — skipping histogram plot")
        return

    combined = np.concatenate(all_vals)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (0,0) — full histogram
    ax = axes[0, 0]
    ax.hist(combined, bins=250, color="steelblue", edgecolor="none", log=True)
    ax.axvline(0.0, color="green", linestyle="-", linewidth=1, label="0.0 (expected nodata)")
    ax.axvline(
        _OFFSET_DEFAULT, color="red", linestyle="--", linewidth=1,
        label=f"offset ({_OFFSET_DEFAULT:.4f})",
    )
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Count (log)")
    ax.set_title("All pixel values")
    ax.legend(fontsize=8)

    # (0,1) — zoom into [0, 0.06]
    ax = axes[0, 1]
    low = combined[combined < 0.06]
    if len(low) > 0:
        ax.hist(low, bins=120, color="coral", edgecolor="none")
        ax.axvline(0.0, color="green", linewidth=1.2, label="0.0")
        ax.axvline(
            _OFFSET_DEFAULT, color="red", linestyle="--", linewidth=1.2,
            label=f"offset ({_OFFSET_DEFAULT:.4f})",
        )
        ax.set_xlabel("Pixel value")
        ax.set_ylabel("Count")
        ax.set_title("Zoom: [0, 0.06] — nodata detection zone")
        ax.legend(fontsize=8)

        # Annotate the diagnostic
        at_zero = int(np.sum(np.abs(combined) < 1e-8))
        near_offset = int(np.sum(np.abs(combined - _OFFSET_DEFAULT) < 0.003))
        if near_offset > at_zero and near_offset > len(combined) * 0.05:
            ax.text(
                0.98, 0.95,
                "BUG: nodata at offset,\nnot at 0.0",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color="red", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
            )
        elif at_zero > len(combined) * 0.01:
            ax.text(
                0.98, 0.95,
                "PASS: nodata at 0.0",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color="green", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
            )

    # (1,0) — data-only histogram
    ax = axes[1, 0]
    data_vals = combined[combined > 1e-3]
    if len(data_vals) > 0:
        ax.hist(data_vals, bins=200, color="seagreen", edgecolor="none")
        ax.set_xlabel("Pixel value (I/F)")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Data pixels only (> 0.001): "
            f"{len(data_vals):,} / {len(combined):,} "
            f"({100 * len(data_vals) / len(combined):.1f}%)"
        )

    # (1,1) — nodata fraction per band-patch
    ax = axes[1, 1]
    if nodata_fracs:
        colours = ["seagreen" if f < 0.5 else "tomato" for f in nodata_fracs]
        ax.bar(range(len(nodata_fracs)), nodata_fracs, color=colours, edgecolor="none")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.5)
        med = np.median(nodata_fracs)
        ax.set_xlabel("Band-patch index")
        ax.set_ylabel("Nodata fraction")
        ax.set_title(f"Nodata fraction per band-patch  (median: {med:.2f})")
        ax.set_ylim(0, 1.05)

    fig.suptitle("Pixel value diagnostics", fontsize=14)
    fig.tight_layout()
    path = out_dir / "validate_histogram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)

    # ---------- Console diagnostic ----------
    at_zero = int(np.sum(np.abs(combined) < 1e-8))
    near_offset = int(np.sum(np.abs(combined - _OFFSET_DEFAULT) < 0.003))
    data_count = int(np.sum(combined > 1e-3))
    n = len(combined)

    logger.info("=" * 60)
    logger.info("NODATA DIAGNOSTIC")
    logger.info("-" * 60)
    logger.info("Total pixels sampled:        %10d", n)
    logger.info("Pixels at 0.0 (±1e-8):       %10d  (%5.1f%%)", at_zero, 100 * at_zero / n)
    logger.info("Pixels near offset (~0.038):  %10d  (%5.1f%%)", near_offset, 100 * near_offset / n)
    logger.info("Pixels with real data (>1e-3):%10d  (%5.1f%%)", data_count, 100 * data_count / n)
    logger.info("-" * 60)

    if near_offset > at_zero and near_offset > n * 0.05:
        logger.warning(
            "BUG DETECTED: nodata pixels are being calibrated to the offset "
            "value (~0.038) instead of remaining at 0.0."
        )
        logger.warning(
            "FIX: In _load_from_jp2, add  nodata_mask = (dest == 0.0)  "
            "before calibration, and  dest[nodata_mask] = 0.0  after."
        )
    elif at_zero > n * 0.01:
        logger.info("PASS: nodata pixels are correctly at 0.0")
    else:
        logger.info(
            "NOTE: very few zero pixels — patches may be fully covered "
            "or the dataset is very sparse."
        )
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Plot 4 — Sample patch thumbnails
# ---------------------------------------------------------------------------


def plot_thumbnails(
        dataset, sampler, out_dir: Path, n_patches: int = 16
) -> None:
    """Grid of actual sample patches with per-patch stats."""
    cols = 4
    rows = max(1, (n_patches + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.8 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]
    if cols == 1:
        axes = axes[:, np.newaxis]

    rng = np.random.default_rng(123)
    idxs = rng.choice(
        len(sampler._centers), min(n_patches, len(sampler._centers)), replace=False
    )

    loaded = 0
    for flat_i, idx in enumerate(idxs):
        r, c = divmod(flat_i, cols)
        ax = axes[r, c]
        cx, cy, _ = sampler._centers[idx]

        try:
            img = _load_patch(dataset, sampler, idx)
            arr = img.numpy()

            # Per-patch stats
            nodata_frac = float((arr < 1e-6).mean())
            data_px = arr[arr > 1e-6]
            dmean = float(data_px.mean()) if len(data_px) > 0 else 0
            dmax = float(data_px.max()) if len(data_px) > 0 else 0

            # Display image
            if arr.shape[0] >= 3:
                disp = arr[:3].transpose(1, 2, 0)  # CHW → HWC
            else:
                disp = arr[0]

            disp = _stretch(disp)

            cmap = None if disp.ndim == 3 else "gray"
            ax.imshow(disp, cmap=cmap, interpolation="nearest")
            ax.set_title(
                f"({cx:.3f}\u00b0, {cy:.3f}\u00b0)\n"
                f"nodata: {nodata_frac:.0%}  "
                f"mean: {dmean:.4f}  max: {dmax:.4f}",
                fontsize=7,
            )
            loaded += 1
        except Exception as exc:
            ax.text(
                0.5, 0.5, f"Load failed:\n{exc!s:.60}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7, color="red", wrap=True,
            )

        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for flat_i in range(len(idxs), rows * cols):
        r, c = divmod(flat_i, cols)
        axes[r, c].set_visible(False)

    fig.suptitle(f"Sample patch thumbnails", fontsize=13)
    fig.tight_layout()
    path = out_dir / "validate_thumbnails.pdf"
    fig.savefig(path, dpi=600)
    path = out_dir / "validate_thumbnails.png"
    fig.savefig(path, dpi=600)
    plt.close(fig)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Plot 5 — Before/after calibration comparison
# ---------------------------------------------------------------------------


def plot_calibration_comparison(
        dataset, sampler, out_dir: Path, n_patches: int = 4
) -> None:
    """Side-by-side: raw calibration (current) vs nodata-masked calibration.

    This directly demonstrates the nodata-mask fix by loading patches and
    showing what the image looks like with and without the fix applied in
    post-processing.
    """
    rng = np.random.default_rng(77)
    idxs = rng.choice(
        len(sampler._centers), min(n_patches, len(sampler._centers)), replace=False
    )

    rows_ok = []
    for idx in idxs:
        try:
            img = _load_patch(dataset, sampler, idx)
            cx, cy, _ = sampler._centers[idx]
            rows_ok.append((img.numpy(), cx, cy))
        except Exception:
            pass

    if not rows_ok:
        logger.warning("No patches loaded — skipping calibration comparison")
        return

    n = len(rows_ok)
    fig, axes = plt.subplots(n, 3, figsize=(14, 4 * n), squeeze=False)

    for row_i, (arr, cx, cy) in enumerate(rows_ok):
        # --- Current (potentially buggy) ---
        if arr.shape[0] >= 3:
            disp_raw = arr[:3].transpose(1, 2, 0)
        else:
            disp_raw = arr[0]

        cmap = None if disp_raw.ndim == 3 else "gray"

        # No stretch — raw values as matplotlib sees them
        ax = axes[row_i, 0]
        ax.imshow(np.clip(disp_raw, 0, 1), cmap=cmap, interpolation="nearest")
        ax.set_title("Raw (no stretch)\nHow matplotlib sees it", fontsize=9)
        ax.set_ylabel(f"({cx:.3f}\u00b0, {cy:.3f}\u00b0)", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        # --- After applying nodata mask in post-processing ---
        fixed = arr.copy()
        # Simulate the fix: any pixel near the offset that's surrounded by
        # other near-offset pixels is likely nodata
        for c in range(fixed.shape[0]):
            band = fixed[c]
            # Heuristic: values within 2% of the offset AND below 0.05 are
            # almost certainly uncalibrated nodata
            nodata_mask = np.abs(band - _OFFSET_DEFAULT) < 0.005
            # Additionally catch exact-zero (already correct if fix applied)
            nodata_mask |= band < 1e-6
            band[nodata_mask] = 0.0

        if fixed.shape[0] >= 3:
            disp_fixed = fixed[:3].transpose(1, 2, 0)
        else:
            disp_fixed = fixed[0]

        ax = axes[row_i, 1]
        disp_fixed_stretched = _stretch(disp_fixed)
        ax.imshow(disp_fixed_stretched, cmap=cmap, interpolation="nearest")
        ax.set_title("After nodata mask + stretch\n(simulated fix)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        # --- Nodata mask visualisation ---
        ax = axes[row_i, 2]
        if fixed.shape[0] >= 3:
            mask_vis = (fixed[:3].sum(axis=0) < 1e-6).astype(float)
        else:
            mask_vis = (fixed[0] < 1e-6).astype(float)
        ax.imshow(mask_vis, cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title("Nodata mask\n(green = data, red = nodata)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Calibration comparison: current vs nodata-masked", fontsize=13, y=1.01,
    )
    fig.tight_layout()
    path = out_dir / "validate_calibration.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate MarsHiRISE strip geometry and sampling pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Outputs saved to --out directory (default: validation/).
            All plots use the Agg backend — safe for headless SSH sessions.
        """),
    )
    parser.add_argument(
        "--root", type=str, default="/scratch/mars_hirise",
        help="Dataset root (default: /scratch/mars_hirise)",
    )
    parser.add_argument(
        "--out", type=str, default="validation",
        help="Output directory for PNGs (default: validation/)",
    )
    parser.add_argument(
        "--bbox", type=float, nargs=4,
        default=[-136, 12, -124, 24],
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="Bounding box in degrees (default: Olympus Mons region)",
    )
    parser.add_argument(
        "--patch-size", type=float, default=0.005,
        help="Patch size in degrees (default: 0.005 ~ 593 px)",
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-thumbnails", type=int, default=16)
    parser.add_argument("--n-hist-patches", type=int, default=20)
    parser.add_argument("--n-calib-patches", type=int, default=4)
    parser.add_argument(
        "--strip-detail", type=int, default=0,
        help="Index of strip to show in the detail view",
    )
    args = parser.parse_args()

    # Resolve imports — add src/ to path
    src_dir = Path(__file__).resolve().parent / "src"
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))
    else:
        # Maybe we're already inside src/, or the user has it on PYTHONPATH
        sys.path.insert(0, str(Path(__file__).resolve().parent))

    # Late imports so sys.path is set
    from dataset.mars_hirise import MarsHiRISE  # noqa: E402
    from dataset.hirise_sampler import HiRISEGeoSampler  # noqa: E402
    from torchgeo.samplers import Units  # noqa: E402

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MarsHiRISE sampling validation")
    logger.info("=" * 60)
    logger.info("Root:         %s", args.root)
    logger.info("Bbox:         %s", args.bbox)
    logger.info("Patch size:   %.4f\u00b0  (~%d px)",
                args.patch_size, round(args.patch_size / (1.0 / 118_502.26)))
    logger.info("Output dir:   %s", out_dir)

    t0 = time.monotonic()

    logger.info("Loading dataset ...")
    dataset = MarsHiRISE(
        root=args.root,
        bbox=tuple(args.bbox),
        channels=["NEAR-INFRARED", "RED", "BLUE-GREEN"],
        download=False,
        reuse_cache=True,
    )
    logger.info("Dataset: %d observations", len(dataset))

    logger.info("Building sampler ...")
    sampler = HiRISEGeoSampler(
        dataset,
        size=args.patch_size,
        length=args.n_samples,
        units=Units.CRS,
    )
    logger.info("Sampler: %d valid centres", len(sampler._centers))

    if len(sampler._centers) == 0:
        logger.error("No valid centres — nothing to validate. "
                     "Check that JP2 files exist and strip polygons are "
                     "larger than the requested patch size.")
        sys.exit(1)

    logger.info("-" * 60)
    logger.info("Generating validation plots ...")
    logger.info("-" * 60)

    # plot_overview(dataset, sampler, out_dir)
    plot_strip_detail(dataset, sampler, out_dir, strip_idx=args.strip_detail)
    # plot_histogram(dataset, sampler, out_dir, n_patches=args.n_hist_patches)
    plot_thumbnails(dataset, sampler, out_dir, n_patches=args.n_thumbnails)
    # plot_calibration_comparison(
    #     dataset, sampler, out_dir, n_patches=args.n_calib_patches
    # )

    elapsed = time.monotonic() - t0
    logger.info("=" * 60)
    logger.info("Validation complete in %.1f s", elapsed)
    logger.info("All plots saved to: %s/", out_dir)
    logger.info("  validate_overview.png      — strip polygons + bboxes + centres")
    logger.info("  validate_strip_detail.png  — single-strip grid pass/fail")
    logger.info("  validate_histogram.png     — pixel distributions + nodata check")
    logger.info("  validate_thumbnails.png    — sample patch thumbnails")
    logger.info("  validate_calibration.png   — before/after nodata-mask fix")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
