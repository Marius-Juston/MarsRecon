"""Compute per-channel statistics across the full MarsHiRISE LitData cache.

Outputs mean, std, min, max, and a 1024-bin histogram per channel for use as
MAE normalization parameters.

Fast Path (LitData)
-------------------
Instead of dynamically loading and orthorectifying `.IMG`/`.JP2` files via GDAL,
this script directly reads the pre-extracted tensors from the LitData cache
specified by the OmegaConf config.

All operations strictly utilize NumPy for numerical stability and compatibility.

Usage
-----
    uv run python src/dataset/compute_dataset_stats.py --config configs/train_hirise.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pathlib
import sys

import numpy as np
from litdata import StreamingDataset, StreamingDataLoader
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm

# Assuming this exists in your codebase for the hash key
from depth_fm.build_litdata_raw import get_litdata_cache_key

_SRC = pathlib.Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_HIST_BINS: int = 2048
OUTPUT_DIR: pathlib.Path = pathlib.Path("dataset_stats")
TEMP_STATS_PATH: str = "/tmp/hirise_stats_rank_{rank}.npy"

HIST_RANGE = {
    "dtm": {"min": -5300.0, "max": 21300.0},
    "image": {"min": 0.0, "max": 1.0}
}

CENTERED_HIST_RANGE = {
    "dtm": {"min": 0.0, "max": 730.5},  # Max expected variance within 590m
    "image": {"min": -0.22, "max": 0.22}
}


# ---------------------------------------------------------------------------
# Core Math & Stats Functions (NumPy)
# ---------------------------------------------------------------------------

def _welford_update(
        n: int, mean: float, M2: float, valid: np.ndarray
) -> tuple[int, float, float]:
    """Update a single-channel Welford accumulator with a batch of valid pixels."""
    m = valid.size
    if m == 0:
        return n, mean, M2

    new_n = n + m
    batch_mean = valid.mean()
    delta_old = batch_mean - mean
    mean_new = mean + delta_old * m / new_n
    batch_var_sum = ((valid - batch_mean) ** 2).sum()
    delta_new = batch_mean - mean_new
    M2_new = M2 + batch_var_sum + delta_old * delta_new * m

    return new_n, mean_new, M2_new


def detrend(z_data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Detrends a 2D elevation array using a least-squares plane fit."""
    h, w = z_data.shape
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    x_valid, y_valid, z_valid = x_grid[mask], y_grid[mask], z_data[mask]

    if z_valid.size == 0:
        return np.array([], dtype=np.float64)

    A = np.c_[x_valid, y_valid, np.ones_like(x_valid)]
    C, _, _, _ = np.linalg.lstsq(A, z_valid, rcond=None)
    detrended = z_data - (C[0] * x_grid + C[1] * y_grid + C[2])

    return np.abs(detrended[mask])


# ---------------------------------------------------------------------------
# Per-Split Worker
# ---------------------------------------------------------------------------

def _worker_fn(split: str, split_path: str, args: dict) -> None:
    """Worker entry point — runs on a LitData split."""
    channels: list[str] = args["channels"]
    C = len(channels)
    path = TEMP_STATS_PATH.format(rank=split)

    if pathlib.Path(path).exists():
        logger.info(f"Compiled file already exists, skipping")
        return

    dataset = StreamingDataset(input_dir=split_path, shuffle=False)

    def collate_fn(data: list):
        """Collates a list of LitData dicts into batched NumPy arrays."""
        return {
            "elevation": [d["elevation"] for d in data],
            "left_red": [d["left_red"] for d in data],
            "right_red": [d["right_red"] for d in data]
        }

    loader = StreamingDataLoader(
        dataset,
        batch_size=8,
        num_workers=min(32, os.cpu_count() or 4),
        collate_fn=collate_fn,
        prefetch_factor=16,
        shuffle=False
    )

    # Accumulators
    count = np.zeros(C, dtype=np.int64)
    mean = np.zeros(C, dtype=np.float64)
    M2 = np.zeros(C, dtype=np.float64)
    ch_min = np.full(C, np.inf, dtype=np.float64)
    ch_max = np.full(C, -np.inf, dtype=np.float64)
    hist = np.zeros((C, N_HIST_BINS), dtype=np.int64)

    # Patch-Centered Accumulators
    c_count = np.zeros(C, dtype=np.int64)
    c_mean = np.zeros(C, dtype=np.float64)
    c_M2 = np.zeros(C, dtype=np.float64)
    c_min = np.full(C, np.inf, dtype=np.float64)
    c_max = np.full(C, -np.inf, dtype=np.float64)
    c_hist = np.zeros((C, N_HIST_BINS), dtype=np.int64)

    n_patches = 0
    total_patches = len(dataset)

    for batch in tqdm(loader, desc=f"Split {split}"):
        batch_size = len(batch["elevation"])

        for i in range(batch_size):
            item_data = {
                "elevation": batch["elevation"][i],
                "left_red": batch["left_red"][i],
                "right_red": batch["right_red"][i]
            }

            for c, ch_name in enumerate(channels):
                residual_histo = False
                channel_pixels: np.ndarray = item_data[ch_name]

                if ch_name == "elevation":
                    if channel_pixels.ndim == 3:
                        channel_pixels = channel_pixels.squeeze(0)

                    valid_mask = np.isfinite(channel_pixels) & (channel_pixels != 0.0)
                    range_key = "dtm"

                    # Detrend elevation
                    residual_np = detrend(channel_pixels, valid_mask)
                    centered_valid = residual_np
                    residual_histo = True

                    c_count[c], c_mean[c], c_M2[c] = _welford_update(
                        c_count[c], c_mean[c], c_M2[c], centered_valid
                    )
                    if centered_valid.size > 0:
                        c_min[c] = min(c_min[c], centered_valid.min())
                        c_max[c] = max(c_max[c], centered_valid.max())

                    h_counts, _ = np.histogram(
                        centered_valid, bins=N_HIST_BINS,
                        range=(CENTERED_HIST_RANGE["dtm"]["min"], CENTERED_HIST_RANGE["dtm"]["max"])
                    )
                    c_hist[c] += h_counts.astype(np.int64)

                else:
                    valid_mask = channel_pixels != 0.0
                    range_key = "image"

                valid = channel_pixels[valid_mask].flatten()
                if valid.size == 0:
                    continue

                # 1. Update Raw Stats
                count[c], mean[c], M2[c] = _welford_update(count[c], mean[c], M2[c], valid)
                ch_min[c] = min(ch_min[c], valid.min())
                ch_max[c] = max(ch_max[c], valid.max())

                h_counts, _ = np.histogram(
                    valid, bins=N_HIST_BINS,
                    range=(HIST_RANGE[range_key]["min"], HIST_RANGE[range_key]["max"])
                )
                hist[c] += h_counts.astype(np.int64)

                if not residual_histo:
                    # 2. Update Patch-Centered Stats
                    patch_mean = valid.mean()
                    centered_valid = valid - patch_mean

                    c_count[c], c_mean[c], c_M2[c] = _welford_update(
                        c_count[c], c_mean[c], c_M2[c], centered_valid
                    )
                    c_min[c] = min(c_min[c], centered_valid.min())
                    c_max[c] = max(c_max[c], centered_valid.max())

                    hc_counts, _ = np.histogram(
                        centered_valid, bins=N_HIST_BINS,
                        range=(CENTERED_HIST_RANGE[range_key]["min"], CENTERED_HIST_RANGE[range_key]["max"])
                    )
                    c_hist[c] += hc_counts.astype(np.int64)

    partial = {
        "rank": split,
        "n_patches": n_patches,
        "count": count,
        "mean": mean,
        "M2": M2,
        "ch_min": ch_min,
        "ch_max": ch_max,
        "hist": hist,
        "c_count": c_count,
        "c_mean": c_mean,
        "c_M2": c_M2,
        "c_min": c_min,
        "c_max": c_max,
        "c_hist": c_hist,
    }

    tmp_path = path + ".tmp.npy"
    np.save(tmp_path, partial, allow_pickle=True)
    os.replace(tmp_path, path)
    logger.info(f"[{split}] done — saved partial stats to {path}")


# ---------------------------------------------------------------------------
# Welford Parallel Combination (NumPy)
# ---------------------------------------------------------------------------

def _combine_welford(partials: list[dict]) -> dict:
    combined = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in partials[0].items()}
    combined["n_patches"] = sum(p["n_patches"] for p in partials)

    for p in partials[1:]:
        C = combined["count"].shape[0]
        for c in range(C):
            # Raw combine
            n_a, n_b = combined["count"][c], p["count"][c]
            if n_b > 0:
                if n_a == 0:
                    for k in ["count", "mean", "M2", "ch_min", "ch_max", "hist"]:
                        combined[k][c] = p[k][c]
                else:
                    n_c = n_a + n_b
                    delta = p["mean"][c] - combined["mean"][c]
                    combined["mean"][c] += delta * n_b / n_c
                    combined["M2"][c] += p["M2"][c] + (delta ** 2) * n_a * n_b / n_c
                    combined["count"][c] = n_c
                    combined["ch_min"][c] = min(combined["ch_min"][c], p["ch_min"][c])
                    combined["ch_max"][c] = max(combined["ch_max"][c], p["ch_max"][c])
                    combined["hist"][c] += p["hist"][c]

            # Centered combine
            c_n_a, c_n_b = combined["c_count"][c], p["c_count"][c]
            if c_n_b > 0:
                if c_n_a == 0:
                    for k in ["c_count", "c_mean", "c_M2", "c_min", "c_max", "c_hist"]:
                        combined[k][c] = p[k][c]
                else:
                    c_n_c = c_n_a + c_n_b
                    delta = p["c_mean"][c] - combined["c_mean"][c]
                    combined["c_mean"][c] += delta * c_n_b / c_n_c
                    combined["c_M2"][c] += p["c_M2"][c] + (delta ** 2) * c_n_a * c_n_b / c_n_c
                    combined["c_count"][c] = c_n_c
                    combined["c_min"][c] = min(combined["c_min"][c], p["c_min"][c])
                    combined["c_max"][c] = max(combined["c_max"][c], p["c_max"][c])
                    combined["c_hist"][c] += p["c_hist"][c]

    return combined


# ---------------------------------------------------------------------------
# Output / Display Generation
# ---------------------------------------------------------------------------

def _calculate_percentile_from_hist(hist_counts, bin_edges, percentile: float) -> float:
    total_pixels = sum(hist_counts)
    if total_pixels == 0:
        return 0.0

    target_count = percentile * total_pixels
    cumulative = 0

    for i, count in enumerate(hist_counts):
        if cumulative + count >= target_count:
            if count == 0: return bin_edges[i]
            fraction_into_bin = (target_count - cumulative) / count
            bin_width = bin_edges[i + 1] - bin_edges[i]
            return bin_edges[i] + (fraction_into_bin * bin_width)
        cumulative += count
    return bin_edges[-1]


def _save_stats(combined: dict, channels: list[str], output_dir: pathlib.Path, args: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    C = len(channels)

    count = combined["count"].tolist()
    mean_vals = combined["mean"].tolist()
    M2_vals = combined["M2"].tolist()
    ch_min = combined["ch_min"].tolist()
    ch_max = combined["ch_max"].tolist()
    hist_counts = combined["hist"].tolist()
    std_vals = [math.sqrt(M2_vals[c] / count[c]) if count[c] > 1 else 0.0 for c in range(C)]

    c_count = combined["c_count"].tolist()
    c_mean_vals = combined["c_mean"].tolist()
    c_M2_vals = combined["c_M2"].tolist()
    c_min_vals = combined["c_min"].tolist()
    c_max_vals = combined["c_max"].tolist()
    c_hist_counts = combined["c_hist"].tolist()
    c_std_vals = [math.sqrt(c_M2_vals[c] / c_count[c]) if c_count[c] > 1 else 0.0 for c in range(C)]

    bin_edges_all, c_bin_edges_all = [], []
    p02_vals, p98_vals = [], []
    c_p02_vals, c_p98_vals = [], []

    for c, ch_name in enumerate(channels):
        range_key = "dtm" if ch_name == "elevation" else "image"

        min_x, max_x = HIST_RANGE[range_key]["min"], HIST_RANGE[range_key]["max"]
        edges = np.linspace(min_x, max_x, N_HIST_BINS + 1).tolist()
        bin_edges_all.append(edges)
        p02_vals.append(_calculate_percentile_from_hist(hist_counts[c], edges, 0.02))
        p98_vals.append(_calculate_percentile_from_hist(hist_counts[c], edges, 0.98))

        c_min_x, c_max_x = CENTERED_HIST_RANGE[range_key]["min"], CENTERED_HIST_RANGE[range_key]["max"]
        c_edges = np.linspace(c_min_x, c_max_x, N_HIST_BINS + 1).tolist()
        c_bin_edges_all.append(c_edges)
        c_p02_vals.append(_calculate_percentile_from_hist(c_hist_counts[c], c_edges, 0.02))
        c_p98_vals.append(_calculate_percentile_from_hist(c_hist_counts[c], c_edges, 0.98))

    stats = {
        "channels": channels,
        "n_valid_patches": combined["n_patches"],
        "n_valid_pixels_per_channel": count,
        "mean": mean_vals, "std": std_vals, "min": ch_min, "max": ch_max,
        "p02": p02_vals, "p98": p98_vals,
        "histogram_bin_edges": bin_edges_all,
        "histogram_counts": hist_counts,
        "centered_mean": c_mean_vals, "centered_std": c_std_vals,
        "centered_min": c_min_vals, "centered_max": c_max_vals,
        "centered_p02": c_p02_vals, "centered_p98": c_p98_vals,
        "centered_histogram_bin_edges": c_bin_edges_all,
        "centered_histogram_counts": c_hist_counts,
    }

    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Plot both sets of histograms
    for is_centered in [False, True]:
        prefix = "centered_" if is_centered else ""

        for c, ch_name in enumerate(channels):
            edges = c_bin_edges_all[c] if is_centered else bin_edges_all[c]
            counts = c_hist_counts[c] if is_centered else hist_counts[c]
            mean_v = c_mean_vals[c] if is_centered else mean_vals[c]
            std_v = c_std_vals[c] if is_centered else std_vals[c]
            p02 = c_p02_vals[c] if is_centered else p02_vals[c]
            p98 = c_p98_vals[c] if is_centered else p98_vals[c]

            bin_centers = [(edges[i] + edges[i + 1]) / 2.0 for i in range(N_HIST_BINS)]
            bar_width = (edges[-1] - edges[0]) / N_HIST_BINS

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(
                bin_centers, counts, width=bar_width,
                align="center", color="steelblue", edgecolor="none",
            )
            ax.set_xscale("symlog")

            ax.axvline(p02, color='red', linestyle='--', linewidth=1, label='2% / 98%')
            ax.axvline(p98, color='red', linestyle='--', linewidth=1)

            x_label = "Elevation (m)" if ch_name == "elevation" else "Calibrated I/F value"
            if is_centered:
                x_label += " (Centered)"

            ax.set_xlabel(x_label)
            ax.set_ylabel("Pixel count")
            title_prefix = f"[Centered] " if is_centered else ""
            ax.set_title(f"{title_prefix}{ch_name}  |  mean={mean_v:.4f}  std={std_v:.4f}")
            ax.set_xlim(edges[0], edges[-1])

            safe_name = ch_name.replace(" ", "_").replace("/", "-")
            png_path = output_dir / f"{prefix}histogram_{safe_name}.png"
            fig.tight_layout()
            fig.savefig(png_path, dpi=150)
            plt.close(fig)

            png_path = output_dir / f"{prefix}histogram_{safe_name}.pdf"
            logger.info("Saving histogram to %s", png_path)
            fig.savefig(png_path, dpi=150)
            plt.close(fig)

    print("\nNormalization parameters for MAE:")
    for c, ch_name in enumerate(channels):
        print(f"  {ch_name}: raw_std={std_vals[c]:.6f}, centered_std={c_std_vals[c]:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    parser = argparse.ArgumentParser(description="Fast LitData statistics pipeline")
    parser.add_argument("--config", type=str, default="configs/train_hirise.yaml",
                        help="Path to OmegaConf training config")
    args = parser.parse_args(argv)

    config = OmegaConf.load(args.config)
    cache_hash = get_litdata_cache_key(config)

    dataset_root = pathlib.Path(config.data.hirise.root)
    litdata_dir = dataset_root / f"litdata_cache_{cache_hash}"

    if not litdata_dir.exists():
        logger.error(f"LitData cache not found at {litdata_dir}.")
        sys.exit(1)

    split_paths = [str(p) for p in litdata_dir.iterdir() if p.is_dir() and (p / "_SUCCESS").exists()]
    if not split_paths:
        logger.error(f"No valid splits found in {litdata_dir}")
        sys.exit(1)

    logger.info(f"Using LitData Cache: {litdata_dir}")
    logger.info(f"Found splits: {[pathlib.Path(p).name for p in split_paths]}")

    stats_channels = ["elevation", "left_red", "right_red"]
    output_dir = OUTPUT_DIR / "dtm"

    worker_args = {
        "channels": stats_channels,
    }

    # Process Splits
    for p in split_paths:
        _worker_fn(pathlib.Path(p).name, p, worker_args)

    # Reduce partials utilizing split names
    logger.info("\nReducing partial statistics…")
    partials: list[dict] = []
    for p in split_paths:
        split_name = pathlib.Path(p).name
        path = TEMP_STATS_PATH.format(rank=split_name)
        partials.append(np.load(path, allow_pickle=True).item())

    combined = _combine_welford(partials)
    _save_stats(combined, stats_channels, output_dir, vars(args))

    # Cleanup temp files
    # for p in split_paths:
    #     path = TEMP_STATS_PATH.format(rank=pathlib.Path(p).name)
    #     if os.path.exists(path):
    #         os.remove(path)


if __name__ == "__main__":
    main()
