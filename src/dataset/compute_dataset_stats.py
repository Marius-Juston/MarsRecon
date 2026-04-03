"""Compute per-channel statistics across the full MarsHiRISE Olympus dataset.

Outputs mean, std, min, max, and a 256-bin histogram per channel for use as
MAE normalization parameters.

Parallelism
-----------
torch.multiprocessing.spawn launches one worker process per GPU (nprocs=4).
Each worker:
  - Owns 1/4 of the pre-computed patch centres (sequential, no replacement)
  - Creates a DataLoader with num_workers=64  →  4 × 64 = 256 total I/O workers
  - Accumulates running statistics on its GPU using the vectorized Chan/Welford
    parallel algorithm (numerically stable, O(1) memory)
  - Saves partial stats to /tmp/hirise_stats_rank{r}.pt

The main process reduces all four partial results with the Welford combination
formula and writes dataset_stats/dataset_stats.json plus per-channel PNG
histograms.

Nodata policy
-------------
Pixels with calibrated value exactly 0.0 are treated as nodata and excluded
from all statistics.  This is correct because MarsHiRISE._load_from_jp2()
explicitly resets nodata locations to 0.0 *after* calibration
(``dest[nodata_mask] = 0.0``).  At default coefficients (offset ≈ 0.038) a
physically valid dark pixel calibrates to ~0.038, never 0.0, so the exclusion
loses negligible valid signal.

Usage
-----
    uv run python src/dataset/compute_dataset_stats.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from dataset.mars_hirise_dtm import MarsHiRISEDTM

_SRC = pathlib.Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hirise_sampler import HiRISEGeoSampler
from mars_hirise import MarsHiRISE
from torchgeo.samplers import Units

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PATCH_SIZE_DEG: float = 0.005  # ~0.005° ≈ 590 m at Mars equator
CHANNELS: list[str] = ["NEAR-INFRARED", "RED", "BLUE-GREEN"]
DTM_CHANNELS: list[str] = ["RED"]
N_HIST_BINS: int = 1024
OUTPUT_DIR: pathlib.Path = pathlib.Path("dataset_stats")
NPROCS: int = 4  # one per GPU
WORKERS_PER_GPU: int = 64  # DataLoader workers per process
TEMP_STATS_PATH: str = "/tmp/hirise_stats_rank{rank}.pt"

BBOX_TUPLE = (-150, 15, -90, 70)

HIST_RANGE = {
    "dtm": {"min": -5300.0, "max": 21300.0},
    "image": {"min": 0.0, "max": 1.0}
}


# ---------------------------------------------------------------------------
# Subset sampler: deterministic sequential pass over a centers sub-list
# ---------------------------------------------------------------------------

class _CenterSubsetSampler:
    """Yield every centre in *centers_slice* exactly once, in order.

    This is intentionally non-random — we want a full, deterministic pass
    over all patches for accurate statistics.

    Args:
        centers_slice: Sub-list of ``(cx, cy, pd.Interval)`` tuples.
        size_tuple:    ``(size_h, size_w)`` in degrees.
    """

    def __init__(self, centers_slice: list, size_tuple: tuple[float, float]) -> None:
        self._centers = centers_slice
        self.size = size_tuple
        self.length = len(centers_slice)

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        half_h, half_w = self.size[0] / 2.0, self.size[1] / 2.0
        for cx, cy, interval in self._centers:
            yield (
                slice(cx - half_w, cx + half_w),
                slice(cy - half_h, cy + half_h),
                slice(interval.left, interval.right),
            )


# ---------------------------------------------------------------------------
# GDAL thread limiter (worker_init_fn for DataLoader)
# ---------------------------------------------------------------------------

def _set_gdal_single_thread(_worker_id: int) -> None:
    """Prevent GDAL from spawning extra threads inside each DataLoader worker.

    With 64 DataLoader workers per GPU process (256 total), each worker
    spinning up GDAL's default thread count would massively over-subscribe
    the CPU.  Setting GDAL_NUM_THREADS=1 keeps parallelism explicit.
    """
    os.environ["GDAL_NUM_THREADS"] = "1"


# ---------------------------------------------------------------------------
# Vectorized Chan/Welford batch update
# ---------------------------------------------------------------------------

def _welford_update(
        n: torch.Tensor,
        mean: torch.Tensor,
        M2: torch.Tensor,
        valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update a single-channel Welford accumulator with a batch of valid pixels.

    Uses the Chan et al. (1979) parallel algorithm applied to a "previous
    accumulator" and a "new batch", giving identical results to processing
    pixels one-at-a-time but in a single vectorized pass.

    Args:
        n:     Scalar int64 tensor — pixels seen so far.
        mean:  Scalar float64 tensor — running mean.
        M2:    Scalar float64 tensor — running sum of squared deviations.
        valid: 1-D float64 tensor of new valid pixel values.

    Returns:
        Updated ``(n, mean, M2)``.
    """
    m = valid.numel()
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


# ---------------------------------------------------------------------------
# Per-GPU worker
# ---------------------------------------------------------------------------

def _worker_fn(rank: int, args: dict) -> None:
    """Worker entry point — runs on a single GPU.

    Iterates over this rank's slice of patch centres, accumulates Welford
    statistics and histogram counts on GPU, then serialises the partial
    results to a temp file for the main process to reduce.

    Args:
        rank: GPU index (0–3).
        args: Dict with keys:
              - ``"slices"``: list of 4 center sub-lists
              - ``"size_tuple"``: (size_h, size_w) in degrees
              - ``"channels"``: list of channel names
              - ``"output_dir"``: not used in worker; kept for reference
    """
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    channels: list[str] = args["channels"]
    C = len(channels)
    centers_slice: list = args["slices"][rank]
    size_tuple: tuple[float, float] = args["size_tuple"]
    dtm: bool = args["dtm"]

    # ----------------------------------------------------------------
    # Build dataset and subset sampler for this rank
    # ----------------------------------------------------------------

    if dtm:
        dataset = MarsHiRISEDTM(
            bbox=BBOX_TUPLE,
            include_ortho=True,
            ortho_type=DTM_CHANNELS,
            download=True,
            reuse_cache=True,
        )
    else:
        dataset = MarsHiRISE(
            target="Olympus",
            channels=channels,
            download=True,
            reuse_cache=True,
        )

    subset_sampler = _CenterSubsetSampler(centers_slice, size_tuple)

    loader = DataLoader(
        dataset,
        sampler=subset_sampler,
        batch_size=1,
        num_workers=WORKERS_PER_GPU,
        multiprocessing_context="spawn",
        prefetch_factor=4,
        worker_init_fn=_set_gdal_single_thread,
        persistent_workers=True,
    )

    # ----------------------------------------------------------------
    # Accumulators (float64 for precision across large pixel counts)
    # ----------------------------------------------------------------
    count = torch.zeros(C, dtype=torch.int64, device=device)
    mean = torch.zeros(C, dtype=torch.float64, device=device)
    M2 = torch.zeros(C, dtype=torch.float64, device=device)
    ch_min = torch.full((C,), float("inf"), dtype=torch.float64, device=device)
    ch_max = torch.full((C,), float("-inf"), dtype=torch.float64, device=device)
    hist = torch.zeros(C, N_HIST_BINS, dtype=torch.int64, device=device)

    n_patches = 0
    log_every = max(1, len(subset_sampler) // 20)

    for batch in loader:
        if not dtm:
            image = batch["image"].squeeze(0).to(device=device, dtype=torch.float64)

        for c, ch_name in enumerate(channels):
            if dtm:
                if ch_name not in batch:
                    continue  # Safe fall-back if a patch misses orthos

                channel_pixels = batch[ch_name].squeeze(0).to(device=device, dtype=torch.float64)

                if ch_name == "elevation":
                    valid_mask = torch.isfinite(channel_pixels)
                    range_key = "dtm"
                else:
                    valid_mask = channel_pixels != 0.0
                    range_key = "image"
            else:
                channel_pixels = image[c]
                valid_mask = channel_pixels != 0.0
                range_key = "image"

            valid = channel_pixels[valid_mask].reshape(-1)
            if valid.numel() == 0:
                continue

            count[c], mean[c], M2[c] = _welford_update(
                count[c], mean[c], M2[c], valid
            )
            ch_min[c] = torch.minimum(ch_min[c], valid.min())
            ch_max[c] = torch.maximum(ch_max[c], valid.max())

            hist[c] += torch.histc(
                valid.float(), bins=N_HIST_BINS, **HIST_RANGE[range_key]
            ).to(torch.int64)

        n_patches += 1
        if n_patches % log_every == 0:
            pct = 100.0 * n_patches / len(subset_sampler)
            print(f"[rank {rank}] {n_patches}/{len(subset_sampler)} patches ({pct:.1f}%)", flush=True)

    # ----------------------------------------------------------------
    # Serialise partial results to temp file
    # ----------------------------------------------------------------
    partial = {
        "rank": rank,
        "n_patches": n_patches,
        "count": count.cpu(),
        "mean": mean.cpu(),
        "M2": M2.cpu(),
        "ch_min": ch_min.cpu(),
        "ch_max": ch_max.cpu(),
        "hist": hist.cpu(),
    }
    path = TEMP_STATS_PATH.format(rank=rank)
    torch.save(partial, path)
    print(f"[rank {rank}] done — saved partial stats to {path}", flush=True)


# ---------------------------------------------------------------------------
# Welford parallel combination
# ---------------------------------------------------------------------------

def _combine_welford(partials: list[dict]) -> dict:
    """Reduce a list of partial Welford accumulators into one combined result.

    Uses the Chan et al. combination formula, applied sequentially across the
    list (equivalent to a binary tree reduction but simpler for 4 workers).

    Args:
        partials: List of dicts, each with tensors for a single rank's stats.

    Returns:
        Single combined dict with the same structure.
    """
    combined = {
        "n_patches": sum(p["n_patches"] for p in partials),
        "count": partials[0]["count"].clone(),
        "mean": partials[0]["mean"].clone(),
        "M2": partials[0]["M2"].clone(),
        "ch_min": partials[0]["ch_min"].clone(),
        "ch_max": partials[0]["ch_max"].clone(),
        "hist": partials[0]["hist"].clone(),
    }

    for p in partials[1:]:
        C = combined["count"].shape[0]
        for c in range(C):
            n_a = combined["count"][c].item()
            n_b = p["count"][c].item()
            if n_b == 0:
                continue
            if n_a == 0:
                combined["count"][c] = p["count"][c]
                combined["mean"][c] = p["mean"][c]
                combined["M2"][c] = p["M2"][c]
                combined["ch_min"][c] = p["ch_min"][c]
                combined["ch_max"][c] = p["ch_max"][c]
                combined["hist"][c] = p["hist"][c]
                continue

            n_c = n_a + n_b
            delta = p["mean"][c] - combined["mean"][c]
            combined["mean"][c] = combined["mean"][c] + delta * n_b / n_c
            combined["M2"][c] = (
                    combined["M2"][c]
                    + p["M2"][c]
                    + delta ** 2 * n_a * n_b / n_c
            )
            combined["count"][c] = n_c
            combined["ch_min"][c] = torch.minimum(combined["ch_min"][c], p["ch_min"][c])
            combined["ch_max"][c] = torch.maximum(combined["ch_max"][c], p["ch_max"][c])
            combined["hist"][c] += p["hist"][c]

    return combined


# ---------------------------------------------------------------------------
# Save JSON + PNGs
# ---------------------------------------------------------------------------

def _calculate_percentile_from_hist(hist_counts: list[int], bin_edges: list[float] | np.ndarray,
                                    percentile: float) -> float:
    """Estimates a percentile by interpolating the CDF of the histogram."""
    total_pixels = sum(hist_counts)
    if total_pixels == 0:
        return 0.0

    target_count = percentile * total_pixels
    cumulative = 0

    for i, count in enumerate(hist_counts):
        if cumulative + count >= target_count:
            if count == 0:
                return bin_edges[i]

            fraction_into_bin = (target_count - cumulative) / count
            bin_width = bin_edges[i + 1] - bin_edges[i]
            return bin_edges[i] + (fraction_into_bin * bin_width)

        cumulative += count

    return bin_edges[-1]  # Fallback to absolute max bin edge


def _save_stats(
        combined: dict,
        channels: list[str],
        patch_size_deg: float,
        output_dir: pathlib.Path,
        args: dict,
) -> None:
    """Compute final stats from combined accumulators and write output files.

    Args:
        combined:      Output of ``_combine_welford``.
        channels:      List of channel names in order.
        patch_size_deg: Patch size used (for metadata).
        output_dir:    Directory to write JSON and PNG files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    C = len(channels)

    count = combined["count"].tolist()
    mean_vals = combined["mean"].tolist()
    M2_vals = combined["M2"].tolist()
    ch_min = combined["ch_min"].tolist()
    ch_max = combined["ch_max"].tolist()
    hist_counts = combined["hist"].tolist()  # list[list[int]], shape (C, 256)

    std_vals = [
        math.sqrt(M2_vals[c] / count[c]) if count[c] > 1 else 0.0
        for c in range(C)
    ]

    bin_edges_all = []
    p02_vals = []
    p98_vals = []

    for c, ch_name in enumerate(channels):
        range_key = "dtm" if ch_name == "elevation" else "image"
        min_x, max_x = HIST_RANGE[range_key]["min"], HIST_RANGE[range_key]["max"]
        range_val = max_x - min_x

        edges = [(i / N_HIST_BINS) * range_val + min_x for i in range(N_HIST_BINS + 1)]
        bin_edges_all.append(edges)

        p02_vals.append(_calculate_percentile_from_hist(hist_counts[c], edges, 0.02))
        p98_vals.append(_calculate_percentile_from_hist(hist_counts[c], edges, 0.98))

    # Maintain strict backwards compatibility if all channels share the exact same edges
    if all(edges == bin_edges_all[0] for edges in bin_edges_all):
        json_bin_edges = bin_edges_all[0]
    else:
        json_bin_edges = bin_edges_all

    stats = {
        "patch_size_deg": patch_size_deg,
        "channels": channels,
        "n_valid_patches": combined["n_patches"],
        "n_valid_pixels_per_channel": count,
        "mean": mean_vals,
        "std": std_vals,
        "min": ch_min,
        "max": ch_max,
        "p02": p02_vals,
        "p98": p98_vals,
        "histogram_bin_edges": json_bin_edges,
        "histogram_counts": hist_counts,
    }

    json_path = output_dir / "dataset_stats.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {json_path}")

    # ----------------------------------------------------------------
    # Per-channel histogram PNGs
    # ----------------------------------------------------------------
    for c, ch_name in enumerate(channels):
        edges = bin_edges_all[c]
        bin_centers = [(edges[i] + edges[i + 1]) / 2.0 for i in range(N_HIST_BINS)]
        bar_width = (edges[-1] - edges[0]) / N_HIST_BINS

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            bin_centers, hist_counts[c], width=bar_width,
            align="center", color="steelblue", edgecolor="none",
        )

        ax.axvline(p02_vals[c], color='red', linestyle='--', linewidth=1, label='2% / 98%')
        ax.axvline(p98_vals[c], color='red', linestyle='--', linewidth=1)

        ax.set_xlabel("Elevation (m)" if ch_name == "elevation" else "Calibrated I/F value")
        ax.set_ylabel("Pixel count")
        ax.set_title(f"{ch_name}  |  mean={mean_vals[c]:.4f}  std={std_vals[c]:.4f}")
        ax.set_xlim(edges[0], edges[-1])

        safe_name = ch_name.replace(" ", "_").replace("/", "-")
        png_path = output_dir / f"histogram_{safe_name}.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"Saved histogram to {png_path}")

    print("\nNormalization parameters for MAE:")
    for c, ch_name in enumerate(channels):
        print(f"  {ch_name}: mean={mean_vals[c]:.6f}, std={std_vals[c]:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    """Build the full sampler in the main process, dispatch to GPU workers,
    reduce partial results, and save statistics.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Get statistics for the dataset"
    )

    parser.add_argument("--dtm", action=argparse.BooleanOptionalAction)
    args = parser.parse_args(argv)

    output_dir = OUTPUT_DIR
    patch_size_deg = PATCH_SIZE_DEG

    # ----------------------------------------------------------------
    # Build the full sampler once to enumerate all valid centres
    # ----------------------------------------------------------------
    print("Building MarsHiRISE dataset and enumerating valid patch centres…")

    if args.dtm:
        output_dir /= "dtm"

        # Dynamic Channels specifically for DTM Stats (Tracking elevation + orthos)
        stats_channels = ["elevation", "left_red", "right_red"]

        dataset = MarsHiRISEDTM(
            bbox=BBOX_TUPLE,
            include_ortho=True,
            ortho_type=DTM_CHANNELS,
            download=True,
            reuse_cache=True,
        )
    else:
        output_dir /= "image"
        stats_channels = CHANNELS

        dataset = MarsHiRISE(
            target="Olympus",
            channels=stats_channels,
            download=True,
            reuse_cache=True,
        )

    full_sampler = HiRISEGeoSampler(
        dataset,
        split=None,
        size=patch_size_deg,
        length=None,  # defaults to all valid centres
        units=Units.CRS,
        replacement=False,
    )
    all_centers = full_sampler._centers

    # import random
    # random.shuffle(all_centers)
    # patch = 500
    # all_centers = all_centers[:patch * NPROCS]

    size_tuple = full_sampler.size
    n_total = len(all_centers)
    print(f"Total valid patch centres: {n_total}")

    if n_total == 0:
        print("No valid patch centres found. Exiting.")
        return

    # ----------------------------------------------------------------
    # Slice centres for each GPU rank
    # ----------------------------------------------------------------
    chunk = n_total // NPROCS
    slices: list[list] = []
    for r in range(NPROCS):
        start = r * chunk
        end = n_total if r == NPROCS - 1 else (r + 1) * chunk
        slices.append(all_centers[start:end])
        print(f"  rank {r}: centres {start}–{end - 1} ({end - start} patches)")

    worker_args = {
        "slices": slices,
        "size_tuple": size_tuple,
        "channels": stats_channels,
        "dtm": args.dtm
    }

    # ----------------------------------------------------------------
    # Launch one worker per GPU
    # ----------------------------------------------------------------
    print(f"\nLaunching {NPROCS} GPU workers (each with {WORKERS_PER_GPU} DataLoader workers)…")
    mp.spawn(_worker_fn, args=(worker_args,), nprocs=NPROCS, join=True)

    # ----------------------------------------------------------------
    # Load partial results and reduce
    # ----------------------------------------------------------------
    print("\nReducing partial statistics…")
    partials: list[dict] = []
    for r in range(NPROCS):
        path = TEMP_STATS_PATH.format(rank=r)
        partials.append(torch.load(path, weights_only=False))

    combined = _combine_welford(partials)

    # ----------------------------------------------------------------
    # Save final stats
    # ----------------------------------------------------------------
    _save_stats(combined, stats_channels, patch_size_deg, output_dir, vars(args))

    # Cleanup
    for r in range(NPROCS):
        path = TEMP_STATS_PATH.format(rank=r)
        os.remove(path)


if __name__ == "__main__":
    main()
