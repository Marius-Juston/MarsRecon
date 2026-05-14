#!/usr/bin/env python3
"""
Overlayed histogram plot (bar-based) for Mars HiRISE channels.

- Uses density normalization
- Clips to non-zero support
- Keeps real histogram bars (not line plots)

Input:
    dataset_stats/image/dataset_stats.json

Output:
    combined_histogram.png / .pdf
"""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

INPUT_JSON = pathlib.Path("dataset_stats/image/dataset_stats.json")
OUTPUT_DIR = INPUT_JSON.parent

CHANNEL_STYLES = {
    "RED": {"color": "#d62728", "label": "Red"},
    "BLUE-GREEN": {"color": "#17becf", "label": "Blue-Green"},
    "NEAR-INFRARED": {"color": "#9467bd", "label": "Near-IR"},
}

ALPHA = 0.45  # transparency for overlap

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

with open(INPUT_JSON, "r") as f:
    stats = json.load(f)

channels = stats["channels"]
hist_counts = stats["histogram_counts"]
bin_edges = stats["histogram_bin_edges"]

# Handle shared vs per-channel edges
if isinstance(bin_edges[0], list):
    edges_per_channel = bin_edges
else:
    edges_per_channel = [bin_edges] * len(channels)

# ---------------------------------------------------------------------
# Determine global non-zero support
# ---------------------------------------------------------------------

global_min_edge = float("inf")
global_max_edge = float("-inf")

for i in range(len(channels)):
    counts = np.array(hist_counts[i])
    edges = np.array(edges_per_channel[i])

    nonzero = np.nonzero(counts)[0]
    if len(nonzero) == 0:
        continue

    first = nonzero[0]
    last = nonzero[-1]

    global_min_edge = min(global_min_edge, edges[first])
    global_max_edge = max(global_max_edge, edges[last + 1])

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(9, 5))

for i, ch in enumerate(channels):
    if ch not in CHANNEL_STYLES:
        continue

    counts = np.array(hist_counts[i], dtype=np.float64)
    edges = np.array(edges_per_channel[i], dtype=np.float64)

    # Clip to global support
    mask = (edges[:-1] >= global_min_edge) & (edges[1:] <= global_max_edge)

    counts = counts[mask]
    edges = edges[np.concatenate([mask, [False]]) | np.concatenate([[False], mask])]

    if counts.sum() == 0:
        continue

    # Density normalization
    bin_width = edges[1] - edges[0]
    density = counts / (counts.sum() * bin_width)

    centers = 0.5 * (edges[:-1] + edges[1:])
    width = bin_width

    style = CHANNEL_STYLES[ch]

    ax.bar(
        centers,
        density,
        width=width,
        color=style["color"],
        alpha=ALPHA,
        label=style["label"],
        edgecolor="none",
    )

# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------

ax.set_xlim(global_min_edge, global_max_edge)

ax.set_xlabel("Calibrated I/F", fontsize=12)
ax.set_ylabel("Density", fontsize=12)

ax.set_title("Mars HiRISE Channel Intensity Distributions", fontsize=14, pad=10)

ax.legend(frameon=True)
ax.grid(True, alpha=0.3)

fig.tight_layout()

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

png_path = OUTPUT_DIR / "combined_histogram.png"
pdf_path = OUTPUT_DIR / "combined_histogram.pdf"

fig.savefig(png_path, dpi=300)
fig.savefig(pdf_path)

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
