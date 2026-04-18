"""Publication-quality visualisations for :mod:`hirise_sampler`.

This module provides figures that compare the two centre-placement strategies
offered by :class:`HiRISEGeoSampler` — ``"simple"`` (bbox grid + overlap
filter) and ``"optimal"`` (valid-centre region + geometric packing).

Entry points
------------

* :func:`visualize_single_strip` — hero figure: one strip, both algorithms
  side-by-side with per-patch intersections shaded and coverage statistics
  printed.

* :func:`visualize_patch_overlap_sweep` — how patch count varies with the
  ``patch_overlap`` knob in optimal mode.

* :func:`visualize_min_overlap_sweep` — how patch count varies with the
  ``min_overlap`` threshold for both algorithms.

* :func:`visualize_multi_strip_grid` — a gallery of strips with both
  algorithms overlaid, useful for showing robustness across shapes.

* :func:`visualize_real_dataset` — overlay the centres from two already-built
  :class:`HiRISEGeoSampler` instances onto the dataset's strip geometries.
  Use this on actual HiRISE data.

* :func:`benchmark_algorithms` — tabular comparison (returns a DataFrame).

* :func:`make_publication_figure` — the multi-panel headline figure combining
  coverage, parameter sweep and per-strip gain histograms.

All figures are saved to disk at high DPI and additionally returned as the
matplotlib :class:`~matplotlib.figure.Figure` for downstream customisation.

Usage without the full TorchGeo dataset
---------------------------------------

``visualize_single_strip``, ``visualize_patch_overlap_sweep``,
``visualize_min_overlap_sweep``, ``visualize_multi_strip_grid``,
``benchmark_algorithms`` and ``make_publication_figure`` all operate directly
on Shapely polygons, so they can be run without constructing a real
:class:`HiRISEGeoSampler`.  :func:`synthetic_hirise_strips` provides
realistic long-narrow rotated polygons for demos and unit tests.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.axisartist import Axes
from shapely.affinity import rotate
from shapely.geometry import Polygon, box

from dataset.min_square_overlap import (
    as_patch_size,
    generate_valid_center_region,
    pack_patches_independent_strips,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Styling constants.  Centralised so every figure in the paper matches.
# ---------------------------------------------------------------------------

# Colour palette chosen to be colourblind-friendly and print-safe.
COLOR_FOOTPRINT = "#4B6F9E"  # steel blue — the strip polygon
COLOR_VALID = "#D4AF37"  # warm gold — the valid-centre region
COLOR_SIMPLE = "#2E7D4A"  # forest green — simple-mode patches
COLOR_OPTIMAL = "#B3321A"  # crimson — optimal-mode patches
COLOR_OVERLAP = "#555555"  # grey — intersections


def _apply_publication_style() -> None:
    """Apply a clean, journal-friendly matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 250,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.4,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.7",
    })


def save_fig(fig: Figure, output_path: str | Path | None) -> None:
    if output_path is not None:
        if not isinstance(output_path, Path):
            output_path = Path(output_path)

        for suf in [".png", ".pdf"]:
            new_path = output_path.with_suffix(suf)

            fig.savefig(new_path, bbox_inches="tight")
            logger.info("Saved %s", new_path)


# ---------------------------------------------------------------------------
# Synthetic HiRISE-like geometry
# ---------------------------------------------------------------------------


def synthetic_hirise_strips(
        n: int = 8,
        seed: int = 0,
        length_range: tuple[float, float] = (18.0, 35.0),
        width_range: tuple[float, float] = (3.0, 6.0),
        rotation_range: tuple[float, float] = (-45.0, 45.0),
        chamfer: bool = True,
) -> list[Polygon]:
    """Generate ``n`` long-narrow rotated polygons mimicking HiRISE strips.

    Real HiRISE strips are long (tens of km) in the along-track direction,
    narrow (a few km) in the cross-track direction, and typically rotated a
    few degrees from axis-aligned due to orbital mechanics.

    Args:
        n: Number of strips to generate.
        seed: RNG seed for reproducibility.
        length_range: Range of lengths (arbitrary units).
        width_range: Range of widths.
        rotation_range: Range of rotation angles in degrees.
        chamfer: If True, corners are chamfered (slightly more realistic).

    Returns:
        List of Shapely polygons centred near the origin.
    """
    rng = np.random.default_rng(seed)
    strips: list[Polygon] = []
    for _ in range(n):
        L = float(rng.uniform(*length_range))
        W = float(rng.uniform(*width_range))
        angle = float(rng.uniform(*rotation_range))
        if chamfer:
            c = min(W, L) * 0.12
            poly = Polygon([
                (c, 0), (L - c, 0),
                (L, c), (L, W - c),
                (L - c, W), (c, W),
                (0, W - c), (0, c),
            ])
        else:
            poly = Polygon([(0, 0), (L, 0), (L, W), (0, W)])
        poly = rotate(poly, angle, origin="center")
        strips.append(poly)
    return strips


# ---------------------------------------------------------------------------
# Low-level centre generators used by the visualisations (and mirroring the
# sampler's behaviour).  Kept separate so plots can run without TorchGeo.
# ---------------------------------------------------------------------------


def _simple_centers(
        footprint: Polygon,
        patch_size: float | tuple[float, float],
        min_overlap: float,
        stride: float | tuple[float, float] | None = None,
) -> list[tuple[float, float]]:
    size_h, size_w = as_patch_size(patch_size)
    stride_h, stride_w = as_patch_size(stride if stride is not None else (size_h, size_w))
    half_h, half_w = size_h / 2.0, size_w / 2.0

    _edge_inset = max(size_h, size_w) * 0.05
    try:
        effective = footprint.buffer(-_edge_inset)
        if effective.is_empty or not effective.is_valid:
            effective = footprint
    except Exception:
        effective = footprint

    minx, miny, maxx, maxy = effective.bounds
    if (maxx - minx) < size_w or (maxy - miny) < size_h:
        return []

    xs = np.arange(minx + half_w, maxx - half_w + stride_w * 1e-6, stride_w)
    ys = np.arange(miny + half_h, maxy - half_h + stride_h * 1e-6, stride_h)

    out: list[tuple[float, float]] = []
    for cy in ys:
        for cx in xs:
            patch = box(cx - half_w, cy - half_h, cx + half_w, cy + half_h)
            if effective.intersection(patch).area / patch.area > min_overlap:
                out.append((float(cx), float(cy)))
    return out


def _optimal_centers(
        footprint: Polygon,
        patch_size: float | tuple[float, float],
        min_overlap: float,
        patch_overlap: float = 0.0,
        phase_steps: int = 20,
        rays: int = 3,
) -> tuple[list[tuple[float, float]], Polygon]:
    """Return (centres, valid_region).  valid_region may be empty."""
    vr = generate_valid_center_region(
        footprint, patch_size, min_overlap, extra_rays_per_edge=rays
    )
    if vr.is_empty:
        return [], vr
    pts = pack_patches_independent_strips(
        vr, patch_size, patch_overlap=patch_overlap, phase_steps=phase_steps
    )
    return pts, vr


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


def _draw_polygon(ax, poly: Polygon, *, edge: str, face: str | None = None,
                  alpha: float = 0.25, lw: float = 1.8, ls: str = "-",
                  label: str | None = None):
    """Draw a single Shapely polygon (exterior only) on ``ax``."""
    if poly.is_empty:
        return None
    if poly.geom_type == "MultiPolygon":
        first = True
        for p in poly.geoms:
            _draw_polygon(ax, p, edge=edge, face=face, alpha=alpha, lw=lw, ls=ls,
                          label=label if first else None)
            first = False
        return None
    xs, ys = poly.exterior.xy
    ax.plot(xs, ys, color=edge, linewidth=lw, linestyle=ls, label=label)
    if face is not None:
        ax.fill(xs, ys, color=face, alpha=alpha)


def _draw_patches(
        ax,
        centers: Sequence[tuple[float, float]],
        patch_size: float | tuple[float, float],
        *,
        edge: str,
        face: str,
        alpha_fill: float = 0.18,
        lw: float = 0.7,
        draw_intersection_with: Polygon | None = None,
):
    """Plot all patches at ``centers`` as thin outlined boxes."""
    size_h, size_w = as_patch_size(patch_size)
    half_h, half_w = size_h / 2.0, size_w / 2.0
    for (cx, cy) in centers:
        rect = Rectangle(
            (cx - half_w, cy - half_h), size_w, size_h,
            edgecolor=edge, facecolor=face, alpha=alpha_fill, linewidth=lw,
        )
        ax.add_patch(rect)
        if draw_intersection_with is not None:
            patch = box(cx - half_w, cy - half_h, cx + half_w, cy + half_h)
            inter = draw_intersection_with.intersection(patch)
            _draw_polygon(ax, inter, edge=COLOR_OVERLAP, face=COLOR_OVERLAP,
                          alpha=0.22, lw=0.0)


def _coverage_stats(
        footprint: Polygon,
        centers: Sequence[tuple[float, float]],
        patch_size: float | tuple[float, float],
) -> dict[str, float]:
    """Compute aggregate coverage statistics for a set of centres."""
    size_h, size_w = as_patch_size(patch_size)
    half_h, half_w = size_h / 2.0, size_w / 2.0
    n = len(centers)
    patch_area = size_h * size_w
    if n == 0:
        return {
            "n_patches": 0,
            "footprint_area": float(footprint.area),
            "total_intersection_area": 0.0,
            "unique_coverage_area": 0.0,
            "coverage_fraction": 0.0,
            "mean_overlap_fraction": 0.0,
        }
    from shapely.ops import unary_union
    patches = [
        box(cx - half_w, cy - half_h, cx + half_w, cy + half_h)
        for (cx, cy) in centers
    ]
    inters = [footprint.intersection(p) for p in patches]
    total = float(sum(i.area for i in inters))
    unique = float(unary_union(inters).area)
    return {
        "n_patches": n,
        "footprint_area": float(footprint.area),
        "total_intersection_area": total,
        "unique_coverage_area": unique,
        "coverage_fraction": unique / float(footprint.area) if footprint.area > 0 else 0.0,
        "mean_overlap_fraction": total / (n * patch_area),
    }


# ---------------------------------------------------------------------------
# 1) Single-strip comparison (hero figure)
# ---------------------------------------------------------------------------


def visualize_single_strip(
        footprint: Polygon,
        patch_size: float | tuple[float, float],
        *,
        min_overlap: float = 0.5,
        patch_overlap: float = 0.0,
        stride: float | tuple[float, float] | None = None,
        output_path: str | Path | None = None,
        title: str | None = None,
        show_valid_region: bool = True,
        show_intersections: bool = True,
        figsize: tuple[float, float] = (12.0, 5.6),
) -> plt.Figure:
    """Side-by-side figure comparing simple vs optimal on a single strip.

    Args:
        footprint: Shapely polygon of the strip footprint.
        patch_size: Scalar (square) or ``(h, w)`` tuple.
        min_overlap: Minimum patch/footprint overlap fraction.
        patch_overlap: Fractional overlap between adjacent patches
            (optimal mode only).
        stride: Stride for simple mode (defaults to ``patch_size``).
        output_path: Where to save the figure (PNG or SVG).  ``None``
            does not save.
        title: Optional super-title.
        show_valid_region: Overlay the valid-centre region on the optimal panel.
        show_intersections: Shade each patch's intersection with the strip.
        figsize: Figure size in inches.
    """
    _apply_publication_style()

    size_h, size_w = as_patch_size(patch_size)
    eq_stride_h = size_h * (1.0 - patch_overlap)
    eq_stride_w = size_w * (1.0 - patch_overlap)

    actual_stride = stride if stride is not None else (eq_stride_h, eq_stride_w)

    simple_centers = _simple_centers(footprint, patch_size, min_overlap, actual_stride)
    optimal_centers, vr = _optimal_centers(
        footprint, patch_size, min_overlap, patch_overlap=patch_overlap,
    )

    s_stats = _coverage_stats(footprint, simple_centers, patch_size)
    o_stats = _coverage_stats(footprint, optimal_centers, patch_size)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # -------- Simple --------
    ax = axes[0]
    _draw_polygon(ax, footprint, edge=COLOR_FOOTPRINT, face=COLOR_FOOTPRINT,
                  alpha=0.12, lw=2.0, label="Strip footprint")
    _draw_patches(
        ax, simple_centers, patch_size,
        edge=COLOR_SIMPLE, face=COLOR_SIMPLE,
        draw_intersection_with=footprint if show_intersections else None,
    )
    if simple_centers:
        xs, ys = zip(*simple_centers)
        ax.scatter(xs, ys, s=6, color=COLOR_SIMPLE, zorder=5, label="Patch centres")
    ax.set_title(
        f"simple  (stride = patch size)\n"
        f"n = {s_stats['n_patches']}  |  "
        f"coverage = {100 * s_stats['coverage_fraction']:.1f}%"
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)

    # -------- Optimal --------
    ax = axes[1]
    _draw_polygon(ax, footprint, edge=COLOR_FOOTPRINT, face=COLOR_FOOTPRINT,
                  alpha=0.12, lw=2.0, label="Strip footprint")
    if show_valid_region and not vr.is_empty:
        _draw_polygon(ax, vr, edge=COLOR_VALID, face=COLOR_VALID,
                      alpha=0.18, lw=1.5, ls="--", label="Valid-centre region")
    _draw_patches(
        ax, optimal_centers, patch_size,
        edge=COLOR_OPTIMAL, face=COLOR_OPTIMAL,
        draw_intersection_with=footprint if show_intersections else None,
    )
    if optimal_centers:
        xs, ys = zip(*optimal_centers)
        ax.scatter(xs, ys, s=6, color=COLOR_OPTIMAL, zorder=5, label="Patch centres")
    ax.set_title(
        f"optimal  (patch_overlap = {patch_overlap:.2f})\n"
        f"n = {o_stats['n_patches']}  |  "
        f"coverage = {100 * o_stats['coverage_fraction']:.1f}%"
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)

    # Lock axis limits to the same bounds for fair visual comparison.
    minx, miny, maxx, maxy = footprint.bounds
    pad = 0.08 * max(maxx - minx, maxy - miny)
    for a in axes:
        a.set_xlim(minx - pad, maxx + pad)
        a.set_ylim(miny - pad, maxy + pad)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()

    if output_path is not None:
        save_fig(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 2) Parameter sweeps
# ---------------------------------------------------------------------------


def visualize_patch_overlap_sweep(
        footprint: Polygon,
        patch_size: float | tuple[float, float],
        *,
        min_overlap: float = 0.5,
        overlap_range: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
        output_path: str | Path | None = None,
        title: str | None = None,
        figsize: tuple[float, float] = (9.0, 4.5),
) -> plt.Figure:
    """Plot patch count & coverage vs ``patch_overlap`` in optimal mode."""
    _apply_publication_style()

    opt_counts, opt_covs = [], []
    simp_counts, simp_covs = [], []

    size_h, size_w = as_patch_size(patch_size)

    for po in overlap_range:
        # Calculate optimal stats
        opt_cen, _ = _optimal_centers(footprint, patch_size, min_overlap, patch_overlap=po)
        opt_stat = _coverage_stats(footprint, opt_cen, patch_size)
        opt_counts.append(opt_stat["n_patches"])
        opt_covs.append(100 * opt_stat["coverage_fraction"])

        # Calculate equivalent simple stats
        eq_stride = (size_h * (1.0 - po), size_w * (1.0 - po))
        simp_cen = _simple_centers(footprint, patch_size, min_overlap, stride=eq_stride)
        simp_stat = _coverage_stats(footprint, simp_cen, patch_size)
        simp_counts.append(simp_stat["n_patches"])
        simp_covs.append(100 * simp_stat["coverage_fraction"])

    fig, (ax_n, ax_c) = plt.subplots(1, 2, figsize=figsize)
    ax_n: Axes
    ax_n.plot(overlap_range, opt_counts, "-o", color=COLOR_OPTIMAL, label="optimal")
    ax_n.plot(overlap_range, simp_counts, "-s", color=COLOR_SIMPLE, label="simple")
    # ax_n.set_yscale("log")
    ax_n.set_xlabel("patch_overlap")
    ax_n.set_ylabel("number of patches")
    ax_n.set_title("Patch count vs patch_overlap")
    ax_n.legend(fontsize=9)

    ax_c.plot(overlap_range, opt_covs, "-o", color=COLOR_OPTIMAL, label="optimal")
    ax_c.plot(overlap_range, simp_covs, "-s", color=COLOR_SIMPLE, label="simple")
    ax_c.set_ylim((0, 102))
    ax_c.set_xlabel("patch_overlap")
    ax_c.set_ylabel("coverage of footprint (%)")
    ax_c.set_title("Coverage vs patch_overlap")
    ax_c.legend(fontsize=9)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()

    if output_path is not None:
        save_fig(fig, output_path)
    return fig


def visualize_min_overlap_sweep(
        footprint: Polygon,
        patch_size: float | tuple[float, float],
        *,
        patch_overlap: float = 0.0,
        min_overlap_range: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
        output_path: str | Path | None = None,
        title: str | None = None,
        figsize: tuple[float, float] = (9.0, 4.5),
) -> plt.Figure:
    """Plot patch count for both algorithms as ``min_overlap`` varies."""
    _apply_publication_style()

    simple_counts: list[int] = []
    optimal_counts: list[int] = []
    for mo in min_overlap_range:
        simple_counts.append(len(_simple_centers(footprint, patch_size, mo)))
        c, _ = _optimal_centers(footprint, patch_size, mo, patch_overlap=patch_overlap)
        optimal_counts.append(len(c))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(min_overlap_range, simple_counts, "-o", color=COLOR_SIMPLE, label="simple")
    ax.plot(min_overlap_range, optimal_counts, "-s", color=COLOR_OPTIMAL, label="optimal")
    ax.set_xlabel("min_overlap threshold")
    ax.set_ylabel("number of valid patches")
    ax.set_title(title or f"Patch count vs min_overlap (patch_overlap = {patch_overlap:.2f})")
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        save_fig(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 3) Multi-strip gallery
# ---------------------------------------------------------------------------


def visualize_multi_strip_grid(
        footprints: Sequence[Polygon],
        patch_size: float | tuple[float, float],
        *,
        min_overlap: float = 0.5,
        patch_overlap: float = 0.0,
        n_cols: int = 3,
        output_path: str | Path | None = None,
        figsize_per: tuple[float, float] = (3.6, 3.2),
        title: str | None = None,
) -> plt.Figure:
    """Gallery: for each footprint, overlay simple (green) and optimal (red) centres.

    Cells show both point clouds simultaneously so the eye can spot where
    optimal gains extra centres along the edges.
    """
    _apply_publication_style()
    n = len(footprints)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per[0] * n_cols, figsize_per[1] * n_rows),
        squeeze=False,
    )

    for i, poly in enumerate(footprints):
        ax = axes[i // n_cols][i % n_cols]
        simple = _simple_centers(poly, patch_size, min_overlap)
        optimal, vr = _optimal_centers(poly, patch_size, min_overlap,
                                       patch_overlap=patch_overlap)

        _draw_polygon(ax, poly, edge=COLOR_FOOTPRINT, face=COLOR_FOOTPRINT,
                      alpha=0.1, lw=1.5)
        if not vr.is_empty:
            _draw_polygon(ax, vr, edge=COLOR_VALID, face=COLOR_VALID,
                          alpha=0.12, lw=1.0, ls="--")

        if simple:
            xs, ys = zip(*simple)
            ax.scatter(xs, ys, s=14, color=COLOR_SIMPLE, marker="o",
                       label=f"simple ({len(simple)})", alpha=0.85)
        if optimal:
            xs, ys = zip(*optimal)
            ax.scatter(xs, ys, s=14, color=COLOR_OPTIMAL, marker="x",
                       linewidths=1.4, label=f"optimal ({len(optimal)})")

        gain = (len(optimal) - len(simple)) / max(1, len(simple)) * 100.0
        ax.set_title(f"strip {i}  (Δ = {gain:+.0f}%)", fontsize=10)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=7.5)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()

    if output_path is not None:
        save_fig(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 4) Real-dataset visualisation
# ---------------------------------------------------------------------------


def visualize_real_dataset(
        sampler_simple,
        sampler_optimal,
        *,
        output_path: str | Path | None = None,
        max_strips: int | None = None,
        title: str | None = None,
        figsize: tuple[float, float] = (12.0, 7.0),
) -> plt.Figure:
    """Overlay both samplers' centres onto their dataset's strip polygons.

    Both sampler instances must share the same underlying dataset and split.
    The figure renders every in-split strip geometry and, on top, the
    centres from each algorithm in distinct colours, so the empirical gain
    from switching to ``"optimal"`` on real HiRISE data is visible at a
    glance.

    Args:
        sampler_simple: A :class:`HiRISEGeoSampler` built with
            ``center_mode="simple"``.
        sampler_optimal: A :class:`HiRISEGeoSampler` built with
            ``center_mode="optimal"``.
        output_path: Where to save the figure.
        max_strips: If not None, only render the first ``max_strips`` in-split
            strips (the centre scatter is always drawn in full).
        title: Optional super-title.
        figsize: Figure size.
    """
    _apply_publication_style()
    assert sampler_simple.split == sampler_optimal.split, (
        "Both samplers must be on the same split"
    )

    fig, (ax_map, ax_bar) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2.2, 1.0]}
    )

    # --- left: map overlay ---
    strips_shown = 0
    for i in range(len(sampler_simple.index)):
        if sampler_simple._assignments.get(i) != sampler_simple.split:
            continue
        if max_strips is not None and strips_shown >= max_strips:
            break
        poly = sampler_simple.index.geometry.iloc[i]
        _draw_polygon(ax_map, poly, edge=COLOR_FOOTPRINT, face=COLOR_FOOTPRINT,
                      alpha=0.08, lw=0.8)
        strips_shown += 1

    s_xy = np.array([(c[0], c[1]) for c in sampler_simple._centers]) \
        if sampler_simple._centers else np.empty((0, 2))
    o_xy = np.array([(c[0], c[1]) for c in sampler_optimal._centers]) \
        if sampler_optimal._centers else np.empty((0, 2))

    if len(s_xy):
        ax_map.scatter(s_xy[:, 0], s_xy[:, 1], s=2.0, color=COLOR_SIMPLE,
                       alpha=0.55, label=f"simple ({len(s_xy)})", zorder=3)
    if len(o_xy):
        ax_map.scatter(o_xy[:, 0], o_xy[:, 1], s=2.0, color=COLOR_OPTIMAL,
                       alpha=0.55, label=f"optimal ({len(o_xy)})", zorder=4)

    ax_map.set_aspect("equal")
    ax_map.set_title(f"Sampler centres on real data  (split = {sampler_simple.split!r})")
    ax_map.legend(loc="upper right", markerscale=3.0)
    ax_map.set_xlabel("x (CRS)")
    ax_map.set_ylabel("y (CRS)")

    # --- right: per-strip bar chart of patch count ---
    s_counts_map = {s["pair_idx"]: s["n_centers"]
                    for s in sampler_simple.per_strip_stats}
    o_counts_map = {s["pair_idx"]: s["n_centers"]
                    for s in sampler_optimal.per_strip_stats}
    idxs = sorted(set(s_counts_map) | set(o_counts_map))
    s_counts = [s_counts_map.get(i, 0) for i in idxs]
    o_counts = [o_counts_map.get(i, 0) for i in idxs]

    order = np.argsort(s_counts)
    s_counts = np.asarray(s_counts)[order]
    o_counts = np.asarray(o_counts)[order]

    y = np.arange(len(idxs))
    bar_h = 0.42
    ax_bar.barh(y - bar_h / 2, s_counts, height=bar_h,
                color=COLOR_SIMPLE, label="simple")
    ax_bar.barh(y + bar_h / 2, o_counts, height=bar_h,
                color=COLOR_OPTIMAL, label="optimal")
    ax_bar.set_xlabel("patches per strip")
    ax_bar.set_ylabel("strip (sorted)")
    ax_bar.set_title("Per-strip yield")
    ax_bar.set_yticks([])
    ax_bar.legend(fontsize=9)

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()

    if output_path is not None:
        save_fig(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 5) Benchmarking
# ---------------------------------------------------------------------------


def benchmark_algorithms(
        footprints: Sequence[Polygon],
        patch_size: float | tuple[float, float],
        *,
        min_overlap: float = 0.5,
        patch_overlap: float = 0.0,
) -> pd.DataFrame:
    """Return a per-strip comparison DataFrame.

    Columns:
        ``strip``, ``simple_n``, ``optimal_n``, ``gain_pct``,
        ``simple_coverage``, ``optimal_coverage``, ``footprint_area``.
    """
    size_h, size_w = as_patch_size(patch_size)
    eq_stride = (size_h * (1.0 - patch_overlap), size_w * (1.0 - patch_overlap))

    rows: list[dict[str, Any]] = []
    for i, poly in enumerate(footprints):
        s = _simple_centers(poly, patch_size, min_overlap, stride=eq_stride)
        o, _ = _optimal_centers(poly, patch_size, min_overlap,
                                patch_overlap=patch_overlap)
        s_stat = _coverage_stats(poly, s, patch_size)
        o_stat = _coverage_stats(poly, o, patch_size)
        simple_n = len(s)
        optimal_n = len(o)
        rows.append({
            "strip": i,
            "simple_n": simple_n,
            "optimal_n": optimal_n,
            "gain_pct": (optimal_n - simple_n) / max(1, simple_n) * 100.0,
            "simple_coverage": s_stat["coverage_fraction"],
            "optimal_coverage": o_stat["coverage_fraction"],
            "footprint_area": float(poly.area),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6) Headline publication figure
# ---------------------------------------------------------------------------


def make_publication_figure(
        footprints: Sequence[Polygon],
        patch_size: float | tuple[float, float],
        *,
        min_overlap: float = 0.5,
        patch_overlap: float = 0.0,
        hero_strip_idx: int = 0,
        overlap_range: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7),
        output_path: str | Path | None = None,
        figsize: tuple[float, float] = (13.0, 9.5),
        title: str = "HiRISE sampler: simple vs optimal patch placement",
) -> plt.Figure:
    """Four-panel headline figure combining hero, sweep, gallery and histogram.

    Panel layout::

        ┌─────────────────────┬─────────────────────┐
        │  (A) simple, hero   │  (B) optimal, hero  │
        ├──────────┬──────────┴─────────────────────┤
        │  (C) sweep          │  (D) per-strip gain │
        └─────────────────────┴─────────────────────┘
    """
    _apply_publication_style()

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.32, wspace=0.22)

    hero = footprints[hero_strip_idx]
    # --- (A) hero: simple ---
    axA = fig.add_subplot(gs[0, 0])

    size_h, size_w = as_patch_size(patch_size)
    eq_stride = (size_h * (1.0 - patch_overlap), size_w * (1.0 - patch_overlap))

    simple = _simple_centers(hero, patch_size, min_overlap, stride=eq_stride)
    _draw_polygon(axA, hero, edge=COLOR_FOOTPRINT, face=COLOR_FOOTPRINT,
                  alpha=0.12, lw=2.0, label="Strip footprint")
    _draw_patches(axA, simple, patch_size, edge=COLOR_SIMPLE, face=COLOR_SIMPLE,
                  draw_intersection_with=hero)
    if simple:
        xs, ys = zip(*simple)
        axA.scatter(xs, ys, s=8, color=COLOR_SIMPLE, zorder=5)
    s_cov = _coverage_stats(hero, simple, patch_size)["coverage_fraction"]
    axA.set_title(f"(A) simple — n = {len(simple)}, coverage = {100 * s_cov:.1f}%")
    axA.set_aspect("equal")

    # --- (B) hero: optimal ---
    axB = fig.add_subplot(gs[0, 1])
    optimal, vr = _optimal_centers(hero, patch_size, min_overlap,
                                   patch_overlap=patch_overlap)
    _draw_polygon(axB, hero, edge=COLOR_FOOTPRINT, face=COLOR_FOOTPRINT,
                  alpha=0.12, lw=2.0, label="Strip footprint")
    if not vr.is_empty:
        _draw_polygon(axB, vr, edge=COLOR_VALID, face=COLOR_VALID,
                      alpha=0.15, lw=1.3, ls="--", label="Valid-centre region")
    _draw_patches(axB, optimal, patch_size, edge=COLOR_OPTIMAL, face=COLOR_OPTIMAL,
                  draw_intersection_with=hero)
    if optimal:
        xs, ys = zip(*optimal)
        axB.scatter(xs, ys, s=8, color=COLOR_OPTIMAL, zorder=5)
    o_cov = _coverage_stats(hero, optimal, patch_size)["coverage_fraction"]
    axB.set_title(
        f"(B) optimal (patch_overlap = {patch_overlap:.2f}) — "
        f"n = {len(optimal)}, coverage = {100 * o_cov:.1f}%"
    )
    axB.set_aspect("equal")

    minx, miny, maxx, maxy = hero.bounds
    pad = 0.08 * max(maxx - minx, maxy - miny)
    for a in (axA, axB):
        a.set_xlim(minx - pad, maxx + pad)
        a.set_ylim(miny - pad, maxy + pad)
        a.legend(loc="upper right", fontsize=8)

    # --- (C) sweep: patch count vs patch_overlap (mean across strips) ---
    axC = fig.add_subplot(gs[1, 0])



    mean_opt = []
    mean_simp = []

    for po in overlap_range:
        # Optimal mean patch count
        c_opt = [len(_optimal_centers(p, patch_size, min_overlap, patch_overlap=po)[0])
                 for p in footprints]
        mean_opt.append(np.mean(c_opt))

        # Simple mean patch count (using equivalent stride)
        eq_stride = (size_h * (1.0 - po), size_w * (1.0 - po))
        c_simp = [len(_simple_centers(p, patch_size, min_overlap, stride=eq_stride))
                  for p in footprints]
        mean_simp.append(np.mean(c_simp))

    axC.plot(overlap_range, mean_opt, "-o", color=COLOR_OPTIMAL, label="optimal")
    axC.plot(overlap_range, mean_simp, "-s", color=COLOR_SIMPLE, label="simple (equivalent stride)")

    axC.set_xlabel("patch_overlap")
    axC.set_ylabel("mean patches / strip")
    axC.set_title("(C) Patch density vs patch_overlap")
    axC.legend(fontsize=9)

    # --- (D) per-strip gain histogram ---
    axD = fig.add_subplot(gs[1, 1])
    df = benchmark_algorithms(footprints, patch_size,
                              min_overlap=min_overlap, patch_overlap=patch_overlap)
    axD.hist(df["gain_pct"], bins=min(20, max(5, len(df) // 2)),
             color=COLOR_OPTIMAL, alpha=0.85, edgecolor="white")
    axD.axvline(df["gain_pct"].median(), color="black", ls="--", lw=1.2,
                label=f"median = {df['gain_pct'].median():.0f}%")
    axD.set_xlabel("optimal vs simple gain (%)")
    axD.set_ylabel("# strips")
    axD.set_title("(D) Per-strip gain distribution")
    axD.legend(fontsize=9)

    fig.suptitle(title, fontsize=14, y=0.995)

    if output_path is not None:
        save_fig(fig, output_path)
    return fig


def plot_polygon(ax, poly, **kwargs):
    """Helper to plot Shapely polygons on a matplotlib axis."""
    x, y = poly.exterior.xy
    ax.plot(x, y, **kwargs)
    ax.fill(x, y, alpha=0.2, color=kwargs.get('color', 'blue'))


def vizualize_overlap_polygon(output_path: str | Path | None = None,
                              figsize: tuple[float, float] = (12.0, 10.0), ):
    # Define test parameters
    L = 4.0
    min_overlap = 0.50  # 50% overlap

    # Define a suite of convex shapes
    test_shapes = {
        "Standard Rectangle": rotate(Polygon([(0, 0), (10, 0), (10, 6), (0, 6)]), 45, origin='center'),
        "Chamfered Rect (Your Use Case)": Polygon([(1, 0), (9, 0), (10, 1), (10, 5), (9, 6), (1, 6), (0, 5), (0, 1)]),
        "Trapezoid": Polygon([(2, 0), (8, 0), (6, 6), (4, 6)]),
        "Hexagon": Polygon([(3, 0), (7, 0), (9, 4), (7, 8), (3, 8), (1, 4)])
    }

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"Valid Center Regions (Square Size: {L}x{L}, Target Overlap: {min_overlap * 100}%)", fontsize=14,
                 y=0.935)

    for ax, (title, base_poly) in zip(axes.flatten(), test_shapes.items()):
        ax.set_title(title)
        ax.set_aspect('equal')

        # 1. Plot Base Polygon
        plot_polygon(ax, base_poly, color='blue', label='Base Polygon', linewidth=2)

        try:
            # 2. Generate and Plot Valid Region
            # Using 3 extra rays per edge. For an 8-point chamfered rect, this is 8 * (1+3) = 32 rays.
            valid_region = generate_valid_center_region(base_poly, L, min_overlap, extra_rays_per_edge=3)
            plot_polygon(ax, valid_region, color='red', label='Valid Region for Center', linestyle='--', linewidth=2)

            # 3. Plot a Sample Square to prove the math
            # Grab a point on the boundary of the valid region
            sample_center = list(valid_region.exterior.coords)[0]
            half_L = L / 2.0
            sample_square = box(sample_center[0] - half_L, sample_center[1] - half_L,
                                sample_center[0] + half_L, sample_center[1] + half_L)

            # Plot the square outline and the intersection
            x, y = sample_square.exterior.xy
            ax.plot(x, y, color='green', linestyle=':', linewidth=2, label='Sample Square on Boundary')

            intersection = base_poly.intersection(sample_square)
            plot_polygon(ax, intersection, color='green')

            # Mark the exact center point
            ax.plot(sample_center[0], sample_center[1], 'ro', markersize=5)

        except Exception as e:
            ax.text(0.5, 0.5, f"Failed: {str(e)}", transform=ax.transAxes, ha='center', color='red')

        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, linestyle=':', alpha=0.6)

    save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# CLI / quick demo
# ---------------------------------------------------------------------------


def run_demo(output_dir: str | Path = "./hirise_viz_out",
             seed: int = 0,
             n_strips: int = 9,
             patch_size: float = 1.5,
             min_overlap: float = 0.5) -> None:
    """Run every visualisation on synthetic HiRISE-like strips.

    Creates a directory of publication-ready PNG + SVG figures.  Intended
    as a smoke test and as the basis for the figures in a manuscript.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strips = synthetic_hirise_strips(n=n_strips, seed=seed)

    # (1) hero at 0% and 50% patch overlap
    for po in (0.0, 0.5):
        tag = f"{int(po * 100):02d}"
        visualize_single_strip(
            strips[0], patch_size,
            min_overlap=min_overlap, patch_overlap=po,
            output_path=output_dir / f"hero_single_strip_po{tag}.png",
            title=f"Simple vs optimal on a HiRISE-like strip (patch_overlap = {po})",
        )
        plt.close()

    # (2) patch-overlap sweep on hero strip
    visualize_patch_overlap_sweep(
        strips[0], patch_size, min_overlap=min_overlap,
        output_path=output_dir / "sweep_patch_overlap.png",
        title="How patch_overlap controls density",
    )
    plt.close()

    # (3) min-overlap sweep
    visualize_min_overlap_sweep(
        strips[0], patch_size,
        output_path=output_dir / "sweep_min_overlap.png",
        title="Sensitivity to the min_overlap threshold",
    )
    plt.close()

    # (4) multi-strip gallery
    visualize_multi_strip_grid(
        strips, patch_size, min_overlap=min_overlap, patch_overlap=0.0,
        output_path=output_dir / "gallery_strips.png",
        title="Simple (green dots) vs optimal (red crosses) across 9 synthetic strips",
    )
    plt.close()

    # (5) benchmark CSV
    df = benchmark_algorithms(strips, patch_size, min_overlap=min_overlap,
                              patch_overlap=0.0)
    df.to_csv(output_dir / "benchmark.csv", index=False)

    # (6) headline publication figure
    make_publication_figure(
        strips, patch_size,
        min_overlap=min_overlap, patch_overlap=0.25,
        output_path=output_dir / "publication_figure.png",
    )
    plt.close()

    vizualize_overlap_polygon(output_path=output_dir / "min_overlap_polygon.png")

    logger.info("Demo complete. Files written to %s", output_dir)
    print(f"Wrote figures + benchmark.csv to {output_dir.resolve()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_demo(output_dir='../outputs/patch/figures')
