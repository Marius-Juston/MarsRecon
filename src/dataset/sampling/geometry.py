"""Geometric patch-packing utilities for footprint polygons.

This module computes, for a given *target polygon* and a patch size, the set
of patch centre locations such that every resulting patch has a guaranteed
minimum fractional overlap with the target.  It then packs patches inside
that *valid-centre region* with optional controllable overlap between
adjacent patches.

Two algorithms are provided:

1. :func:`generate_valid_center_region` — constructs the region in which a
   patch centre may be placed while preserving a minimum intersection area
   with the target.  Convex targets are supported; non-convex targets will
   still usually work provided the area-vs-radius function is monotone
   along rays from the centroid.

2. :func:`pack_patches_independent_strips` — sweeps rows and columns
   through the valid-centre region to place as many patch centres as
   possible, supporting an overlap parameter (0 ≤ overlap < 1) that
   controls the fractional overlap between adjacent patches.

Patch dimensions are expressed as ``(height, width)`` to match TorchGeo's
convention.  A scalar is accepted for square patches.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.optimize import brentq
from shapely.affinity import translate
from shapely.geometry import LineString, Point, Polygon, box

__all__ = [
    "generate_valid_center_region",
    "pack_patches_independent_strips",
    "pack_patches_grid",
    "as_patch_size",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def as_patch_size(size: float | tuple[float, float]) -> tuple[float, float]:
    """Normalise ``size`` to ``(height, width)``."""
    if isinstance(size, (int, float)):
        return float(size), float(size)
    h, w = size
    return float(h), float(w)


def _get_optimized_angles(
        target_polygon: Polygon,
        centroid: Point,
        extra_rays_per_edge: int = 2,
) -> np.ndarray:
    """Return a set of angles (radians) from the centroid to sample the boundary.

    Includes every vertex direction plus ``extra_rays_per_edge`` uniformly
    spaced angles between consecutive vertices so that the resulting polygon
    hugs the true valid region more closely.
    """
    cx, cy = centroid.x, centroid.y
    coords = list(target_polygon.exterior.coords)[:-1]
    angles: list[float] = []
    for i in range(len(coords)):
        vx, vy = coords[i]
        a_v = np.arctan2(vy - cy, vx - cx) % (2 * np.pi)
        angles.append(a_v)

        nx, ny = coords[(i + 1) % len(coords)]
        a_n = np.arctan2(ny - cy, nx - cx) % (2 * np.pi)
        if a_n < a_v:
            a_n += 2 * np.pi
        step = (a_n - a_v) / (extra_rays_per_edge + 1)
        for j in range(1, extra_rays_per_edge + 1):
            angles.append((a_v + j * step) % (2 * np.pi))
    return np.sort(np.unique(angles))


# ---------------------------------------------------------------------------
# Valid centre region
# ---------------------------------------------------------------------------


def generate_valid_center_region(
        target_polygon: Polygon,
        patch_size: float | tuple[float, float],
        overlap_percentage: float,
        extra_rays_per_edge: int = 3,
) -> Polygon:
    """Return the polygon of valid patch-centre locations.

    For a rectangular patch of size ``patch_size`` (height, width), a centre
    inside the returned polygon is guaranteed to produce a patch whose
    intersection area with ``target_polygon`` is at least
    ``overlap_percentage × (patch_width × patch_height)``.

    Args:
        target_polygon: The footprint polygon (typically convex).
        patch_size: Scalar (square) or ``(height, width)`` tuple.
        overlap_percentage: Required minimum fractional overlap in ``[0, 1]``.
        extra_rays_per_edge: Resolution of the boundary sampling. Higher
            values give tighter approximations at cost of speed.

    Returns:
        A Shapely :class:`Polygon`.  An empty polygon is returned when the
        target is too small to satisfy the overlap constraint at any centre.
    """
    if not (0.0 <= overlap_percentage <= 1.0):
        raise ValueError("overlap_percentage must be in [0, 1]")

    size_h, size_w = as_patch_size(patch_size)
    if size_h <= 0 or size_w <= 0:
        raise ValueError("patch_size must be positive")

    patch_area = size_h * size_w
    target_area = patch_area * overlap_percentage
    half_h = size_h / 2.0
    half_w = size_w / 2.0
    base_patch = box(-half_w, -half_h, half_w, half_h)

    def area_difference(r: float, theta: float, cx: float, cy: float) -> float:
        test_x = cx + r * np.cos(theta)
        test_y = cy + r * np.sin(theta)
        translated = translate(base_patch, xoff=test_x, yoff=test_y)
        return target_polygon.intersection(translated).area - target_area

    centroid = target_polygon.centroid
    x0, y0 = centroid.x, centroid.y

    # Infeasible: even the centroid can't host a patch with enough overlap.
    if area_difference(0.0, 0.0, x0, y0) < 0:
        return Polygon()

    coords = np.array(target_polygon.exterior.coords)
    diag = float(np.hypot(size_h, size_w))
    r_max = float(np.max(np.linalg.norm(coords - [x0, y0], axis=1))) + diag

    angles = _get_optimized_angles(target_polygon, centroid, extra_rays_per_edge)
    boundary_points: list[tuple[float, float]] = []

    for theta in angles:
        try:
            r_star = brentq(area_difference, a=0.0, b=r_max, args=(theta, x0, y0))
            boundary_points.append(
                (x0 + r_star * np.cos(theta), y0 + r_star * np.sin(theta))
            )
        except ValueError:
            # No sign change on this ray; skip.
            continue

    if len(boundary_points) < 3:
        return Polygon()

    region = Polygon(boundary_points)
    if not region.is_valid:
        region = region.buffer(0)  # fix self-intersections, common after sampling
    if region.is_empty or region.geom_type != "Polygon":
        return Polygon()
    return region


# ---------------------------------------------------------------------------
# Packing algorithms
# ---------------------------------------------------------------------------


def pack_patches_independent_strips(
        valid_region: Polygon,
        patch_size: float | tuple[float, float],
        patch_overlap: float = 0.0,
        phase_steps: int = 20,
) -> list[tuple[float, float]]:
    """Pack patch centres inside ``valid_region`` using row/column sweeps.

    Tries both horizontal and vertical sweeps across a range of phase
    offsets and returns the configuration with the most centres.

    Args:
        valid_region: Output of :func:`generate_valid_center_region`.
        patch_size: Scalar (square) or ``(height, width)`` tuple.
        patch_overlap: Fractional overlap between adjacent patches, in
            ``[0, 1)``.  ``0.0`` yields edge-to-edge packing, ``0.5`` means
            adjacent patches share 50% of their side length, etc.
        phase_steps: Number of phase offsets to try per direction.

    Returns:
        List of ``(x, y)`` centre coordinates (possibly empty).
    """
    if valid_region.is_empty:
        return []
    if not (0.0 <= patch_overlap < 1.0):
        raise ValueError("patch_overlap must be in [0, 1)")

    size_h, size_w = as_patch_size(patch_size)
    stride_x = size_w * (1.0 - patch_overlap)
    stride_y = size_h * (1.0 - patch_overlap)
    if stride_x <= 0 or stride_y <= 0:  # pragma: no cover - defensive; size>0 and overlap in [0,1) guarantee stride>0
        raise ValueError("patch_overlap produces non-positive stride")

    minx, miny, maxx, maxy = valid_region.bounds
    best_points: list[tuple[float, float]] = []
    max_count = 0

    # --- Horizontal rows ---
    for dy in np.linspace(0.0, stride_y, phase_steps, endpoint=False):
        pts: list[tuple[float, float]] = []
        y = miny + dy
        while y <= maxy:
            sweep = LineString([(minx - 1.0, y), (maxx + 1.0, y)])
            inter = valid_region.intersection(sweep)
            if not inter.is_empty:
                # Enumerate each connected segment on this row.
                segments: Iterable = (
                    inter.geoms if inter.geom_type == "MultiLineString" else [inter]
                )
                for seg in segments:
                    if seg.geom_type != "LineString" or seg.is_empty:
                        continue
                    ix_min, _, ix_max, _ = seg.bounds
                    width = ix_max - ix_min
                    if width < 0:  # pragma: no cover - defensive; .bounds width is never negative
                        continue
                    count = int(width // stride_x) + 1
                    used = (count - 1) * stride_x
                    x0 = ix_min + (width - used) / 2.0
                    for i in range(count):
                        pts.append((x0 + i * stride_x, y))
            y += stride_y
        if len(pts) > max_count:
            max_count = len(pts)
            best_points = pts

    # --- Vertical columns ---
    for dx in np.linspace(0.0, stride_x, phase_steps, endpoint=False):
        pts = []
        x = minx + dx
        while x <= maxx:
            sweep = LineString([(x, miny - 1.0), (x, maxy + 1.0)])
            inter = valid_region.intersection(sweep)
            if not inter.is_empty:
                segments = (
                    inter.geoms if inter.geom_type == "MultiLineString" else [inter]
                )
                for seg in segments:
                    if seg.geom_type != "LineString" or seg.is_empty:
                        continue
                    _, iy_min, _, iy_max = seg.bounds
                    height = iy_max - iy_min
                    if height < 0:  # pragma: no cover - defensive; .bounds height is never negative
                        continue
                    count = int(height // stride_y) + 1
                    used = (count - 1) * stride_y
                    y0 = iy_min + (height - used) / 2.0
                    for i in range(count):
                        pts.append((x, y0 + i * stride_y))
            x += stride_x
        if len(pts) > max_count:
            max_count = len(pts)
            best_points = pts

    return best_points


def pack_patches_grid(
        valid_region: Polygon,
        patch_size: float | tuple[float, float],
        patch_overlap: float = 0.0,
        phase_steps: int = 10,
) -> list[tuple[float, float]]:
    """Pack patch centres using a rigid rectangular grid with phase search.

    This is the original strategy: a single axis-aligned grid is swept in
    both x and y phases and the highest-count phase wins.  Produces a
    regular lattice of centres (unlike the row/column sweep, which relaxes
    column alignment between rows).
    """
    if valid_region.is_empty:
        return []
    if not (0.0 <= patch_overlap < 1.0):
        raise ValueError("patch_overlap must be in [0, 1)")

    size_h, size_w = as_patch_size(patch_size)
    stride_x = size_w * (1.0 - patch_overlap)
    stride_y = size_h * (1.0 - patch_overlap)

    minx, miny, maxx, maxy = valid_region.bounds
    best_points: list[tuple[float, float]] = []
    max_count = -1

    # Prepare shapely prep for fast containment
    from shapely.prepared import prep

    prepared = prep(valid_region)

    for dx in np.linspace(0.0, stride_x, phase_steps, endpoint=False):
        for dy in np.linspace(0.0, stride_y, phase_steps, endpoint=False):
            xs = np.arange(minx + dx, maxx + stride_x, stride_x)
            ys = np.arange(miny + dy, maxy + stride_y, stride_y)
            pts = [
                (float(x), float(y))
                for y in ys
                for x in xs
                if prepared.contains(Point(x, y))
            ]
            if len(pts) > max_count:
                max_count = len(pts)
                best_points = pts
    return best_points
