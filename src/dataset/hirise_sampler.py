"""HiRISE-aware geospatial sampler with train/val/test splitting for TorchGeo.

HiRISE satellite passes produce long, narrow, slightly rotated image strips.
Sampling patches uniformly from axis-aligned bounding boxes wastes 60–90 % of
samples on empty (zero-value) pixels that fall in the corners outside the
actual strip.

:class:`HiRISEGeoSampler` avoids this by pre-computing a grid of patch centres
that are confirmed to lie within each strip's polygon footprint, then sampling
from that pre-computed set at each epoch.  Construction runs once; iteration is
O(1) per sample.

Split support
~~~~~~~~~~~~~

The sampler partitions the dataset's stereo pairs (index rows) into
**train / val / test** subsets.  Two splitting strategies are provided:

* ``"geographic"`` — sorts stereo pairs along a spatial axis (longitude or
  latitude) and assigns contiguous blocks to each split.  This prevents
  spatial data leakage: nearby strips never appear on both sides of the split.

* ``"random"`` — assigns stereo pairs uniformly at random using a
  deterministic seed.

Both strategies support **K-fold cross-validation**.  When ``n_folds`` is set,
the data is partitioned into K equally-sized folds.  ``fold_idx`` selects which
fold is used as the test set; the remaining folds are re-split into train and
val according to ``val_fraction``.

Split assignments are **cached to disk** so that:

1. Every process in a distributed training run sees the same split.
2. Re-instantiating the sampler with identical parameters reuses the same
   assignment without recomputing.
3. The cache key incorporates all split-relevant parameters (dataset root,
   target, bbox, split method, seed, K, fold, fractions) so that changing
   any parameter produces a fresh split.

Cache files are written next to the dataset's spatial index cache (under
``<root>/.cache/``) with a filename derived from the configuration hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from collections.abc import Iterator
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from shapely.geometry import box as shapely_box
from torchgeo.datasets.geo import GeoDataset
from torchgeo.samplers import GeoSampler, Units

from dataset.mars_hirise_base import MARS_PROJECTED_CRS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Split types
# ---------------------------------------------------------------------------

VALID_SPLITS = frozenset({"train", "val", "test"})
VALID_SPLIT_METHODS = frozenset({"geographic", "random"})


# ---------------------------------------------------------------------------
# Split assignment logic
# ---------------------------------------------------------------------------

def _compute_split_assignments(
    n_pairs: int,
    pair_coords: np.ndarray | None,
    *,
    method: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    n_folds: int | None,
    fold_idx: int,
) -> dict[int, str]:
    """Assign each stereo-pair index to 'train', 'val', or 'test'.

    Returns:
        Dict mapping pair positional index → split label.
    """
    rng = np.random.default_rng(seed)

    if n_folds is not None and n_folds > 1:
        return _kfold_split(
            n_pairs, pair_coords, method=method,
            n_folds=n_folds, fold_idx=fold_idx,
            val_fraction=val_fraction, rng=rng,
        )

    # ── Standard percentage split ──
    indices = np.arange(n_pairs)

    if method == "geographic" and pair_coords is not None:
        # Sort by spatial coordinate → contiguous blocks
        order = np.argsort(pair_coords)
    else:
        # Random permutation
        order = rng.permutation(n_pairs)

    n_test = max(1, int(round(n_pairs * test_fraction)))
    n_val = max(1, int(round(n_pairs * val_fraction)))
    n_train = n_pairs - n_test - n_val

    if n_train < 1:
        raise ValueError(
            f"Split fractions leave no training data: "
            f"train={train_fraction}, val={val_fraction}, test={test_fraction} "
            f"with {n_pairs} stereo pairs."
        )

    assignments = {}
    for i, idx in enumerate(order):
        if i < n_train:
            assignments[int(idx)] = "train"
        elif i < n_train + n_val:
            assignments[int(idx)] = "val"
        else:
            assignments[int(idx)] = "test"

    return assignments


def _kfold_split(
    n_pairs: int,
    pair_coords: np.ndarray | None,
    *,
    method: str,
    n_folds: int,
    fold_idx: int,
    val_fraction: float,
    rng: np.random.Generator,
) -> dict[int, str]:
    """K-fold cross-validation split.

    The data is divided into ``n_folds`` equally-sized folds.
    ``fold_idx`` selects the test fold.  The remaining folds are
    re-split into train and val using ``val_fraction`` (relative to the
    non-test portion).
    """
    if fold_idx < 0 or fold_idx >= n_folds:
        raise ValueError(
            f"fold_idx={fold_idx} is out of range for n_folds={n_folds}. "
            f"Valid range: 0 to {n_folds - 1}."
        )

    if method == "geographic" and pair_coords is not None:
        order = np.argsort(pair_coords)
    else:
        order = rng.permutation(n_pairs)

    # Assign each position in `order` to a fold
    fold_ids = np.zeros(n_pairs, dtype=int)
    fold_size = n_pairs // n_folds
    remainder = n_pairs % n_folds

    start = 0
    for f in range(n_folds):
        # Distribute remainder across the first `remainder` folds
        end = start + fold_size + (1 if f < remainder else 0)
        fold_ids[start:end] = f
        start = end

    # Test fold
    test_mask = fold_ids == fold_idx

    # Among non-test folds, split into train/val
    non_test_positions = np.where(~test_mask)[0]
    n_non_test = len(non_test_positions)
    n_val = max(1, int(round(n_non_test * val_fraction)))

    # Deterministic val selection: use a sub-permutation seeded by fold_idx
    val_rng = np.random.default_rng(rng.integers(0, 2**31) + fold_idx)
    val_positions = set(
        val_rng.choice(non_test_positions, size=n_val, replace=False).tolist()
    )

    assignments = {}
    for pos_in_order, pair_idx in enumerate(order):
        if test_mask[pos_in_order]:
            assignments[int(pair_idx)] = "test"
        elif pos_in_order in val_positions:
            assignments[int(pair_idx)] = "val"
        else:
            assignments[int(pair_idx)] = "train"

    return assignments


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _split_cache_key(
    dataset_root: str,
    dataset_target: str | None,
    dataset_bbox: tuple | None,
    method: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    n_folds: int | None,
    fold_idx: int,
    size: tuple[float, float],
    stride: tuple[float, float],
    min_overlap: float,
    ortho_types: list[str] | None,
) -> str:
    """Compute a deterministic hash key for the split configuration."""
    key_parts = {
        "root": str(dataset_root),
        "target": dataset_target,
        "bbox": dataset_bbox,
        "method": method,
        "train_frac": train_fraction,
        "val_frac": val_fraction,
        "test_frac": test_fraction,
        "seed": seed,
        "n_folds": n_folds,
        "fold_idx": fold_idx,
        "size": size,
        "stride": stride,
        "min_overlap": min_overlap,
        "ortho_types": sorted(ortho_types) if ortho_types else None,
    }
    raw = json.dumps(key_parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _split_cache_dir(dataset_root: str) -> pathlib.Path:
    return pathlib.Path(dataset_root) / ".cache" / "sampler_splits"


def _load_cached_split(cache_path: pathlib.Path) -> dict[int, str] | None:
    if not cache_path.exists():
        return None
    try:
        with open(cache_path) as f:
            data = json.load(f)
        # Convert string keys back to ints
        return {int(k): v for k, v in data["assignments"].items()}
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("Corrupt split cache %s: %s", cache_path, e)
        return None


def _save_cached_split(
    cache_path: pathlib.Path,
    assignments: dict[int, str],
    metadata: dict,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "assignments": {str(k): v for k, v in assignments.items()},
    }
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Saved split cache: %s", cache_path)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class HiRISEGeoSampler(GeoSampler):
    """Sampler that restricts patches to within HiRISE strip polygon footprints,
    with built-in train/val/test splitting and K-fold cross-validation.

    Construction pre-computes a regular grid of candidate patch centres for
    every strip in ``dataset.index``, keeping only centres whose corresponding
    patch intersects the polygon footprint (not merely its bounding box).  At
    each epoch, ``length`` centres are drawn uniformly at random from this set.

    The split is performed at the **stereo-pair level** — entire strips are
    assigned to train, val, or test.  This prevents spatial data leakage
    (nearby terrain never appears on both sides of the split).

    Args:
        dataset: The :class:`~MarsHiRISEDTM` dataset to sample from.
        size: Patch height and width in CRS units (degrees when
            ``units=Units.CRS``) or pixels (when ``units=Units.PIXELS``).
            A single float sets both dimensions equal.
        split: Which split to sample from: ``"train"``, ``"val"``, or
            ``"test"``.
        split_fractions: ``(train, val, test)`` fractions summing to 1.0.
        split_method: ``"geographic"`` (spatially contiguous blocks) or
            ``"random"`` (uniformly at random).
        split_axis: For geographic splits: ``"longitude"`` or ``"latitude"``.
        n_folds: Number of folds for K-fold cross-validation.  ``None``
            disables K-fold and uses ``split_fractions`` directly.
        fold_idx: Which fold to use as the test set (0 to ``n_folds - 1``).
        seed: Random seed for reproducible split assignment.
        length: Number of patches to yield per epoch.  Defaults to the total
            number of pre-computed valid centres for this split.
        stride: Centre-to-centre grid spacing in the same units as ``size``.
            Defaults to ``size`` (non-overlapping grid).
        roi: Optional Shapely Polygon to further restrict the spatial domain.
        toi: Optional :class:`pandas.Interval` to restrict the temporal domain.
        units: Whether *size* and *stride* are given in CRS units or pixels.
        generator: Optional :class:`torch.Generator` for reproducible sampling.
        min_overlap: Minimum fraction of patch area that must overlap the
            strip footprint to be considered valid (default 0.5).
        replacement: Sample with replacement if ``True``.
        reuse_cache: Reuse cached split assignment if available.

    Example::

        from hirise_sampler import HiRISEGeoSampler
        from torchgeo.samplers import Units

        # Standard train/val/test
        train_sampler = HiRISEGeoSampler(
            dataset, size=0.005, split="train",
            split_fractions=(0.8, 0.1, 0.1),
            seed=42,
        )
        val_sampler = HiRISEGeoSampler(
            dataset, size=0.005, split="val",
            split_fractions=(0.8, 0.1, 0.1),
            seed=42,
        )

        # 5-fold cross-validation, fold 0 as test
        train_sampler = HiRISEGeoSampler(
            dataset, size=0.005, split="train",
            n_folds=5, fold_idx=0, seed=42,
        )
        test_sampler = HiRISEGeoSampler(
            dataset, size=0.005, split="test",
            n_folds=5, fold_idx=0, seed=42,
        )
    """

    def __init__(
            self,
            dataset: GeoDataset,
            size: float | tuple[float, float],
            *,
            split: Literal["train", "test", "val", "all"] | None = "train",
            split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
            split_method: str = "geographic",
            split_axis: str = "longitude",
            n_folds: int | None = None,
            fold_idx: int = 0,
            seed: int = 42,
            length: int | None = None,
            stride: float | tuple[float, float] | None = None,
            roi=None,
            toi=None,
            units: Units = Units.CRS,
            generator: torch.Generator | None = None,
            min_overlap: float = 0.5,
            replacement: bool = False,
            reuse_cache: bool = True,
    ) -> None:
        super().__init__(dataset, roi, toi)

        # Use the whole dataset
        if split is None:
            split = "all"

        if split == "all":
            split = "train"
            split_fractions = (1.0, 0.0, 0.0)

        # ── Validate split parameters ──
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of {sorted(VALID_SPLITS)}."
            )
        if split_method not in VALID_SPLIT_METHODS:
            raise ValueError(
                f"Invalid split_method '{split_method}'. "
                f"Must be one of {sorted(VALID_SPLIT_METHODS)}."
            )

        train_f, val_f, test_f = split_fractions
        if n_folds is None and abs(train_f + val_f + test_f - 1.0) > 1e-6:
            raise ValueError(
                f"split_fractions must sum to 1.0, got "
                f"{train_f} + {val_f} + {test_f} = {train_f + val_f + test_f}"
            )

        self.split = split
        self.split_fractions = split_fractions
        self.split_method = split_method
        self.split_axis = split_axis
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.seed = seed
        self.replacement = replacement
        self.min_overlap = min_overlap

        # ── Resolve size / stride to (height_deg, width_deg) ──
        size_h, size_w = _to_tuple(size)
        if units == Units.PIXELS:
            xres, yres = dataset.res
            size_h *= yres
            size_w *= xres

        if stride is None:
            stride_h, stride_w = size_h, size_w
        else:
            stride_h, stride_w = _to_tuple(stride)
            if units == Units.PIXELS:
                stride_h *= yres
                stride_w *= xres

        self.size = (size_h, size_w)
        self.stride = (stride_h, stride_w)
        self.generator = generator

        # ── Compute or load split assignments ──
        self._assignments = self._get_split_assignments(dataset, reuse_cache)

        # Log split distribution
        split_counts = {}
        for s in VALID_SPLITS:
            split_counts[s] = sum(1 for v in self._assignments.values() if v == s)
        n_total = len(self._assignments)
        logger.info(
            "Split assignment: %d stereo pairs → train=%d, val=%d, test=%d "
            "(method=%s, seed=%d%s)",
            n_total,
            split_counts.get("train", 0),
            split_counts.get("val", 0),
            split_counts.get("test", 0),
            split_method,
            seed,
            f", fold={fold_idx}/{n_folds}" if n_folds else "",
        )

        # ── Pre-compute valid patch centres for this split only ──
        self._centers: list[tuple[float, float, pd.Interval]] = []
        self._build_valid_centers()

        n = len(self._centers)
        self.length = length if length is not None else max(1, n)

        if not self.replacement and self.length > n:
            logger.warning(
                "length (%d) > available centres (%d) with replacement=False; "
                "capping to %d. Use replacement=True for oversampling.",
                self.length, n, n,
            )
            self.length = n

        if not self._centers:
            logger.warning(
                "HiRISEGeoSampler: No valid patch centres found for split='%s'. "
                "Check that strip polygons are larger than the requested patch "
                "size (%.6f° × %.6f°) and that the split has data assigned.",
                self.split,
                size_h,
                size_w,
            )

    # ------------------------------------------------------------------
    # Split assignment
    # ------------------------------------------------------------------

    def _get_split_assignments(
        self,
        dataset: GeoDataset,
        reuse_cache: bool,
    ) -> dict[int, str]:
        """Compute or load cached split assignments for all stereo pairs."""

        n_pairs = len(self.index)
        train_f, val_f, test_f = self.split_fractions

        # Extract dataset-level metadata for cache key
        ds_root = str(getattr(dataset, "root", "unknown"))
        ds_target = getattr(dataset, "_target_filter", None) or \
                    getattr(dataset, "target", None)
        ds_bbox = None
        if hasattr(dataset, "_user_bbox"):
            ds_bbox = dataset._user_bbox
        elif hasattr(dataset, "bounds"):
            b: tuple[slice, slice, slice] = dataset.bounds
            ds_bbox = (b[0].start, b[1].start, b[0].stop, b[1].stop)

        ortho_types = getattr(dataset, "ortho_types", None)

        cache_hash = _split_cache_key(
            dataset_root=ds_root,
            dataset_target=ds_target,
            dataset_bbox=ds_bbox,
            method=self.split_method,
            train_fraction=train_f,
            val_fraction=val_f,
            test_fraction=test_f,
            seed=self.seed,
            n_folds=self.n_folds,
            fold_idx=self.fold_idx,
            size=self.size,
            stride=self.stride,
            min_overlap=self.min_overlap,
            ortho_types=ortho_types,
        )

        cache_dir = _split_cache_dir(ds_root)
        cache_path = cache_dir / f"split_{cache_hash}.json"

        # Try cache
        if reuse_cache:
            cached = _load_cached_split(cache_path)
            if cached is not None and len(cached) == n_pairs:
                logger.info(
                    "Loaded cached split from %s (%d pairs)", cache_path, n_pairs
                )
                return cached
            elif cached is not None:
                logger.warning(
                    "Cached split has %d pairs but dataset has %d; recomputing.",
                    len(cached), n_pairs,
                )

        # Compute pair coordinates for geographic splitting
        pair_coords = None
        if self.split_method == "geographic":
            pair_coords = self._extract_pair_coordinates(self.split_axis)

        assignments = _compute_split_assignments(
            n_pairs=n_pairs,
            pair_coords=pair_coords,
            method=self.split_method,
            train_fraction=train_f,
            val_fraction=val_f,
            test_fraction=test_f,
            seed=self.seed,
            n_folds=self.n_folds,
            fold_idx=self.fold_idx,
        )

        # Cache the result
        metadata = {
            "dataset_root": ds_root,
            "dataset_target": ds_target,
            "dataset_bbox": ds_bbox,
            "n_pairs": n_pairs,
            "method": self.split_method,
            "split_axis": self.split_axis,
            "train_fraction": train_f,
            "val_fraction": val_f,
            "test_fraction": test_f,
            "seed": self.seed,
            "n_folds": self.n_folds,
            "fold_idx": self.fold_idx,
            "split_counts": {
                s: sum(1 for v in assignments.values() if v == s)
                for s in VALID_SPLITS
            },
        }
        _save_cached_split(cache_path, assignments, metadata)

        return assignments

    def _extract_pair_coordinates(self, axis: str) -> np.ndarray:
        """Extract the centroid coordinate along the split axis for each pair."""
        projected = self.index.to_crs(MARS_PROJECTED_CRS)
        centroids = projected.geometry.centroid.to_crs(self.index.crs)
        if axis == "longitude":
            return centroids.x.to_numpy()
        elif axis == "latitude":
            return centroids.y.to_numpy()
        else:
            raise ValueError(
                f"Unknown split_axis '{axis}'. Use 'longitude' or 'latitude'."
            )

    # ------------------------------------------------------------------
    # Centre grid computation (split-aware)
    # ------------------------------------------------------------------

    def _build_valid_centers(self) -> None:
        """Grid each strip polygon and keep centres whose patch intersects it.

        Only processes strips assigned to ``self.split``.
        """
        size_h, size_w = self.size
        stride_h, stride_w = self.stride
        half_h = size_h / 2.0
        half_w = size_w / 2.0

        n_added = 0
        n_too_small = 0
        n_skipped_split = 0

        _edge_inset = max(size_h, size_w) * 0.05

        for i in range(len(self.index)):
            # ── Split filtering: skip pairs not in this split ──
            if self._assignments.get(i) != self.split:
                n_skipped_split += 1
                continue

            footprint = self.index.geometry.iloc[i]
            interval = self.index.index[i]

            try:
                effective = footprint.buffer(-_edge_inset)
                if effective.is_empty or not effective.is_valid:
                    effective = footprint
            except Exception:
                effective = footprint

            minx, miny, maxx, maxy = effective.bounds

            if (maxx - minx) < size_w or (maxy - miny) < size_h:
                n_too_small += 1
                continue

            xs = np.arange(minx + half_w, maxx - half_w + stride_w * 1e-6, stride_w)
            ys = np.arange(miny + half_h, maxy - half_h + stride_h * 1e-6, stride_h)

            pd_interval = pd.Interval(interval.left, interval.right, closed="both")

            for cy in ys:
                for cx in xs:
                    patch = shapely_box(
                        cx - half_w, cy - half_h, cx + half_w, cy + half_h
                    )
                    overlap = effective.intersection(patch).area / patch.area
                    if overlap > self.min_overlap:
                        self._centers.append((cx, cy, pd_interval))
                        n_added += 1

        logger.info(
            "HiRISEGeoSampler [%s]: %d valid centres from %d strips "
            "(%d in other splits, %d too small).",
            self.split,
            n_added,
            len(self.index),
            n_skipped_split,
            n_too_small,
        )

    # ------------------------------------------------------------------
    # Sampler protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[slice, slice, slice]]:
        """Yield random patch slices guaranteed to intersect strip data.

        Yields:
            ``(x_slice, y_slice, t_slice)`` tuples compatible with
            :meth:`~MarsHiRISEDTM.__getitem__`.
        """
        n = len(self._centers)
        if n == 0:
            return

        half_h, half_w = self.size[0] / 2.0, self.size[1] / 2.0

        if self.replacement:
            indices = torch.randint(n, (self.length,), generator=self.generator).tolist()
        else:
            indices = torch.randperm(n, generator=self.generator).tolist()[:self.length]

        for idx in indices:
            cx, cy, interval = self._centers[idx]
            yield (
                slice(cx - half_w, cx + half_w),
                slice(cy - half_h, cy + half_h),
                slice(interval.left, interval.right),
            )

    def __len__(self) -> int:
        return self.length

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def split_summary(self) -> dict[str, str | int | dict[str, int] | Any]:
        """Return the number of stereo pairs and patch centres per split."""
        pair_counts = {s: 0 for s in VALID_SPLITS}
        for v in self._assignments.values():
            pair_counts[v] += 1
        return {
            "split": self.split,
            "pairs_in_split": pair_counts[self.split],
            "centres_in_split": len(self._centers),
            "total_pairs": len(self._assignments),
            "pairs_per_split": pair_counts,
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _to_tuple(value: float | tuple[float, float]) -> tuple[float, float]:
    """Normalise a scalar or 2-tuple to ``(height, width)``."""
    if isinstance(value, (int, float)):
        return float(value), float(value)
    return float(value[0]), float(value[1])