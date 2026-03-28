"""HiRISE-aware geospatial sampler for TorchGeo.

HiRISE satellite passes produce long, narrow, slightly rotated image strips.
Sampling patches uniformly from axis-aligned bounding boxes wastes 60–90 % of
samples on empty (zero-value) pixels that fall in the corners outside the
actual strip.

:class:`HiRISEGeoSampler` avoids this by pre-computing a grid of patch centres
that are confirmed to lie within each strip's polygon footprint, then sampling
from that pre-computed set at each epoch.  Construction runs once; iteration is
O(1) per sample.
"""

import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd
import torch
from shapely.geometry import box as shapely_box
from torchgeo.datasets.geo import GeoDataset
from torchgeo.samplers import GeoSampler, Units

logger = logging.getLogger(__name__)


class HiRISEGeoSampler(GeoSampler):
    """Sampler that restricts patches to within HiRISE strip polygon footprints.

    Construction pre-computes a regular grid of candidate patch centres for
    every strip in ``dataset.index``, keeping only centres whose corresponding
    patch intersects the polygon footprint (not merely its bounding box).  At
    each epoch, ``length`` centres are drawn uniformly at random from this set.

    Args:
        dataset: The :class:`~temp.MarsHiRISE` dataset to sample from.
        size: Patch height and width in CRS units (degrees when
            ``units=Units.CRS``) or pixels (when ``units=Units.PIXELS``).
            A single float sets both dimensions equal.
        length: Number of patches to yield per epoch.  Defaults to the total
            number of pre-computed valid centres (roughly one non-overlapping
            pass over all strips).
        stride: Centre-to-centre grid spacing in the same units as ``size``.
            Defaults to ``size`` (non-overlapping grid).  Use a smaller value
            (e.g. ``0.5 * size``) for denser / overlapping sampling.
        roi: Optional Shapely Polygon to further restrict the spatial domain.
        toi: Optional :class:`pandas.Interval` to restrict the temporal domain.
        units: Whether *size* and *stride* are given in CRS units
            (:attr:`~torchgeo.samplers.Units.CRS`) or pixels
            (:attr:`~torchgeo.samplers.Units.PIXELS`).
        generator: Optional :class:`torch.Generator` for reproducible sampling.

    Example::

        from hirise_sampler import HiRISEGeoSampler
        from torchgeo.samplers import Units

        sampler = HiRISEGeoSampler(dataset, size=0.005, length=500,
                                   units=Units.CRS)
        dataloader = DataLoader(dataset, sampler=sampler)
    """

    def __init__(
            self,
            dataset: GeoDataset,
            size: float | tuple[float, float],
            length: int | None = None,
            stride: float | tuple[float, float] | None = None,
            roi=None,
            toi=None,
            units: Units = Units.CRS,
            generator: torch.Generator | None = None,
            min_overlap: int = 0.5
    ) -> None:
        super().__init__(dataset, roi, toi)

        # ----------------------------------------------------------------
        # Resolve size / stride to (height_deg, width_deg)
        # ----------------------------------------------------------------

        size_h, size_w = _to_tuple(size)
        if units == Units.PIXELS:
            xres, yres = dataset.res  # GeoDataset.res is always a (xres, yres) tuple
            size_h = size_h * yres
            size_w = size_w * xres

        if stride is None:
            stride_h, stride_w = size_h, size_w
        else:
            stride_h, stride_w = _to_tuple(stride)
            if units == Units.PIXELS:
                stride_h = stride_h * yres
                stride_w = stride_w * xres

        self.size = (size_h, size_w)
        self.stride = (stride_h, stride_w)
        self.generator = generator
        self.min_overlap = min_overlap

        # ----------------------------------------------------------------
        # Pre-compute valid patch centres once at construction time
        # ----------------------------------------------------------------
        self._centers: list[tuple[float, float, pd.Interval]] = []
        self._build_valid_centers()

        self.length = length if length is not None else max(1, len(self._centers))

        if not self._centers:
            logger.warning(
                "HiRISEGeoSampler: No valid patch centres found.  "
                "Check that strip polygons are larger than the requested patch "
                "size (%.6f° × %.6f°).",
                size_h,
                size_w,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_valid_centers(self) -> None:
        """Grid each strip polygon and keep centres whose patch intersects it."""
        size_h, size_w = self.size
        stride_h, stride_w = self.stride
        half_h = size_h / 2.0
        half_w = size_w / 2.0

        n_added = 0
        n_too_small = 0

        # Small inset applied to each strip footprint before computing candidate
        # centres.  This prevents generating centres at the precise polygon
        # boundary where floating-point differences between the index geometry
        # (Shapely) and rasterio's recomputed bounds can cause the early-exit
        # check in _load_from_jp2 to reject an otherwise-valid patch.
        # 5 % of the patch size ≈ a few tens of metres at HiRISE resolution.
        _edge_inset = max(size_h, size_w) * 0.05

        for i in range(len(self.index)):
            footprint = self.index.geometry.iloc[i]
            interval = self.index.index[i]

            # Apply edge inset for robustness; fall back to original if result
            # is empty (strip close to patch size).
            try:
                effective = footprint.buffer(-_edge_inset)
                if effective.is_empty or not effective.is_valid:
                    effective = footprint
            except Exception:
                effective = footprint

            minx, miny, maxx, maxy = effective.bounds

            # Skip strips that are narrower or shorter than one patch.
            if (maxx - minx) < size_w or (maxy - miny) < size_h:
                n_too_small += 1
                logger.debug(
                    "Strip %d is smaller than patch size (%.6f° × %.6f°); skipping.",
                    i,
                    maxx - minx,
                    maxy - miny,
                )
                continue

            # Generate candidate centres on a regular grid within the bbox,
            # inset by half-patch so every patch fits within the bbox.
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
            "HiRISEGeoSampler: %d valid centres across %d strips "
            "(%d strips skipped — smaller than patch).",
            n_added,
            len(self.index),
            n_too_small,
        )

    # ------------------------------------------------------------------
    # Sampler protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[slice, slice, slice]]:
        """Yield random patch slices guaranteed to intersect strip data.

        Yields:
            ``(x_slice, y_slice, t_slice)`` tuples compatible with
            :meth:`~temp.MarsHiRISE.__getitem__`.
        """
        n = len(self._centers)
        if n == 0:
            return

        half_h, half_w = self.size[0] / 2.0, self.size[1] / 2.0

        indices = torch.randint(n, (self.length,), generator=self.generator).tolist()
        for idx in indices:
            cx, cy, interval = self._centers[idx]
            yield (
                slice(cx - half_w, cx + half_w),
                slice(cy - half_h, cy + half_h),
                slice(interval.left, interval.right),
            )

    def __len__(self) -> int:
        return self.length


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _to_tuple(value: float | tuple[float, float]) -> tuple[float, float]:
    """Normalise a scalar or 2-tuple to ``(height, width)``."""
    if isinstance(value, (int, float)):
        return (float(value), float(value))
    return (float(value[0]), float(value[1]))
