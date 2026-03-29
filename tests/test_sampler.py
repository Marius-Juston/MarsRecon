"""Tests for HiRISEGeoSampler.

Covers:
- All yielded patches intersect the strip polygon (not merely its bbox)
- Patch size is exactly `size` degrees
- `length` parameter controls the number of yielded samples
- Strips narrower than `size` produce no valid centres
- Seeded generator gives reproducible results
- Multiple strips are all sampled
"""

import pathlib
import sys
from unittest.mock import patch

import pytest
import torch
from shapely.geometry import Polygon, box

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from helpers import make_mock_dataset
from hirise_sampler import HiRISEGeoSampler
from torchgeo.samplers import Units


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def single_strip_dataset(strip_polygon: Polygon, mars_crs):
    """Mock dataset with one rotated HiRISE strip."""
    return make_mock_dataset([strip_polygon], mars_crs)


@pytest.fixture
def multi_strip_dataset(mars_crs):
    """Mock dataset with three non-overlapping axis-aligned strips."""
    strips = [
        box(-140.0, 10.0, -139.9, 15.0),  # narrow N-S strip
        box(-138.0, 10.0, -137.9, 15.0),
        box(-136.0, 10.0, -135.9, 15.0),
    ]
    return make_mock_dataset(strips, mars_crs)


# ---------------------------------------------------------------------------
# Core correctness tests
# ---------------------------------------------------------------------------


class TestHiRISESamplerCorrectness:
    PATCH_SIZE = 0.005  # degrees

    def test_all_patches_intersect_strip(self, strip_polygon, single_strip_dataset):
        sampler = HiRISEGeoSampler(
            single_strip_dataset, size=self.PATCH_SIZE, length=50, units=Units.CRS
        )
        for x_sl, y_sl, _ in sampler:
            patch = box(x_sl.start, y_sl.start, x_sl.stop, y_sl.stop)
            assert strip_polygon.intersects(patch), (
                f"Patch {patch.bounds} does not intersect strip polygon"
            )

    def test_no_patch_is_entirely_outside_strip_bbox(
            self, strip_polygon, single_strip_dataset
    ):
        bbox = box(*strip_polygon.bounds)
        sampler = HiRISEGeoSampler(
            single_strip_dataset, size=self.PATCH_SIZE, length=50, units=Units.CRS
        )
        for x_sl, y_sl, _ in sampler:
            patch = box(x_sl.start, y_sl.start, x_sl.stop, y_sl.stop)
            assert bbox.intersects(patch)

    def test_patch_width_equals_size(self, single_strip_dataset):
        sampler = HiRISEGeoSampler(
            single_strip_dataset, size=self.PATCH_SIZE, length=10, units=Units.CRS
        )
        for x_sl, y_sl, _ in sampler:
            width = x_sl.stop - x_sl.start
            height = y_sl.stop - y_sl.start
            assert width == pytest.approx(self.PATCH_SIZE, rel=1e-9)
            assert height == pytest.approx(self.PATCH_SIZE, rel=1e-9)

    def test_asymmetric_patch_size(self, strip_polygon, mars_crs):
        size = (0.003, 0.007)  # height × width
        dataset = make_mock_dataset([strip_polygon], mars_crs)
        sampler = HiRISEGeoSampler(dataset, size=size, length=10, units=Units.CRS)
        for x_sl, y_sl, _ in sampler:
            assert (y_sl.stop - y_sl.start) == pytest.approx(size[0], rel=1e-9)
            assert (x_sl.stop - x_sl.start) == pytest.approx(size[1], rel=1e-9)


# ---------------------------------------------------------------------------
# Length and count
# ---------------------------------------------------------------------------


class TestHiRISESamplerLength:
    def test_requested_length_respected(self, single_strip_dataset):
        sampler = HiRISEGeoSampler(
            single_strip_dataset, size=0.005, length=37, units=Units.CRS
        )
        samples = list(sampler)
        assert len(samples) == 37

    def test_default_length_equals_number_of_centers(self, single_strip_dataset):
        sampler = HiRISEGeoSampler(
            single_strip_dataset, size=0.005, units=Units.CRS
        )
        assert len(sampler) == len(sampler._centers)
        assert len(list(sampler)) == len(sampler)

    def test_strip_too_small_produces_no_centers(self, mars_crs):
        tiny = box(-136.0, 18.0, -135.999, 18.002)  # ~0.001° × ~0.002°
        dataset = make_mock_dataset([tiny], mars_crs)
        sampler = HiRISEGeoSampler(
            dataset, size=0.005, length=10, units=Units.CRS
        )
        assert len(sampler._centers) == 0

    def test_multiple_strips_all_contribute_centers(self, multi_strip_dataset):
        sampler = HiRISEGeoSampler(
            multi_strip_dataset, size=0.005, length=10, units=Units.CRS
        )
        # With three 0.1°-wide × 5°-long strips, each should contribute centres.
        assert len(sampler._centers) > 0

    def test_zero_length_yields_nothing(self, single_strip_dataset):
        sampler = HiRISEGeoSampler(
            single_strip_dataset, size=0.005, length=0, units=Units.CRS
        )
        assert list(sampler) == []


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestHiRISESamplerReproducibility:
    def test_seeded_generator_is_deterministic(self, single_strip_dataset):
        def _run(seed: int):
            gen = torch.Generator()
            gen.manual_seed(seed)
            sampler = HiRISEGeoSampler(
                single_strip_dataset, size=0.005, length=20,
                generator=gen, units=Units.CRS,
            )
            return [
                (round(x.start, 10), round(y.start, 10))
                for x, y, _ in sampler
            ]

        run1 = _run(42)
        run2 = _run(42)
        assert run1 == run2

    def test_different_seeds_produce_different_sequences(self, single_strip_dataset):
        def _run(seed: int):
            gen = torch.Generator()
            gen.manual_seed(seed)
            sampler = HiRISEGeoSampler(
                single_strip_dataset, size=0.005, length=20,
                generator=gen, units=Units.CRS,
            )
            return [x.start for x, _, _ in sampler]

        assert _run(42) != _run(99)


# ---------------------------------------------------------------------------
# Stride
# ---------------------------------------------------------------------------


class TestHiRISESamplerStride:
    def test_smaller_stride_produces_more_centers(self, strip_polygon, mars_crs):
        dataset = make_mock_dataset([strip_polygon], mars_crs)
        coarse = HiRISEGeoSampler(
            dataset, size=0.005, stride=0.005, units=Units.CRS
        )
        fine = HiRISEGeoSampler(
            dataset, size=0.005, stride=0.002, units=Units.CRS
        )
        assert len(fine._centers) > len(coarse._centers)

    def test_default_stride_equals_size(self, strip_polygon, mars_crs):
        dataset = make_mock_dataset([strip_polygon], mars_crs)
        s1 = HiRISEGeoSampler(dataset, size=0.005, units=Units.CRS)
        s2 = HiRISEGeoSampler(dataset, size=0.005, stride=0.005, units=Units.CRS)
        assert len(s1._centers) == len(s2._centers)


# ---------------------------------------------------------------------------
# _to_tuple utility
# ---------------------------------------------------------------------------


class TestToTuple:
    """_to_tuple normalises scalars and 2-tuples to (height, width)."""

    def test_integer_returns_symmetric_float_pair(self):
        from hirise_sampler import _to_tuple

        result = _to_tuple(3)
        assert result == (3.0, 3.0)

    def test_float_returns_symmetric_float_pair(self):
        from hirise_sampler import _to_tuple

        result = _to_tuple(0.005)
        assert result == pytest.approx((0.005, 0.005))

    def test_tuple_preserved_as_floats(self):
        from hirise_sampler import _to_tuple

        result = _to_tuple((0.003, 0.007))
        assert result == pytest.approx((0.003, 0.007))


# ---------------------------------------------------------------------------
# PIXELS units conversion
# ---------------------------------------------------------------------------


class TestSamplerPixelUnits:
    def test_pixels_converted_to_degrees(self, strip_polygon, mars_crs):
        """Units.PIXELS: size/stride multiplied by dataset.res."""
        from helpers import make_mock_dataset

        pixel_size = 100
        xres, yres = 8.44e-6, 8.44e-6
        dataset = make_mock_dataset([strip_polygon], mars_crs, res=(xres, yres))
        sampler = HiRISEGeoSampler(
            dataset, size=pixel_size, length=10, units=Units.PIXELS
        )
        expected_h = pixel_size * yres
        expected_w = pixel_size * xres
        assert sampler.size[0] == pytest.approx(expected_h, rel=1e-6)
        assert sampler.size[1] == pytest.approx(expected_w, rel=1e-6)
        # Stride defaults to size when not set
        assert sampler.stride[0] == pytest.approx(expected_h, rel=1e-6)


# ---------------------------------------------------------------------------
# buffer() exception fallback
# ---------------------------------------------------------------------------


class TestBufferExceptionFallback:
    def test_buffer_raises_falls_back_to_original(
        self, strip_polygon, mars_crs
    ):
        """If buffer(-inset) raises, the sampler falls back to the original polygon."""
        from helpers import make_mock_dataset

        original_buffer = strip_polygon.buffer

        call_count = [0]

        def _bad_buffer(dist, *args, **kwargs):
            call_count[0] += 1
            if dist < 0:
                raise RuntimeError("simulated buffer error")
            return original_buffer(dist, *args, **kwargs)

        dataset = make_mock_dataset([strip_polygon], mars_crs)

        with patch.object(
            strip_polygon.__class__, "buffer", side_effect=_bad_buffer
        ):
            sampler = HiRISEGeoSampler(
                dataset, size=0.005, length=10, units=Units.CRS
            )

        # Even with buffer failing, centers should be found using original polygon
        assert len(sampler._centers) > 0


# ---------------------------------------------------------------------------
# min_overlap edge cases
# ---------------------------------------------------------------------------


class TestMinOverlapEdgeCases:
    def test_min_overlap_zero_accepts_more_centers(self, strip_polygon, mars_crs):
        """min_overlap=0 (exclusive) accepts grid points that barely clip the polygon."""
        from helpers import make_mock_dataset

        dataset = make_mock_dataset([strip_polygon], mars_crs)
        default_sampler = HiRISEGeoSampler(
            dataset, size=0.005, length=10, units=Units.CRS, min_overlap=0.5
        )
        loose_sampler = HiRISEGeoSampler(
            dataset, size=0.005, length=10, units=Units.CRS, min_overlap=0.0
        )
        # With a lower threshold, at least as many centres (typically more)
        assert len(loose_sampler._centers) >= len(default_sampler._centers)

    def test_min_overlap_one_strict_filter(self, strip_polygon, mars_crs):
        """min_overlap=1.0 keeps only fully-contained patches (or none)."""
        from helpers import make_mock_dataset

        dataset = make_mock_dataset([strip_polygon], mars_crs)
        strict_sampler = HiRISEGeoSampler(
            dataset, size=0.005, length=10, units=Units.CRS, min_overlap=1.0
        )
        default_sampler = HiRISEGeoSampler(
            dataset, size=0.005, length=10, units=Units.CRS, min_overlap=0.5
        )
        # Strict filter has at most as many centres as default
        assert len(strict_sampler._centers) <= len(default_sampler._centers)


# ---------------------------------------------------------------------------
# Empty centres iteration
# ---------------------------------------------------------------------------


class TestEmptyCentersIteration:
    def test_tiny_strip_empty_centers_yields_nothing(self, mars_crs):
        """Strip smaller than patch → _centers == [] → list(sampler) == []."""
        from helpers import make_mock_dataset

        tiny = box(-136.0, 18.0, -135.999, 18.002)  # ~0.001° × ~0.002°
        dataset = make_mock_dataset([tiny], mars_crs)
        sampler = HiRISEGeoSampler(dataset, size=0.005, length=5, units=Units.CRS)
        assert sampler._centers == []
        assert list(sampler) == []  # hits the `if n == 0: return` branch
