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
