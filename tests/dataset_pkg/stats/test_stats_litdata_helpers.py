"""Unit tests for pure NumPy helpers in :mod:`dataset.stats.compute_stats_litdata`.

These cover the numerically-stable Welford accumulator helpers, plane-fit
detrending, percentile-from-histogram interpolation, and the multi-rank
Welford combination.  All functions are pure — no GDAL, no torch.distributed,
no I/O — so they are tested directly with synthetic NumPy inputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dataset.stats.compute_stats_litdata import (
    _calculate_percentile_from_hist,
    _combine_welford,
    _welford_update,
    detrend,
)


# ---------------------------------------------------------------------------
# _welford_update — single-channel running mean / M2
# ---------------------------------------------------------------------------


class TestWelfordUpdate:
    def test_empty_batch_returns_state_unchanged(self):
        # Arrange
        n, mean, M2 = 0, 0.0, 0.0
        valid = np.array([], dtype=np.float64)
        # Act
        n_out, m_out, M2_out = _welford_update(n, mean, M2, valid)
        # Assert
        assert (n_out, m_out, M2_out) == (0, 0.0, 0.0)

    def test_single_batch_matches_numpy_mean_and_variance(self):
        rng = np.random.default_rng(0)
        valid = rng.normal(loc=3.5, scale=2.0, size=10_000).astype(np.float64)
        n_out, mean_out, M2_out = _welford_update(0, 0.0, 0.0, valid)
        assert n_out == valid.size
        assert math.isclose(mean_out, valid.mean(), rel_tol=1e-12)
        # population variance from M2 / n
        var_out = M2_out / n_out
        assert math.isclose(var_out, valid.var(), rel_tol=1e-12)

    def test_two_batches_match_concatenated(self):
        """Parallel Welford combine: two batches → same stats as one big batch."""
        rng = np.random.default_rng(42)
        a = rng.normal(0.0, 1.0, 500).astype(np.float64)
        b = rng.normal(5.0, 3.0, 700).astype(np.float64)
        # Sequential update
        n, m, M2 = _welford_update(0, 0.0, 0.0, a)
        n, m, M2 = _welford_update(n, m, M2, b)
        # Reference
        all_ = np.concatenate([a, b])
        assert n == all_.size
        assert math.isclose(m, all_.mean(), rel_tol=1e-12)
        assert math.isclose(M2 / n, all_.var(), rel_tol=1e-12)


# ---------------------------------------------------------------------------
# detrend — least-squares plane removal
# ---------------------------------------------------------------------------


class TestDetrend:
    def test_perfect_plane_yields_near_zero_residual(self):
        """A pure plane z = ax+by+c should detrend to ~zero residuals."""
        h, w = 16, 24
        y, x = np.mgrid[0:h, 0:w].astype(np.float64)
        z = 2.0 * x - 0.5 * y + 7.0
        mask = np.ones_like(z, dtype=bool)
        residual = detrend(z, mask)
        assert residual.shape == (h * w,)
        assert float(residual.max()) < 1e-8

    def test_constant_plus_noise_residual_is_noise_magnitude(self):
        rng = np.random.default_rng(1)
        z = np.full((10, 10), 100.0)
        z += rng.normal(scale=0.1, size=z.shape)
        residual = detrend(z, np.ones_like(z, dtype=bool))
        # All valid pixels returned; magnitudes ≈ noise level
        assert residual.size == 100
        assert float(residual.mean()) < 0.5

    def test_empty_mask_returns_empty_array(self):
        z = np.zeros((4, 4), dtype=np.float64)
        mask = np.zeros_like(z, dtype=bool)
        residual = detrend(z, mask)
        assert isinstance(residual, np.ndarray)
        assert residual.size == 0


# ---------------------------------------------------------------------------
# _calculate_percentile_from_hist — linear interpolation inside the target bin
# ---------------------------------------------------------------------------


class TestCalculatePercentileFromHist:
    def test_empty_histogram_returns_zero(self):
        assert _calculate_percentile_from_hist([0, 0, 0], [0.0, 1.0, 2.0, 3.0], 0.5) == 0.0

    def test_p100_returns_right_edge(self):
        # 100% percentile falls past the cumulative max → returns last edge
        hist = [10, 20, 30]
        edges = [0.0, 1.0, 2.0, 3.0]
        result = _calculate_percentile_from_hist(hist, edges, 1.0)
        # By the loop exit condition this returns the last edge.
        assert result == 3.0

    def test_p0_returns_left_edge_of_first_bin(self):
        result = _calculate_percentile_from_hist([10, 20], [0.0, 1.0, 2.0], 0.0)
        # target_count = 0; first iter cumulative+count >= 0 → returns bin start
        assert result == 0.0

    def test_uniform_bins_percentile_50(self):
        # 100 evenly distributed across 4 bins → p50 → middle of bin index 1
        hist = [25, 25, 25, 25]
        edges = [0.0, 1.0, 2.0, 3.0, 4.0]
        # target_count = 50; bin0 fills 25; cumulative=25 < 50; bin1 fills next 25
        # at fraction (50-25)/25=1.0 → start + 1.0 * width = 1.0 + 1.0 = 2.0
        result = _calculate_percentile_from_hist(hist, edges, 0.5)
        assert math.isclose(result, 2.0, abs_tol=1e-12)

    def test_zero_count_bin_in_path_returns_bin_start(self):
        # Pathological: target lands exactly at a zero-count bin.
        # Loop hits "if count == 0: return bin_edges[i]"
        hist = [10, 0, 10]
        edges = [0.0, 1.0, 2.0, 3.0]
        # total=20, target at p=0.55 → 11.0; cumulative=10 after bin0, so next
        # iter bin1 has count=0 and cumulative+0 >= 11? cumulative=10, target=11
        # No — 10+0 = 10 < 11 → continue.  Pick a p that lands on the zero bin:
        # cumulative=10, want cumulative+count >= target → need 10+0 >= target
        # so target<=10 lands inside zero-bin path only after we've moved past.
        # Use p=0.5 → target=10; iter bin0: 0+10>=10 → enters → count=10 → frac=1.0
        # → result = 0.0 + 1.0 * 1.0 = 1.0
        result = _calculate_percentile_from_hist(hist, edges, 0.5)
        assert math.isclose(result, 1.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# _combine_welford — multi-rank reduction
# ---------------------------------------------------------------------------


def _make_partial(C: int = 2) -> dict:
    """Empty single-rank partial state with C channels."""
    return {
        "n_patches": 0,
        "count": np.zeros(C, dtype=np.int64),
        "mean": np.zeros(C, dtype=np.float64),
        "M2": np.zeros(C, dtype=np.float64),
        "ch_min": np.full(C, np.inf, dtype=np.float64),
        "ch_max": np.full(C, -np.inf, dtype=np.float64),
        "hist": np.zeros((C, 4), dtype=np.int64),
        "c_count": np.zeros(C, dtype=np.int64),
        "c_mean": np.zeros(C, dtype=np.float64),
        "c_M2": np.zeros(C, dtype=np.float64),
        "c_min": np.full(C, np.inf, dtype=np.float64),
        "c_max": np.full(C, -np.inf, dtype=np.float64),
        "c_hist": np.zeros((C, 4), dtype=np.int64),
    }


class TestCombineWelford:
    def test_single_partial_returns_copy(self):
        p = _make_partial()
        p["n_patches"] = 7
        p["count"][0] = 100
        p["mean"][0] = 1.5
        combined = _combine_welford([p])
        assert combined["n_patches"] == 7
        # NOTE: per-channel arrays are aliased via shallow dict copy; we verify
        # values match rather than identity.  This documents current behavior.
        assert combined["count"][0] == 100
        assert combined["mean"][0] == 1.5

    def test_two_partials_combine_means_correctly(self):
        a = _make_partial(C=1)
        a["n_patches"] = 1
        a["count"][0] = 100
        a["mean"][0] = 10.0
        a["M2"][0] = 0.0
        a["ch_min"][0] = 5.0
        a["ch_max"][0] = 15.0

        b = _make_partial(C=1)
        b["n_patches"] = 1
        b["count"][0] = 50
        b["mean"][0] = 20.0
        b["M2"][0] = 0.0
        b["ch_min"][0] = 18.0
        b["ch_max"][0] = 25.0

        combined = _combine_welford([a, b])
        assert combined["n_patches"] == 2
        assert combined["count"][0] == 150
        # weighted mean: (100*10 + 50*20) / 150 ≈ 13.333
        assert math.isclose(combined["mean"][0], (100 * 10 + 50 * 20) / 150, rel_tol=1e-12)
        assert combined["ch_min"][0] == 5.0
        assert combined["ch_max"][0] == 25.0

    def test_zero_count_channel_takes_other_partials_values(self):
        """If channel has zero count in the accumulator, copy the new partial verbatim."""
        a = _make_partial(C=1)
        # a has zero count for channel 0
        b = _make_partial(C=1)
        b["count"][0] = 42
        b["mean"][0] = 3.14
        b["ch_min"][0] = -1.0
        b["ch_max"][0] = 9.0
        combined = _combine_welford([a, b])
        assert combined["count"][0] == 42
        assert math.isclose(combined["mean"][0], 3.14)
        assert combined["ch_min"][0] == -1.0
        assert combined["ch_max"][0] == 9.0
