"""Regression tests for successive updates in the illumination estimator."""

import importlib.util
from pathlib import Path

import pytest
import torch


# Load this numerical helper without initializing the model/training package
# (which imports GPU and checkpoint-related dependencies).
_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src/depth_fm/data/image_processing/sun_vector.py"
)
_SPEC = importlib.util.spec_from_file_location("sun_vector_under_test", _SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.fixture
def scene():
    """Nonplanar terrain and positive imagery with sparse bright outliers."""
    axis = torch.linspace(-1, 1, 32)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    dtm = (0.15 * torch.sin(3 * x) + 0.12 * torch.cos(4 * y))[None]
    image = (0.6 + 0.15 * torch.cos(3 * x) + 0.1 * torch.sin(4 * y))[None]
    image[:, ::5, ::7] += 0.8
    return dtm, image, torch.ones_like(dtm)


def record_solves(monkeypatch):
    original = torch.linalg.lstsq
    calls = []

    def recording_lstsq(a, b, *args, **kwargs):
        result = original(a, b, *args, **kwargs)
        calls.append((a.clone(), result.solution.clone()))
        return result

    monkeypatch.setattr(torch.linalg, "lstsq", recording_lstsq)
    return calls


def assert_returns_last_fit(result, calls):
    direction, intensity, _ = result
    expected = calls[-1][1][:3, 0].clone()
    expected[2] = expected[2].abs()
    assert torch.linalg.vector_norm(expected) > 1e-4
    torch.testing.assert_close(direction * intensity, expected)


@pytest.mark.parametrize("max_iter", [1, 4])
def test_iteration_budget_returns_latest_weighted_fit(scene, monkeypatch, max_iter):
    calls = record_solves(monkeypatch)
    result = _MODULE.estimate_sun_vector_irls(*scene, max_iter=max_iter, tol=0.0)

    assert len(calls) == max_iter + 1  # Initial OLS plus weighted solves.
    assert torch.linalg.vector_norm(calls[1][1] - calls[0][1]) > 1e-4
    assert_returns_last_fit(result, calls)
    if max_iter > 1:
        # A changed coefficient must produce new residual-based weights.
        assert not torch.allclose(calls[1][0], calls[2][0])


def test_convergence_stops_after_accepting_candidate(scene, monkeypatch):
    calls = record_solves(monkeypatch)
    result = _MODULE.estimate_sun_vector_irls(*scene, max_iter=15, tol=100.0)

    assert len(calls) == 2
    assert_returns_last_fit(result, calls)
