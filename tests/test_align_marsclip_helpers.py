"""Unit tests for small helpers in ``stage_b.align_marsclip`` (B1a-pairs)."""

from __future__ import annotations

import pytest
import torch

from stage_b.align_marsclip import PAIRED_VIEWS_CHOICES, make_local_view


def test_make_local_view_preserves_batch_shape() -> None:
    b, c, h, w = 3, 3, 256, 256
    x = torch.randn(b, c, h, w)
    y = make_local_view(x, crop_fraction=0.5)
    assert y.shape == (b, c, h, w)


def test_make_local_view_center_crop_fraction_deterministic() -> None:
    """Inner half of a constant plane should equal the outer after resize."""
    x = torch.ones(1, 1, 4, 4)
    y = make_local_view(x, crop_fraction=0.5)
    assert torch.allclose(y, torch.ones_like(y))


@pytest.mark.parametrize("bad", (-0.1, 0.0, 1.0, 2.0))
def test_make_local_view_rejects_invalid_crop_fraction(bad: float) -> None:
    x = torch.randn(1, 3, 32, 32)
    with pytest.raises(ValueError, match="local_crop_fraction"):
        make_local_view(x, crop_fraction=bad)


def test_paired_views_choices() -> None:
    assert PAIRED_VIEWS_CHOICES == ("none", "local_global")

