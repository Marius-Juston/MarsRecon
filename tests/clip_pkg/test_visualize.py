"""Tests for MarsCLIP preview utilities."""

from __future__ import annotations

import pathlib
import sys

import torch

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from clip.visualize_marsclip import _build_mask_overlay, save_sample_preview


def test_build_mask_overlay_is_informative_for_full_valid_mask():
    image = torch.zeros(3, 4, 4, dtype=torch.float32)
    image[1] = 0.5
    valid_mask = torch.ones(4, 4, dtype=torch.bool)

    overlay = _build_mask_overlay(image, valid_mask)

    assert overlay.shape == (4, 4, 3)
    assert overlay.dtype.kind == "f"
    assert overlay[..., 1].mean() > overlay[..., 0].mean()


def test_build_mask_overlay_marks_invalid_pixels_red():
    image = torch.ones(3, 4, 4, dtype=torch.float32) * 0.3
    valid_mask = torch.ones(4, 4, dtype=torch.bool)
    valid_mask[0, 0] = False

    overlay = _build_mask_overlay(image, valid_mask)

    assert overlay[0, 0, 0] > overlay[0, 0, 1]
    assert overlay[0, 0, 0] > overlay[0, 0, 2]


def test_save_sample_preview_writes_png(tmp_path):
    sample = {
        "image": torch.rand(3, 8, 8),
        "valid_mask": torch.ones(8, 8, dtype=torch.bool),
        "rationale_raw": "Olympus Mons lava channels",
        "rationale_expanded": "Expanded geology context",
        "metadata": {
            "obs_id": "OBS_A",
            "overall_valid_fraction": 1.0,
        },
    }
    out_path = tmp_path / "preview.png"

    result = save_sample_preview([sample], out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
